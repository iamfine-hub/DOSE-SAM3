"""
SAM3 + SEED 适配器 FSS 评测脚本 (纯 A, 训练 + 评估一体化).
对每个 fold:
  1. 用其他 3 个 fold 的训练数据训练 A (排除当前 fold 测试类)
  2. 在当前 fold 上用 A 评测

协议: PASCAL-5i / COCO-20i, 4-fold 交叉验证, seed 42

用法:
  python FSS_sam3_seed.py --checkpoint /root/checkpoints/sam3.pt --dataset pascal
"""

import argparse
import ast
import json
import os
import random
import sys
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ---- 类别名称 ----

PASCAL_CLASSES = {
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor",
}

COCO_CLASSES = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 12: "stop sign", 13: "parking meter", 14: "bench", 15: "bird",
    16: "cat", 17: "dog", 18: "horse", 19: "sheep", 20: "cow",
    21: "elephant", 22: "bear", 23: "zebra", 24: "giraffe", 25: "backpack",
    26: "umbrella", 27: "handbag", 28: "tie", 29: "suitcase", 30: "frisbee",
    31: "skis", 32: "snowboard", 33: "sports ball", 34: "kite", 35: "baseball bat",
    36: "baseball glove", 37: "skateboard", 38: "surfboard", 39: "tennis racket", 40: "bottle",
    41: "wine glass", 42: "cup", 43: "fork", 44: "knife", 45: "spoon",
    46: "bowl", 47: "banana", 48: "apple", 49: "sandwich", 50: "orange",
    51: "broccoli", 52: "carrot", 53: "hot dog", 54: "pizza", 55: "donut",
    56: "cake", 57: "chair", 58: "couch", 59: "potted plant", 60: "bed",
    61: "dining table", 62: "toilet", 63: "tv", 64: "laptop", 65: "mouse",
    66: "remote", 67: "keyboard", 68: "cell phone", 69: "microwave", 70: "oven",
    71: "toaster", 72: "sink", 73: "refrigerator", 74: "book", 75: "clock",
    76: "vase", 77: "scissors", 78: "teddy bear", 79: "hair drier", 80: "toothbrush",
}


# ---- Fold 划分 ----

def get_fold_classes(dataset, split):
    """返回 (train_classes, test_classes)."""
    if dataset == "pascal":
        if split == 0:
            return list(range(6, 21)), list(range(1, 6))
        elif split == 1:
            return list(range(1, 6)) + list(range(11, 21)), list(range(6, 11))
        elif split == 2:
            return list(range(1, 11)) + list(range(16, 21)), list(range(11, 16))
        elif split == 3:
            return list(range(1, 16)), list(range(16, 21))
    elif dataset == "coco":
        all_classes = list(range(1, 81))
        if split == 0:
            test = list(range(1, 78, 4))
        elif split == 1:
            test = list(range(2, 79, 4))
        elif split == 2:
            test = list(range(3, 80, 4))
        elif split == 3:
            test = list(range(4, 81, 4))
        return list(set(all_classes) - set(test)), test
    return None, None


def get_class_name(dataset, class_id):
    if dataset == "pascal":
        return PASCAL_CLASSES.get(class_id, None)
    return COCO_CLASSES.get(class_id, None)


# ---- 评估指标 ----

class MetricAccumulator:
    def __init__(self):
        self.fg_inter = 0.0
        self.fg_union = 0.0
        self.bg_inter = 0.0
        self.bg_union = 0.0

    def update(self, pred_mask, gt_mask, valid_mask):
        pred_valid = pred_mask & valid_mask
        gt_valid = gt_mask & valid_mask

        self.fg_inter += (pred_valid & gt_valid).sum()
        self.fg_union += (pred_valid | gt_valid).sum()

        pred_bg = valid_mask & (~pred_mask)
        gt_bg = valid_mask & (~gt_mask)
        self.bg_inter += (pred_bg & gt_bg).sum()
        self.bg_union += (pred_bg | gt_bg).sum()

    def fg_iou(self):
        return self.fg_inter / self.fg_union if self.fg_union > 0 else float("nan")

    def miou(self):
        return self.fg_iou()

    def fbiou(self):
        return (self.fg_iou() + (self.bg_inter / self.bg_union if self.bg_union > 0 else 0)) / 2


# ---- 数据加载 ----

def load_sub_class_file_list(path):
    with open(path) as f:
        content = f.read()
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    return ast.literal_eval(content)


def clean_path(s):
    return s.replace("\r", "")


def load_mask(path, class_id):
    mask = np.array(Image.open(path))
    valid = mask != 255
    binary = mask == class_id
    return binary, valid


def resolve_path(list_path, dataset, args):
    if dataset == "coco":
        rel = list_path.replace("../data/MSCOCO2014/", "")
        return os.path.join(args.coco_root, rel)
    else:
        rel = list_path.replace("../data/VOCdevkit2012/VOC2012/", "")
        return os.path.join(args.pascal_root, rel)


# ---- 训练数据池 ----

def build_training_pool(lists_dir, dataset, dataset_root, val_fold):
    """
    从 val_fold 之外的 3 个 fold 的 train split 构建训练池.
    排除 val_fold 的测试类, 避免数据泄漏.
    返回 {class_id: [(img_path, mask_path), ...]}.
    """
    _, test_classes = get_fold_classes(dataset, val_fold)
    val_test_classes = set(test_classes)

    pool = defaultdict(list)
    seen = set()

    for fold in range(4):
        if fold == val_fold:
            continue
        list_path = os.path.join(lists_dir, dataset, "fss_list", "train",
                                 f"sub_class_file_list_{fold}.txt")
        if not os.path.exists(list_path):
            continue

        data = load_sub_class_file_list(list_path)
        for class_id, pairs in data.items():
            if class_id in val_test_classes:
                continue
            for img_rel, mask_rel in pairs:
                img_rel, mask_rel = clean_path(img_rel), clean_path(mask_rel)
                # 构造完整路径
                if dataset == "coco":
                    rel = img_rel.replace("../data/MSCOCO2014/", "")
                    img_path = os.path.join(dataset_root, rel)
                    rel_m = mask_rel.replace("../data/MSCOCO2014/", "")
                    mask_path = os.path.join(dataset_root, rel_m)
                else:
                    rel = img_rel.replace("../data/VOCdevkit2012/VOC2012/", "")
                    img_path = os.path.join(dataset_root, rel)
                    rel_m = mask_rel.replace("../data/VOCdevkit2012/VOC2012/", "")
                    mask_path = os.path.join(dataset_root, rel_m)

                key = (class_id, img_path)
                if key not in seen and os.path.exists(img_path) and os.path.exists(mask_path):
                    seen.add(key)
                    pool[class_id].append((img_path, mask_path))

    # 仅保留 ≥ 6 张的类 (1 张 support + 5 张 query)
    pool = {cid: pairs for cid, pairs in pool.items() if len(pairs) >= 6}
    names = [get_class_name(dataset, cid) for cid in sorted(pool.keys())]
    print(f"  训练池: {len(pool)} 类 ({names}), {sum(len(v) for v in pool.values())} 对")
    return pool


def sample_episode(pool, rng, n_support=1, n_query=5):
    """从池中随机采样一个 episode."""
    class_id = rng.choice(list(pool.keys()))
    pairs = pool[class_id]
    sampled = rng.sample(pairs, n_support + n_query)
    support = sampled[:n_support]
    queries = sampled[n_support:]
    return support, queries, class_id


# ---- 训练 A ----

def train_fold_A(model, adapter, dataset, dataset_root, lists_dir, val_fold, args):
    """为当前 val_fold 训练 A (用其他 3 个 fold 的数据). 返回训练好的 adapter."""
    from sam3.model.sam3_image_processor_seed import Sam3ProcessorSEED

    pool = build_training_pool(lists_dir, dataset, dataset_root, val_fold)
    if not pool:
        print(f"  [警告] 训练池为空, 跳过 fold {val_fold} 的训练")
        return adapter

    processor = Sam3ProcessorSEED(model, adapter, device=args.device, confidence_threshold=0.0)
    optimizer_A = torch.optim.Adam(processor.get_A_params(), lr=args.lr_outer)
    rng = random.Random(args.seed)

    print(f"  训练 fold {val_fold} 的 A ({args.epochs} epochs × {args.episodes_per_epoch} episodes)...")

    for epoch in range(args.epochs):
        processor.adapter.train()
        loss_a_accum = 0.0
        count = 0

        pbar = tqdm(range(args.episodes_per_epoch), desc=f"    Epoch {epoch + 1}/{args.epochs}")
        for ep_idx in pbar:
            _, queries, class_id = sample_episode(pool, rng)
            class_name = get_class_name(dataset, class_id)
            if class_name is None:
                continue

            # ---- query 上更新 A ----
            total_loss_q = 0.0
            valid_count = 0

            for q_img_path, q_mask_path in queries:
                try:
                    q_img = Image.open(q_img_path).convert("RGB")
                    q_mask = np.array(Image.open(q_mask_path))
                except Exception:
                    continue

                state_q = processor.set_image(q_img)
                processor._encode_text(class_name, state_q)

                lang_feat = state_q["backbone_out"]["language_features"]
                adapted_q = processor.adapter.forward_A(lang_feat)
                state_q["backbone_out"]["language_features"] = adapted_q

                state_q = processor._forward_grounding_grad(state_q)

                probs = state_q.get("masks_logits", None)
                scores = state_q.get("scores", [])
                if probs is None or probs.numel() == 0:
                    continue

                probs = probs.squeeze(1)
                if probs.ndim == 2:
                    probs = probs.unsqueeze(0)

                if len(scores) > 1:
                    s_list = [s.item() if hasattr(s, 'item') else s for s in scores]
                    best_idx = int(np.argmax(s_list))
                else:
                    best_idx = 0

                pred = probs[best_idx]
                gt_t = torch.from_numpy(q_mask).to(processor.device)
                valid = gt_t != 255
                gt_fg = (gt_t == class_id) & valid

                loss_q = F.binary_cross_entropy(pred, gt_fg.float(), reduction="mean")
                total_loss_q += loss_q
                valid_count += 1

            if valid_count > 0:
                avg_loss_q = total_loss_q / valid_count
                optimizer_A.zero_grad()
                avg_loss_q.backward()
                optimizer_A.step()
                loss_a_accum += avg_loss_q.item()

            count += 1

            pbar.set_postfix({"loss_A": f"{loss_a_accum / count:.4f}"})

        print(f"    Epoch {epoch + 1}: loss_A={loss_a_accum / count:.6f}")

    # 清理
    del processor
    torch.cuda.empty_cache()
    return adapter


def best_mask_from_result(result, target_shape=None):
    """从 predict_query 结果中提取最高置信度 mask (bool numpy), 必要时 resize."""
    masks = result.get("masks", [])
    scores = result.get("scores", [])
    if len(masks) == 0:
        return None
    best_idx = 0
    if len(scores) > 1:
        s_list = [s.item() if hasattr(s, 'item') else s for s in scores]
        best_idx = int(np.argmax(s_list))
    pred = masks[best_idx].detach().cpu().numpy().squeeze().astype(bool)
    if target_shape is not None and pred.shape != tuple(target_shape):
        pred = np.array(Image.fromarray(pred).resize(
            (target_shape[1], target_shape[0]), Image.NEAREST))
    return pred


def adapt_fc2_predict(processor, adapter, sup_img, sup_mask, qry_img,
                      class_id, class_name, steps, lr):
    """消融: 用 support 图微调 A 的 fc2, 预测 query, 最后恢复原始权重 (异常也保证恢复)."""
    saved_state = {k: v.clone() for k, v in adapter.block_a.state_dict().items()}
    try:
        fc2_params = [adapter.block_a.fc2.weight, adapter.block_a.fc2.bias]
        optimizer = torch.optim.SGD(fc2_params, lr=lr)

        for step in range(steps):
            state_s = processor.set_image(sup_img)
            processor._encode_text(class_name, state_s)
            lang_feat = state_s["backbone_out"]["language_features"]
            state_s["backbone_out"]["language_features"] = processor.adapter.forward_A(lang_feat)
            state_s = processor._forward_grounding_grad(state_s)

            probs = state_s.get("masks_logits", None)
            scores = state_s.get("scores", [])
            if probs is None or probs.numel() == 0:
                continue
            probs = probs.squeeze(1)
            if probs.ndim == 2:
                probs = probs.unsqueeze(0)

            if len(scores) > 1:
                s_list = [s.item() if hasattr(s, 'item') else s for s in scores]
                best_idx = int(np.argmax(s_list))
            else:
                best_idx = 0

            pred = probs[best_idx]
            gt_t = torch.from_numpy(sup_mask).to(processor.device)
            valid = gt_t != 255
            gt_fg = (gt_t == class_id) & valid
            loss = F.binary_cross_entropy(pred, gt_fg.float(), reduction="mean")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return processor.predict_query(qry_img, class_name)
    finally:
        adapter.block_a.load_state_dict(saved_state)


# ---- 评测一个 fold ----

def evaluate_fold(model, adapter, dataset, fold, args):
    """在当前 fold 上评测 (可选 --adapt_fc2 微调 fc2 / --mode 切换 B 消融)."""
    from sam3.model.sam3_image_processor_seed import Sam3ProcessorSEED

    _, test_classes = get_fold_classes(dataset, fold)

    lists_dir = os.path.join(args.lists_dir, dataset, "fss_list")
    val_list_path = os.path.join(lists_dir, "val", f"sub_class_file_list_{fold}.txt")
    if not os.path.exists(val_list_path):
        print(f"  [跳过] Fold {fold}: {val_list_path} 不存在")
        return {}, None, None

    sub_class_data = load_sub_class_file_list(val_list_path)

    # 构建 query 池
    query_pool = {}
    for class_id in test_classes:
        pairs = sub_class_data.get(class_id, [])
        valid_pairs = []
        for img_rel, mask_rel in pairs:
            img_rel, mask_rel = clean_path(img_rel), clean_path(mask_rel)
            if os.path.exists(resolve_path(img_rel, dataset, args)) and \
               os.path.exists(resolve_path(mask_rel, dataset, args)):
                valid_pairs.append((img_rel, mask_rel))
        if valid_pairs:
            query_pool[class_id] = valid_pairs

    if not query_pool:
        print(f"  Fold {fold}: 无有效数据")
        return {}, None, None

    print(f"\n{'=' * 60}")
    test_names = [get_class_name(dataset, cid) for cid in sorted(query_pool.keys())]
    test_counts = {cid: len(pairs) for cid, pairs in query_pool.items()}
    print(f"Fold {fold} | 测试类别: {test_names}")
    print(f"  各类图片数: {test_counts}")
    print(f"{'=' * 60}")

    processor = Sam3ProcessorSEED(model, adapter, device=args.device,
                                   confidence_threshold=0.0)
    rng = random.Random(42)
    num_episodes = args.test_episodes

    # 检查 A 的 fc2 是否真的训练到了
    fc2_w_norm = adapter.block_a.fc2.weight.data.norm().item()
    fc2_b_norm = adapter.block_a.fc2.bias.data.norm().item()
    print(f"  [调试] A.fc2 weight norm: {fc2_w_norm:.6f}, bias norm: {fc2_b_norm:.6f}")

    fold_accum = MetricAccumulator()
    cls_accumulators = {cid: MetricAccumulator() for cid in query_pool}

    first_exc = None
    for ep in tqdm(range(num_episodes), desc=f"  评测 Fold {fold}"):
        class_id = rng.choice(list(query_pool.keys()))
        class_name = get_class_name(dataset, class_id)
        pool = query_pool[class_id]

        sampled = rng.sample(pool, 2)
        sup_img_rel, sup_mask_rel = sampled[0]
        qry_img_rel, qry_mask_rel = sampled[1]

        sup_img_path = resolve_path(sup_img_rel, dataset, args)
        sup_mask_path = resolve_path(sup_mask_rel, dataset, args)
        qry_img_path = resolve_path(qry_img_rel, dataset, args)
        qry_mask_path = resolve_path(qry_mask_rel, dataset, args)

        try:
            # ---- Query 预测 (可选: 先微调 fc2 / B 内循环) ----
            qry_img = Image.open(qry_img_path).convert("RGB")
            gt_binary, valid_mask = load_mask(qry_mask_path, class_id)

            if args.adapt_fc2:
                sup_img = Image.open(sup_img_path).convert("RGB")
                sup_mask = np.array(Image.open(sup_mask_path))
                result = adapt_fc2_predict(processor, adapter, sup_img, sup_mask,
                                           qry_img, class_id, class_name,
                                           args.adapt_steps, args.adapt_lr)
            elif args.mode != "a":
                sup_img = Image.open(sup_img_path).convert("RGB")
                sup_mask = np.array(Image.open(sup_mask_path))
                processor.reset_episode()

                state_s = processor.set_image(sup_img)
                processor._encode_text(class_name, state_s)
                lang_feat_orig_s = state_s["backbone_out"]["language_features"]

                for step in range(args.K_inner):
                    state_s["backbone_out"]["language_features"] = lang_feat_orig_s
                    loss_s = processor.inner_loop_step(state_s, sup_mask, class_id,
                                                       mode=args.mode)
                    if loss_s.item() == 0.0:
                        break
                    b_params = processor.get_B_params()
                    grads = torch.autograd.grad(loss_s, b_params, retain_graph=False)
                    for p, g in zip(b_params, grads):
                        if g is not None:
                            p.data = p.data - args.lr_inner * g

                result = processor.predict_query(qry_img, class_name, mode=args.mode)
            else:
                result = processor.predict_query(qry_img, class_name)
            pred_np = best_mask_from_result(result, gt_binary.shape)
            if pred_np is None:
                continue

            cls_accumulators[class_id].update(pred_np, gt_binary, valid_mask)
            fold_accum.update(pred_np, gt_binary, valid_mask)

        except Exception as e:
            if first_exc is None:
                first_exc = e
            continue

    if first_exc is not None:
        print("  [调试] 评测中出现异常 (episode 被跳过), 首个异常堆栈:")
        traceback.print_exception(type(first_exc), first_exc, first_exc.__traceback__)

    # 汇总
    for cid in sorted(cls_accumulators.keys()):
        acc = cls_accumulators[cid]
        if acc.fg_union > 0:
            print(f"  类别 {cid} ({get_class_name(dataset, cid)}): mIoU={acc.miou():.4f}")
        else:
            print(f"  类别 {cid} ({get_class_name(dataset, cid)}): *** 无结果 ***")

    cls_mious = {cid: acc.miou() for cid, acc in cls_accumulators.items() if acc.fg_union > 0}
    fold_miou = np.mean(list(cls_mious.values())) if cls_mious else 0.0
    fold_fbiou = fold_accum.fbiou() if fold_accum.fg_union > 0 else 0.0
    print(f"\n  Fold {fold} mIoU: {fold_miou:.4f}  FB-IoU: {fold_fbiou:.4f}")

    del processor
    torch.cuda.empty_cache()
    return cls_mious, fold_miou, fold_fbiou


# ---- 主函数 ----

def main():
    parser = argparse.ArgumentParser(description="SAM3 + SEED 适配器 FSS 评测 (训练+评估)")
    # 数据
    parser.add_argument("--coco_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/MSCOCO2014")
    parser.add_argument("--pascal_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/pascal/VOCdevkit/VOC2012")
    parser.add_argument("--lists_dir", type=str, default="/root/SAM_clone/lists")
    # 模型
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM3 权重路径")
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "coco", "both"])
    parser.add_argument("--device", type=str, default="cuda")
    # 训练 A
    parser.add_argument("--epochs", type=int, default=1, help="每个 fold 训练 A 的 epoch 数")
    parser.add_argument("--episodes_per_epoch", type=int, default=200)
    parser.add_argument("--lr_outer", type=float, default=1e-4, help="A 学习率")
    parser.add_argument("--seed", type=int, default=42)
    # 保存
    parser.add_argument("--test_episodes", type=int, default=1000, help="每 fold 评测 episode 数")
    parser.add_argument("--adapt_fc2", action="store_true",
                        help="消融: 评测时用 support 图微调 A 的 fc2 (每 episode 微调后预测, 用完恢复)")
    parser.add_argument("--adapt_steps", type=int, default=5, help="fc2 微调步数")
    parser.add_argument("--adapt_lr", type=float, default=0.01, help="fc2 微调学习率")
    parser.add_argument("--mode", type=str, default="a",
                        choices=["a", "b", "ab_parallel", "ab_serial"],
                        help="消融: a=静态A; b=仅B(支持图适应); ab_parallel=并行A+B; ab_serial=串行B(A)")
    parser.add_argument("--K_inner", type=int, default=10, help="B 内循环步数 (mode≠a 时)")
    parser.add_argument("--lr_inner", type=float, default=0.1, help="B 内循环学习率 (mode≠a 时)")
    parser.add_argument("--save_adapters", type=str, default=None,
                        help="保存各 fold A 权重的目录 (默认不保存)")
    parser.add_argument("--load_adapters", type=str, default=None,
                        help="加载已训好的 A 权重目录 (跳过训练, 文件名为 adapter_A_fold{fold}.pt)")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # ---- 加载 SAM3 ----
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.text_adapter_seed import TextAdapterSEED

    print("加载 SAM3...")
    model = build_sam3_image_model(
        bpe_path=os.path.join(os.path.dirname(__file__), "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device=args.device,
        eval_mode=True,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
    )
    for p in model.parameters():
        p.requires_grad = False

    datasets = []
    if args.dataset in ("pascal", "both"):
        datasets.append("pascal")
    if args.dataset in ("coco", "both"):
        datasets.append("coco")

    all_results = {}

    for dataset in datasets:
        dataset_root = args.pascal_root if dataset == "pascal" else args.coco_root
        print(f"\n{'#' * 60}")
        print(f"# {dataset.upper()} SEED 训练 + 评估")
        print(f"{'#' * 60}")

        all_fold_mious = []
        all_fold_fbious = []
        all_per_class = {}

        for fold in range(4):
            print(f"\n{'─' * 60}")
            print(f">>> Fold {fold}")

            # 每 fold 用全新的适配器
            adapter = TextAdapterSEED()
            print(f"  适配器参数量: A={adapter.num_params['A']}, B={adapter.num_params['B']}")

            # ---- 训练或加载 A ----
            if args.load_adapters:
                load_path = os.path.join(args.load_adapters, f"adapter_A_fold{fold}.pt")
                if not os.path.exists(load_path):
                    print(f"  [跳过] 未找到 A 权重: {load_path}")
                    continue
                adapter.block_a.load_state_dict(
                    torch.load(load_path, map_location=args.device)
                )
                print(f"  已加载 A 权重: {load_path}")
            else:
                adapter = train_fold_A(model, adapter, dataset, dataset_root,
                                       args.lists_dir, fold, args)

            if args.save_adapters:
                os.makedirs(args.save_adapters, exist_ok=True)
                save_path = os.path.join(args.save_adapters, f"adapter_A_fold{fold}.pt")
                torch.save(adapter.block_a.state_dict(), save_path)

            # ---- 评测 ----
            cls_mious, fold_miou, fold_fbiou = evaluate_fold(
                model, adapter, dataset, fold, args
            )

            if fold_miou is not None:
                all_fold_mious.append(fold_miou)
                all_fold_fbious.append(fold_fbiou)
                for cid, miou in cls_mious.items():
                    if cid not in all_per_class:
                        all_per_class[cid] = []
                    all_per_class[cid].append(miou)

        # ---- 最终汇总 ----
        print(f"\n{'=' * 60}")
        print(f"  {dataset.upper()} 最终结果")
        print(f"{'=' * 60}")
        mean_miou = np.mean(all_fold_mious) if all_fold_mious else 0.0
        mean_fbiou = np.mean(all_fold_fbious) if all_fold_fbious else 0.0
        print(f"  4-fold 平均 mIoU:   {mean_miou:.4f}")
        print(f"  4-fold 平均 FB-IoU: {mean_fbiou:.4f}")
        print(f"  各 Fold mIoU:   {[f'{x:.4f}' for x in all_fold_mious]}")
        print(f"  各 Fold FB-IoU: {[f'{x:.4f}' for x in all_fold_fbious]}")

        print(f"\n  各类别结果:")
        for cid in sorted(all_per_class.keys()):
            cname = get_class_name(dataset, cid)
            cmiou = np.mean(all_per_class[cid])
            print(f"    {cid:3d} {cname:20s}:  mIoU={cmiou:.4f}")

        all_results[dataset] = {
            "mIoU": float(mean_miou),
            "FB-IoU": float(mean_fbiou),
            "fold_mIoUs": [float(x) for x in all_fold_mious],
            "fold_FB_IoUs": [float(x) for x in all_fold_fbious],
            "per_class_mIoU": {int(k): float(np.mean(v)) for k, v in all_per_class.items()},
        }

    # 保存 JSON
    result_path = os.path.join(os.path.dirname(__file__), f"FSS_sam3_seed_{args.dataset}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存 → {result_path}")


if __name__ == "__main__":
    main()