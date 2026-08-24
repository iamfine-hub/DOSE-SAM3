"""
跨数据集泛化实验: 用 PASCAL 数据训练 A → 测 COCO; 用 COCO 数据训练 A → 测 PASCAL.

目标: 区分"通用域偏移" vs "数据集特定修正".

关键设计 (排除测试 fold 类名):
  对每个目标 fold k, 用"源数据集"构建训练池时, 额外排除那些"类名与目标数据集
  该 fold 测试类重叠"的源类. 这样训练出的 A 对目标 fold 的每个测试类名都真正未见,
  是严格的跨类/跨域泛化测试.

  · PASCAL→COCO: 训练类 = PASCAL 的类, 剔除 "类名 ∈ COCO fold k 测试类名" 的类.
  · COCO→PASCAL: 训练类 = COCO 的类, 剔除 "类名 ∈ PASCAL fold k 测试类名" 的类.
  · 每个方向每个 fold 训练一个独立的 A, 测对应 fold.

用法 (服务器, 会训练):
  python exp_cross_dataset.py --checkpoint /root/checkpoints/sam3.pt \
      --dataset_source pascal --dataset_target coco \
      --coco_root ... --pascal_root ... --lists_dir ... \
      --epochs 1 --episodes_per_epoch 200 --lr_outer 1e-4 --seed 42 \
      --test_episodes 100 --device cuda --folds 0 \
      --out_dir /root/SAM_clone/exp_out

说明:
  · 本脚本独立训练 + 测试, 不依赖任何已保存的权重.
  · 与主实验协议一致: 训练池=源数据集其他 fold 的 train split, 排除目标 fold 测试类名.
  · 输出: 每方向每 fold 一行 CSV (miou / overlap 分组), 并打印逐类结果.
"""

import argparse
import csv
import os
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm

from sam3.model.text_adapter_seed import TextAdapterSEED
from sam3.model.sam3_image_processor_seed import Sam3ProcessorSEED

# ---- 类表 ----

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


def get_fold_classes(dataset, split):
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


def class_names(dataset):
    return PASCAL_CLASSES if dataset == "pascal" else COCO_CLASSES


# ---- 路径与数据 ----

def load_sub_class_file_list(path):
    import ast
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


# ---- 训练池 (复制自 FSS_sam3_seed.py, 增加 extra_exclude_names) ----

def build_training_pool(lists_dir, dataset, dataset_root, val_fold, extra_exclude_names=None):
    """
    从 val_fold 之外的 3 个 fold 的 train split 构建训练池.
    extra_exclude_names: 额外排除的"类名"集合 (用于跨数据集剔除重叠类名).
    """
    _, test_classes = get_fold_classes(dataset, val_fold)
    val_test_classes = set(test_classes)
    if extra_exclude_names:
        # 把"类名"映射回本数据集的 class_id 并排除
        names_map = class_names(dataset)
        for cid, name in names_map.items():
            if name in extra_exclude_names:
                val_test_classes.add(cid)

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

    pool = {cid: pairs for cid, pairs in pool.items() if len(pairs) >= 6}
    return pool


def sample_episode(pool, rng, n_support=1, n_query=5):
    class_id = rng.choice(list(pool.keys()))
    pairs = pool[class_id]
    sampled = rng.sample(pairs, n_support + n_query)
    return sampled[:n_support], sampled[n_support:], class_id


# ---- 训练 A (复制自 FSS_sam3_seed.py) ----

def train_A(model, dataset, dataset_root, lists_dir, val_fold, extra_exclude_names, args):
    """为跨数据集实验训练 A: 源数据集上训练, 排除目标 fold 测试类名."""
    pool = build_training_pool(lists_dir, dataset, dataset_root, val_fold,
                               extra_exclude_names=extra_exclude_names)
    if not pool:
        print(f"  [警告] 训练池为空, 跳过")
        return None

    adapter = TextAdapterSEED().to(args.device)
    processor = Sam3ProcessorSEED(model, adapter, device=args.device, confidence_threshold=0.0)
    optimizer_A = torch.optim.Adam(processor.get_A_params(), lr=args.lr_outer)
    rng = random.Random(args.seed)

    names = [get_class_name(dataset, cid) for cid in sorted(pool.keys())]
    print(f"  训练类 ({len(pool)}): {names}")

    for epoch in range(args.epochs):
        processor.adapter.train()
        loss_a_accum = 0.0
        count = 0
        pbar = tqdm(range(args.episodes_per_epoch), desc=f"    Epoch {epoch+1}/{args.epochs}")
        for ep_idx in pbar:
            _, queries, class_id = sample_episode(pool, rng)
            class_name = get_class_name(dataset, class_id)
            if class_name is None:
                continue

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
            pbar.set_postfix({"loss_A": f"{loss_a_accum / max(count,1):.4f}"})

        print(f"    Epoch {epoch+1}: loss_A={loss_a_accum / max(count,1):.6f}")

    del processor
    torch.cuda.empty_cache()
    return adapter


# ---- 评测一个 fold ----

class MetricAccumulator:
    def __init__(self):
        self.fg_inter = 0.0
        self.fg_union = 0.0

    def update(self, pred_mask, gt_mask, valid_mask):
        pred_valid = pred_mask & valid_mask
        gt_valid = gt_mask & valid_mask
        self.fg_inter += (pred_valid & gt_valid).sum()
        self.fg_union += (pred_valid | gt_valid).sum()

    def miou(self):
        return self.fg_inter / self.fg_union if self.fg_union > 0 else float("nan")


def load_sam3(checkpoint, device):
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def predict_query_plain(processor, image, prompt):
    state = processor.set_image(image)
    processor._encode_text(prompt, state)
    x = state["backbone_out"]["language_features"]
    with torch.no_grad():
        x_scaled = processor.adapter.forward_A(x)
    state["backbone_out"]["language_features"] = x_scaled
    state = processor._forward_grounding(state)
    return state


def best_mask_from_state(state, target_shape):
    masks = state.get("masks", [])
    scores = state.get("scores", [])
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


def evaluate_fold(model, adapter, target_dataset, args, target_fold, test_episodes):
    _, test_classes = get_fold_classes(target_dataset, target_fold)
    lists_dir_f = os.path.join(args.lists_dir, target_dataset, "fss_list")
    val_list_path = os.path.join(lists_dir_f, "val", f"sub_class_file_list_{target_fold}.txt")
    if not os.path.exists(val_list_path):
        print(f"  [跳过] Fold {target_fold}: {val_list_path} 不存在")
        return {}, None

    sub_class_data = load_sub_class_file_list(val_list_path)
    query_pool = {}
    for class_id in test_classes:
        pairs = sub_class_data.get(class_id, [])
        valid_pairs = []
        for img_rel, mask_rel in pairs:
            img_rel, mask_rel = clean_path(img_rel), clean_path(mask_rel)
            if os.path.exists(resolve_path(img_rel, target_dataset, args)) and \
               os.path.exists(resolve_path(mask_rel, target_dataset, args)):
                valid_pairs.append((img_rel, mask_rel))
        if valid_pairs:
            query_pool[class_id] = valid_pairs

    if not query_pool:
        return {}, None

    processor = Sam3ProcessorSEED(model, adapter, device=args.device, confidence_threshold=0.0)
    rng = random.Random(42)
    cls_accum = {cid: MetricAccumulator() for cid in query_pool}

    first_exc = None
    for _ in range(test_episodes):
        class_id = rng.choice(list(query_pool.keys()))
        class_name = get_class_name(target_dataset, class_id)
        pool = query_pool[class_id]
        sampled = rng.sample(pool, 2)
        qry_img_rel, qry_mask_rel = sampled[1]
        qry_img_path = resolve_path(qry_img_rel, target_dataset, args)
        qry_mask_path = resolve_path(qry_mask_rel, target_dataset, args)

        try:
            qry_img = Image.open(qry_img_path).convert("RGB")
            gt_binary, valid_mask = load_mask(qry_mask_path, class_id)
            state = predict_query_plain(processor, qry_img, class_name)
            pred_np = best_mask_from_state(state, gt_binary.shape)
            if pred_np is None:
                continue
            cls_accum[class_id].update(pred_np, gt_binary, valid_mask)
        except Exception as e:
            if first_exc is None:
                first_exc = e
            continue

    if first_exc is not None:
        print("  [调试] 评测异常 (episode 被跳过):")
        traceback.print_exception(type(first_exc), first_exc, first_exc.__traceback__)

    cls_mious = {cid: acc.miou() for cid, acc in cls_accum.items() if acc.fg_union > 0}
    fold_miou = float(np.mean(list(cls_mious.values()))) if cls_mious else None
    del processor
    torch.cuda.empty_cache()
    return cls_mious, fold_miou


# ---- 主流程 ----

def main():
    parser = argparse.ArgumentParser(description="跨数据集泛化实验 (训练 + 测试)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_source", type=str, required=True,
                        choices=["pascal", "coco"], help="训练 A 用的数据集")
    parser.add_argument("--dataset_target", type=str, required=True,
                        choices=["pascal", "coco"], help="测试 A 用的数据集")
    parser.add_argument("--coco_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/MSCOCO2014")
    parser.add_argument("--pascal_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/pascal/VOCdevkit/VOC2012")
    parser.add_argument("--lists_dir", type=str, default="/root/SAM_clone/lists")
    parser.add_argument("--device", type=str, default="cuda")
    # 训练超参 (与主实验一致)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--episodes_per_epoch", type=int, default=200)
    parser.add_argument("--lr_outer", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    # 测试
    parser.add_argument("--test_episodes", type=int, default=100)
    parser.add_argument("--folds", type=str, default="0,1,2,3")
    parser.add_argument("--out_csv", type=str, default="cross_dataset.csv")
    args = parser.parse_args()

    src, tgt = args.dataset_source, args.dataset_target
    if src == tgt:
        raise SystemExit("source 和 target 不能相同")

    folds = [int(f) for f in args.folds.split(",")]
    src_names = class_names(src)
    tgt_names = class_names(tgt)

    print(f"加载 SAM3: {args.checkpoint}")
    model = load_sam3(args.checkpoint, args.device)

    rows = []
    for fold in folds:
        # 目标 fold 的测试类名
        _, tgt_test = get_fold_classes(tgt, fold)
        tgt_test_names = {get_class_name(tgt, cid) for cid in tgt_test}
        # 训练时需排除的源类: 源类名 ∈ 目标 fold 测试类名
        extra_exclude_names = tgt_test_names

        # 源数据集根目录
        src_root = args.pascal_root if src == "pascal" else args.coco_root

        print(f"\n{'='*60}")
        print(f"Fold {fold}: 用 {src} 训练 (排除类名 {sorted(tgt_test_names)}) → 测 {tgt} fold{fold}")
        print(f"{'='*60}")

        adapter = train_A(model, src, src_root, args.lists_dir, fold,
                          extra_exclude_names, args)
        if adapter is None:
            continue

        cls_mious, fold_miou = evaluate_fold(model, adapter, tgt, args, fold,
                                             args.test_episodes)
        if fold_miou is None:
            continue

        # 逐类打印
        for cid in sorted(cls_mious):
            print(f"    {tgt} 类 {cid} ({get_class_name(tgt, cid)}): mIoU={cls_mious[cid]:.4f}")

        # 按类名是否与源数据集重叠分组
        overlap_mious = [m for c, m in cls_mious.items()
                         if get_class_name(tgt, c) in set(src_names.values())]
        novel_mious = [m for c, m in cls_mious.items()
                       if get_class_name(tgt, c) not in set(src_names.values())]
        ov_mean = float(np.mean(overlap_mious)) if overlap_mious else None
        nv_mean = float(np.mean(novel_mious)) if novel_mious else None
        ov_str = f"{ov_mean:.4f}" if ov_mean is not None else "n/a"
        nv_str = f"{nv_mean:.4f}" if nv_mean is not None else "n/a"
        print(f"  Fold {fold}: {tgt} mIoU={fold_miou:.4f}  (overlap={ov_str}, novel={nv_str})")

        rows.append({
            "src": src, "tgt": tgt, "fold": fold,
            "miou": round(fold_miou, 4),
            "overlap_miou": ov_str if ov_mean is not None else "",
            "novel_miou": nv_str if nv_mean is not None else "",
        })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["src", "tgt", "fold", "miou",
                                               "overlap_miou", "novel_miou"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已保存 → {args.out_csv}")


if __name__ == "__main__":
    main()
