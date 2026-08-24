"""
偏移敏感度实验: 对学到的偏移 Δ = A(x) - x 施加缩放系数 α, 观察 mIoU / top1 置信度 / AP 的变化.

核心逻辑: 加载已保存的 adapter_A_fold{0..3}.pt (--save_adapters 输出), 对每个 α:
    x_scaled = x + alpha * (A(x) - x)
    用 x_scaled 替代语言特征走完整 grounding 前向.
  α=1 即完整 DOSE, α=0 即原生 SAM3. 同一批 episode (rng seed 42), 保证 α 间可比.

用法 (在服务器上运行):
  python exp_offset_sensitivity.py --checkpoint /root/checkpoints/sam3.pt \
      --adapter_dir /root/SAM_clone/adapters_A --dataset pascal \
      --test_episodes 100 --device cuda --out_csv /root/SAM_clone/exp_out/sens_pascal.csv

输出: CSV 每行一个 (fold, alpha), 含 miou / top1_conf / ap.
"""

import argparse
import csv
import os
import random
import traceback

import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from tqdm import tqdm

from sam3.model.text_adapter_seed import TextAdapterSEED
from sam3.model.sam3_image_processor_seed import Sam3ProcessorSEED

# ---- 复制的辅助函数 (与 FSS_sam3_seed.py 保持一致) ----

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


def get_class_name(dataset, class_id):
    if dataset == "pascal":
        return PASCAL_CLASSES.get(class_id, None)
    return COCO_CLASSES.get(class_id, None)


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


# ---- 指标 ----

def compute_mask_iou(pred_bool, gt_binary, valid_mask):
    """逐 mask 计算与 GT 的 IoU (只在前景上, 忽略 255)."""
    pred = pred_bool & valid_mask
    gt = gt_binary & valid_mask
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    if union == 0:
        return 0.0
    return inter / union


def average_precision(scores, labels):
    """scores: 置信度列表; labels: 是否正确的 0/1 列表. 返回 AP (0-1)."""
    if not labels or sum(labels) == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores), kind="stable")
    s = np.asarray(scores)[order]
    l = np.asarray(labels)[order]
    n_pos = l.sum()
    tp = np.cumsum(l)
    fp = np.cumsum(1 - l)
    recall = tp / n_pos
    precision = tp / (tp + fp + 1e-10)
    # 11-point AP 或面积法: 用面积法 (积分)
    ap = 0.0
    prev_r = 0.0
    for r, p in zip(recall, precision):
        ap += (r - prev_r) * p
        prev_r = r
    return ap


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


# ---- 模型加载 (复制自 FSS_sam3_seed.py 的 main 逻辑) ----

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


# ---- 偏移缩放预测 (核心) ----

def predict_query_scaled(processor, image, prompt, alpha):
    """
    带 α 缩放的 query 预测:
       x = 语言特征;  x_scaled = x + alpha * (A(x) - x)
    """
    state = processor.set_image(image)
    processor._encode_text(prompt, state)
    x = state["backbone_out"]["language_features"]
    with torch.no_grad():
        xA = processor.adapter.forward_A(x)
        x_scaled = x + alpha * (xA - x)
    state["backbone_out"]["language_features"] = x_scaled
    state = processor._forward_grounding(state)
    return state


def best_mask_from_state(state, target_shape):
    masks = state.get("masks", [])
    scores = state.get("scores", [])
    logits = state.get("masks_logits", [])
    if len(masks) == 0:
        return None, None, None
    best_idx = 0
    if len(scores) > 1:
        s_list = [s.item() if hasattr(s, 'item') else s for s in scores]
        best_idx = int(np.argmax(s_list))
    pred = masks[best_idx].detach().cpu().numpy().squeeze().astype(bool)
    if target_shape is not None and pred.shape != tuple(target_shape):
        pred = np.array(Image.fromarray(pred).resize(
            (target_shape[1], target_shape[0]), Image.NEAREST))
    top1_conf = scores[best_idx].item() if hasattr(scores[best_idx], 'item') else float(scores[best_idx])
    # masks_logits 为连续概率, 用于 AP: 用 best mask 的 mean logit 作为检测置信度
    mlogit = logits[best_idx].detach().cpu().numpy().squeeze()
    ap_score = float(mlogit.mean()) if mlogit.size > 0 else 0.0
    return pred, top1_conf, ap_score


# ---- 单 fold 单 alpha 评测 ----

def evaluate_fold_alpha(model, adapter, dataset, data_args, lists_dir, fold, alpha,
                        test_episodes, device):
    _, test_classes = get_fold_classes(dataset, fold)
    lists_dir_f = os.path.join(lists_dir, dataset, "fss_list")
    val_list_path = os.path.join(lists_dir_f, "val", f"sub_class_file_list_{fold}.txt")
    if not os.path.exists(val_list_path):
        print(f"  [跳过] Fold {fold}: {val_list_path} 不存在")
        return None

    sub_class_data = load_sub_class_file_list(val_list_path)
    query_pool = {}
    for class_id in test_classes:
        pairs = sub_class_data.get(class_id, [])
        valid_pairs = []
        for img_rel, mask_rel in pairs:
            img_rel, mask_rel = clean_path(img_rel), clean_path(mask_rel)
            if os.path.exists(resolve_path(img_rel, dataset, data_args)) and \
               os.path.exists(resolve_path(mask_rel, dataset, data_args)):
                valid_pairs.append((img_rel, mask_rel))
        if valid_pairs:
            query_pool[class_id] = valid_pairs

    if not query_pool:
        return None

    processor = Sam3ProcessorSEED(model, adapter, device=device, confidence_threshold=0.0)
    rng = random.Random(42)  # 固定种子: 保证不同 α 用同一批 episode
    num_episodes = test_episodes

    cls_accum = {cid: MetricAccumulator() for cid in query_pool}
    fold_accum = MetricAccumulator()
    conf_list = []
    ep_iou_list = []
    ap_score_list = []
    ap_label_list = []

    first_exc = None
    for _ in range(num_episodes):
        class_id = rng.choice(list(query_pool.keys()))
        class_name = get_class_name(dataset, class_id)
        pool = query_pool[class_id]
        sampled = rng.sample(pool, 2)
        qry_img_rel, qry_mask_rel = sampled[1]  # 只取 query, 不用 support
        qry_img_path = resolve_path(qry_img_rel, dataset, data_args)
        qry_mask_path = resolve_path(qry_mask_rel, dataset, data_args)

        try:
            qry_img = Image.open(qry_img_path).convert("RGB")
            gt_binary, valid_mask = load_mask(qry_mask_path, class_id)

            state = predict_query_scaled(processor, qry_img, class_name, alpha)
            pred_np, top1_conf, ap_score = best_mask_from_state(state, gt_binary.shape)
            if pred_np is None:
                continue

            cls_accum[class_id].update(pred_np, gt_binary, valid_mask)
            fold_accum.update(pred_np, gt_binary, valid_mask)
            conf_list.append(top1_conf)

            # 逐 episode IoU (用于 100 个 episode 的 mean±std)
            iou = compute_mask_iou(pred_np, gt_binary, valid_mask)
            ep_iou_list.append(iou)
            ap_score_list.append(ap_score)
            ap_label_list.append(1 if iou >= 0.5 else 0)

        except Exception as e:
            if first_exc is None:
                first_exc = e
            continue

    if first_exc is not None:
        print("  [调试] 评测异常 (episode 被跳过):")
        traceback.print_exception(type(first_exc), first_exc, first_exc.__traceback__)

    cls_mious = {cid: acc.miou() for cid, acc in cls_accum.items() if acc.fg_union > 0}
    fold_miou = float(np.mean(list(cls_mious.values()))) if cls_mious else 0.0
    mean_conf = float(np.mean(conf_list)) if conf_list else 0.0
    ap = average_precision(ap_score_list, ap_label_list)

    # 100 个 episode 的均值与标准差
    ep_miou = float(np.mean(ep_iou_list)) if ep_iou_list else 0.0
    ep_miou_std = float(np.std(ep_iou_list)) if len(ep_iou_list) > 1 else 0.0
    conf_std = float(np.std(conf_list)) if len(conf_list) > 1 else 0.0

    del processor
    torch.cuda.empty_cache()
    return ep_miou, ep_miou_std, mean_conf, conf_std, ap


# ---- 主流程 ----

def main():
    parser = argparse.ArgumentParser(description="偏移敏感度实验 (α 缩放 Δ)")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="含 adapter_A_fold{0..3}.pt 的目录 (save_adapters 输出)")
    parser.add_argument("--dataset", type=str, default="pascal", choices=["pascal", "coco"])
    parser.add_argument("--coco_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/MSCOCO2014")
    parser.add_argument("--pascal_root", type=str,
                        default="/root/autodl-tmp/datasets/FSS_datasets/pascal/VOCdevkit/VOC2012")
    parser.add_argument("--lists_dir", type=str, default="/root/SAM_clone/lists")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--test_episodes", type=int, default=500)
    parser.add_argument("--alphas", type=str, default="-1.0,-0.5,0.0,0.25,0.5,0.75,1.0,1.5,2.0",
                        help="逗号分隔的 α 列表")
    parser.add_argument("--folds", type=str, default="0",
                        help="评测哪些 fold (默认只跑 fold 0, 偏移敏感度是趋势分析, 单个 fold 足够)")
    parser.add_argument("--out_csv", type=str, default="offset_sensitivity.csv")
    args = parser.parse_args()

    alphas = [float(a) for a in args.alphas.split(",")]
    folds = [int(f) for f in args.folds.split(",")]

    print(f"加载 SAM3: {args.checkpoint}")
    model = load_sam3(args.checkpoint, args.device)

    rows = []
    for fold in folds:
        adapter_path = os.path.join(args.adapter_dir, f"adapter_A_fold{fold}.pt")
        if not os.path.exists(adapter_path):
            print(f"  [跳过] 未找到 {adapter_path}")
            continue
        adapter = TextAdapterSEED().to(args.device)
        adapter.block_a.load_state_dict(torch.load(adapter_path, map_location=args.device))
        print(f"\n=== Fold {fold} 使用 {adapter_path} ===")

        for alpha in alphas:
            print(f"  α={alpha:+.2f} ...", flush=True)
            res = evaluate_fold_alpha(
                model, adapter, args.dataset, args, args.lists_dir, fold, alpha,
                args.test_episodes, args.device)
            if res is None:
                continue
            miou, miou_std, conf, conf_std, ap = res
            print(f"    mIoU={miou:.4f}±{miou_std:.4f}  "
                  f"top1_conf={conf:.4f}±{conf_std:.4f}  AP={ap:.4f}")
            rows.append({
                "fold": fold, "alpha": alpha,
                "miou": round(miou, 4), "miou_std": round(miou_std, 4),
                "top1_conf": round(conf, 4), "conf_std": round(conf_std, 4),
                "ap": round(ap, 4),
            })

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "alpha", "miou", "miou_std",
                                               "top1_conf", "conf_std", "ap"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n结果已保存 → {args.out_csv}")


if __name__ == "__main__":
    main()
