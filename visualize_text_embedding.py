"""
SAM3 文本嵌入在 A 适配前后的可视化 (机制分析).

功能:
  1. PCA 主图: 20 个 PASCAL 类的文本嵌入, vanilla vs A-only 的移动 (4 个 fold 面板, 共享全局 PCA 轴)
  2. 逐 token 残差热力图: 20 类 × 32 token 的 ||Δ|| (Δ = A(x) - x)
  3. 量化指标: 平均类间余弦距离 / 逐类分离度 / 残差相对幅度与旋转角
  4. 20×20 余弦距离矩阵热力图 (vanilla vs A-only)

前提:
  已用 FSS_sam3_seed.py --save_adapters 保存了 adapter_A_fold{0..3}.pt.

用法:
  python visualize_text_embedding.py --checkpoint /root/checkpoints/sam3.pt \
      --adapter_dir ./adapters_A --out_dir ./figs
"""

import argparse
import csv
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 可调配置区 (改类、prompt、阈值、输出格式都在这里)
# ============================================================

# PASCAL-5i 的 20 个类 (class_id -> 名称)
PASCAL_CLASSES = {
    1: "aeroplane", 2: "bicycle", 3: "bird", 4: "boat", 5: "bottle",
    6: "bus", 7: "car", 8: "cat", 9: "chair", 10: "cow",
    11: "dining table", 12: "dog", 13: "horse", 14: "motorbike", 15: "person",
    16: "potted plant", 17: "sheep", 18: "sofa", 19: "train", 20: "tv monitor",
}

# 每个 fold 的测试类 (与 FSS_sam3_seed.py 的 get_fold_classes 一致)
FOLD_CLASSES = {
    0: list(range(1, 6)),     # 1-5
    1: list(range(6, 11)),    # 6-10
    2: list(range(11, 16)),   # 11-15
    3: list(range(16, 21)),   # 16-20
}

# 内容 token 判定: 跨所有 prompt 的逐 token 方差低于此阈值视为 padding
CONTENT_VAR_THRESHOLD = 1e-6

# PCA 主成分数 (固定 2 维投影)
PCA_N_COMPONENTS = 2

# 输出图片格式 (矢量图)
OUTPUT_FORMATS = ["pdf", "svg"]

# 字号 (论文要求 ≥8pt)
BASE_FONT_SIZE = 10
LEGEND_FONT_SIZE = 8


# ---- 模型 / 适配器加载 ----

def load_sam3(checkpoint: str, device: str):
    """加载冻结的 SAM3."""
    from sam3.model_builder import build_sam3_image_model
    model = build_sam3_image_model(
        bpe_path=os.path.join(os.path.dirname(__file__),
                              "sam3/assets/bpe_simple_vocab_16e6.txt.gz"),
        device=device,
        eval_mode=True,
        checkpoint_path=checkpoint,
        load_from_HF=False,
    )
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_adapter(adapter_dir: str, fold: int, device: str):
    """加载第 fold 个 adapter (只取 block_a). 返回 TextAdapterSEED."""
    from sam3.model.text_adapter_seed import TextAdapterSEED
    path = os.path.join(adapter_dir, f"adapter_A_fold{fold}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"找不到 A 权重: {path}\n"
            "请先用 FSS_sam3_seed.py --save_adapters 保存各 fold 的 A。")
    adapter = TextAdapterSEED().to(device)
    adapter.block_a.load_state_dict(torch.load(path, map_location=device))
    return adapter


def text_features(model, prompt: str, device: str) -> torch.Tensor:
    """返回该 prompt 的 language_features, 形状 [32, 1, 256]."""
    text_outputs = model.backbone.forward_text([prompt], device=device)
    return text_outputs["language_features"]


# ---- 池化 / 距离 ----

def content_token_mask(vanilla: dict) -> np.ndarray:
    """检测内容 token: 跨 20 个 prompt 的逐 token 方差低于阈值视为 padding."""
    X = np.stack([vanilla[cid].squeeze(1) for cid in sorted(vanilla)])  # [20,32,256]
    token_var = X.var(axis=0).mean(axis=-1)  # [32]
    return token_var > CONTENT_VAR_THRESHOLD


def pooled(feat: dict, mask: np.ndarray) -> dict:
    """对内容 token 求均值, 得到每类一个 256 维嵌入."""
    return {cid: feat[cid].squeeze(1)[mask].mean(axis=0) for cid in feat}


def pairwise_cosine_distance(emb: dict, class_ids):
    """20 类两两余弦距离矩阵, 返回 (上三角均值, 上三角标准差, 矩阵)."""
    E = np.stack([emb[cid] for cid in class_ids])  # [20,256]
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - E @ E.T                              # [20,20], 余弦距离 ∈ [0,2]
    iu = np.triu_indices(len(class_ids), 1)
    return D[iu].mean(), D[iu].std(), D


def per_class_distance(emb: dict, class_ids) -> np.ndarray:
    """每类到其余 19 类的平均余弦距离, 返回 [20]."""
    E = np.stack([emb[cid] for cid in class_ids])
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - E @ E.T
    off = 1.0 - np.eye(len(class_ids))
    return (D * off).sum(axis=1) / (len(class_ids) - 1)


# ---- 主流程 ----

def main():
    parser = argparse.ArgumentParser(description="SAM3 文本嵌入适配前后可视化")
    parser.add_argument("--checkpoint", type=str, required=True, help="SAM3 权重路径")
    parser.add_argument("--adapter_dir", type=str, required=True,
                        help="--save_adapters 输出的目录 (含 adapter_A_fold{0..3}.pt)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=str, default=".", help="图片输出目录")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    # 论文要求 Times New Roman; 服务器若无该字体, 自动回退到 DejaVu Serif.
    # 需要严格 Times New Roman 时, 在装有该字体的机器上运行, 或用 pca_coords.csv 自行成图.
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = BASE_FONT_SIZE

    class_ids = list(range(1, 21))

    # ---- 1. 提取每类的 vanilla / A-only 特征 ----
    print("加载 SAM3...")
    model = load_sam3(args.checkpoint, device)

    vanilla = {}   # cid -> [32,1,256] 原始
    adapted = {}   # cid -> [32,1,256] 该类所在 fold 的 A 适配后
    residual = {}  # cid -> [32,1,256] Δ = A(x) - x

    with torch.no_grad():
        for fold, cids in FOLD_CLASSES.items():
            adapter = load_adapter(args.adapter_dir, fold, device)
            print(f"  fold {fold} 适配器已加载")
            for cid in cids:
                name = PASCAL_CLASSES[cid]
                x = text_features(model, name, device)   # [32,1,256]
                xA = adapter.forward_A(x)                # [32,1,256]
                vanilla[cid] = x.detach().cpu()
                adapted[cid] = xA.detach().cpu()
                residual[cid] = (xA - x).detach().cpu()

    # ---- 2. 内容 token 检测 + 池化 ----
    mask = content_token_mask(vanilla)
    n_content = int(mask.sum())
    print(f"  内容 token: {n_content}/32, padding: {32 - n_content}/32")
    vanilla_p = pooled(vanilla, mask)
    adapted_p = pooled(adapted, mask)

    # ---- 3. PCA (numpy SVD, 40 个向量共享同一组轴) ----
    data = np.stack(list(vanilla_p.values()) + list(adapted_p.values()))  # [40,256]
    data_c = data - data.mean(axis=0, keepdims=True)
    _, S, Vt = np.linalg.svd(data_c, full_matrices=False)
    proj = data_c @ Vt[:PCA_N_COMPONENTS].T            # [40,2]
    evr = (S[:PCA_N_COMPONENTS] ** 2) / (S ** 2).sum()
    proj_v = proj[:20]                                 # vanilla
    proj_a = proj[20:]                                 # A-only

    all_xy = np.vstack([proj_v, proj_a])
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad = 0.1 * max(xmax - xmin, ymax - ymin)

    # ---- 3.5 输出每类 2D 坐标 (供外部工具自行画图) ----
    coords_path = os.path.join(args.out_dir, "pca_coords.csv")
    with open(coords_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "fold",
                         "pc1_vanilla", "pc2_vanilla", "pc1_a", "pc2_a"])
        for fold, cids in FOLD_CLASSES.items():
            for cid in cids:
                writer.writerow([cid, PASCAL_CLASSES[cid], fold,
                                 f"{proj_v[cid - 1, 0]:.6f}", f"{proj_v[cid - 1, 1]:.6f}",
                                 f"{proj_a[cid - 1, 0]:.6f}", f"{proj_a[cid - 1, 1]:.6f}"])
    print(f"\n每类 2D 坐标已保存 → {coords_path}")
    print(f"  {'id':<3} {'class':<14} {'fold':<5} {'PC1_v':>9} {'PC2_v':>9} "
          f"{'PC1_a':>9} {'PC2_a':>9}")
    for fold, cids in FOLD_CLASSES.items():
        for cid in cids:
            print(f"  {cid:<3} {PASCAL_CLASSES[cid]:<14} {fold:<5} "
                  f"{proj_v[cid - 1, 0]:>9.4f} {proj_v[cid - 1, 1]:>9.4f} "
                  f"{proj_a[cid - 1, 0]:>9.4f} {proj_a[cid - 1, 1]:>9.4f}")

    # ---- 4. 主图: 逐 fold 面板 (颜色=类别, 形状=配置, 箭头=移动) ----
    colors = plt.get_cmap("tab20").colors
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    for idx, (fold, cids) in enumerate(FOLD_CLASSES.items()):
        ax = axes[idx // 2][idx % 2]
        for cid in cids:
            c = colors[cid - 1]
            vx, vy = proj_v[cid - 1]
            axx, axy = proj_a[cid - 1]
            ax.scatter(vx, vy, marker="o", facecolors="none",
                       edgecolors=c, s=90, linewidths=1.5)          # vanilla 空心
            ax.scatter(axx, axy, marker="^", color=c, s=90)          # A-only 实心
            ax.annotate("", xy=(axx, axy), xytext=(vx, vy),
                        arrowprops=dict(arrowstyle="->", color=c, lw=1.2))
        ax.set_title(f"Fold {fold} (classes {cids[0]}-{cids[-1]})",
                     fontsize=BASE_FONT_SIZE)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC1 ({evr[0] * 100:.1f}%)")
        ax.set_ylabel(f"PC2 ({evr[1] * 100:.1f}%)")
    handles = [plt.Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=colors[cid - 1], markersize=7,
                          label=f"{cid} {PASCAL_CLASSES[cid]}") for cid in class_ids]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               fontsize=LEGEND_FONT_SIZE, frameon=False)
    fig.text(0.5, 0.03,
             "circle = vanilla (hollow)   triangle = A-only (filled)   "
             "arrow = shift of each class embedding by A",
             ha="center", fontsize=LEGEND_FONT_SIZE)
    fig.suptitle("PASCAL-5i text embeddings: Vanilla vs Branch A (global PCA)",
                 fontsize=BASE_FONT_SIZE + 2)
    fig.tight_layout(rect=[0, 0.07, 1, 0.97])
    for fmt in OUTPUT_FORMATS:
        fig.savefig(os.path.join(args.out_dir, f"pca_folds.{fmt}"))
    plt.close(fig)

    # ---- 5. 逐 token 残差热力图 (行=类按 fold 分组, 列=32 token) ----
    R = np.stack([residual[cid].squeeze(1) for cid in class_ids])   # [20,32,256]
    R_norm = np.linalg.norm(R, axis=-1)                             # [20,32]
    fig2, ax2 = plt.subplots(figsize=(8, 7))
    im = ax2.imshow(R_norm, aspect="auto", cmap="viridis")
    ax2.set_xticks(range(32))
    ax2.set_yticks(range(20))
    ax2.set_yticklabels([PASCAL_CLASSES[cid] for cid in class_ids],
                        fontsize=LEGEND_FONT_SIZE)
    ax2.set_xlabel("token index")
    ax2.set_title(r"Per-token residual norm $\|\Delta\|$: $\Delta = A(x) - x$")
    for k in (5, 10, 15):                                            # fold 分隔线
        ax2.axhline(k - 0.5, color="white", linewidth=1.2)
    if n_content < 32:
        ax2.axvline(n_content - 0.5, color="red", linestyle="--", linewidth=1.0,
                    label=f"content/padding boundary (n={n_content})")
    fig2.colorbar(im, ax=ax2, label=r"$\|\Delta\|$")
    ax2.legend(fontsize=LEGEND_FONT_SIZE, loc="upper right")
    fig2.tight_layout()
    for fmt in OUTPUT_FORMATS:
        fig2.savefig(os.path.join(args.out_dir, f"token_residual.{fmt}"))
    plt.close(fig2)

    # ---- 6. 量化指标 ----
    m1v, s1v, Dv = pairwise_cosine_distance(vanilla_p, class_ids)
    m1a, s1a, Da = pairwise_cosine_distance(adapted_p, class_ids)
    sep_v = per_class_distance(vanilla_p, class_ids)
    sep_a = per_class_distance(adapted_p, class_ids)

    dx = np.stack([adapted_p[cid] - vanilla_p[cid] for cid in class_ids])
    xv = np.stack([vanilla_p[cid] for cid in class_ids])
    xa = np.stack([adapted_p[cid] for cid in class_ids])
    rel_norm = np.linalg.norm(dx, axis=1) / (np.linalg.norm(xv, axis=1) + 1e-12)
    cos_ang = (xv * xa).sum(axis=1) / (
        np.linalg.norm(xv, axis=1) * np.linalg.norm(xa, axis=1) + 1e-12)
    angle = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))

    lines = []
    lines.append("========== 量化指标 (内容 token 池化) ==========")
    lines.append(f"[指标1] 平均类间余弦距离: vanilla {m1v:.4f} ± {s1v:.4f}  |  "
                 f"A-only {m1a:.4f} ± {s1a:.4f}  |  Δ {m1a - m1v:+.4f}")
    lines.append("")
    lines.append("[指标2] 逐类分离度 (该类到其余 19 类的平均余弦距离):")
    lines.append(f"  {'id':<3} {'class':<14} {'vanilla':>9} {'A-only':>9} {'Δ':>9}")
    for i, cid in enumerate(class_ids):
        lines.append(f"  {cid:<3} {PASCAL_CLASSES[cid]:<14} {sep_v[i]:>9.4f} "
                     f"{sep_a[i]:>9.4f} {sep_a[i] - sep_v[i]:>+9.4f}")
    lines.append("")
    lines.append(f"[指标3] 残差相对幅度 mean(||Δ||/||x||) = {rel_norm.mean():.4f}")
    lines.append(f"[指标3] 旋转角 mean = {angle.mean():.2f}°, "
                 f"范围 [{angle.min():.2f}, {angle.max():.2f}]°")

    metrics_path = os.path.join(args.out_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines))
    print(f"\n量化指标已保存 → {metrics_path}")

    # ---- 7. 20×20 余弦距离矩阵热力图 ----
    vmax = max(Dv.max(), Da.max())
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 7))
    for ax, D, title in zip(axes3, (Dv, Da), ("Vanilla", "A-only")):
        im = ax.imshow(D, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"{title} pairwise cosine distance", fontsize=BASE_FONT_SIZE)
        ax.set_xticks(range(20))
        ax.set_yticks(range(20))
        ax.set_xticklabels([str(c) for c in class_ids], fontsize=LEGEND_FONT_SIZE)
        ax.set_yticklabels([PASCAL_CLASSES[c] for c in class_ids], fontsize=LEGEND_FONT_SIZE)
    fig3.colorbar(im, ax=axes3, label="cosine distance (0-2)")
    fig3.tight_layout()
    for fmt in OUTPUT_FORMATS:
        fig3.savefig(os.path.join(args.out_dir, f"dist_matrix.{fmt}"))
    plt.close(fig3)

    print(f"\n图片已保存到 {args.out_dir}:")
    for fmt in OUTPUT_FORMATS:
        print(f"  pca_folds.{fmt}, token_residual.{fmt}, dist_matrix.{fmt}")


if __name__ == "__main__":
    main()
