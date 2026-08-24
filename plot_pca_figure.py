"""
绘制 PASCAL-5i 文本嵌入适配前后 PCA 图 (论文图 4-4-1).

数据来源: figs/pca_coords.csv (由 visualize_text_embedding.py 输出).
与 pca_folds.svg 完全同款, 但细节全部放在顶部可调配置区,
方便针对论文排版逐项调整 (标记样式/字号/标题/图例/坐标轴标签等).

用法:
  python plot_pca_figure.py                # 默认读 figs/pca_coords.csv, 输出 figs/4-4-1.{svg,pdf,png}
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ============================================================
# 可调配置区 (改这里即可调整细节, 无需动下面代码)
# ============================================================

CSV_PATH = os.path.join("figs", "pca_coords.csv")   # 数据来源
OUT_DIR = "figs"                                     # 输出目录
OUT_NAME = "4-4-1"                                   # 输出文件名 (不含后缀)
OUTPUT_FORMATS = ["svg", "pdf", "png"]               # 输出格式

# --- 坐标轴标签 (方差占比取自 pca_folds.svg, 如需更新需重算全局 PCA) ---
EVAR = {"PC1": 14.6, "PC2": 11.9}

# --- 字号 (期刊要求 >=8pt) ---
BASE_FONT_SIZE = 18
LEGEND_FONT_SIZE = 14

# --- 标记样式: 每类 vanilla=A 适配前, A-only=适配后 ---
VANILLA_MARKER = "o"        # vanilla 形状 (空心)
VANILLA_EDGE_WIDTH = 1.5    # vanilla 空心圈线宽
ADAPTED_MARKER = "^"        # A-only 形状 (实心)
MARKER_SIZE = 40            # 散点面积 (s= 参数, 调小避免遮挡箭头)
ARROW_LW = 1.2              # 箭头线宽
ARROW_STYLE = "->"          # 箭头样式

# --- 子图标题 / 整图 ---
FOLD_TITLE = "Fold {fold} (classes {first}-{last})"
SUPTITLE = "PASCAL-5i text embeddings: Vanilla vs Branch A (global PCA)"
SUPTITLE_FONT_DELTA = 2     # suptitle 相对 BASE_FONT_SIZE 的增量
LEGEND_NCOLS = 5
AX_PAD_RATIO = 0.1          # 坐标轴留白比例
FOLDS = [0, 1, 2, 3]        # 子图顺序

# 注意: 颜色表每类一色 (按 class_id 1..20 取 tab20)
COLORMAP = "tab20"

# --- 颜色方案: False=每类一色, True=同一 fold 的类用同一色 ---
COLOR_BY_FOLD = False
FOLD_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]   # fold 0..3 顺序

# --- 布局: False=2x2 每 fold 一个面板, True=全部类合到一张图 ---
SINGLE_PANEL = False

# --- 是否画 vanilla → A-only 箭头 ---
SHOW_ARROWS = True

# --- 是否把类名标注在点旁边 (代替底部图例) ---
LABEL_POINTS = False
LABEL_OFFSET = (5, 5)        # 标注相对 vanilla 点的像素偏移 (x, y)

# ============================================================
# 读取数据
# ============================================================

def load_coords(path: str):
    """读取 pca_coords.csv, 返回 class_id -> {pc1_v, pc2_v, pc1_a, pc2_a, name, fold}."""
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows[int(r["class_id"])] = {
                "name": r["class_name"],
                "fold": int(r["fold"]),
                "pc1_v": float(r["pc1_vanilla"]),
                "pc2_v": float(r["pc2_vanilla"]),
                "pc1_a": float(r["pc1_a"]),
                "pc2_a": float(r["pc2_a"]),
            }
    return rows


# ============================================================
# 绘图
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PASCAL-5i 文本嵌入 PCA 图")
    parser.add_argument("--out-name", type=str, default=OUT_NAME,
                        help="输出文件名 (不含后缀)")
    parser.add_argument("--single", action="store_true",
                        help="全部类合到一张图 (覆盖 SINGLE_PANEL)")
    parser.add_argument("--no-arrows", action="store_true",
                        help="不画箭头 (覆盖 SHOW_ARROWS)")
    parser.add_argument("--label-points", action="store_true",
                        help="把类名标在点旁边 (覆盖 LABEL_POINTS)")
    parser.add_argument("--fold-color", action="store_true",
                        help="同一 fold 同一色 (覆盖 COLOR_BY_FOLD)")
    args = parser.parse_args()
    out_name = args.out_name
    single = SINGLE_PANEL or args.single
    show_arrows = SHOW_ARROWS and not args.no_arrows
    label_points = LABEL_POINTS or args.label_points
    color_by_fold = COLOR_BY_FOLD or args.fold_color

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = load_coords(CSV_PATH)

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
    plt.rcParams["font.size"] = BASE_FONT_SIZE

    colors = plt.get_cmap(COLORMAP).colors

    all_v = np.array([[rows[c]["pc1_v"], rows[c]["pc2_v"]] for c in sorted(rows)])
    all_a = np.array([[rows[c]["pc1_a"], rows[c]["pc2_a"]] for c in sorted(rows)])
    all_xy = np.vstack([all_v, all_a])
    xmin, xmax = all_xy[:, 0].min(), all_xy[:, 0].max()
    ymin, ymax = all_xy[:, 1].min(), all_xy[:, 1].max()
    pad = AX_PAD_RATIO * max(xmax - xmin, ymax - ymin)

    def draw_class(ax, cid):
        r = rows[cid]
        c = FOLD_COLORS[r["fold"]] if color_by_fold else colors[cid - 1]
        vx, vy = r["pc1_v"], r["pc2_v"]
        axx, axy = r["pc1_a"], r["pc2_a"]
        ax.scatter(vx, vy, marker=VANILLA_MARKER, facecolors="none",
                   edgecolors=c, s=MARKER_SIZE, linewidths=VANILLA_EDGE_WIDTH)
        ax.scatter(axx, axy, marker=ADAPTED_MARKER, color=c, s=MARKER_SIZE)
        if show_arrows:
            ax.annotate("", xy=(axx, axy), xytext=(vx, vy),
                        arrowprops=dict(arrowstyle=ARROW_STYLE, color=c, lw=ARROW_LW))
        if label_points:
            ax.annotate(rows[cid]["name"], xy=(vx, vy), xytext=LABEL_OFFSET,
                        textcoords="offset points", fontsize=LEGEND_FONT_SIZE)

    if single:
        fig, ax = plt.subplots(figsize=(9, 9))
        for cid in sorted(rows):
            draw_class(ax, cid)
        ax.set_xlim(xmin - pad, xmax + pad)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"PC1 ({EVAR['PC1']:.1f}%)")
        ax.set_ylabel(f"PC2 ({EVAR['PC2']:.1f}%)")
        rect = [0, 0.12, 1, 0.95]
    else:
        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        for idx, fold in enumerate(FOLDS):
            ax = axes[idx // 2][idx % 2]
            cids = [c for c in sorted(rows) if rows[c]["fold"] == fold]
            for cid in cids:
                draw_class(ax, cid)
            ax.set_title(FOLD_TITLE.format(fold=fold, first=cids[0], last=cids[-1]),
                         fontsize=BASE_FONT_SIZE)
            ax.set_xlim(xmin - pad, xmax + pad)
            ax.set_ylim(ymin - pad, ymax + pad)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(f"PC1 ({EVAR['PC1']:.1f}%)")
            ax.set_ylabel(f"PC2 ({EVAR['PC2']:.1f}%)")
        rect = [0, 0.05, 1, 0.97]

    if label_points:
        rect = [rect[0], 0.02, rect[2], rect[3]]
    elif color_by_fold:
        handles = [plt.Line2D([0], [0], marker="s", color=FOLD_COLORS[fold],
                              linestyle="none", markersize=12,
                              label=f"Fold {fold}")
                   for fold in FOLDS]
        handles += [
            plt.Line2D([0], [0], marker="o", markerfacecolor="none",
                       markeredgecolor="k", linestyle="none", markersize=12,
                       label="vanilla"),
            plt.Line2D([0], [0], marker="^", color="k", linestyle="none",
                       markersize=12, label="A-only"),
        ]
        if single:
            ax.legend(handles=handles, loc="lower right", fontsize=LEGEND_FONT_SIZE)
        else:
            fig.legend(handles=handles, loc="lower right",
                       fontsize=LEGEND_FONT_SIZE)
    else:
        handles = [plt.Line2D([0], [0], marker=VANILLA_MARKER, color="w",
                              markerfacecolor=colors[cid - 1], markersize=9,
                              label=f"{cid} {rows[cid]['name']}") for cid in sorted(rows)]
        fig.legend(handles=handles, loc="lower center", ncol=LEGEND_NCOLS,
                   fontsize=LEGEND_FONT_SIZE, frameon=False)
    fig.suptitle(SUPTITLE, fontsize=BASE_FONT_SIZE + SUPTITLE_FONT_DELTA)
    fig.tight_layout(rect=rect)

    for fmt in OUTPUT_FORMATS:
        out_path = os.path.join(OUT_DIR, f"{out_name}.{fmt}")
        fig.savefig(out_path)
        print(f"已保存 → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
