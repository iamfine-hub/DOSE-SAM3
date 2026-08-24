"""
绘制跨数据集泛化图: PASCAL-A → COCO 与 COCO-A → PASCAL 的 mIoU (每 fold + 均值).

数据来源: exp_cross_dataset.py 输出的 CSV (可多文件合并, 如 cross_p2c.csv + cross_c2p.csv).
用法:
  python plot_cross_dataset.py \
      --csv_p2c /root/SAM_clone/exp_out/cross_p2c.csv \
      --csv_c2p /root/SAM_clone/exp_out/cross_c2p.csv \
      --out_dir figs --out_name cross_dataset
输出: figs/cross_dataset.pdf / .svg / .png

列格式: src,tgt,fold,miou,overlap_miou,novel_miou
"""

import argparse
import csv
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["font.size"] = 11


def load_csv(path):
    if not path or not os.path.exists(path):
        return []
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def parse_float(v):
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_p2c", type=str, default=None, help="PASCAL→COCO 结果 CSV")
    parser.add_argument("--csv_c2p", type=str, default=None, help="COCO→PASCAL 结果 CSV")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--out_name", type=str, default="cross_dataset")
    args = parser.parse_args()

    # 标注每个方向的原生 SAM3 基线 (同数据集原生)
    BASELINE = {
        "coco": 64.1,    # 原生 SAM3 在 COCO-20i 上的 mIoU (论文表3)
        "pascal": 78.5,  # 原生 SAM3 在 PASCAL-5i 上的 mIoU (论文表3)
    }

    rows_p2c = load_csv(args.csv_p2c)
    rows_c2p = load_csv(args.csv_c2p)
    if not rows_p2c and not rows_c2p:
        raise SystemExit("未找到任何跨数据集结果 CSV, 请先确认路径")

    os.makedirs(args.out_dir, exist_ok=True)

    def collect(rows):
        """返回 {fold: miou} 与均值."""
        d = {}
        for r in rows:
            m = parse_float(r.get("miou"))
            if m is not None:
                d[int(r["fold"])] = m
        mean = float(np.mean(list(d.values()))) if d else None
        return d, mean

    data = []
    if rows_p2c:
        d, mean = collect(rows_p2c)
        data.append(("PASCAL-A → COCO", "coco", d, mean))
    if rows_c2p:
        d, mean = collect(rows_c2p)
        data.append(("COCO-A → PASCAL", "pascal", d, mean))

    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_dirs = len(data)
    width = 0.22
    for i, (label, tgt, d, mean) in enumerate(data):
        base = BASELINE[tgt]
        folds = sorted(d.keys())
        x = np.arange(len(folds)) + i * width
        vals = [d[f] for f in folds]
        bars = ax.bar(x, vals, width=width, label=f"{label} (mean {mean:.2f})")
        # 每根柱顶标数值
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.3, f"{v:.1f}", ha="center", fontsize=9)
        # 原生基线横线
        ax.axhline(base, color=plt.get_cmap("tab10")(i), ls="--", lw=1.2,
                   alpha=0.7)
        ax.text(len(folds) - 0.5 + i * width, base + 0.3,
                f"vanilla SAM3 on {tgt}: {base}", fontsize=9,
                color=plt.get_cmap("tab10")(i), ha="right")

    ax.set_xticks(np.arange(4) + width * (n_dirs - 1) / 2)
    ax.set_xticklabels(["Fold 0", "Fold 1", "Fold 2", "Fold 3"])
    ax.set_ylabel("mIoU (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_title("Cross-dataset generalization: trained A tested on unseen dataset")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    for fmt in ["pdf", "svg", "png"]:
        fig.savefig(os.path.join(args.out_dir, f"{args.out_name}.{fmt}"))
        print(f"已保存 → {args.out_dir}/{args.out_name}.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    main()
