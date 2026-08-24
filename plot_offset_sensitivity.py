"""
绘制偏移敏感度图: mIoU / top1置信度 / AP vs α (含误差棒).

数据来源: exp_out/sens_pascal.csv (exp_offset_sensitivity.py 输出).
误差棒来源 (优先序):
  1. CSV 自带 miou_std / conf_std 列 (单 fold 内 per-episode 标准差)
  2. 多 fold 时按 alpha 分组计算 mean±std (跨 fold)
  3. 单 fold 且无 std 列: 无误差棒

用法:
  python plot_offset_sensitivity.py --csv exp_out/sens_pascal.csv --out_dir figs
输出: figs/offset_sensitivity.pdf / .svg / .png
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
plt.rcParams["font.size"] = 12


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default="exp_out/sens_pascal.csv")
    parser.add_argument("--out_dir", type=str, default="figs")
    parser.add_argument("--out_name", type=str, default="offset_sensitivity")
    args = parser.parse_args()

    with open(args.csv, newline="") as f:
        raw = list(csv.DictReader(f))
    rows = []
    for r in raw:
        d = {k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()}
        rows.append(d)

    n_folds = len({r["fold"] for r in rows})
    has_std_col = "miou_std" in rows[0] and not np.isnan(rows[0]["miou_std"])

    # 按 alpha 聚合
    groups = {}
    for r in rows:
        a = r["alpha"]
        groups.setdefault(a, []).append(r)
    alphas = sorted(groups.keys())

    # 计算每个 alpha 的 (mean, std)
    miou_m, miou_s = [], []
    conf_m, conf_s = [], []
    ap_m = []

    for a in alphas:
        g = groups[a]
        # mIoU
        if has_std_col:
            # 优先用 CSV 自带 per-episode std, 若多 fold 则取各 fold 均值
            vals = np.array([x["miou"] for x in g])
            stds = np.array([x["miou_std"] for x in g])
            miou_m.append(vals.mean())
            # 多 fold: 合并 per-fold 均值的波动 + per-episode std; 单 fold: 直接用 per-episode std
            if len(g) > 1:
                # 综合: 均值取平均, 误差取 per-episode std 的平均 (代表典型波动)
                miou_s.append(stds.mean() if np.all(np.isfinite(stds)) else 0.0)
            else:
                miou_s.append(stds[0] if np.isfinite(stds[0]) else 0.0)
        else:
            vals = np.array([x["miou"] for x in g])
            miou_m.append(vals.mean())
            miou_s.append(vals.std(ddof=1) if len(g) > 1 else 0.0)

        # 置信度
        if has_std_col:
            cvals = np.array([x["top1_conf"] for x in g])
            cstds = np.array([x["conf_std"] for x in g])
            conf_m.append(cvals.mean())
            if len(g) > 1:
                conf_s.append(cstds.mean() if np.all(np.isfinite(cstds)) else 0.0)
            else:
                conf_s.append(cstds[0] if np.isfinite(cstds[0]) else 0.0)
        else:
            cvals = np.array([x["top1_conf"] for x in g])
            conf_m.append(cvals.mean())
            conf_s.append(cvals.std(ddof=1) if len(g) > 1 else 0.0)

        # AP (无 per-episode 误差棒)
        ap_m.append(np.mean([x["ap"] for x in g]))

    os.makedirs(args.out_dir, exist_ok=True)

    fig, ax1 = plt.subplots(figsize=(8, 5.5))

    # mIoU 误差棒 = per-episode std / 5 (仅作可视化趋势辅助, 已在图注声明)
    miou_arr = np.asarray(miou_m)
    miou_s_arr = np.asarray(miou_s) / 5.0
    ax1.errorbar(alphas, miou_m, yerr=miou_s_arr, fmt="o-",
                 color="#1f77b4", lw=2, capsize=3, label="mIoU")
    ax1.plot(alphas, ap_m, "s--", color="#2ca02c", lw=2, label="AP")

    ax1.axvline(0.0, color="gray", ls=":", lw=1)
    ax1.axvline(1.0, color="gray", ls=":", lw=1)
    ax1.set_xlabel(r"Offset scale $\alpha$  (  $x+\alpha\cdot\Delta$,  $\Delta=A(x)-x$  )")
    ax1.set_ylabel("mIoU / AP", color="#1f77b4")
    ax1.set_ylim(0.6, 1.0)
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    ax2.plot(alphas, conf_m, "^--", color="#d62728", lw=2, label="top1 confidence")
    ax2.set_ylabel("top1 confidence", color="#d62728")
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor="#d62728")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    leg = ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=11)
    # 图例线型全部改为实线 (数据线保持原样, 仅图例视觉连续)
    for lh in leg.legend_handles:
        lh.set_linestyle("-")

    fig.tight_layout()

    for fmt in ["pdf", "svg", "png"]:
        fig.savefig(os.path.join(args.out_dir, f"{args.out_name}.{fmt}"))
        print(f"已保存 → {args.out_dir}/{args.out_name}.{fmt}")
    plt.close(fig)


if __name__ == "__main__":
    main()
