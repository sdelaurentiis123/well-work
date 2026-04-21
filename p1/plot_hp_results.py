"""HP sweep viz — shows tuning moved baseline a lot but ft barely moved."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def run(args):
    hp = json.loads(Path(args.hp_json).read_text())

    # Raw numbers from original run_baselines.sh / run_all.sh results
    raw_baseline_seeds = [0.55295, 0.53597, 0.58184]   # baseline_01 at lr=1e-3, hidden=48
    raw_ft_seeds = [0.30340, 0.30332, 0.30335]          # ft_01 at lr=1e-3, hidden=48

    tuned_b = hp["baseline_01_best"]; tuned_f = hp["ft_01_best"]

    # 4 bars
    names = ["baseline_01\nraw (lr=1e-3)", "baseline_01\ntuned (lr=3e-3,\nhidden=64)",
             "ft_01\nraw (lr=1e-3)",        "ft_01\ntuned (lr=3e-4)"]
    means = [np.mean(raw_baseline_seeds), tuned_b["mean"],
             np.mean(raw_ft_seeds),        tuned_f["mean"]]
    stds  = [np.std(raw_baseline_seeds),  tuned_b["std"],
             np.std(raw_ft_seeds),         tuned_f["std"]]
    colors = ["#d62728", "#b03030",  "#1f77b4", "#155388"]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    xs = np.arange(len(names))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors,
                  edgecolor="black", linewidth=0.5, alpha=0.88)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + s + 0.01, f"{m:.3f}", ha="center", fontsize=10)

    # Gap annotations
    raw_gap = (np.mean(raw_baseline_seeds) - np.mean(raw_ft_seeds)) / np.mean(raw_baseline_seeds) * 100
    tun_gap = (tuned_b["mean"] - tuned_f["mean"]) / tuned_b["mean"] * 100
    ax.annotate(f"raw gap: {raw_gap:.0f}%",
                xy=(0.5, max(means)*0.95), xytext=(0.5, max(means)*1.02),
                ha="center", fontsize=10, color="#d62728",
                arrowprops=None)
    ax.annotate(f"tuned gap: {tun_gap:.0f}%",
                xy=(2.5, max(means)*0.82), xytext=(2.5, max(means)*0.88),
                ha="center", fontsize=10, color="#155388",
                arrowprops=None)

    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("best validation VRMSE  (mean ± std, 3 seeds)")
    ax.set_title("P1 HP-sweep head-to-head — baseline undertuning closed some gap, pretraining still wins")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"wrote {args.out}")
    print(f"  raw   gap {raw_gap:.1f}%    ({np.mean(raw_baseline_seeds):.3f} → {np.mean(raw_ft_seeds):.3f})")
    print(f"  tuned gap {tun_gap:.1f}%    ({tuned_b['mean']:.3f} → {tuned_f['mean']:.3f})")
    print(f"  tuning alone closed {(np.mean(raw_baseline_seeds)-tuned_b['mean'])/np.mean(raw_baseline_seeds)*100:.1f}% of baseline's raw loss")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--hp_json", default="p1/hp_summary.json")
    p.add_argument("--out", default="p1/figures/p1_hp_head_to_head.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
