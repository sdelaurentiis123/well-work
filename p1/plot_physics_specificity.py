"""Task 4 headline: physics-specific transfer — 3-way bar chart.

MHD pretrain → FT: transfers physics, wins 28% vs scratch.
NS pretrain → FT: wrong physics, LOSES 23% vs scratch.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def best_val(log_path: Path) -> float:
    if not log_path.exists(): return math.nan
    b = math.inf
    for line in log_path.read_text().splitlines():
        if not line.strip(): continue
        v = json.loads(line).get("val_vrmse", math.inf)
        if v < b: b = v
    return b


def seeds_for(prefix, runs):
    import re
    rx = re.compile(rf"^{re.escape(prefix)}(_s\d+)?$")
    out = []
    for d in runs.iterdir():
        if d.is_dir() and rx.match(d.name):
            v = best_val(d / "log.jsonl")
            if math.isfinite(v): out.append(v)
    return out


def run(args):
    runs = Path(args.runs_dir)
    hp = json.loads(Path(args.hp_json).read_text())

    # Three configurations to compare at 1% data:
    #   baseline_01 tuned  — from hp_summary
    #   MHD→ft_01 tuned    — from hp_summary
    #   NS→ft_01           — from runs/ns_ft_01_s{0,1,2}
    ns_vals = seeds_for("ns_ft_01", runs)

    bars = [
        ("from scratch\n(tuned baseline)", hp["baseline_01_best"]["mean"],
         hp["baseline_01_best"]["std"], "#d62728"),
        ("NS pretrain → FT\n(supernova_explosion_64)", np.mean(ns_vals),
         np.std(ns_vals), "#ff8c00"),
        ("MHD pretrain → FT\n(M_A=2.0 → M_A=0.7, tuned)", hp["ft_01_best"]["mean"],
         hp["ft_01_best"]["std"], "#1f77b4"),
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    xs = np.arange(len(bars))
    heights = [b[1] for b in bars]
    errs = [b[2] for b in bars]
    colors = [b[3] for b in bars]
    names = [b[0] for b in bars]

    ax.bar(xs, heights, yerr=errs, capsize=6, color=colors,
           edgecolor="black", linewidth=0.6, alpha=0.9)
    for i, (h, e) in enumerate(zip(heights, errs)):
        ax.text(i, h + e + 0.012, f"{h:.3f} ± {e:.3f}", ha="center", fontsize=10)

    # Delta annotations
    b_scratch = hp["baseline_01_best"]["mean"]
    ns_delta = (np.mean(ns_vals) - b_scratch) / b_scratch * 100
    mhd_delta = (hp["ft_01_best"]["mean"] - b_scratch) / b_scratch * 100
    ax.annotate(f"{ns_delta:+.0f}% vs scratch",
                xy=(1, np.mean(ns_vals) + np.std(ns_vals) + 0.03),
                ha="center", fontsize=11, color="#ff8c00", fontweight="bold")
    ax.annotate(f"{mhd_delta:+.0f}% vs scratch",
                xy=(2, hp["ft_01_best"]["mean"] + hp["ft_01_best"]["std"] + 0.03),
                ha="center", fontsize=11, color="#1f77b4", fontweight="bold")

    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("best validation VRMSE on M_A=0.7 (mean ± std, 3 seeds)")
    ax.set_title("Physics-specific transfer — MHD pretrain helps, NS pretrain HURTS\n"
                 "All runs: 1% target data, FNO3D 18.6M params, epochs=40, 3 seeds")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(heights) + max(errs) + 0.08)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150); plt.close(fig)
    print(f"wrote {args.out}")
    print(f"\nNS→ft_01 per-seed: {ns_vals}")
    print(f"NS→ft_01 mean ± std: {np.mean(ns_vals):.4f} ± {np.std(ns_vals):.4f}")
    print(f"NS vs scratch:    {ns_delta:+.1f}% ({'WORSE' if ns_delta>0 else 'better'})")
    print(f"MHD vs scratch:   {mhd_delta:+.1f}%")
    print(f"MHD vs NS:        {(hp['ft_01_best']['mean'] - np.mean(ns_vals))/np.mean(ns_vals)*100:+.1f}%")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="p1/runs")
    p.add_argument("--hp_json",  default="p1/hp_summary.json")
    p.add_argument("--out", default="p1/figures/p1_physics_specificity.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
