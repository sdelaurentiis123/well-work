"""Task 2(a): cross-model rollout VRMSE vs step, from existing results.json files.

No new compute — reads `rollout_vrmse_mean_per_step` / `rollout_vrmse_std_per_step`
that `eval_full.py` already wrote for each checkpoint.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

MODELS = [
    ("baseline",     "baseline (M_A=0.7, 100% data, scratch)", "#888888", "-"),
    ("ft_100",       "ft_100 (pretrain → 100% M_A=0.7)",       "#1f77b4", "-"),
    ("baseline_10",  "baseline_10 (scratch, 10%)",             "#d62728", "--"),
    ("ft_10",        "ft_10 (pretrain → 10%)",                 "#1f77b4", "--"),
    ("baseline_01",  "baseline_01 (scratch, 1%)",              "#d62728", ":"),
    ("ft_01",        "ft_01 (pretrain → 1%)",                  "#1f77b4", ":"),
    ("pretrain",     "pretrain (zero-shot, no M_A=0.7 fine-tune)", "#2ca02c", "-."),
]

MARKED_STEPS = [1, 5, 10, 25, 50]


def run(args):
    root = Path(args.evals_dir)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for name, label, color, ls in MODELS:
        rj = root / name / "results.json"
        if not rj.exists():
            print(f"  missing {rj}"); continue
        r = json.loads(rj.read_text())
        mu = np.array(r.get("rollout_vrmse_mean_per_step", []))
        sd = np.array(r.get("rollout_vrmse_std_per_step", []))
        if len(mu) == 0:
            print(f"  {name}: no rollout data"); continue
        steps = 1 + np.arange(len(mu))
        ax.plot(steps, mu, ls, lw=1.8, color=color, label=label)
        ax.fill_between(steps, mu - sd, mu + sd, color=color, alpha=0.1)

    ax.set_xlabel("autoregressive rollout step")
    ax.set_ylabel("VRMSE vs ground truth  (mean ± std over test trajectories)")
    ax.set_title("P1 multi-step rollout — all models, M_A=0.7 test trajectories\n"
                 "pretrained models outperform at short horizons; reverse at long horizons")
    ax.grid(True, alpha=0.3)
    # mark standard eval steps
    for s in MARKED_STEPS:
        ax.axvline(s, ls=":", color="#cccccc", lw=0.6, zorder=0)
    ax.set_xlim(1, None)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"\nwrote {args.out}")

    # Also print the numeric table at selected steps
    print("\n=== Rollout VRMSE at marked steps (mean ± std) ===")
    header = f"{'config':<20s}  " + "  ".join(f"step{s:>3d}" for s in MARKED_STEPS)
    print(header); print("-" * len(header))
    for name, _, _, _ in MODELS:
        rj = root / name / "results.json"
        if not rj.exists(): continue
        r = json.loads(rj.read_text())
        mu = r.get("rollout_vrmse_mean_per_step", [])
        sd = r.get("rollout_vrmse_std_per_step", [])
        row = f"{name:<20s}  "
        for s in MARKED_STEPS:
            if s - 1 < len(mu):
                row += f"{mu[s-1]:.3f}±{sd[s-1]:.3f}  "
            else:
                row += "---  "
        print(row)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--evals_dir", default="p1/evals2")
    p.add_argument("--out", default="p1/figures/p1_rollout_comparison.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
