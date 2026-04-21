"""Physics probe 5: scaling invariance violation per model."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS_FOUR = ["baseline", "baseline_01", "ft_01", "ft_100"]
COLORS = {"baseline": "#888888", "baseline_01": "#d62728",
          "ft_01": "#1f77b4", "ft_100": "#2ca02c"}
CHANNELS = ["density","B_x","B_y","B_z","v_x","v_y","v_z"]


def run(args):
    root = Path(args.physics_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for i, dev_key in enumerate(["dev_A", "dev_B"]):
        ax = axes[i]
        width = 0.2
        xpos = np.arange(len(CHANNELS))
        for j, m in enumerate(MODELS_FOUR):
            fp = root / m / "scaling.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            vals = d[dev_key]
            ax.bar(xpos + (j - 1.5) * width, vals, width,
                   label=m, color=COLORS[m], edgecolor="black", linewidth=0.3)
        ax.set_xticks(xpos); ax.set_xticklabels(CHANNELS, rotation=30)
        ax.set_ylabel("relative deviation after rescaling+inverse")
        ax.set_title(("(a) β=2 scaling (B, v → 2x)" if i == 0 else
                      "(b) α=4 scaling (ρ → 4x)"))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        ax.axhline(0.01, ls=":", color="gray", alpha=0.5)  # ideal-invariance threshold
    fig.suptitle("P1 scaling invariance — ideal MHD predicts zero deviation (dashed = 0.01 noise line)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f"wrote {args.out}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--physics_dir", default="p1/evals/physics")
    p.add_argument("--out", default="p1/figures/physics/scaling_invariance.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
