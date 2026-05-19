"""Task 2: cascade evolution at multiple rollout steps.

Reads field_snapshots.npz (which has trajectory-averaged 3D fields at steps
{1, 5, 10, 25, 50}), computes E(k_perp) at low k_par for B_y and B_z separately,
overlays vs truth at each step. Identifies whether energy piles up at high k
(numerical small-scale instability), low k (wrong cascade direction), or
inflates uniformly.

Outputs:
  - <out>                              combined 2-row figure (B_y and B_z)
  - <out>.with_suffix('.png')
  - <out_stem>_By.pdf / _By.png        standalone single-row figure for B_y
  - <out_stem>_Bz.pdf / _Bz.png        standalone single-row figure for B_z
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .extract_physics import anisotropic_spectrum, b0_direction


def low_kpar_slice(H: np.ndarray, edges: np.ndarray, npar_slice: int = 2) -> np.ndarray:
    """H shape (n_par, n_perp). Sum over first npar_slice k_par bins -> E(k_perp)."""
    return H[:npar_slice, :].sum(axis=0)


CONFIG_STYLES = [
    ("walrus", "#d62728", "-"),
    ("fno_ft", "#1f77b4", "-"),
    ("fno_pretrain_ood", "#ff7f0e", ":"),
    ("fno_baseline", "#888888", "--"),
]


def _draw_component_row(axes_row, snaps, steps, ch_name, ch_idx, b0):
    """Draw a single magnetic component's cascade evolution across rollout steps
    onto the provided 1D array of axes (length == len(steps))."""
    for col, s in enumerate(steps):
        ax = axes_row[col]
        for cfg_name, color, ls in CONFIG_STYLES:
            pred = snaps[cfg_name][f"pred_step{s}"][ch_idx]
            edges, H = anisotropic_spectrum(pred, b0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            E_perp = low_kpar_slice(H, edges)
            ax.loglog(centers, E_perp + 1e-30, color=color, ls=ls, lw=1.4,
                      label=cfg_name if col == 0 else None)
        truth = snaps["walrus"][f"truth_step{s}"][ch_idx]
        edges, Ht = anisotropic_spectrum(truth, b0)
        centers = 0.5 * (edges[:-1] + edges[1:])
        E_truth = low_kpar_slice(Ht, edges)
        ax.loglog(centers, E_truth + 1e-30, color="black", lw=1.7, alpha=0.8,
                  label="truth" if col == 0 else None)
        ax.set_title(f"step {s}, {ch_name}")
        ax.grid(alpha=0.3, which="both")
        ax.set_xlabel(r"$k_\perp$")
    axes_row[0].set_ylabel(r"$E(k_\perp)$ at low $k_\parallel$")
    axes_row[0].legend(fontsize=7, loc="lower left")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    snaps = {
        c: np.load(args.results_root / c / "field_snapshots.npz")
        for c in ["fno_baseline", "fno_ft", "fno_pretrain_ood", "walrus"]
    }
    steps = list(snaps["walrus"]["steps"])
    print(f"steps: {steps}")

    # Use truth from any config (identical) to define b0 axis
    truth_b1 = snaps["walrus"]["truth_step1"][1:4]  # (3, 64, 64, 64) - B at step 1
    b0 = b0_direction(truth_b1)
    print(f"b0 (from truth step 1): {b0}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    # ---- Combined 2-row figure (retained for completeness) ----
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 7),
                             sharey="row")
    _draw_component_row(axes[0], snaps, steps, "B_y", 2, b0)
    _draw_component_row(axes[1], snaps, steps, "B_z", 3, b0)
    fig.suptitle("Perpendicular cascade evolution: Walrus per-component pile-up")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    # ---- Standalone per-component figures ----
    for ch_name, ch_idx, suffix in [("B_y", 2, "By"), ("B_z", 3, "Bz")]:
        fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 3.6),
                                 sharey=True)
        if len(steps) == 1:
            axes = np.array([axes])
        _draw_component_row(axes, snaps, steps, ch_name, ch_idx, b0)
        fig.suptitle(f"Perpendicular cascade evolution: {ch_name} component")
        fig.tight_layout()
        out_single = args.out.with_name(args.out.stem + f"_{suffix}").with_suffix(".pdf")
        fig.savefig(out_single, bbox_inches="tight")
        fig.savefig(out_single.with_suffix(".png"), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {out_single}")

    # Quantitative: where does the energy concentrate?
    print("\n=== quantitative: walrus B_y / B_z energy by k-band ===")
    print(f"  {'step':<6}  {'ch':<4}  {'low-k E':>12}  {'mid-k E':>12}  {'high-k E':>12}  {'total':>12}")
    for s in steps:
        for ch_name, ch_idx in [("B_y", 2), ("B_z", 3)]:
            pred = snaps["walrus"][f"pred_step{s}"][ch_idx]
            edges, H = anisotropic_spectrum(pred, b0)
            E_perp = low_kpar_slice(H, edges)  # (n_perp,)
            n = len(E_perp)
            low = E_perp[: n // 4].sum()
            mid = E_perp[n // 4 : 3 * n // 4].sum()
            hi = E_perp[3 * n // 4 :].sum()
            tot = E_perp.sum()
            print(f"  {s:<6}  {ch_name:<4}  {low:>12.3e}  {mid:>12.3e}  {hi:>12.3e}  {tot:>12.3e}")


if __name__ == "__main__":
    main()
