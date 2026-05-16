"""Task 2: cascade evolution at multiple rollout steps.

Reads field_snapshots.npz (which has trajectory-averaged 3D fields at steps
{1, 5, 10, 25, 50}), computes E(k_perp) at low k_par for B_y and B_z
separately, overlays vs truth at each step.

Per reviewer request, this script emits the cascade evolution as two
separate figures (one per perpendicular component) rather than a single 2-row
grid. Both go to the same out_dir:

    fig_cascade_evolution_By.pdf
    fig_cascade_evolution_Bz.pdf

A combined figure (fig_cascade_evolution.pdf) is also written for backward
compatibility but is no longer referenced in the paper.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .extract_physics import anisotropic_spectrum, b0_direction


def low_kpar_slice(H: np.ndarray, edges: np.ndarray, npar_slice: int = 2) -> np.ndarray:
    """H shape (n_par, n_perp). Sum over first npar_slice k_par bins to give E(k_perp)."""
    return H[:npar_slice, :].sum(axis=0)


def _plot_one_component(snaps, steps, b0, ch_name, ch_idx, out_path):
    """Render a single perpendicular-component cascade evolution figure."""
    fig, axes = plt.subplots(1, len(steps), figsize=(4 * len(steps), 3.6), sharey=True)
    if len(steps) == 1:
        axes = [axes]
    for col, s in enumerate(steps):
        ax = axes[col]
        for cfg_name, color, ls in [
            ("walrus", "#d62728", "-"),
            ("fno_ft", "#1f77b4", "-"),
            ("fno_pretrain_ood", "#ff7f0e", ":"),
            ("fno_baseline", "#888888", "--"),
        ]:
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
        if col == 0:
            ax.set_ylabel(r"$E(k_\perp)$ at low $k_\parallel$")
    axes[0].legend(fontsize=7, loc="lower left")
    fig.suptitle(f"Perpendicular cascade evolution: {ch_name}")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _plot_combined(snaps, steps, b0, out_path):
    """Retained for backward compatibility - the paper no longer references it."""
    fig, axes = plt.subplots(2, len(steps), figsize=(4 * len(steps), 7), sharey="row")
    for col, s in enumerate(steps):
        for row, ch_name, ch_idx in [(0, "B_y", 2), (1, "B_z", 3)]:
            ax = axes[row, col]
            for cfg_name, color, ls in [
                ("walrus", "#d62728", "-"),
                ("fno_ft", "#1f77b4", "-"),
                ("fno_pretrain_ood", "#ff7f0e", ":"),
                ("fno_baseline", "#888888", "--"),
            ]:
                pred = snaps[cfg_name][f"pred_step{s}"][ch_idx]
                edges, H = anisotropic_spectrum(pred, b0)
                centers = 0.5 * (edges[:-1] + edges[1:])
                E_perp = low_kpar_slice(H, edges)
                ax.loglog(centers, E_perp + 1e-30, color=color, ls=ls, lw=1.4,
                          label=cfg_name if (row == 0 and col == 0) else None)
            truth = snaps["walrus"][f"truth_step{s}"][ch_idx]
            edges, Ht = anisotropic_spectrum(truth, b0)
            centers = 0.5 * (edges[:-1] + edges[1:])
            E_truth = low_kpar_slice(Ht, edges)
            ax.loglog(centers, E_truth + 1e-30, color="black", lw=1.7, alpha=0.8,
                      label="truth" if (row == 0 and col == 0) else None)
            ax.set_title(f"step {s}, {ch_name}")
            ax.grid(alpha=0.3, which="both")
            if row == 1:
                ax.set_xlabel(r"$k_\perp$")
            if col == 0:
                ax.set_ylabel(r"$E(k_\perp)$ at low $k_\parallel$")
    axes[0, 0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Perpendicular cascade evolution: Walrus per-component pile-up")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True,
                   help="Path to the combined figure; per-component figures are "
                        "derived from it by appending _By / _Bz to the stem.")
    args = p.parse_args()

    snaps = {
        c: np.load(args.results_root / c / "field_snapshots.npz")
        for c in ["fno_baseline", "fno_ft", "fno_pretrain_ood", "walrus"]
    }
    steps = list(snaps["walrus"]["steps"])
    print(f"steps: {steps}")

    truth_b1 = snaps["walrus"]["truth_step1"][1:4]
    b0 = b0_direction(truth_b1)
    print(f"b0 (from truth step 1): {b0}")

    # Per-component figures (the ones the paper now cites).
    stem = args.out.with_suffix("")
    _plot_one_component(snaps, steps, b0, "B_y", 2,
                        stem.with_name(stem.name + "_By").with_suffix(".pdf"))
    _plot_one_component(snaps, steps, b0, "B_z", 3,
                        stem.with_name(stem.name + "_Bz").with_suffix(".pdf"))

    # Combined figure retained for compatibility / appendix.
    _plot_combined(snaps, steps, b0, args.out)

    # Quantitative: where does the energy concentrate?
    print("\n=== quantitative: walrus B_y / B_z energy by k-band ===")
    print(f"  {'step':<6}  {'ch':<4}  {'low-k E':>12}  {'mid-k E':>12}  {'high-k E':>12}  {'total':>12}")
    for s in steps:
        for ch_name, ch_idx in [("B_y", 2), ("B_z", 3)]:
            pred = snaps["walrus"][f"pred_step{s}"][ch_idx]
            edges, H = anisotropic_spectrum(pred, b0)
            E_perp = low_kpar_slice(H, edges)
            n = len(E_perp)
            low = E_perp[: n // 4].sum()
            mid = E_perp[n // 4 : 3 * n // 4].sum()
            hi = E_perp[3 * n // 4 :].sum()
            tot = E_perp.sum()
            print(f"  {s:<6}  {ch_name:<4}  {low:>12.3e}  {mid:>12.3e}  {hi:>12.3e}  {tot:>12.3e}")


if __name__ == "__main__":
    main()
