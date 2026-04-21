"""Task 2(b) + 2(c): aniso comparison + cascade slopes.

Reads evals/physics/<model>/aniso_step1.npz for each of the 4 models + ground truth.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS = ["baseline", "baseline_01", "ft_01", "ft_100"]


def run(args):
    root = Path(args.physics_dir)
    data = {}
    for m in MODELS:
        fp = root / m / "aniso_step1.npz"
        if not fp.exists():
            print(f"  missing {fp}"); continue
        d = np.load(fp)
        data[m] = {"pred": d["pred"], "truth": d["truth"], "edges": d["edges"]}

    if not data:
        print("no aniso data found"); return
    # Ground truth is the same across models (same test set); average.
    truths = np.stack([data[m]["truth"] for m in data])
    truth_mean = truths.mean(axis=0)
    edges = data[next(iter(data))]["edges"]

    # ----- (b) 2x2 heatmap comparison -----
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [("ground truth", truth_mean),
              ("baseline (100% scratch)", data.get("baseline", {}).get("pred", truth_mean)),
              ("baseline_01 (1% scratch)", data.get("baseline_01", {}).get("pred", truth_mean)),
              ("ft_01 (pretrain + 1% FT)", data.get("ft_01", {}).get("pred", truth_mean))]
    # log scale
    vmax = np.log10(truth_mean.max() + 1e-20)
    vmin = vmax - 6
    for ax, (title, H) in zip(axes.flat, panels):
        im = ax.imshow(np.log10(H + 1e-20).T,
                       origin="lower", aspect="auto",
                       extent=[edges[0], edges[-1], edges[0], edges[-1]],
                       vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel(r"k$_\parallel$"); ax.set_ylabel(r"k$_\perp$")
    plt.colorbar(im, ax=list(axes.flat), label=r"log$_{10}$ E(k$_\parallel$, k$_\perp$)")
    fig.suptitle("P1 anisotropic spectrum E(k∥, k⊥) — B_x channel, step-1 prediction on M_A=0.7 test",
                 fontsize=12)
    fig.savefig(args.aniso_out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {args.aniso_out}")

    # ----- (c) 1D cascade slopes -----
    # k_perp cascade: average H over small k_∥ bins (low k_∥), plot vs k_⊥
    # k_par cascade: average H over small k_⊥ bins (low k_⊥), plot vs k_∥
    centers = 0.5 * (edges[:-1] + edges[1:])
    i_lowpar = slice(0, max(2, len(centers) // 8))     # lowest couple k_∥ bins
    i_lowperp = slice(0, max(2, len(centers) // 8))    # lowest couple k_⊥ bins

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"baseline": "#888888", "baseline_01": "#d62728",
              "ft_01": "#1f77b4", "ft_100": "#2ca02c"}

    def plot_slice(ax, arr, label, color, ls="-"):
        # k_perp cascade: integrate over low k_∥, plot vs k_⊥
        e_perp = arr[i_lowpar, :].mean(axis=0)
        # positive values only for loglog
        mask = e_perp > 0
        ax.loglog(centers[mask], e_perp[mask], ls, lw=2, color=color, label=label)

    for m in MODELS:
        if m in data:
            plot_slice(ax1, data[m]["pred"], m, colors[m])
    plot_slice(ax1, truth_mean, "truth", "black", ls=":")
    # Reference slope k_⊥^(-5/3)
    k_ref = centers[(centers >= 2) & (centers <= 10)]
    if len(k_ref):
        ref = 1e-4 * k_ref ** (-5/3)
        ax1.loglog(k_ref, ref, "k--", lw=1, alpha=0.5, label=r"$k_\perp^{-5/3}$ (GS95)")
    ax1.set_xlabel(r"k$_\perp$"); ax1.set_ylabel(r"E(k$_\perp$ | low k$_\parallel$)")
    ax1.set_title("Perpendicular cascade")
    ax1.grid(True, which="both", alpha=0.3); ax1.legend(fontsize=8)

    def plot_slice2(ax, arr, label, color, ls="-"):
        e_par = arr[:, i_lowperp].mean(axis=1)
        mask = e_par > 0
        ax.loglog(centers[mask], e_par[mask], ls, lw=2, color=color, label=label)

    for m in MODELS:
        if m in data:
            plot_slice2(ax2, data[m]["pred"], m, colors[m])
    plot_slice2(ax2, truth_mean, "truth", "black", ls=":")
    # Reference slope k_∥^(-2)
    if len(k_ref):
        ref = 1e-4 * k_ref ** (-2)
        ax2.loglog(k_ref, ref, "k--", lw=1, alpha=0.5, label=r"$k_\parallel^{-2}$ (GS95)")
    ax2.set_xlabel(r"k$_\parallel$"); ax2.set_ylabel(r"E(k$_\parallel$ | low k$_\perp$)")
    ax2.set_title("Parallel cascade")
    ax2.grid(True, which="both", alpha=0.3); ax2.legend(fontsize=8)

    fig.suptitle("P1 cascade slopes — B_x channel, step-1 prediction")
    fig.tight_layout()
    fig.savefig(args.cascade_out, dpi=140); plt.close(fig)
    print(f"wrote {args.cascade_out}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--physics_dir", default="p1/evals/physics")
    p.add_argument("--aniso_out", default="p1/figures/p1_aniso_spectrum_comparison.png")
    p.add_argument("--cascade_out", default="p1/figures/p1_cascade_slopes.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
