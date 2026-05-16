"""Overlay-figure generation for the Walrus diagnostic note.

Reads .npz outputs from results/shifted_window/{fno_baseline, fno_ft,
fno_pretrain_ood, walrus}/ and produces 6 overlay figures comparing all four
configs on the same shifted window (frames [3..52]):

    fig_vrmse_overlay.pdf      - VRMSE vs rollout step
    fig_cascade.pdf            - perpendicular cascade at step 1
    fig_bx_norm.pdf            - guide-field magnitude trajectory
    fig_divb.pdf               - div B floor accumulation
    fig_equipartition.pdf      - E_B / E_K trajectory
    fig_per_traj.pdf           - per-trajectory VRMSE, all four models overlaid
                                 in a single axes (10 trajectories each)

Confound disclosure (printed in figure captions and findings.md):
  - scale: Walrus 1.3B vs FNO 18M (~70x params)
  - contamination: Walrus saw MHD_64 train; FNO trained on M_A=2.0 only
  - conditioning: Walrus T_in=3 vs FNO T_in=1 (small but nonzero)

Usage:
    python -m src.overlay_figures \
        --results_root results/shifted_window \
        --out_dir figures/
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CONFIGS = ["fno_baseline", "fno_ft", "fno_pretrain_ood", "walrus"]
LABELS = {
    "fno_baseline":     "FNO scratch (1% data)",
    "fno_ft":           "FNO MHD-pretrain + FT",
    "fno_pretrain_ood": "FNO MHD-pretrain zero-FT",
    "walrus":           "Walrus 1.3B zero-shot",
}
COLORS = {
    "fno_baseline":     "#888888",
    "fno_ft":           "#1f77b4",
    "fno_pretrain_ood": "#ff7f0e",
    "walrus":           "#d62728",
}
LINESTYLES = {
    "fno_baseline":     "--",
    "fno_ft":           "-",
    "fno_pretrain_ood": ":",
    "walrus":           "-",
}


def load_npz(root: Path, cfg: str, name: str):
    p = root / cfg / name
    if not p.exists():
        return None
    return np.load(p)


# ---------------- figure builders ----------------------------------------
def fig_vrmse(root: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for cfg in CONFIGS:
        d = load_npz(root, cfg, "rollout_vrmse_full.npz")
        if d is None: continue
        v = d["vrmse"]
        m = v.mean(0); s = v.std(0)
        x = np.arange(1, len(m) + 1)
        ax.plot(x, m, color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.7, label=LABELS[cfg])
        ax.fill_between(x, m - s, m + s, color=COLORS[cfg], alpha=0.12)
    ax.set_xlabel("rollout step")
    ax.set_ylabel("VRMSE")
    ax.set_yscale("log")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax.set_title("VRMSE vs rollout step (10 trajectories, shared window)")
    return fig


def fig_cascade(root: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    # E(k_perp) integrated over low k_par.
    for cfg in CONFIGS:
        d = load_npz(root, cfg, "aniso_step1.npz")
        if d is None: continue
        edges = d["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        Hp = d["pred"]
        # low-k_par slice
        lowpar = slice(0, max(1, len(edges) // 8))
        ek_pred = Hp[lowpar, :].sum(0)
        ax.loglog(centers, ek_pred + 1e-30, color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.7, label=LABELS[cfg])
    # Truth curve (use whichever config has it; they should all match)
    for cfg in CONFIGS:
        d = load_npz(root, cfg, "aniso_step1.npz")
        if d is None: continue
        edges = d["edges"]
        centers = 0.5 * (edges[:-1] + edges[1:])
        Ht = d["truth"]
        lowpar = slice(0, max(1, len(edges) // 8))
        ek_truth = Ht[lowpar, :].sum(0)
        ax.loglog(centers, ek_truth + 1e-30, color="black", lw=2.0, alpha=0.6, label="ground truth")
        break
    # GS95 + Boldyrev reference slopes
    k_ref = np.array([3.0, 16.0])
    ax.loglog(k_ref, k_ref ** (-5/3) * 1e-2, "k--", alpha=0.4, lw=0.9, label=r"$k_\perp^{-5/3}$ (GS95)")
    ax.loglog(k_ref, k_ref ** (-3/2) * 1e-2, "k:",  alpha=0.4, lw=0.9, label=r"$k_\perp^{-3/2}$ (Boldyrev)")
    ax.set_xlabel(r"$k_\perp$")
    ax.set_ylabel(r"$E_{B_x}(k_\perp)$ at low $k_\parallel$")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Perpendicular cascade at step 1 (B_x channel)")
    return fig


def fig_bx_norm(root: Path):
    """Mean |B_x| over rollout - the guide-field-collapse diagnostic."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    truth_drawn = False
    for cfg in CONFIGS:
        d = load_npz(root, cfg, "variance.npz")
        if d is None: continue
        c = load_npz(root, cfg, "conservation.npz")
        if c is None: continue
        E_B = c["pred_E_B"].mean(0)  # (K+1,)
        proxy = np.sqrt(2.0 * np.maximum(E_B, 0))
        ax.plot(np.arange(len(proxy)), proxy, color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.7, label=LABELS[cfg])
        if not truth_drawn:
            E_B_t = c["truth_E_B"].mean(0)
            proxy_t = np.sqrt(2.0 * np.maximum(E_B_t, 0))
            ax.plot(np.arange(len(proxy_t)), proxy_t, color="black", lw=2.0, alpha=0.6, label="ground truth")
            truth_drawn = True
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$\langle |B| \rangle \approx \sqrt{2 E_B}$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Magnetic-field magnitude trajectory")
    return fig


def fig_divb(root: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    truth_drawn = False
    for cfg in CONFIGS:
        c = load_npz(root, cfg, "conservation.npz")
        if c is None: continue
        d = c["pred_divB_norm"].mean(0)
        ax.plot(np.arange(len(d)), d, color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.7, label=LABELS[cfg])
        if not truth_drawn:
            dt = c["truth_divB_norm"].mean(0)
            ax.plot(np.arange(len(dt)), dt, color="black", lw=2.0, alpha=0.6, label="ground truth")
            truth_drawn = True
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$|\nabla \cdot B| / |B|$")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    ax.set_title("Solenoidal-constraint floor")
    return fig


def fig_equipart(root: Path):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    truth_drawn = False
    for cfg in CONFIGS:
        c = load_npz(root, cfg, "conservation.npz")
        if c is None: continue
        r = c["pred_E_ratio"].mean(0)
        ax.plot(np.arange(len(r)), r, color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.7, label=LABELS[cfg])
        if not truth_drawn:
            rt = c["truth_E_ratio"].mean(0)
            ax.plot(np.arange(len(rt)), rt, color="black", lw=2.0, alpha=0.6, label="ground truth")
            truth_drawn = True
    ax.axhline(0.25, color="gray", lw=0.8, alpha=0.5, ls=":")
    ax.text(1, 0.27, r"$1/M_A^2 = 0.25$ (M_A=2 source)", fontsize=8, color="gray")
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$E_B / E_K$")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Equipartition trajectory")
    return fig


def fig_per_traj(root: Path):
    """Per-trajectory rollout VRMSE, all four configurations overlaid on a
    single axes. Each configuration contributes up to 10 thin trajectory
    traces plus a thicker mean line; the legend identifies configurations
    by colour."""
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for cfg in CONFIGS:
        d = load_npz(root, cfg, "rollout_vrmse_full.npz")
        if d is None:
            continue
        v = d["vrmse"]
        x = np.arange(1, v.shape[1] + 1)
        # Individual trajectories: thin, semi-transparent.
        for i in range(v.shape[0]):
            ax.plot(x, v[i], color=COLORS[cfg], lw=0.5, alpha=0.35)
        # Mean line carries the legend entry.
        ax.plot(x, v.mean(0), color=COLORS[cfg], ls=LINESTYLES[cfg], lw=1.8,
                label=LABELS[cfg])
    ax.set_yscale("log")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("VRMSE")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title("Per-trajectory rollout VRMSE (all four models overlaid)")
    fig.tight_layout()
    return fig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    builders = {
        "fig_vrmse_overlay.pdf":  fig_vrmse,
        "fig_cascade.pdf":        fig_cascade,
        "fig_bx_norm.pdf":        fig_bx_norm,
        "fig_divb.pdf":           fig_divb,
        "fig_equipartition.pdf":  fig_equipart,
        "fig_per_traj.pdf":       fig_per_traj,
    }
    for name, builder in builders.items():
        try:
            f = builder(args.results_root)
            f.savefig(args.out_dir / name, bbox_inches="tight")
            f.savefig(args.out_dir / name.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
            plt.close(f)
            print(f"wrote {args.out_dir / name}")
        except Exception as e:
            print(f"FAIL {name}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
