"""All physics interpretability plots that derive from physics/<model>/ .npz files.

Produces:
  figures/physics/conservation_drift.png     (Set B Task 1: mass, E_B, E_K, ratio)
  figures/physics/divergence_violation.png   (Set B Task 1: ∇·B)
  figures/physics/equipartition_evolution.png (Set B Task 2)
  figures/physics/variance_evolution.png     (Task 5/Set B Task 5c)
  figures/physics/failure_modes.png          (Task 5 main synthesis)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS_FOUR = ["baseline", "baseline_01", "ft_01", "ft_100"]
COLORS = {"baseline": "#888888", "baseline_01": "#d62728",
          "ft_01": "#1f77b4", "ft_100": "#2ca02c"}
LS_TRUTH = dict(color="black", ls=":", lw=1.5, label="truth")


def load_cons(root):
    """Return dict: model -> {metric -> (n_traj, K+1)} for pred and 'truth' shared."""
    out = {"pred": {}, "truth": None}
    for m in MODELS_FOUR:
        fp = root / m / "conservation.npz"
        if not fp.exists():
            continue
        d = np.load(fp)
        out["pred"][m] = {k: d[f"pred_{k}"] for k in ("mass","E_B","E_K","E_ratio","divB_norm")}
        # Truth is same per trajectory across models — take last-loaded
        out["truth"] = {k: d[f"truth_{k}"] for k in ("mass","E_B","E_K","E_ratio","divB_norm")}
    return out


def plot_conservation_drift(cons, out_path):
    """4-panel: mass, E_B, E_K, E_ratio relative drift vs step."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    titles = [("mass",    axes[0, 0], r"relative drift in $\int\rho\,dV$"),
              ("E_B",     axes[0, 1], r"relative drift in $\int B^2\,dV$"),
              ("E_K",     axes[1, 0], r"relative drift in $\int\frac{1}{2}\rho v^2\,dV$"),
              ("E_ratio", axes[1, 1], r"$E_B/E_K$ (absolute)"),]
    for key, ax, ylab in titles:
        for m, d in cons["pred"].items():
            v = d[key]
            if key != "E_ratio":
                ref = v[:, 0:1]
                drift = (v - ref) / (np.abs(ref) + 1e-12)
            else:
                drift = v
            mu = drift.mean(axis=0); sd = drift.std(axis=0)
            steps = np.arange(mu.shape[0])
            ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
            ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
        if cons["truth"] is not None:
            v = cons["truth"][key]
            if key != "E_ratio":
                ref = v[:, 0:1]
                drift = (v - ref) / (np.abs(ref) + 1e-12)
            else:
                drift = v
            mu = drift.mean(axis=0)
            ax.plot(np.arange(mu.shape[0]), mu, **LS_TRUTH)
        ax.set_xlabel("rollout step"); ax.set_ylabel(ylab)
        ax.set_title(key.replace("_"," "))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("P1 conservation drift — relative to initial, 10 M_A=0.7 test trajectories")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


def plot_divB(cons, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in cons["pred"].items():
        v = d["divB_norm"]
        mu = v.mean(axis=0); sd = v.std(axis=0)
        steps = np.arange(mu.shape[0])
        ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
        ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
    if cons["truth"] is not None:
        v = cons["truth"]["divB_norm"]
        ax.plot(np.arange(v.shape[1]), v.mean(axis=0), **LS_TRUTH)
    ax.set_xlabel("rollout step")
    ax.set_ylabel(r"$\|\nabla \cdot B\|_2 / \langle|B|\rangle$  (monopole production)")
    ax.set_yscale("log")
    ax.set_title("P1 ∇·B violation — magnetic monopole production over rollout")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


def plot_equipartition(cons, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m, d in cons["pred"].items():
        v = d["E_ratio"]
        mu = v.mean(axis=0); sd = v.std(axis=0)
        steps = np.arange(mu.shape[0])
        ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
        ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
    if cons["truth"] is not None:
        v = cons["truth"]["E_ratio"]
        ax.plot(np.arange(v.shape[1]), v.mean(axis=0), **LS_TRUTH)
    # Reference lines for the two source regimes
    # M_A=0.7: E_B/E_K ~ 1/M_A^2 ≈ 2
    # M_A=2.0: E_B/E_K ~ 1/M_A^2 ≈ 0.25
    ax.axhline(1/0.7**2, ls="--", color="#1f77b4", alpha=0.5,
               label=r"M_A=0.7 theory  $1/M_A^2 \approx 2$")
    ax.axhline(1/2.0**2, ls="--", color="#888888", alpha=0.5,
               label=r"M_A=2.0 theory (pretrain regime)  $1/M_A^2 \approx 0.25$")
    ax.set_xlabel("rollout step"); ax.set_ylabel(r"$E_B / E_K$")
    ax.set_title("P1 equipartition — does the model drift toward source or target regime?")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


def plot_variance_evolution(root, out_path):
    """Each model: variance of each field vs rollout step."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    channels = ["density","B_x","B_y","B_z","v_x","v_y","v_z"]
    for ci, ch in enumerate(channels):
        ax = axes.flat[ci]
        for m in MODELS_FOUR:
            fp = root / m / "variance.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            var = d["variance_per_step"][:, :, ci]   # (n_traj, K+1)
            mu = var.mean(axis=0); sd = var.std(axis=0)
            steps = np.arange(mu.shape[0])
            ax.plot(steps, mu, color=COLORS[m], lw=1.6, label=m)
            ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
        ax.set_title(ch); ax.set_xlabel("step"); ax.set_ylabel("variance")
        ax.grid(True, alpha=0.3)
        if ci == 0: ax.legend(fontsize=7)
    axes.flat[-1].axis("off")
    fig.suptitle("P1 variance evolution during rollout — pretrained models may inflate, baselines decay")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


def plot_failure_modes(root, out_path):
    """Composite: (A) rollout VRMSE, (B) E_B drift, (C) variance collapse, (D) one real-space slice."""
    fig = plt.figure(figsize=(14, 10))
    # Panel A: rollout VRMSE
    ax = fig.add_subplot(2, 2, 1)
    for m in MODELS_FOUR:
        fp = root / m / "rollout_vrmse_full.npz"
        if not fp.exists(): continue
        v = np.load(fp)["vrmse"]   # (n_traj, K)
        mu = v.mean(axis=0); sd = v.std(axis=0)
        steps = 1 + np.arange(mu.shape[0])
        ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
        ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
    ax.set_xlabel("step"); ax.set_ylabel("VRMSE vs truth")
    ax.set_title("(a) Rollout accuracy"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # Panel B: E_B drift
    cons = load_cons(root)
    ax = fig.add_subplot(2, 2, 2)
    for m, d in cons["pred"].items():
        v = d["E_B"]; ref = v[:, 0:1]
        drift = (v - ref) / (np.abs(ref) + 1e-12)
        mu = drift.mean(axis=0); sd = drift.std(axis=0)
        steps = np.arange(mu.shape[0])
        ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
        ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
    if cons["truth"]:
        v = cons["truth"]["E_B"]; ref = v[:, 0:1]
        drift = (v - ref) / (np.abs(ref) + 1e-12)
        ax.plot(np.arange(drift.shape[1]), drift.mean(axis=0), **LS_TRUTH)
    ax.set_xlabel("step"); ax.set_ylabel(r"$(E_B - E_{B,0})/E_{B,0}$")
    ax.set_title("(b) Magnetic energy drift"); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # Panel C: density variance evolution
    ax = fig.add_subplot(2, 2, 3)
    for m in MODELS_FOUR:
        fp = root / m / "variance.npz"
        if not fp.exists(): continue
        v = np.load(fp)["variance_per_step"][:, :, 0]   # density
        mu = v.mean(axis=0)
        ax.plot(np.arange(mu.shape[0]), mu / mu[0], color=COLORS[m], lw=1.8, label=m)
    ax.set_xlabel("step"); ax.set_ylabel("density variance / initial")
    ax.set_title("(c) Density-field variance evolution\n(collapse → smooth averaging, inflation → wild predictions)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    # Panel D: real-space snapshot comparison at step 25 for ft_01 and baseline_01
    ax = fig.add_subplot(2, 2, 4)
    picked = None; step_ref = 25
    for m in ("ft_01", "baseline_01"):
        fp = root / m / "field_snapshots.npz"
        if fp.exists():
            d = np.load(fp)
            if f"pred_step{step_ref}" in d:
                picked = d; break
    if picked is not None:
        truth_slice = picked[f"truth_step{step_ref}"][0, :, :, 32]   # density z-midplane
        ft_slice = np.load(root / "ft_01" / "field_snapshots.npz")[f"pred_step{step_ref}"][0, :, :, 32]
        bl_slice = np.load(root / "baseline_01" / "field_snapshots.npz")[f"pred_step{step_ref}"][0, :, :, 32]
        # three-way strip
        combined = np.concatenate([truth_slice, ft_slice, bl_slice], axis=1)
        im = ax.imshow(combined, cmap="viridis")
        ax.set_title(f"(d) density z-midplane at step {step_ref}  —  truth | ft_01 | baseline_01")
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, shrink=0.7)
    else:
        ax.axis("off"); ax.text(0.5, 0.5, "snapshots unavailable", ha="center")

    fig.suptitle("P1 long-horizon failure modes — pretraining wins short, diverges long", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


def run(args):
    root = Path(args.physics_dir)
    outdir = Path(args.figures_dir); outdir.mkdir(parents=True, exist_ok=True)

    cons = load_cons(root)
    if cons["pred"]:
        plot_conservation_drift(cons, outdir / "conservation_drift.png")
        plot_divB(cons, outdir / "divergence_violation.png")
        plot_equipartition(cons, outdir / "equipartition_evolution.png")
    plot_variance_evolution(root, outdir / "variance_evolution.png")
    plot_failure_modes(root, outdir / "failure_modes.png")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--physics_dir", default="p1/evals/physics")
    p.add_argument("--figures_dir", default="p1/figures/physics")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
