"""Per-trajectory rollout VRMSE for four conditions. Checks for outliers driving means."""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

conditions = [
    ("baseline_01",   "scratch 1%",           "#c0392b"),
    ("baseline",      "scratch 100%",         "#7f8c8d"),
    ("ft_01",         "MHD pretrain + FT",    "#2c5aa0"),
    ("pretrain_ood",  "pretrain zero-FT (OOD)","#e67e22"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharey=False)
for ax, (name, lbl, color) in zip(axes.flat, conditions):
    p = f"p1/evals/physics/{name}/rollout_vrmse_full.npz"
    try:
        v = np.load(p)["vrmse"]
    except Exception as e:
        ax.set_title(f"{lbl} — missing data"); continue
    steps = 1 + np.arange(v.shape[1])
    # plot each traj as thin line, mean bold
    for i in range(v.shape[0]):
        ax.plot(steps, v[i], lw=0.8, color=color, alpha=0.35)
    ax.plot(steps, v.mean(0), lw=2.2, color=color, label="mean")
    ax.plot(steps, np.median(v, axis=0), lw=1.4, color="black", ls="--", label="median")
    ax.set_title(f"{lbl}\nstep-1 VRMSE = {v[:,0].mean():.3f} ± {v[:,0].std():.3f}  •  step-50 VRMSE = {v[:,49].mean():.3f} ± {v[:,49].std():.3f}", fontsize=10)
    ax.set_xlabel("rollout step"); ax.set_ylabel("VRMSE vs truth")
    ax.set_yscale("log")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

fig.suptitle("Per-trajectory rollout VRMSE (thin lines = individual trajectories)\n"
             "Check whether outliers drive means, and whether conditions are bimodal", fontsize=11)
fig.tight_layout()
fig.savefig("p1/figures/p1_per_traj_rollout.png", dpi=150)
print("wrote p1/figures/p1_per_traj_rollout.png")

# Also compute: does any trajectory in each condition show qualitatively different behavior?
print("\n=== Per-trajectory step-50 VRMSE ===")
for name, lbl, _ in conditions:
    p = f"p1/evals/physics/{name}/rollout_vrmse_full.npz"
    try:
        v = np.load(p)["vrmse"][:, 49]
        print(f"{lbl:<30s}  [{', '.join(f'{x:.2f}' for x in sorted(v))}]")
    except: pass
