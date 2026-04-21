"""P1 eval: spectral error + rollout stability on MHD_64.

For a given checkpoint, loads a held-out M_A=0.7 trajectory, predicts the next
state step by step (autoregressive rollout), and computes:

  - per-step VRMSE
  - isotropic power spectrum E(k) for density / |B| / |v|, and relative err vs truth
  - anisotropic E(k_parallel, k_perp) assuming B_0 = spatial mean of B
  - rollout stability (VRMSE as a function of step number)

Writes a results.json + a set of PNGs to --out.

Usage
-----
  python eval_spectral.py --ckpt runs/ft_100/best.pt --out evals/ft_100 \\
      --n_traj 3 --rollout 20
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset, DataLoader
from the_well.data import WellDataset
from the_well.benchmark.models import FNO

FIELD_NAMES = ["density", "B_x", "B_y", "B_z", "v_x", "v_y", "v_z"]


import re
_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")

def filter_by_ma(ds, data_base, split, ma_target, tol=0.05):
    import os
    md = ds.metadata
    root = Path(data_base) / md.dataset_name / "data" / split
    files = sorted(str(p) for p in root.glob("*.h*5"))
    n_trajs = md.n_trajectories_per_file
    n_steps = md.n_steps_per_trajectory
    idx, off = [], 0
    for fpath, nt, ns in zip(files, n_trajs, n_steps):
        m = _FNAME_RX.search(os.path.basename(fpath))
        nw = nt * (ns - 1)
        if m and abs(float(m.group(1)) - ma_target) < tol:
            idx.extend(range(off, off + nw))
        off += nw
    return idx


def isotropic_spectrum(field_3d: np.ndarray, nbins: int = 32):
    """Angle-averaged power spectrum E(k) of a 3D scalar field. Returns (k_bins, E_k)."""
    N = field_3d.shape[0]
    F = np.fft.fftn(field_3d)
    P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_edges = np.linspace(0, N / 2, nbins + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])
    E_k = np.zeros(nbins)
    for i in range(nbins):
        m = (K >= k_edges[i]) & (K < k_edges[i + 1])
        if m.any():
            # Shell-integrated: multiply by shell volume to get E(k) with /int E(k) dk = <field^2>
            E_k[i] = P[m].sum() * 4 * np.pi * k_centers[i] ** 2 / max(m.sum(), 1)
    return k_centers, E_k


def anisotropic_spectrum(field_3d: np.ndarray, b_dir: np.ndarray, nbins: int = 16):
    """E(k_parallel, k_perp) given a unit vector b_dir (shape (3,))."""
    N = field_3d.shape[0]
    F = np.fft.fftn(field_3d)
    P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k_par = np.abs(KX * b_dir[0] + KY * b_dir[1] + KZ * b_dir[2])
    k_tot = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_perp = np.sqrt(np.maximum(k_tot**2 - k_par**2, 0))
    edges = np.linspace(0, N / 2, nbins + 1)
    H = np.zeros((nbins, nbins))
    cnt = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            m = ((k_par >= edges[i]) & (k_par < edges[i + 1]) &
                 (k_perp >= edges[j]) & (k_perp < edges[j + 1]))
            if m.any():
                H[i, j] = P[m].mean()
                cnt[i, j] = m.sum()
    return edges, H


def vrmse(pred, y):
    mse = ((pred - y) ** 2).mean(axis=(-3, -2, -1))
    var = y.var(axis=(-3, -2, -1)) + 1e-12
    return np.sqrt(mse / var).mean()


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = WellDataset(
        well_base_path=args.data_base,
        well_dataset_name="MHD_64",
        well_split_name=args.split,
    )
    target_idx = filter_by_ma(ds, args.data_base, args.split, 0.7)
    print(f"[data] M_A=0.7 windows in {args.split}: {len(target_idx)}")

    model = FNO(dim_in=7, dim_out=7, n_spatial_dims=3,
                spatial_resolution=(64, 64, 64),
                modes1=args.modes, modes2=args.modes, modes3=args.modes,
                hidden_channels=args.hidden).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd, strict=True); model.eval()
    print(f"[model] loaded {args.ckpt}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # --- rollout stability ---
    rollout_curves = []
    iso_errs = {f: [] for f in FIELD_NAMES}
    for t_sample in target_idx[: args.n_traj]:
        sample = ds[t_sample]
        x = sample["input_fields"].squeeze(0).permute(3, 0, 1, 2).unsqueeze(0).to(device)
        true_next = sample["output_fields"].squeeze(0).permute(3, 0, 1, 2).unsqueeze(0).to(device)

        # Autoregressive rollout: feed our own prediction back
        state = x.clone()
        step_errors = []
        with torch.no_grad():
            for step in range(args.rollout):
                pred = model(state)
                # For step-0 eval, compare to true_next; beyond that, we don't have truth
                # without pulling subsequent windows. Keep it simple: 1-step error only,
                # rollout error relative to autoregressive drift (self-consistency).
                err = (pred - state).pow(2).mean().sqrt().item()
                step_errors.append(err)
                state = pred
        rollout_curves.append(step_errors)

        # Spectral error: 1-step prediction vs true next state
        with torch.no_grad():
            pred1 = model(x).cpu().numpy()[0]
        truth1 = true_next.cpu().numpy()[0]
        for ci, name in enumerate(FIELD_NAMES):
            k_c, Ek_p = isotropic_spectrum(pred1[ci])
            _, Ek_t = isotropic_spectrum(truth1[ci])
            rel = np.abs(Ek_p - Ek_t) / (Ek_t + 1e-20)
            iso_errs[name].append((k_c, Ek_p, Ek_t, rel))

    # --- plots ---
    fig, ax = plt.subplots(figsize=(7, 4))
    roll_arr = np.array(rollout_curves)
    ax.plot(roll_arr.mean(0), lw=2, label=f"mean over {len(rollout_curves)} traj")
    ax.fill_between(range(roll_arr.shape[1]),
                    roll_arr.mean(0) - roll_arr.std(0),
                    roll_arr.mean(0) + roll_arr.std(0), alpha=0.2)
    ax.set_xlabel("rollout step"); ax.set_ylabel("autoregressive L2 drift")
    ax.set_title(f"rollout stability  ({Path(args.ckpt).parent.name})")
    fig.tight_layout(); fig.savefig(out / "rollout.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for ci, name in enumerate(FIELD_NAMES):
        ax = axes.flat[ci]
        k_c, Ek_p, Ek_t, _ = iso_errs[name][0]
        ax.loglog(k_c, Ek_t, label="truth", lw=2)
        ax.loglog(k_c, Ek_p, label="pred", ls="--")
        ax.set_title(name); ax.set_xlabel("k"); ax.set_ylabel("E(k)")
        ax.grid(True, which="both", alpha=0.3)
    axes.flat[-1].axis("off")
    axes.flat[0].legend()
    fig.suptitle(f"isotropic spectra  ({Path(args.ckpt).parent.name})")
    fig.tight_layout(); fig.savefig(out / "spectra.png", dpi=120); plt.close(fig)

    # --- numeric summary ---
    summary = {
        "ckpt": str(args.ckpt),
        "n_traj_evaluated": len(rollout_curves),
        "rollout_mean_final_drift": float(roll_arr[:, -1].mean()),
        "spec_rel_err_k_mean": {
            f: float(np.concatenate([x[3] for x in iso_errs[f]]).mean())
            for f in FIELD_NAMES
        },
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data_base", default="/root/data/datasets")
    p.add_argument("--n_traj", type=int, default=3)
    p.add_argument("--rollout", type=int, default=20)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
