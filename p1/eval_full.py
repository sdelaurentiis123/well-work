"""P1 full eval suite — addresses Tier 2 Q6 + Tier 3 Q8,9,11,14 + Tier 5 Q16.

For a given checkpoint, on held-out M_A=0.7 test trajectories:
  Q6: spectral eval with --n_traj 15 (tight error bars).
  Q8: multi-step rollout vs ground truth — per-step VRMSE and spectral error.
  Q9: anisotropic E(k_parallel, k_perp) with B_0 aligned via per-sample mean B.
  Q11: absolute spectral error alongside relative (kills B_y/B_z divide-by-near-zero).
  Q14: real-space pred vs truth snapshot slices for density, |B|, |v|.
  Q16: compute-cost accounting from log.jsonl (wall-clock, total steps, hours).

Writes PNGs + results.json to --out.
"""
from __future__ import annotations
import argparse, json, math, time
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Subset
from the_well.data import WellDataset
from the_well.benchmark.models import FNO

import re, os
_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")
FIELD_NAMES = ["density", "B_x", "B_y", "B_z", "v_x", "v_y", "v_z"]


# --- helpers ----------------------------------------------------------------
def filter_by_ma(ds, data_base, split, ma_target, tol=0.05):
    md = ds.metadata
    root = Path(data_base) / md.dataset_name / "data" / split
    files = sorted(str(p) for p in root.glob("*.h*5"))
    idx, off = [], 0
    for fpath, nt, ns in zip(files, md.n_trajectories_per_file, md.n_steps_per_trajectory):
        m = _FNAME_RX.search(os.path.basename(fpath))
        nw = nt * (ns - 1)
        if m and abs(float(m.group(1)) - ma_target) < tol:
            idx.extend(range(off, off + nw))
        off += nw
    return idx


def to_chw(sample, key="input_fields"):
    """(1, T=1, D, H, W, C) or (T=1, D, H, W, C) -> (C, D, H, W) numpy."""
    arr = sample[key]
    if arr.ndim == 6:
        arr = arr[0]
    return arr.squeeze(0).permute(3, 0, 1, 2).numpy()


def vrmse_cxyz(pred, truth):
    """Per-sample per-field VRMSE. pred/truth shape (C, D, H, W)."""
    mse = ((pred - truth) ** 2).mean(axis=(1, 2, 3))
    var = truth.var(axis=(1, 2, 3)) + 1e-12
    return np.sqrt(mse / var).mean()


# --- spectra ----------------------------------------------------------------
def isotropic_spectrum(field_3d: np.ndarray, nbins: int = 32):
    N = field_3d.shape[0]
    F = np.fft.fftn(field_3d)
    P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    edges = np.linspace(0, N / 2, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    Ek = np.zeros(nbins)
    for i in range(nbins):
        m = (K >= edges[i]) & (K < edges[i + 1])
        if m.any():
            Ek[i] = P[m].sum()
    return centers, Ek


def anisotropic_spectrum(field_3d: np.ndarray, b_dir: np.ndarray, nbins: int = 12):
    """E(k_par, k_perp) given unit vector b_dir."""
    N = field_3d.shape[0]
    F = np.fft.fftn(field_3d)
    P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k_par = np.abs(KX * b_dir[0] + KY * b_dir[1] + KZ * b_dir[2])
    k_mag = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_perp = np.sqrt(np.maximum(k_mag**2 - k_par**2, 0))
    edges = np.linspace(0, N / 2, nbins + 1)
    H = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            m = ((k_par >= edges[i]) & (k_par < edges[i + 1]) &
                 (k_perp >= edges[j]) & (k_perp < edges[j + 1]))
            if m.any():
                H[i, j] = P[m].sum()
    return edges, H


def b0_direction(B_xyz: np.ndarray) -> np.ndarray:
    """Unit vector along spatial-mean B. Shape (3,)."""
    mean_B = B_xyz.mean(axis=(1, 2, 3))  # (3,)
    n = mean_B / (np.linalg.norm(mean_B) + 1e-12)
    return n


# --- multi-step rollout -----------------------------------------------------
def load_trajectory_windows(ds, start_idx, K):
    """Return (state_0, state_1, ..., state_K) truth chain from the dataset.

    Assumes the dataset stores sliding-window pairs of adjacent timesteps in
    file order. start_idx is a window index (0..num_windows-1). We read K+1
    successive windows' outputs as the K+1 ground-truth states starting from
    window start_idx's input and its outputs.
    """
    # state_0 = input of start_idx
    s0 = to_chw(ds[start_idx], "input_fields")
    truth = [s0]
    for k in range(K):
        j = start_idx + k
        if j >= len(ds):
            break
        truth.append(to_chw(ds[j], "output_fields"))
    return truth  # len = K+1 (in best case)


def rollout_vs_truth(model, device, ds, indices, K=50):
    """Autoregressive rollout from N starts, compare per-step vs truth."""
    step_vrmse = np.full((len(indices), K), np.nan)
    step_rel_iso = np.full((len(indices), K, len(FIELD_NAMES)), np.nan)
    for i, start in enumerate(indices):
        truth = load_trajectory_windows(ds, start, K)
        if len(truth) < 2:
            continue
        state = torch.tensor(truth[0]).unsqueeze(0).to(device)
        with torch.no_grad():
            for k in range(min(K, len(truth) - 1)):
                pred = model(state)
                p_np = pred.cpu().numpy()[0]
                t_np = truth[k + 1]
                step_vrmse[i, k] = vrmse_cxyz(p_np, t_np)
                # spectral density rel error for each channel
                for ci in range(len(FIELD_NAMES)):
                    _, Ep = isotropic_spectrum(p_np[ci])
                    _, Et = isotropic_spectrum(t_np[ci])
                    # use log-space error to avoid near-zero blowup
                    step_rel_iso[i, k, ci] = float(
                        np.nanmean(np.abs(np.log10(Ep + 1e-20) - np.log10(Et + 1e-20)))
                    )
                state = pred
    return step_vrmse, step_rel_iso


# --- plotting ---------------------------------------------------------------
def plot_rollout(step_vrmse, out_path, title):
    K = step_vrmse.shape[1]
    mu = np.nanmean(step_vrmse, axis=0)
    sd = np.nanstd(step_vrmse, axis=0)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(1 + np.arange(K), mu, lw=2)
    ax.fill_between(1 + np.arange(K), mu - sd, mu + sd, alpha=0.2)
    ax.set_xlabel("rollout step"); ax.set_ylabel("VRMSE vs ground truth")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_anisotropic(edges_pred, H_pred, edges_truth, H_truth, out_path, title):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmax = np.log10(max(H_truth.max(), H_pred.max()) + 1e-20)
    vmin = vmax - 6
    kwargs = dict(origin="lower", aspect="auto",
                  extent=[edges_truth[0], edges_truth[-1], edges_truth[0], edges_truth[-1]],
                  vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].imshow(np.log10(H_truth + 1e-20), **kwargs)
    axes[0].set_title(r"truth  log$_{10}$ E(k$_\parallel$, k$_\perp$)")
    axes[1].imshow(np.log10(H_pred + 1e-20), **kwargs)
    axes[1].set_title(r"pred  log$_{10}$ E(k$_\parallel$, k$_\perp$)")
    diff = np.log10(H_pred + 1e-20) - np.log10(H_truth + 1e-20)
    im = axes[2].imshow(diff, origin="lower", aspect="auto",
                        extent=[edges_truth[0], edges_truth[-1], edges_truth[0], edges_truth[-1]],
                        vmin=-2, vmax=2, cmap="RdBu_r")
    axes[2].set_title(r"$\log_{10}$(pred/truth)")
    plt.colorbar(im, ax=axes[2])
    for ax in axes:
        ax.set_xlabel(r"k$_\parallel$"); ax.set_ylabel(r"k$_\perp$")
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out_path, dpi=120); plt.close(fig)


def plot_snapshots(pred, truth, out_path, title):
    # Middle z slice for each key field
    fields = {"density": 0, "|B|": None, "|v|": None}
    B_pred = np.sqrt(pred[1:4]**2).sum(0) ** 0.5
    # oops — that's wrong, |B| = sqrt(B_x^2 + B_y^2 + B_z^2)
    B_pred = np.sqrt((pred[1:4]**2).sum(0))
    B_truth = np.sqrt((truth[1:4]**2).sum(0))
    v_pred = np.sqrt((pred[4:7]**2).sum(0))
    v_truth = np.sqrt((truth[4:7]**2).sum(0))
    items = [
        ("density", pred[0], truth[0]),
        ("|B|", B_pred, B_truth),
        ("|v|", v_pred, v_truth),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(11, 10))
    z = pred.shape[-1] // 2
    for row, (name, p, t) in enumerate(items):
        vmin = min(p[:, :, z].min(), t[:, :, z].min())
        vmax = max(p[:, :, z].max(), t[:, :, z].max())
        axes[row, 0].imshow(t[:, :, z], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 0].set_title(f"{name} — truth"); axes[row, 0].set_ylabel(name)
        axes[row, 1].imshow(p[:, :, z], vmin=vmin, vmax=vmax, cmap="viridis")
        axes[row, 1].set_title(f"{name} — pred")
        axes[row, 2].imshow(p[:, :, z] - t[:, :, z], cmap="RdBu_r",
                            vmin=-(t[:, :, z].std()), vmax=t[:, :, z].std())
        axes[row, 2].set_title(f"{name} — diff")
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title); fig.tight_layout()
    fig.savefig(out_path, dpi=120); plt.close(fig)


def compute_cost_from_log(log_path):
    if not log_path.exists():
        return None
    total_s = 0.0; n_ep = 0
    for line in log_path.read_text().splitlines():
        if not line.strip(): continue
        r = json.loads(line)
        total_s += r.get("time_s", 0); n_ep += 1
    return {"total_wall_s": total_s, "hours": total_s / 3600, "n_epochs": n_ep}


# --- main -------------------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = WellDataset(well_base_path=args.data_base, well_dataset_name="MHD_64",
                     well_split_name=args.split)
    target_idx = filter_by_ma(ds, args.data_base, args.split, 0.7)
    print(f"[data] {args.split} M_A=0.7 windows: {len(target_idx)}")

    model = FNO(dim_in=7, dim_out=7, n_spatial_dims=3,
                spatial_resolution=(64, 64, 64),
                modes1=args.modes, modes2=args.modes, modes3=args.modes,
                hidden_channels=args.hidden).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd, strict=True); model.eval()

    # --- Q6: tight spectral eval on n_traj test windows ---
    n_traj = min(args.n_traj, len(target_idx))
    rng = np.random.default_rng(0)
    chosen = list(rng.choice(target_idx, size=n_traj, replace=False))

    step_vrmse_all, rel_iso_all, abs_iso_all = [], [], []
    snapshot_pred, snapshot_truth = None, None
    aniso_pred_sum = aniso_truth_sum = None
    aniso_edges = None

    for i, idx in enumerate(chosen):
        sample = ds[idx]
        x = torch.tensor(to_chw(sample, "input_fields")).unsqueeze(0).to(device)
        truth = to_chw(sample, "output_fields")
        with torch.no_grad():
            pred = model(x).cpu().numpy()[0]
        step_vrmse_all.append(vrmse_cxyz(pred, truth))

        rel_ch, abs_ch = [], []
        for ci in range(len(FIELD_NAMES)):
            k_c, Ep = isotropic_spectrum(pred[ci])
            _, Et = isotropic_spectrum(truth[ci])
            rel_ch.append(float(np.mean(np.abs(Ep - Et) / (Et + 1e-20))))
            abs_ch.append(float(np.mean(np.abs(Ep - Et))))
        rel_iso_all.append(rel_ch); abs_iso_all.append(abs_ch)

        # Q9: anisotropic for B_x (guide-field along x̂ in sub-Alfvénic regime)
        b_dir = b0_direction(truth[1:4])
        edges, Hp = anisotropic_spectrum(pred[1], b_dir)
        _, Ht = anisotropic_spectrum(truth[1], b_dir)
        if aniso_pred_sum is None:
            aniso_pred_sum = Hp; aniso_truth_sum = Ht; aniso_edges = edges
        else:
            aniso_pred_sum += Hp; aniso_truth_sum += Ht

        if i == 0:
            snapshot_pred, snapshot_truth = pred, truth

    rel_iso_mean = np.mean(rel_iso_all, axis=0); rel_iso_std = np.std(rel_iso_all, axis=0)
    abs_iso_mean = np.mean(abs_iso_all, axis=0); abs_iso_std = np.std(abs_iso_all, axis=0)

    # --- Q8: multi-step rollout vs ground truth ---
    # Pick a few starts with enough future windows for K-step rollout
    rollout_starts = [s for s in chosen[:args.n_rollout] if s + args.K < len(ds)]
    K_eff = args.K
    step_v, step_rel = rollout_vs_truth(model, device, ds, rollout_starts, K=K_eff)
    plot_rollout(step_v, out / "rollout_vs_truth.png",
                 f"rollout vs ground truth — {Path(args.ckpt).parent.name}")

    # --- Q9: aniso plot ---
    aniso_pred_mean = aniso_pred_sum / n_traj
    aniso_truth_mean = aniso_truth_sum / n_traj
    plot_anisotropic(aniso_edges, aniso_pred_mean, aniso_edges, aniso_truth_mean,
                     out / "aniso_spectrum_Bx.png",
                     f"B_x anisotropic spectrum — {Path(args.ckpt).parent.name}")

    # --- Q14: real-space snapshots ---
    plot_snapshots(snapshot_pred, snapshot_truth, out / "snapshots.png",
                   f"real-space pred vs truth — {Path(args.ckpt).parent.name}")

    # --- Q16: compute cost from log ---
    log_path = Path(args.ckpt).parent / "log.jsonl"
    cost = compute_cost_from_log(log_path)

    summary = {
        "ckpt": str(args.ckpt),
        "n_traj": n_traj,
        "step1_vrmse_mean": float(np.mean(step_vrmse_all)),
        "step1_vrmse_std": float(np.std(step_vrmse_all)),
        "iso_rel_err_mean": {n: float(rel_iso_mean[i]) for i, n in enumerate(FIELD_NAMES)},
        "iso_rel_err_std": {n: float(rel_iso_std[i]) for i, n in enumerate(FIELD_NAMES)},
        "iso_abs_err_mean": {n: float(abs_iso_mean[i]) for i, n in enumerate(FIELD_NAMES)},
        "iso_abs_err_std": {n: float(abs_iso_std[i]) for i, n in enumerate(FIELD_NAMES)},
        "rollout_K": K_eff,
        "rollout_vrmse_mean_per_step": np.nanmean(step_v, axis=0).tolist(),
        "rollout_vrmse_std_per_step": np.nanstd(step_v, axis=0).tolist(),
        "train_cost": cost,
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2, default=float)[:1500])
    print(f"\nwrote {out}/")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--data_base", default="/root/data/datasets")
    p.add_argument("--n_traj", type=int, default=15)
    p.add_argument("--n_rollout", type=int, default=5)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
