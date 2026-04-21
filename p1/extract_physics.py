"""Master physics extraction — one pass per checkpoint.

For a given (ckpt, test-data) pair, rolls forward K steps from N test ICs and dumps:
  - conservation.npz : per-step mass / E_B / E_K / div_B / equipartition
  - aniso_step1.npz  : 1-step anisotropic spectrum (pred + truth) averaged over N ICs
  - field_snapshots.npz : middle-z slices at reference steps for later visualization
  - rollout_vrmse.npz : per-step VRMSE vs ground truth (subsumes eval_full's number)

Reads test data directly via h5py (no the_well dependency needed).
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import torch, h5py
from the_well.benchmark.models import FNO

REF_STEPS = [1, 5, 10, 25, 50]
ROLLOUT_K = 50
CHANNELS = ["density","B_x","B_y","B_z","v_x","v_y","v_z"]
_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")


# ------------ data ----------------------------------------------------------
def load_test_trajectories(test_dir: Path, n_traj: int, ma_target=0.7, K=ROLLOUT_K, seed=0):
    """Grab n_traj (7, K+1, 64, 64, 64) trajectories from M_A=0.7 test files.

    Returns array of shape (n_traj, K+1, 7, 64, 64, 64).
    """
    files = sorted(p for p in test_dir.glob("*.h*5")
                   if abs(float(_FNAME_RX.search(p.name).group(1)) - ma_target) < 0.05)
    rng = np.random.default_rng(seed)
    trajs = []
    # Each test file has 1 trajectory of 100 steps. Pull multiple (distinct) start times
    # per file to reach n_traj samples.
    per_file = int(np.ceil(n_traj / len(files)))
    for fp in files:
        with h5py.File(fp, "r") as h:
            n_st = h["t0_fields"]["density"].shape[1]
            max_start = n_st - K - 1
            if max_start <= 0:
                starts = [0]
            else:
                # evenly spaced start times
                starts = np.linspace(0, max_start, per_file, dtype=int).tolist()
            for t0 in starts:
                if len(trajs) >= n_traj: break
                dens = h["t0_fields"]["density"][0, t0:t0+K+1]
                B = h["t1_fields"]["magnetic_field"][0, t0:t0+K+1]
                v = h["t1_fields"]["velocity"][0, t0:t0+K+1]
                traj = np.empty((K+1, 7, 64, 64, 64), dtype=np.float32)
                traj[:, 0] = dens
                traj[:, 1:4] = np.moveaxis(B, -1, 1)
                traj[:, 4:7] = np.moveaxis(v, -1, 1)
                trajs.append(traj)
        if len(trajs) >= n_traj: break
    return np.stack(trajs[:n_traj], axis=0)


# ------------ diagnostics ---------------------------------------------------
def divergence_B(B: np.ndarray):
    """∇·B using centered FD with periodic BC.  B shape: (3, D, H, W) -> (D, H, W)."""
    dx = np.roll(B[0], -1, 0) - np.roll(B[0], 1, 0)
    dy = np.roll(B[1], -1, 1) - np.roll(B[1], 1, 1)
    dz = np.roll(B[2], -1, 2) - np.roll(B[2], 1, 2)
    return 0.5 * (dx + dy + dz)


def conservation_step(state: np.ndarray):
    """state shape (7, D, H, W). Returns dict of scalars for this snapshot."""
    rho = state[0]
    B = state[1:4]
    v = state[4:7]
    N3 = rho.size
    mass = rho.sum() / N3
    E_B = (B * B).sum() / (2.0 * N3)                     # ∫|B|²/2 dV per unit vol
    E_K = 0.5 * (rho * (v * v).sum(0)).sum() / N3         # ∫½ρv² dV per unit vol
    divB = divergence_B(B)
    Bmag = np.sqrt((B * B).sum(0)) + 1e-12
    divB_norm = float(np.sqrt((divB * divB).mean())) / float(Bmag.mean())
    return dict(mass=float(mass), E_B=float(E_B), E_K=float(E_K),
                E_ratio=float(E_B / (E_K + 1e-12)), divB_norm=divB_norm)


def isotropic_spectrum(field: np.ndarray, nbins: int = 32):
    N = field.shape[0]
    F = np.fft.fftn(field); P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K = np.sqrt(KX**2 + KY**2 + KZ**2)
    edges = np.linspace(0, N/2, nbins+1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    E = np.zeros(nbins)
    for i in range(nbins):
        m = (K >= edges[i]) & (K < edges[i+1])
        if m.any(): E[i] = P[m].sum()
    return centers, E


def anisotropic_spectrum(field: np.ndarray, b_dir: np.ndarray, nbins: int = 16):
    N = field.shape[0]
    F = np.fft.fftn(field); P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k_par = np.abs(KX*b_dir[0] + KY*b_dir[1] + KZ*b_dir[2])
    k_tot = np.sqrt(KX**2+KY**2+KZ**2)
    k_perp = np.sqrt(np.maximum(k_tot**2 - k_par**2, 0))
    edges = np.linspace(0, N/2, nbins+1)
    H = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            m = ((k_par >= edges[i]) & (k_par < edges[i+1]) &
                 (k_perp >= edges[j]) & (k_perp < edges[j+1]))
            if m.any(): H[i,j] = P[m].sum()
    return edges, H


def b0_direction(B: np.ndarray):
    mean_B = B.mean(axis=(1,2,3))
    n = mean_B / (np.linalg.norm(mean_B) + 1e-12)
    return n


# ------------ main ----------------------------------------------------------
def run(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                         else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device}")

    model = FNO(dim_in=7, dim_out=7, n_spatial_dims=3,
                spatial_resolution=(64,64,64),
                modes1=12, modes2=12, modes3=12,
                hidden_channels=48).to(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd, strict=True); model.eval()
    print(f"loaded {args.ckpt}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    # Load test data once
    print(f"loading {args.n_traj} test trajectories...")
    truth = load_test_trajectories(Path(args.test_dir), args.n_traj, K=args.K)
    print(f"truth shape: {truth.shape}")

    # Arrays to collect
    cons_pred = {k: np.zeros((args.n_traj, args.K+1)) for k in
                 ("mass","E_B","E_K","E_ratio","divB_norm")}
    cons_truth = {k: np.zeros((args.n_traj, args.K+1)) for k in cons_pred}
    rollout_vrmse = np.zeros((args.n_traj, args.K))
    variance_per_step = np.zeros((args.n_traj, args.K+1, 7))

    # Aniso spectra: accumulate 2D histograms at step 1 (pred vs truth)
    aniso_pred = None; aniso_truth = None; aniso_edges = None

    # Field snapshots at reference steps for later visualization
    ref_snaps = {"truth": {}, "pred": {}}
    for s in REF_STEPS:
        if s <= args.K:
            ref_snaps["truth"][s] = np.zeros((7, 64, 64, 64), dtype=np.float32)
            ref_snaps["pred"][s] = np.zeros((7, 64, 64, 64), dtype=np.float32)

    with torch.no_grad():
        for t_i in range(args.n_traj):
            print(f"  traj {t_i+1}/{args.n_traj}")
            state = torch.tensor(truth[t_i, 0]).unsqueeze(0).to(device)
            # conservation at step 0
            for kname, v in conservation_step(truth[t_i, 0]).items():
                cons_pred[kname][t_i, 0] = v
                cons_truth[kname][t_i, 0] = v
            variance_per_step[t_i, 0] = truth[t_i, 0].reshape(7,-1).var(axis=1)

            for step in range(1, args.K+1):
                pred = model(state).cpu().numpy()[0]      # (7,64,64,64)
                true_k = truth[t_i, step]
                # per-step conservation / variance
                for kname, v in conservation_step(pred).items():
                    cons_pred[kname][t_i, step] = v
                for kname, v in conservation_step(true_k).items():
                    cons_truth[kname][t_i, step] = v
                variance_per_step[t_i, step] = pred.reshape(7,-1).var(axis=1)
                # VRMSE
                mse = ((pred - true_k) ** 2).mean(axis=(1,2,3))
                var = true_k.var(axis=(1,2,3)) + 1e-12
                rollout_vrmse[t_i, step-1] = float(np.sqrt(mse/var).mean())
                # ref snaps
                if step in ref_snaps["pred"]:
                    ref_snaps["pred"][step] = ref_snaps["pred"][step] + pred / args.n_traj
                    ref_snaps["truth"][step] = ref_snaps["truth"][step] + true_k / args.n_traj
                # step-1 aniso
                if step == 1:
                    b_dir = b0_direction(true_k[1:4])
                    edges, Hp = anisotropic_spectrum(pred[1], b_dir)
                    _, Ht = anisotropic_spectrum(true_k[1], b_dir)
                    if aniso_pred is None:
                        aniso_pred = Hp; aniso_truth = Ht; aniso_edges = edges
                    else:
                        aniso_pred += Hp; aniso_truth += Ht
                state = torch.tensor(pred).unsqueeze(0).to(device)

    aniso_pred /= args.n_traj
    aniso_truth /= args.n_traj

    # Save (flat numeric arrays only — no dicts)
    np.savez_compressed(out / "conservation.npz",
        **{f"pred_{k}": v for k, v in cons_pred.items()},
        **{f"truth_{k}": v for k, v in cons_truth.items()})
    np.savez_compressed(out / "aniso_step1.npz",
                       pred=aniso_pred, truth=aniso_truth, edges=aniso_edges)
    np.savez_compressed(out / "rollout_vrmse_full.npz",
                       vrmse=rollout_vrmse, n_traj=args.n_traj)
    np.savez_compressed(out / "variance.npz",
                       variance_per_step=variance_per_step, channels=np.array(CHANNELS))
    snap_dict = {}
    for s in ref_snaps["truth"]:
        snap_dict[f"truth_step{s}"] = ref_snaps["truth"][s]
        snap_dict[f"pred_step{s}"] = ref_snaps["pred"][s]
    np.savez_compressed(out / "field_snapshots.npz",
                       steps=np.array(list(ref_snaps["truth"].keys())), **snap_dict)
    print(f"wrote {out}/")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test_dir", default="data/MHD_64/data/test")
    p.add_argument("--n_traj", type=int, default=10)
    p.add_argument("--K", type=int, default=ROLLOUT_K)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
