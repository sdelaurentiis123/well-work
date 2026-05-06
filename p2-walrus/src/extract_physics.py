"""Physics-feature diagnostics, refactored to be model-agnostic.

Lifted from p1/extract_physics.py with two changes:
  1. Rollout is delegated to a `RolloutAdapter` (FNO or Walrus) — no model logic here.
  2. `load_test_trajectories` accepts a `start_offset` so we can rollout from frame 10
     instead of frame 0 (Walrus needs 10 history frames; for fair comparison FNO
     rolls out from the same physical state).

Output `.npz` schema is identical to Paper 1's `p1/evals/physics/{config}/` layout
so the existing `make_figures.py` overlay code works unchanged.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Protocol
import numpy as np
import h5py

REF_STEPS = [1, 5, 10, 25, 50]
ROLLOUT_K = 50
CHANNELS = ["density", "B_x", "B_y", "B_z", "v_x", "v_y", "v_z"]
_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")


# ============================================================================
# Adapter protocol
# ============================================================================
class RolloutAdapter(Protocol):
    """Model-agnostic rollout interface.

    n_history: number of past frames the model needs as input.
        FNO: 1 (single-frame Markov)
        Walrus: 10 (10-frame history conditioning)

    rollout(history, K) -> ndarray of shape (K, 7, 64, 64, 64):
        history: ndarray of shape (n_history, 7, 64, 64, 64).
                 The n_history frames immediately preceding the prediction target.
        Returns: K predicted next-frames.
    """
    n_history: int

    def rollout(self, history: np.ndarray, K: int) -> np.ndarray: ...


# ============================================================================
# Data loading
# ============================================================================
def load_test_trajectories(
    test_dir: Path,
    n_traj: int,
    ma_target: float = 0.7,
    n_frames: int = ROLLOUT_K + 1,
    seed: int = 0,
):
    """Load (n_traj, n_frames, 7, 64, 64, 64) from M_A=0.7 test files.

    Each trajectory is `n_frames` consecutive timesteps starting from a random
    (seeded) offset within the source h5 file.

    For Paper 1's setup (FNO, 2-frame Markov): n_frames = K+1 = 51.
    For shifted-window (Walrus + FNO comparison): n_frames = max_history + K = 60.
        First `max_history` frames are conditioning input; remaining K are targets.
    """
    files = sorted(
        p
        for p in test_dir.glob("*.h*5")
        if abs(float(_FNAME_RX.search(p.name).group(1)) - ma_target) < 0.05
    )
    if not files:
        raise FileNotFoundError(f"No M_A={ma_target} files in {test_dir}")
    rng = np.random.default_rng(seed)
    trajs = []
    per_file = int(np.ceil(n_traj / len(files)))
    for fp in files:
        with h5py.File(fp, "r") as h:
            n_st = h["t0_fields"]["density"].shape[1]
            max_start = n_st - n_frames
            if max_start < 0:
                raise ValueError(
                    f"{fp.name} has only {n_st} timesteps; need ≥{n_frames}."
                )
            elif max_start == 0:
                starts = [0]
            else:
                starts = np.linspace(0, max_start, per_file, dtype=int).tolist()
            for t0 in starts:
                if len(trajs) >= n_traj:
                    break
                dens = h["t0_fields"]["density"][0, t0 : t0 + n_frames]
                B = h["t1_fields"]["magnetic_field"][0, t0 : t0 + n_frames]
                v = h["t1_fields"]["velocity"][0, t0 : t0 + n_frames]
                traj = np.empty((n_frames, 7, 64, 64, 64), dtype=np.float32)
                traj[:, 0] = dens
                traj[:, 1:4] = np.moveaxis(B, -1, 1)
                traj[:, 4:7] = np.moveaxis(v, -1, 1)
                trajs.append(traj)
        if len(trajs) >= n_traj:
            break
    if len(trajs) < n_traj:
        raise RuntimeError(
            f"Only collected {len(trajs)} trajectories from {len(files)} files; "
            f"asked for {n_traj}. Increase per_file or check h5 timestep counts."
        )
    return np.stack(trajs[:n_traj], axis=0)


# ============================================================================
# Pure diagnostic functions (lifted verbatim from p1/extract_physics.py)
# ============================================================================
def divergence_B(B: np.ndarray):
    """∇·B via centered FD, periodic BC. B shape (3, D, H, W) -> (D, H, W)."""
    dx = np.roll(B[0], -1, 0) - np.roll(B[0], 1, 0)
    dy = np.roll(B[1], -1, 1) - np.roll(B[1], 1, 1)
    dz = np.roll(B[2], -1, 2) - np.roll(B[2], 1, 2)
    return 0.5 * (dx + dy + dz)


def conservation_step(state: np.ndarray):
    """state shape (7, D, H, W). Returns scalar dict for this snapshot."""
    rho = state[0]
    B = state[1:4]
    v = state[4:7]
    N3 = rho.size
    mass = rho.sum() / N3
    E_B = (B * B).sum() / (2.0 * N3)
    E_K = 0.5 * (rho * (v * v).sum(0)).sum() / N3
    divB = divergence_B(B)
    Bmag = np.sqrt((B * B).sum(0)) + 1e-12
    divB_norm = float(np.sqrt((divB * divB).mean())) / float(Bmag.mean())
    return dict(
        mass=float(mass),
        E_B=float(E_B),
        E_K=float(E_K),
        E_ratio=float(E_B / (E_K + 1e-12)),
        divB_norm=divB_norm,
    )


def anisotropic_spectrum(field: np.ndarray, b_dir: np.ndarray, nbins: int = 16):
    N = field.shape[0]
    F = np.fft.fftn(field)
    P = (F * F.conj()).real / N**6
    kx = np.fft.fftfreq(N) * N
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    k_par = np.abs(KX * b_dir[0] + KY * b_dir[1] + KZ * b_dir[2])
    k_tot = np.sqrt(KX**2 + KY**2 + KZ**2)
    k_perp = np.sqrt(np.maximum(k_tot**2 - k_par**2, 0))
    edges = np.linspace(0, N / 2, nbins + 1)
    H = np.zeros((nbins, nbins))
    for i in range(nbins):
        for j in range(nbins):
            m = (
                (k_par >= edges[i])
                & (k_par < edges[i + 1])
                & (k_perp >= edges[j])
                & (k_perp < edges[j + 1])
            )
            if m.any():
                H[i, j] = P[m].sum()
    return edges, H


def b0_direction(B: np.ndarray):
    mean_B = B.mean(axis=(1, 2, 3))
    return mean_B / (np.linalg.norm(mean_B) + 1e-12)


# ============================================================================
# Diagnostic driver — model-agnostic
# ============================================================================
def run_diagnostics(
    adapter: RolloutAdapter,
    truth: np.ndarray,
    K: int,
    out_dir: Path,
    max_history: int | None = None,
):
    """Run rollout + compute diagnostic suite.

    Parameters
    ----------
    adapter : RolloutAdapter with .n_history and .rollout(history, K).
    truth : ndarray (n_traj, max_history + K, 7, 64, 64, 64).
        First max_history frames = conditioning window (full window for Walrus,
        last frame only for FNO). Remaining K frames = ground-truth targets.
    K : number of prediction steps.
    out_dir : path for .npz outputs.
    max_history : overrides truth.shape[1] - K. Default = adapter.n_history.
        Should be ≥ adapter.n_history. Use 10 for shifted-window comparison
        (so all configs see identical starting state at frame max_history-1).

    Output files (matching p1/evals/physics/<config>/ schema):
        rollout_vrmse_full.npz  : vrmse (n_traj, K), n_traj
        conservation.npz        : pred_*/truth_* keys, each (n_traj, K+1)
        aniso_step1.npz         : pred (16,16), truth (16,16), edges (17,)
        field_snapshots.npz     : truth_step{1,5,10,25,50}, pred_step{1,5,10,25,50}
        variance.npz            : variance_per_step (n_traj, K+1, 7), channels
    """
    n_traj = truth.shape[0]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if max_history is None:
        max_history = adapter.n_history
    if max_history < adapter.n_history:
        raise ValueError(
            f"max_history={max_history} < adapter.n_history={adapter.n_history}"
        )
    if truth.shape[1] != max_history + K:
        raise ValueError(
            f"truth.shape[1]={truth.shape[1]} != max_history({max_history}) + K({K})"
        )

    # The "starting state" (used for conservation_step at index 0) is the last
    # conditioning frame (frame max_history - 1 in truth). Targets are
    # truth[:, max_history:max_history+K].
    state0_idx = max_history - 1

    cons_keys = ("mass", "E_B", "E_K", "E_ratio", "divB_norm")
    cons_pred = {k: np.zeros((n_traj, K + 1)) for k in cons_keys}
    cons_truth = {k: np.zeros((n_traj, K + 1)) for k in cons_keys}
    rollout_vrmse = np.zeros((n_traj, K))
    variance_per_step = np.zeros((n_traj, K + 1, 7))

    aniso_pred = aniso_truth = aniso_edges = None

    ref_snaps = {"truth": {}, "pred": {}}
    for s in REF_STEPS:
        if s <= K:
            ref_snaps["truth"][s] = np.zeros((7, 64, 64, 64), dtype=np.float32)
            ref_snaps["pred"][s] = np.zeros((7, 64, 64, 64), dtype=np.float32)

    for t_i in range(n_traj):
        print(f"  traj {t_i + 1}/{n_traj}")

        # Conditioning window for the adapter (last n_history frames)
        history_start = max_history - adapter.n_history
        history = truth[t_i, history_start : max_history]  # (n_history, 7, 64³)

        # Rollout
        pred = adapter.rollout(history, K)  # (K, 7, 64, 64, 64)

        # Step 0 = starting state (same for pred and truth, just for conservation continuity)
        for kname, val in conservation_step(truth[t_i, state0_idx]).items():
            cons_pred[kname][t_i, 0] = val
            cons_truth[kname][t_i, 0] = val
        variance_per_step[t_i, 0] = truth[t_i, state0_idx].reshape(7, -1).var(axis=1)

        # Steps 1..K = predictions vs targets
        for step in range(1, K + 1):
            pred_k = pred[step - 1]
            true_k = truth[t_i, max_history + step - 1]

            for kname, val in conservation_step(pred_k).items():
                cons_pred[kname][t_i, step] = val
            for kname, val in conservation_step(true_k).items():
                cons_truth[kname][t_i, step] = val
            variance_per_step[t_i, step] = pred_k.reshape(7, -1).var(axis=1)

            mse = ((pred_k - true_k) ** 2).mean(axis=(1, 2, 3))
            var = true_k.var(axis=(1, 2, 3)) + 1e-12
            rollout_vrmse[t_i, step - 1] = float(np.sqrt(mse / var).mean())

            if step in ref_snaps["pred"]:
                ref_snaps["pred"][step] = ref_snaps["pred"][step] + pred_k / n_traj
                ref_snaps["truth"][step] = ref_snaps["truth"][step] + true_k / n_traj

            if step == 1:
                b_dir = b0_direction(true_k[1:4])
                edges, Hp = anisotropic_spectrum(pred_k[1], b_dir)
                _, Ht = anisotropic_spectrum(true_k[1], b_dir)
                if aniso_pred is None:
                    aniso_pred = Hp
                    aniso_truth = Ht
                    aniso_edges = edges
                else:
                    aniso_pred += Hp
                    aniso_truth += Ht

    aniso_pred /= n_traj
    aniso_truth /= n_traj

    np.savez_compressed(
        out_dir / "conservation.npz",
        **{f"pred_{k}": v for k, v in cons_pred.items()},
        **{f"truth_{k}": v for k, v in cons_truth.items()},
    )
    np.savez_compressed(
        out_dir / "aniso_step1.npz",
        pred=aniso_pred,
        truth=aniso_truth,
        edges=aniso_edges,
    )
    np.savez_compressed(
        out_dir / "rollout_vrmse_full.npz",
        vrmse=rollout_vrmse,
        n_traj=n_traj,
    )
    np.savez_compressed(
        out_dir / "variance.npz",
        variance_per_step=variance_per_step,
        channels=np.array(CHANNELS),
    )
    snap_dict = {}
    for s in ref_snaps["truth"]:
        snap_dict[f"truth_step{s}"] = ref_snaps["truth"][s]
        snap_dict[f"pred_step{s}"] = ref_snaps["pred"][s]
    np.savez_compressed(
        out_dir / "field_snapshots.npz",
        steps=np.array(list(ref_snaps["truth"].keys())),
        **snap_dict,
    )
    print(f"wrote {out_dir}/")
