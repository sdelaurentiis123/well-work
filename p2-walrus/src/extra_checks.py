"""Extra sanity checks: (a) FP64 rollout + (d) perturbation-seed test.

Two modes via --mode:

  --mode=fp64
      Run Walrus rollout on 1 trajectory with model + inputs cast to torch.float64.
      Compare per-step variance trajectory to FP32 baseline. If divergence is
      numerical, FP64 should slow it dramatically. If structural, FP64 explodes
      at the same rate.

  --mode=perturb
      Run Walrus rollout on the SAME trajectory N=5 times, each with a different
      Gaussian perturbation (std=1e-6) added to the 3-frame conditioning input.
      If symmetry breaks the same way every time → learned asymmetry in Walrus
      (uninteresting). If splits between B_y and B_z across seeds → genuine
      dynamical instability with seed-dependent basin selection (publishable).

Usage:
    python -m src.extra_checks --mode fp64 \
        --test_dir <path> --ckpt_path <path> --config_path <path> \
        --well_base_path <path> --out_dir <path>

    python -m src.extra_checks --mode perturb --n_seeds 5 \
        --test_dir <path> --ckpt_path <path> --config_path <path> \
        --well_base_path <path> --out_dir <path>
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch

from .extract_physics import load_test_trajectories, conservation_step
from .rollout_adapter import WalrusAdapter


def _per_step_variance(traj: np.ndarray) -> np.ndarray:
    """traj: (K, 7, 64, 64, 64). Returns (K, 7) per-step per-channel spatial variance."""
    return traj.reshape(traj.shape[0], 7, -1).var(axis=2)


def _per_step_conservation(traj: np.ndarray) -> dict:
    """traj: (K, 7, 64, 64, 64). Returns dict of (K,) arrays."""
    K = traj.shape[0]
    out = {k: np.zeros(K) for k in ("E_B", "E_K", "E_ratio", "divB_norm")}
    for s in range(K):
        c = conservation_step(traj[s])
        for k in out:
            out[k][s] = c[k]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["fp64", "perturb"], required=True)
    p.add_argument("--n_seeds", type=int, default=5, help="for --mode=perturb")
    p.add_argument("--perturb_amplitude", type=float, default=1e-6)
    p.add_argument("--test_dir", required=True, type=Path)
    p.add_argument("--ckpt_path", required=True, type=Path)
    p.add_argument("--config_path", required=True, type=Path)
    p.add_argument("--well_base_path", required=True, type=Path)
    p.add_argument("--out_dir", required=True, type=Path)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--max_history", type=int, default=3)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    device = torch.device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # We use trajectory index 0 as the canonical test case for both checks.
    print(f"[extra_checks] mode={args.mode} loading 1 truth trajectory...")
    truth = load_test_trajectories(
        args.test_dir, n_traj=1, n_frames=args.max_history + args.K,
    )  # (1, 53, 7, 64³)
    history = truth[0, : args.max_history]    # (3, 7, 64³)
    targets = truth[0, args.max_history : args.max_history + args.K]  # (50, 7, 64³)
    print(f"[extra_checks] truth shape: {truth.shape}")
    print(f"[extra_checks] history shape: {history.shape}, targets shape: {targets.shape}")

    print(f"[extra_checks] loading Walrus...")
    adapter = WalrusAdapter(
        checkpoint_path=args.ckpt_path,
        config_path=args.config_path,
        well_base_path=args.well_base_path,
        device=device,
    )

    if args.mode == "fp64":
        # Cast model to fp64 (and revin internals follow data dtype)
        print(f"[extra_checks] casting model to fp64")
        adapter.model = adapter.model.double()
        # We need history to be passed as fp64; rollout signature takes np.ndarray.
        # Convert history to float64; the adapter's _build_batch_from_history uses
        # torch.tensor(np.array) which preserves dtype.
        hist64 = history.astype(np.float64)
        try:
            pred = adapter.rollout(hist64, args.K)  # (K, 7, 64, 64, 64)
            print(f"[extra_checks] pred dtype: {pred.dtype}, shape: {pred.shape}")
        except Exception as e:
            print(f"[extra_checks] FP64 rollout errored: {e}")
            # fall back to checking why
            raise
        var = _per_step_variance(pred)         # (K, 7)
        cons = _per_step_conservation(pred)
        np.savez(
            args.out_dir / "fp64_rollout.npz",
            pred_var=var,
            pred_E_B=cons["E_B"],
            pred_divB_norm=cons["divB_norm"],
            pred_E_ratio=cons["E_ratio"],
            history=history,
            targets=targets,
            channels=np.array(["rho", "Bx", "By", "Bz", "vx", "vy", "vz"]),
        )
        print(f"[extra_checks] wrote {args.out_dir / 'fp64_rollout.npz'}")
        # Report key numbers
        print(f"\n  per-step variance (FP64) at selected steps:")
        for s in [0, 10, 25, 49]:
            print(f"    step {s+1:>2}: " + ", ".join(
                f"{c}={var[s, i]:.3e}"
                for i, c in enumerate(["rho", "Bx", "By", "Bz", "vx", "vy", "vz"])
            ))

    elif args.mode == "perturb":
        print(f"[extra_checks] running {args.n_seeds} perturbation seeds, amplitude={args.perturb_amplitude}")
        rng = np.random.default_rng(42)
        all_var = np.zeros((args.n_seeds, args.K, 7))
        all_div = np.zeros((args.n_seeds, args.K))
        all_eb = np.zeros((args.n_seeds, args.K))
        for s in range(args.n_seeds):
            perturbed_history = history + args.perturb_amplitude * rng.standard_normal(history.shape).astype(history.dtype)
            print(f"\n[extra_checks] seed {s}:")
            pred = adapter.rollout(perturbed_history, args.K)  # (K, 7, 64³)
            all_var[s] = _per_step_variance(pred)
            cons = _per_step_conservation(pred)
            all_div[s] = cons["divB_norm"]
            all_eb[s] = cons["E_B"]
            # report which component blew up at step 50
            v50 = all_var[s, -1]
            ch = ["rho", "Bx", "By", "Bz", "vx", "vy", "vz"]
            big = [(ch[i], v50[i]) for i in range(7) if v50[i] > 100]
            print(f"  step-50 channels with var>100: {big}")
        np.savez(
            args.out_dir / "perturb_rollouts.npz",
            per_seed_variance=all_var,
            per_seed_divB_norm=all_div,
            per_seed_E_B=all_eb,
            seeds=np.arange(args.n_seeds),
            amplitude=args.perturb_amplitude,
        )
        print(f"\n[extra_checks] wrote {args.out_dir / 'perturb_rollouts.npz'}")

    print("[extra_checks] done")


if __name__ == "__main__":
    main()
