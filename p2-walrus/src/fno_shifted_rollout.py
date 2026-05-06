"""Driver: run all 3 FNO baselines on the shifted window (frames [10] -> [11..60]).

Loads each FNO checkpoint, builds an FNOAdapter, runs run_diagnostics, writes
.npz files to results/shifted_window/<config>/.

Designed to be called from either local (Mac) for quick test runs, or Ginsburg
for the full 10-trajectory campaign. Same code path either way; --device picks.

Usage:
    python -m src.fno_shifted_rollout \
        --test_dir <path-to-MHD_64-test> \
        --ckpt_dir <path-to-FNO-checkpoints> \
        --out_root <path-to-results/shifted_window> \
        --n_traj 10 --K 50 --max_history 10 [--device cuda|cpu|mps]
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from .extract_physics import load_test_trajectories, run_diagnostics
from .rollout_adapter import FNOAdapter


# Map of config name -> checkpoint filename within ckpt_dir.
# Adjust if user organizes checkpoints differently.
FNO_CONFIGS = {
    "fno_baseline": "baseline_01.pt",       # p1/runs/baseline_01/best.pt
    "fno_ft":       "ft_01.pt",             # p1/runs/ft_01/best.pt
    "fno_pretrain_ood": "pretrain.pt",      # p1/runs/pretrain/best.pt (zero-FT control)
}


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True, type=Path)
    p.add_argument("--ckpt_dir", required=True, type=Path)
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--n_traj", type=int, default=10)
    p.add_argument("--K", type=int, default=50)
    p.add_argument(
        "--max_history",
        type=int,
        default=3,
        help="Conditioning window. 1 = Paper-1-style. 3 matches Walrus's empirical T_in for fair comparison.",
    )
    p.add_argument("--device", default=None, help="cuda/cpu/mps; auto-detect if omitted")
    p.add_argument(
        "--configs",
        nargs="+",
        default=list(FNO_CONFIGS.keys()),
        help="Subset of FNO configs to run.",
    )
    return p.parse_args()


def pick_device(arg):
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse()
    device = pick_device(args.device)
    print(f"[fno_shifted_rollout] device={device}")
    print(f"[fno_shifted_rollout] loading {args.n_traj} test trajectories of {args.max_history + args.K} frames each")

    truth = load_test_trajectories(
        args.test_dir,
        n_traj=args.n_traj,
        n_frames=args.max_history + args.K,
    )
    print(f"[fno_shifted_rollout] truth shape: {truth.shape}")

    for cfg in args.configs:
        if cfg not in FNO_CONFIGS:
            print(f"[fno_shifted_rollout] WARN: unknown config {cfg}, skipping")
            continue
        ckpt = args.ckpt_dir / FNO_CONFIGS[cfg]
        if not ckpt.exists():
            print(f"[fno_shifted_rollout] WARN: missing checkpoint {ckpt}, skipping {cfg}")
            continue
        print(f"\n[fno_shifted_rollout] === {cfg} <- {ckpt.name} ===")
        adapter = FNOAdapter(ckpt, device)
        run_diagnostics(
            adapter=adapter,
            truth=truth,
            K=args.K,
            out_dir=args.out_root / cfg,
            max_history=args.max_history,
        )

    print("\n[fno_shifted_rollout] done")


if __name__ == "__main__":
    main()
