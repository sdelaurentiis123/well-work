"""Driver: run Walrus zero-shot on the shifted window (frames [10] -> [11..60]).

Loads Walrus 1.3B once, builds a WalrusAdapter, runs run_diagnostics on the
same 10 M_A=0.7 trajectories the FNO driver uses, writes .npz files to
results/shifted_window/walrus/.

Usage:
    python -m src.walrus_rollout \
        --test_dir <path-to-MHD_64-test> \
        --ckpt_path <path-to-walrus.pt> \
        --config_path <path-to-extended_config.yaml> \
        --well_base_path <path-containing-MHD_64-dir> \
        --out_root <path-to-results/shifted_window> \
        --n_traj 10 --K 50 --max_history 10
"""
from __future__ import annotations
import argparse
from pathlib import Path
import torch

from .extract_physics import load_test_trajectories, run_diagnostics
from .rollout_adapter import WalrusAdapter


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--test_dir", required=True, type=Path)
    p.add_argument("--ckpt_path", required=True, type=Path)
    p.add_argument("--config_path", required=True, type=Path)
    p.add_argument(
        "--well_base_path",
        required=True,
        type=Path,
        help="Directory containing MHD_64/ subdir for Walrus's data_module.",
    )
    p.add_argument("--out_root", required=True, type=Path)
    p.add_argument("--n_traj", type=int, default=10)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--max_history", type=int, default=10)
    p.add_argument("--device", default=None)
    return p.parse_args()


def main():
    args = parse()
    if args.max_history != 10:
        raise ValueError(
            "Walrus is hardcoded for n_history=10 (per configs/data/MHD_64.yaml:8)"
        )

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[walrus_rollout] device={device}")

    print(f"[walrus_rollout] loading {args.n_traj} test trajectories of {args.max_history + args.K} frames")
    truth = load_test_trajectories(
        args.test_dir,
        n_traj=args.n_traj,
        n_frames=args.max_history + args.K,
    )
    print(f"[walrus_rollout] truth shape: {truth.shape}")

    print(f"[walrus_rollout] loading Walrus from {args.ckpt_path}")
    adapter = WalrusAdapter(
        checkpoint_path=args.ckpt_path,
        config_path=args.config_path,
        well_base_path=args.well_base_path,
        device=device,
    )

    print(f"[walrus_rollout] running diagnostics → {args.out_root / 'walrus'}")
    run_diagnostics(
        adapter=adapter,
        truth=truth,
        K=args.K,
        out_dir=args.out_root / "walrus",
        max_history=args.max_history,
    )
    print("[walrus_rollout] done")


if __name__ == "__main__":
    main()
