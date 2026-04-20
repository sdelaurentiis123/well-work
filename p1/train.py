"""P1 training: FNO3D on MHD_64, split by M_A.

Modes
-----
  baseline : train from scratch on M_A=0.7 (target) only
  pretrain : train from scratch on M_A=2.0 (source) only, save ckpt
  finetune : load --init_ckpt, fine-tune on M_A=0.7 at --data_frac

Usage
-----
  python train.py --mode pretrain  --epochs 20 --out runs/pretrain
  python train.py --mode baseline  --epochs 20 --out runs/baseline
  python train.py --mode finetune  --init_ckpt runs/pretrain/best.pt \\
                  --data_frac 0.1 --epochs 20 --out runs/ft_10
"""
from __future__ import annotations
import argparse, os, json, re, time, math, random, glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from the_well.data import WellDataset
from the_well.benchmark.models import FNO

TARGET_MA = 0.7   # anisotropic, sub-Alfvenic, fusion-analog
SOURCE_MA = 2.0   # isotropic, super-Alfvenic, ISM-regime
MA_TOL = 0.05     # float match tolerance

_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")


def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def filter_indices_by_ma(ds: WellDataset, data_base: str, split: str,
                         ma_target: float, tol: float = MA_TOL) -> list[int]:
    """Fast filter: parse M_A from HDF5 filenames (MHD_Ma_X_Ms_Y.hdf5) and map to
    window indices via the dataset's n_trajectories_per_file metadata. O(n_files).

    Assumes WellDataset visits files in sorted filename order and that
    n_trajectories_per_file aligns with that order. Window count per trajectory
    = n_steps_per_trajectory - 1 (pairs of adjacent steps).
    """
    md = ds.metadata
    root = Path(data_base) / md.dataset_name / "data" / split
    files = sorted(str(p) for p in root.glob("*.h*5"))
    if not files:
        raise RuntimeError(f"filter_indices_by_ma: no HDF5 files in {root}")

    n_trajs = md.n_trajectories_per_file
    n_steps = md.n_steps_per_trajectory
    if len(n_trajs) != len(files):
        raise RuntimeError(f"n_trajectories_per_file has {len(n_trajs)} entries but {len(files)} files in {root}")

    idx = []
    window_offset = 0
    for fpath, n_traj, n_step in zip(files, n_trajs, n_steps):
        m = _FNAME_RX.search(os.path.basename(fpath))
        n_windows_in_file = n_traj * (n_step - 1)
        if m and abs(float(m.group(1)) - ma_target) < tol:
            idx.extend(range(window_offset, window_offset + n_windows_in_file))
        window_offset += n_windows_in_file
    return idx


def prep(batch, device):
    # (B, T=1, D, H, W, C) -> (B, C, D, H, W)
    x = batch["input_fields"].squeeze(1).permute(0, 4, 1, 2, 3).contiguous().to(device)
    y = batch["output_fields"].squeeze(1).permute(0, 4, 1, 2, 3).contiguous().to(device)
    return x, y


def vrmse(pred, y):
    """Variance-scaled RMSE, per the Well's default metric."""
    mse = (pred - y).pow(2).mean(dim=(2, 3, 4))     # (B, C)
    var = y.var(dim=(2, 3, 4), unbiased=False) + 1e-12
    return (mse / var).sqrt().mean()


def make_model(args, device):
    m = FNO(
        dim_in=7, dim_out=7, n_spatial_dims=3,
        spatial_resolution=(64, 64, 64),
        modes1=args.modes, modes2=args.modes, modes3=args.modes,
        hidden_channels=args.hidden,
    ).to(device)
    return m


def load_ckpt(model, path, device):
    sd = torch.load(path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded {path}  missing={len(missing)} unexpected={len(unexpected)}")


def save_ckpt(model, opt, epoch, val, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": epoch, "val_vrmse": val}, path)


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] mode={args.mode} device={device} out={args.out}")

    # --- wandb ---
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT", "well-work-p1"),
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=args.run_name or Path(args.out).name,
                   config=vars(args), tags=[args.mode])

    # --- data ---
    print(f"[data] base={args.data_base}  resolving MHD_64 windows by M_A split...")
    ds = WellDataset(
        well_base_path=args.data_base,
        well_dataset_name="MHD_64",
        well_split_name="train",
    )
    val_ds = WellDataset(
        well_base_path=args.data_base,
        well_dataset_name="MHD_64",
        well_split_name="valid",
    )

    train_ma = TARGET_MA if args.mode in ("baseline", "finetune") else SOURCE_MA
    val_ma = TARGET_MA if args.mode in ("baseline", "finetune") else SOURCE_MA

    t0 = time.time()
    train_idx = filter_indices_by_ma(ds, args.data_base, "train", train_ma)
    val_idx = filter_indices_by_ma(val_ds, args.data_base, "valid", val_ma)
    print(f"[data] train(M_A={train_ma}): {len(train_idx)} windows  "
          f"val(M_A={val_ma}): {len(val_idx)} windows  ({time.time()-t0:.1f}s)")

    # sub-sample for fine-tune data fraction
    if args.data_frac < 1.0:
        random.shuffle(train_idx)
        k = max(1, int(len(train_idx) * args.data_frac))
        train_idx = train_idx[:k]
        print(f"[data] fine-tune data_frac={args.data_frac} -> {len(train_idx)} train windows")

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=args.bs,
                              shuffle=True, num_workers=args.workers,
                              pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(val_ds, val_idx), batch_size=args.bs,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    # --- model ---
    model = make_model(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] FNO3D  params={n_params/1e6:.2f}M")
    if args.init_ckpt:
        load_ckpt(model, args.init_ckpt, device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * max(1, len(train_loader)))

    # --- loop ---
    best_val = math.inf
    log_path = Path(args.out) / "log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    step = 0
    for ep in range(args.epochs):
        model.train()
        ep_t0, losses = time.time(), []
        for batch in train_loader:
            x, y = prep(batch, device)
            pred = model(x)
            loss = vrmse(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); step += 1
            if use_wandb and step % 10 == 0:
                wandb.log({"train/vrmse": loss.item(), "lr": sched.get_last_lr()[0], "step": step})
        tr = float(np.mean(losses))

        # --- val ---
        model.eval()
        with torch.no_grad():
            vl = []
            for batch in val_loader:
                x, y = prep(batch, device)
                vl.append(vrmse(model(x), y).item())
        val = float(np.mean(vl)) if vl else math.nan
        ep_dt = time.time() - ep_t0
        print(f"[ep {ep:02d}] train={tr:.4f}  val={val:.4f}  time={ep_dt:.1f}s")
        rec = {"epoch": ep, "train_vrmse": tr, "val_vrmse": val, "time_s": ep_dt}
        log_path.open("a").write(json.dumps(rec) + "\n")
        if use_wandb:
            wandb.log({"epoch/train_vrmse": tr, "epoch/val_vrmse": val, "epoch": ep})

        if val < best_val:
            best_val = val
            save_ckpt(model, opt, ep, val, Path(args.out) / "best.pt")
            print(f"  -> new best {val:.4f}, ckpt saved")

    save_ckpt(model, opt, args.epochs - 1, val, Path(args.out) / "last.pt")
    print(f"[done] best_val_vrmse={best_val:.4f}")
    if use_wandb:
        wandb.summary["best_val_vrmse"] = best_val
        wandb.finish()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["baseline", "pretrain", "finetune"], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--data_base", default="/root/data",
                   help="local data root (with datasets/MHD_64/ inside) or hf://datasets/polymathic-ai/")
    p.add_argument("--init_ckpt", default=None,
                   help="state_dict to load before training (finetune mode)")
    p.add_argument("--data_frac", type=float, default=1.0,
                   help="fraction of target training set to use (finetune)")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--modes", type=int, default=12)
    p.add_argument("--hidden", type=int, default=48)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run_name", default=None)
    p.add_argument("--no_wandb", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
