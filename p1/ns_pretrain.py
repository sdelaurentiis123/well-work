"""Task 4: NS-pretrain control — pretrain FNO3D on a non-MHD dataset.

Loads HDF5 files from a Polymathic-Well-format dataset (e.g. supernova_explosion_64)
via h5py directly. Detects all scalar + vector fields, packs them into a channel
axis. Zero-pads (or truncates) to 7 channels so the architecture exactly matches
our MHD-pretrain + ft pipeline.

Saves checkpoint at runs/ns_pretrain/best.pt and last.pt, compatible with
train.py --init_ckpt.
"""
from __future__ import annotations
import argparse, json, math, os, random, time, glob
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from the_well.benchmark.models import FNO


TARGET_CHANNELS = 7


def set_seed(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class NonMHDTrajectoryDataset(Dataset):
    """Opens all *.hdf5 files under a dir, yields consecutive-timestep pairs,
    packs (t0_fields scalars + t1_fields vectors) into a fixed 7-channel tensor.

    Zero-pads if the dataset has fewer than 7 channels; truncates if more.
    """
    def __init__(self, root: str):
        self.files = sorted(glob.glob(os.path.join(root, "*.h*5")))
        if not self.files:
            raise RuntimeError(f"no HDF5 files in {root}")
        # Inventory: for each file, (n_trajectories, n_steps, n_channels_total)
        self.index = []   # list of (file_idx, traj_idx, step_idx) tuples — each is one "input at t, output at t+1" sample
        self.schema = None
        for fi, fp in enumerate(self.files):
            with h5py.File(fp, "r") as h:
                n_tr, n_st = self._detect_shape(h)
                schema = self._detect_schema(h)
                if self.schema is None: self.schema = schema
                for tr in range(n_tr):
                    for st in range(n_st - 1):
                        self.index.append((fi, tr, st))
        print(f"[NS-Dataset] {len(self.files)} files, {len(self.index)} windows")
        print(f"[NS-Dataset] schema: {self.schema}")
        # Rank counts: scalars (t0), vectors (t1, 3 components each). Total channels.
        scalars = len(self.schema.get("t0_fields", []))
        vectors = len(self.schema.get("t1_fields", [])) * 3
        self._raw_channels = scalars + vectors
        print(f"[NS-Dataset] raw channels={self._raw_channels} -> padded to {TARGET_CHANNELS}")

    @staticmethod
    def _detect_shape(h):
        # try t0_fields first (scalars)
        if "t0_fields" in h:
            grp = h["t0_fields"]
            name = list(grp.keys())[0]
            arr = grp[name]
            return int(arr.shape[0]), int(arr.shape[1])
        elif "t1_fields" in h:
            grp = h["t1_fields"]
            name = list(grp.keys())[0]
            arr = grp[name]
            return int(arr.shape[0]), int(arr.shape[1])
        else:
            raise RuntimeError(f"no t0_fields or t1_fields in {h.filename}")

    @staticmethod
    def _detect_schema(h):
        schema = {}
        for key in ("t0_fields", "t1_fields", "t2_fields"):
            if key in h:
                schema[key] = sorted(list(h[key].keys()))
        return schema

    def _load_state(self, h, traj, step):
        """Returns (channels, D, H, W) float32."""
        chans = []
        for name in self.schema.get("t0_fields", []):
            a = h["t0_fields"][name][traj, step]     # (D,H,W)
            chans.append(a)
        for name in self.schema.get("t1_fields", []):
            v = h["t1_fields"][name][traj, step]     # (D,H,W,3)
            for c in range(3):
                chans.append(v[..., c])
        state = np.stack(chans, axis=0).astype(np.float32)   # (raw_C, D, H, W)
        # pad/truncate to 7
        C, D, H, W = state.shape
        if C < TARGET_CHANNELS:
            pad = np.zeros((TARGET_CHANNELS - C, D, H, W), dtype=np.float32)
            state = np.concatenate([state, pad], axis=0)
        elif C > TARGET_CHANNELS:
            state = state[:TARGET_CHANNELS]
        return state

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        fi, tr, st = self.index[idx]
        with h5py.File(self.files[fi], "r") as h:
            x = self._load_state(h, tr, st)
            y = self._load_state(h, tr, st + 1)
        return torch.from_numpy(x), torch.from_numpy(y)


def vrmse(pred, y):
    mse = (pred - y).pow(2).mean(dim=(2, 3, 4))
    var = y.var(dim=(2, 3, 4), unbiased=False) + 1e-12
    return (mse / var).sqrt().mean()


def run(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"[cfg] out={out}  device={device}")

    # wandb optional
    use_wb = not args.no_wandb
    if use_wb:
        import wandb
        wandb.init(project=os.environ.get("WANDB_PROJECT","well-work-p1"),
                   entity=os.environ.get("WANDB_ENTITY"),
                   name=args.run_name or out.name, config=vars(args), tags=["ns_pretrain"])

    # Dataset
    train_ds = NonMHDTrajectoryDataset(args.train_dir)
    # Use first ~10% of windows as "valid" for periodic checkpointing
    rng = np.random.default_rng(0)
    idx_all = np.arange(len(train_ds))
    rng.shuffle(idx_all)
    n_val = max(32, len(idx_all) // 10)
    val_idx = idx_all[:n_val]; train_idx = idx_all[n_val:]
    from torch.utils.data import Subset
    train_loader = DataLoader(Subset(train_ds, train_idx.tolist()), batch_size=args.bs,
                              shuffle=True, num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(Subset(train_ds, val_idx.tolist()), batch_size=args.bs,
                            shuffle=False, num_workers=args.workers, pin_memory=True)
    print(f"[data] train_windows={len(train_idx)}  val_windows={len(val_idx)}")

    # Model
    model = FNO(dim_in=7, dim_out=7, n_spatial_dims=3,
                spatial_resolution=(64, 64, 64),
                modes1=args.modes, modes2=args.modes, modes3=args.modes,
                hidden_channels=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] FNO3D params={n_params/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    steps_per = max(1, len(train_loader))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps_per)

    log_path = out / "log.jsonl"; log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val = math.inf; step = 0
    for ep in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            pred = model(x); loss = vrmse(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            losses.append(loss.item()); step += 1
            if use_wb and step % 10 == 0:
                wandb.log({"train/vrmse": loss.item(), "lr": sched.get_last_lr()[0], "step": step})
        tr = float(np.mean(losses))
        model.eval()
        vls = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                vls.append(vrmse(model(x), y).item())
        val = float(np.mean(vls)) if vls else math.nan
        dt = time.time() - t0
        print(f"[ep {ep:02d}] train={tr:.4f}  val={val:.4f}  time={dt:.1f}s")
        log_path.open("a").write(json.dumps({"epoch":ep,"train_vrmse":tr,"val_vrmse":val,"time_s":dt}) + "\n")
        if use_wb:
            wandb.log({"epoch/train_vrmse": tr, "epoch/val_vrmse": val, "epoch": ep})
        if val < best_val:
            best_val = val
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "epoch": ep, "val_vrmse": val}, out / "best.pt")
            print(f"  -> new best {val:.4f}, ckpt saved")
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                "epoch": args.epochs - 1, "val_vrmse": val}, out / "last.pt")
    print(f"[done] best_val_vrmse={best_val:.4f}")
    if use_wb:
        wandb.summary["best_val_vrmse"] = best_val
        wandb.finish()


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--train_dir", required=True, help="dir with *.hdf5 files for pretrain")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--bs", type=int, default=8)
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
