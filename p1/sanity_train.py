"""P1 day 2: sanity train FNO3D on a handful of MHD_64 windows (MPS).

Tiny model, tiny dataset, few steps. Just proves the pipeline works end to end.
"""
import os, time
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from the_well.data import WellDataset
from the_well.benchmark.models import FNO

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("device:", DEVICE)

# --- Data -------------------------------------------------------------------
ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="MHD_64",
    well_split_name="train",
)
N_TRAIN = 16
subset = Subset(ds, list(range(N_TRAIN)))
loader = DataLoader(subset, batch_size=1, num_workers=0, shuffle=True)
print(f"train windows: {len(subset)}")

# --- Model ------------------------------------------------------------------
C = 7  # density + B_xyz + v_xyz
model = FNO(
    dim_in=C,
    dim_out=C,
    n_spatial_dims=3,
    spatial_resolution=(64, 64, 64),
    modes1=8, modes2=8, modes3=8,
    hidden_channels=32,
).to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"FNO params: {n_params/1e6:.2f}M")

opt = torch.optim.Adam(model.parameters(), lr=1e-3)

def prep(batch):
    # input_fields: (B, T=1, 64, 64, 64, C) -> (B, C, 64, 64, 64)
    x = batch["input_fields"].squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
    y = batch["output_fields"].squeeze(1).permute(0, 4, 1, 2, 3).contiguous()
    return x.to(DEVICE), y.to(DEVICE)

# --- Few steps --------------------------------------------------------------
model.train()
t0 = time.time()
for step, batch in enumerate(loader):
    x, y = prep(batch)
    # the_well FNO expects (B, D, H, W, C) per its patcher; try both
    try:
        pred = model(x)
    except Exception:
        # fall back: maybe it wants channels-last
        x_cl = x.permute(0, 2, 3, 4, 1).contiguous()
        pred = model(x_cl).permute(0, 4, 1, 2, 3).contiguous()
    if pred.shape != y.shape:
        # normalize shape one more way
        if pred.ndim == 5 and pred.shape[-1] == C and y.shape[1] == C:
            pred = pred.permute(0, 4, 1, 2, 3).contiguous()
    loss = nn.functional.mse_loss(pred, y)
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"  step {step:02d}  loss={loss.item():.4f}  pred.shape={tuple(pred.shape)}")
    if step >= 5:
        break
print(f"elapsed: {time.time()-t0:.1f}s")
