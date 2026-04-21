"""Physics probe 5: scaling invariance test.

MHD is invariant under: ρ → α ρ, B → β B, v → β/√α v, t → t √α / β.
Test whether each model respects it.

Given a test snapshot:
  Copy A: β=2, α=1  ⇒  ρ'=ρ, B'=2B, v'=2v. Expected dimensionless dynamics identical.
  Copy B: α=4, β=1  ⇒  ρ'=4ρ, B'=B, v'=v/2. Expected dimensionless dynamics identical.

For scale-invariance of dt: transform A takes t'→t/2, B takes t'→2t. A predicted next
state corresponds to a *different* dimensionless interval under each transform.
Cleanest comparison: rescale the predicted scaled snapshot back to the original units
and compare to the model's prediction on the un-scaled snapshot.
"""
from __future__ import annotations
import argparse, re
from pathlib import Path
import numpy as np
import torch, h5py
from the_well.benchmark.models import FNO

_FNAME_RX = re.compile(r"MHD_Ma_([\d.]+)_Ms_([\d.]+)\.h(?:df5|5)$")


def load_one_snapshot(test_dir: Path, ma_target=0.7, t0=50):
    """Return one (7, 64, 64, 64) snapshot at t=t0 from the first M_A=0.7 test file."""
    files = sorted(p for p in test_dir.glob("*.h*5")
                   if abs(float(_FNAME_RX.search(p.name).group(1)) - ma_target) < 0.05)
    with h5py.File(files[0], "r") as h:
        dens = h["t0_fields"]["density"][0, t0]
        B = h["t1_fields"]["magnetic_field"][0, t0]
        v = h["t1_fields"]["velocity"][0, t0]
    state = np.empty((7, 64, 64, 64), dtype=np.float32)
    state[0] = dens
    state[1:4] = np.moveaxis(B, -1, 0)
    state[4:7] = np.moveaxis(v, -1, 0)
    return state


def transform(state: np.ndarray, alpha: float, beta: float):
    """Apply ρ → α ρ, B → β B, v → β/√α v."""
    out = np.empty_like(state)
    out[0] = alpha * state[0]
    out[1:4] = beta * state[1:4]
    out[4:7] = (beta / np.sqrt(alpha)) * state[4:7]
    return out


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

    snap = load_one_snapshot(Path(args.test_dir))
    # Original prediction
    x = torch.tensor(snap).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_orig = model(x).cpu().numpy()[0]

    # Scaling A: β=2, α=1
    snap_A = transform(snap, alpha=1.0, beta=2.0)
    xA = torch.tensor(snap_A).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_A = model(xA).cpu().numpy()[0]
    # Rescale pred_A back to original units (inverse transform)
    pred_A_back = transform(pred_A, alpha=1.0, beta=0.5)
    # Deviation
    dev_A = np.sqrt(((pred_A_back - pred_orig) ** 2).mean(axis=(1,2,3))) \
            / (np.sqrt((pred_orig ** 2).mean(axis=(1,2,3))) + 1e-12)

    # Scaling B: α=4, β=1
    snap_B = transform(snap, alpha=4.0, beta=1.0)
    xB = torch.tensor(snap_B).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_B = model(xB).cpu().numpy()[0]
    # Inverse: α=1/4, β=1
    pred_B_back = transform(pred_B, alpha=0.25, beta=1.0)
    dev_B = np.sqrt(((pred_B_back - pred_orig) ** 2).mean(axis=(1,2,3))) \
            / (np.sqrt((pred_orig ** 2).mean(axis=(1,2,3))) + 1e-12)

    np.savez_compressed(out / "scaling.npz",
                       dev_A=dev_A, dev_B=dev_B,
                       pred_orig=pred_orig, pred_A_back=pred_A_back, pred_B_back=pred_B_back)
    print(f"wrote {out}/scaling.npz")
    print(f"  dev_A (β=2) per-channel: {dev_A}")
    print(f"  dev_B (α=4) per-channel: {dev_B}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--test_dir", default="data/MHD_64/data/test")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
