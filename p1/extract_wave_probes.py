"""Physics probe 3+4: synthetic Alfvén + magnetosonic wave propagation.

Constructs small-amplitude linear waves on a uniform background, rolls each model
forward, and saves the predicted state at every step to a .npz.

Alfvén:  ρ=1, B=(1,0,0), v_y(x) = A sin(2π k x / L).  Theoretical v_A = 1/√(4πρ) ≈ 0.282.
Magnetosonic:  ρ(x) = 1 + A sin(2π k x / L), v_x paired via ideal-gas linear mode.
                Theoretical v_ms = √(c_s² + v_A²).  We assume c_s = 0.4 (Ms=1-ish regime).

Outputs one .npz per (model, wave_type) with field trajectories.

Usage:
  python p1/extract_wave_probes.py --ckpt runs/ft_01/best.pt --out evals/physics/wave_probes/ft_01
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from the_well.benchmark.models import FNO

N = 64                  # grid
L = 1.0                 # box size (dimensionless)
A = 0.01                # perturbation amplitude (linear regime)
K_MODE = 4              # wavenumber (mode number)
STEPS = 50              # rollout length
RHO_0 = 1.0             # background density
B0 = 1.0                # background B_x
CS = 0.4                # sound speed (assumed — document in writeup)
V_A_TH = B0 / np.sqrt(4 * np.pi * RHO_0)        # theoretical Alfvén speed
V_MS_TH = np.sqrt(CS ** 2 + V_A_TH ** 2)         # theoretical magnetosonic speed


def build_alfven_ic() -> np.ndarray:
    """7-channel uniform background + transverse v_y perturbation. Shape (7,64,64,64)."""
    out = np.zeros((7, N, N, N), dtype=np.float32)
    x = np.arange(N) / N * L
    pert = A * np.sin(2 * np.pi * K_MODE * x / L)      # (N,)
    out[0] = RHO_0                                     # density
    out[1] = B0                                        # B_x uniform
    # B_y, B_z = 0; v_x, v_z = 0
    # v_y depends on x only; broadcast to the 3D grid
    out[5] = pert[:, None, None] * np.ones((1, N, N), dtype=np.float32)   # v_y(x)
    return out


def build_magsonic_ic() -> np.ndarray:
    """Isothermal magnetosonic fast-mode IC: coupled δρ and δv_x along x.

    Linear dispersion for fast mode propagating along B_0: ω² = k² v_ms².
    For a plane wave δρ = A cos(kx - ωt):
        δv_x = (ω/k) · δρ / ρ_0 = v_ms · δρ / ρ_0.
    """
    out = np.zeros((7, N, N, N), dtype=np.float32)
    x = np.arange(N) / N * L
    pert = A * np.sin(2 * np.pi * K_MODE * x / L)      # (N,)
    out[0] = RHO_0 + pert[:, None, None]               # density
    out[1] = B0                                        # B_x uniform
    out[4] = (V_MS_TH * pert / RHO_0)[:, None, None]   # v_x coupled (linear mode)
    return out


def make_model(device):
    m = FNO(dim_in=7, dim_out=7, n_spatial_dims=3,
            spatial_resolution=(N, N, N),
            modes1=12, modes2=12, modes3=12,
            hidden_channels=48).to(device)
    return m


def rollout(model, ic, device, steps):
    """Return (steps+1, 7, N, N, N) array including initial condition."""
    traj = np.zeros((steps + 1,) + ic.shape, dtype=np.float32)
    traj[0] = ic
    state = torch.tensor(ic).unsqueeze(0).to(device)
    with torch.no_grad():
        for t in range(steps):
            pred = model(state)
            traj[t + 1] = pred.cpu().numpy()[0]
            state = pred
    return traj


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available()
                         else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"device={device}")

    model = make_model(device)
    sd = torch.load(args.ckpt, map_location=device, weights_only=True)
    sd = sd["model"] if "model" in sd else sd
    model.load_state_dict(sd, strict=True); model.eval()
    print(f"loaded {args.ckpt}")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    alfven_ic = build_alfven_ic()
    magsonic_ic = build_magsonic_ic()

    alfven_traj = rollout(model, alfven_ic, device, args.steps)
    magsonic_traj = rollout(model, magsonic_ic, device, args.steps)

    np.savez_compressed(out / "alfven.npz",
                       traj=alfven_traj,
                       ic=alfven_ic,
                       v_A_theory=V_A_TH,
                       k_mode=K_MODE, amplitude=A, steps=args.steps)
    np.savez_compressed(out / "magsonic.npz",
                       traj=magsonic_traj,
                       ic=magsonic_ic,
                       v_ms_theory=V_MS_TH,
                       c_s=CS, v_A=V_A_TH,
                       k_mode=K_MODE, amplitude=A, steps=args.steps)
    print(f"wrote {out}/alfven.npz and {out}/magsonic.npz")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=STEPS)
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
