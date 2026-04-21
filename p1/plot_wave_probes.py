"""Physics probe 3+4 figures — Alfvén + magnetosonic wave propagation.

Reads evals/physics/wave_probes/<model>/{alfven,magsonic}.npz.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS_FOUR = ["baseline", "baseline_01", "ft_01", "ft_100"]
COLORS = {"baseline": "#888888", "baseline_01": "#d62728",
          "ft_01": "#1f77b4", "ft_100": "#2ca02c"}


def extract_wave_stats(traj: np.ndarray, kind: str):
    """From (T, 7, 64, 64, 64) rollout, extract (position_peak, amplitude, k-spectrum) per step.

    For Alfvén: track v_y(x) along x-axis averaged over (y, z).
    For magnetosonic: track δρ(x).
    """
    if kind == "alfven":
        signal = traj[:, 5].mean(axis=(-1, -2))   # v_y along x, avg over y,z  -> (T, Nx)
    elif kind == "magsonic":
        signal = traj[:, 0].mean(axis=(-1, -2)) - traj[:, 0].mean(axis=(-1, -2)).mean(axis=-1, keepdims=True)
    else:
        raise ValueError(kind)
    T, Nx = signal.shape
    # peak position: argmax
    peak_x = np.argmax(signal, axis=-1)
    # amplitude: peak - mean
    amp = signal.max(axis=-1) - signal.mean(axis=-1)
    # FFT: power in each k
    S = np.abs(np.fft.rfft(signal, axis=-1)) ** 2
    return signal, peak_x, amp, S


def run(args):
    root = Path(args.wave_dir)
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))

    for wave_i, wave in enumerate(["alfven", "magsonic"]):
        # --- panel 1: position vs time, with theoretical line ---
        ax_pos = axes[wave_i, 0]
        ax_amp = axes[wave_i, 1]
        ax_shape = axes[wave_i, 2]
        ax_spec = axes[wave_i, 3]

        theory_speed = None
        for m in MODELS_FOUR:
            fp = root / m / f"{wave}.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            traj = d["traj"]
            theory_speed = float(d["v_A_theory"] if wave == "alfven" else d["v_ms_theory"])
            k_mode = int(d["k_mode"])
            _, peak_x, amp, S = extract_wave_stats(traj, wave)
            t = np.arange(traj.shape[0])
            # fit position line
            slope, _ = np.polyfit(t, peak_x.astype(float) / traj.shape[-1], 1)
            ax_pos.plot(t, peak_x.astype(float) / traj.shape[-1],
                        color=COLORS[m], lw=1.6,
                        label=f"{m}: v_eff={slope:.3f}")
            ax_amp.plot(t, amp, color=COLORS[m], lw=1.6, label=m)
            # shape at t=0 and t=50 (if present)
            final_t = traj.shape[0] - 1
            sig_end = traj[final_t, 5 if wave=="alfven" else 0].mean(axis=(-1,-2))
            ax_shape.plot(sig_end, color=COLORS[m], lw=1.5, label=m)
            ax_spec.semilogy(np.arange(S.shape[1]), S[final_t], color=COLORS[m], lw=1.6, label=m)

        # theory line in position plot
        if theory_speed is not None:
            ax_pos.plot(t, (theory_speed * t / 64) % 1, "k--", lw=1, alpha=0.5,
                        label=f"theory v={theory_speed:.3f}")
        ax_pos.set_xlabel("step"); ax_pos.set_ylabel("peak position / L")
        ax_pos.set_title(f"{wave.title()}: peak position"); ax_pos.grid(True, alpha=0.3)
        ax_pos.legend(fontsize=7)

        ax_amp.set_xlabel("step"); ax_amp.set_ylabel("amplitude")
        ax_amp.set_title(f"{wave.title()}: amplitude"); ax_amp.grid(True, alpha=0.3)
        ax_amp.legend(fontsize=8)

        sig_init = traj[0, 5 if wave=="alfven" else 0].mean(axis=(-1,-2))
        ax_shape.plot(sig_init, "k:", lw=1.5, label="initial")
        ax_shape.set_xlabel("x index"); ax_shape.set_ylabel("field value")
        ax_shape.set_title(f"{wave.title()}: shape at final step")
        ax_shape.grid(True, alpha=0.3); ax_shape.legend(fontsize=8)

        ax_spec.set_xlabel("k"); ax_spec.set_ylabel("|S(k)|²")
        ax_spec.set_title(f"{wave.title()}: spectrum at final step")
        ax_spec.axvline(k_mode, color="k", ls=":", alpha=0.5, label=f"initial k={k_mode}")
        ax_spec.grid(True, alpha=0.3, which="both"); ax_spec.legend(fontsize=8)

    fig.suptitle("P1 synthetic wave probes — do models propagate Alfvén + magnetosonic waves correctly?",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140); plt.close(fig)
    print(f"wrote {args.out}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--wave_dir", default="p1/evals/physics/wave_probes")
    p.add_argument("--out", default="p1/figures/physics/wave_probes.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
