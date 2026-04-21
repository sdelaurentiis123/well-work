"""Second round — fixes + richer plots."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

MODELS_FOUR = ["baseline", "baseline_01", "ft_01", "ft_100"]
ALL_CONFIGS = ["baseline","baseline_10","baseline_10_s1","baseline_10_s2",
               "baseline_01","baseline_01_s1","baseline_01_s2",
               "ft_100","ft_10","ft_10_s1","ft_10_s2",
               "ft_01","ft_01_s1","ft_01_s2","pretrain"]
COLORS = {"baseline":"#888888","baseline_01":"#d62728","ft_01":"#1f77b4","ft_100":"#2ca02c"}
CHANNELS = ["density","B_x","B_y","B_z","v_x","v_y","v_z"]


def iso_spec(field):
    N = field.shape[0]
    F = np.fft.fftn(field); P = (F*F.conj()).real / N**6
    kx = np.fft.fftfreq(N)*N
    KX,KY,KZ = np.meshgrid(kx,kx,kx,indexing='ij')
    K = np.sqrt(KX**2+KY**2+KZ**2)
    nbins=32
    edges = np.linspace(0,N/2,nbins+1)
    c = 0.5*(edges[:-1]+edges[1:])
    E = np.zeros(nbins)
    for i in range(nbins):
        m = (K>=edges[i]) & (K<edges[i+1])
        if m.any(): E[i] = P[m].sum()
    return c, E


# --- fixed spectral heatmap (absolute, normalized by truth total power) -----
def plot_spectral_channel_heatmap_v2(physics_dir, out_path):
    ref_steps = [1, 5, 10, 25, 50]
    data = np.full((len(MODELS_FOUR), len(ref_steps), len(CHANNELS)), np.nan)
    for mi, m in enumerate(MODELS_FOUR):
        fp = physics_dir/m/"field_snapshots.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        for si, s in enumerate(ref_steps):
            pk = f"pred_step{s}"; tk = f"truth_step{s}"
            if pk not in d: continue
            pred = d[pk]; truth = d[tk]
            for ci in range(7):
                _, Ep = iso_spec(pred[ci])
                _, Et = iso_spec(truth[ci])
                # absolute error divided by truth total power
                total_t = Et.sum() + 1e-20
                data[mi, si, ci] = np.sum(np.abs(Ep - Et)) / total_t
    fig, axes = plt.subplots(1, len(ref_steps), figsize=(18, 4.5), sharey=True)
    vmax = np.nanpercentile(data, 95)
    for si, s in enumerate(ref_steps):
        ax = axes[si]
        im = ax.imshow(data[:, si, :], aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax, origin="upper")
        ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS, rotation=45)
        if si == 0:
            ax.set_yticks(range(len(MODELS_FOUR))); ax.set_yticklabels(MODELS_FOUR)
        ax.set_title(f"step {s}")
        # overlay values
        for i in range(data.shape[0]):
            for j in range(data.shape[2]):
                v = data[i, si, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            color="white" if v > vmax*0.5 else "black", fontsize=7)
    fig.colorbar(im, ax=axes, label=r"$\sum_k |E_p(k)-E_t(k)| / \sum_k E_t(k)$")
    fig.suptitle("Per-channel absolute spectral error (normalized by truth total power)")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


# --- spectrum evolution per model -------------------------------------------
def plot_spectrum_evolution(physics_dir, out_path):
    ref_steps = [1, 5, 10, 25, 50]
    fig, axes = plt.subplots(len(MODELS_FOUR), 3, figsize=(15, 4*len(MODELS_FOUR)))
    for mi, m in enumerate(MODELS_FOUR):
        fp = physics_dir/m/"field_snapshots.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        # Plot 3 fields: density, B_x, v_x
        for ci_i, (ch_name, ch_idx) in enumerate([("density",0),("B_x",1),("v_x",4)]):
            ax = axes[mi, ci_i]
            # truth at step 1
            _, Et_ref = iso_spec(d["truth_step1"][ch_idx])
            centers, _ = iso_spec(d["truth_step1"][ch_idx])
            ax.loglog(centers, Et_ref, "k:", lw=2, alpha=0.7, label="truth (step 1)")
            cmap = plt.get_cmap("plasma")
            for si, s in enumerate(ref_steps):
                pk = f"pred_step{s}"
                if pk not in d: continue
                _, Ep = iso_spec(d[pk][ch_idx])
                mask = Ep > 0
                ax.loglog(centers[mask], Ep[mask], lw=1.6, color=cmap(si/len(ref_steps)),
                          label=f"step {s}")
            ax.set_title(f"{m}  {ch_name}"); ax.grid(True, which="both", alpha=0.3)
            ax.set_xlabel("k"); ax.set_ylabel("E(k)")
            if mi == 0 and ci_i == 0:
                ax.legend(fontsize=7, loc="lower left")
    fig.suptitle("E(k) evolution through rollout — does high-k inflate? low-k decay?")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- wave amplitude growth rate (abs, not ratio) ----------------------------
def plot_wave_amplitude_growth(physics_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for wi, wave in enumerate(["alfven", "magsonic"]):
        ax = axes[wi]
        for m in MODELS_FOUR:
            fp = physics_dir/"wave_probes"/m/f"{wave}.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            traj = d["traj"]
            ch = 5 if wave == "alfven" else 0
            signal = traj[:, ch].mean(axis=(-1,-2))
            if wave == "magsonic":
                signal = signal - signal.mean(axis=-1, keepdims=True)
            amp = signal.max(axis=-1) - signal.min(axis=-1)
            ax.semilogy(np.arange(amp.shape[0]), amp, color=COLORS[m], lw=1.8, label=m)
        # theoretical: amplitude should be constant in ideal linear wave
        ax.axhline(0.02, ls="--", color="black", alpha=0.5, label="initial amplitude (2·A)")
        ax.set_xlabel("step"); ax.set_ylabel(f"peak-to-peak amplitude  (log)")
        ax.set_title(f"{wave.title()} wave — amplitude over time")
        ax.grid(True, alpha=0.3, which="both"); ax.legend(fontsize=8)
    fig.suptitle("Wave amplitude — linear theory predicts flat; growth = nonlinear destabilization")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- short-long quantified with regression + spearman -----------------------
def plot_short_long_quant(evals2_dir, out_path):
    pts = []
    for cfg in ALL_CONFIGS:
        rp = evals2_dir/cfg/"results.json"
        if not rp.exists(): continue
        r = json.loads(rp.read_text())
        mu = r.get("rollout_vrmse_mean_per_step", [])
        if len(mu) < 50: continue
        if "_01" in cfg: frac = 0.01
        elif "_10" in cfg: frac = 0.10
        elif cfg == "pretrain": frac = None
        else: frac = 1.0
        kind = "ft" if cfg.startswith("ft") else ("baseline" if cfg.startswith("baseline") else "pretrain")
        pts.append((cfg, kind, frac, mu[0], mu[49]))

    fig, ax = plt.subplots(figsize=(9, 6))
    for cfg, kind, frac, s1, s50 in pts:
        c = "#d62728" if kind == "baseline" else ("#1f77b4" if kind == "ft" else "#2ca02c")
        ms = 160 if frac == 0.01 else (100 if frac == 0.10 else 60)
        ax.scatter(s1, s50, s=ms, c=c, alpha=0.75, edgecolor="black", linewidth=0.6)
        ax.annotate(cfg, (s1, s50), fontsize=7, xytext=(4, 4), textcoords="offset points")

    s1_arr = np.array([p[3] for p in pts])
    s50_arr = np.array([p[4] for p in pts])
    # Spearman across all
    rho, p = spearmanr(s1_arr, s50_arr)
    # Spearman restricted to ft only
    ft_pts = [(p[3], p[4]) for p in pts if p[1] == "ft" and p[2] is not None]
    if len(ft_pts) > 2:
        ft_s1 = np.array([p[0] for p in ft_pts])
        ft_s50 = np.array([p[1] for p in ft_pts])
        rho_ft, p_ft = spearmanr(ft_s1, ft_s50)
    else:
        rho_ft = p_ft = np.nan

    ax.text(0.02, 0.98,
            f"Spearman ρ (all 15): {rho:.2f} (p={p:.2f})\n"
            f"Spearman ρ (FT only): {rho_ft:.2f} (p={p_ft:.2f})",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_xlabel("step-1 VRMSE"); ax.set_ylabel("step-50 VRMSE  (log)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Short vs long horizon — correlation quantified across all 15 runs")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d62728", ms=10, lw=0, label="from scratch"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4", ms=10, lw=0, label="pretrain + FT"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ca02c", ms=10, lw=0, label="pretrain zero-shot"),
    ], loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- main -------------------------------------------------------------------
def run(args):
    phys = Path(args.physics_dir); evals2 = Path(args.evals2_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    plot_spectral_channel_heatmap_v2(phys, out/"spectral_channel_heatmap_v2.png")
    plot_spectrum_evolution(phys, out/"spectrum_evolution_per_model.png")
    plot_wave_amplitude_growth(phys, out/"wave_amplitude_growth.png")
    plot_short_long_quant(evals2, out/"short_vs_long_quantified.png")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--physics_dir", default="p1/evals/physics")
    p.add_argument("--evals2_dir", default="p1/evals2")
    p.add_argument("--out_dir", default="p1/figures/deep")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
