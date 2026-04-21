"""Deep-dive plots that squeeze the existing data.

Each function produces one figure and tells one story. Runs from what's already
in p1/evals/physics/ and p1/evals2/ — no GPU needed.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

MODELS_FOUR = ["baseline", "baseline_01", "ft_01", "ft_100"]
ALL_CONFIGS = ["baseline","baseline_10","baseline_10_s1","baseline_10_s2",
               "baseline_01","baseline_01_s1","baseline_01_s2",
               "ft_100","ft_10","ft_10_s1","ft_10_s2",
               "ft_01","ft_01_s1","ft_01_s2","pretrain"]
COLORS = {"baseline":"#888888","baseline_01":"#d62728","ft_01":"#1f77b4","ft_100":"#2ca02c"}
CHANNELS = ["density","B_x","B_y","B_z","v_x","v_y","v_z"]


# --- helper -----------------------------------------------------------------
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


# --- (1) pretrain persistence map -------------------------------------------
def plot_pretrain_persistence(physics_dir, out_path):
    """6-panel: drift of field-level stats over rollout, with theory refs."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    panels = [
        ("mean |B_x|",      "mean_Bx",   None),
        (r"$E_B/E_K$",      "E_ratio",   [(1/0.7**2, "M_A=0.7 target (theory)", "#1f77b4"),
                                           (1/2.0**2, "M_A=2.0 pretrain",        "#888")]),
        ("density std",     "std_rho",   [(0.16, "M_A=0.7 obs", "#1f77b4")]),
        (r"$|B|^2$ volume avg", "E_B",   None),
        ("v_x std",         "std_vx",    [(0.4, "M_A=0.7 obs", "#1f77b4")]),
        (r"$\nabla\cdot B / |B|$", "divB_norm", None),
    ]
    for ax, (ylab, key, refs) in zip(axes.flat, panels):
        for m in MODELS_FOUR:
            cp = physics_dir/m/"conservation.npz"
            vp = physics_dir/m/"variance.npz"
            fp = physics_dir/m/"field_snapshots.npz"
            if key in ("E_B","E_K","E_ratio","mass","divB_norm"):
                if not cp.exists(): continue
                d = np.load(cp)
                v = d[f"pred_{key}"]  # (n_traj, K+1)
            elif key == "std_rho":
                if not vp.exists(): continue
                d = np.load(vp)
                v = np.sqrt(d["variance_per_step"][:,:,0])
            elif key == "std_vx":
                if not vp.exists(): continue
                d = np.load(vp)
                v = np.sqrt(d["variance_per_step"][:,:,4])
            elif key == "mean_Bx":
                # No per-step mean stored globally — approximate from snapshots
                if not fp.exists(): continue
                d = np.load(fp)
                steps = d["steps"]
                traj = np.array([d[f"pred_step{int(s)}"][1].mean() for s in steps])
                ax.plot(steps.astype(int), traj, color=COLORS[m], marker="o", lw=1.8, label=m)
                continue
            mu = v.mean(0); sd = v.std(0)
            steps = np.arange(len(mu))
            ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
            ax.fill_between(steps, mu-sd, mu+sd, color=COLORS[m], alpha=0.12)
        # truth
        if key in ("E_B","E_K","E_ratio","mass","divB_norm"):
            fb = physics_dir/"baseline"/"conservation.npz"
            if fb.exists():
                d = np.load(fb); v = d[f"truth_{key}"]
                ax.plot(np.arange(v.shape[1]), v.mean(0), color="black", ls=":", lw=1.5, label="truth")
        if refs:
            for val, lbl, c in refs:
                ax.axhline(val, ls="--", color=c, alpha=0.5, label=lbl)
        ax.set_title(ylab); ax.set_xlabel("step"); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Pretrain-persistence map — which field-level statistics drift toward the pretrain regime?")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (2) cascade cutoff vs data fraction ------------------------------------
def compute_cutoff(Ek_truth, Ek_pred, threshold=0.1):
    """Smallest k above which |pred|/truth < threshold."""
    ratio = Ek_pred / (Ek_truth + 1e-20)
    mask = (Ek_truth > 0) & (ratio < threshold)
    if not mask.any():
        return None
    return int(np.argmax(mask))


def plot_cascade_cutoff(physics_dir, out_path):
    """k_perp cutoff per config vs data fraction. Uses aniso_step1 histograms."""
    pts = []
    for cfg in ALL_CONFIGS:
        ap = physics_dir/cfg/"aniso_step1.npz"
        if not ap.exists(): continue
        d = np.load(ap)
        Hp = d["pred"]; Ht = d["truth"]; edges = d["edges"]
        centers = 0.5*(edges[:-1]+edges[1:])
        # perpendicular cascade: integrate over low k_par bins
        i_lowpar = slice(0, max(2, len(centers)//8))
        Ek_p = Hp[i_lowpar, :].mean(axis=0)
        Ek_t = Ht[i_lowpar, :].mean(axis=0)
        cutoff = compute_cutoff(Ek_t, Ek_p, 0.1)
        # classify
        if cfg == "pretrain": frac, kind = None, "pretrain"
        elif "_01" in cfg: frac = 0.01
        elif "_10" in cfg: frac = 0.10
        else: frac = 1.0
        kind = "ft" if cfg.startswith("ft") else ("baseline" if cfg.startswith("baseline") else "pretrain")
        if cutoff is None: continue
        k_val = centers[cutoff]
        pts.append((cfg, frac, kind, k_val))

    fig, ax = plt.subplots(figsize=(8, 5))
    # group by kind + data frac, show error bars over seeds
    groups = {}
    for cfg, frac, kind, k in pts:
        if frac is None: continue
        groups.setdefault((kind, frac), []).append(k)
    for (kind, frac), ks in groups.items():
        c = "#d62728" if kind == "baseline" else "#1f77b4"
        marker = "o" if kind == "baseline" else "s"
        label = f"{kind} (n={len(ks)})" if len(ks) > 1 else None
        ax.errorbar(frac, np.mean(ks), yerr=np.std(ks), marker=marker, ms=8,
                    color=c, capsize=5, label=label if label not in [l.get_label() for l in ax.get_lines()] else None)
    # custom legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="#d62728", ms=8, lw=0, label="from scratch"),
        Line2D([0],[0], marker="s", color="#1f77b4", ms=8, lw=0, label="pretrained + FT")])
    ax.set_xscale("log")
    ax.set_xlabel("fraction of M_A=0.7 training data")
    ax.set_ylabel(r"cascade cutoff wavenumber k$_\perp$")
    ax.set_title("Cascade cutoff — smallest k_⊥ where pred/truth drops below 0.1, lower = worse")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (3) per-channel spectral error heatmap ---------------------------------
def plot_spectral_channel_heatmap(physics_dir, out_path):
    """For each (model, channel, ref_step), compute relative isotropic spectral error."""
    ref_steps = [1, 5, 10, 25, 50]
    # collect: (model, step, channel) -> scalar
    data = np.full((len(MODELS_FOUR), len(ref_steps), len(CHANNELS)), np.nan)
    for mi, m in enumerate(MODELS_FOUR):
        fp = physics_dir/m/"field_snapshots.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        steps = d["steps"]
        for si, s in enumerate(ref_steps):
            if s not in steps: continue
            pred = d[f"pred_step{s}"]; truth = d[f"truth_step{s}"]
            for ci in range(7):
                _, Ep = iso_spec(pred[ci])
                _, Et = iso_spec(truth[ci])
                rel_err = np.mean(np.abs(Ep - Et) / (Et + 1e-20))
                # clip to avoid extreme values from near-zero denominators
                data[mi, si, ci] = min(rel_err, 5.0)
    fig, axes = plt.subplots(1, len(ref_steps), figsize=(18, 4.5), sharey=True)
    for si, s in enumerate(ref_steps):
        ax = axes[si]
        im = ax.imshow(data[:, si, :], aspect="auto", cmap="viridis",
                       vmin=0, vmax=2.0, origin="upper")
        ax.set_xticks(range(len(CHANNELS))); ax.set_xticklabels(CHANNELS, rotation=45)
        if si == 0:
            ax.set_yticks(range(len(MODELS_FOUR))); ax.set_yticklabels(MODELS_FOUR)
        ax.set_title(f"step {s}")
    fig.colorbar(im, ax=axes, label="rel spectral err  (clipped at 2.0)")
    fig.suptitle("Per-channel isotropic spectral error — which fields fail first for each model?")
    fig.savefig(out_path, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out_path}")


# --- (4) short vs long horizon scatter --------------------------------------
def plot_short_long_scatter(evals2_dir, out_path):
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

    fig, ax = plt.subplots(figsize=(8, 6))
    for cfg, kind, frac, s1, s50 in pts:
        c = "#d62728" if kind == "baseline" else ("#1f77b4" if kind == "ft" else "#2ca02c")
        marker_size = 150 if frac == 0.01 else (90 if frac == 0.10 else 50)
        ax.scatter(s1, s50, s=marker_size, c=c, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.annotate(cfg, (s1, s50), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("step-1 VRMSE on M_A=0.7 test"); ax.set_ylabel("step-50 VRMSE on M_A=0.7 test")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.set_title("Short vs long horizon — does winning step-1 predict winning step-50?")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d62728", ms=10, lw=0, label="from scratch"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1f77b4", ms=10, lw=0, label="pretrain + FT"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#2ca02c", ms=10, lw=0, label="pretrain zero-shot"),
    ], loc="lower right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (5) wave-mode energy persistence ---------------------------------------
def plot_wave_mode_persistence(physics_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for wi, wave in enumerate(["alfven", "magsonic"]):
        ax = axes[wi]
        for m in MODELS_FOUR:
            fp = physics_dir/"wave_probes"/m/f"{wave}.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            traj = d["traj"]   # (T, 7, N, N, N)
            k_mode = int(d["k_mode"])
            signal = (traj[:, 5 if wave=="alfven" else 0]
                      .mean(axis=(-1,-2)))  # (T, N)
            if wave == "magsonic":
                signal = signal - signal.mean(axis=-1, keepdims=True)
            S = np.abs(np.fft.rfft(signal, axis=-1)) ** 2  # (T, N//2+1)
            total = S.sum(axis=-1) + 1e-12
            frac_in_mode = S[:, k_mode] / total
            ax.plot(np.arange(S.shape[0]), frac_in_mode, color=COLORS[m], lw=1.7, label=m)
        ax.set_xlabel("step"); ax.set_ylabel(f"fraction of energy at k={k_mode}")
        ax.set_title(f"{wave.title()} wave — mode purity over time")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.suptitle("Wave-mode energy persistence — clean propagation → stays near 1.0")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (6) spatial error structure --------------------------------------------
def plot_spatial_error_maps(physics_dir, out_path, step=25):
    fig, axes = plt.subplots(len(MODELS_FOUR), 3, figsize=(12, 4*len(MODELS_FOUR)))
    for mi, m in enumerate(MODELS_FOUR):
        fp = physics_dir/m/"field_snapshots.npz"
        if not fp.exists():
            for col in range(3): axes[mi, col].axis("off"); continue
        d = np.load(fp)
        if f"pred_step{step}" not in d:
            for col in range(3): axes[mi, col].axis("off"); continue
        pred = d[f"pred_step{step}"]; truth = d[f"truth_step{step}"]
        z = pred.shape[-1]//2
        # density error
        err_dens = (pred[0] - truth[0])[:,:,z]
        # |B| error
        Bp = np.sqrt((pred[1:4]**2).sum(0)); Bt = np.sqrt((truth[1:4]**2).sum(0))
        err_B = (Bp - Bt)[:,:,z]
        # |v| error
        vp = np.sqrt((pred[4:7]**2).sum(0)); vt = np.sqrt((truth[4:7]**2).sum(0))
        err_v = (vp - vt)[:,:,z]
        for ci, (err, title) in enumerate([(err_dens,"Δρ"),(err_B,"Δ|B|"),(err_v,"Δ|v|")]):
            vlim = max(abs(err.min()), abs(err.max()))
            im = axes[mi, ci].imshow(err, cmap="RdBu_r", vmin=-vlim, vmax=vlim)
            axes[mi, ci].set_title(f"{m}  {title}  (max={vlim:.3f})", fontsize=9)
            axes[mi, ci].set_xticks([]); axes[mi, ci].set_yticks([])
    fig.suptitle(f"Spatial error maps at rollout step {step}  —  z-midplane slices")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (7) divergence-onset distribution --------------------------------------
def plot_divergence_onset(physics_dir, out_path, threshold=1.5):
    fig, ax = plt.subplots(figsize=(8, 5))
    for m in MODELS_FOUR:
        fp = physics_dir/m/"rollout_vrmse_full.npz"
        if not fp.exists(): continue
        v = np.load(fp)["vrmse"]   # (n_traj, K)
        # first step per trajectory where VRMSE > threshold
        onsets = []
        for t_i in range(v.shape[0]):
            exceed = np.where(v[t_i] > threshold)[0]
            onsets.append(exceed[0] + 1 if len(exceed) else v.shape[1] + 1)
        onsets = np.array(onsets)
        ax.hist(onsets, bins=np.arange(0, 52, 2), alpha=0.5, label=f"{m} (med={np.median(onsets):.1f})",
                color=COLORS[m], edgecolor="black", linewidth=0.3)
    ax.set_xlabel(f"step at which VRMSE > {threshold}")
    ax.set_ylabel("trajectories")
    ax.set_title(f"Divergence-onset distribution across test trajectories (threshold VRMSE={threshold})")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- (8) conservation-violation rate (d/dt of divB, E_B) --------------------
def plot_conservation_rates(physics_dir, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [("E_B", r"$dE_B/dt$  (relative)"),
              ("divB_norm", r"$d\|\nabla\cdot B\|/dt$")]
    for ax, (key, ylab) in zip(axes, panels):
        for m in MODELS_FOUR:
            fp = physics_dir/m/"conservation.npz"
            if not fp.exists(): continue
            d = np.load(fp)
            v = d[f"pred_{key}"]
            if key == "E_B":
                # relative rate: d(E_B)/dt / E_B[0]
                dv = np.diff(v, axis=1) / (v[:, 0:1] + 1e-12)
            else:
                dv = np.diff(v, axis=1)
            mu = dv.mean(0); sd = dv.std(0)
            steps = 0.5 + np.arange(len(mu))
            ax.plot(steps, mu, color=COLORS[m], lw=1.8, label=m)
            ax.fill_between(steps, mu - sd, mu + sd, color=COLORS[m], alpha=0.15)
        ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.grid(True, alpha=0.3)
        ax.axhline(0, color="black", ls=":", lw=0.8)
        ax.set_title(key)
        ax.legend(fontsize=8)
    fig.suptitle("Conservation-violation rates over rollout  (truth ideally = 0)")
    fig.tight_layout(); fig.savefig(out_path, dpi=140); plt.close(fig)
    print(f"wrote {out_path}")


# --- main -------------------------------------------------------------------
def run(args):
    phys = Path(args.physics_dir); evals2 = Path(args.evals2_dir)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    plot_pretrain_persistence(phys, out/"pretrain_persistence_map.png")
    plot_cascade_cutoff(phys, out/"cascade_cutoff_vs_data.png")
    plot_spectral_channel_heatmap(phys, out/"spectral_channel_heatmap.png")
    plot_short_long_scatter(evals2, out/"short_vs_long_horizon.png")
    plot_wave_mode_persistence(phys, out/"wave_mode_persistence.png")
    plot_spatial_error_maps(phys, out/"spatial_error_maps_step25.png", step=25)
    plot_divergence_onset(phys, out/"divergence_onset.png", threshold=1.5)
    plot_conservation_rates(phys, out/"conservation_rates.png")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--physics_dir", default="p1/evals/physics")
    p.add_argument("--evals2_dir", default="p1/evals2")
    p.add_argument("--out_dir", default="p1/figures/deep")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
