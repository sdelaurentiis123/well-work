"""Paper-quality figures for ML4PS 2026 workshop submission.

Generates 6 figures as both PDF (vector, for LaTeX) and PNG (for README),
with consistent styling. Reads from committed artifacts in p1/ subdirs —
no compute required.

Figures:
  fig1_headline_physics_specificity — 3-bar (scratch / NS→ft / MHD→ft)
  fig2_data_efficiency              — VRMSE vs data fraction
  fig3_cascade                      — perpendicular cascade preservation
  fig4_long_horizon_failure         — VRMSE vs rollout step
  fig5_equipartition_drift          — E_B/E_K evolution during rollout
  fig6_conservation_violation       — ∇·B and E_B drift per model
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# --- unified styling --------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

C_SCRATCH  = "#c0392b"     # red — from-scratch baseline
C_NS       = "#e67e22"     # orange — non-MHD pretrain
C_MHD_FT   = "#2c5aa0"     # deep blue — MHD pretrain + FT
C_TRUTH    = "#000000"
C_BASELINE = "#7f8c8d"     # gray — full-data scratch reference

LINEWIDTH = 1.6
BAR_EDGECOLOR = "black"
BAR_LINEWIDTH = 0.4

ROOT = Path(__file__).resolve().parent.parent            # p1/
OUT  = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def save(fig, name, width_in=3.5, height_in=2.6):
    fig.set_size_inches(width_in, height_in)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300)
    plt.close(fig)
    print(f"  wrote {name}.pdf and {name}.png")


# --- fig1 headline: physics specificity ------------------------------------
def fig1_headline():
    hp = json.loads((ROOT / "hp_summary.json").read_text())
    import math as _m
    # NS seeds
    ns = []
    for p in (ROOT / "runs").iterdir():
        if p.is_dir() and p.name.startswith("ns_ft_01"):
            lines = [json.loads(l) for l in (p/"log.jsonl").read_text().splitlines() if l.strip()]
            if lines:
                ns.append(min(r["val_vrmse"] for r in lines))
    bars = [
        ("from scratch\n(HP-tuned)",  hp["baseline_01_best"]["mean"], hp["baseline_01_best"]["std"], C_SCRATCH),
        ("NS pretrain + FT\n(non-MHD control)", float(np.mean(ns)), float(np.std(ns)), C_NS),
        ("MHD pretrain + FT\n(tuned)",          hp["ft_01_best"]["mean"],       hp["ft_01_best"]["std"],       C_MHD_FT),
    ]
    fig, ax = plt.subplots()
    xs = np.arange(len(bars))
    h = [b[1] for b in bars]; e = [b[2] for b in bars]
    ax.bar(xs, h, yerr=e, capsize=3, color=[b[3] for b in bars],
           edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH, width=0.6)
    for i, (hh, ee) in enumerate(zip(h, e)):
        ax.text(i, hh + ee + 0.015, f"{hh:.3f}", ha="center", fontsize=8)
    # deltas
    b_ref = hp["baseline_01_best"]["mean"]
    ns_delta = (float(np.mean(ns)) - b_ref) / b_ref * 100
    mhd_delta = (hp["ft_01_best"]["mean"] - b_ref) / b_ref * 100
    ax.text(1, max(h)+0.05, f"{ns_delta:+.0f}%", ha="center", fontsize=8, color=C_NS,    fontweight="bold")
    ax.text(2, max(h)+0.05, f"{mhd_delta:+.0f}%", ha="center", fontsize=8, color=C_MHD_FT, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("best val VRMSE  (mean ± std, 3 seeds)")
    ax.set_ylim(0, max(h) + max(e) + 0.10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Physics-specific transfer at 1% target data", fontsize=10)
    save(fig, "fig1_headline_physics_specificity", width_in=3.6, height_in=2.8)


# --- fig2 data-efficiency --------------------------------------------------
def best_val_from(run_dir):
    logp = run_dir / "log.jsonl"
    if not logp.exists(): return None
    lines = [json.loads(l) for l in logp.read_text().splitlines() if l.strip()]
    return min(r["val_vrmse"] for r in lines) if lines else None


def fig2_data_efficiency():
    runs = ROOT / "runs"
    hp = json.loads((ROOT / "hp_summary.json").read_text())

    def seeds_for(prefix):
        rx = re.compile(rf"^{re.escape(prefix)}(_s\d+)?$")
        out = []
        for d in runs.iterdir():
            if d.is_dir() and rx.match(d.name):
                v = best_val_from(d)
                if v is not None and math.isfinite(v): out.append(v)
        return out

    scratch = {
        1.00: seeds_for("baseline"),
        0.10: seeds_for("baseline_10"),
        0.01: [hp["baseline_01_best"]["mean"]],          # HP-tuned
    }
    ft = {
        1.00: seeds_for("ft_100"),
        0.10: seeds_for("ft_10"),
        0.01: [hp["ft_01_best"]["mean"]],                # HP-tuned
    }
    scratch_std = {
        1.00: np.std(scratch[1.00]) if len(scratch[1.00])>1 else 0.0,
        0.10: np.std(scratch[0.10]) if len(scratch[0.10])>1 else 0.0,
        0.01: hp["baseline_01_best"]["std"],
    }
    ft_std = {
        1.00: np.std(ft[1.00]) if len(ft[1.00])>1 else 0.0,
        0.10: np.std(ft[0.10]) if len(ft[0.10])>1 else 0.0,
        0.01: hp["ft_01_best"]["std"],
    }

    fig, ax = plt.subplots()
    fracs = [0.01, 0.10, 1.00]
    m_s = [np.mean(scratch[f]) for f in fracs]
    m_f = [np.mean(ft[f]) for f in fracs]
    s_s = [scratch_std[f] for f in fracs]
    s_f = [ft_std[f]      for f in fracs]
    ax.errorbar(fracs, m_s, yerr=s_s, marker="o", ms=6, lw=LINEWIDTH,
                color=C_SCRATCH, capsize=3, label="from scratch", zorder=3)
    ax.errorbar(fracs, m_f, yerr=s_f, marker="s", ms=6, lw=LINEWIDTH,
                color=C_MHD_FT, capsize=3, label="MHD pretrain + FT", zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("fraction of target (M_A=0.7) training data")
    ax.set_ylabel("best val VRMSE")
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    ax.set_title("Data efficiency of MHD pretraining", fontsize=10)
    save(fig, "fig2_data_efficiency", width_in=3.6, height_in=2.6)


# --- fig3 cascade preservation ---------------------------------------------
def fig3_cascade():
    # Build the perpendicular cascade figure from the stored aniso_step1 npz.
    phys = ROOT / "evals" / "physics"
    fig, ax = plt.subplots()
    configs = [("baseline_01", "from scratch\n(lr=1e-3 un-tuned)",  C_SCRATCH, ":"),
               ("baseline",    "from scratch (100% data)",           C_BASELINE, "-"),
               ("ft_01",       "MHD pretrain + FT",                  C_MHD_FT,  "-")]
    truth_E = None
    for name, lbl, color, ls in configs:
        fp = phys / name / "aniso_step1.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        Hp = d["pred"]; Ht = d["truth"]; edges = d["edges"]
        centers = 0.5*(edges[:-1]+edges[1:])
        i_lowpar = slice(0, max(2, len(centers)//8))
        Ek_p = Hp[i_lowpar, :].mean(axis=0)
        Ek_t = Ht[i_lowpar, :].mean(axis=0)
        if truth_E is None: truth_E = (centers, Ek_t)
        mask = Ek_p > 0
        ax.loglog(centers[mask], Ek_p[mask], ls, lw=LINEWIDTH, color=color, label=lbl)
    if truth_E:
        ax.loglog(truth_E[0], truth_E[1], "k--", lw=1.2, alpha=0.6, label="ground truth")
    # GS95 slope reference
    k_ref = centers[(centers >= 2) & (centers <= 10)]
    if len(k_ref):
        ref = 2e-4 * k_ref ** (-5/3)
        ax.loglog(k_ref, ref, color="gray", ls=(0,(1,1)), lw=0.8, alpha=0.6,
                  label=r"$k_{\perp}^{-5/3}$ (GS95)")
    ax.set_xlabel(r"k$_{\perp}$")
    ax.set_ylabel(r"E(k$_{\perp}$ | low k$_{\parallel}$)")
    ax.set_title("Perpendicular cascade preservation", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, "fig3_cascade", width_in=3.6, height_in=2.8)


# --- fig4 long-horizon failure ---------------------------------------------
def fig4_long_horizon():
    e2 = ROOT / "evals2"
    fig, ax = plt.subplots()
    models = [("baseline",    "baseline (100% data, scratch)", C_BASELINE, "-"),
              ("baseline_01", "scratch (1% data)",              C_SCRATCH,  "-"),
              ("ft_01",       "MHD pretrain + FT (1% data)",    C_MHD_FT,  "-")]
    for name, lbl, color, ls in models:
        rj = e2 / name / "results.json"
        if not rj.exists(): continue
        r = json.loads(rj.read_text())
        mu = np.array(r.get("rollout_vrmse_mean_per_step", []))
        sd = np.array(r.get("rollout_vrmse_std_per_step", []))
        if len(mu) == 0: continue
        steps = 1 + np.arange(len(mu))
        ax.plot(steps, mu, ls, lw=LINEWIDTH, color=color, label=lbl)
        ax.fill_between(steps, mu-sd, mu+sd, color=color, alpha=0.15)
    ax.set_xlabel("autoregressive rollout step")
    ax.set_ylabel("VRMSE vs ground truth")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Long-horizon rollout — pretraining flips failure mode", fontsize=10)
    save(fig, "fig4_long_horizon_failure", width_in=3.6, height_in=2.6)


# --- fig5 equipartition drift ---------------------------------------------
def fig5_equipartition():
    phys = ROOT / "evals" / "physics"
    fig, ax = plt.subplots()
    configs = [("baseline_01", "scratch (1% data)", C_SCRATCH,  "-"),
               ("ft_01",       "MHD pretrain + FT", C_MHD_FT,  "-"),
               ("baseline",    "scratch (100% data)", C_BASELINE, "-")]
    truth_curve = None
    for name, lbl, color, ls in configs:
        fp = phys / name / "conservation.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        v = d["pred_E_ratio"]
        mu = v.mean(axis=0); sd = v.std(axis=0)
        steps = np.arange(mu.shape[0])
        ax.plot(steps, mu, ls, lw=LINEWIDTH, color=color, label=lbl)
        ax.fill_between(steps, mu-sd, mu+sd, color=color, alpha=0.15)
        if truth_curve is None:
            t = d["truth_E_ratio"]
            truth_curve = (np.arange(t.shape[1]), t.mean(axis=0))
    if truth_curve:
        ax.plot(truth_curve[0], truth_curve[1], "k--", lw=1.2, alpha=0.6, label="ground truth")
    # theory refs
    ax.axhline(1/0.7**2, ls=":", color=C_MHD_FT, alpha=0.5, lw=0.8)
    ax.axhline(1/2.0**2, ls=":", color="gray",   alpha=0.5, lw=0.8)
    ax.text(45, 1/0.7**2 - 0.18, "target regime (M_A=0.7)", fontsize=7, color=C_MHD_FT)
    ax.text(45, 1/2.0**2 + 0.05, "pretrain regime (M_A=2.0)", fontsize=7, color="gray")
    ax.set_xlabel("rollout step"); ax.set_ylabel(r"$E_B / E_K$")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title("Pretrain bias persists through fine-tuning", fontsize=10)
    save(fig, "fig5_equipartition_drift", width_in=3.6, height_in=2.6)


# --- fig6 conservation violation -------------------------------------------
def fig6_conservation():
    phys = ROOT / "evals" / "physics"
    fig, (ax_eb, ax_div) = plt.subplots(1, 2)
    configs = [("baseline_01", "scratch (1% data)",  C_SCRATCH, "-"),
               ("ft_01",       "MHD pretrain + FT",  C_MHD_FT, "-"),
               ("baseline",    "scratch (100% data)",C_BASELINE,"-")]
    for name, lbl, color, ls in configs:
        fp = phys / name / "conservation.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        # relative E_B drift
        v = d["pred_E_B"]; ref = v[:, 0:1]
        drift = (v - ref) / (np.abs(ref) + 1e-12)
        mu = drift.mean(axis=0); sd = drift.std(axis=0)
        steps = np.arange(mu.shape[0])
        ax_eb.plot(steps, mu, ls, lw=LINEWIDTH, color=color, label=lbl)
        ax_eb.fill_between(steps, mu-sd, mu+sd, color=color, alpha=0.15)
        # divB_norm absolute
        v2 = d["pred_divB_norm"]
        mu2 = v2.mean(axis=0); sd2 = v2.std(axis=0)
        ax_div.plot(steps, mu2, ls, lw=LINEWIDTH, color=color, label=lbl)
        ax_div.fill_between(steps, mu2-sd2, mu2+sd2, color=color, alpha=0.15)
    # truth overlay for divB
    fp = phys / "baseline_01" / "conservation.npz"
    if fp.exists():
        d = np.load(fp); t = d["truth_divB_norm"]
        ax_div.plot(np.arange(t.shape[1]), t.mean(axis=0), "k--", lw=1.0, alpha=0.5, label="truth")
    ax_eb.set_xlabel("rollout step"); ax_eb.set_ylabel(r"$(E_B - E_{B,0})/E_{B,0}$")
    ax_eb.set_title("(a) magnetic energy drift", fontsize=9)
    ax_div.set_xlabel("rollout step"); ax_div.set_ylabel(r"$\|\nabla\cdot B\| / \langle|B|\rangle$")
    ax_div.set_title("(b) monopole production", fontsize=9)
    ax_div.set_yscale("log")
    ax_eb.legend(fontsize=7, loc="upper left")
    save(fig, "fig6_conservation_violation", width_in=7.0, height_in=2.6)


# --- main -------------------------------------------------------------------
def main():
    print("Generating paper-quality figures...")
    fig1_headline()
    fig2_data_efficiency()
    fig3_cascade()
    fig4_long_horizon()
    fig5_equipartition()
    fig6_conservation()
    print(f"\nAll figures in {OUT}/")


if __name__ == "__main__":
    main()
