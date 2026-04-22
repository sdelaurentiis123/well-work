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

    def best_val(p):
        b = 1e9
        for l in open(p):
            r = json.loads(l); v = r.get("val_vrmse")
            if v is not None and v < b: b = v
        return b

    # Matched-architecture scratch: lr=3e-3, hidden=48 (same as pretrained FT's arch)
    matched_scratch = [0.4463268383856743, 0.4612095355987549, 0.4885952770709991]

    # 3-pretrain-seed MHD FTs: pretrain_s0 (represented by 3-FT-seed mean from hp_summary),
    # pretrain_s1 + pretrain_s2 from runs_from_seed_runner
    SEED_DIR = ROOT / "runs_from_seed_runner"
    mhd_pts = [
        hp["ft_01_best"]["mean"],
        best_val(SEED_DIR / "ft_01_from_pretrain_s1" / "log.jsonl"),
        best_val(SEED_DIR / "ft_01_from_pretrain_s2" / "log.jsonl"),
    ]
    # 3-pretrain-seed NS FTs: ns_pretrain_s0 (represented by 3-FT-seed mean from local
    # ns_ft_01_s{0,1,2} runs), ns_pretrain_s1 + s2 from runs_from_seed_runner
    ns_s0_vals = []
    for p in (ROOT / "runs").iterdir():
        if p.is_dir() and p.name.startswith("ns_ft_01_s"):
            ns_s0_vals.append(best_val(p / "log.jsonl"))
    ns_pts = [
        float(np.mean(ns_s0_vals)),
        best_val(SEED_DIR / "ns_ft_01_from_pretrain_s1" / "log.jsonl"),
        best_val(SEED_DIR / "ns_ft_01_from_pretrain_s2" / "log.jsonl"),
    ]

    bars = [
        ("from scratch\n(h=48, 3 seeds)",        float(np.mean(matched_scratch)), float(np.std(matched_scratch)), C_SCRATCH),
        ("NS pretrain + FT\n(non-MHD, 3 seeds)", float(np.mean(ns_pts)),          float(np.std(ns_pts)),           C_NS),
        ("MHD pretrain + FT\n(3 seeds)",         float(np.mean(mhd_pts)),         float(np.std(mhd_pts)),          C_MHD_FT),
    ]
    fig, ax = plt.subplots()
    xs = np.arange(len(bars))
    h = [b[1] for b in bars]; e = [b[2] for b in bars]
    ax.bar(xs, h, yerr=e, capsize=3, color=[b[3] for b in bars],
           edgecolor=BAR_EDGECOLOR, linewidth=BAR_LINEWIDTH, width=0.6)
    for i, (hh, ee) in enumerate(zip(h, e)):
        ax.text(i, hh + ee + 0.015, f"{hh:.3f}", ha="center", fontsize=8)
    # deltas vs matched-architecture scratch baseline
    b_ref = float(np.mean(matched_scratch))
    ns_delta = (float(np.mean(ns_pts)) - b_ref) / b_ref * 100
    mhd_delta = (float(np.mean(mhd_pts)) - b_ref) / b_ref * 100
    ax.text(1, max(h)+0.05, f"{ns_delta:+.0f}%", ha="center", fontsize=8, color=C_NS,    fontweight="bold")
    ax.text(2, max(h)+0.05, f"{mhd_delta:+.0f}%", ha="center", fontsize=8, color=C_MHD_FT, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars])
    ax.set_ylabel("best val VRMSE  (mean ± std, 3 seeds)")
    ax.set_ylim(0, max(h) + max(e) + 0.10)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("Matched-architecture transfer at 1% target data", fontsize=10)
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

    # Matched-architecture (h=48) scratch baseline at 1% — same 3 seeds used in Fig 1.
    # Pulled from hp_summary baseline_rows where cfg = {lr=3e-3, hidden=48, epochs=40}.
    matched_h48_seeds = None
    for cfg, seeds in hp["baseline_rows"]:
        if cfg.get("lr") == 3e-3 and cfg.get("hidden") == 48 and cfg.get("epochs") == 40:
            matched_h48_seeds = seeds
            break
    assert matched_h48_seeds is not None, "matched h=48 baseline not in hp_summary"

    scratch = {
        1.00: seeds_for("baseline"),
        0.10: seeds_for("baseline_10"),
        0.01: matched_h48_seeds,            # matched-arch (h=48), 3 seeds, consistent with Fig 1
    }
    ft = {
        1.00: seeds_for("ft_100"),
        0.10: seeds_for("ft_10"),
        0.01: [hp["ft_01_best"]["mean"]],   # FT h=48 (pretrained width), 3 seeds
    }
    scratch_std = {
        1.00: np.std(scratch[1.00]) if len(scratch[1.00])>1 else 0.0,
        0.10: np.std(scratch[0.10]) if len(scratch[0.10])>1 else 0.0,
        0.01: float(np.std(matched_h48_seeds)),
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
    # Averages E(k_perp) over the lowest-k_parallel slab on held-out M_A=0.7 test
    # trajectories' B_x field at step 1. "Ground truth" = the true simulated
    # spectrum from these test trajectories (not a fit).
    phys = ROOT / "evals" / "physics"
    fig, ax = plt.subplots()
    configs = [("baseline_01", "from scratch (1% data)",   C_SCRATCH,  "-"),
               ("baseline",    "from scratch (100% data)", C_BASELINE, "-"),
               ("ft_01",       "MHD pretrain + FT (1%)",   C_MHD_FT,   "-")]
    truth_curve = None
    centers_all = None
    for name, lbl, color, ls in configs:
        fp = phys / name / "aniso_step1.npz"
        if not fp.exists(): continue
        d = np.load(fp)
        Hp = d["pred"]; Ht = d["truth"]; edges = d["edges"]
        centers = 0.5*(edges[:-1]+edges[1:])
        centers_all = centers
        i_lowpar = slice(0, max(2, len(centers)//8))
        Ek_p = Hp[i_lowpar, :].mean(axis=0)
        Ek_t = Ht[i_lowpar, :].mean(axis=0)
        if truth_curve is None: truth_curve = (centers, Ek_t)
        mask = Ek_p > 0
        ax.loglog(centers[mask], Ek_p[mask], ls, lw=LINEWIDTH, color=color,
                  marker="o", markersize=3.5, label=lbl)

    # Ground truth first — heavy black dashed, with markers so it reads clearly.
    if truth_curve is not None:
        ck, ek = truth_curve; m = ek > 0
        ax.loglog(ck[m], ek[m], "k--", lw=1.8, marker="s", markersize=4, alpha=0.85,
                  label=r"ground truth (M$_A$=0.7 test, B$_x$)")

    # Reference slopes in the inertial range k∈[2,10], anchored to truth at k=3
    # so they sit visually among the data curves rather than off-panel.
    if centers_all is not None and truth_curve is not None:
        ck, ek = truth_curve
        i_anchor = int(np.argmin(np.abs(ck - 3.0)))
        E_anchor = ek[i_anchor]
        k_anchor = ck[i_anchor]
        k_ref = np.linspace(2.0, 10.0, 40)
        # GS95 k^{-5/3}
        ref_gs = E_anchor * (k_ref / k_anchor) ** (-5/3)
        ax.loglog(k_ref, ref_gs, color="#666666", ls=":", lw=1.5,
                  label=r"GS95 $k_{\perp}^{-5/3}$")
        # Boldyrev k^{-3/2} (prediction for sub-Alfv\'enic guide-field with dynamic alignment)
        ref_bo = E_anchor * (k_ref / k_anchor) ** (-3/2)
        ax.loglog(k_ref, ref_bo, color="#b59410", ls="-.", lw=1.5,
                  label=r"Boldyrev $k_{\perp}^{-3/2}$")

    ax.set_xlabel(r"$k_{\perp}$")
    ax.set_ylabel(r"$E(k_{\perp}\,|\,\text{low }k_{\parallel})$ — $B_x$ channel")
    ax.set_xlim(0.9, 40)
    ax.set_title("Perpendicular cascade preservation (step-1 prediction)",
                 fontsize=9.5)
    ax.legend(fontsize=6.5, loc="lower left", frameon=True, framealpha=0.9,
              ncol=1, handlelength=2.4)
    ax.grid(True, which="both", alpha=0.2, linewidth=0.3)
    save(fig, "fig3_cascade", width_in=3.6, height_in=2.8)


# --- fig4 long-horizon failure ---------------------------------------------
def fig4_long_horizon():
    # Source: evals/physics/*/rollout_vrmse_full.npz (10 held-out test trajectories,
    # same set used in Fig 7). Per-step mean±std across those 10 trajectories.
    phys = ROOT / "evals" / "physics"
    fig, ax = plt.subplots()
    models = [("baseline",      "scratch (100% data)",        C_BASELINE, "-"),
              ("baseline_01",   "scratch (1% data)",          C_SCRATCH,  "-"),
              ("ft_01",         "MHD pretrain + FT (1%)",     C_MHD_FT,   "-"),
              ("pretrain_ood",  "MHD pretrain zero-FT (OOD)", C_NS,       "--")]
    for name, lbl, color, ls in models:
        fp = phys / name / "rollout_vrmse_full.npz"
        if not fp.exists(): continue
        v = np.load(fp)["vrmse"]         # (n_traj, n_steps)
        mu = v.mean(axis=0); sd = v.std(axis=0)
        steps = 1 + np.arange(mu.shape[0])
        ax.plot(steps, mu, ls, lw=LINEWIDTH, color=color, label=lbl)
        ax.fill_between(steps, np.clip(mu-sd, 1e-3, None), mu+sd,
                        color=color, alpha=0.15)
    ax.set_xlabel("autoregressive rollout step")
    ax.set_ylabel("VRMSE vs ground truth")
    ax.set_yscale("log")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.set_title("Long-horizon rollout — fine-tuning introduces late instability",
                 fontsize=9.5)
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
    # Reference levels:
    #  - Target-regime line is EMPIRICALLY measured from truth_E_ratio on M_A=0.7
    #    test trajectories in conservation.npz (any config; they all share the
    #    same ground-truth rollouts). Mean over all steps × all 10 trajectories.
    #  - Source-regime line is the THEORETICAL 1/M_A^2 = 0.25 for M_A=2.0
    #    (M_A=2.0 test data is out of scope for this evaluation).
    any_cons = np.load(phys / "ft_01" / "conservation.npz")
    target_empirical = float(any_cons["truth_E_ratio"].mean())   # ≈ 2.13
    source_theoretical = 1.0 / 2.0**2                            # = 0.25
    ax.axhline(target_empirical, ls=":", color=C_MHD_FT, alpha=0.55, lw=1.0)
    ax.axhline(source_theoretical, ls=":", color="gray",  alpha=0.55, lw=1.0)
    ax.text(45, target_empirical - 0.18,
            f"target M$_A$=0.7 (empirical, {target_empirical:.2f})",
            fontsize=6.5, color=C_MHD_FT)
    ax.text(45, source_theoretical + 0.05,
            "source M$_A$=2.0 (theoretical 1/M$_A^2$=0.25)",
            fontsize=6.5, color="gray")
    ax.set_xlabel("rollout step"); ax.set_ylabel(r"$E_B / E_K$")
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(r"$E_B/E_K$ drift during autoregressive rollout", fontsize=10)
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

    # Annotate the step-50 floor-excess ratio used in §4.3:
    # ratio = (ft_01 divB - truth divB) / (scratch_01 divB - truth divB), final step
    try:
        dft   = np.load(phys / "ft_01"       / "conservation.npz")
        dsc01 = np.load(phys / "baseline_01" / "conservation.npz")
        truth50 = dsc01["truth_divB_norm"][:, -1].mean()
        ft_ex   = dft["pred_divB_norm"][:, -1].mean()    - truth50
        sc_ex   = dsc01["pred_divB_norm"][:, -1].mean()  - truth50
        ratio = ft_ex / sc_ex if sc_ex > 0 else float("nan")
        ax_div.text(0.97, 0.20,
                    f"step-50 floor-excess:\nFT / scratch $\\approx$ {ratio:.1f}$\\times$",
                    transform=ax_div.transAxes, ha="right", va="bottom", fontsize=7,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888", lw=0.4))
    except Exception:
        pass

    save(fig, "fig6_conservation_violation", width_in=7.0, height_in=2.6)


# --- fig7 per-trajectory rollout -------------------------------------------
def fig7_per_trajectory():
    """4-panel per-trajectory rollout VRMSE; verifies that mean behavior
    in fig4 is not outlier-driven and shows the distinct OOD failure mode."""
    phys = ROOT / "evals" / "physics"
    conditions = [
        ("baseline_01",   "scratch (1% data)",          C_SCRATCH),
        ("baseline",      "scratch (100% data)",        C_BASELINE),
        ("ft_01",         "MHD pretrain + FT (1%)",     C_MHD_FT),
        ("pretrain_ood",  "MHD pretrain zero-FT (OOD)", C_NS),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.4), sharey=True)
    for ax, (name, lbl, color) in zip(axes.flat, conditions):
        p = phys / name / "rollout_vrmse_full.npz"
        if not p.exists():
            ax.set_title(f"{lbl} — missing data"); continue
        v = np.load(p)["vrmse"]
        steps = 1 + np.arange(v.shape[1])
        for i in range(v.shape[0]):
            ax.plot(steps, v[i], lw=0.6, color=color, alpha=0.35)
        ax.plot(steps, v.mean(0),   lw=1.8, color=color, label="mean")
        ax.plot(steps, np.median(v, axis=0), lw=1.1, color="black", ls="--",
                label="median")
        s1_m = v[:, 0].mean();  s1_s = v[:, 0].std()
        s50_m = v[:, 49].mean(); s50_s = v[:, 49].std()
        ax.set_title(
            f"{lbl}\nstep-1 {s1_m:.2f}$\\pm${s1_s:.2f}  "
            f"step-50 {s50_m:.2f}$\\pm${s50_s:.2f}",
            fontsize=8.5)
        ax.set_yscale("log")
        ax.set_xlabel("rollout step")
        ax.legend(loc="lower right", fontsize=7, frameon=False)
    axes[0, 0].set_ylabel("VRMSE vs truth")
    axes[1, 0].set_ylabel("VRMSE vs truth")
    fig.suptitle("Per-trajectory rollout VRMSE — thin lines are individual "
                 "test trajectories (n=10)", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig7_per_trajectory_rollout", width_in=7.0, height_in=4.4)


# --- main -------------------------------------------------------------------
def main():
    print("Generating paper-quality figures...")
    fig1_headline()
    fig2_data_efficiency()
    fig3_cascade()
    fig4_long_horizon()
    fig5_equipartition()
    fig6_conservation()
    fig7_per_trajectory()
    print(f"\nAll figures in {OUT}/")


if __name__ == "__main__":
    main()
