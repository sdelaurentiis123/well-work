"""Headline figure v2 — data-efficiency with error bars across seeds."""
from __future__ import annotations
import argparse, json, math, re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SEED_RX = re.compile(r"^(?P<base>.+?)(?:_s\d+)?$")


def best_val(log_path: Path) -> float:
    if not log_path.exists(): return math.nan
    best = math.inf
    for line in log_path.read_text().splitlines():
        if not line.strip(): continue
        v = json.loads(line).get("val_vrmse", math.inf)
        if v < best: best = v
    return best


def run(args):
    runs = Path(args.runs_dir)
    # Collect per-(config, seed) best_val.
    configs = {
        1.0:  {"baseline": "baseline",    "ft": "ft_100"},
        0.1:  {"baseline": "baseline_10", "ft": "ft_10"},
        0.01: {"baseline": "baseline_01", "ft": "ft_01"},
    }
    results = {}
    for frac, cfg in configs.items():
        for kind, name in cfg.items():
            seeds = []
            for d in runs.glob(f"{name}*"):
                if not (d / "log.jsonl").exists(): continue
                v = best_val(d / "log.jsonl")
                if math.isfinite(v): seeds.append(v)
            results[(frac, kind)] = seeds
            print(f"  {name}: {len(seeds)} seeds  best={seeds}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"baseline": "#d62728", "ft": "#1f77b4"}
    labels = {"baseline": "from scratch (M_A=0.7)",
              "ft": "pretrained on M_A=2.0, fine-tuned"}
    marker = {"baseline": "o", "ft": "s"}
    for kind in ("baseline", "ft"):
        xs, means, stds = [], [], []
        for frac in sorted(configs.keys()):
            vals = results.get((frac, kind), [])
            if not vals: continue
            xs.append(frac); means.append(np.mean(vals)); stds.append(np.std(vals))
        xs, means, stds = np.array(xs), np.array(means), np.array(stds)
        ax.errorbar(xs, means, yerr=stds, marker=marker[kind], ms=8, lw=2,
                    capsize=5, label=labels[kind], color=colors[kind])

    ax.set_xscale("log")
    ax.set_xlabel("fraction of M_A=0.7 training data")
    ax.set_ylabel("best validation VRMSE  (mean ± std across seeds)")
    ax.set_title("P1 transfer study: ISM-regime → fusion-regime MHD\n"
                 "FNO3D on Polymathic Well MHD_64")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")

    # Print summary table with error bars
    print("\n=== Headline table (with error bars) ===")
    print(f"{'Data':<8}  {'From scratch (μ±σ)':<22}  {'FT (μ±σ)':<22}  {'Δ%':>7}")
    for frac in sorted(configs.keys(), reverse=True):
        b = results.get((frac, "baseline"), [])
        f = results.get((frac, "ft"), [])
        if not (b and f): continue
        bm, bs = np.mean(b), np.std(b)
        fm, fs = np.mean(f), np.std(f)
        delta = 100 * (fm - bm) / bm
        print(f"{frac*100:>5.0f}%    {bm:.4f} ± {bs:.4f}      {fm:.4f} ± {fs:.4f}      {delta:+.1f}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="p1/figures/p1_data_efficiency_v2.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
