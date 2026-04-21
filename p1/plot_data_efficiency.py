"""P1 headline figure: data-efficiency curves, pretrained vs from-scratch.

Reads log.jsonl files written by train.py and plots best val VRMSE vs data
fraction for both regimes.
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import matplotlib.pyplot as plt


def best_val(log_path: Path) -> float:
    best = math.inf
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        v = rec.get("val_vrmse", math.inf)
        if v < best:
            best = v
    return best


def run(args):
    runs_dir = Path(args.runs_dir)
    points = {}
    for name, frac in [
        ("baseline", 1.0),
        ("baseline_10", 0.1),
        ("baseline_01", 0.01),
        ("ft_100", 1.0),
        ("ft_10", 0.1),
        ("ft_01", 0.01),
    ]:
        log = runs_dir / name / "log.jsonl"
        if not log.exists():
            print(f"  missing: {log}")
            continue
        points[name] = (frac, best_val(log))

    print("points:", points)

    scratch = [(f, v) for k, (f, v) in points.items() if k.startswith("baseline")]
    pretrn = [(f, v) for k, (f, v) in points.items() if k.startswith("ft_")]
    scratch.sort(); pretrn.sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    if scratch:
        fs, vs = zip(*scratch)
        ax.plot(fs, vs, "o-", lw=2, ms=8, label="from scratch (M_A=0.7)", color="#d62728")
    if pretrn:
        fp, vp = zip(*pretrn)
        ax.plot(fp, vp, "s-", lw=2, ms=8, label="pretrained on M_A=2.0, fine-tuned", color="#1f77b4")

    ax.set_xscale("log")
    ax.set_xlabel("fraction of M_A=0.7 training data")
    ax.set_ylabel("best validation VRMSE")
    ax.set_title("P1 transfer study: ISM-regime → fusion-regime MHD\n"
                 "FNO3D on Polymathic Well MHD_64")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--runs_dir", default="runs")
    p.add_argument("--out", default="p1_data_efficiency.png")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
