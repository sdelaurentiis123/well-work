"""Task 3: HP search on baseline_01 (scratch, 1% data) and ft_01 (pretrain + 1% FT).

Phase 1 — coarse single-seed sweep over {lr, hidden, epochs} for baseline_01, and
          {lr} only for ft_01 (fixed hidden=48 since pretrained weights expect it,
          fixed epochs=40 to keep budget fair).
Phase 2 — top-5 baseline_01 configs at 3 seeds; top-3 ft_01 configs at 3 seeds.

Writes results to runs/hp_baseline01/<cfg_hash>/ and runs/hp_ft01/<cfg_hash>/.
Idempotent — skips any completed config (log.jsonl with correct epoch count).
"""
from __future__ import annotations
import argparse, itertools, json, hashlib, os, subprocess, sys, time
from pathlib import Path


BASELINE_GRID = {
    "lr":     [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden": [32, 48, 64],
    "epochs": [40, 80, 120],
}
FT_GRID = {
    "lr":     [1e-4, 3e-4, 1e-3, 3e-3],
    "hidden": [48],     # fixed — pretrained weights expect hidden=48
    "epochs": [40],     # fixed — budget parity with ft_01 baseline
}


def cfg_id(cfg: dict) -> str:
    s = json.dumps(cfg, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def iter_grid(grid):
    keys = list(grid.keys())
    for vals in itertools.product(*[grid[k] for k in keys]):
        yield dict(zip(keys, vals))


def run_one(cfg: dict, mode: str, init_ckpt: str, seed: int, out_base: Path,
            data_base: str, run_name_prefix: str, log_progress: bool):
    cid = cfg_id(cfg)
    out_dir = out_base / f"{cid}_s{seed}"
    logp = out_dir / "log.jsonl"
    if logp.exists():
        # count epochs completed
        lines = [l for l in logp.read_text().splitlines() if l.strip()]
        if len(lines) >= cfg["epochs"]:
            print(f"  SKIP  {mode} {cid} seed={seed}  (already {len(lines)}ep)")
            return out_dir, "skipped"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "python", "-u", "p1/train.py",
        "--mode", mode,
        "--out", str(out_dir),
        "--data_base", data_base,
        "--bs", "8",
        "--lr", str(cfg["lr"]),
        "--modes", "12",
        "--hidden", str(cfg["hidden"]),
        "--epochs", str(cfg["epochs"]),
        "--seed", str(seed),
        "--workers", "4",
        "--data_frac", "0.01",
        "--run_name", f"{run_name_prefix}_{cid}_s{seed}",
    ]
    if mode == "finetune" and init_ckpt:
        cmd += ["--init_ckpt", init_ckpt]
    if log_progress:
        print("  CMD:", " ".join(cmd))
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("WANDB_PROJECT", "well-work-p1")
    env.pop("WANDB_ENTITY", None)
    try:
        with open(out_dir/"stdout.log", "w") as f:
            subprocess.run(cmd, check=True, stdout=f, stderr=subprocess.STDOUT, env=env)
        elapsed = time.time() - t0
        lines = [json.loads(l) for l in (out_dir/"log.jsonl").read_text().splitlines() if l.strip()]
        best = min(r["val_vrmse"] for r in lines) if lines else float("inf")
        print(f"  DONE  {mode} {cid} seed={seed}  best={best:.4f}  {elapsed/60:.1f}min")
        return out_dir, best
    except subprocess.CalledProcessError as e:
        print(f"  FAIL  {mode} {cid} seed={seed}  rc={e.returncode}")
        return out_dir, "failed"


def best_val(path: Path) -> float:
    logp = path / "log.jsonl"
    if not logp.exists(): return float("inf")
    lines = [json.loads(l) for l in logp.read_text().splitlines() if l.strip()]
    return min((r["val_vrmse"] for r in lines), default=float("inf"))


def phase1_coarse(grid, mode, out_base: Path, init_ckpt, data_base, run_name_prefix, log_progress):
    """Run all grid configs at seed 0, return ranked list."""
    print(f"\n=== Phase 1: coarse sweep ({mode}) — {sum(1 for _ in iter_grid(grid))} configs ===")
    results = []
    for cfg in iter_grid(grid):
        out_dir, best = run_one(cfg, mode, init_ckpt, 0, out_base, data_base,
                                run_name_prefix, log_progress)
        if best != "failed":
            results.append((cfg, out_dir, best if best != "skipped" else best_val(out_dir)))
    results.sort(key=lambda x: x[2])
    return results


def phase2_refine(top_cfgs, mode, out_base: Path, init_ckpt, data_base, run_name_prefix, log_progress):
    """Rerun top-n configs at seeds 1, 2."""
    print(f"\n=== Phase 2: refining {len(top_cfgs)} configs × 2 additional seeds ===")
    rows = []
    for cfg, out_dir_s0, best_s0 in top_cfgs:
        seeds_best = [best_s0]
        for s in (1, 2):
            _, best = run_one(cfg, mode, init_ckpt, s, out_base, data_base,
                              run_name_prefix, log_progress)
            if best == "failed": continue
            if best == "skipped":
                best = best_val(out_base / f"{cfg_id(cfg)}_s{s}")
            seeds_best.append(best)
        rows.append((cfg, seeds_best))
    return rows


def report(rows, title):
    print(f"\n=== {title} ===")
    print(f"  {'cfg':<56}  mean ± std  (per seed)")
    import numpy as np
    best_overall = None
    for cfg, seeds_best in sorted(rows, key=lambda r: sum(r[1])/len(r[1])):
        s = np.array(seeds_best)
        mean = s.mean(); std = s.std()
        cfg_str = ", ".join(f"{k}={v}" for k,v in cfg.items())
        per = ", ".join(f"{v:.4f}" for v in seeds_best)
        print(f"  {cfg_str:<56}  {mean:.4f} ± {std:.4f}   [{per}]")
        if best_overall is None: best_overall = (cfg, mean, std)
    return best_overall


def run(args):
    data_base = args.data_base
    out_baseline = Path(args.out_root) / "hp_baseline01"
    out_ft = Path(args.out_root) / "hp_ft01"
    out_baseline.mkdir(parents=True, exist_ok=True)
    out_ft.mkdir(parents=True, exist_ok=True)

    # --- baseline_01 coarse ---
    top_b = phase1_coarse(BASELINE_GRID, "baseline", out_baseline, None, data_base,
                         "hpb", args.log_progress)
    top_b = top_b[:args.top_k]
    # refine
    rows_b = phase2_refine(top_b, "baseline", out_baseline, None, data_base, "hpb", args.log_progress)
    best_b = report(rows_b, "baseline_01 HP sweep results")

    # --- ft_01 sweep ---
    top_f = phase1_coarse(FT_GRID, "finetune", out_ft, args.pretrain_ckpt, data_base,
                         "hpf", args.log_progress)
    top_f = top_f[:args.top_k_ft]
    rows_f = phase2_refine(top_f, "finetune", out_ft, args.pretrain_ckpt, data_base,
                          "hpf", args.log_progress)
    best_f = report(rows_f, "ft_01 HP sweep results")

    summary = {
        "baseline_01_best": {"cfg": best_b[0], "mean": best_b[1], "std": best_b[2]} if best_b else None,
        "ft_01_best":       {"cfg": best_f[0], "mean": best_f[1], "std": best_f[2]} if best_f else None,
        "baseline_rows":    [(c, [float(x) for x in s]) for c, s in rows_b],
        "ft_rows":          [(c, [float(x) for x in s]) for c, s in rows_f],
    }
    (Path(args.out_root) / "hp_summary.json").write_text(json.dumps(summary, indent=2, default=str))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_base", default="/root/data/datasets")
    p.add_argument("--out_root", default="runs")
    p.add_argument("--pretrain_ckpt", default="runs/pretrain/best.pt")
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--top_k_ft", type=int, default=3)
    p.add_argument("--log_progress", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse())
