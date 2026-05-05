# p2-walrus — Walrus diagnostic study

Follow-up note to Paper 1 (within-MHD transfer with FNO3D, plasma-tinkering @ 3b348d4).
Runs the Paper 1 physics-feature diagnostic suite on Walrus 1.3B (Polymathic AI's
cross-domain foundation model) on the same 10 M_A=0.7 MHD_64 test trajectories.

**Framing: characterization, not benchmarking.** Three confounds (scale 70×,
contamination — Walrus saw MHD_64 train, conditioning — 10-frame vs 2-frame Markov)
are explicit and disclosed in every figure.

## Methodological correction (vs naive plan)

Walrus needs 10 input frames; FNO uses 2-frame Markov. Naively shifting only
Walrus's window biases three diagnostics (`aniso_step1`, `E_ratio`, `divB_norm`).
**Fix: shift FNO too.** All four configs (3 FNO baselines + Walrus) run on
identical frames [10] → [11..59], 50 prediction steps from frame 10. Paper 1's
frozen `p1/evals/physics/` artifacts stay untouched.

## Layout

```
src/                       refactored extract_physics, rollout adapters, drivers, overlay plotting
slurm/                     SLURM batch + cheatsheet
results/shifted_window/    .npz diagnostic summaries (committed; ~50 MB)
figures/                   overlay PDFs (committed)
findings.md                one-page diagnostic profile (after Phase 5)
data/, ckpts/, outputs/    .gitignore'd; live on Ginsburg
env/                       .gitignore'd; conda env in-tree for clean wipe
```

## Compute

Ginsburg HPC (Columbia), group `astro`, scratch `/burg/astro/users/sod2112/p2-walrus/`.
A40 (18 GPUs) primary, A100 (16 GPUs) fallback. bf16 required → V100S/RTX 8000
ruled out.

## Status

See `noble-scribbling-hopper.md` plan in `~/.claude/plans/` for phase tracking.
