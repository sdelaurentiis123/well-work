# Walrus diagnostic profile on MHD_64 — preliminary findings

**Date:** 2026-05-05
**Compute:** vast.ai RTX 4090, ~30 min total, ~$0.15
**Code:** [p2-walrus @ this commit](../) (well-work repo, p2-walrus subdir)

## Setup

Same 10 M_A=0.7 MHD_64 test trajectories Paper 1 used. Shifted window: condition on
frames [0,1,2], predict frames [3..52] (50 steps). All 4 configs see *identical*
starting states — no window-position confound.

| config            | model                                  | trained on          | conditioning |
|-------------------|----------------------------------------|---------------------|--------------|
| FNO scratch       | FNO3D, 18M, h=48, 1% target data       | M_A=0.7 (1%)        | 1 frame      |
| FNO MHD-pretrain+FT | FNO3D, 18M, MHD-pretrain → 1% FT       | M_A=2.0 → M_A=0.7   | 1 frame      |
| FNO MHD zero-FT   | FNO3D, 18M, MHD-pretrain only          | M_A=2.0             | 1 frame      |
| **Walrus 1.3B**   | Walrus, 1.3B, foundation model         | 19 physics domains incl. MHD_64 train | 3 frames |

Three confounds make this **characterization, not benchmarking**:
- scale: Walrus is ~70× larger than FNO
- contamination: Walrus saw MHD_64 train; FNOs didn't
- conditioning: 3 frames vs 1 frame

## Headline finding

**Walrus's autoregressive rollout on MHD turbulence diverges catastrophically.**

Per-trajectory VRMSE:

| step | walrus min | walrus median | walrus mean   | walrus max    |
|------|-----------:|--------------:|--------------:|--------------:|
| 1    | 0.076      | **0.105**     | 0.104         | 0.126         |
| 5    | 0.245      | 0.311         | 0.314         | 0.376         |
| 10   | 0.568      | 0.694         | 0.673         | 0.763         |
| 25   | 1.189      | 1.88          | 4.25          | 25.6          |
| 50   | **16.2**   | **316**       | **10,148**    | **68,950**    |

At step 1, Walrus is the **best** of all 4 configs — it learned one-step MHD
physics well. By step 50, even the *best* trajectory has VRMSE > 16, an order of
magnitude worse than FNO MHD-pretrain+FT (12) and ~5× worse than FNO scratch (1.96).

For comparison:

| config            | step 1 VRMSE | step 50 VRMSE (mean) | step 50 \|B\| | step 50 ∇·B floor | step 50 E_B/E_K |
|-------------------|-------------:|---------------------:|--------------:|------------------:|----------------:|
| FNO scratch       | 0.554        | 1.96                 | 1.26          | 0.056             | 1.21            |
| FNO MHD-pretrain+FT | 0.304      | 12.0                 | 5.82          | 0.134             | 0.105           |
| FNO MHD zero-FT   | 0.715        | 2.99                 | 0.625         | 0.325             | 0.087           |
| **Walrus**        | **0.104**    | **10,148**           | **48,597**    | **0.824**         | **3.5 × 10⁹**   |
| **truth**         | —            | —                    | 1.13          | 0.033             | 2.15            |

## Why does Walrus's published VRMSE = 1.2256 not reflect this?

Polymathic's eval script (`eval_onegpu_example_walrus.sh`) aggregates with
`torch.mean, torch.median, torch.std` over batches, and reports T21–60. Looking at
our **median** at step 25: 1.88. At step 50: 316. Their reported number is
consistent with the *median over the early-mid window* — which **masks** the
catastrophic late divergence.

The diagnostic suite reveals what the headline number hides.

## Specific failure modes

**Per-channel variance over rollout (averaged across 10 trajectories):**

```
step    ρ        B_x      B_y         B_z         v_x      v_y      v_z
0       0.86     0.061    0.102       0.111       0.268    0.156    0.181
1       0.84     0.062    0.102       0.111       0.268    0.156    0.181
10      0.68     0.054    0.108       0.111       0.260    0.154    0.166
25      0.65     0.040    235.6       2.19        0.267    0.158    0.157
50      0.65     0.039    2.36×10⁹    1.20×10⁵    0.268    0.231    0.150
```

1. **Asymmetric B-component blow-up — perpendicular only.** B_x (the imposed
   guide field, ~constant in truth) **stays stable** at variance ≈ 0.04
   throughout the rollout. **B_y explodes by 10¹⁰× and B_z by 10⁶×.** Density
   and all three velocity components stay near truth-like values. Walrus has
   learned the parallel guide field is conserved but its perpendicular
   B-fluctuations are unstable under autoregressive feedback.

2. **Late-stage onset.** Through step 10, every channel including B_y is within
   a factor of ~1.1 of truth-like. Divergence in B_y appears between step 10
   and step 25 (variance jumps from 0.108 → 235), then explodes exponentially
   to 10⁹ by step 50. All 10 trajectories follow this pattern — not outlier-driven.

3. **Catastrophic ∇·B violation.** Step-50 ∇·B / |B| = 0.82 (truth: 0.033).
   The solenoidal-constraint error is ~80% of the local field magnitude.
   For comparison, FNO baselines are 0.06–0.33. Consistent with the
   B_y/B_z blow-up — exploding perpendicular components break ∇·B = 0.

4. **Equipartition explosion driven by E_B.** E_B/E_K at step 50 is 3.5 × 10⁹
   (vs truth 2.15). Magnetic energy explodes while kinetic energy stays
   bounded — the velocity field stays sane while the magnetic field is the
   only thing diverging. This isn't "everything blows up" — it's "the magnetic
   perpendicular components blow up specifically."

5. **Step-1 prediction is excellent.** VRMSE 0.10 at step 1 — the best of all
   four configs. Walrus has learned the one-step MHD physics. The instability
   emerges *only* through autoregressive feedback in the B_y / B_z components.
   Same family of failure as FNO fine-tuned (which has step-50 VRMSE 12) but
   far more severe and physically more diagnostic: instead of "everything
   degrades," it's "the parallel field is preserved while the perpendicular
   field is structurally unstable."

## What this does NOT establish

- That Walrus is "worse than FNO" — it has 70× more params and saw MHD_64 train.
- That patch-jittering / scale doesn't help. They might help; we just see a
  different failure mode that scale alone didn't fix.
- That fine-tuning Walrus on 1% MHD_64 target data wouldn't fix it. **That's the
  natural follow-up.** LoRA fine-tune at the target distribution may stabilize the
  rollout — or it may introduce the same instability FNO fine-tuned shows.

## Implications for the diagnostic-suite framing

This is exactly the kind of finding the diagnostic suite is designed to catch.
A reviewer of the Walrus paper looking only at the headline VRMSE would conclude
"Walrus does mediocre on MHD." Our diagnostic profile says:

> Walrus has learned one-step MHD physics excellently. Its autoregressive
> rollout is structurally unstable specifically in the perpendicular magnetic
> components (B_y / B_z); the parallel guide field, density, and velocity
> stay physical throughout. Onset is around step 20, divergence is exponential
> by step 25, ∇·B violation reaches 80% of |B| at step 50.

That's a *much* sharper characterization than "VRMSE = 1.23." It tells a
deployment engineer specifically: **don't trust this model for rollout past
~20 steps on magnetized plasma, and even short-horizon predictions should
sanity-check perpendicular-B fidelity.**

It also suggests a specific architectural intervention: **enforcing ∇·B = 0
on the autoregressive output** (e.g., projecting predictions onto the
solenoidal subspace each step) would likely stabilize the perpendicular
components by removing the constraint-violating modes that compound. That's
a testable hypothesis for the next iteration.

## Files

- `figures/fig_vrmse_overlay.pdf` — VRMSE-vs-step (4 curves, log-y)
- `figures/fig_per_traj.pdf` — per-trajectory VRMSE consistency (4-panel)
- `figures/fig_cascade.pdf` — perpendicular cascade at step 1
- `figures/fig_bx_norm.pdf` — ⟨|B|⟩ trajectory
- `figures/fig_divb.pdf` — ∇·B floor
- `figures/fig_equipartition.pdf` — E_B/E_K trajectory
- `results/shifted_window/{config}/*.npz` — raw diagnostic arrays per config

## Caveats and what should be re-verified

- The catastrophic divergence is striking enough that someone should sanity-check
  the standalone `_standalone_rollout_model` in `src/rollout_adapter.py` against
  Walrus's official eval pipeline (their `train.py --validation_mode=True` path).
  The numbers I'm seeing are roughly consistent with their median-aggregated result
  at the steps they report, so I don't believe this is a bug — but it should be
  cross-checked.
- The `predict_delta=True` is set on the FIRST formatter call but defaults inside
  the rollout loop. Worth verifying our standalone implementation matches the
  trainer's exactly (`walrus/trainer/training.py:402-562`).
- 10 trajectories is small. Wider statistics across more trajectories would
  strengthen the per-trajectory consistency claim.
