# Diagnostic profile of zero-shot Walrus on MHD_64 plasma turbulence

**Date:** 2026-05-06
**Scope:** zero-shot pretrained Walrus 1.3B (`polymathic-ai/walrus`), evaluated against the same
10 M_A=0.7 MHD_64 test trajectories Paper 1 used (well-work@3b348d4).
**Compute:** ~$10 total on vast.ai (RTX 4090 + A100 80GB).
**Code:** `p2-walrus/` subdir of well-work repo.

## What this is and is not

**This is:** a structural-failure-mode characterization of *zero-shot pretrained* Walrus
on plasma MHD test data, using Paper 1's physics-feature diagnostic apparatus.

**This is not:** a comparison to Walrus's published Table 1 MHD T21-60 = 1.2256 number.
That number is for *Walrus fine-tuned on MHD per Sec 5.2 cross-domain protocol* (additional
500K samples). Polymathic has not released a fine-tuned MHD checkpoint (HF: only base
`polymathic-ai/walrus` available).

**The paper itself shows zero-shot Walrus blows up on MHD.** Figure 15 (Appendix E.4.1)
of the Walrus paper plots the "Walrus-PT" (pretrained, no fine-tune) rollout curve for
in-distribution datasets. The MHD (3D) panel shows it climbing from VRMSE ~1 at step 1
to roughly **10⁶ by step 100** — consistent with our findings. Their Table 7 also lists
MHD (3D) as one of only 3/19 pretraining datasets with median trajectory-averaged
VRMSE ≥ 10 *with* patch jitter on. So the catastrophic zero-shot divergence is the
paper's own published result. **What we add is the physics-feature decomposition of
that failure mode**, not the observation that it exists.

The natural follow-up — running the Sec 5.2 fine-tune protocol and applying this diagnostic
suite to the result — is **future work** (estimated ~$200–400 on 4×H100). Best done as a
Polymathic intern deliverable, not on personal compute budget.

## Setup

**Configs evaluated** (all on shifted/aligned windows so models start from same physical state):

| config              | model                    | trained on            | conditioning |
|---------------------|--------------------------|-----------------------|--------------|
| FNO scratch         | FNO3D, 18M, h=48         | M_A=0.7 (1% data)     | 1 frame      |
| FNO MHD-pretrain+FT | FNO3D, 18M               | M_A=2.0 → M_A=0.7 FT  | 1 frame      |
| FNO MHD zero-FT     | FNO3D, 18M               | M_A=2.0 only          | 1 frame      |
| **Walrus zero-shot**| Walrus 1.3B (base)       | 19 domains incl. MHD train | 3 frames |

**Two evaluation windows tested** (both find the same structural pattern):

- **Early window:** input frames [0,1,2] → predict frames [3..52], K=50, n_traj=10
- **Paper window:** input frames [14,15,16] → predict frames [17..76], K=60, n_traj=5
  (matches paper Sec 5 convention "all predicted trajectories begin from T=17")

**Three confounds make this characterization, not benchmarking:**
- scale: Walrus 70× larger than FNO
- contamination: Walrus saw MHD_64 train; FNOs didn't
- conditioning: 3 frames vs 1 frame

## Headline finding

**Zero-shot Walrus exhibits a structural basin instability in the perpendicular MHD plane,
emerging around rollout step 20 in both evaluation windows, with deterministic
input-state-dependent selection between the B_y and B_z basins.**

Per-step median VRMSE evolution (paper window, n=5):

| step | median VRMSE | mean VRMSE | range |
|-----:|-------------:|-----------:|---------------------:|
| 1    | 0.10         | 0.10       | [0.08, 0.12] |
| 20   | 1.26         | 1.38       | [1.14, 1.68] |
| 30   | 4.55         | 8.13       | [1.73, 20.8] |
| 50   | 297.5        | 6,835      | [9.2, 31,528] |
| 60   | 1,713        | 311,822    | [10.0, 1.5×10⁶] |

Step-1 prediction is the **best** of all 4 configs (0.10 — Walrus has clearly learned
one-step MHD physics excellently). The instability emerges *only* through autoregressive
feedback past step ~20.

## Five evidence chains supporting the structural-instability claim

### (1) Per-trajectory basin selection — symmetry breaking, not random

Per-trajectory variance at step 50 (paper window):

```
traj           rho       B_x      B_y         B_z         v_x       v_y       v_z
0          1.7e-02   5.4e-02  4.4e-02     1.8e+05     1.8e-01   7.4e-02   8.8e-01    ← B_z basin
1          4.3e-02   4.2e-02  3.3e-01     9.8e+09     1.6e-01   1.8e-01   1.6e-02    ← B_z basin (large)
2          2.7e-01   2.2e-02  7.1e-02     1.6e+07     2.5e-01   1.3e-01   7.5e-01    ← B_z basin
3          6.2e-01   2.6e-02  2.8e+02     2.6e-01     3.4e-01   1.7e-01   1.2e-01    ← B_y basin
4          2.2e+00   2.4e-02  1.0e+13     1.5e-01     2.9e-01   8.3e-02   1.5e-01    ← B_y basin (giant)
```

- Each trajectory picks **exactly one** of {B_y, B_z} to amplify. Never both equally.
- Choice is determined by the input state, not random noise (perturbation test below).
- B_x (parallel guide field), density, all velocity components stay near truth-like throughout.
- Same physical trajectory file can flip basins depending on which 3-frame slice is conditioned on
  (early-window traj 0 → B_y basin; paper-window traj 0 → B_z basin).

### (2) Perturbation-seed test — basin is deterministic per input

For **early-window trajectory 0** (which selects the B_y basin in unperturbed inference),
5 different 1e-6 amplitude Gaussian perturbations of the 3-frame input:

| seed | step-50 B_y variance | step-50 B_z variance |
|------|---------------------:|---------------------:|
| 0    | 9.3 × 10⁷           | 0.11                 |
| 1    | 1.8 × 10⁸           | 0.09                 |
| 2    | 2.0 × 10⁸           | 0.09                 |
| 3    | 5.0 × 10⁸           | 0.10                 |
| 4    | 4.2 × 10⁷           | 0.12                 |

**5/5 perturbations stayed in B_y basin.** The 12× spread in B_y magnitude is chaotic
exponential amplification of the noise *within* the basin — but the basin selection itself
is robust to infinitesimal perturbation.

→ Combined with (1): **basin selection depends on the specific 3-frame input slice**, not
the underlying physical trajectory identity, and is robust to infinitesimal perturbations
of any given input slice. Same simulation file can map to B_y or B_z basin depending on
which 3-frame slice you condition on (early-window traj 0 → B_y basin; paper-window traj 0
→ B_z basin).

### (3) FP64 rollout — instability is structural, not numerical

Walrus was trained in FP32 (paper Sec B.1). We default to FP32 inference. To rule out
"divergence is FP32 round-off compounding", we ran FP64 inference on A100 80GB:

| step | FP32 B_y var (early, traj 0) | FP64 B_y var (early, traj 0) |
|-----:|---:|---:|
| 1    | 0.10 | 0.11 |
| 11   | 0.11 | 0.12 |
| 26   | ~few | 11.7 |
| 50   | 1.1 × 10⁸ | 9.1 × 10⁷ |

FP64 B_z paper-window (traj 0): step-31=19.7, step-51=2.3×10⁴, step-60=4.1×10⁵ — same order
of magnitude as FP32 paper-window step-60 (1.8×10⁵).

**FP64 does not suppress the divergence.** Same magnitude at step 50 in both windows.
The instability is **structural in Walrus's autoregressive map**, not a numerical artifact.

### (4) Pipeline equivalence — substitution-correct vs trainer

Line-by-line diff of our standalone `_standalone_rollout_model` (lifted from
`walrus/demo_notebooks/walrus_example_1_RunningWalrus.ipynb`) vs
`walrus/trainer/training.py:rollout_model` (lines 402–562):

| # | Difference | Functional impact for MHD |
|---|---|---|
| 1 | Hardcode `predict_delta=True` vs config-driven | extended_config has `prediction_type: delta` → identical |
| 2 | Hardcode `train_rollout_limit=1` vs `T_in if (train and causal) else 1` | We pass train=False → identical |
| 3 | No ensemble averaging vs `validation_one_step_ensemble_size` | Default = 1 in extended_config → identical |
| 4 | No autocast wrapper on denormalization | We don't enable autocast at all; model runs fp32 → moot |
| 5 | Skip `padded_field_mask` resize hack | Comment says "Quick hack for Neutron"; doesn't trigger for MHD (both = 7) |
| 6 | Always check mask presence vs AND with `masked_loss_for_objects` | MHD `constant_field_names[0] = []` → both → mask=None |

All differences functionally equivalent for MHD_64. Substitution of our 3-frame history into
the dataloader's batch template populates `boundary_conditions`, `padded_field_mask`,
`field_indices`, etc. correctly from the dataloader.

### (5) Cascade evolution — preferential high-k pile-up (Gibbs-style aliasing)

Trajectory-averaged B_y perpendicular-cascade energy at low-k_∥, by k-band, plus growth
factors from step 1 to step 50:

| step | low-k E | mid-k E | high-k E | total |
|-----:|--------:|--------:|---------:|------:|
| 1    | 8.2e-3  | 1.5e-3  | 5.4e-5   | 9.7e-3 |
| 5    | 8.2e-3  | 1.4e-3  | 5.6e-5   | 9.7e-3 |
| 10   | 8.9e-3  | 1.7e-3  | 1.0e-4   | 1.1e-2 |
| 25   | 2.6     | 6.7     | 1.4      | 1.1e+1 |
| 50   | 2.8e+7  | 4.6e+7  | 8.1e+6   | 8.2e+7 |
| **growth (step 50 / step 1)** | **3.4×10⁹** | **3.1×10¹⁰** | **1.5×10¹¹** | — |

**High-k modes grew ~50× faster than low-k modes.** The high/low ratio collapses from
~150 (truth-like, low-k dominated) at step 1 to ~3 at step 50. The spectrum flattens during
rollout, with energy preferentially piling up at high wavenumbers.

This is **Gibbs-style aliasing accumulating at the patch grid scale** — exactly the kind of
small-scale numerical-instability signature that motivated Walrus's patch-jitter mechanism
in the first place (paper Sec 3.2: "Aliasing can lead to spectral artifacts… which leads to
this grid-imprinting"). Patch jitter helps but does not fully suppress it under autoregressive
feedback.

**This strengthens the architectural-intervention story:** divergence-free spectral projection
at the AR output would damp specifically the constraint-violating high-k modes that compound
into Gibbs accumulation. The instability is structurally amenable to spectral filtering.

## Conservation diagnostics at step 50

| config | step-50 \|B\| mean | step-50 \|B\| median | step-50 \|B\| range | step-50 ∇·B/\|B\| | step-50 E_B/E_K |
|-------|----:|----:|----:|----:|---:|
| FNO scratch       | 1.26   | 1.26 | [1.24, 1.29]    | 0.056 | 1.21 |
| FNO MHD-pretrain+FT | 5.51 | 5.06 | [2.80, 9.04]    | 0.134 | 0.11 |
| FNO MHD zero-FT   | 0.62   | 0.63 | [0.48, 0.70]    | 0.325 | 0.09 |
| **Walrus**        | **21,651** | **690** | [32, 147,879]   | **0.82** | **3.5×10⁹** |
| **truth**         | 1.13   | —    | —               | 0.033 | 2.15 |

Walrus's step-50 |B| distribution is highly skewed: median ≈ 690, mean ≈ 22,000, dominated
by 1-2 trajectories with |B| ~ 10⁵. With ∇·B/|B| = 0.82 across all trajectories:
- Using **median |B|** ≈ 690: absolute ∇·B ≈ 0.82 × 690 ≈ **570** → ~15,000× truth's 0.037
- Using **mean |B|** ≈ 22,000: absolute ∇·B ≈ 0.82 × 22,000 ≈ **18,000** → ~5×10⁵× truth's 0.037

Either way, massive solenoidal-constraint failure co-localized with the perpendicular
B-component blow-up. Median is more representative of the "typical" trajectory; mean is
dominated by the most-diverged ones.

## Five-point Walrus-paper audit

Verified against the paper text and walrus repo before reporting:

| # | Check | Status |
|---|---|---|
| V1 | Validation time stride = 1 (consecutive frames) | ✅ matches (`multidatamodule.py:284` forces max=min=1 for valid) |
| V2 | Number of trajectories | ⚠️ ours: 5/10; paper Sec 3.2 says "20 validation trajectories" while Sec E.2 says "32 trajectories" (paper-internal inconsistency); Sec 5.2 N for Table 1 unspecified → **median variance caveat** |
| V3 | Boundary tagging periodic for MHD_64 | ✅ matches (BC tensor = `[[2,2],[2,2],[2,2]]` = all periodic) |
| V4 | Patch jitter at evaluation | ✅ matches (`jitter_patches: true` in extended_config; no train/eval branching — paper eval ALSO has jitter on) |
| V5 | Field transformations for MHD | ✅ matches (no MHD transforms in walrus configs/data/MHD_64.yaml; none in paper Sec C.1) |

Only V2 is a documented caveat. With heavy-tailed step-50 distributions, our medians (e.g.,
T21-60 = 18.25) have wider error bars than would be obtained with paper-scale N. The
qualitative pattern (catastrophic divergence, basin selection, FP64-confirmed structural)
is N-invariant.

## Mechanism interpretation

**The instability is on an unstable manifold with two physically-equivalent perpendicular
sub-directions** (B_y and B_z, statistically equivalent in M_A=0.7 turbulence with B_0 ‖ x̂).
Walrus's autoregressive feedback selects whichever sub-direction is more aligned with its
learned unstable eigendirection at the start of the rollout, then amplifies it exponentially.

This is mechanistically the kind of instability that **divergence-free projection at the
autoregressive output** would damp. A solenoidal projection step would zero out
constraint-violating modes regardless of which perpendicular component they're concentrated
in — and the projection is symmetric in B_y/B_z by construction.

## Limitations and future work

- **N=5 paper-window / N=10 early-window** — heavy-tailed distributions, wider error bars
  than ideal. Future replication should use ≥32 trajectories.
- **Zero-shot only** — does this instability survive the paper's Sec 5.2 fine-tune protocol
  (additional 500K MHD samples)? **Open question; the natural follow-up experiment.**
  Polymathic has not released a fine-tuned MHD checkpoint, so this would require running
  the fine-tune ourselves (~$200–400 compute, 1–2 days).
- **VRMSE formula** — paper Sec E.1.1 uses joint space+channel averaging; we per-channel
  sqrt then mean. Cross-config comparisons are apples-to-apples; comparison to paper's
  Table 1 magnitudes is muddied (but moot since Table 1 is fine-tuned anyway).
- **Two distinct claims, different scopes:**
  - The **B_y/B_z basin-selection signature** is observed in Walrus 1.3B specifically;
    we have no comparable observation on FNO. May be an idiosyncrasy of Walrus's
    transformer + tokenizer + jitter combination.
  - The broader **late-stage ∇·B-violation-driven divergence pattern** generalizes:
    Paper 1's FNO MHD-pretrain+FT also exhibits late-stage rollout instability
    (step-50 VRMSE ~12, ∇·B/|B| 0.13) consistent with the same underlying mechanism
    — one-step-loss-trained models lack solenoidal-constraint enforcement, so ∇·B
    error accumulates and amplifies under autoregressive feedback. The architectural
    intervention (divergence-free projection) would help both architectures.

## Implications

Even reframed as zero-shot diagnostic: this is a defensible characterization of a
foundation-model failure mode that the published metrics (median VRMSE on a fine-tuned
checkpoint) would not surface to a downstream user. The diagnostic apparatus is the
methodological contribution; Walrus is one case study.

For deployment: zero-shot pretrained Walrus cannot be trusted for autoregressive plasma
rollouts past step ~20. Whether the Sec 5.2 fine-tuning protocol resolves this is the
critical follow-up question.

For architecture: the instability is symmetric in B_y/B_z and concentrated on
constraint-violating modes — argues for divergence-free projection at AR output as a
testable architectural intervention, regardless of fine-tuning outcome.

## Files

- `results/shifted_window/{fno_baseline,fno_ft,fno_pretrain_ood,walrus}/*.npz` — early-window diagnostics
- `results/paper_window/walrus/*.npz` — paper-window diagnostics
- `results/extra_checks/{perturb_rollouts.npz,fp64_rollout.npz}` — early-window FP64 + perturb
- `results/extra_checks_paper_window/fp64_rollout.npz` — paper-window FP64
- `figures/fig_vrmse_overlay.pdf` — VRMSE vs step (4 configs, log-y, early window)
- `figures/fig_cascade.pdf` — perpendicular cascade at step 1
- `figures/fig_cascade_evolution.pdf` — cascade at steps 1, 5, 10, 25, 50
- `figures/fig_bx_norm.pdf`, `fig_divb.pdf`, `fig_equipartition.pdf`, `fig_per_traj.pdf`

## Friday pitch (for Shirley meeting)

> "I extended Paper 1's physics-feature diagnostic suite to zero-shot pretrained Walrus
> on the same MHD_64 test trajectories. Found a structural basin instability in the
> perpendicular magnetic plane that emerges around step 20, with input-state-dependent
> deterministic selection between the B_y and B_z basins. FP64-confirmed structural
> rather than numerical; perturbation-confirmed deterministic basin selection.
>
> Section 5.2 of the Walrus paper reports a per-task fine-tuning protocol that achieves
> T21-60 median VRMSE = 1.2256 on MHD *for the fine-tuned model*; the natural next
> experiment is whether that fine-tuning protocol also fixes the basin instability or
> whether it survives.
>
> That's what I'd want to do as a Polymathic intern — run the Sec 5.2 fine-tune for MHD,
> apply this diagnostic suite, characterize whether per-task fine-tuning is sufficient
> for plasma stability or whether architectural intervention (e.g., solenoidal projection
> at AR output) is needed for deployment-grade stability."
