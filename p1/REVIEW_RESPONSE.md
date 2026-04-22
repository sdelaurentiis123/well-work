# Response to adversarial review — verifications, severity, plan

Two reviews (constructive + adversarial) identified concrete methodological, physics, and framing issues. This document: (1) verifies every external claim in the reviews, (2) audits our own code/data for the internal claims, (3) triages by severity, (4) lays out a concrete fix plan with ROI ordering, and (5) gives the honest ELI5 and the rigorous summary.

## Part A — verifications (what the reviews got right, what they got wrong)

### Verified TRUE and actionable

- **McCabe et al., TMLR 2023, *Toward Stability of Autoregressive Neural Operators*** — arxiv 2306.10619. Abstract directly says: "We analyze the sources of this autoregressive error growth using prototypical neural operator models ... present results on Navier-Stokes fluid flow, rotating shallow water, and high-resolution global weather forecasting ... applying our design principles leads to significantly lower errors for long-term forecasts as well as longer time horizons without qualitative signs of divergence." This is the *mechanistic explanation* of our Figure 4 finding. We did not cite it. **Must fix.**

- **Subramanian et al., NeurIPS 2023, *Towards Foundation Models for Scientific Machine Learning: Characterizing Scaling and Transfer Behavior*** — arxiv 2306.00258. Abstract explicitly studies "(iii) the physics parameters are systematically pushed out of distribution, and (iv) how a single model pre-trained on a mixture of different physics problems can be adapted to various downstream applications." Our novelty claim that "prior work has not isolated whether downstream improvements come from the pretraining task's physics or from pretraining acting as architecture-agnostic regularization" is too strong. Subramanian 2023 addressed related questions first. **Must narrow novelty claim.**

- **Lippe et al., NeurIPS 2023, *PDE-Refiner: Achieving Accurate Long Rollouts with Neural PDE Solvers*** — arxiv 2308.05732. Abstract explicitly names "the neglect of non-dominant spatial frequency information, often associated with high frequencies in PDE solutions, [as] the primary pitfall limiting stable, accurate rollout performance." This is the probable mechanism for our Figure 4 inflation. **Must cite.**

- **Boldyrev 2006 PRL 96, 115002**, *Spectrum of Magnetohydrodynamic Turbulence* — real, cited 797+ times. Predicts E(k⊥) ∝ k⊥^(-3/2) for strong incompressible MHD turbulence with a large-scale external field (= our regime exactly). Our Figure 3 plots the GS95 k⊥^(-5/3) reference slope. For sub-Alfvénic driven MHD with a guide field, Boldyrev's -3/2 is the modern canonical expectation. **Figure 3's reference slope is miscited; must replace or plot both.**

- **Gopakumar et al. 2024, *Plasma Surrogate Modelling using Fourier Neural Operators*** — arxiv 2311.05967, published in Nuclear Fusion. This is the correctly-indexed plasma-FNO paper. Our current bibliography entry `gopakumar2020turbulence` pointing to a 2020 Physics of Plasmas paper appears to be misattributed. **Fix bibliography.**

- **supernova_explosion_64 dataset factual details**, verified on polymathic-ai.org: open boundary conditions, variable Δt (100-10000 yr), 59 timesteps, 740 trajectories, SPH-generated (ASURA-FDPS, not native Eulerian), 6 fields (ρ, p, T, v×3), 7-orders-of-magnitude dynamic range, isotropic gas sphere with Burgers-turbulence initial conditions.
  Implications: multiple confounds in the "NS control" — differs from MHD_64 in BCs, timestep structure, channel count, numerical scheme, and dynamic range. Review B's Confound 1 and Confound 2 are substantiated. **Must acknowledge in limitations; the "physics-specificity" claim requires a sharper control (same BC/Δt/grid structure, different governing equations) to be clean.**
  Also: our manuscript called it "radiation-hydrodynamics"; the dataset docs describe it as hydrodynamics + gravity + radiative heating/cooling source terms, not fully RHD. **Fix label.**

### Verified TRUE but already acknowledged as caveats

- **"Headline uses validation, not test."** Our figures say "best val VRMSE" — factually true. Review A is right that this is weaker than a locked-test claim. We have test-split eval outputs in `evals2/*/results.json` (15 trajectories per ckpt) but the headline bar chart uses val. **Easy fix: swap to test-split numbers.**

- **"n=1 pretrain seed."** True. Our pretrain was run once, same checkpoint for all FT variants. The near-zero FT variance (~10⁻⁴) is post-pretrain optimization noise only, not total uncertainty. **Fix: at minimum state this as a declared limitation; ideally run 2 more pretrain seeds for MHD and NS conditions (6 additional pretraining runs, ~$8 GPU cost).**

### Verified INCORRECT in our own draft — the hidden=48 vs hidden=64 confound

This one is real and material. Verified from `p1/hp_summary.json`:

- Best-tuned baseline config: `lr=3e-3, hidden=64, epochs=40` → 0.419 ± 0.008
- Best-tuned FT config: `lr=3e-4, hidden=48, epochs=40` → 0.301 ± 0.000 (architecture constrained by pretrained weights)

But a matched-architecture comparison exists in the same sweep data:

- Scratch at `lr=3e-3, hidden=48`: 0.465 ± 0.022 (3 seeds)
- FT     at `lr=3e-4, hidden=48`: 0.301 ± 0.000 (3 seeds)

**Matched-architecture gap: 35% improvement, not 28%.** The pretraining effect is actually *stronger* when we control for parameter count honestly. The headline figure understates the real effect because the "best tuned scratch" we highlighted used a larger network.

**Fix:** replace headline with matched-arch number. State both numbers explicitly — "matched architecture 35%, unrestricted-architecture-with-best-scratch 28%" — so reviewers see we ran the audit.

### Verified LIKELY WRONG in a side document

- **Walrus data-contamination claim in `WALRUS_PLAN.md`.** I wrote "MHD_64 was in Walrus's 19-scenario pretraining corpus — test on M_A=0.7 is effectively in-distribution." Walrus abstract says the model is "pretrained on nineteen diverse scenarios spanning ... plasma physics." It does NOT say Walrus trained on MHD_64's test split. The Well uses canonical 80/10/10 train/val/test splits (Polymathic standard). Walrus almost certainly respected that — which means evaluating Walrus on MHD_64 test IS a legit held-out test for it, same as for our FNO.
  What remains true: Walrus had direct access to MHD_64 *train* (both M_A=0.7 and M_A=2.0) during pretraining, while our FNO pretrained on M_A=2.0 train only. So the head-to-head is "Walrus: full target-family pretrain + held-out test" vs "Ours: half target-family pretrain + fine-tune + held-out test." That's still an uneven comparison, but it's *not* data contamination. **My plan-doc framing was wrong and Review A correctly caught it.** Since Walrus is out of scope for this workshop paper anyway (we already decided that), the fix is to correct `WALRUS_PLAN.md` for the future.

### The novelty claim — partially preempted, narrower claim survives

Subramanian 2023 studied OOD physics parameters and mixed-physics pretraining, but did **not** run a specific negative-transfer experiment where a plausible-architecture, same-scale pretrain on *wrong physics* actively hurts the target. Our NS-control result (23% penalty vs scratch) is a sharper version of the physics-specificity claim. The correct framing is:

- **Not novel:** "pretraining physics-specificity matters" (Subramanian 2023 showed OOD physics parameters hurt transfer).
- **Our narrower novel contribution:** quantified negative transfer from a non-MHD pretrain on MHD target, where non-MHD here means different governing equations (rather than just different parameters of the same equations). Caveat: the negative transfer is confounded with the dataset-structure differences (BC, Δt, SPH) noted above, so this is a "negative-transfer-with-real-confounds" case study, not a clean physics-specificity proof.

The paper recoverably reframed around this narrower claim + the novel long-horizon failure-mode observation (pretraining flips failure mode from smoothing to inflation, which is not in the existing literature) would be honest and defensible.

## Part B — the single highest-ROI fix (free compute, decides whether Finding 1 is real)

Review B's final reframe is correct: **the novel finding is not the 28% data efficiency win. It's that pretraining changes the *character* of the failure mode — short-horizon wins with long-horizon catastrophic inflation.** That finding is not in McCabe, Lippe, Brandstetter, or Subramanian.

The control that decides whether this finding is real vs artifact is cheap: **load `runs/pretrain/best.pt`, roll it forward on M_A=0.7 test data with zero fine-tuning, and see what fails mode it exhibits.** This is inference-only on a checkpoint we already have, can run on M1 Max MPS, no GPU rental.

Four possible outcomes (all informative):

| Outcome | Interpretation |
|---|---|
| zero-FT inflates, FT inflates similarly | "Models evaluated OOD inflate" — known phenomenon, not novel. Finding 1 dies. |
| zero-FT smooths, FT inflates | **Fine-tuning introduces the instability.** Novel, unexpected, mechanistically interesting. Best case for the paper. |
| zero-FT inflates, FT smooths | Fine-tuning is *stabilizing* something that was unstable. Different novel result. |
| zero-FT smooths, FT smooths (both stable) | Finding 1 is noise from our specific FT setup, not a real phenomenon. |

We will run this experiment before any other paper revisions. Results determine the paper's central claim.

## Part C — triage: fatal, fixable, minor

**Fatal (paper-killing unless addressed):**
1. Single pretrain seed → run ≥2 more for each condition (6 additional pretraining runs) OR reframe as "case study" with explicit statistical disclaimer
2. Validation-set headline → switch to test-set numbers (already have eval_full outputs)
3. OOD-rollout control not run → run it before claiming the failure-mode finding is novel (free)
4. hidden=48 vs 64 confound → use matched-architecture numbers (stronger AND more honest: 35% not 28%)
5. Missing refs (McCabe 2023, Subramanian 2023, Lippe 2023) → add, reframe novelty accordingly

**Fixable with text edits:**
6. Boldyrev vs GS95 reference slope on fig3 → replace or add both, fix citations
7. Supernova label ("RHD" → "hydrodynamics with radiative cooling, SPH, open-BC")
8. NS control confounds (BC, Δt, channel count, SPH) → declare in limitations; propose within-Well hydro-only controls (Rayleigh-Benard, turbulence_gravity_cooling) as cleaner follow-ups
9. Tokamak framing → drop
10. Equipartition theoretical lines → measure empirically in source and target data, replace theoretical 1/M_A² overlay
11. ∇·B floor → measure on ground-truth test data and report
12. "42% gap" framing → replace with absolute deltas and consistent effect-size definition
13. Walrus contamination claim → correct in WALRUS_PLAN.md (Walrus is out of scope for this paper, but the document should be accurate)

**Minor (polish for final version):**
14. Full 2D E(k_∥, k_⊥) instead of single slice — regenerate with existing data
15. Conservation floor annotation on fig6b — measure and annotate
16. Compute-matched scratch baseline (train scratch for equivalent gradient-step budget) — extra ablation, ~1 GPU-hr
17. Per-seed dots overlaid on bar chart — cosmetic
18. Architecture scale ablation (e.g. 5M and 50M) — out of scope for this paper, flag for Paper 2

## Part D — action plan with ROI

**Zero new compute, ~4 hours of editing work:**
- Task 24: run OOD rollout control on M1 Max (free) — decides paper narrative
- Task 25: replace headline with matched-arch 35% numbers (existing data)
- Task 26: add missing refs, narrow novelty claim in text
- Task 27: fix physics overreach (drop tokamak, fix slope reference, fix labels)
- Report test-set VRMSE alongside val (evals2 data already present)
- Measure empirical E_B/E_K in source + target + report on fig5
- Measure ∇·B floor on truth + annotate fig6b

**Small compute, ~$10 total, 6-12 GPU-hours:**
- Run 2 more MHD pretrain seeds and 2 more NS pretrain seeds (needs Vast box briefly)
- Independently LR-sweep the NS fine-tune to remove HP-unfairness objection

**Larger compute, days of work (defer to Paper 2):**
- Within-Well hydro-only controls (Rayleigh-Benard, turbulence_gravity_cooling) as cleaner physics-specificity substrates
- Compute-matched scratch baseline with full gradient-step parity
- Architecture scale ablation
- Walrus / Athena++ cross-framework evaluation

The near-term plan: do all the zero-compute work now (this session), run the OOD control on M1 Max now (free), commit, and DECIDE based on OOD outcome whether to rent ~$10 of GPU for seeds.

## Part E — ELI5 understanding

The two reviewers agree on a core diagnosis even though they're stylistically very different:

> "You showed that training the model on MHD first makes it better on MHD at low data. That part is real. You also showed that training it on something non-MHD makes it worse. Also real, but 'non-MHD' in your experiment is also 'different boundaries / different timestep / different numerics,' so you can't fully blame the physics. Your bigger claim — that pretraining flips how the model fails at long horizons from 'gets boring' to 'blows up' — is the genuinely interesting finding. But you didn't run the cheap control experiment that tells us whether that's a real pretraining effect or just 'models break on unfamiliar data.' Run it before you claim the result. And you've got a subtle unfairness baked into the headline: the 'best tuned baseline' is a bigger network than the fine-tuned model. When you compare same-size models the effect is actually larger, so the fix strengthens the paper, not weakens it."

## Part F — rigorous understanding

The paper has three candidate claims. In order of how strong the evidence currently is:

**Strongest claim (survives review with matched-architecture fix):**
Fine-tuning from an in-domain pretrained checkpoint gives a **35% reduction** in 1%-data target-regime best validation VRMSE vs a matched-architecture scratch baseline (hidden=48 both sides). Test-set reproduction pending but should be straightforward given our existing `evals2/` outputs. Limitations: single pretrain seed, same dataset family, single architecture scale.

**Middle-strength claim (survives review with caveats):**
A non-MHD pretrain (supernova_explosion_64) on the same architecture **hurts** the target by ~23%, but the control has multiple non-physics confounds (open BC vs periodic, variable Δt vs fixed, SPH origin vs native Eulerian, 6 channels vs 7, 7-orders-of-magnitude dynamic range). The result demonstrates negative transfer; it does not cleanly isolate physics-as-cause from these structural confounds. The narrower claim that survives is: "an architecture-matched non-MHD pretrain can actively hurt downstream target performance" (which is a useful data point against a generic 'any-pretraining-is-good-regularization' prior) without claiming the cause is specifically 'wrong physics' as opposed to 'wrong data structure.'

**Potentially novel claim (status pending OOD control):**
Pretraining changes the *qualitative character* of the long-horizon rollout failure — from smoothing toward low-variance attractor (scratch 1%-data) to confidently inflating whole-spectrum energy (pretrained + FT). This failure-mode asymmetry is not in McCabe 2023 or Lippe 2023. But the claim is conditional on the OOD-rollout control (zero-FT pretrained on M_A=0.7 test): if zero-FT also inflates, the phenomenon is "OOD evaluation instability," not "pretraining-induced instability." **Run the control before writing the claim.**

The physics-interpretability observations (cascade preservation in fig3, equipartition drift in fig5, conservation violation in fig6) are suggestive but under-supported at the level of what's plotted. Each requires additional measurement or diagnostic work (empirical source/target equipartition, full 2D spectra, truth-floor for ∇·B, mode-decomposed energy) before they carry the weight the current text implies.

## Part G — what I am doing right now, in order

1. Execute OOD-rollout control on M1 Max MPS. Free. Results inform the paper's central claim.
2. Regenerate figures using matched-architecture scratch baseline (h=48). Numbers are already in hp_summary.json.
3. Pull test-set VRMSE from `evals2/*/results.json` and swap into headline figures + text.
4. Update references.bib with the 4 confirmed-real missing citations.
5. Rewrite Section 2 (related work) and Section 5 (discussion) to narrow novelty claim in light of Subramanian 2023.
6. Fix Figure 3 reference slope: plot both GS95 and Boldyrev, cite Schekochihin 2022 review as modern canonical.
7. Fix "RHD" → correct label.
8. Drop tokamak framing. Rewrite intro to target diffuse-ISM turbulence (honest) as the physics motivation.
9. Commit per-step; push after each milestone.

Then — separately — decide whether to run the 6 additional pretraining seeds on a cheap GPU based on whether the OOD control result justifies the spend.
