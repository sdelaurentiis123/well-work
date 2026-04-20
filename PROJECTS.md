# Plasma turbulence × foundation models — project sketches

Context: PhD work. Ecosystem is Polymathic (The Well + Walrus + Overtone + MPP) plus
TORAX (differentiable tokamak transport in JAX). MHD and fusion turbulence data are
the main currency.

Each project is sized so the **first milestone fits in a week on a workstation or
one GPU**, and scales up with cluster access. Rank ordered by research-return-per-effort.

---

## P1 — Transfer study: Well-MHD pretrain → fusion-relevant MHD fine-tune
**One-liner.** Does pretraining on astrophysical compressible MHD (Well `MHD_256`,
`MHD_64`) actually help on fusion-relevant MHD, or is the inductive bias in the
wrong regime?

**Why it's interesting.** The Well's MHD data is ISM turbulence (subsonic→supersonic,
sub-/super-Alfvénic, periodic box). Fusion MHD lives in a totally different corner:
low-β, strongly magnetized, toroidal geometry, anisotropic. If transfer works, the
Well becomes a legitimate pretraining corpus for fusion surrogates. If it doesn't,
the *failure modes* are a paper by themselves (where does the prior help, where does
it hurt, which fields / scales are preserved).

**Concrete plan.**
1. Baseline: FNO and a small ViT-PDE trained from scratch on a fusion-relevant MHD
   dataset. Two candidates for fine-tuning targets:
   - BOUT++ / JOREK public reduced-MHD ELM or edge-turbulence dumps.
   - Self-generated: Athena or Dedalus sims of a slab/annulus with a strong guide
     field, scanning β and η.
2. Pretrain the same architecture on Well `MHD_64` (cheap, 100 traj).
3. Fine-tune on 1%, 10%, 100% of the fusion target set. Plot data-efficiency curves.
4. Spectral analysis: where in k-space does the pretrained model do better/worse?
   (Power spectrum, anisotropy ratio k_∥/k_⊥.)

**First milestone (week).** Baseline FNO on `MHD_64` trains and hits a reasonable
power-spectrum match. No fine-tuning yet.

**Risk.** If you can't source a fusion-MHD dataset, fall back to a synthetic anisotropic
MHD dataset made with Dedalus — still a real result.

---

## P2 — Walrus as drop-in turbulent-flux surrogate inside TORAX
**One-liner.** Replace / augment QLKNN with a Walrus-style transformer that eats a
short history of (T_i, T_e, n_e, q, s) profiles and predicts turbulent heat/particle
fluxes.

**Why it's interesting.** TORAX's default turbulent transport model (QLKNN_7_11) is a
small MLP fit to gyrokinetic QuaLiKiz outputs — point-wise, no history, no
neighborhood. A transformer that sees radial neighborhood + short time history could
capture non-local transport, stiffness, and intermittency that QLKNN smears. And
because TORAX is fully differentiable, the whole hybrid pipeline is still end-to-end
autodiff'able — gradients flow through the surrogate for trajectory optimization and
controller design.

**Concrete plan.**
1. Generate a training set by running QuaLiKiz (or CGYRO/GENE if available) on a grid
   of (∇T, ∇n, q, s, ν*, β) points — this is the same thing QLKNN was trained on, but
   you control the sampling.
2. Train a small (<20M param) transformer to predict (χ_i, χ_e, D_e) from a radial
   patch of inputs + a few timesteps of history.
3. Wrap it in TORAX's transport model interface (there's an API for this in
   `torax/transport_model`). Run an ITER rampup config and compare to QLKNN on:
   predicted stored energy trajectory, wall-clock, smoothness of χ profiles.
4. Paper angle: "neural transport closure with memory" + ablation on how much
   history/neighborhood matters.

**First milestone (week).** Toy version: train the transformer on a *synthetic*
closure (say χ = χ_GB × f(∇T/T)) and show it plugs into TORAX and reproduces the
analytic run. De-risks the infra before you touch real gyrokinetic data.

**Stretch.** Pretrain on Walrus weights then fine-tune — does large-scale physics
pretraining transfer to a 0D→scalar flux regression task? (Probably via the spatial
encoder, not the whole model.)

---

## P3 — Overtone-style cyclic patching for tokamak MHD rollouts
**One-liner.** Port Overtone's CSM/CKM onto a patch-based transformer trained on
Well MHD and measure whether patch-cycling fixes the grid-lattice spectral artifacts
that show up in long fusion-MHD rollouts.

**Why it's interesting.** Long-horizon rollout stability is the blocking issue for
practical use of learned emulators in fusion (you want minutes of plasma time, not
microseconds). Overtone's Mar 2026 result is that fixed-patch transformers accumulate
structured errors at lattice frequencies, and cycling the patch size breaks that up.
Nobody has tested this specifically on plasma/MHD rollouts yet.

**Concrete plan.**
1. Reproduce Overtone's result on `MHD_64` — one model, three baselines at
   patch sizes 4/8/16, Overtone CSM trained once. Check: compute-accuracy Pareto
   curve, spectral artifact amplitude at k = 2π/patch.
2. Extend to 3D MHD_256 if compute allows, or to an anisotropic fusion-like dataset
   from P1.
3. Publish a short note: "Cyclic patch modulation for MHD emulation" — small scope,
   fast turnaround, clean figures.

**First milestone (week).** Get the Overtone code
(https://github.com/PolymathicAI — haven't confirmed Overtone is a separate repo yet,
may live inside walrus/) running on a baseline patch-ViT you train on MHD_64.

**Why it's fast.** Overtone is architecture-agnostic and small — CSM is a few-line
change to the patch embedding. The experiment design is already done; you just swap
the dataset.

---

## P4 — TORAX-native pretraining corpus for small plasma foundation models
**One-liner.** Use TORAX's speed + differentiability to build a big parameter-swept
corpus of tokamak transport trajectories, train a small (<100M) plasma-specific
foundation model, benchmark it against Walrus + MPP.

**Why it's interesting.** Walrus is 1.3B params across everything. A narrow,
plasma-specific foundation model might beat it on fusion tasks with 10× less compute.
And TORAX can emit 10⁴–10⁵ trajectories cheaply — each ITER-hybrid rampup runs in
seconds. Output: a HuggingFace dataset + a small open-weight model.

**Concrete plan.**
1. Parameter-sweep TORAX: vary I_p(t), n_e(t), P_aux, geometry (CHEASE eq basis set),
   pedestal parameters, impurity levels. 10k trajectories × ~100 timesteps.
2. Build a WellDataset-compatible loader (HDF5 with the Well's tensor layout) so the
   data slots into any Polymathic training script.
3. Pretrain a Walrus-architecture model at 50M params on this corpus alone, and a
   second version initialized from Walrus-1.3B weights and fine-tuned.
4. Eval tasks: predict T_i/T_e profile evolution, predict disruption-like stiff
   transitions, predict sensitivity to control inputs. Compare to Walrus zero-shot,
   MPP zero-shot, Walrus fine-tuned.

**First milestone (week).** Sweep 100 TORAX runs, dump to HDF5, verify WellDataset
loads them and a baseline trains one epoch.

**Why it's valuable.** Even the *dataset* is a publication — "ToraxBench" / "TBench":
a standardized fusion transport benchmark for ML surrogates. This is the kind of
artifact a PhD is remembered for.

**Stretch.** Use TORAX's differentiability for adversarial sampling: take gradients
of the surrogate's error w.r.t. scenario parameters to find the hardest trajectories,
add them to the corpus, retrain. Self-improving benchmark.

---

## P5 (exploratory, low-priority) — Gyrokinetic-to-transport distillation
**One-liner.** Train a neural operator that maps a gyrokinetic configuration
(profiles + geometry) to a full turbulent flux *profile*, not just a point-wise Q —
distilling a gyrokinetic code (GENE/CGYRO) into something that runs inside TORAX at
TORAX speed.

This is P2 with a much bigger ambition and a much harder data-generation problem
(GENE sims cost 10³–10⁴ core-hours each). Flag for later when you have cluster
access and a collaborator with GENE data.

---

## Cross-cutting infra to set up first
Independent of which project you pick, these pay off everywhere:

1. `pip install the_well` into `toraxvenv` (or a separate venv — check PyTorch/JAX
   coexistence). Stream `MHD_64` from HF to validate the loader end-to-end.
2. Pull `walrus` repo, get the HF checkpoint loading, run inference on one MHD_64
   trajectory. Zero-shot rollout is a free baseline for any downstream project.
3. Decide PyTorch vs JAX story. TORAX is JAX; Walrus is PyTorch. The Well loader is
   torch-native. Cleanest path: keep TORAX data generation in JAX, train ML in
   PyTorch, move arrays across via numpy.

---

## Recommended order
P1 **or** P3 first (both are self-contained, 1–2 week papers-worth of work, both are
de-risked). Then P2 once the transport surrogate infra is in place. P4 is the
ambitious thesis-chapter version; start collecting infrastructure for it in parallel.

## Open questions to pin down before committing
- What fusion-MHD datasets do you actually have access to? (Answers change P1 scope
  drastically.)
- What's your compute budget? (P4 in full needs multi-GPU; P3 fits on one A100.)
- Are you collaborating with anyone on GENE/CGYRO runs? (Gates P5 and the
  serious version of P2.)
