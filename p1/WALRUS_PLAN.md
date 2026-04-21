# Walrus comparison — rigorous execution plan

## Reframing forced by Phase 1

The single most important Phase 1 finding is that **MHD_64 was in Walrus's 19-scenario pretraining corpus**. Our originally-planned experiment — "Walrus zero-shot vs our narrow pretrain on unseen MHD" — was based on a false premise. Walrus has seen M_A=0.7 trajectories during pretraining.

This **kills** a naive "can Walrus generalize to our data?" framing, but it **does not** kill the experiment. It changes the question:

- **Old framing (wrong):** Does Walrus generalize from its broad pretrain to a specific MHD regime it hasn't seen?
- **New framing:** Given that Walrus was massively pretrained on 19 physical scenarios including MHD_64, does *our narrow M_A=2.0 → M_A=0.7 pretrain* deliver comparable or better quality on M_A=0.7 than Walrus's broad pretrain?

The new framing is scientifically cleaner, more defensible, and more relevant to the "do we need plasma-specific foundation models" question the paper is actually about.

## The paper claim we want to support

> "A narrow single-regime pretrain on 3,960 MHD windows produces an FNO3D that matches or exceeds a 1.3B-parameter transformer foundation model (Walrus) pretrained on 19 cross-domain scenarios — on M_A=0.7 next-step VRMSE, on cascade preservation, and on short-horizon rollout — at ~1000× less pretrain compute. For the fusion-like sub-Alfvénic MHD target, narrow specialization beats broad scale."

or the inverse, if the data says so:

> "Walrus's broad pretrain dominates our narrow pretrain on target-regime accuracy, even though Walrus was not regime-specialized. Scale appears to substitute for specialization in this domain."

The experiment is a falsification either way — we don't know which answer we'll get.

## Infrastructure plan

### Instance choice

**Decision: start with RTX 4090 24GB; upgrade to A6000 48GB only if needed.**

- Zero-shot Walrus inference fits in ~15–20 GB fp16 with gradient checkpointing → 24 GB is enough for inference.
- Fine-tuning 1.3B params requires gradients + Adam state → 35–45 GB → needs A6000 48 GB.

Phased GPU rental:
1. **Phase 2 (zero-shot inference):** RTX 4090 24GB, $0.40/hr, US-East or US-Central for good HF peering. Budget 2 hours.
2. **Phase 4 (fine-tuning):** only rent A6000 48GB if Phase 2 revealed the zero-shot baseline is interesting enough to fine-tune. $0.70/hr, budget 4 hours.

Total estimated compute: **~$4 for zero-shot + ~$3 for fine-tuning = $7**. Well under the $20 budget.

### Region choice

**US-East or US-Central** — based on this session's empirical finding that Hungary and Texas hosts got 80 MB/s to HuggingFace while Spain hosts got 16 MB/s. Whatever the reason (ISP peering, CDN edge), EU hosts are slow. Pick a US host with advertised CUDA 12.4+ and inet_down ≥ 1 Gbps.

## Phase 2 — GPU setup + sanity check (RTX 4090, ~1 hr)

1. Rent RTX 4090 in US region, min 100 GB disk, CUDA 12.4+.
2. Install Walrus repo: `pip install -e .` (the_well is an upstream dep; cu version matters — need torch 2.5.1 matching Walrus pyproject, not our current 2.11.0).
3. **Compatibility risk:** Walrus pins torch==2.5.1. Our FNO stack uses torch 2.11. Don't mix — use a fresh venv for Walrus work. Document.
4. Download Walrus weights from HF: `walrus.pt` (5.1 GB) + `extended_config.yaml`.
5. Download MHD_64 test M_A=0.7 files (3.4 GB local subset — can rsync from laptop or download fresh from HF).
6. **Sanity check:** run notebook's Part 2 (non-Well data synthetic example) to verify model loads and forward pass works on dummy data at correct shape.
7. Write minimal inference script `walrus_comparison/walrus_infer.py` that:
   - loads Walrus
   - loads one MHD_64 M_A=0.7 trajectory via our existing HDF5 loader
   - constructs the proper batch dict (field_indices, metadata, etc.)
   - runs a single forward pass
   - verifies output shape and scale
8. **Do not proceed to Phase 3 until the sanity check passes.** Sanity budget: 1 hour of debugging. If it doesn't work in 1 hour, stop and diagnose before burning more GPU time.

### Phase 2 exit criteria

- Walrus model loads on GPU with no shape mismatches.
- Forward pass on one real 64³ × T=6 × 7 input produces finite output of shape [1, T_out, 64, 64, 64, 7].
- Output values are in physically reasonable range (density ~1, |B| ~1, |v| ~0.4).

## Phase 3 — zero-shot inference + metric suite (~2 hrs GPU)

On 15 M_A=0.7 test trajectories (same sample as our FNO evaluation for direct comparability — match by dataset-index not by content):

1. **Next-step prediction (T=1 lookahead):** fresh VRMSE number per field per trajectory.
2. **Autoregressive rollout** at K ∈ {1, 5, 10, 25, 50}. We skip K=100 because: (a) our FNO eval was K=50, (b) MHD_64 trajectories are only 100 steps so K=100 gives ~0 test windows per trajectory.
3. **Physics extraction** (same as `p1/extract_physics.py`): per-step mass, E_B, E_K, E_ratio, divB_norm, variance. Save as `evals/walrus/zero_shot/<trajectory>/conservation.npz`.
4. **Isotropic spectrum** per field per trajectory at step 1 and step 10.
5. **Synthetic Alfvén wave probe:** reuse our `p1/extract_wave_probes.py` script with Walrus as the forward model. This should work out of the box since the wave probe only needs a forward model at 64³.
6. **Anisotropic spectrum:** compute at step 1 using `aniso_spectrum` from our existing pipeline.

### What we explicitly skip for time

- **Scaling invariance test**: Walrus's `revin` normalization likely makes this trivial/uninteresting (model normalizes internally). Skip unless a reviewer asks.
- **Multi-step rollout at K=100**: not enough ground-truth data.

### Output convention

Store Walrus results in the SAME format as our FNO `evals/physics/<config>/` so our existing plotters work out of the box:
- `evals/walrus/zero_shot/<trajectory_i>/conservation.npz`
- `evals/walrus/zero_shot/rollout_vrmse_full.npz`  (shape: n_traj × K)
- `evals/walrus/zero_shot/aniso_step1.npz`
- `evals/walrus/zero_shot/field_snapshots.npz`
- `evals/walrus/wave_probes/alfven.npz`, `magsonic.npz`

## Phase 4 — light fine-tuning (A6000 48GB, ~2 hrs if we do it)

**Decision tree before renting A6000:**
- If Phase 3 shows Walrus zero-shot already matches or beats our ft_01 (0.301), fine-tuning is less informative — Walrus wins without any target-domain adaptation. Document that and skip fine-tuning.
- If Phase 3 shows Walrus zero-shot is between our scratch baseline (0.419) and our ft_01 (0.301), fine-tuning is interesting — does light FT close the gap?
- If Walrus zero-shot is worse than our scratch baseline (unlikely given the in-distribution caveat, but possible), skip fine-tuning and report as "broad pretrain loses to narrow pretrain."

### If we fine-tune

- Unfreeze the last 3 processor blocks (of 12) + decoder. All other params frozen.
- Data: 1% M_A=0.7 train, same subset as our ft_01 (seed=0 subsample).
- Optimizer: AdamW, lr=1e-4 (lower than Walrus's pretrain lr=2e-4 since we're fine-tuning a tiny head).
- Epochs: 10 (vs our FNO's 40 — Walrus's pretrain is much further along).
- Seeds: 1 for budget, 3 if time permits.

## Phase 5 — head-to-head figures (local, no GPU, after rsync)

After rsyncing Walrus eval outputs locally, reuse our existing plotters:

- `p1/figures/p1_walrus_vs_fno_vrmse.png` — 5-bar chart: scratch_tuned, NS→ft, MHD→ft, Walrus zero-shot, Walrus-FT
- `p1/figures/p1_walrus_vs_fno_cascade.png` — cascade slopes with Walrus added to the existing 4 models
- `p1/figures/p1_walrus_vs_fno_rollout.png` — rollout comparison with Walrus added
- `p1/figures/p1_walrus_vs_fno_conservation.png` — conservation panels with Walrus

Most figures are extensions of existing `plot_*.py` scripts — just adding a Walrus column to existing multi-model plots.

## Phase 6 — writeup + cleanup

- `p1/Walrus_comparison_results.md` — narrative summary of what Walrus looked like on each metric, head-to-head with our models, failure modes observed.
- Destroy Walrus instance. Document total compute cost.
- Update `P1_v3_SUMMARY.md` Section 4.4 with Walrus numbers.

## Execution checklist (copy to the actual run log)

- [ ] **Pre-GPU:** verify local has MHD_64 M_A=0.7 test data (3.4 GB) — already done ✓
- [ ] **Rent:** US-region RTX 4090, 100+ GB disk, CUDA 12.4+, reliability ≥ 0.98
- [ ] **Setup:** fresh venv, install Walrus + deps (torch 2.5.1 pin!)
- [ ] **Download:** walrus.pt + extended_config.yaml from HF (5.1 GB)
- [ ] **Sanity:** run demo notebook Part 2 synthetic example end-to-end
- [ ] **Loader:** write MHD_64 → Walrus batch converter. Verify field_indices, metadata, BCs
- [ ] **Single forward:** one real M_A=0.7 trajectory, get output, check shape + scale
- [ ] **Zero-shot inference:** 15 trajectories × rollout K=50 + step-1 + spectra
- [ ] **Decision point:** is fine-tuning worth the A6000 cost?
- [ ] **(Optional) Fine-tune:** unfreeze last 3 blocks + decoder, 1% data, 10 epochs
- [ ] **Rsync results:** evals/walrus/*.npz pulled to laptop
- [ ] **Figures:** extend existing plotters
- [ ] **Destroy instance**
- [ ] **Writeup:** Walrus_comparison_results.md + Section 4.4 of paper

## Budget hard cap

$25 total. If we exceed this, stop and reassess scope.

## Known unknowns I'll document as we hit them

- Whether `rollout_model` function exists in the distribution or is notebook-only
- Whether our 64³ × T=6 input is too big for RTX 4090 at fp16 (might need gradient checkpointing, smaller batches)
- Whether jitter_patches needs to be disabled for reproducibility in inference
- Whether our fine-tune decision to unfreeze last 3 blocks is enough to move the needle
- Whether Walrus's `predict_delta=True` matches our FNO's direct-next-state prediction (these are different objectives — may need translation)

## What this plan explicitly does NOT cover (for scope discipline)

- **Training Walrus from scratch on any of our data.** Massively out of scope; we only use the pretrained checkpoint.
- **Comparing at other data fractions beyond 1%.** Our headline is 1%; fine-tuning at 10%/100% is nice-to-have but not required.
- **Reproducing Walrus's paper results.** We trust the checkpoint is what they released.
- **Handling Walrus architecture modifications** (patch size sweep, adaptive compute budget). Single default configuration throughout.

## Exit conditions (move to paper-writing regardless)

1. **Walrus beats our MHD→ft clearly** (<0.28 VRMSE) → paper framing becomes "scale dominates narrow specialization"
2. **Walrus loses to our MHD→ft clearly** (>0.35 VRMSE) → paper framing becomes "narrow specialization beats scale at 1000× less compute"
3. **Walrus matches our MHD→ft within ~0.02** → paper framing becomes "comparable quality at different compute points; depends on downstream task regime"
4. **Walrus setup fails after 1-hour debug budget** → skip Walrus entirely, document as a limitation in the paper, frame our result as standalone

Whichever branch we land on, we commit the eval outputs and move to the paper.
