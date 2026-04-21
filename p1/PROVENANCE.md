# P1 Full Provenance Document

Hardware, data, code, and config sufficient to reproduce every number in `p1/RESULTS.md`.

## Hardware

- **Training box:** Vast.ai instance `35331407` (host IP 69.162.77.37, rented 2026-04-20).
- **GPU:** 1× NVIDIA RTX 4090 (24 GB, CUDA 13.0, driver 580.95.05).
- **CPU/RAM/Disk:** AMD EPYC 7502 (32/64 cores leased), 64 GB RAM, 100 GB NVMe (Kingston SEDC1000BM8960G).
- **Container image:** `vastai/pytorch` (PyTorch 2.11.0 + CUDA 13 + cuDNN, Python 3.12.13 via `/venv/main`).
- **Local dev:** Apple M1 Max, PyTorch 2.11 MPS, for sanity only — all quoted numbers are from the Vast box.

## Data

- **Source:** Polymathic AI `The Well`, dataset `MHD_64`, downloaded 2026-04-20 via `the-well-download` CLI (`the_well` package v1.2.0) from the HuggingFace dataset `polymathic-ai/MHD_64`.
- **Total size on disk:** 71 GB (train 57 GB / valid 6.9 GB / test 6.9 GB), unmodified after download.
- **Physics:** Isothermal compressible MHD (Burkhart et al. 2020 ISM turbulence simulations), 64³ grid, periodic BC, 100 timesteps per trajectory. Filenames encode parameters: `MHD_Ma_<M_A>_Ms_<M_s>.hdf5`.
- **Parameter grid:** `M_A ∈ {0.7, 2.0}` × `M_s ∈ {0.5, 0.7, 1.5, 2.0, 7.0}` = 10 files per split.
- **Channels (7):** density (scalar), B_x, B_y, B_z, v_x, v_y, v_z. Shape per sample window: `(64, 64, 64, 7)`.
- **Windows per file per split:** file has `n_trajectories × (n_steps − 1)` sliding pairs. train metadata reports `n_trajectories_per_file = [5, 8, 8, 8, 8, 8, 8, 8, 8, 8]`. So train has 7623 windows; valid 990; test 990.
- **M_A split (filename-based, exact):**
  - train: M_A=0.7 has 5 files (3,663 windows); M_A=2.0 has 5 files (3,960 windows).
  - valid: M_A=0.7 has 5 files (495 windows); M_A=2.0 has 5 files (495 windows).
  - test: M_A=0.7 has 5 files (495 windows); M_A=2.0 has 5 files (495 windows).
- **Physical note:** M_A=0.7 has an imposed background field `B_0 ∥ x̂` (per-sample mean `B_x ≈ 1.0`, `B_y/B_z ≈ 0`). This is the anisotropy source and why we treat it as a "fusion-analog" — it has a real guide field.
- **Disjointness:** train / valid / test are Polymathic-curated disjoint splits (content hashes of first 1MB of `MHD_Ma_0.7_Ms_0.5.hdf5` differ across splits: `4f59…`, `129f…`, `140c…`). Within-split, the M_A=2.0 pretrain files and M_A=0.7 fine-tune files are disjoint by construction (different filenames, different parameters).

## Preprocessing

- **Normalization: none applied by our pipeline.** `WellDataset` is instantiated with `use_normalization=False` (default). Fields are delivered in simulation units: density ≈ 1 ± 0.16 (code units), |B| ≈ 1 (sub-Alfvénic Burkhart units — the mean field has amplitude 1), |v| ≈ 0.4.
- **Implication for VRMSE:** since variance is computed per-sample per-field (not using a global normalization constant), VRMSE is scale-invariant and comparable across fields and M_A regimes.
- **No domain augmentation** (flips / rotations / crops) applied.
- **Time handling:** `(T_in=1, T_out=1)` — one-step-ahead prediction on adjacent sliding-window pairs.

## Model

- **Architecture:** `the_well.benchmark.models.FNO` with `dim_in=7, dim_out=7, n_spatial_dims=3, spatial_resolution=(64,64,64), modes1=modes2=modes3=12, hidden_channels=48`.
- **Parameter count:** 18.62 M (identical across all runs; this is a single architecture).
- **Implementation:** wraps `neuraloperator==0.3.0` (upstream FNO3D). Spectral convolutions via `torch.fft.{fftn,irfftn}`.
- **Framework:** PyTorch 2.11.0, AdamW optimizer (`weight_decay=1e-5`), cosine LR schedule over the full run (`T_max = epochs × steps_per_epoch`), gradient clipping at L2 norm 1.0.

## Training config (per run)

All runs share: `bs=8`, `lr=1e-3`, `modes=12`, `hidden=48`, `workers=4`, `seed=0`, AdamW weight_decay=1e-5, cosine LR, grad clip 1.0.

The *only* differences are mode / target split / init / epochs:

| Run | Mode | Train split | `init_ckpt` | `data_frac` | Epochs |
|-----|------|------------|-------------|-------------|--------|
| `pretrain` | pretrain | train M_A=2.0 | — | 1.00 | 20 |
| `baseline` | baseline | train M_A=0.7 | — | 1.00 | 20 |
| `baseline_10` | baseline | train M_A=0.7 | — | 0.10 | 25 |
| `baseline_01` | baseline | train M_A=0.7 | — | 0.01 | 40 |
| `ft_100` | finetune | train M_A=0.7 | `runs/pretrain/best.pt` | 1.00 | 15 |
| `ft_10` | finetune | train M_A=0.7 | `runs/pretrain/best.pt` | 0.10 | 25 |
| `ft_01` | finetune | train M_A=0.7 | `runs/pretrain/best.pt` | 0.01 | 40 |

Validation: always on the `valid` split at the same M_A as training (for pretrain, M_A=2.0 valid; for all M_A=0.7 runs, M_A=0.7 valid). 495 windows.

Best checkpoint is the one with lowest `val_vrmse`.

**Data-fraction subsampling determinism:** seed=0 is set before the subsample. `random.shuffle(train_idx)` + `[:k]` is deterministic under a fixed seed, so `baseline_10` and `ft_10` train on the **same 10% subset of M_A=0.7 windows**. Likewise `baseline_01` and `ft_01`. (Verified by code: `p1/train.py:97-104`.)

## Evaluation (for `p1/evals/*/`)

- **Split:** `test` (Polymathic-curated held-out, separate from train and valid).
- **M_A filter:** 0.7 only (target regime).
- **Metrics:**
  - `val_vrmse` during training (on `valid` split, logged per-epoch).
  - Isotropic power spectrum `E(k)` for each of 7 channels, angle-averaged in 32 k-bins, with relative error `|E_pred - E_true| / (E_true + 1e-20)` per channel.
  - Autoregressive rollout drift: L2 of `||state_{t+1} - state_t||` for 20 steps (this measures self-consistency, not error vs ground truth — known weak metric).
- **Test trajectories evaluated:** 3 (first 3 M_A=0.7 test windows). **This is too few** for tight error bars — see caveats below.

## Software versions (pinned on the box)

```
torch==2.11.0+cu130
the_well==1.2.0
neuraloperator==0.3.0
tensorly-torch==0.5.0
einops==0.8.2
h5py==3.16.0
numpy==2.4.4
scipy==1.17.1
matplotlib==3.10.8
wandb==0.26.0
```

## Code

- Git: `git@github-personal:sdelaurentiis123/well-work.git`, branch `main`.
- HEAD at time of results: `f9b6d30` (P1 results + eval artifacts).
- Entry points:
  - `p1/train.py` — single-run training driver (argparse-based).
  - `p1/run_all.sh` — sequential driver for pretrain+baseline+ft_100+ft_10+ft_01.
  - `p1/run_baselines.sh` — sequential driver for baseline_10+baseline_01.
  - `p1/eval_spectral.py` — per-checkpoint spectral + rollout eval.
  - `p1/plot_data_efficiency.py` — generates the headline figure.

## Artifacts

- `runs/<name>/log.jsonl` — one record per epoch: `{epoch, train_vrmse, val_vrmse, time_s}`.
- `runs/<name>/best.pt` — best-val checkpoint (state_dict + optimizer + epoch + val).
- `runs/<name>/last.pt` — final-epoch checkpoint.
- wandb project: `sdelaurentiis123-columbia-university/well-work-p1` (6 runs logged).
- `evals/<name>/{results.json,spectra.png,rollout.png}` — per-ckpt eval outputs.

## Known caveats (declared, not discovered)

1. **Single seed per configuration.** All 7 runs used `seed=0`. The 45% gap at 1% data is almost certainly real given its magnitude, but the 7.2% gap at 10% data and the 1.4% gap at 100% data could in principle be within seed variance. **Addressed by the seed-rerun experiment below.**
2. **n_traj=3 for spectral metrics.** Too few for confidence intervals on spectral error.
3. **Rollout drift ≠ rollout error.** The metric used in `evals/` is autoregressive L2 self-drift, not error vs ground truth rollouts. A proper multi-step evaluation requires reconstructing ground-truth rollouts from consecutive test windows.
4. **B_y / B_z relative-spectrum error is dominated by near-zero denominators** in the sub-Alfvénic regime. Density / B_x / v_{xyz} are the load-bearing spectral metrics.
5. **Same-physics transfer.** Source and target both compressible isothermal MHD, differing only in M_A. The "fusion-analog" framing is a proxy; real fusion MHD has different physics (low-β, toroidal, kinetic effects).

All of these are called out in `p1/RESULTS.md` "Limitations / next steps."
