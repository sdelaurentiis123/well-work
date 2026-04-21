# Walrus schema mapping — MHD_64 → Walrus input

Discovered from `polymathic-ai/walrus` repo + HF checkpoint config (`extended_config.yaml`), demo notebook `walrus_example_1_RunningWalrus.ipynb`, and `isotropic_model.py` source.

## Checkpoint inventory (HuggingFace `polymathic-ai/walrus`)

| file | size |
|---|---|
| `walrus.pt` | 5.14 GB (fp32 state dict, full 1.3B params) |
| `walrus.safetensors` | 5.14 GB (same weights, safetensors format) |
| `extended_config.yaml` | <1 KB but load-bearing |
| `README.md` | model card |

No quantized or distilled variants released. Single checkpoint.

## Model specification (from `configs/model/isotropic_model.yaml`)

- Class: `walrus.models.IsotropicModel`
- hidden_dim = 768
- processor_blocks = 12 (factorized space + time attention)
- projection_dim = 96
- intermediate_dim = 192
- groups = 12
- max_d = 3 (force all data to 3D)
- jitter_patches = True (random translation of patch grid at inference — on by default)
- causal_in_time = False (bidirectional in time)
- Parameters: ~1.3 B total

## Field-to-index vocabulary (from `config.data.field_index_map_override`)

Our MHD_64's 7 fields map cleanly into Walrus's pretrain vocabulary — no novel fields, no zero-padding needed:

| our channel | Walrus field name | index |
|---|---|---|
| density    | `density`          | 28 |
| B_x        | `magnetic_field_x` | 39 |
| B_y        | `magnetic_field_y` | 40 |
| B_z        | `magnetic_field_z` | 41 |
| v_x        | `velocity_x`       | 4 |
| v_y        | `velocity_y`       | 5 |
| v_z        | `velocity_z`       | 6 |

All 7 fields were in Walrus's training corpus (MHD_64 itself was one of the 19 pretraining scenarios). Walrus has *seen* MHD_64 during pretraining.

**Critical implication for "zero-shot":** Walrus's "zero-shot on M_A=0.7" is really "test on a subset of its own pretrain distribution." It is NOT the same kind of zero-shot as our FNO3D, which never saw M_A=0.7 at any point in its MHD→MHD pretrain. Document this distinction in the paper.

## Input tensor signature (from notebook cells 24-25)

Model expects a dict with:

```
input_fields    float    [B, T_in,  H, W, D, C_var]
output_fields   float    [B, T_out, H, W, D, C_var]
constant_fields float    [B, H, W, D, C_con]           # C_con can be 0
boundary_conditions int  [B, 3, 2]                      # 0=wall, 1=open, 2=periodic
padded_field_mask bool   [C_var]                        # True = real field, False = zero-padded
field_indices    int     [C_var + C_con]                # integer indices from the map above
metadata         WellMetadata                            # required for formatters
```

For MHD_64 M_A=0.7:
- B = 1 (inference one trajectory at a time to fit in VRAM)
- T_in = 6 (history length — notebook uses T_in=6 for the `revin` normalization)
- T_out = 10–100 depending on rollout length requested
- H = W = D = 64
- C_var = 7
- C_con = 0
- boundary_conditions = `[[[2,2],[2,2],[2,2]]]` (periodic on all 3 axes)
- padded_field_mask = `[True]*7`
- field_indices = `[28, 39, 40, 41, 4, 5, 6]`
- metadata: need to construct `WellMetadata(dataset_name="MHD_64", n_spatial_dims=3, field_names={0:['density'], 1:['magnetic_field','velocity'], 2:[]}, ...)`

## Temporal window

Walrus expects T_in = 6 history frames as the prompt for next-step prediction. Our MHD_64 trajectories have 100 timesteps, so we can pull a window `[t, t+1, ..., t+5]` as `input_fields` and `[t+6, ..., t+6+T_out-1]` as `output_fields` for ground-truth comparison.

Our FNO evaluation used T_in=1 (single frame input). **Comparability note:** Walrus inherently has 6× more context per prediction than our FNO3D did. The head-to-head is not architecture-matched — it's "pretrain-scale-matched within each architecture's native operating mode." Flag this in the paper Section 4.4.

## Normalization

`SamplewiseRevNormalization` via `revin` — applied inside the forward pass. The model handles all normalization internally; we feed it raw physical units.

## Patch size / internal resolution

Walrus uses `variable_downsample=True` encoder stride modulation to target internal resolution ~16–17 per spatial dim in 3D. Our 64³ input will be downsampled by stride ≥ 4 in each spatial axis before processing. Output is upsampled back to 64³.

## VRAM estimate

- Weights fp32: 5.14 GB → fp16: 2.57 GB
- Attention activations for T=6, hidden=768, 64³ input, 12 blocks ≈ 10–15 GB under fp16 + activation checkpointing (controlled by `gradient_checkpointing_freq`)
- **Total inference estimate: ~15–20 GB on fp16 with grad ckpt.** Should fit on a 24 GB RTX 4090 comfortably.
- For fine-tuning (gradients + Adam state): ~35–45 GB. **Requires 48 GB A6000 or 80 GB A100.** If budget-bounded, unfreeze only the last 2–3 decoder layers.

## Zero-shot inference pipeline (derived from notebook)

```python
from walrus.models import IsotropicModel
from walrus.data.well_to_multi_transformer import ChannelsFirstWithTimeFormatter
from walrus.trainer.normalization_strat import SamplewiseRevNormalization
from hydra.utils import instantiate
from omegaconf import OmegaConf

config = OmegaConf.load("extended_config.yaml")
ckpt = torch.load("walrus.pt", map_location="cpu", weights_only=True)["app"]["model"]
n_fields = max(config.data.field_index_map_override.values()) + 1
model = instantiate(config.model, n_states=n_fields).cuda()
model.load_state_dict(ckpt); model.eval()
formatter = ChannelsFirstWithTimeFormatter()
revin = instantiate(config.trainer.revin)()

# Build batch dict as above, then:
inputs, y_ref = formatter.process_input(batch, causal_in_time=False,
                                          predict_delta=True, train=False)
y_pred, y_ref = rollout_model(model, revin, batch, formatter,
                               max_rollout_steps=K, device="cuda")
```

## Things to verify on first inference

1. `walrus.pt` loads into the `IsotropicModel` config without shape mismatches.
2. Forward pass on a single real MHD_64 window produces finite, non-NaN output of shape `[B, T_out, H, W, D, C_var]`.
3. The `padded_field_mask=[True]*7` path works — notebook's example used one masked-out field (False), but ours has no padding. Verify the full-True case.
4. Output resolution matches input (64³ in → 64³ out).
5. `rollout_model` function actually exists at `walrus.trainer.training` — the notebook imports it but didn't show its definition. Check.

## Open risks going into Phase 2

- **"Zero-shot" caveat**: MHD_64 WAS in pretraining. Our M_A=0.7 test trajectories may have been seen. Check train/valid/test split — if Polymathic's released model was trained on the **full** MHD_64 train split, there's no data leakage (we evaluate on test). If they trained on train+valid+test... flag and reconsider framing.
- **Jitter_patches at inference**: non-deterministic. May need to set seed or disable for reproducibility.
- **rollout_model signature**: might require specific metadata format. First forward pass will reveal.
- **Field ordering in channel axis**: notebook example uses `[velocity_x, velocity_y, density, blubber, velocity_z]` order (un-sorted by index). Walrus handles channel order via the `field_indices` tensor — ordering in the channel axis of `input_fields` must align with the order of integers in `field_indices`. We'll use: `[density, B_x, B_y, B_z, v_x, v_y, v_z]` in channels, and `[28, 39, 40, 41, 4, 5, 6]` in `field_indices`.
