"""Rollout adapters for FNO and Walrus.

Adapter interface (per src/extract_physics.py:RolloutAdapter):
    n_history: int
    rollout(history, K) -> ndarray (K, 7, 64, 64, 64)
"""
from __future__ import annotations
import copy
from pathlib import Path
import numpy as np
import torch


# ============================================================================
# FNO adapter — single-frame Markov, autoregressive loop
# ============================================================================
class FNOAdapter:
    """Wraps a Paper 1 FNO3D checkpoint as a rollout adapter."""

    n_history = 1

    def __init__(self, ckpt_path: str | Path, device: torch.device):
        from the_well.benchmark.models import FNO

        self.device = device
        self.model = FNO(
            dim_in=7,
            dim_out=7,
            n_spatial_dims=3,
            spatial_resolution=(64, 64, 64),
            modes1=12,
            modes2=12,
            modes3=12,
            hidden_channels=48,
        ).to(device)
        sd = torch.load(ckpt_path, map_location=device, weights_only=True)
        sd = sd["model"] if "model" in sd else sd
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

    @torch.no_grad()
    def rollout(self, history: np.ndarray, K: int) -> np.ndarray:
        # history shape (1, 7, 64, 64, 64); take the last frame as starting state.
        state = torch.tensor(history[-1]).unsqueeze(0).to(self.device)
        out = np.zeros((K, 7, 64, 64, 64), dtype=np.float32)
        for step in range(K):
            pred = self.model(state)  # (1, 7, 64, 64, 64)
            out[step] = pred[0].cpu().numpy()
            state = pred
        return out


# ============================================================================
# Walrus adapter — 10-frame conditioning, native rollout
# ============================================================================
class WalrusAdapter:
    """Wraps Walrus 1.3B as a rollout adapter.

    Uses the standalone rollout_model() pattern from
    walrus/demo_notebooks/walrus_example_1_RunningWalrus.ipynb so we don't have to
    instantiate the LightningModule.

    Construction requires:
      - checkpoint_path: path to walrus.pt (the_well download)
      - config_path: path to extended_config.yaml (the_well download)
      - well_base_path: path to MHD_64 dataset on disk (Walrus's data module needs
        it to compute field stats during instantiation)
    """

    n_history = 10

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        well_base_path: str | Path,
        device: torch.device,
    ):
        from omegaconf import OmegaConf, open_dict
        from hydra.utils import instantiate
        from walrus.data.well_to_multi_transformer import (
            ChannelsFirstWithTimeFormatter,
        )

        self.device = device
        config = OmegaConf.load(config_path)

        # Strip non-Well datasets that reference paths that don't exist on our system
        with open_dict(config):
            mod = config.data.module_parameters
            for k in list(mod.well_dataset_info.keys()):
                if not k.startswith("MHD"):
                    del mod.well_dataset_info[k]

        self.data_module = instantiate(
            mod,
            well_base_path=str(well_base_path),
            world_size=1,
            rank=0,
            data_workers=1,
            field_index_map_override=config.data.get(
                "field_index_map_override", {}
            ),
            prefetch_field_names=False,
        )

        # Walrus data module exposes datasets in rollout_val_datasets
        # We pick the MHD_64 dataset (should be the only one after pruning)
        self._mhd_dataset = self.data_module.rollout_val_datasets[0].sub_dsets[0]
        self.metadata = self._mhd_dataset.metadata
        self.field_to_index_map = self.data_module.train_dataset.field_to_index_map
        n_fields = max(self.field_to_index_map.values()) + 1

        # Instantiate model + load weights
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        # Lightning checkpoint structure: ckpt["app"]["model"]
        state_dict = ckpt["app"]["model"] if "app" in ckpt else ckpt
        self.model = instantiate(config.model, n_states=n_fields).to(device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.formatter = ChannelsFirstWithTimeFormatter()
        self.revin = instantiate(config.trainer.revin)()
        self.config = config

    @torch.no_grad()
    def rollout(self, history: np.ndarray, K: int) -> np.ndarray:
        """Roll out K steps from a 10-frame history.

        history: (10, 7, 64, 64, 64) — channel order [density, B_x, B_y, B_z, v_x, v_y, v_z]
        Returns: (K, 7, 64, 64, 64)
        """
        # Build a Walrus-format batch from the history. The exact batch structure
        # is determined by the MHD_64 dataset's __getitem__; we mimic that here.
        # Walrus MHD batch keys: input_fields, metadata, boundary_conditions,
        # padded_field_mask, constant_fields, output_fields.
        batch = self._build_batch_from_history(history, K)

        y_pred, _ = _standalone_rollout_model(
            model=self.model,
            revin=self.revin,
            batch=batch,
            formatter=self.formatter,
            max_rollout_steps=K,
            device=self.device,
        )
        # y_pred shape per demo: (B, T, ..., C). Need to convert back to
        # (K, 7, 64, 64, 64) with our channel order.
        y_pred = y_pred[0].cpu().numpy()  # drop batch dim -> (T, H, W, D, C)
        # Permute channels-last → channels-first: (T, H, W, D, C) → (T, C, H, W, D)
        y_pred = np.moveaxis(y_pred, -1, 1)
        # Restrict to first 7 channels (MHD) in the order Walrus stored them.
        y_pred = y_pred[:K, :7]
        # TODO: verify the channel ordering Walrus emits matches our [density,
        # B_x, B_y, B_z, v_x, v_y, v_z]. May need a permutation here.
        return y_pred.astype(np.float32)

    def _build_batch_from_history(self, history: np.ndarray, K: int):
        """Construct the dict Walrus's rollout_model expects.

        Walrus expects (per demo notebook line ~150):
          batch = {
            "input_fields":  Tensor (B, T_in, H, W, D, C),
            "metadata":      MetaData,
            "boundary_conditions": ...,
            "padded_field_mask": Tensor (C,) bool,
            "constant_fields":   Tensor (B, ?, ..., C_const),
          }

        Easiest path: pull a real batch from the dataset, then overwrite
        `input_fields` with our history. This guarantees we don't miss any
        required key / metadata / shape constraint.
        """
        # Pull one sample from the MHD dataset to get a properly-shaped template.
        loader = self.data_module.rollout_val_dataloaders()[0]
        template = next(iter(loader))

        # history shape (10, 7, 64, 64, 64) — channels-first
        # Need to transpose to channels-last and pad to Walrus's full channel set.
        # template["input_fields"].shape gives us the full (B, T_in, H, W, D, C_full).
        T_in = template["input_fields"].shape[1]
        if T_in != self.n_history:
            raise RuntimeError(
                f"Walrus T_in={T_in} but adapter.n_history={self.n_history}"
            )
        C_full = template["input_fields"].shape[-1]

        hist_chan_last = np.moveaxis(history, 1, -1)  # (10, 64, 64, 64, 7)
        # Pad along channel axis to C_full (other channels are masked out by
        # padded_field_mask).
        if C_full > 7:
            padded = np.zeros(
                (self.n_history, 64, 64, 64, C_full), dtype=np.float32
            )
            # We need to know where MHD's 7 channels live in the full C_full
            # vector. field_to_index_map gives us this — TODO refine.
            padded[..., :7] = hist_chan_last  # placeholder; will need the real index map
            hist_chan_last = padded

        template["input_fields"] = torch.tensor(hist_chan_last).unsqueeze(0)
        return template


# ============================================================================
# Standalone rollout_model (lifted from the Walrus demo notebook)
# ============================================================================
@torch.no_grad()
def _standalone_rollout_model(
    model,
    revin,
    batch: dict,
    formatter,
    max_rollout_steps: int = 50,
    model_epsilon: float = 1e-5,
    device: torch.device = torch.device("cpu"),
):
    """Standalone rollout — does not depend on the LightningModule.

    Lifted from walrus/demo_notebooks/walrus_example_1_RunningWalrus.ipynb
    (the "Simplified version of the trainer method for demo purposes" cell).
    """
    from walrus.trainer.training import expand_mask_to_match

    metadata = batch["metadata"]
    batch = {
        k: v.to(device) if k not in {"metadata", "boundary_conditions"} else v
        for k, v in batch.items()
    }

    if "mask" in batch["metadata"].constant_field_names[0]:
        mask_index = batch["metadata"].constant_field_names[0].index("mask")
        mask = batch["constant_fields"][..., mask_index : mask_index + 1].to(
            device, dtype=torch.bool
        )
    else:
        mask = None

    inputs, y_ref = formatter.process_input(
        batch,
        causal_in_time=model.causal_in_time,
        predict_delta=True,
        train=False,
    )

    T_in = batch["input_fields"].shape[1]
    max_rollout_steps = max_rollout_steps + (T_in - 1)
    rollout_steps = min(y_ref.shape[1], max_rollout_steps)
    train_rollout_limit = 1

    y_ref = y_ref[:, :rollout_steps]
    moving_batch = copy.deepcopy(batch)
    y_preds = []
    for i in range(train_rollout_limit - 1, rollout_steps):
        inputs, _ = formatter.process_input(moving_batch)
        inputs = list(inputs)
        normalization_stats = revin.compute_stats(
            inputs[0], metadata, epsilon=model_epsilon
        )
        normalized_inputs = inputs[:]
        normalized_inputs[0] = revin.normalize_stdmean(
            normalized_inputs[0], normalization_stats
        )
        y_pred = model(
            normalized_inputs[0],
            normalized_inputs[1],
            normalized_inputs[2].tolist(),
            metadata=metadata,
        )
        if model.causal_in_time:
            y_pred = y_pred[-1:]
        y_pred = inputs[0][-y_pred.shape[0] :].float() + revin.denormalize_delta(
            y_pred, normalization_stats
        )
        y_pred = formatter.process_output(y_pred, metadata)[..., : y_ref.shape[-1]]
        if mask is not None:
            mask_pred = expand_mask_to_match(mask, y_pred)
            y_pred.masked_fill_(mask_pred, 0)
        y_pred = y_pred.masked_fill(~batch["padded_field_mask"], 0.0)
        if i != rollout_steps - 1:
            moving_batch["input_fields"] = torch.cat(
                [moving_batch["input_fields"][:, 1:], y_pred[:, -1:]], dim=1
            )
        if model.causal_in_time and i == train_rollout_limit - 1:
            y_preds.append(y_pred)
        else:
            y_preds.append(y_pred[:, -1:])
    y_pred_out = torch.cat(y_preds, dim=1)
    return y_pred_out, y_ref
