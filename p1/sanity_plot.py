"""P1 day 1: plot a slice of one MHD_64 trajectory + histogram of M_A, M_s."""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

import numpy as np
import matplotlib.pyplot as plt
from the_well.data import WellDataset
from torch.utils.data import DataLoader

ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="MHD_64",
    well_split_name="train",
)
print(f"n_samples={len(ds)}  scalar_names={ds.metadata.constant_scalar_names}")

# Sweep constant_scalars across the whole dataset via a loader.
# With 7623 windows this is cheap (scalars only); trajectories shared within same file.
loader = DataLoader(ds, batch_size=32, num_workers=0)
all_scalars = []
for i, batch in enumerate(loader):
    all_scalars.append(batch["constant_scalars"].numpy())
    if i >= 20:  # 640 windows is enough to see the grid
        break
scalars = np.concatenate(all_scalars, axis=0)
ma, ms = scalars[:, 0], scalars[:, 1]
print(f"M_A range: {np.unique(np.round(ma, 3))}")
print(f"M_s range: {np.unique(np.round(ms, 3))}")

# Plot a z-midplane slice of sample 0: density, |B|, |v|
s0 = ds[0]
x = s0["input_fields"].numpy()[0]  # (64,64,64,7): [density, B_xyz, v_xyz]
rho = x[..., 0]
B = x[..., 1:4]
v = x[..., 4:7]
Bmag = np.sqrt((B * B).sum(-1))
vmag = np.sqrt((v * v).sum(-1))

z = 32
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(rho[:, :, z], cmap="viridis"); axes[0].set_title(r"$\rho$")
axes[1].imshow(Bmag[:, :, z], cmap="inferno"); axes[1].set_title(r"$|B|$")
axes[2].imshow(vmag[:, :, z], cmap="plasma"); axes[2].set_title(r"$|v|$")
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
fig.suptitle(f"MHD_64 sample 0, z=32  (M_A={s0['constant_scalars'][0]:.2f}, M_s={s0['constant_scalars'][1]:.2f})")
fig.tight_layout()
fig.savefig("p1/mhd64_sample_slice.png", dpi=120)
print("wrote p1/mhd64_sample_slice.png")

# Scalar sweep scatter
uniq_pairs, counts = np.unique(scalars.round(3), axis=0, return_counts=True)
print("\n(M_A, M_s) -> window count (from 640-sample probe):")
for (a, s), c in zip(uniq_pairs, counts):
    print(f"  M_A={a:.2f} M_s={s:.2f}  n={c}")

fig2, ax = plt.subplots(figsize=(6, 4))
ax.scatter(uniq_pairs[:, 1], uniq_pairs[:, 0], s=counts * 4, alpha=0.7)
ax.set_xlabel(r"$M_s$ (sonic Mach)"); ax.set_ylabel(r"$M_A$ (Alfvén Mach)")
ax.set_title("MHD_64 parameter grid (marker size = window count in probe)")
ax.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig("p1/mhd64_param_grid.png", dpi=120)
print("wrote p1/mhd64_param_grid.png")
