"""P1 day 1: stream one MHD_64 trajectory from HF, inspect, plot a slice."""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from the_well.data import WellDataset
from torch.utils.data import DataLoader

ds = WellDataset(
    well_base_path="hf://datasets/polymathic-ai/",
    well_dataset_name="MHD_64",
    well_split_name="train",
)

print("dataset length:", len(ds))
print("metadata:", ds.metadata)
print("field names:", getattr(ds.metadata, "field_names", None))

loader = DataLoader(ds, batch_size=1, num_workers=0)
batch = next(iter(loader))
print("\nbatch keys:", list(batch.keys()))
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    else:
        print(f"  {k}: {type(v).__name__}")
