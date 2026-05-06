#!/bin/bash
# A100 80GB bootstrap for FP64-only sanity check.
# Faster than rerunning everything: only need 1 trajectory's worth of data.
set -euo pipefail
cd /workspace
apt-get update -qq && apt-get install -y -qq git rsync curl

[ -d code ] || git clone -q https://github.com/sdelaurentiis123/well-work.git code
[ -d walrus_repo ] || git clone -q https://github.com/PolymathicAI/walrus.git walrus_repo

cd /workspace/code/p2-walrus
mkdir -p ckpts/walrus-1.3b data logs results/extra_checks

pip install -q --no-cache-dir hydra-core omegaconf pytorch-lightning einops h5py \
    huggingface_hub the_well matplotlib tqdm
pip install -q --no-cache-dir -e /workspace/walrus_repo

# Walrus checkpoint + MHD test (HF, fast)
hf download polymathic-ai/walrus --local-dir ckpts/walrus-1.3b
hf download polymathic-ai/MHD_64 --repo-type=dataset --include "data/test/*" --local-dir data_hf

# Restructure data layout
mkdir -p data/datasets/MHD_64/data
cp -r data_hf/data/test data/datasets/MHD_64/data/test
cd data/datasets/MHD_64/data && ln -sf test train && ln -sf test valid && cd /workspace/code/p2-walrus

# Run FP64 (50 steps, 1 trajectory)
python -m src.extra_checks --mode fp64 --K 50 \
    --test_dir data/datasets/MHD_64/data/test \
    --ckpt_path ckpts/walrus-1.3b/walrus.pt \
    --config_path ckpts/walrus-1.3b/extended_config.yaml \
    --well_base_path data/datasets \
    --out_dir results/extra_checks \
    2>&1 | tee logs/fp64.log

echo "=== FP64 done ==="
ls -lah results/extra_checks/
