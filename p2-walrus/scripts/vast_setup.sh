#!/bin/bash
# Vast.ai cloud-instance bootstrap. Run from /workspace or $HOME.
#
# Idempotent: re-running picks up where it left off.
#
# Sets up the same env as Ginsburg, then downloads Walrus + MHD_64 test in
# parallel. ~10-15 min total before the rollouts can start.
set -euo pipefail

WORK=/workspace/p2-walrus
mkdir -p "$WORK" && cd "$WORK"

echo "=== [1/6] system tools ==="
apt-get update -qq
apt-get install -y -qq git rsync curl

echo "=== [2/6] clone repos ==="
[ -d code ] || git clone https://github.com/sdelaurentiis123/well-work.git code
[ -d walrus_repo ] || git clone https://github.com/PolymathicAI/walrus.git walrus_repo

cd "$WORK/code/p2-walrus"
mkdir -p ckpts/walrus-1.3b ckpts/fno data logs outputs results/shifted_window

echo "=== [3/6] python deps ==="
# Image already has torch 2.5.1 + cuda. Install the rest.
pip install -q --no-cache-dir hydra-core omegaconf pytorch-lightning einops h5py \
    huggingface_hub the_well matplotlib tqdm
pip install -q -e "$WORK/walrus_repo"

echo "=== [4/6] background downloads (Walrus + MHD_64 test) ==="
nohup hf download polymathic-ai/walrus --local-dir ckpts/walrus-1.3b > logs/hf_walrus.log 2>&1 < /dev/null &
echo "hf_pid=$!"
nohup the-well-download --dataset MHD_64 --split test --base-path data --parallel > logs/well_dl.log 2>&1 < /dev/null &
echo "well_pid=$!"

echo "=== [5/6] await downloads ==="
wait
echo "downloads done"
du -sh ckpts/walrus-1.3b/ data/

echo "=== [6/6] symlink train/valid -> test (Walrus data_module needs them) ==="
cd data/datasets/MHD_64/data
ln -sf test train
ln -sf test valid
ls -la

echo "=== ALL READY ==="
