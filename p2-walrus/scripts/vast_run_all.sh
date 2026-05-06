#!/bin/bash
# Run after vast_setup downloads finish: symlinks + FNO + Walrus rollouts.
set -euo pipefail
cd /workspace/code/p2-walrus

echo "=== symlink train, valid -> test ==="
cd data/datasets/MHD_64/data
[ -e train ] || ln -sf test train
[ -e valid ] || ln -sf test valid
ls -la
cd /workspace/code/p2-walrus

echo "=== FNO baselines (3 configs × 10 trajectories × 50 steps) ==="
python -m src.fno_shifted_rollout \
    --test_dir data/datasets/MHD_64/data/test \
    --ckpt_dir ckpts/fno \
    --out_root results/shifted_window \
    --n_traj 10 --K 50 --max_history 3 \
    --device cuda 2>&1 | tee logs/fno_run.log

echo
echo "=== Walrus zero-shot (1 model × 10 trajectories × 50 steps) ==="
export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1
python -m src.walrus_rollout \
    --test_dir data/datasets/MHD_64/data/test \
    --ckpt_path ckpts/walrus-1.3b/walrus.pt \
    --config_path ckpts/walrus-1.3b/extended_config.yaml \
    --well_base_path data/datasets \
    --out_root results/shifted_window \
    --n_traj 10 --K 50 --max_history 3 \
    --device cuda 2>&1 | tee logs/walrus_run.log

echo
echo "=== ALL DONE ==="
ls -lah results/shifted_window/
for cfg in fno_baseline fno_ft fno_pretrain_ood walrus; do
    echo "--- $cfg ---"
    ls -lah results/shifted_window/$cfg/ 2>&1 | head
done
