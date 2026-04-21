#!/usr/bin/env bash
# Wait for MHD_64 train+valid downloads to finish, then kick off HP sweep.
set -euo pipefail

cd ~/well-work
source ~/.bashrc_p1
unset WANDB_ENTITY

echo "[chain] $(date) waiting for downloads..."
while pgrep -f the-well-download > /dev/null; do sleep 30; done
echo "[chain] $(date) downloads done; starting HP sweep"

# Verify data is present and intact
test -f /root/data/datasets/MHD_64/data/train/MHD_Ma_0.7_Ms_0.5.hdf5 || { echo "MISSING TRAIN FILE"; exit 2; }
test -f /root/data/datasets/MHD_64/data/valid/MHD_Ma_0.7_Ms_0.5.hdf5 || { echo "MISSING VALID FILE"; exit 2; }

python -u p1/hp_sweep.py --data_base /root/data/datasets \
    --out_root runs --pretrain_ckpt runs/pretrain/best.pt \
    --top_k 5 --top_k_ft 3 --log_progress \
    2>&1 | tee runs/hp_sweep.log

echo "[chain] $(date) HP SWEEP DONE"
