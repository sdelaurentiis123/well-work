#!/usr/bin/env bash
# From-scratch baselines at 10% and 1% data — direct counterparts to ft_10 and ft_01.
# Same seed so the train subset matches the fine-tune subset.
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.bashrc_p1
unset WANDB_ENTITY

DATA=/root/data/datasets
COMMON="--data_base $DATA --bs 8 --lr 1e-3 --modes 12 --hidden 48 --workers 4"

echo "[run_baselines] $(date) baseline 10% ..."
python -u p1/train.py --mode baseline --out runs/baseline_10 \
    --data_frac 0.1 --epochs 25 --run_name baseline_10 $COMMON \
    2>&1 | tee runs/baseline_10.log

echo "[run_baselines] $(date) baseline 1% ..."
python -u p1/train.py --mode baseline --out runs/baseline_01 \
    --data_frac 0.01 --epochs 40 --run_name baseline_01 $COMMON \
    2>&1 | tee runs/baseline_01.log

echo "[run_baselines] $(date) DONE"
