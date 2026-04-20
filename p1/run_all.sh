#!/usr/bin/env bash
# Sequential driver: baseline + 3 fine-tunes. Run under nohup.
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.bashrc_p1
unset WANDB_ENTITY

DATA=/root/data/datasets
COMMON="--data_base $DATA --bs 8 --lr 1e-3 --modes 12 --hidden 48 --workers 4"

echo "[run_all] $(date) baseline M_A=0.7 ..."
python -u p1/train.py --mode baseline --out runs/baseline \
    --epochs 20 --run_name baseline_MA07 $COMMON \
    2>&1 | tee runs/baseline.log

echo "[run_all] $(date) finetune 100% ..."
python -u p1/train.py --mode finetune --out runs/ft_100 \
    --init_ckpt runs/pretrain/best.pt --data_frac 1.0 \
    --epochs 15 --run_name ft_100 $COMMON \
    2>&1 | tee runs/ft_100.log

echo "[run_all] $(date) finetune 10% ..."
python -u p1/train.py --mode finetune --out runs/ft_10 \
    --init_ckpt runs/pretrain/best.pt --data_frac 0.1 \
    --epochs 25 --run_name ft_10 $COMMON \
    2>&1 | tee runs/ft_10.log

echo "[run_all] $(date) finetune 1% ..."
python -u p1/train.py --mode finetune --out runs/ft_01 \
    --init_ckpt runs/pretrain/best.pt --data_frac 0.01 \
    --epochs 40 --run_name ft_01 $COMMON \
    2>&1 | tee runs/ft_01.log

echo "[run_all] $(date) DONE"
