#!/usr/bin/env bash
# T2 Q5: rerun the 4 critical configs at seeds 1 and 2.
# baseline_10 / ft_10 / baseline_01 / ft_01. Pretrain is shared — seed=0 is fine for the foundation.
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.bashrc_p1
unset WANDB_ENTITY

DATA=/root/data/datasets
COMMON="--data_base $DATA --bs 8 --lr 1e-3 --modes 12 --hidden 48 --workers 4"
PRETRAIN_CKPT=runs/pretrain/best.pt

for SEED in 1 2; do
  for cfg in \
      "baseline_10:baseline:0.1:25::" \
      "ft_10:finetune:0.1:25:$PRETRAIN_CKPT" \
      "baseline_01:baseline:0.01:40::" \
      "ft_01:finetune:0.01:40:$PRETRAIN_CKPT" ; do
    IFS=":" read -r name mode frac epochs init <<< "$cfg"
    OUT=runs/${name}_s${SEED}
    NAME=${name}_s${SEED}
    INIT_ARG=""
    [ -n "$init" ] && INIT_ARG="--init_ckpt $init"
    echo "[seeds] $(date)  $NAME  mode=$mode frac=$frac epochs=$epochs seed=$SEED init=$init"
    python -u p1/train.py --mode $mode --out $OUT --data_frac $frac --epochs $epochs \
        --seed $SEED --run_name $NAME $INIT_ARG $COMMON \
        2>&1 | tee $OUT.log
  done
done

echo "[seeds] $(date) DONE"
