#!/usr/bin/env bash
# After all training finishes: evaluate every checkpoint with eval_full.py.
# Writes evals2/<run>/ with 15-traj spectra, rollout-vs-truth, anisotropic, snapshots.
set -euo pipefail

cd "$(dirname "$0")/.."
source ~/.bashrc_p1
unset WANDB_ENTITY

DATA=/root/data/datasets

for ckpt in runs/*/best.pt; do
  name=$(basename $(dirname "$ckpt"))
  out=evals2/$name
  [ -d "$out" ] && echo "[evals] skip (exists): $name" && continue
  echo "[evals] $(date)  $name ..."
  python -u p1/eval_full.py --ckpt "$ckpt" --out "$out" \
      --n_traj 15 --n_rollout 5 --K 50 --data_base $DATA \
      2>&1 | tail -30
done

echo "[evals] $(date)  plotting data-efficiency v2..."
python -u p1/plot_data_efficiency_v2.py --runs_dir runs --out p1/figures/p1_data_efficiency_v2.png 2>&1 | tail -20

echo "[evals] $(date) DONE"
