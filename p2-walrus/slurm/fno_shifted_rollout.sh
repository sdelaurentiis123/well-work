#!/bin/bash -l
#SBATCH --job-name=p2_fno
#SBATCH --account=astro
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint=a40
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/fno_%j.out
#SBATCH --error=slurm/fno_%j.err

set -euo pipefail
cd /ginsburg/astro/users/sod2112/well-work/code/p2-walrus

source /etc/profile.d/modules.sh
module load anaconda/3-2023.09
# Do NOT load cuda12.0/toolkit — torch ships its own CUDA libs and the system
# CUDA module conflicts with the cu124-pinned wheels Walrus brought in.

echo "=== job start $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

cd code 2>/dev/null || true  # in case cwd already correct

conda run -p ./env python -m src.fno_shifted_rollout \
    --test_dir data/datasets/MHD_64/data/test \
    --ckpt_dir ckpts/fno \
    --out_root results/shifted_window \
    --n_traj 10 --K 50 --max_history 3 \
    --device cuda

echo "=== job end $(date) ==="
ls -lah results/shifted_window/
