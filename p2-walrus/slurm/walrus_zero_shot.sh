#!/bin/bash -l
#SBATCH --job-name=p2_walrus
#SBATCH --account=astro
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=96G
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm/walrus_%j.out
#SBATCH --error=slurm/walrus_%j.err

set -euo pipefail
cd /ginsburg/astro/users/sod2112/well-work/code/p2-walrus

source /etc/profile.d/modules.sh
module load anaconda/3-2023.09
# Don't load cuda module — wheels ship their own libs.

export HDF5_USE_FILE_LOCKING=FALSE
export HYDRA_FULL_ERROR=1

echo "=== job start $(date) ==="
echo "host: $(hostname)"
nvidia-smi --query-gpu=name,memory.free --format=csv,noheader

conda run -p ./env python -m src.walrus_rollout \
    --test_dir data/datasets/MHD_64/data/test \
    --ckpt_path ckpts/walrus-1.3b/walrus.pt \
    --config_path ckpts/walrus-1.3b/extended_config.yaml \
    --well_base_path data/datasets \
    --out_root results/shifted_window \
    --n_traj 10 --K 50 --max_history 3 \
    --device cuda

echo "=== job end $(date) ==="
ls -lah results/shifted_window/walrus/ 2>&1
