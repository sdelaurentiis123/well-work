#!/usr/bin/env bash
# Bootstrap a fresh Vast.ai (PyTorch (Vast) template) box for P1.
# Usage on the box:
#   bash setup_remote.sh
# Expects env vars HF_TOKEN and WANDB_API_KEY already set (or edit ~/.bashrc_p1).
set -euo pipefail

export HF_HUB_ENABLE_HF_TRANSFER=1
source /venv/main/bin/activate

# Python deps on top of the preinstalled torch+cuda image.
pip install --no-cache-dir the_well "the_well[benchmark]" wandb hf_transfer

# Clone repo if not already there.
[ -d ~/well-work ] || git clone https://github.com/sdelaurentiis123/well-work.git ~/well-work

# Download MHD_64 to local disk (faster than streaming; avoids HF rate limits).
mkdir -p /root/data
the-well-download --base-path /root/data --dataset MHD_64 --parallel

# wandb login (non-interactive if WANDB_API_KEY is set).
wandb login --relogin "${WANDB_API_KEY:-}" || true

echo "[setup] done. cd ~/well-work and run p1/train.py."
