#!/usr/bin/env bash
# Wait for run_seeds to finish, then run evals, then (optionally) self-shutdown.
# Usage: nohup bash p1/chain_after_seeds.sh > runs/chain.log 2>&1 &
set -euo pipefail

cd "$(dirname "$0")/.."

# Wait for the run_seeds.sh loop to finish (it writes DONE at the end)
echo "[chain] $(date) waiting for run_seeds DONE..."
while ! grep -q "\[seeds\] .* DONE" runs/run_seeds.log 2>/dev/null; do
    sleep 60
done
echo "[chain] $(date) run_seeds finished — launching evals"

bash p1/run_evals.sh 2>&1 | tee runs/evals.log

echo "[chain] $(date) CHAIN DONE"
