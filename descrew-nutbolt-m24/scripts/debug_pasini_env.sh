#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/debug_pasini_env.sh GPU_ID SEED [EXTRA_HYDRA_OVERRIDES...]
# Example:
#   scripts/debug_pasini_env.sh 0 42 headless=False task.env.numEnvs=1 steps=500 random_actions=True

GPU=${1:-0}
SEED=${2:-42}
shift 2 || true

export CUDA_VISIBLE_DEVICES=${GPU}

python scripts/debug_pasini_env.py \
  task=XHandPasiniScrewDriver \
  seed=${SEED} \
  sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=${GPU} \
  headless=True \
  task.env.numEnvs=6 \
  "$@"
