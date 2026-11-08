#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
ensure_isaacgym_env

GPUS=$1
SEED=$2
CACHE=$3

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:3:$len}")

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraScrewDriver headless=False seed=${SEED} \
task.env.numEnvs=10 test=True \
train.algo=PureBC \
train.ppo.proprio_adapt=True \
wandb_activate=False \
task.env.reset_dist_threshold=0.12 \
"checkpoint=outputs/XHandHoraScrewDriver_student_purebc/${CACHE}/stage2_bc_nn/model_best.ckpt" \
"${EXTRA_ARGS[@]}"
