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
train.algo=DiffusionLatentStudent \
train.ppo.proprio_adapt=True \
wandb_activate=False \
task.env.reset_dist_threshold=0.12 \
+train.ppo.diffusion_steps=10 \
+train.ppo.diffusion_steps_infer=10 \
"checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/${CACHE}/stage2_diffusion_nn/model_best.ckpt" \
"${EXTRA_ARGS[@]}"
