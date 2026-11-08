#!/bin/bash
set -euo pipefail

# Usage:
#   scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh \
#     GPU_ID SEED TEACHER_CACHE ROLLOUT_PT OUT_CACHE [PRETRAIN_UPDATES] [extra overrides...]
#
# Example:
#   scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh \
#     0 42 run_a outputs/XHandHoraScrewDriver_teacher/run_a/teacher_rollouts/run_a.pt \
#     run_a_action_chunk_rollout 2000

GPU_ID=$1
SEED=$2
TEACHER_CACHE=$3
ROLLOUT_PT=$4
OUT_CACHE=$5
PRETRAIN_UPDATES=${6:-2000}

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:6:$len}")

scripts/screwdriver_student_diffusion_action_chunk.sh "${GPU_ID}" "${SEED}" "${TEACHER_CACHE}" \
  train.ppo.output_name=XHandHoraScrewDriver_student_diffusion_action_chunk/"${OUT_CACHE}" \
  +train.ppo.rollout_pretrain_path="${ROLLOUT_PT}" \
  +train.ppo.rollout_pretrain_updates="${PRETRAIN_UPDATES}" \
  +train.ppo.rollout_pretrain_batch_size=2048 \
  +train.ppo.rollout_pretrain_log_interval=100 \
  "${EXTRA_ARGS[@]}"
