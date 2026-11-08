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

CKPT_DIR="outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/${CACHE}/stage2_diffusion_action_chunk_nn"
CKPT_PATH=""
for CAND in \
  "model_best_student_reward.ckpt" \
  "model_best_student.ckpt" \
  "model_last.ckpt" \
  "model_best.ckpt"
do
  if [[ -f "${CKPT_DIR}/${CAND}" ]]; then
    CKPT_PATH="${CKPT_DIR}/${CAND}"
    break
  fi
done
if [[ -z "${CKPT_PATH}" ]]; then
  echo "No checkpoint found under ${CKPT_DIR}"
  exit 1
fi
echo "Using checkpoint: ${CKPT_PATH}"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraScrewDriver headless=False seed=${SEED} \
task.env.numEnvs=10 test=True \
train.algo=DiffusionActionChunkStudent \
train.ppo.proprio_adapt=True \
wandb_activate=False \
task.env.reset_dist_threshold=0.12 \
+train.ppo.action_chunk_len=8 \
+train.ppo.action_chunk_diffusion_steps=10 \
+train.ppo.action_chunk_diffusion_steps_infer=10 \
+train.ppo.action_chunk_stochastic_infer=False \
"checkpoint=${CKPT_PATH}" \
"${EXTRA_ARGS[@]}"
