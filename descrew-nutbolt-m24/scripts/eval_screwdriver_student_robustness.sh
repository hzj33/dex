#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/eval_screwdriver_student_robustness.sh \
#     GPU_ID SEED ALGO CHECKPOINT [STEPS] [EVAL_CACHE] [extra overrides...]
#
# Example (nominal):
#   scripts/eval_screwdriver_student_robustness.sh \
#     0 42 DiffusionActionChunkStudent \
#     outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt \
#     256 nominal_diff
#
# Example (perturbed):
#   scripts/eval_screwdriver_student_robustness.sh \
#     0 42 DiffusionActionChunkStudent <ckpt> 256 perturb_diff \
#     task.env.randomization.obs_noise_e_scale=0.03 \
#     task.env.randomization.obs_noise_t_scale=0.015 \
#     task.env.forceScale=1.0 \
#     task.env.randomForceProbScalar=0.2

GPU_ID=${1:-0}
SEED=${2:-42}
ALGO=${3:?Missing ALGO (e.g. ProprioAdapt / DiffusionLatentStudent / DiffusionActionChunkStudent)}
CKPT=${4:?Missing CHECKPOINT}
STEPS=${5:-256}
EVAL_CACHE=${6:-eval_default}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
ensure_isaacgym_env

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:6:$len}")

PY_ARGS=(
  train.py
  task=XHandHoraScrewDriver
  headless=True
  seed=${SEED}
  sim_device=cuda:${GPU_ID}
  rl_device=cuda:${GPU_ID}
  graphics_device_id=7
  test=True
  +test_num_steps=${STEPS}
  train.algo=${ALGO}
  train.ppo.proprio_adapt=True
  train.ppo.output_name=XHandHoraScrewDriver_eval_robustness/${EVAL_CACHE}
  "checkpoint=${CKPT}"
  wandb_activate=False
  task.env.numEnvs=48
  task.env.reset_dist_threshold=0.15
  task.env.randomization.randomizePDGains=False
  task.env.randomization.action_noise_e_scale=0.0
  task.env.randomization.action_noise_t_scale=0.0
  task.env.randomization.obs_noise_e_scale=0.0
  task.env.randomization.obs_noise_t_scale=0.0
  task.env.randomization.noisy_rpy_scale=0.0
  task.env.randomization.noisy_pos_scale=0.0
  task.env.forceScale=0.0
  task.env.randomForceProbScalar=0.0
)

PY_ARGS+=("${EXTRA_ARGS[@]}")
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PY_ARGS[@]}"
