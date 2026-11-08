#!/usr/bin/env bash
set -euo pipefail

# Robustness-oriented latent diffusion training entry.
# Usage:
#   scripts/screwdriver_student_diffusion_latent_robust_light.sh GPU_ID SEED CACHE [extra overrides...]

GPU_ID=${1:-0}
SEED=${2:-42}
CACHE=${3:?Missing CACHE}

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:3:$len}")

PY_ARGS=(
  train.py
  task=XHandHoraScrewDriver
  headless=True
  seed=${SEED}
  train.algo=DiffusionLatentStudent
  train.ppo.proprio_adapt=True
  train.ppo.output_name=XHandHoraScrewDriver_student_diffusion_latent/${CACHE}
  experiment=student_sim_diffusion_latent_robust_light
  task.env.numEnvs=48
  task.env.reset_dist_threshold=0.15
  task.env.randomization.obs_noise_t_scale=0.01
  task.env.randomization.obs_noise_e_scale=0.02
  task.env.forceScale=0.5
  task.env.randomForceProbScalar=0.1
  wandb_activate=False
  "checkpoint=outputs/XHandHoraScrewDriver_teacher/${CACHE}/stage1_nn/best_reward_*.pth"
  +train.ppo.diffusion_steps=10
  +train.ppo.diffusion_steps_infer=10
  +train.ppo.diffusion_lr=3e-4
  +train.ppo.diffusion_loss_coef=1.0
  +train.ppo.bc_loss_coef=1.0
)

PY_ARGS+=("${EXTRA_ARGS[@]}")
CUDA_VISIBLE_DEVICES=${GPU_ID} python "${PY_ARGS[@]}"
