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

PY_ARGS=(
    train.py
    task=Dexh13HoraLightbulbThesisTwoFinger
    headless=False
    seed=${SEED}
    sim_device=cuda:${GPUS}
    rl_device=cuda:${GPUS}
    graphics_device_id=7
    task.env.numEnvs=1
    test=True
    train.algo=ProprioAdapt
    train.ppo.proprio_adapt=True
    train.ppo.minibatch_size=12
    task.env.randomization.randomizePDGains=False
    task.env.randomization.action_noise_e_scale=0.0
    task.env.randomization.action_noise_t_scale=0.0
    task.env.randomization.obs_noise_e_scale=0.0
    task.env.randomization.obs_noise_t_scale=0.0
    task.env.randomization.noisy_rpy_scale=0.0
    task.env.randomization.noisy_pos_scale=0.0
    task.env.forceScale=0.0
    task.env.randomForceProbScalar=0.0
    wandb_activate=False
    train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_thesis_twofinger/${CACHE}
    "checkpoint=outputs/Dexh13HoraLightbulb_student_padapt_thesis_twofinger/${CACHE}/stage2_nn/model_best.ckpt"
)

PY_ARGS+=("${EXTRA_ARGS[@]}")
CUDA_VISIBLE_DEVICES=${GPUS} python "${PY_ARGS[@]}"
