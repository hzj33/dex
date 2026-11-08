#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3
HEADLESS=${4:-True}

array=("$@")
EXTRA_ARGS=("${array[@]:4}")

NUM_ENVS_ARGS=()
if [ "$HEADLESS" = "False" ] || [ "$HEADLESS" = "false" ]; then
    NUM_ENVS_ARGS=(task.env.numEnvs=1 train.ppo.minibatch_size=12)
fi

PY_ARGS=(
    train.py
    task=Dexh13HoraLightbulbThesisTwoFinger
    headless=${HEADLESS}
    seed=${SEED}
    experiment=thesis_twofinger
    train.algo=PPO
    wandb_activate=False
    train.ppo.output_name=Dexh13HoraLightbulb_teacher_thesis_twofinger/${CACHE}
    task.env.termination.grace_steps=150
    task.env.termination.enable_finger_dist=True
    task.env.termination.enable_nut_stagnation=True
    task.env.termination.enable_no_contact=True
    task.env.termination.enable_screw_limit=True
)

PY_ARGS+=("${NUM_ENVS_ARGS[@]}")
PY_ARGS+=("${EXTRA_ARGS[@]}")

CUDA_VISIBLE_DEVICES=${GPUS} python "${PY_ARGS[@]}"
