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
    task=Dexh13HoraLightbulbSim2RealTwoFingerThumbNoSlip
    headless=${HEADLESS}
    seed=${SEED}
    experiment=sim2real_twofinger_thumb_noslip
    train.algo=PPO
    wandb_activate=False
    train.ppo.output_name=Dexh13HoraLightbulb_teacher_sim2real_twofinger_thumb_noslip/${CACHE}
)

PY_ARGS+=("${NUM_ENVS_ARGS[@]}")
PY_ARGS+=("${EXTRA_ARGS[@]}")

CUDA_VISIBLE_DEVICES=${GPUS} python "${PY_ARGS[@]}"
