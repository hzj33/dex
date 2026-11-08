#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

# 默认可通过环境变量覆盖：NUM_ENVS
NUM_ENVS=${NUM_ENVS:-4096}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulb headless=True seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.numEnvs=${NUM_ENVS} \
task.env.termination.grace_steps=150 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
wandb_activate=True \
train.ppo.output_name=Dexh13HoraLightbulb_teacher/${CACHE} \
${EXTRA_ARGS}

