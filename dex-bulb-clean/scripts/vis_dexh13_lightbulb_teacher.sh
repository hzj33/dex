#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulb seed=${SEED} experiment=rl train.algo=PPO \
  test=True wandb_activate=False \
  checkpoint=${CKPT} \
  headless=False graphics_device_id=0 task.env.numEnvs=1 \
  task.env.termination.grace_steps=800 \
  task.env.termination.enable_finger_dist=False \
  task.env.termination.enable_nut_stagnation=False \
  task.env.termination.enable_no_contact=False \
  task.env.termination.enable_screw_limit=False \
  ${EXTRA_ARGS}

