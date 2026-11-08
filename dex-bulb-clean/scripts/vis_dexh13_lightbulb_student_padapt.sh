#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulb headless=False seed=${SEED} \
  task.env.numEnvs=1 test=True \
  train.algo=ProprioAdapt \
  train.ppo.proprio_adapt=True \
  wandb_activate=False \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  train.load_path=${CKPT} \
  ${EXTRA_ARGS}

