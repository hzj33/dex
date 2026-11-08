#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
TEACHER_CKPT=$3
STUDENT_CKPT=$4

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg.yaml \
  task=Dexh13HoraLightbulb \
  train=Dexh13HoraLightbulb_DOTPG \
  checkpoint=${TEACHER_CKPT} \
  test=True wandb_activate=False \
  train.load_path=${STUDENT_CKPT} \
  seed=${SEED} \
  task.env.numEnvs=1 \
  headless=False graphics_device_id=0 \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  ${EXTRA_ARGS}

