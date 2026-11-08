#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg_student.yaml \
  task=RBHandHoraLightbulb train=RBHandHoraLightbulb_DOTPG_Student \
  headless=False graphics_device_id=0 seed=${SEED} \
  task.env.numEnvs=1 test=True \
  wandb_activate=False \
  task.env.wrist.enable=True task.env.wrist.action_index=0 task.env.wrist.yaw_scale=-0.05 task.env.wrist.limit_deg=25.0 task.env.wrist.smooth=True task.env.wrist.pivot_mode=nut_pos task.env.wrist.write_to_obs=True \
  task.env.termination.grace_steps=0 task.env.termination.enable_finger_dist=True task.env.termination.enable_nut_stagnation=True task.env.termination.enable_no_contact=True task.env.termination.enable_screw_limit=True \
  train.load_path=${CKPT} \
  ${EXTRA_ARGS}
