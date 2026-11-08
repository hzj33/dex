#!/bin/bash
set -euo pipefail

# BC 学生模型可视化（Dexh13 + lightbulb）
# Usage:
#   ./scripts/vis_dexh13_lightbulb_student_bc.sh <GPU> <SEED> <STUDENT_CKPT> [hydra overrides...]
#
# 例：./scripts/vis_dexh13_lightbulb_student_bc.sh 0 42 outputs/bc_dexh13_lightbulb_student/stage2_nn/model_best.ckpt

GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_bc_student.yaml \
  task=Dexh13HoraLightbulb train=Dexh13HoraLightbulb_BC_Student \
  headless=False graphics_device_id=0 seed=${SEED} \
  task.env.numEnvs=1 test=True \
  wandb_activate=False \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  train.load_path=${CKPT} \
  ${EXTRA_ARGS}
