#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
TEACHER_CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

NUM_ENVS=${NUM_ENVS:-2048}
ADAPT_WARMUP_STEPS=${ADAPT_WARMUP_STEPS:-4000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
EXPL_NOISE=${EXPL_NOISE:-0.005}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg_student.yaml \
  task=RBHandHoraLightbulb \
  train=RBHandHoraLightbulb_DOTPG_Student \
  checkpoint=${TEACHER_CKPT} \
  headless=True graphics_device_id=-1 \
  seed=${SEED} \
  task.env.numEnvs=${NUM_ENVS} \
  task.env.wrist.enable=True \
  task.env.wrist.action_index=0 \
  task.env.wrist.yaw_scale=-0.05 \
  task.env.wrist.limit_deg=25.0 \
  task.env.wrist.smooth=True \
  task.env.wrist.pivot_mode=nut_pos \
  task.env.wrist.write_to_obs=True \
  train.dotpg.adapt_warmup_steps=${ADAPT_WARMUP_STEPS} \
  train.dotpg.warmup_steps=${WARMUP_STEPS} \
  train.dotpg.exploration_noise=${EXPL_NOISE} \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  wandb_activate=True \
  ${EXTRA_ARGS}
