#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

# RBHand 任务更重一些，默认用 4096 env；可通过 NUM_ENVS 覆盖
NUM_ENVS=${NUM_ENVS:-4096}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=RBHandHoraLightbulb headless=True seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.numEnvs=${NUM_ENVS} \
task.env.wrist.enable=True \
task.env.wrist.action_index=0 \
task.env.wrist.yaw_scale=-0.05 \
task.env.wrist.limit_deg=25.0 \
task.env.wrist.smooth=True \
task.env.wrist.pivot_mode=nut_pos \
task.env.wrist.write_to_obs=True \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
wandb_activate=True \
train.ppo.output_name=RBHandHoraLightbulb_teacher/${CACHE} \
${EXTRA_ARGS}
