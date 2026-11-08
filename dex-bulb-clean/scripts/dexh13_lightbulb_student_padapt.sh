#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3
TEACHER_CKPT=$4

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

NUM_ENVS=${NUM_ENVS:-4096}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulb headless=True seed=${SEED} \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True train.ppo.output_name=Dexh13HoraLightbulb_student_padapt/${CACHE} \
experiment=student_sim \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=${NUM_ENVS} \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
wandb_activate=False \
checkpoint=${TEACHER_CKPT} \
${EXTRA_ARGS}

