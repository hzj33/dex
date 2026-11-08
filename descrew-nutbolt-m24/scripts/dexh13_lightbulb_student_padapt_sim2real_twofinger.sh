#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3
TEACHER_CKPT=$4

array=("$@")
EXTRA_ARGS=("${array[@]:4}")

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=${SEED} \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/${CACHE} \
experiment=student_sim2real_twofinger_padapt \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
train.ppo.minibatch_size=576 \
wandb_activate=False \
"checkpoint=${TEACHER_CKPT}" \
"${EXTRA_ARGS[@]}"
