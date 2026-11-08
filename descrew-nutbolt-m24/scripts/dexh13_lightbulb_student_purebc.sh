#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulb headless=True seed=${SEED} \
train.algo=PureBC \
train.ppo.proprio_adapt=True train.ppo.output_name=Dexh13HoraLightbulb_student_purebc/${CACHE} \
experiment=student_sim_purebc \
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
"checkpoint=outputs/Dexh13HoraLightbulb_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
${EXTRA_ARGS}
