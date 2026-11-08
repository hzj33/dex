#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

TEACHER_CKPT="outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth"

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbThesisTwoFinger headless=True seed=${SEED} \
train.algo=ProprioAdapt \
train.ppo.proprio_adapt=True train.ppo.output_name=Dexh13HoraLightbulb_student_padapt_thesis_twofinger/${CACHE} \
experiment=student_sim_thesis_twofinger_padapt \
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
${EXTRA_ARGS}
