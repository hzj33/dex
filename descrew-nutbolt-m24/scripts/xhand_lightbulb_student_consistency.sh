#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraLightbulb headless=True seed=${SEED} \
train.algo=ConsistencyLatentStudent \
train.ppo.proprio_adapt=True train.ppo.output_name=XHandHoraLightbulb_student_consistency/${CACHE} \
experiment=student_sim_consistency \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
train.ppo.minibatch_size=576 \
wandb_activate=False \
"checkpoint=outputs/XHandHoraLightbulb_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
+train.ppo.consistency_lr=3e-4 \
+train.ppo.consistency_loss_coef=1.0 \
+train.ppo.consistency_boundary_coef=0.5 \
+train.ppo.consistency_num_scales=10 \
+train.ppo.bc_loss_coef=1.0 \
${EXTRA_ARGS}
