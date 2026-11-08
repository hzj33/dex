#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraScrewDriver headless=True seed=${SEED} \
train.algo=DiffusionActionChunkStudent \
train.ppo.proprio_adapt=True train.ppo.output_name=XHandHoraScrewDriver_student_diffusion_action_chunk/${CACHE} \
experiment=student_sim_diffusion_action_chunk \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
wandb_activate=False \
task.env.reset_dist_threshold=0.15 \
"checkpoint=outputs/XHandHoraScrewDriver_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
+train.ppo.action_chunk_len=8 \
+train.ppo.action_chunk_diffusion_steps=10 \
+train.ppo.action_chunk_diffusion_steps_infer=10 \
+train.ppo.action_chunk_diffusion_lr=3e-4 \
+train.ppo.action_chunk_diffusion_loss_coef=1.0 \
+train.ppo.action_chunk_first_action_bc_loss_coef=1.0 \
+train.ppo.action_chunk_bc_loss_coef=0.1 \
+train.ppo.action_chunk_stochastic_infer=False \
+train.ppo.action_chunk_teacher_mix_steps=120000 \
${EXTRA_ARGS}
