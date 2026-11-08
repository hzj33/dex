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
train.algo=DOTPG \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_dotpg_sim2real_twofinger/${CACHE} \
experiment=student_sim2real_twofinger_dotpg \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=48 \
train.ppo.minibatch_size=576 \
++train.dotpg.state_mode=student \
++train.dotpg.dynamic_state=True \
++train.dotpg.replay_buffer_device=cpu \
++train.dotpg.replay_buffer_dtype=float16 \
++train.dotpg.expert_buffer_device=cpu \
++train.dotpg.expert_buffer_dtype=float16 \
++train.dotpg.buffer_size=200000 \
++train.dotpg.expert_buffer_size=200000 \
++train.dotpg.max_auto_expert_buffer_size=200000 \
++train.dotpg.expert_add_num_envs=16 \
++train.dotpg.warmup_steps=5000 \
++train.dotpg.batch_size=256 \
++train.dotpg.adapt_warmup_steps=1000 \
++train.dotpg.bc_coef=2.5 \
++train.dotpg.bc_pretrain_steps=2000 \
++train.dotpg.bc_pretrain_lr=0.0003 \
++train.dotpg.bc_batch_size=512 \
++train.dotpg.online_expert=True \
++train.dotpg.reuse_expert_buffer=True \
wandb_activate=False \
"checkpoint=${TEACHER_CKPT}" \
"${EXTRA_ARGS[@]}"
