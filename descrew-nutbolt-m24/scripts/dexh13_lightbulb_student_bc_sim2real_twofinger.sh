#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3
TEACHER_CKPT=$4

array=("$@")
EXTRA_ARGS=("${array[@]:4}")

NUM_ENVS=${NUM_ENVS:-48}
ADD_NUM_ENVS=${ADD_NUM_ENVS:-16}
DEMO_STEPS=${DEMO_STEPS:-5000}
BATCH_SIZE=${BATCH_SIZE:-512}
UPDATES_PER_COLLECT=${UPDATES_PER_COLLECT:-4}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-2000}
BUFFER_SIZE=${BUFFER_SIZE:-200000}
BUFFER_DEVICE=${BUFFER_DEVICE:-cpu}
BUFFER_DTYPE=${BUFFER_DTYPE:-float16}
ACTION_BC_COEF=${ACTION_BC_COEF:-1.0}
LATENT_COEF=${LATENT_COEF:-1.0}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-32}
MAX_AGENT_STEPS=${MAX_AGENT_STEPS:-0}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFinger headless=True seed=${SEED} \
train.algo=BCStudent \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_bc_sim2real_twofinger/${CACHE} \
experiment=student_sim2real_twofinger_bc \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=${NUM_ENVS} \
train.ppo.minibatch_size=576 \
++train.bc.demo_collect_steps=${DEMO_STEPS} \
++train.bc.add_num_envs=${ADD_NUM_ENVS} \
++train.bc.batch_size=${BATCH_SIZE} \
++train.bc.updates_per_collect=${UPDATES_PER_COLLECT} \
++train.bc.pretrain_steps=${PRETRAIN_STEPS} \
++train.bc.buffer_size=${BUFFER_SIZE} \
++train.bc.buffer_device=${BUFFER_DEVICE} \
++train.bc.buffer_dtype=${BUFFER_DTYPE} \
++train.bc.action_bc_coef=${ACTION_BC_COEF} \
++train.bc.latent_coef=${LATENT_COEF} \
++train.bc.eval_interval_agent_steps=${EVAL_INTERVAL} \
++train.bc.eval_num_episodes=${EVAL_EPISODES} \
++train.bc.max_agent_steps=${MAX_AGENT_STEPS} \
wandb_activate=False \
"checkpoint=${TEACHER_CKPT}" \
"${EXTRA_ARGS[@]}"
