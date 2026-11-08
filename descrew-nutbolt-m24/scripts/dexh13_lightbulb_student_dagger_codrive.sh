#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
CACHE=$3
TEACHER_CKPT=$4

EXTRA_ARGS=("${@:5}")

NUM_ENVS=${NUM_ENVS:-48}
ADD_NUM_ENVS=${ADD_NUM_ENVS:-16}
BATCH_SIZE=${BATCH_SIZE:-512}
UPDATES_PER_COLLECT=${UPDATES_PER_COLLECT:-4}
BUFFER_SIZE=${BUFFER_SIZE:-200000}
BUFFER_DEVICE=${BUFFER_DEVICE:-cpu}
BUFFER_DTYPE=${BUFFER_DTYPE:-float32}
BETA_START=${BETA_START:-1.0}
BETA_MIN=${BETA_MIN:-0.0}
BETA_DECAY=${BETA_DECAY:-0.0002}
ACTION_MIXING=${ACTION_MIXING:-switch}
PURE_STUDENT_ROLLOUT=${PURE_STUDENT_ROLLOUT:-False}
WARMUP_COLLECT_STEPS=${WARMUP_COLLECT_STEPS:-64}
EVAL_INTERVAL=${EVAL_INTERVAL:-100000}
EVAL_EPISODES=${EVAL_EPISODES:-32}
EVAL_MAX_STEPS=${EVAL_MAX_STEPS:-0}
SANITY_MAX_STEPS=${SANITY_MAX_STEPS:-1000}
EVAL_SELECT_ENABLED=${EVAL_SELECT_ENABLED:-False}
EVAL_SELECT_INTERVAL=${EVAL_SELECT_INTERVAL:-100000}
EVAL_SELECT_NUM_STEPS=${EVAL_SELECT_NUM_STEPS:-256}
EVAL_SELECT_DONE_PENALTY=${EVAL_SELECT_DONE_PENALTY:-2000.0}
MAX_AGENT_STEPS=${MAX_AGENT_STEPS:-0}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=True seed=${SEED} \
train.algo=DAggerStudent \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_dagger_codrive/${CACHE} \
experiment=student_codrive_dagger \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=${NUM_ENVS} \
train.ppo.minibatch_size=576 \
++train.dagger.add_num_envs=${ADD_NUM_ENVS} \
++train.dagger.batch_size=${BATCH_SIZE} \
++train.dagger.updates_per_collect=${UPDATES_PER_COLLECT} \
++train.dagger.buffer_size=${BUFFER_SIZE} \
++train.dagger.buffer_device=${BUFFER_DEVICE} \
++train.dagger.buffer_dtype=${BUFFER_DTYPE} \
++train.dagger.beta_start=${BETA_START} \
++train.dagger.beta_min=${BETA_MIN} \
++train.dagger.beta_decay=${BETA_DECAY} \
++train.dagger.action_mixing=${ACTION_MIXING} \
++train.dagger.pure_student_rollout=${PURE_STUDENT_ROLLOUT} \
++train.dagger.warmup_collect_steps=${WARMUP_COLLECT_STEPS} \
++train.dagger.eval_interval_agent_steps=${EVAL_INTERVAL} \
++train.dagger.eval_num_episodes=${EVAL_EPISODES} \
++train.dagger.eval_max_steps=${EVAL_MAX_STEPS} \
++train.dagger.sanity_max_steps=${SANITY_MAX_STEPS} \
++train.dagger.max_agent_steps=${MAX_AGENT_STEPS} \
++train.ppo.eval_select.enabled=${EVAL_SELECT_ENABLED} \
++train.ppo.eval_select.interval_agent_steps=${EVAL_SELECT_INTERVAL} \
++train.ppo.eval_select.num_steps=${EVAL_SELECT_NUM_STEPS} \
++train.ppo.eval_select.done_penalty=${EVAL_SELECT_DONE_PENALTY} \
wandb_activate=False \
"checkpoint=${TEACHER_CKPT}" \
"${EXTRA_ARGS[@]}"
