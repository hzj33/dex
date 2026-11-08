#!/bin/bash
set -euo pipefail

GPUS=$1
SEED=$2
TEACHER_CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

NUM_ENVS=${NUM_ENVS:-2048}
BETA_START=${BETA_START:-1.0}
BETA_MIN=${BETA_MIN:-0.0}
# β→0 从 env step 2000→10000，teacher 指导窗口 ×5
BETA_DECAY=${BETA_DECAY:-1e-4}
UPDATES_PER_COLLECT=${UPDATES_PER_COLLECT:-4}
BATCH_SIZE=${BATCH_SIZE:-4096}
# 对齐 DOTPG max_auto_expert_buffer_size=8000000；fp16 与 DOTPG replay 同 dtype
BUFFER_SIZE=${BUFFER_SIZE:-8000000}
BUFFER_DTYPE=${BUFFER_DTYPE:-float16}
WARMUP_COLLECT_STEPS=${WARMUP_COLLECT_STEPS:-64}
MIN_BUFFER_FOR_UPDATE=${MIN_BUFFER_FOR_UPDATE:-32768}
EVAL_INTERVAL=${EVAL_INTERVAL:-500000}
EVAL_EPISODES=${EVAL_EPISODES:-32}
SANITY=${SANITY:-True}
STRICT_BASE=${STRICT_BASE:-True}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dagger_student.yaml \
  task=Dexh13HoraLightbulb \
  train=Dexh13HoraLightbulb_DAgger_Student \
  checkpoint=${TEACHER_CKPT} \
  headless=True graphics_device_id=-1 \
  seed=${SEED} \
  task.env.numEnvs=${NUM_ENVS} \
  train.dagger.beta_start=${BETA_START} \
  train.dagger.beta_min=${BETA_MIN} \
  train.dagger.beta_decay=${BETA_DECAY} \
  train.dagger.updates_per_collect=${UPDATES_PER_COLLECT} \
  train.dagger.batch_size=${BATCH_SIZE} \
  train.dagger.buffer_size=${BUFFER_SIZE} \
  train.dagger.buffer_dtype=${BUFFER_DTYPE} \
  train.dagger.warmup_collect_steps=${WARMUP_COLLECT_STEPS} \
  train.dagger.min_buffer_for_update=${MIN_BUFFER_FOR_UPDATE} \
  train.dagger.eval_interval_agent_steps=${EVAL_INTERVAL} \
  train.dagger.eval_num_episodes=${EVAL_EPISODES} \
  train.dagger.sanity_check_on_restore=${SANITY} \
  train.dagger.strict_base_policy=${STRICT_BASE} \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  wandb_activate=True \
  task.env.termination.log=True \
  ${EXTRA_ARGS}
