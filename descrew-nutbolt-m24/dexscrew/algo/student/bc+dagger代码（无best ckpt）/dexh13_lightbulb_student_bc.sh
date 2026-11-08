#!/bin/bash
set -euo pipefail

# 纯 BC 基线（Dexh13 + lightbulb）
# Usage:
#   ./scripts/dexh13_lightbulb_student_bc.sh <GPU> <SEED> <TEACHER_CKPT> [hydra overrides...]
#
# 例：./scripts/dexh13_lightbulb_student_bc.sh 0 42 outputs/dexh13_lightbulb_teacher/stage1_nn/model_best.ckpt

GPUS=$1
SEED=$2
TEACHER_CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

NUM_ENVS=${NUM_ENVS:-2048}
DEMO_STEPS=${DEMO_STEPS:-16000}
# 512 × DEMO_STEPS(16000) = 8.192M ≈ BUFFER_SIZE(8M)，时间维度均匀覆盖 teacher rollout。
# >= NUM_ENVS 时自动回退到 NUM_ENVS（见 bc.py._collect_demos）
ADD_NUM_ENVS=${ADD_NUM_ENVS:-512}
BATCH_SIZE=${BATCH_SIZE:-4096}
UPDATES_PER_COLLECT=${UPDATES_PER_COLLECT:-4}
LR=${LR:-3e-4}
ACTION_BC_COEF=${ACTION_BC_COEF:-1.0}
# 默认 1.0：adapt_tconv 直接走 latent MSE 拿梯度，不再受冻结 actor_mlp / mu clamp 稀释。
LATENT_COEF=${LATENT_COEF:-1.0}
PRETRAIN_STEPS=${PRETRAIN_STEPS:-5000}
# 对齐 DOTPG max_auto_expert_buffer_size=8000000；fp16 与 DOTPG replay 同 dtype
BUFFER_SIZE=${BUFFER_SIZE:-8000000}
BUFFER_DTYPE=${BUFFER_DTYPE:-float16}
MIN_BUFFER_FOR_UPDATE=${MIN_BUFFER_FOR_UPDATE:-32768}
EVAL_INTERVAL=${EVAL_INTERVAL:-500000}
EVAL_EPISODES=${EVAL_EPISODES:-32}
REUSE_DEMO=${REUSE_DEMO:-True}
SANITY=${SANITY:-True}
STRICT_BASE=${STRICT_BASE:-True}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_bc_student.yaml \
  task=Dexh13HoraLightbulb \
  train=Dexh13HoraLightbulb_BC_Student \
  checkpoint=${TEACHER_CKPT} \
  headless=True graphics_device_id=-1 \
  seed=${SEED} \
  task.env.numEnvs=${NUM_ENVS} \
  train.bc.demo_collect_steps=${DEMO_STEPS} \
  train.bc.add_num_envs=${ADD_NUM_ENVS} \
  train.bc.batch_size=${BATCH_SIZE} \
  train.bc.updates_per_collect=${UPDATES_PER_COLLECT} \
  train.bc.lr=${LR} \
  train.bc.action_bc_coef=${ACTION_BC_COEF} \
  train.bc.latent_coef=${LATENT_COEF} \
  train.bc.pretrain_steps=${PRETRAIN_STEPS} \
  train.bc.buffer_size=${BUFFER_SIZE} \
  train.bc.buffer_dtype=${BUFFER_DTYPE} \
  train.bc.min_buffer_for_update=${MIN_BUFFER_FOR_UPDATE} \
  train.bc.eval_interval_agent_steps=${EVAL_INTERVAL} \
  train.bc.eval_num_episodes=${EVAL_EPISODES} \
  train.bc.reuse_demo_buffer=${REUSE_DEMO} \
  train.bc.sanity_check_on_restore=${SANITY} \
  train.bc.strict_base_policy=${STRICT_BASE} \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  wandb_activate=True \
  task.env.termination.log=True \
  ${EXTRA_ARGS}
