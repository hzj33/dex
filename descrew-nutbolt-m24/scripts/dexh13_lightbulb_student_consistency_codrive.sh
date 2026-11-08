#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <gpu_ids> <seed> <run_tag> [teacher_ckpt] [hydra overrides...]" >&2
  exit 2
fi

GPUS=$1
SEED=$2
CACHE=$3
TEACHER_CKPT=${4:-sim2real/codrive/best_reward_4159.37.pth}

if [[ $# -ge 4 ]]; then
  shift 4
else
  shift 3
fi
EXTRA_ARGS=("$@")

NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-200000}
STUDENT_PROGRESS_LOG_INTERVAL=${STUDENT_PROGRESS_LOG_INTERVAL:-25000}
CONSISTENCY_LR=${CONSISTENCY_LR:-3e-4}
CONSISTENCY_LOSS_COEF=${CONSISTENCY_LOSS_COEF:-1.0}
CONSISTENCY_BOUNDARY_COEF=${CONSISTENCY_BOUNDARY_COEF:-0.5}
CONSISTENCY_NUM_SCALES=${CONSISTENCY_NUM_SCALES:-10}
CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}
CONSISTENCY_USE_EMA_TARGET=${CONSISTENCY_USE_EMA_TARGET:-False}
CONSISTENCY_EMA_DECAY=${CONSISTENCY_EMA_DECAY:-0.999}
CONSISTENCY_INFER_USE_EMA=${CONSISTENCY_INFER_USE_EMA:-False}
CONSISTENCY_TRAIN_ALIGN_INFER=${CONSISTENCY_TRAIN_ALIGN_INFER:-False}
CONSISTENCY_LR_SCHEDULE=${CONSISTENCY_LR_SCHEDULE:-none}
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS:-0}
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS:-0}
CONSISTENCY_LR_FINAL_SCALE=${CONSISTENCY_LR_FINAL_SCALE:-1.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
BASE_ACTION_ANCHOR_COEF=${BASE_ACTION_ANCHOR_COEF:-0.0}
CONSISTENCY_ACTION_L2_COEF=${CONSISTENCY_ACTION_L2_COEF:-0.0}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py \
task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive \
headless=True \
seed=${SEED} \
sim_device=cuda:0 \
rl_device=cuda:0 \
graphics_device_id=0 \
train.algo=ConsistencyLatentStudent \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_consistency_codrive/${CACHE} \
experiment=student_codrive_consistency \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=${NUM_ENVS} \
train.ppo.minibatch_size=${MINIBATCH} \
wandb_activate=False \
checkpoint=${TEACHER_CKPT} \
+train.ppo.consistency_lr=${CONSISTENCY_LR} \
+train.ppo.consistency_loss_coef=${CONSISTENCY_LOSS_COEF} \
+train.ppo.consistency_boundary_coef=${CONSISTENCY_BOUNDARY_COEF} \
+train.ppo.consistency_num_scales=${CONSISTENCY_NUM_SCALES} \
+train.ppo.consistency_infer_steps=${CONSISTENCY_INFER_STEPS} \
+train.ppo.consistency_use_ema_target=${CONSISTENCY_USE_EMA_TARGET} \
+train.ppo.consistency_ema_decay=${CONSISTENCY_EMA_DECAY} \
+train.ppo.consistency_infer_use_ema=${CONSISTENCY_INFER_USE_EMA} \
+train.ppo.consistency_stochastic_infer=False \
+train.ppo.consistency_train_align_infer=${CONSISTENCY_TRAIN_ALIGN_INFER} \
+train.ppo.consistency_lr_schedule=${CONSISTENCY_LR_SCHEDULE} \
+train.ppo.consistency_lr_decay_start_agent_steps=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS} \
+train.ppo.consistency_lr_decay_end_agent_steps=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS} \
+train.ppo.consistency_lr_final_scale=${CONSISTENCY_LR_FINAL_SCALE} \
+train.ppo.bc_loss_coef=${BC_LOSS_COEF} \
+train.ppo.base_action_anchor_coef=${BASE_ACTION_ANCHOR_COEF} \
+train.ppo.consistency_action_l2_coef=${CONSISTENCY_ACTION_L2_COEF} \
+train.ppo.student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS} \
+train.ppo.student_progress_log_interval_agent_steps=${STUDENT_PROGRESS_LOG_INTERVAL} \
"${EXTRA_ARGS[@]}"
