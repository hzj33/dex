#!/bin/bash
set -euo pipefail

GPUS=${1:-0}
SEED=${2:-42}
CACHE=${3:-residual_consistency_codrive_s${SEED}_$(date +%Y%m%d_%H%M%S)}
BASE_CKPT=${4:-sim2real/codrive/model_best_codrive.ckpt}

EXTRA_ARGS=("${@:5}")

NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}
CONSISTENCY_LR=${CONSISTENCY_LR:-3e-4}
CONSISTENCY_LOSS_COEF=${CONSISTENCY_LOSS_COEF:-1.0}
CONSISTENCY_BOUNDARY_COEF=${CONSISTENCY_BOUNDARY_COEF:-0.5}
CONSISTENCY_NUM_SCALES=${CONSISTENCY_NUM_SCALES:-10}
CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
BASE_ACTION_ANCHOR_COEF=${BASE_ACTION_ANCHOR_COEF:-0.05}
RESIDUAL_TARGET_SCALE=${RESIDUAL_TARGET_SCALE:-1.0}
RESIDUAL_GATE=${RESIDUAL_GATE:-0.25}
RESIDUAL_RECON_COEF=${RESIDUAL_RECON_COEF:-0.25}
RESIDUAL_ACTION_DELTA_COEF=${RESIDUAL_ACTION_DELTA_COEF:-0.0}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-0}
EVAL_SELECT_ENABLED=${EVAL_SELECT_ENABLED:-False}
EVAL_SELECT_INTERVAL=${EVAL_SELECT_INTERVAL:-100000}
EVAL_SELECT_NUM_STEPS=${EVAL_SELECT_NUM_STEPS:-256}
EVAL_SELECT_DONE_PENALTY=${EVAL_SELECT_DONE_PENALTY:-2000.0}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=True seed=${SEED} \
sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 \
train.algo=ResidualConsistencyLatentStudent \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_residual_consistency_codrive/${CACHE} \
experiment=student_codrive_residual_consistency \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.obs_noise_t_scale=0.01 \
task.env.randomization.obs_noise_e_scale=0.02 \
task.env.numEnvs=${NUM_ENVS} \
train.ppo.minibatch_size=${MINIBATCH} \
++train.ppo.consistency_lr=${CONSISTENCY_LR} \
++train.ppo.consistency_loss_coef=${CONSISTENCY_LOSS_COEF} \
++train.ppo.consistency_boundary_coef=${CONSISTENCY_BOUNDARY_COEF} \
++train.ppo.consistency_num_scales=${CONSISTENCY_NUM_SCALES} \
++train.ppo.consistency_infer_steps=${CONSISTENCY_INFER_STEPS} \
++train.ppo.consistency_stochastic_infer=False \
++train.ppo.consistency_train_align_infer=False \
++train.ppo.bc_loss_coef=${BC_LOSS_COEF} \
++train.ppo.base_action_anchor_coef=${BASE_ACTION_ANCHOR_COEF} \
++train.ppo.consistency_residual_target_scale=${RESIDUAL_TARGET_SCALE} \
++train.ppo.consistency_residual_gate=${RESIDUAL_GATE} \
++train.ppo.consistency_residual_recon_coef=${RESIDUAL_RECON_COEF} \
++train.ppo.consistency_residual_action_delta_coef=${RESIDUAL_ACTION_DELTA_COEF} \
++train.ppo.student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS} \
++train.ppo.eval_select.enabled=${EVAL_SELECT_ENABLED} \
++train.ppo.eval_select.interval_agent_steps=${EVAL_SELECT_INTERVAL} \
++train.ppo.eval_select.num_steps=${EVAL_SELECT_NUM_STEPS} \
++train.ppo.eval_select.done_penalty=${EVAL_SELECT_DONE_PENALTY} \
wandb_activate=False \
"checkpoint=${BASE_CKPT}" \
"${EXTRA_ARGS[@]}"
