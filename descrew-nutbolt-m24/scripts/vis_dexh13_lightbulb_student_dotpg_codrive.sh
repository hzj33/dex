#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
ensure_isaacgym_env

GPUS=$1
SEED=$2
CACHE=$3

array=("$@")
EXTRA_ARGS=("${array[@]:3}")

CKPT="outputs/Dexh13HoraLightbulb_student_dotpg_codrive/${CACHE}/student_output/dotpg_nn/model_best.ckpt"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=False seed=${SEED} \
sim_device=cuda:${GPUS} \
rl_device=cuda:${GPUS} \
graphics_device_id=${GPUS} \
train.algo=DOTPG \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_dotpg_codrive/${CACHE}_vis \
experiment=student_codrive_dotpg_vis \
test=True \
task.env.numEnvs=1 \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.randomizePDGains=False \
task.env.randomization.noisy_pos_scale=0.0 \
task.env.randomization.noisy_rpy_scale=0.0 \
task.env.randomization.obs_noise_t_scale=0.0 \
task.env.randomization.obs_noise_e_scale=0.0 \
task.env.randomization.action_noise_e_scale=0.0 \
task.env.randomization.action_noise_t_scale=0.0 \
task.env.forceScale=0.0 \
task.env.randomForceProbScalar=0.0 \
train.ppo.minibatch_size=12 \
++train.dotpg.state_mode=student \
++train.dotpg.dynamic_state=True \
++train.dotpg.policy_arch=teacher_actor \
++train.dotpg.policy_output_mode=clamp \
++train.dotpg.policy_loss_mode=dual \
++train.dotpg.test_num_episodes=999999 \
++train.dotpg.test_max_steps=1000000 \
"checkpoint=${CKPT}" \
"${EXTRA_ARGS[@]}"
