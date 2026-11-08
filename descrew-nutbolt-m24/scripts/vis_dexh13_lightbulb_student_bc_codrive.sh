#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"

GPUS=$1
SEED=$2
CACHE=$3

array=("$@")
EXTRA_ARGS=("${array[@]:3}")

CKPT="outputs/Dexh13HoraLightbulb_student_bc_codrive/${CACHE}/bc_nn/model_best.ckpt"

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive headless=False seed=${SEED} \
train.algo=BCStudent \
train.ppo.proprio_adapt=True \
train.ppo.output_name=Dexh13HoraLightbulb_student_bc_codrive/${CACHE}_vis \
experiment=student_codrive_bc_vis \
test=True \
task.env.numEnvs=1 \
task.env.termination.grace_steps=0 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
task.env.randomization.noisy_pos_scale=0.0 \
task.env.randomization.noisy_rpy_scale=0.0 \
task.env.randomization.obs_noise_t_scale=0.0 \
task.env.randomization.obs_noise_e_scale=0.0 \
task.env.randomization.action_noise_e_scale=0.0 \
task.env.randomization.action_noise_t_scale=0.0 \
task.env.forceScale=0.0 \
train.ppo.minibatch_size=12 \
++train.bc.test_num_episodes=999999 \
++train.bc.test_max_steps=1000000 \
"checkpoint=${CKPT}" \
"${EXTRA_ARGS[@]}"
