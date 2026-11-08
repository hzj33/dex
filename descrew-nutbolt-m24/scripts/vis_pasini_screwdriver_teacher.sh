#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

# Live training with viewer to inspect initial pose / contacts.
# Use a small env count and matching PPO batch params so it runs without the batch-size assertion.
# python train.py task=XHandPasiniScrewDriver headless=False seed=${SEED} \
# sim_device=cuda:${GPUS} rl_device=cuda:${GPUS} graphics_device_id=7 \
# task.env.numEnvs=6 test=False \
# train.algo=PPO \
# task.env.reset_dist_threshold=0.1 \
# wandb_activate=False \
# train.ppo.output_name=XHandPasiniScrewDriver_teacher/${CACHE}_vis \
# train.ppo.minibatch_size=72 train.ppo.mini_epochs=1 \
# ${EXTRA_ARGS}
python train.py task=XHandPasiniScrewDriver headless=False seed=${SEED} \
sim_device=cuda:${GPUS} rl_device=cuda:${GPUS} graphics_device_id=7 \
task.env.numEnvs=12 test=True \
train.algo=PPO \
wandb_activate=False \
"checkpoint=outputs/XHandPasiniScrewDriver_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
${EXTRA_ARGS}
