#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

python train.py task=XHandPasiniBulb headless=False seed=${SEED} \
sim_device=cuda:${GPUS} rl_device=cuda:${GPUS} graphics_device_id=7 \
task.env.numEnvs=6 test=True \
train.algo=PPO \
wandb_activate=False \
"checkpoint=outputs/XHandPasiniBulb_teacher/${CACHE}/stage1_nn/best_reward_*.pth" \
${EXTRA_ARGS}
