#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
HEADLESS=${4:-True}

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

NUM_ENVS_ARG=""
if [ "$HEADLESS" = "False" ] || [ "$HEADLESS" = "false" ]; then
    NUM_ENVS_ARG="task.env.numEnvs=1 train.ppo.minibatch_size=12"
fi

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandPasiniLightbulb headless=${HEADLESS} seed=${SEED} \
experiment=rl \
train.algo=PPO \
wandb_activate=False \
train.ppo.output_name=XHandPasiniLightbulb_teacher/${CACHE} \
${NUM_ENVS_ARG} \
${EXTRA_ARGS}
