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
python train.py task=Dexh13HoraLightbulbMiddleContactViz headless=${HEADLESS} seed=${SEED} \
experiment=middle_contactviz \
train.algo=PPO \
wandb_activate=False \
train.ppo.output_name=Dexh13HoraLightbulb_teacher_middle_contactviz/${CACHE} \
task.env.termination.grace_steps=150 \
task.env.termination.enable_finger_dist=True \
task.env.termination.enable_nut_stagnation=True \
task.env.termination.enable_no_contact=True \
task.env.termination.enable_screw_limit=True \
${NUM_ENVS_ARG} \
${EXTRA_ARGS}
