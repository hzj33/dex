#!/bin/bash
GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

NUM_ENVS=${NUM_ENVS:-2048}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
EXPL_NOISE=${EXPL_NOISE:-0.005}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg.yaml \
  task=XHandHoraScrewDriver \
  checkpoint=${CKPT} \
  headless=True graphics_device_id=-1 \
  seed=${SEED} \
  num_envs=${NUM_ENVS} \
  train.dotpg.state_mode=teacher \
  train.dotpg.warmup_steps=${WARMUP_STEPS} \
  train.dotpg.exploration_noise=${EXPL_NOISE} \
  wandb_activate=True \
  task.env.reset_dist_threshold=0.15 \
  ${EXTRA_ARGS}
