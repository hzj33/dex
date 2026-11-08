#!/bin/bash
GPUS=$1
SEED=$2
CKPT=$3

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg_student.yaml \
  task=XHandHoraScrewDriver headless=False seed=${SEED} \
  num_envs=10 test=True \
  wandb_activate=False \
  train.load_path=${CKPT} \
  ${EXTRA_ARGS}

