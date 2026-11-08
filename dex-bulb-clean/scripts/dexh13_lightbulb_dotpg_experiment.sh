#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "Usage: bash scripts/dexh13_lightbulb_dotpg_experiment.sh <gpu> <seed> <teacher_ckpt> <full|no_lipschitz|no_dual_value|no_proprio_hist> [extra hydra args...]"
  exit 1
fi

GPUS=$1
SEED=$2
TEACHER_CKPT=$3
ABLATION=$4

array=( "$@" )
len=${#array[@]}
EXTRA_ARGS=${array[@]:4:$len}

NUM_ENVS=${NUM_ENVS:-2048}
ADAPT_WARMUP_STEPS=${ADAPT_WARMUP_STEPS:-4000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
EXPL_NOISE=${EXPL_NOISE:-0.005}

OUTPUT_NAME="dotpg_exp_dexh13_lightbulb_${ABLATION}_seed${SEED}"

ABLATION_ARGS=()
case "${ABLATION}" in
  full)
    ABLATION_ARGS=(
      train.dotpg.ablation=full
      train.dotpg.use_proprio_hist=True
      train.dotpg.use_dual_value=True
      train.dotpg.use_q_value=True
      train.dotpg.use_spectral_norm=True
      train.dotpg.use_gradient_penalty=True
      train.dotpg.lambda_gp=10.0
    )
    ;;
  no_lipschitz)
    ABLATION_ARGS=(
      train.dotpg.ablation=no_lipschitz
      train.dotpg.use_proprio_hist=True
      train.dotpg.use_dual_value=True
      train.dotpg.use_q_value=True
      train.dotpg.use_spectral_norm=False
      train.dotpg.use_gradient_penalty=False
      train.dotpg.lambda_gp=0.0
    )
    ;;
  no_dual_value)
    ABLATION_ARGS=(
      train.dotpg.ablation=no_dual_value
      train.dotpg.use_proprio_hist=True
      train.dotpg.use_dual_value=False
      train.dotpg.use_q_value=False
      train.dotpg.use_spectral_norm=True
      train.dotpg.use_gradient_penalty=True
      train.dotpg.lambda_gp=10.0
    )
    ;;
  no_proprio_hist)
    ABLATION_ARGS=(
      train.dotpg.ablation=no_proprio_hist
      train.dotpg.use_proprio_hist=False
      train.dotpg.use_dual_value=True
      train.dotpg.use_q_value=True
      train.dotpg.use_spectral_norm=True
      train.dotpg.use_gradient_penalty=True
      train.dotpg.lambda_gp=10.0
      train.dotpg.adapt_warmup_steps=0
      train.dotpg.freeze_adapt_after_warmup=True
    )
    ;;
  *)
    echo "Unknown ablation: ${ABLATION}"
    echo "Expected one of: full, no_lipschitz, no_dual_value, no_proprio_hist"
    exit 1
    ;;
esac

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py --config-name=config_dotpg_student.yaml \
  task=Dexh13HoraLightbulb \
  train=Dexh13HoraLightbulb_DOTPG_Experiment \
  checkpoint=${TEACHER_CKPT} \
  headless=True graphics_device_id=-1 \
  seed=${SEED} \
  task.env.numEnvs=${NUM_ENVS} \
  train.ppo.output_name=${OUTPUT_NAME} \
  wandb_name=${OUTPUT_NAME} \
  train.dotpg.adapt_warmup_steps=${ADAPT_WARMUP_STEPS} \
  train.dotpg.warmup_steps=${WARMUP_STEPS} \
  train.dotpg.exploration_noise=${EXPL_NOISE} \
  task.env.termination.grace_steps=0 \
  task.env.termination.enable_finger_dist=True \
  task.env.termination.enable_nut_stagnation=True \
  task.env.termination.enable_no_contact=True \
  task.env.termination.enable_screw_limit=True \
  wandb_activate=True \
  "${ABLATION_ARGS[@]}" \
  ${EXTRA_ARGS}
