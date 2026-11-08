#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

# Optional 4th arg: HEADLESS (True/False/1/0). Backward compatible:
# - If omitted, defaults to True.
# - If the 4th arg doesn't look like a boolean, it is treated as an extra hydra override.
HEADLESS_RAW=$4

is_bool_arg() {
	case "$1" in
		True|False|true|false|1|0) return 0 ;;
		*) return 1 ;;
	esac
}

to_hydra_bool() {
	case "$1" in
		True|true|1) echo "True" ;;
		False|false|0) echo "False" ;;
		*) echo "True" ;;
	esac
}

if is_bool_arg "$HEADLESS_RAW"; then
	HEADLESS=$(to_hydra_bool "$HEADLESS_RAW")
	EXTRA_START_IDX=4
else
	HEADLESS=True
	EXTRA_START_IDX=3
fi

NUM_ENVS_OVERRIDE=""
if [ "$HEADLESS" = "False" ]; then
	# Visualize while training; reduce load.
	NUM_ENVS_OVERRIDE="task.env.numEnvs=16 train.ppo.minibatch_size=12"
fi

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:$EXTRA_START_IDX:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandHoraScrewDriver headless=${HEADLESS} seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=0.1 \
wandb_activate=False \
train.ppo.output_name=XHandHoraScrewDriver_teacher/${CACHE} \
${NUM_ENVS_OVERRIDE} \
${EXTRA_ARGS}
