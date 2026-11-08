#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3
STEPS=${4:-256}
TAG=${5:-run_a}
COLLECT_CACHE="${CACHE}_collect"

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:5:$len}")

if ! python - <<'PY' >/dev/null 2>&1
import isaacgym  # noqa: F401
import sys
assert sys.version_info[:2] == (3, 8)
PY
then
  echo "Local runtime is not Isaac Gym compatible (need python 3.8 + isaacgym)."
  echo "Use docker wrapper instead:"
  echo "  scripts/collect_screwdriver_teacher_rollout_docker.sh ${GPUS} ${SEED} ${CACHE} ${STEPS} ${TAG}"
  exit 1
fi

PY_ARGS=(
    train.py
    task=XHandHoraScrewDriver
    train.algo=PPO
    headless=True
    seed=${SEED}
    sim_device=cuda:${GPUS}
    rl_device=cuda:${GPUS}
    graphics_device_id=7
    task.env.numEnvs=64
    test=True
    wandb_activate=False
    train.ppo.output_name=XHandHoraScrewDriver_teacher/${COLLECT_CACHE}
    "checkpoint=outputs/XHandHoraScrewDriver_teacher/${CACHE}/stage1_nn/best_reward_*.pth"
    task.env.randomization.randomizePDGains=False
    task.env.randomization.action_noise_e_scale=0.0
    task.env.randomization.action_noise_t_scale=0.0
    task.env.randomization.obs_noise_e_scale=0.0
    task.env.randomization.obs_noise_t_scale=0.0
    task.env.randomization.noisy_rpy_scale=0.0
    task.env.randomization.noisy_pos_scale=0.0
    task.env.forceScale=0.0
    task.env.randomForceProbScalar=0.0
    +collect_rollout=True
    +collect_steps=${STEPS}
    +collect_save_point_cloud=True
    +collect_out=outputs/teacher_rollouts/XHandHoraScrewDriver_teacher/${CACHE}/${TAG}_seed${SEED}_steps${STEPS}.pt
)

PY_ARGS+=("${EXTRA_ARGS[@]}")
python "${PY_ARGS[@]}"
