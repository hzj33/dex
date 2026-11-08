#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-42}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-5000}"
ACTION_MODE="${ACTION_MODE:-random}"
# Per-step target delta (radians). Increase for more visible finger motion.
ACTION_SCALE="${ACTION_SCALE:-0.25}"
# IsaacGym DOF order for this asset (see `--print_dof_map`):
# 0:j_00 1:j11 2:j12 3:j2 4:j3 5:j4 6:j5 7:j6 8:j7 9:j8 10:j9 11:j10
# Randomize thumb(j2/j3/j4)+index(j5/j6)+middle(j7/j8); keep ring(j9/j10) & pinky(j11/j12) still.
RANDOM_ACTION_INDICES="${RANDOM_ACTION_INDICES:-3,4,5,6,7,8,9}"

python scripts/debug_env.py \
  --task_config configs/task/RBHandHoraLightbulb.yaml \
  --sim_device "cuda:${GPU_ID}" \
  --graphics_device_id 0 \
  --num_envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --action_mode "${ACTION_MODE}" \
  --action_scale "${ACTION_SCALE}" \
  --random_action_indices "${RANDOM_ACTION_INDICES}" \
  --seed "${SEED}"
