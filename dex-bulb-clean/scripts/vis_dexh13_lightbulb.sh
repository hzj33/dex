#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-42}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-5000}"
PRINT_EVERY="${PRINT_EVERY:-200}"
ACTION_MODE="${ACTION_MODE:-random}"
ACTION_SCALE="${ACTION_SCALE:-0.25}"
SIM_DEVICE="${SIM_DEVICE:-cuda:${GPU_ID}}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"
HEADLESS="${HEADLESS:-0}"

# Dexh13 (URDF 默认关节顺序通常为):
# 0-3: index, 4-7: middle, 8-11: ring, 12-15: thumb
# 默认随机化：食指/中指/拇指的弯曲关节（不随机化每指的 joint_0 侧摆）
RANDOM_ACTION_INDICES="${RANDOM_ACTION_INDICES:-1,2,3,5,6,7,13,14,15}"

EXTRA_ARGS=()
if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" ]]; then
  EXTRA_ARGS+=(--headless)
fi

python scripts/debug_env.py \
  --task_config configs/task/Dexh13HoraLightbulb.yaml \
  --sim_device "${SIM_DEVICE}" \
  --graphics_device_id "${GRAPHICS_DEVICE_ID}" \
  --num_envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --print_every "${PRINT_EVERY}" \
  --action_mode "${ACTION_MODE}" \
  --action_scale "${ACTION_SCALE}" \
  --random_action_indices "${RANDOM_ACTION_INDICES}" \
  --seed "${SEED}" \
  "${EXTRA_ARGS[@]}"
