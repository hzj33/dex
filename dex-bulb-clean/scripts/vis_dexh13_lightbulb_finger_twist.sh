#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-42}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-3000}"
PRINT_EVERY="${PRINT_EVERY:-100}"
HEADLESS="${HEADLESS:-0}"
GRAPHICS_DEVICE_ID="${GRAPHICS_DEVICE_ID:-0}"

# PD（torque_control 下生效）：适当增大 D 可减少抖动（可按需覆盖）
PD_PGAIN="${PD_PGAIN:-}"
PD_DGAIN="${PD_DGAIN:-0.05}"

# 提高接触摩擦，减少打滑（仅用于 demo 观察；训练建议用域随机化处理）
SHAPE_FRICTION="${SHAPE_FRICTION:-4.0}"
SHAPE_RESTITUTION="${SHAPE_RESTITUTION:-0.0}"

# 开环周期：抓取 -> 扭转 -> 松开
GRASP_STEPS="${GRASP_STEPS:-80}"
TWIST_STEPS="${TWIST_STEPS:-240}"
RELEASE_STEPS="${RELEASE_STEPS:-80}"

# 抓取阶段（合拢/张开）：控制弯曲关节（是“比例”，不是弧度）
# - index/middle/ring：joint_1/2/3
# - thumb：joint_2/3（joint_1 用 THUMB_J1_BIAS 单独控制横摆）
# - OPEN_FRACTION：更张开（0~1）
# - CLOSE_FRACTION：更合拢（0~1）
OPEN_FRACTION="${OPEN_FRACTION:-0.0}"
CLOSE_FRACTION="${CLOSE_FRACTION:-0.65}"

# 约定：-1 表示“沿用全局 OPEN_FRACTION / CLOSE_FRACTION”
USE_GLOBAL_FRACTION="-1"

# ── 拇指分两组独立控制 ──
# A) joint_1（横摆根关节）：常量偏置 THUMB_J1_BIAS（rad），控制靠拢灯泡
# B) joint_2/3（弯曲关节）：比例控制 THUMB_FLEX_*_FRACTION（0~1，-1=沿用全局）
THUMB_J1_BIAS="${THUMB_J1_BIAS:-1.25}"
THUMB_FLEX_OPEN_FRACTION="${THUMB_FLEX_OPEN_FRACTION:-${USE_GLOBAL_FRACTION}}"
THUMB_FLEX_CLOSE_FRACTION="${THUMB_FLEX_CLOSE_FRACTION:-${USE_GLOBAL_FRACTION}}"

# ── 扭转阶段（joint_0 正弦摆动，单位 rad）──
# index/middle/ring：幅度 TWIST_AMP，周期 TWIST_PERIOD（步数）
# 拇指 joint_0：偏置 THUMB_ROOT_BIAS + 摆幅 THUMB_TWIST_AMP（0=不摆）
TWIST_AMP="${TWIST_AMP:-0.25}"
TWIST_PERIOD="${TWIST_PERIOD:-60}"
DEFAULT_THUMB_ROOT_BIAS="-0.2"
THUMB_ROOT_BIAS="${THUMB_ROOT_BIAS:-${DEFAULT_THUMB_ROOT_BIAS}}"
THUMB_TWIST_AMP="${THUMB_TWIST_AMP:-0}"

# 可选：不改 YAML 的快速微调（CSV）
# HAND_ROOT_POS="0.12,0.02,0.10"
# HAND_ROOT_RPY="3.1416,0,3.1416"
HAND_ROOT_POS="${HAND_ROOT_POS:-}"
HAND_ROOT_RPY="${HAND_ROOT_RPY:-}"

ARGS=(
  --task_config configs/task/Dexh13HoraLightbulb.yaml \
  --sim_device "cuda:${GPU_ID}" \
  --graphics_device_id "${GRAPHICS_DEVICE_ID}" \
  --num_envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --print_every "${PRINT_EVERY}" \
  --seed "${SEED}" \
  --shape_friction "${SHAPE_FRICTION}" \
  --shape_restitution "${SHAPE_RESTITUTION}" \
  --pd_dgain "${PD_DGAIN}" \
  --grasp_steps "${GRASP_STEPS}" \
  --twist_steps "${TWIST_STEPS}" \
  --release_steps "${RELEASE_STEPS}" \
  --open_fraction "${OPEN_FRACTION}" \
  --close_fraction "${CLOSE_FRACTION}" \
  --thumb_j1_bias "${THUMB_J1_BIAS}" \
  --thumb_open_fraction "${THUMB_FLEX_OPEN_FRACTION}" \
  --thumb_close_fraction "${THUMB_FLEX_CLOSE_FRACTION}" \
  --twist_amp "${TWIST_AMP}" \
  --thumb_root_bias "${THUMB_ROOT_BIAS}" \
  --thumb_twist_amp "${THUMB_TWIST_AMP}" \
  --twist_period "${TWIST_PERIOD}"
)

if [[ "${HEADLESS}" == "1" || "${HEADLESS}" == "true" ]]; then
  ARGS+=(--headless)
fi

if [[ -n "${PD_PGAIN}" ]]; then
  ARGS+=(--pd_pgain "${PD_PGAIN}")
fi
if [[ -n "${HAND_ROOT_POS}" ]]; then
  ARGS+=(--hand_root_pos "${HAND_ROOT_POS}")
fi
if [[ -n "${HAND_ROOT_RPY}" ]]; then
  ARGS+=(--hand_root_rpy "${HAND_ROOT_RPY}")
fi

python scripts/debug_env_dexh13_finger_twist.py "${ARGS[@]}"
