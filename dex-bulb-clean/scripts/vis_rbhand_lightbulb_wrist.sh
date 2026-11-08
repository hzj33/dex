#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-0}"
SEED="${2:-42}"

NUM_ENVS="${NUM_ENVS:-1}"
STEPS="${STEPS:-3000}"

# Reduce micro jitter by default (you can override via env vars)
PD_PGAIN="${PD_PGAIN:-}"
PD_DGAIN="${PD_DGAIN:-0.05}"

# Increase contact friction to reduce slip (override as needed)
SHAPE_FRICTION="${SHAPE_FRICTION:-4.0}"
SHAPE_RESTITUTION="${SHAPE_RESTITUTION:-0.0}"

# Viewer helper: draw local axes on the rotating "nut" (bulb) so rotation is obvious.
DRAW_AXES="${DRAW_AXES:-1}"
AXIS_LENGTH="${AXIS_LENGTH:-0.08}"
AXIS_THICKNESS="${AXIS_THICKNESS:-0.01}"
AXES_FRAME="${AXES_FRAME:-nut}"

# Wrist orbit params (around +Z, centered at nut_pos by default)
# NOTE: In +Z right-hand rule, +yaw is counter-clockwise (拆卸). Use negative for clockwise (装配).
YAW_DEG="${YAW_DEG:--25}"
GRASP_STEPS="${GRASP_STEPS:-40}"
YAW_STEPS="${YAW_STEPS:-60}"
RELEASE_STEPS="${RELEASE_STEPS:-40}"
RETURN_STEPS="${RETURN_STEPS:-60}"

# For this RBHand asset, hand DOF order (see scripts/vis_rbhand_lightbulb.sh):
# 0:j_00 1:j11 2:j12 3:j2 4:j3 5:j4 6:j5 7:j6 8:j7 9:j8 10:j9 11:j10
# Default: thumb(j2/j3/j4)+index(j5/j6)+middle(j7/j8); keep ring(j9/j10) & pinky(j11/j12) still.
ACTIVE_DOF_INDICES="${ACTIVE_DOF_INDICES:-3,4,5,6,7,8,9}"

# 0=open_end (closest to 0), 1=close_end (largest |value|). Tune if grasp is too loose/tight.
OPEN_FRACTION="${OPEN_FRACTION:-0.0}"
CLOSE_FRACTION="${CLOSE_FRACTION:-0.9}"

ARGS=(
  --task_config configs/task/RBHandHoraLightbulb.yaml \
  --sim_device "cuda:${GPU_ID}" \
  --graphics_device_id 0 \
  --num_envs "${NUM_ENVS}" \
  --steps "${STEPS}" \
  --seed "${SEED}" \
  --shape_friction "${SHAPE_FRICTION}" \
  --shape_restitution "${SHAPE_RESTITUTION}" \
  --pd_dgain "${PD_DGAIN}" \
  --yaw_deg "${YAW_DEG}" \
  --grasp_steps "${GRASP_STEPS}" \
  --yaw_steps "${YAW_STEPS}" \
  --release_steps "${RELEASE_STEPS}" \
  --return_steps "${RETURN_STEPS}" \
  --active_dof_indices "${ACTIVE_DOF_INDICES}" \
  --open_fraction "${OPEN_FRACTION}" \
  --close_fraction "${CLOSE_FRACTION}"
)

if [[ -n "${PD_PGAIN}" ]]; then
  ARGS+=(--pd_pgain "${PD_PGAIN}")
fi

if [[ "${DRAW_AXES}" == "1" ]]; then
  ARGS+=(--draw_axes --axis_length "${AXIS_LENGTH}" --axis_thickness "${AXIS_THICKNESS}" --axes_frame "${AXES_FRAME}")
fi

python scripts/debug_env_wrist_twist.py "${ARGS[@]}"
