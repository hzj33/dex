#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUN_ID="${RUN_ID:-m24_usersaved_contactpose_s42_20260805_153516_ppo30m}"
TASK_CONFIG="${M24_TASK_CONFIG:-XHandPasiniM24NutBolt}"
E002_CALIBRATION_JSON="${E002_CALIBRATION_JSON:-}"
STAGE_DIR="${PROJECT_DIR}/outputs/XHandPasiniM24NutBolt_teacher/${RUN_ID}/stage1_nn"
ISAACGYM_DIR="${ISAACGYM_DIR:-/data/Codefield/third_party/isaacgym_preview4_py38_clean}"
CONTAINER_NAME="${DEXSCREW_CONTAINER_NAME:-dexscrew_m24_ppo_viewer}"
EXTRA_ARGS=("$@")
CALIBRATION_ARGS=()

if [[ -z "${DISPLAY:-}" ]]; then
  echo "DISPLAY is empty. Run this command from the Ubuntu desktop terminal." >&2
  exit 1
fi

CHECKPOINT="$({
  find "${STAGE_DIR}" -maxdepth 1 -type f -name 'best_reward_*.pth' \
    -printf '%T@ %p\n' 2>/dev/null || true
} | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "${CHECKPOINT}" ]] || [[ ! -f "${CHECKPOINT}" ]]; then
  echo "No PPO checkpoint found under: ${STAGE_DIR}" >&2
  exit 1
fi

SNAPSHOT_DIR="${STAGE_DIR}/eval_snapshots"
mkdir -p "${SNAPSHOT_DIR}"
SNAPSHOT="${SNAPSHOT_DIR}/$(date +%Y%m%d_%H%M%S)_$(basename "${CHECKPOINT}")"
cp -- "${CHECKPOINT}" "${SNAPSHOT}"
CHECKPOINT="${SNAPSHOT}"

CHECKPOINT="${CHECKPOINT#${PROJECT_DIR}/}"
echo "Loading M24 PPO evaluation snapshot: ${CHECKPOINT}"

if [[ "${TASK_CONFIG}" == "XHandPasiniM24NutBoltE002" ]]; then
  if [[ -z "${E002_CALIBRATION_JSON}" || ! -f "${E002_CALIBRATION_JSON}" ]]; then
    echo "E002 visualization requires E002_CALIBRATION_JSON from its cloud pipeline." >&2
    exit 1
  fi
  POSE_OVERRIDE="$(python3 - "${E002_CALIBRATION_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
if data.get("status") != "passed":
    raise SystemExit("E002 calibration did not pass")
values = ",".join(f"{float(x):.10g}" for x in data["custom_init_dof_pos"])
print(f"+task.env.customInitDofPos=[{values}]")
PY
)"
  FORCE_SIGN_OVERRIDE="$(python3 - "${E002_CALIBRATION_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"task.env.e002_motion.force_sign={float(data['force_sign']):.1f}")
PY
)"
  CALIBRATION_ARGS=("${POSE_OVERRIDE}" "${FORCE_SIGN_OVERRIDE}")
fi

cd "${PROJECT_DIR}"
ISAACGYM_DIR="${ISAACGYM_DIR}" \
DEXSCREW_CONTAINER_NAME="${CONTAINER_NAME}" \
  exec "${PROJECT_DIR}/docker-run-isaacgym.sh" python -u train.py \
    "task=${TASK_CONFIG}" \
    headless=False \
    test=True \
    seed=42 \
    experiment=rl \
    train.algo=PPO \
    task.env.numEnvs=1 \
    sim_device=cuda:0 \
    rl_device=cuda:0 \
    graphics_device_id=0 \
    wandb_activate=False \
    "checkpoint=${CHECKPOINT}" \
    "${CALIBRATION_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
