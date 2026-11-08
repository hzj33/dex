#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONTAINER_NAME="dexscrew_pasini_m24_nut_tuner"
ISAACGYM_ROOT="${ISAACGYM_DIR:-/data/Codefield/third_party/isaacgym_preview4_py38_clean}"
OUTPUT_PATH="outputs/initpose_tuning/XHandPasiniM24NutBolt_current.yaml"

cd "${REPO_DIR}"

if docker inspect --format '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null | rg -qx 'true'; then
  echo "Paxini M24 nut pose tuner is already running; activating the Isaac Gym window."
  if command -v xdotool >/dev/null 2>&1 && [[ -n "${DISPLAY:-}" ]]; then
    TUNER_XAUTHORITY="${XAUTHORITY:-/run/user/$(id -u)/gdm/Xauthority}"
    DISPLAY="${DISPLAY}" XAUTHORITY="${TUNER_XAUTHORITY}" \
      xdotool search --onlyvisible --name '^Isaac Gym$' windowactivate 2>/dev/null || true
  fi
  exit 0
fi

if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

if [[ ! -f "${ISAACGYM_ROOT}/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so" ]]; then
  echo "IsaacGym Preview 4 Python 3.8 binding not found under: ${ISAACGYM_ROOT}" >&2
  exit 1
fi

export ISAACGYM_DIR="${ISAACGYM_ROOT}"
export DEXSCREW_CONTAINER_NAME="${CONTAINER_NAME}"

exec ./docker-run-isaacgym.sh \
  python scripts/tune_dexh13_lightbulb_initpose.py \
  --task XHandPasiniM24NutBolt \
  --gpu 0 \
  --seed 42 \
  --out "${OUTPUT_PATH}"
