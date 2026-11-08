#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

RUN_ID="${RUN_ID:-m24_latestpose_s42_$(date +%Y%m%d_%H%M%S)_ppo8192}"
SEED="${SEED:-42}"
NUM_ENVS="${NUM_ENVS:-8192}"
MINIBATCH_SIZE="${MINIBATCH_SIZE:-16384}"
NUM_THREADS="${NUM_THREADS:-16}"
TRAIN_TIMEOUT_SECONDS="${TRAIN_TIMEOUT_SECONDS:-0}"
ISAACGYM_DIR="${ISAACGYM_DIR:-/data/Codefield/third_party/isaacgym_preview4_py38_clean}"
CONTAINER_NAME="${DEXSCREW_CONTAINER_NAME:-dexscrew_m24_ppo_8192}"

PIPELINE_DIR="${PROJECT_DIR}/outputs/local_pipeline_m24_ppo/${RUN_ID}"
TRAIN_OUTPUT_NAME="XHandPasiniM24NutBolt_teacher/${RUN_ID}"
TRAIN_OUTPUT_DIR="${PROJECT_DIR}/outputs/${TRAIN_OUTPUT_NAME}"
LOG_PATH="${PIPELINE_DIR}/train.log"
STATUS_PATH="${PIPELINE_DIR}/status"

mkdir -p "${PIPELINE_DIR}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Docker container already exists: ${CONTAINER_NAME}" >&2
  exit 1
fi

TRAIN_COMMAND=(
  python -u train.py
  task=XHandPasiniM24NutBolt
  headless=True
  "seed=${SEED}"
  experiment=rl
  train.algo=PPO
  "task.env.numEnvs=${NUM_ENVS}"
  "train.ppo.num_actors=${NUM_ENVS}"
  "train.ppo.minibatch_size=${MINIBATCH_SIZE}"
  "train.ppo.output_name=${TRAIN_OUTPUT_NAME}"
  wandb_activate=False
  "num_threads=${NUM_THREADS}"
  graphics_device_id=0
)

if [[ "${TRAIN_TIMEOUT_SECONDS}" -gt 0 ]]; then
  CONTAINER_COMMAND=(
    timeout
    --signal=TERM
    --kill-after=30s
    "${TRAIN_TIMEOUT_SECONDS}"
    "${TRAIN_COMMAND[@]}"
  )
else
  CONTAINER_COMMAND=("${TRAIN_COMMAND[@]}")
fi

COMMAND=(
  "${PROJECT_DIR}/docker-run-isaacgym.sh"
  "${CONTAINER_COMMAND[@]}"
)

printf '%s\n' "${RUN_ID}" > "${PIPELINE_DIR}/run_id.txt"
printf '%s\n' "$$" > "${PIPELINE_DIR}/launcher.pid"
printf '%s\n' "${CONTAINER_NAME}" > "${PIPELINE_DIR}/container_name.txt"
printf '%s\n' "${TRAIN_OUTPUT_DIR}" > "${PIPELINE_DIR}/train_output_dir.txt"
printf '%s\n' "$(git -C "${PROJECT_DIR}" rev-parse HEAD)" > "${PIPELINE_DIR}/git_commit.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "${PIPELINE_DIR}/started_at.txt"
printf 'NUM_ENVS=%s\nMINIBATCH_SIZE=%s\nNUM_THREADS=%s\nSEED=%s\nTRAIN_TIMEOUT_SECONDS=%s\n' \
  "${NUM_ENVS}" "${MINIBATCH_SIZE}" "${NUM_THREADS}" "${SEED}" \
  "${TRAIN_TIMEOUT_SECONDS}" \
  > "${PIPELINE_DIR}/capacity.env"
printf '%q ' "${COMMAND[@]}" > "${PIPELINE_DIR}/command.txt"
printf '\n' >> "${PIPELINE_DIR}/command.txt"
printf 'running\n' > "${STATUS_PATH}"

cd "${PROJECT_DIR}"
set +e
ISAACGYM_DIR="${ISAACGYM_DIR}" \
DEXSCREW_CONTAINER_NAME="${CONTAINER_NAME}" \
  "${COMMAND[@]}" >> "${LOG_PATH}" 2>&1
EXIT_STATUS=$?
set -e

printf '%s\n' "${EXIT_STATUS}" > "${PIPELINE_DIR}/exit_status.txt"
printf '%s\n' "$(date --iso-8601=seconds)" > "${PIPELINE_DIR}/finished_at.txt"
if [[ "${EXIT_STATUS}" -eq 0 ]]; then
  printf 'completed\n' > "${STATUS_PATH}"
elif [[ "${EXIT_STATUS}" -eq 124 ]] && [[ "${TRAIN_TIMEOUT_SECONDS}" -gt 0 ]]; then
  printf 'completed:expected_timeout\n' > "${STATUS_PATH}"
else
  printf 'failed:%s\n' "${EXIT_STATUS}" > "${STATUS_PATH}"
fi

exit "${EXIT_STATUS}"
