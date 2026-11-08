#!/usr/bin/env bash
set -euo pipefail

# One bounded cloud experiment: E001 / R1-motion.  Capacity probes are short
# and serial; exactly one formal PPO phase is launched from scratch after the
# first healthy probe.  This script is intentionally independent of legacy
# R0/R1/R3 reward-teaching pipelines.

ROOT="${ROOT:-/root/code/dexscrew-repro}"
CONDA_ENV="${CONDA_ENV:-dexscrew-ig}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
NUM_THREADS="${NUM_THREADS:-16}"
PROBE_TIMEOUT_SECONDS="${PROBE_TIMEOUT_SECONDS:-180}"
FORMAL_TIMEOUT_SECONDS="${FORMAL_TIMEOUT_SECONDS:-1800}"
RUN_TAG="${RUN_TAG:-m24_humantwist_e001_r1motion_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="${PIPE_DIR:-outputs/cloud_pipeline_m24_humantwist_e001_${RUN_TAG}}"
TASK="XHandPasiniM24NutBolt"

cd "${ROOT}"
if [[ ! "${RUN_TAG}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG must contain only letters, digits, dot, underscore, or dash." >&2
  exit 2
fi
if [[ -e "${PIPE_DIR}" ]]; then
  echo "Refusing to overwrite pipeline directory: ${PIPE_DIR}" >&2
  exit 2
fi
if ! nvidia-smi -i "${GPU}" >/dev/null 2>&1; then
  echo "GPU ${GPU} is not available." >&2
  exit 2
fi
if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck disable=SC1091
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "Could not locate conda.sh for the cloud runtime." >&2
  exit 2
fi
conda activate "${CONDA_ENV}"

mkdir -p "${PIPE_DIR}"/{commands,configs,logs,status,manifests}
started_at="$(date --iso-8601=seconds)"
git_commit="$(git rev-parse HEAD)"
printf '%s\n' "running" > "${PIPE_DIR}/phase.txt"
printf '%s\n' "${started_at}" > "${PIPE_DIR}/started_at.txt"
printf '%s\n' "${git_commit}" > "${PIPE_DIR}/git_commit.txt"
git status --short > "${PIPE_DIR}/git_status_at_start.txt"
git diff --binary HEAD > "${PIPE_DIR}/git_diff_at_start.patch"
cp configs/task/XHandPasiniM24NutBolt.yaml "${PIPE_DIR}/configs/"
cp configs/task/XHandPasiniNutBolt.yaml "${PIPE_DIR}/configs/"
cp configs/train/XHandPasiniNutBolt.yaml "${PIPE_DIR}/configs/"
cp "$0" "${PIPE_DIR}/cloud_run_m24_humantwist_e001.sh.used"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version \
  --format=csv,noheader > "${PIPE_DIR}/gpu_at_start.csv"

has_fatal_marker() {
  grep -aEq 'Traceback \(most recent call last\)|CUDA out of memory|illegal memory access|Segmentation fault|PhysX.*[Ee]rror|Error executing job' "$1"
}

probe_healthy() {
  local exit_status="$1"
  local log_path="$2"
  [[ "${exit_status}" == "0" || "${exit_status}" == "124" ]] || return 1
  grep -aq 'Agent Steps:' "${log_path}" || return 1
  ! has_fatal_marker "${log_path}"
}

select_best_checkpoint() {
  local output_dir="$1"
  find "${output_dir}/stage1_nn" -maxdepth 1 -type f -name 'best_reward_*.pth' \
    -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -n 1 | cut -f2-
}

run_phase() {
  local phase="$1"
  local num_envs="$2"
  local minibatch="$3"
  local phase_timeout="$4"
  local output_name="XHandPasiniM24NutBolt_teacher/${RUN_TAG}_${phase}"
  local output_dir="outputs/${output_name}"
  local log_path="${PIPE_DIR}/logs/${phase}.log"
  local command_path="${PIPE_DIR}/commands/${phase}.sh"
  local phase_started phase_finished exit_status result checkpoint
  local -a command=(
    timeout --signal=TERM --kill-after=30s "${phase_timeout}"
    python -u train.py
    "task=${TASK}"
    headless=True
    "seed=${SEED}"
    experiment=E001_R1_motion
    train.algo=PPO
    sim_device=cuda:0
    rl_device=cuda:0
    graphics_device_id=0
    "num_threads=${NUM_THREADS}"
    "task.env.numEnvs=${num_envs}"
    "train.ppo.num_actors=${num_envs}"
    "train.ppo.minibatch_size=${minibatch}"
    "train.ppo.output_name=${output_name}"
    wandb_activate=False
  )
  {
    printf '# started_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
    printf '%q ' "${command[@]}"
    printf '\n'
  } > "${command_path}"
  phase_started="$(date --iso-8601=seconds)"
  printf 'running\n' > "${PIPE_DIR}/status/${phase}.status"
  {
    echo "[${phase}] start=${phase_started} envs=${num_envs} minibatch=${minibatch}"
    cat "${command_path}"
  } > "${log_path}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}" >> "${log_path}" 2>&1
  exit_status=$?
  set -e
  phase_finished="$(date --iso-8601=seconds)"
  if [[ "${exit_status}" == "0" ]]; then
    result="completed"
  elif [[ "${exit_status}" == "124" ]]; then
    result="completed:expected_timeout"
  else
    result="failed:${exit_status}"
  fi
  checkpoint="$(select_best_checkpoint "${output_dir}")"
  {
    echo "[${phase}] end=${phase_finished} exit_status=${exit_status} result=${result}"
    echo "[${phase}] selected_best_checkpoint=${checkpoint:-none}"
  } >> "${log_path}"
  printf '%s\n' "${exit_status}" > "${PIPE_DIR}/status/${phase}.exit_status"
  printf '%s\n' "${result}" > "${PIPE_DIR}/status/${phase}.status"
  printf '%s\n' "${phase_started}" > "${PIPE_DIR}/status/${phase}.started_at"
  printf '%s\n' "${phase_finished}" > "${PIPE_DIR}/status/${phase}.finished_at"
  printf '%s\n' "${checkpoint}" > "${PIPE_DIR}/status/${phase}.best_checkpoint"
  if [[ -n "${checkpoint}" ]]; then sha256sum "${checkpoint}" > "${PIPE_DIR}/status/${phase}.best_checkpoint.sha256"; fi
  python - "${PIPE_DIR}/manifests/${phase}.json" <<PY
import json
data = {
  "experiment": "E001", "reward_version": "R1-motion", "phase": "${phase}",
  "started_at": "${phase_started}", "finished_at": "${phase_finished}",
  "exit_status": ${exit_status}, "result": "${result}", "git_commit": "${git_commit}",
  "seed": ${SEED}, "num_envs": ${num_envs}, "minibatch_size": ${minibatch},
  "timeout_seconds": ${phase_timeout}, "command_file": "${command_path}",
  "log": "${log_path}", "output_dir": "${output_dir}",
  "selected_best_checkpoint": "${checkpoint}" or None,
  "action_mask_indices": [4, 5, 6, 7, 8, 9, 10, 11],
}
with open(__import__('sys').argv[1], 'w', encoding='utf-8') as handle:
    json.dump(data, handle, indent=2, sort_keys=True)
    handle.write('\n')
PY
  LAST_EXIT_STATUS="${exit_status}"
  LAST_LOG_PATH="${log_path}"
  LAST_CHECKPOINT="${checkpoint}"
}

# Ordered capacity contract for >=24 GB usable VRAM.
declare -a CAPACITY_PAIRS=("12288 24576" "8192 16384" "6144 12288")
selected_envs=""
selected_minibatch=""
# First verify that R1's phase buffers and diagnostics survive a real one-env
# GPU rollout before allocating the high-capacity PhysX/contact buffers.
run_phase smoke_1env 1 12 60
if ! probe_healthy "${LAST_EXIT_STATUS}" "${LAST_LOG_PATH}"; then
  printf 'failed:one_env_smoke\n' > "${PIPE_DIR}/phase.txt"
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_finish.csv" || true
  exit 1
fi
for pair in "${CAPACITY_PAIRS[@]}"; do
  read -r candidate_envs candidate_minibatch <<< "${pair}"
  probe_name="probe_${candidate_envs}_${candidate_minibatch}"
  run_phase "${probe_name}" "${candidate_envs}" "${candidate_minibatch}" "${PROBE_TIMEOUT_SECONDS}"
  if probe_healthy "${LAST_EXIT_STATUS}" "${LAST_LOG_PATH}"; then
    selected_envs="${candidate_envs}"
    selected_minibatch="${candidate_minibatch}"
    printf '%s\n' "${selected_envs} ${selected_minibatch}" > "${PIPE_DIR}/selected_capacity.txt"
    break
  fi
done

if [[ -z "${selected_envs}" ]]; then
  printf 'failed:no_healthy_capacity_probe\n' > "${PIPE_DIR}/phase.txt"
  nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_finish.csv" || true
  exit 1
fi

run_phase formal "${selected_envs}" "${selected_minibatch}" "${FORMAL_TIMEOUT_SECONDS}"
if [[ "${LAST_EXIT_STATUS}" == "0" ]]; then
  printf 'completed\n' > "${PIPE_DIR}/phase.txt"
elif [[ "${LAST_EXIT_STATUS}" == "124" ]]; then
  printf 'completed:expected_timeout\n' > "${PIPE_DIR}/phase.txt"
else
  printf 'failed:formal_exit_%s\n' "${LAST_EXIT_STATUS}" > "${PIPE_DIR}/phase.txt"
fi
printf '%s\n' "$(date --iso-8601=seconds)" > "${PIPE_DIR}/finished_at.txt"
nvidia-smi --query-gpu=name,memory.used,memory.free,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_finish.csv" || true
printf '%s\n' "${LAST_CHECKPOINT}" > "${PIPE_DIR}/selected_teacher_checkpoint.txt"
if [[ "${LAST_EXIT_STATUS}" == "0" || "${LAST_EXIT_STATUS}" == "124" ]]; then
  exit 0
fi
exit "${LAST_EXIT_STATUS}"
