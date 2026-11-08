#!/usr/bin/env bash
set -euo pipefail

# E002-A0b/A0c -> A1b only. This pipeline requires automatic stable-pair
# reset/action/sign calibration before one-environment smoke and one five-minute
# nominal PPO probe. It never promotes a probe checkpoint to a 30-minute run.
ROOT="${ROOT:-/root/code/dexscrew-repro}"
CONDA_ENV="${CONDA_ENV:-dexscrew-ig}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-m24_e002_a1b_s${SEED}_$(date +%Y%m%d_%H%M%S)}"
PIPE_DIR="${PIPE_DIR:-outputs/cloud_pipeline_m24_humantwist_e002_${RUN_TAG}}"
TASK="XHandPasiniM24NutBoltE002"
NUM_THREADS="${NUM_THREADS:-16}"
CALIBRATION_CANDIDATES="${CALIBRATION_CANDIDATES:-1024}"
CALIBRATION_TIMEOUT="${CALIBRATION_TIMEOUT:-300}"
CALIBRATION_JSON="${PIPE_DIR}/manifests/a0_calibration.json"

cd "${ROOT}"
[[ ! -e "${PIPE_DIR}" ]] || { echo "Refusing to overwrite ${PIPE_DIR}" >&2; exit 2; }
source /root/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"
mkdir -p "${PIPE_DIR}"/{commands,configs,logs,status,manifests}
trap 'rc=$?; printf "%s\n" "$rc" > "${PIPE_DIR}/exit_status"; date --iso-8601=seconds > "${PIPE_DIR}/finished_at.txt"' EXIT
printf 'running\n' > "${PIPE_DIR}/phase.txt"
date --iso-8601=seconds > "${PIPE_DIR}/started_at.txt"
git rev-parse HEAD > "${PIPE_DIR}/git_commit.txt"
git status --short > "${PIPE_DIR}/git_status_at_start.txt"
cp configs/task/XHandPasiniM24NutBoltE002.yaml "${PIPE_DIR}/configs/"
cp configs/task/XHandPasiniM24NutBolt.yaml "${PIPE_DIR}/configs/"
cp configs/train/XHandPasiniM24NutBoltE002.yaml "${PIPE_DIR}/configs/train_XHandPasiniM24NutBoltE002.yaml"
cp "$0" "${PIPE_DIR}/cloud_run_m24_e002_a1.sh.used"
cp scripts/train_m24_e002_bounded.py "${PIPE_DIR}/train_m24_e002_bounded.py.used"
sha256sum scripts/train_m24_e002_bounded.py dexscrew/algo/ppo/ppo.py dexscrew/tasks/xhand_pasini.py \
  dexscrew/tasks/contact_moments.py configs/task/XHandPasiniM24NutBoltE002.yaml > "${PIPE_DIR}/source_SHA256SUMS"
cp scripts/calibrate_m24_e002_a0.py "${PIPE_DIR}/calibrate_m24_e002_a0.py.used"
cp scripts/trace_m24_e002_a0.py "${PIPE_DIR}/trace_m24_e002_a0.py.used"
nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader > "${PIPE_DIR}/gpu_at_start.csv"

run_a0() {
  set +e
  python scripts/test_m24_e002_a0.py > "${PIPE_DIR}/logs/a0.log" 2>&1
  local rc=$?
  set -e
  printf '%s\n' "${rc}" > "${PIPE_DIR}/status/a0.exit_status"
  [[ "${rc}" == 0 ]] && printf 'passed\n' > "${PIPE_DIR}/status/a0.status" || printf 'failed:%s\n' "${rc}" > "${PIPE_DIR}/status/a0.status"
  return "${rc}"
}

run_calibration() {
  local log="${PIPE_DIR}/logs/a0_calibration.log"
  local command_file="${PIPE_DIR}/commands/a0_calibration.sh"
  local rc
  if [[ -n "${A0C_DIAGNOSTIC_JSON:-}" ]]; then
    # Reuse the accepted fixed-pose diagnostic; never replace it with a fresh
    # A0b/random-action search while transitioning to the authorized short run.
    cp scripts/import_m24_a0c_for_training.py "${PIPE_DIR}/import_m24_a0c_for_training.py.used"
    printf 'python scripts/import_m24_a0c_for_training.py %q %q\n' \
      "${A0C_DIAGNOSTIC_JSON}" "${CALIBRATION_JSON}" > "${command_file}"
    set +e
    python scripts/import_m24_a0c_for_training.py "${A0C_DIAGNOSTIC_JSON}" "${CALIBRATION_JSON}" > "${log}" 2>&1
    rc=$?
    set -e
    printf '%s\n' "${rc}" > "${PIPE_DIR}/status/a0_calibration.exit_status"
    [[ "${rc}" == 0 ]] && printf 'passed:imported_a0c\n' > "${PIPE_DIR}/status/a0_calibration.status" || printf 'failed:a0c_import\n' > "${PIPE_DIR}/status/a0_calibration.status"
    return "${rc}"
  fi
  local -a command=(timeout --signal=TERM --kill-after=30s "${CALIBRATION_TIMEOUT}" python -u
    scripts/calibrate_m24_e002_a0.py "task=${TASK}" headless=True
    sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 "num_threads=${NUM_THREADS}"
    "task.env.numEnvs=${CALIBRATION_CANDIDATES}"
    "+calibration_candidates=${CALIBRATION_CANDIDATES}"
    "+calibration_out=${CALIBRATION_JSON}" "+calibration_seed=${SEED}")
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}" > "${command_file}"
  printf '%q ' "${command[@]}" >> "${command_file}"
  printf '\n' >> "${command_file}"
  printf 'running\n' > "${PIPE_DIR}/status/a0_calibration.status"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}" > "${log}" 2>&1
  rc=$?
  set -e
  printf '%s\n' "${rc}" > "${PIPE_DIR}/status/a0_calibration.exit_status"
  if [[ "${rc}" == 0 ]] && python - "${CALIBRATION_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if data.get("status") == "passed" else 1)
PY
  then
    printf 'passed\n' > "${PIPE_DIR}/status/a0_calibration.status"
    return 0
  fi
  printf 'failed:%s\n' "${rc}" > "${PIPE_DIR}/status/a0_calibration.status"
  return 1
}

load_calibration_overrides() {
  POSE_OVERRIDE="$(python - "${CALIBRATION_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
values = ",".join(f"{float(x):.10g}" for x in data["custom_init_dof_pos"])
print(f"+task.env.customInitDofPos=[{values}]")
PY
)"
  FORCE_SIGN_OVERRIDE="$(python - "${CALIBRATION_JSON}" <<'PY'
import json
import sys
data = json.load(open(sys.argv[1], encoding="utf-8"))
print(f"task.env.e002_motion.force_sign={float(data['force_sign']):.1f}")
PY
)"
  printf '%s\n%s\n' "${POSE_OVERRIDE}" "${FORCE_SIGN_OVERRIDE}" > "${PIPE_DIR}/manifests/calibration_overrides.txt"
}

run_a0_trace() {
  local log="${PIPE_DIR}/logs/a0_trace.log"
  local command_file="${PIPE_DIR}/commands/a0_trace.sh"
  local trace_csv="${PIPE_DIR}/manifests/a0_trace.csv"
  local rc
  local trace_steps
  trace_steps="$(python -c 'import json,sys; print(max(16, json.load(open(sys.argv[1])).get("thresholds", {}).get("probe_steps", 16)))' "${CALIBRATION_JSON}")"
  local -a command=(timeout --signal=TERM --kill-after=30s 180 python -u
    scripts/trace_m24_e002_a0.py "task=${TASK}" headless=True
    sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 "num_threads=${NUM_THREADS}"
    task.env.numEnvs=1 "${POSE_OVERRIDE}" "${FORCE_SIGN_OVERRIDE}"
    task.env.e002_motion.contact_acquire_deadline_steps=10000
    task.env.e002_motion.contact_loss_grace_steps=10000
    "+trace_steps=${trace_steps}" "+trace_calibration=${CALIBRATION_JSON}"
    "+trace_out=${trace_csv}")
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}" > "${command_file}"
  printf '%q ' "${command[@]}" >> "${command_file}"
  printf '\n' >> "${command_file}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}" > "${log}" 2>&1
  rc=$?
  set -e
  printf '%s\n' "${rc}" > "${PIPE_DIR}/status/a0_trace.exit_status"
  [[ "${rc}" == 0 ]] && printf 'passed\n' > "${PIPE_DIR}/status/a0_trace.status" || printf 'failed:%s\n' "${rc}" > "${PIPE_DIR}/status/a0_trace.status"
  return "${rc}"
}

run_phase() {
  local phase="$1" envs="$2" minibatch="$3" limit="$4"
  local output_name="XHandPasiniM24NutBolt_teacher/${RUN_TAG}_${phase}"
  local output_dir="outputs/${output_name}"
  local log="${PIPE_DIR}/logs/${phase}.log"
  local command_file="${PIPE_DIR}/commands/${phase}.sh"
  local started finished rc result checkpoint
  local -a command=(timeout --signal=TERM --kill-after=30s "${limit}" python -u scripts/train_m24_e002_bounded.py
    "task=${TASK}" headless=True "seed=${SEED}" experiment=E002_A1b train.algo=PPO
    sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 "num_threads=${NUM_THREADS}"
    "task.env.numEnvs=${envs}" "train.ppo.num_actors=${envs}"
    "train.ppo.minibatch_size=${minibatch}" "train.ppo.output_name=${output_name}"
    "${POSE_OVERRIDE}" "${FORCE_SIGN_OVERRIDE}"
    wandb_activate=False)
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}" > "${command_file}"
  printf '%q ' "${command[@]}" >> "${command_file}"
  printf '\n' >> "${command_file}"
  started="$(date --iso-8601=seconds)"
  printf 'running\n' > "${PIPE_DIR}/status/${phase}.status"
  { echo "[${phase}] start=${started} envs=${envs} minibatch=${minibatch}"; cat "${command_file}"; } > "${log}"
  set +e
  CUDA_VISIBLE_DEVICES="${GPU}" "${command[@]}" >> "${log}" 2>&1
  rc=$?
  set -e
  finished="$(date --iso-8601=seconds)"
  [[ "${rc}" == 0 ]] && result=completed || { [[ "${rc}" == 124 ]] && result=completed:expected_timeout || result="failed:${rc}"; }
  checkpoint="$(find "${output_dir}/stage1_nn" -maxdepth 1 -type f -name 'best_reward_*.pth' -printf '%T@\t%p\n' 2>/dev/null | sort -rn | head -n 1 | cut -f2- || true)"
  printf '%s\n' "${rc}" > "${PIPE_DIR}/status/${phase}.exit_status"
  printf '%s\n' "${result}" > "${PIPE_DIR}/status/${phase}.status"
  printf '%s\n' "${checkpoint}" > "${PIPE_DIR}/status/${phase}.best_checkpoint"
  if [[ -f "${output_dir}/stage1_nn/last.pth" ]]; then
    printf '%s\n' "${output_dir}/stage1_nn/last.pth" > "${PIPE_DIR}/status/${phase}.last_checkpoint"
  fi
  { echo "[${phase}] end=${finished} exit=${rc} result=${result}"; echo "[${phase}] checkpoint=${checkpoint:-none}"; } >> "${log}"
  PHASE_RC="${rc}"; PHASE_LOG="${log}"; PHASE_CHECKPOINT="${checkpoint}"
}

healthy() {
  [[ "${PHASE_RC}" == 0 || "${PHASE_RC}" == 124 ]] &&
    grep -aq 'Agent Steps:' "${PHASE_LOG}" &&
    ! grep -aEq 'Traceback|CUDA out of memory|illegal memory access|Segmentation fault|PhysX.*[Ee]rror|Error executing job' "${PHASE_LOG}"
}

if ! run_a0; then printf 'failed:a0\n' > "${PIPE_DIR}/phase.txt"; exit 1; fi
if ! run_calibration; then printf 'failed:a0_calibration\n' > "${PIPE_DIR}/phase.txt"; exit 1; fi
load_calibration_overrides
if ! run_a0_trace; then printf 'failed:a0_trace\n' > "${PIPE_DIR}/phase.txt"; exit 1; fi
run_phase smoke_1env 1 12 60
if ! healthy; then printf 'failed:smoke\n' > "${PIPE_DIR}/phase.txt"; exit 1; fi

for pair in '12288 24576' '8192 16384' '6144 12288'; do
  read -r envs minibatch <<< "${pair}"
  run_phase "probe_${envs}_${minibatch}" "${envs}" "${minibatch}" 300
  if healthy; then
    printf '%s %s\n' "${envs}" "${minibatch}" > "${PIPE_DIR}/selected_capacity.txt"
    printf '%s\n' "${PHASE_CHECKPOINT}" > "${PIPE_DIR}/selected_probe_checkpoint.txt"
    printf 'completed:awaiting_learning_evaluation\n' > "${PIPE_DIR}/phase.txt"
    date --iso-8601=seconds > "${PIPE_DIR}/finished_at.txt"
    exit 0
  fi
done
printf 'failed:no_healthy_capacity\n' > "${PIPE_DIR}/phase.txt"
date --iso-8601=seconds > "${PIPE_DIR}/finished_at.txt"
exit 1
