#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-g1_residual_consistency_s${SEED}_$(date +%Y%m%d_%H%M%S)}
PROBE_ROOT=${PROBE_ROOT:-outputs/local_probe_residual_consistency_g1}
PROBE_DIR="${PROBE_ROOT}/${RUN_TAG}"
LOG_DIR="${PROBE_DIR}/logs"

BASE_CKPT=${BASE_CKPT:-sim2real/codrive/model_best_codrive.ckpt}
TRAIN_WINDOW_SEC=${TRAIN_WINDOW_SEC:-1800}
NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-200000}

EVAL_SEEDS=${EVAL_SEEDS:-42}
EVAL_STEPS=${EVAL_STEPS:-256}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-600}

GPU_UTIL_MAX=${GPU_UTIL_MAX:-80}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-80}
GPU_GUARD_SKIP=${GPU_GUARD_SKIP:-False}

RESIDUAL_TARGET_SCALE=${RESIDUAL_TARGET_SCALE:-1.0}
RESIDUAL_GATE=${RESIDUAL_GATE:-0.25}
RESIDUAL_RECON_COEF=${RESIDUAL_RECON_COEF:-0.25}
RESIDUAL_ACTION_DELTA_COEF=${RESIDUAL_ACTION_DELTA_COEF:-0.0}
BASE_ACTION_ANCHOR_COEF=${BASE_ACTION_ANCHOR_COEF:-0.05}
CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}

EXTRA_TRAIN_ARGS=("$@")

guard_gpu() {
  if [[ "${GPU_GUARD_SKIP}" == "True" || "${GPU_GUARD_SKIP}" == "true" || "${GPU_GUARD_SKIP}" == "1" ]]; then
    echo "[gpu_guard] skipped"
    return 0
  fi

  local line used total util mem_pct
  line="$(nvidia-smi --id="${GPU}" --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits)"
  IFS=',' read -r used total util <<< "${line}"
  used="${used// /}"
  total="${total// /}"
  util="${util// /}"
  mem_pct=$(( used * 100 / total ))
  echo "[gpu_guard] gpu=${GPU} mem=${used}/${total}MiB (${mem_pct}%) util=${util}% limits=mem<=${GPU_MEM_PCT_MAX}% util<=${GPU_UTIL_MAX}%"
  if (( mem_pct > GPU_MEM_PCT_MAX || util > GPU_UTIL_MAX )); then
    echo "[gpu_guard] over budget; not starting local diffusion probe" >&2
    exit 75
  fi
}

run_and_log() {
  local log_path="$1"
  shift
  {
    echo "[run] start $(date -Iseconds)"
    printf '[run] command:'
    printf ' %q' "$@"
    printf '\n'
  } | tee "${log_path}"
  set +e
  "$@" 2>&1 | tee -a "${log_path}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "[run] exit_status=${status} end $(date -Iseconds)" | tee -a "${log_path}"
  return "${status}"
}

extract_metric() {
  local key="$1"
  local line="$2"
  printf '%s\n' "${line}" | sed -nE "s/.*${key}=([-+0-9.eE]+).*/\\1/p"
}

if [[ ! -f "${BASE_CKPT}" ]]; then
  echo "Missing PAdapt base checkpoint: ${BASE_CKPT}" >&2
  exit 2
fi

guard_gpu

mkdir -p "${LOG_DIR}"

cp configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml "${PROBE_DIR}/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml.used"
cp configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml "${PROBE_DIR}/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml.used"
cp "${BASE_CKPT}" "${PROBE_DIR}/model_best_codrive.ckpt.used"

{
  echo "run_tag=${RUN_TAG}"
  echo "date=$(date -Iseconds)"
  echo "root=${ROOT}"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "base_ckpt=${BASE_CKPT}"
  echo "train_window_sec=${TRAIN_WINDOW_SEC}"
  echo "num_envs=${NUM_ENVS}"
  echo "minibatch=${MINIBATCH}"
  echo "student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS}"
  echo "residual_target_scale=${RESIDUAL_TARGET_SCALE}"
  echo "residual_gate=${RESIDUAL_GATE}"
  echo "residual_recon_coef=${RESIDUAL_RECON_COEF}"
  echo "residual_action_delta_coef=${RESIDUAL_ACTION_DELTA_COEF}"
  echo "base_action_anchor_coef=${BASE_ACTION_ANCHOR_COEF}"
  echo "consistency_infer_steps=${CONSISTENCY_INFER_STEPS}"
  echo "eval_seeds=${EVAL_SEEDS}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
} > "${PROBE_DIR}/manifest.env"

train_cmd=(
  ./docker-run-isaacgym.sh
  timeout "${TRAIN_WINDOW_SEC}"
  scripts/run_with_cleanup.sh
  env
  DEXSCREW_SKIP_GIT_DIFF=1
  NUM_ENVS="${NUM_ENVS}"
  MINIBATCH="${MINIBATCH}"
  STUDENT_MAX_AGENT_STEPS="${STUDENT_MAX_AGENT_STEPS}"
  RESIDUAL_TARGET_SCALE="${RESIDUAL_TARGET_SCALE}"
  RESIDUAL_GATE="${RESIDUAL_GATE}"
  RESIDUAL_RECON_COEF="${RESIDUAL_RECON_COEF}"
  RESIDUAL_ACTION_DELTA_COEF="${RESIDUAL_ACTION_DELTA_COEF}"
  BASE_ACTION_ANCHOR_COEF="${BASE_ACTION_ANCHOR_COEF}"
  CONSISTENCY_INFER_STEPS="${CONSISTENCY_INFER_STEPS}"
  bash scripts/dexh13_lightbulb_student_residual_consistency_codrive.sh
  "${GPU}" "${SEED}" "${RUN_TAG}" "${BASE_CKPT}"
  "${EXTRA_TRAIN_ARGS[@]}"
)

set +e
run_and_log "${LOG_DIR}/train.log" "${train_cmd[@]}"
train_status=$?
set -e
echo "${train_status}" > "${PROBE_DIR}/train.status"
if [[ "${train_status}" -ne 0 && "${train_status}" -ne 124 ]]; then
  echo "Training failed with status ${train_status}; see ${LOG_DIR}/train.log" >&2
  exit "${train_status}"
fi

CKPT="outputs/Dexh13HoraLightbulb_student_residual_consistency_codrive/${RUN_TAG}/stage2_residual_consistency_nn/model_best.ckpt"
if [[ ! -f "${CKPT}" ]]; then
  echo "Training did not produce expected checkpoint: ${CKPT}" >&2
  exit 3
fi
echo "${CKPT}" > "${PROBE_DIR}/checkpoint.txt"

printf 'seed\tstatus\tsteps\tavg_reward\tavg_done_rate\tlatent_mse\tresidual_mse\taction_mse_to_teacher\tdelta_to_base_ratio\taction_mse_to_base\tsaturation_ratio\tlog\n' > "${PROBE_DIR}/eval_summary.tsv"

read -r -a eval_seed_array <<< "${EVAL_SEEDS}"
for eval_seed in "${eval_seed_array[@]}"; do
  guard_gpu
  eval_log="${LOG_DIR}/eval_seed${eval_seed}.log"
  eval_cmd=(
    ./docker-run-isaacgym.sh
    env
    CUDA_VISIBLE_DEVICES="${GPU}"
    timeout "${EVAL_TIMEOUT_SEC}"
    scripts/run_with_cleanup.sh
    python train.py
    task=Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
    train.algo=ResidualConsistencyLatentStudent
    test=True
    headless=True
    train.ppo.proprio_adapt=True
    seed="${eval_seed}"
    sim_device=cuda:0
    rl_device=cuda:0
    graphics_device_id=0
    wandb_activate=False
    task.env.numEnvs="${EVAL_NUM_ENVS}"
    train.ppo.minibatch_size=12
    train.ppo.output_name="eval_${RUN_TAG}_seed${eval_seed}"
    task.env.termination.grace_steps=0
    task.env.termination.enable_finger_dist=True
    task.env.termination.enable_nut_stagnation=True
    task.env.termination.enable_no_contact=True
    task.env.termination.enable_screw_limit=True
    task.env.randomization.obs_noise_t_scale=0.01
    task.env.randomization.obs_noise_e_scale=0.02
    checkpoint="${CKPT}"
    +test_num_steps="${EVAL_STEPS}"
    ++train.ppo.consistency_residual_target_scale="${RESIDUAL_TARGET_SCALE}"
    ++train.ppo.consistency_residual_gate="${RESIDUAL_GATE}"
    ++train.ppo.consistency_infer_steps="${CONSISTENCY_INFER_STEPS}"
    ++train.ppo.consistency_stochastic_infer=False
    ++train.ppo.consistency_train_align_infer=False
  )
  set +e
  run_and_log "${eval_log}" "${eval_cmd[@]}"
  eval_status=$?
  set -e

  eval_summary="$(rg 'EvalSummary' "${eval_log}" | tail -1 || true)"
  recon_summary="$(rg 'EvalReconSummary' "${eval_log}" | tail -1 || true)"
  residual_summary="$(rg 'EvalResidualSummary' "${eval_log}" | tail -1 || true)"

  steps="$(extract_metric steps "${eval_summary}")"
  reward="$(extract_metric avg_reward "${eval_summary}")"
  done_rate="$(extract_metric avg_done_rate "${eval_summary}")"
  latent_mse="$(extract_metric latent_mse "${recon_summary}")"
  residual_mse="$(extract_metric residual_mse "${recon_summary}")"
  action_mse_teacher="$(extract_metric action_mse_to_teacher "${recon_summary}")"
  delta_ratio="$(extract_metric delta_to_base_ratio "${residual_summary}")"
  action_mse_base="$(extract_metric action_mse_to_base "${residual_summary}")"
  saturation="$(extract_metric saturation_ratio "${residual_summary}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${eval_seed}" "${eval_status}" "${steps}" "${reward}" "${done_rate}" \
    "${latent_mse}" "${residual_mse}" "${action_mse_teacher}" \
    "${delta_ratio}" "${action_mse_base}" "${saturation}" "${eval_log}" \
    >> "${PROBE_DIR}/eval_summary.tsv"
done

echo "probe_dir=${PROBE_DIR}"
echo "checkpoint=${CKPT}"
echo "eval_summary=${PROBE_DIR}/eval_summary.tsv"
