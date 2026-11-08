#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

CKPT=${CKPT:?Set CKPT to a ConsistencyLatentStudent checkpoint}
GPU=${GPU:-0}
RUN_TAG=${RUN_TAG:-eval_consistency_$(date +%Y%m%d_%H%M%S)}
EVAL_ROOT=${EVAL_ROOT:-outputs/local_eval_consistency_g2}
EVAL_DIR="${EVAL_ROOT}/${RUN_TAG}"
LOG_DIR="${EVAL_DIR}/logs"

EVAL_SEEDS=${EVAL_SEEDS:-42}
EVAL_STEPS=${EVAL_STEPS:-256}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-600}
CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}
CONSISTENCY_USE_EMA_TARGET=${CONSISTENCY_USE_EMA_TARGET:-False}
CONSISTENCY_EMA_DECAY=${CONSISTENCY_EMA_DECAY:-0.999}
CONSISTENCY_INFER_USE_EMA=${CONSISTENCY_INFER_USE_EMA:-False}
CONSISTENCY_TRAIN_ALIGN_INFER=${CONSISTENCY_TRAIN_ALIGN_INFER:-False}

GPU_UTIL_MAX=${GPU_UTIL_MAX:-80}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-80}
GPU_GUARD_SKIP=${GPU_GUARD_SKIP:-False}

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
    echo "[gpu_guard] over budget; not starting local consistency eval" >&2
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

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing checkpoint: ${CKPT}" >&2
  exit 2
fi

guard_gpu
mkdir -p "${LOG_DIR}"

{
  echo "run_tag=${RUN_TAG}"
  echo "date=$(date -Iseconds)"
  echo "root=${ROOT}"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "checkpoint=${CKPT}"
  echo "eval_seeds=${EVAL_SEEDS}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
  echo "consistency_infer_steps=${CONSISTENCY_INFER_STEPS}"
  echo "consistency_use_ema_target=${CONSISTENCY_USE_EMA_TARGET}"
  echo "consistency_ema_decay=${CONSISTENCY_EMA_DECAY}"
  echo "consistency_infer_use_ema=${CONSISTENCY_INFER_USE_EMA}"
  echo "consistency_train_align_infer=${CONSISTENCY_TRAIN_ALIGN_INFER}"
} > "${EVAL_DIR}/manifest.env"

printf 'seed\tstatus\tsteps\tavg_reward\tavg_done_rate\tlatent_mse\tlatent_l1\taction_mse_to_teacher\tlog\n' > "${EVAL_DIR}/eval_summary.tsv"

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
    train.algo=ConsistencyLatentStudent
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
    ++train.ppo.consistency_infer_steps="${CONSISTENCY_INFER_STEPS}"
    ++train.ppo.consistency_use_ema_target="${CONSISTENCY_USE_EMA_TARGET}"
    ++train.ppo.consistency_ema_decay="${CONSISTENCY_EMA_DECAY}"
    ++train.ppo.consistency_infer_use_ema="${CONSISTENCY_INFER_USE_EMA}"
    ++train.ppo.consistency_stochastic_infer=False
    ++train.ppo.consistency_train_align_infer="${CONSISTENCY_TRAIN_ALIGN_INFER}"
  )
  set +e
  run_and_log "${eval_log}" "${eval_cmd[@]}"
  eval_status=$?
  set -e

  eval_summary="$(rg 'EvalSummary' "${eval_log}" | tail -1 || true)"
  recon_summary="$(rg 'EvalReconSummary' "${eval_log}" | tail -1 || true)"

  steps="$(extract_metric steps "${eval_summary}")"
  reward="$(extract_metric avg_reward "${eval_summary}")"
  done_rate="$(extract_metric avg_done_rate "${eval_summary}")"
  latent_mse="$(extract_metric latent_mse "${recon_summary}")"
  latent_l1="$(extract_metric latent_l1 "${recon_summary}")"
  action_mse_teacher="$(extract_metric action_mse_to_teacher "${recon_summary}")"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${eval_seed}" "${eval_status}" "${steps}" "${reward}" "${done_rate}" \
    "${latent_mse}" "${latent_l1}" "${action_mse_teacher}" "${eval_log}" \
    >> "${EVAL_DIR}/eval_summary.tsv"
done

echo "eval_dir=${EVAL_DIR}"
echo "eval_summary=${EVAL_DIR}/eval_summary.tsv"
