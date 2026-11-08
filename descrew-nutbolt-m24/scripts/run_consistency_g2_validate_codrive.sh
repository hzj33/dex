#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
SEED=${SEED:-42}
RUN_TAG=${RUN_TAG:-g2_consistency_s${SEED}_$(date +%Y%m%d_%H%M%S)}
PROBE_ROOT=${PROBE_ROOT:-outputs/local_probe_consistency_g2}
PROBE_DIR="${PROBE_ROOT}/${RUN_TAG}"
LOG_DIR="${PROBE_DIR}/logs"

TEACHER_CKPT=${TEACHER_CKPT:-sim2real/codrive/best_reward_4159.37.pth}
TRAIN_WINDOW_SEC=${TRAIN_WINDOW_SEC:-1800}
NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-200000}
STUDENT_PROGRESS_LOG_INTERVAL=${STUDENT_PROGRESS_LOG_INTERVAL:-25000}

EVAL_SEEDS=${EVAL_SEEDS:-42}
EVAL_STEPS=${EVAL_STEPS:-256}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-600}

GPU_UTIL_MAX=${GPU_UTIL_MAX:-80}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-80}
GPU_GUARD_SKIP=${GPU_GUARD_SKIP:-False}

CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}
CONSISTENCY_NUM_SCALES=${CONSISTENCY_NUM_SCALES:-10}
CONSISTENCY_LR=${CONSISTENCY_LR:-3e-4}
CONSISTENCY_LOSS_COEF=${CONSISTENCY_LOSS_COEF:-1.0}
CONSISTENCY_BOUNDARY_COEF=${CONSISTENCY_BOUNDARY_COEF:-0.5}
CONSISTENCY_USE_EMA_TARGET=${CONSISTENCY_USE_EMA_TARGET:-False}
CONSISTENCY_EMA_DECAY=${CONSISTENCY_EMA_DECAY:-0.999}
CONSISTENCY_INFER_USE_EMA=${CONSISTENCY_INFER_USE_EMA:-False}
CONSISTENCY_TRAIN_ALIGN_INFER=${CONSISTENCY_TRAIN_ALIGN_INFER:-False}
CONSISTENCY_LR_SCHEDULE=${CONSISTENCY_LR_SCHEDULE:-none}
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS:-0}
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS:-0}
CONSISTENCY_LR_FINAL_SCALE=${CONSISTENCY_LR_FINAL_SCALE:-1.0}
BC_LOSS_COEF=${BC_LOSS_COEF:-1.0}
BASE_ACTION_ANCHOR_COEF=${BASE_ACTION_ANCHOR_COEF:-0.0}
CONSISTENCY_ACTION_L2_COEF=${CONSISTENCY_ACTION_L2_COEF:-0.0}

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
    echo "[gpu_guard] over budget; not starting local consistency probe" >&2
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

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Missing CoDrive teacher checkpoint: ${TEACHER_CKPT}" >&2
  exit 2
fi

guard_gpu

mkdir -p "${LOG_DIR}"

cp configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml "${PROBE_DIR}/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml.used"
cp configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml "${PROBE_DIR}/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml.used"
cp "${TEACHER_CKPT}" "${PROBE_DIR}/best_reward_4159.37.pth.used"

{
  echo "run_tag=${RUN_TAG}"
  echo "date=$(date -Iseconds)"
  echo "root=${ROOT}"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "teacher_ckpt=${TEACHER_CKPT}"
  echo "train_window_sec=${TRAIN_WINDOW_SEC}"
  echo "num_envs=${NUM_ENVS}"
  echo "minibatch=${MINIBATCH}"
  echo "student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS}"
  echo "student_progress_log_interval=${STUDENT_PROGRESS_LOG_INTERVAL}"
  echo "consistency_infer_steps=${CONSISTENCY_INFER_STEPS}"
  echo "consistency_num_scales=${CONSISTENCY_NUM_SCALES}"
  echo "consistency_lr=${CONSISTENCY_LR}"
  echo "consistency_loss_coef=${CONSISTENCY_LOSS_COEF}"
  echo "consistency_boundary_coef=${CONSISTENCY_BOUNDARY_COEF}"
  echo "consistency_use_ema_target=${CONSISTENCY_USE_EMA_TARGET}"
  echo "consistency_ema_decay=${CONSISTENCY_EMA_DECAY}"
  echo "consistency_infer_use_ema=${CONSISTENCY_INFER_USE_EMA}"
  echo "consistency_train_align_infer=${CONSISTENCY_TRAIN_ALIGN_INFER}"
  echo "consistency_lr_schedule=${CONSISTENCY_LR_SCHEDULE}"
  echo "consistency_lr_decay_start_agent_steps=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS}"
  echo "consistency_lr_decay_end_agent_steps=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS}"
  echo "consistency_lr_final_scale=${CONSISTENCY_LR_FINAL_SCALE}"
  echo "bc_loss_coef=${BC_LOSS_COEF}"
  echo "base_action_anchor_coef=${BASE_ACTION_ANCHOR_COEF}"
  echo "consistency_action_l2_coef=${CONSISTENCY_ACTION_L2_COEF}"
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
  STUDENT_PROGRESS_LOG_INTERVAL="${STUDENT_PROGRESS_LOG_INTERVAL}"
  CONSISTENCY_INFER_STEPS="${CONSISTENCY_INFER_STEPS}"
  CONSISTENCY_NUM_SCALES="${CONSISTENCY_NUM_SCALES}"
  CONSISTENCY_LR="${CONSISTENCY_LR}"
  CONSISTENCY_LOSS_COEF="${CONSISTENCY_LOSS_COEF}"
  CONSISTENCY_BOUNDARY_COEF="${CONSISTENCY_BOUNDARY_COEF}"
  CONSISTENCY_USE_EMA_TARGET="${CONSISTENCY_USE_EMA_TARGET}"
  CONSISTENCY_EMA_DECAY="${CONSISTENCY_EMA_DECAY}"
  CONSISTENCY_INFER_USE_EMA="${CONSISTENCY_INFER_USE_EMA}"
  CONSISTENCY_TRAIN_ALIGN_INFER="${CONSISTENCY_TRAIN_ALIGN_INFER}"
  CONSISTENCY_LR_SCHEDULE="${CONSISTENCY_LR_SCHEDULE}"
  CONSISTENCY_LR_DECAY_START_AGENT_STEPS="${CONSISTENCY_LR_DECAY_START_AGENT_STEPS}"
  CONSISTENCY_LR_DECAY_END_AGENT_STEPS="${CONSISTENCY_LR_DECAY_END_AGENT_STEPS}"
  CONSISTENCY_LR_FINAL_SCALE="${CONSISTENCY_LR_FINAL_SCALE}"
  BC_LOSS_COEF="${BC_LOSS_COEF}"
  BASE_ACTION_ANCHOR_COEF="${BASE_ACTION_ANCHOR_COEF}"
  CONSISTENCY_ACTION_L2_COEF="${CONSISTENCY_ACTION_L2_COEF}"
  bash scripts/dexh13_lightbulb_student_consistency_codrive.sh
  "${GPU}" "${SEED}" "${RUN_TAG}" "${TEACHER_CKPT}"
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

CKPT_BEST="outputs/Dexh13HoraLightbulb_student_consistency_codrive/${RUN_TAG}/stage2_consistency_nn/model_best.ckpt"
CKPT_LAST="outputs/Dexh13HoraLightbulb_student_consistency_codrive/${RUN_TAG}/stage2_consistency_nn/model_last.ckpt"
if [[ -f "${CKPT_BEST}" ]]; then
  CKPT="${CKPT_BEST}"
elif [[ -f "${CKPT_LAST}" ]]; then
  CKPT="${CKPT_LAST}"
  echo "[checkpoint] model_best missing; using model_last" | tee -a "${LOG_DIR}/train.log"
else
  echo "Training did not produce a consistency checkpoint under ${RUN_TAG}" >&2
  exit 3
fi
echo "${CKPT}" > "${PROBE_DIR}/checkpoint.txt"

printf 'seed\tstatus\tsteps\tavg_reward\tavg_done_rate\tlatent_mse\tlatent_l1\taction_mse_to_teacher\tlog\n' > "${PROBE_DIR}/eval_summary.tsv"

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
    >> "${PROBE_DIR}/eval_summary.tsv"
done

echo "probe_dir=${PROBE_DIR}"
echo "checkpoint=${CKPT}"
echo "eval_summary=${PROBE_DIR}/eval_summary.tsv"
