#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
BATCH_TAG=${BATCH_TAG:-g2_consistency_formal_$(date +%Y%m%d_%H%M%S)}
FORMAL_ROOT=${FORMAL_ROOT:-outputs/local_formal_consistency_g2}
FORMAL_DIR="${FORMAL_ROOT}/${BATCH_TAG}"
LOG_DIR="${FORMAL_DIR}/logs"
PER_SEED_ROOT="${FORMAL_DIR}/per_seed"

TRAIN_SEEDS=${TRAIN_SEEDS:-"42 43 44"}
EVAL_SEEDS=${EVAL_SEEDS:-"42 43 44"}
TEACHER_CKPT=${TEACHER_CKPT:-sim2real/codrive/best_reward_4159.37.pth}

TRAIN_WINDOW_SEC=${TRAIN_WINDOW_SEC:-4200}
NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-900000}
STUDENT_PROGRESS_LOG_INTERVAL=${STUDENT_PROGRESS_LOG_INTERVAL:-100000}

EVAL_STEPS=${EVAL_STEPS:-2048}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-2400}

CONSISTENCY_INFER_STEPS=${CONSISTENCY_INFER_STEPS:-1}
CONSISTENCY_NUM_SCALES=${CONSISTENCY_NUM_SCALES:-10}
BASE_ACTION_ANCHOR_COEF=${BASE_ACTION_ANCHOR_COEF:-0.0}

GPU_UTIL_MAX=${GPU_UTIL_MAX:-80}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-80}
GPU_GUARD_SKIP=${GPU_GUARD_SKIP:-False}

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "Missing CoDrive teacher checkpoint: ${TEACHER_CKPT}" >&2
  exit 2
fi

mkdir -p "${LOG_DIR}" "${PER_SEED_ROOT}"

git status --short --branch --untracked-files=all > "${FORMAL_DIR}/git_status_short.txt" 2>/dev/null || true
git diff --stat > "${FORMAL_DIR}/git_diff_stat.txt" 2>/dev/null || true
git diff -- dexscrew/algo/ppo/consistency_latent_student.py train.py dexscrew/algo/student/__init__.py \
  > "${FORMAL_DIR}/git_diff_tracked.patch" 2>/dev/null || true
mkdir -p "${FORMAL_DIR}/scripts_used"
cp scripts/dexh13_lightbulb_student_consistency_codrive.sh \
  scripts/run_consistency_g2_validate_codrive.sh \
  scripts/eval_consistency_codrive_checkpoint.sh \
  scripts/run_consistency_g2_formal_codrive.sh \
  "${FORMAL_DIR}/scripts_used/"

{
  echo "batch_tag=${BATCH_TAG}"
  echo "date=$(date -Iseconds)"
  echo "root=${ROOT}"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "teacher_ckpt=${TEACHER_CKPT}"
  echo "train_seeds=${TRAIN_SEEDS}"
  echo "eval_seeds=${EVAL_SEEDS}"
  echo "train_window_sec=${TRAIN_WINDOW_SEC}"
  echo "num_envs=${NUM_ENVS}"
  echo "minibatch=${MINIBATCH}"
  echo "student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS}"
  echo "student_progress_log_interval=${STUDENT_PROGRESS_LOG_INTERVAL}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
  echo "eval_timeout_sec=${EVAL_TIMEOUT_SEC}"
  echo "consistency_infer_steps=${CONSISTENCY_INFER_STEPS}"
  echo "consistency_num_scales=${CONSISTENCY_NUM_SCALES}"
  echo "base_action_anchor_coef=${BASE_ACTION_ANCHOR_COEF}"
  echo "gpu_util_max=${GPU_UTIL_MAX}"
  echo "gpu_mem_pct_max=${GPU_MEM_PCT_MAX}"
} > "${FORMAL_DIR}/manifest.env"

printf 'train_seed\trun_tag\tstatus\tprobe_dir\tcheckpoint\teval_summary\n' > "${FORMAL_DIR}/train_runs.tsv"
printf 'train_seed\trun_tag\teval_seed\tstatus\tsteps\tavg_reward\tavg_done_rate\tlatent_mse\tlatent_l1\taction_mse_to_teacher\tcheckpoint\tlog\n' > "${FORMAL_DIR}/formal_eval_summary.tsv"

overall_status=0
read -r -a train_seed_array <<< "${TRAIN_SEEDS}"
for train_seed in "${train_seed_array[@]}"; do
  run_tag="${BATCH_TAG}_train_s${train_seed}"
  probe_dir="${PER_SEED_ROOT}/${run_tag}"
  wrapper_log="${LOG_DIR}/train_seed${train_seed}.wrapper.log"

  echo "[formal] train_seed=${train_seed} run_tag=${run_tag} start $(date -Iseconds)" | tee "${wrapper_log}"

  set +e
  GPU="${GPU}" \
  SEED="${train_seed}" \
  RUN_TAG="${run_tag}" \
  PROBE_ROOT="${PER_SEED_ROOT}" \
  TEACHER_CKPT="${TEACHER_CKPT}" \
  TRAIN_WINDOW_SEC="${TRAIN_WINDOW_SEC}" \
  NUM_ENVS="${NUM_ENVS}" \
  MINIBATCH="${MINIBATCH}" \
  STUDENT_MAX_AGENT_STEPS="${STUDENT_MAX_AGENT_STEPS}" \
  STUDENT_PROGRESS_LOG_INTERVAL="${STUDENT_PROGRESS_LOG_INTERVAL}" \
  EVAL_STEPS="${EVAL_STEPS}" \
  EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" \
  EVAL_SEEDS="${EVAL_SEEDS}" \
  EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
  CONSISTENCY_INFER_STEPS="${CONSISTENCY_INFER_STEPS}" \
  CONSISTENCY_NUM_SCALES="${CONSISTENCY_NUM_SCALES}" \
  BASE_ACTION_ANCHOR_COEF="${BASE_ACTION_ANCHOR_COEF}" \
  GPU_UTIL_MAX="${GPU_UTIL_MAX}" \
  GPU_MEM_PCT_MAX="${GPU_MEM_PCT_MAX}" \
  GPU_GUARD_SKIP="${GPU_GUARD_SKIP}" \
  bash scripts/run_consistency_g2_validate_codrive.sh 2>&1 | tee -a "${wrapper_log}"
  status=${PIPESTATUS[0]}
  set -e

  checkpoint=""
  if [[ -f "${probe_dir}/checkpoint.txt" ]]; then
    checkpoint="$(<"${probe_dir}/checkpoint.txt")"
  fi
  eval_summary="${probe_dir}/eval_summary.tsv"
  if [[ ! -f "${eval_summary}" ]]; then
    eval_summary=""
  fi

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${train_seed}" "${run_tag}" "${status}" "${probe_dir}" "${checkpoint}" "${eval_summary}" \
    >> "${FORMAL_DIR}/train_runs.tsv"

  if [[ -n "${eval_summary}" ]]; then
    awk -v train_seed="${train_seed}" -v run_tag="${run_tag}" -v checkpoint="${checkpoint}" \
      'BEGIN { FS=OFS="\t" } NR > 1 { print train_seed, run_tag, $1, $2, $3, $4, $5, $6, $7, $8, checkpoint, $9 }' \
      "${eval_summary}" >> "${FORMAL_DIR}/formal_eval_summary.tsv"
  fi

  echo "[formal] train_seed=${train_seed} status=${status} end $(date -Iseconds)" | tee -a "${wrapper_log}"
  if [[ "${status}" -ne 0 && "${overall_status}" -eq 0 ]]; then
    overall_status="${status}"
  fi
  if [[ "${status}" -eq 75 ]]; then
    echo "[formal] stopping because GPU guard reported over-budget" | tee -a "${wrapper_log}"
    break
  fi
done

awk 'BEGIN { FS=OFS="\t" }
  NR > 1 && $4 == 0 && $6 != "" {
    n += 1
    reward[n] = $6
    done += $7
    latent_mse += $8
    latent_l1 += $9
    action_mse += $10
  }
  END {
    if (n == 0) {
      print "valid_eval_rows=0"
      exit
    }
    reward_mean = 0
    for (i = 1; i <= n; i++) {
      reward_mean += reward[i]
    }
    reward_mean /= n
    reward_var = 0
    for (i = 1; i <= n; i++) {
      diff = reward[i] - reward_mean
      reward_var += diff * diff
    }
    reward_std = sqrt(reward_var / n)
    printf "valid_eval_rows=%d\n", n
    printf "reward_mean=%.6f\n", reward_mean
    printf "reward_std=%.6f\n", reward_std
    printf "done_rate_mean=%.6f\n", done / n
    printf "latent_mse_mean=%.6f\n", latent_mse / n
    printf "latent_l1_mean=%.6f\n", latent_l1 / n
    printf "action_mse_to_teacher_mean=%.6f\n", action_mse / n
  }' "${FORMAL_DIR}/formal_eval_summary.tsv" > "${FORMAL_DIR}/aggregate_stats.txt"

echo "formal_dir=${FORMAL_DIR}"
echo "train_runs=${FORMAL_DIR}/train_runs.tsv"
echo "formal_eval_summary=${FORMAL_DIR}/formal_eval_summary.tsv"
echo "aggregate_stats=${FORMAL_DIR}/aggregate_stats.txt"
exit "${overall_status}"
