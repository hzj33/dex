#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
TRAIN_SEEDS=${TRAIN_SEEDS:-"42 43 44"}
SUITE_TAG=${SUITE_TAG:-g4c_consistency_selector_$(date +%Y%m%d_%H%M%S)}
PROBE_ROOT=${PROBE_ROOT:-outputs/local_probe_consistency_g4c}
EVAL_ROOT=${EVAL_ROOT:-outputs/local_eval_consistency_g4c}
SUITE_DIR="${PROBE_ROOT}/${SUITE_TAG}"

TRAIN_WINDOW_SEC=${TRAIN_WINDOW_SEC:-4200}
STUDENT_MAX_AGENT_STEPS=${STUDENT_MAX_AGENT_STEPS:-450000}
STUDENT_PROGRESS_LOG_INTERVAL=${STUDENT_PROGRESS_LOG_INTERVAL:-100000}
NUM_ENVS=${NUM_ENVS:-16}
MINIBATCH=${MINIBATCH:-192}

EVAL_STEPS=${EVAL_STEPS:-512}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_SEEDS=${EVAL_SEEDS:-"42 43 44"}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-900}

GPU_UTIL_MAX=${GPU_UTIL_MAX:-80}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-80}
TEACHER_CKPT=${TEACHER_CKPT:-sim2real/codrive/best_reward_4159.37.pth}
CONSISTENCY_TRAIN_ALIGN_INFER=${CONSISTENCY_TRAIN_ALIGN_INFER:-False}
CONSISTENCY_LR_SCHEDULE=${CONSISTENCY_LR_SCHEDULE:-none}
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS:-0}
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS:-0}
CONSISTENCY_LR_FINAL_SCALE=${CONSISTENCY_LR_FINAL_SCALE:-1.0}

mkdir -p "${SUITE_DIR}"

{
  echo "suite_tag=${SUITE_TAG}"
  echo "date=$(date -Iseconds)"
  echo "root=${ROOT}"
  echo "branch=$(git branch --show-current 2>/dev/null || true)"
  echo "commit=$(git rev-parse --short HEAD 2>/dev/null || true)"
  echo "dirty_status_begin<<EOF"
  git status --short --untracked-files=all 2>/dev/null || true
  echo "EOF"
  echo "teacher_ckpt=${TEACHER_CKPT}"
  echo "train_seeds=${TRAIN_SEEDS}"
  echo "train_window_sec=${TRAIN_WINDOW_SEC}"
  echo "student_max_agent_steps=${STUDENT_MAX_AGENT_STEPS}"
  echo "student_progress_log_interval=${STUDENT_PROGRESS_LOG_INTERVAL}"
  echo "num_envs=${NUM_ENVS}"
  echo "minibatch=${MINIBATCH}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
  echo "eval_seeds=${EVAL_SEEDS}"
  echo "gpu=${GPU}"
  echo "gpu_util_max=${GPU_UTIL_MAX}"
  echo "gpu_mem_pct_max=${GPU_MEM_PCT_MAX}"
  echo "consistency_train_align_infer=${CONSISTENCY_TRAIN_ALIGN_INFER}"
  echo "consistency_lr_schedule=${CONSISTENCY_LR_SCHEDULE}"
  echo "consistency_lr_decay_start_agent_steps=${CONSISTENCY_LR_DECAY_START_AGENT_STEPS}"
  echo "consistency_lr_decay_end_agent_steps=${CONSISTENCY_LR_DECAY_END_AGENT_STEPS}"
  echo "consistency_lr_final_scale=${CONSISTENCY_LR_FINAL_SCALE}"
} > "${SUITE_DIR}/suite_manifest.env"

printf 'train_seed\trun_tag\tcheckpoint_kind\tcheckpoint\tsha256\teval_summary\n' \
  > "${SUITE_DIR}/checkpoint_eval_index.tsv"

run_eval_checkpoint() {
  local train_seed="$1"
  local run_tag="$2"
  local kind="$3"
  local ckpt="$4"

  if [[ ! -f "${ckpt}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${train_seed}" "${run_tag}" "${kind}" "${ckpt}" "MISSING" "" \
      >> "${SUITE_DIR}/checkpoint_eval_index.tsv"
    return 0
  fi

  local sha
  sha="$(sha256sum "${ckpt}" | awk '{print $1}')"
  local eval_tag="${run_tag}_${kind}_eval${EVAL_STEPS}"

  GPU="${GPU}" \
  CKPT="${ckpt}" \
  RUN_TAG="${eval_tag}" \
  EVAL_ROOT="${EVAL_ROOT}/${SUITE_TAG}" \
  EVAL_STEPS="${EVAL_STEPS}" \
  EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" \
  EVAL_SEEDS="${EVAL_SEEDS}" \
    EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
    GPU_UTIL_MAX="${GPU_UTIL_MAX}" \
    GPU_MEM_PCT_MAX="${GPU_MEM_PCT_MAX}" \
    CONSISTENCY_TRAIN_ALIGN_INFER="${CONSISTENCY_TRAIN_ALIGN_INFER}" \
    bash scripts/eval_consistency_codrive_checkpoint.sh

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${train_seed}" "${run_tag}" "${kind}" "${ckpt}" "${sha}" \
    "${EVAL_ROOT}/${SUITE_TAG}/${eval_tag}/eval_summary.tsv" \
    >> "${SUITE_DIR}/checkpoint_eval_index.tsv"
}

read -r -a train_seed_array <<< "${TRAIN_SEEDS}"
for train_seed in "${train_seed_array[@]}"; do
  run_tag="${SUITE_TAG}_train_s${train_seed}"

  GPU="${GPU}" \
  SEED="${train_seed}" \
  RUN_TAG="${run_tag}" \
  PROBE_ROOT="${SUITE_DIR}/selected" \
  TEACHER_CKPT="${TEACHER_CKPT}" \
  TRAIN_WINDOW_SEC="${TRAIN_WINDOW_SEC}" \
  STUDENT_MAX_AGENT_STEPS="${STUDENT_MAX_AGENT_STEPS}" \
  STUDENT_PROGRESS_LOG_INTERVAL="${STUDENT_PROGRESS_LOG_INTERVAL}" \
  NUM_ENVS="${NUM_ENVS}" \
  MINIBATCH="${MINIBATCH}" \
  EVAL_STEPS="${EVAL_STEPS}" \
  EVAL_NUM_ENVS="${EVAL_NUM_ENVS}" \
  EVAL_SEEDS="${EVAL_SEEDS}" \
  EVAL_TIMEOUT_SEC="${EVAL_TIMEOUT_SEC}" \
  GPU_UTIL_MAX="${GPU_UTIL_MAX}" \
  GPU_MEM_PCT_MAX="${GPU_MEM_PCT_MAX}" \
  CONSISTENCY_TRAIN_ALIGN_INFER="${CONSISTENCY_TRAIN_ALIGN_INFER}" \
  CONSISTENCY_LR_SCHEDULE="${CONSISTENCY_LR_SCHEDULE}" \
  CONSISTENCY_LR_DECAY_START_AGENT_STEPS="${CONSISTENCY_LR_DECAY_START_AGENT_STEPS}" \
  CONSISTENCY_LR_DECAY_END_AGENT_STEPS="${CONSISTENCY_LR_DECAY_END_AGENT_STEPS}" \
  CONSISTENCY_LR_FINAL_SCALE="${CONSISTENCY_LR_FINAL_SCALE}" \
  bash scripts/run_consistency_g2_validate_codrive.sh \
    ++train.ppo.eval_select.enabled=True \
    ++train.ppo.eval_select.interval_agent_steps=100000 \
    ++train.ppo.eval_select.min_agent_steps=100000 \
    ++train.ppo.eval_select.num_steps=512 \
    ++train.ppo.eval_select.done_penalty=0.0 \
    ++train.ppo.eval_select.min_score_improvement=0.0 \
    ++train.ppo.eval_select.final_eval=True \
    ++train.ppo.eval_select.save_deploy_best=True

  ckpt_dir="outputs/Dexh13HoraLightbulb_student_consistency_codrive/${run_tag}/stage2_consistency_nn"
  selected_summary="${SUITE_DIR}/selected/${run_tag}/eval_summary.tsv"

  if [[ -f "${ckpt_dir}/model_best.ckpt" ]]; then
    selected_sha="$(sha256sum "${ckpt_dir}/model_best.ckpt" | awk '{print $1}')"
  else
    selected_sha="MISSING"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${train_seed}" "${run_tag}" "model_best_alias" \
    "${ckpt_dir}/model_best.ckpt" "${selected_sha}" "${selected_summary}" \
    >> "${SUITE_DIR}/checkpoint_eval_index.tsv"

  run_eval_checkpoint "${train_seed}" "${run_tag}" "model_best_train" \
    "${ckpt_dir}/model_best_train.ckpt"
  run_eval_checkpoint "${train_seed}" "${run_tag}" "model_last" \
    "${ckpt_dir}/model_last.ckpt"

  if [[ -f "${ckpt_dir}/model_best_eval.ckpt" && -f "${ckpt_dir}/model_best.ckpt" ]]; then
    sha256sum "${ckpt_dir}/model_best.ckpt" \
      "${ckpt_dir}/model_best_eval.ckpt" \
      "${ckpt_dir}/model_best_deploy.ckpt" \
      "${ckpt_dir}/model_best_train.ckpt" \
      "${ckpt_dir}/model_last.ckpt" \
      > "${SUITE_DIR}/${run_tag}_checkpoint_sha256.txt"
  fi
done

awk -F'\t' '
  FNR==1 && NR==1 { next }
  FILENAME==ARGV[1] { next }
  FNR==1 { next }
  {
    n[FILENAME]++
    s[FILENAME]+=$4
    ss[FILENAME]+=$4*$4
    d[FILENAME]+=$5
    lm[FILENAME]+=$6
    l1[FILENAME]+=$7
    am[FILENAME]+=$8
  }
  END {
    print "eval_summary\tn\tmean_reward\tstd_reward\tdone_mean\tlatent_mse\tlatent_l1\taction_mse_to_teacher"
    for (f in n) {
      mean=s[f]/n[f]
      var=ss[f]/n[f]-mean*mean
      if (var < 0) var=0
      printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n", f, n[f], mean, sqrt(var), d[f]/n[f], lm[f]/n[f], l1[f]/n[f], am[f]/n[f]
    }
  }
' "${SUITE_DIR}/checkpoint_eval_index.tsv" \
  $(awk -F'\t' 'NR>1 && $6 != "" {print $6}' "${SUITE_DIR}/checkpoint_eval_index.tsv") \
  > "${SUITE_DIR}/eval_aggregate.tsv"

{
  echo "dirty_status_end<<EOF"
  git status --short --untracked-files=all 2>/dev/null || true
  echo "EOF"
} >> "${SUITE_DIR}/suite_manifest.env"

echo "suite_dir=${SUITE_DIR}"
echo "checkpoint_eval_index=${SUITE_DIR}/checkpoint_eval_index.tsv"
echo "eval_aggregate=${SUITE_DIR}/eval_aggregate.tsv"
