#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

GPU=${GPU:-0}
SUITE_TAG=${SUITE_TAG:-g4c_existing_checkpoint_eval_$(date +%Y%m%d_%H%M%S)}
PROBE_ROOT=${PROBE_ROOT:-outputs/local_probe_consistency_g4c_external}
EVAL_ROOT=${EVAL_ROOT:-outputs/local_eval_consistency_g4c_external}
SUITE_DIR="${PROBE_ROOT}/${SUITE_TAG}"

RUNS=${RUNS:-"42:g4c_consistency_selector_20260602_032255_train_s42 43:g4c_consistency_trainonly_20260602_040200_s43 44:g4c_consistency_trainonly_20260602_043500_s44"}
CHECKPOINT_KINDS=${CHECKPOINT_KINDS:-"model_best model_best_train model_last"}

EVAL_STEPS=${EVAL_STEPS:-512}
EVAL_NUM_ENVS=${EVAL_NUM_ENVS:-16}
EVAL_SEEDS=${EVAL_SEEDS:-"42 43 44"}
EVAL_TIMEOUT_SEC=${EVAL_TIMEOUT_SEC:-900}
GPU_UTIL_MAX=${GPU_UTIL_MAX:-95}
GPU_MEM_PCT_MAX=${GPU_MEM_PCT_MAX:-95}

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
  echo "runs=${RUNS}"
  echo "checkpoint_kinds=${CHECKPOINT_KINDS}"
  echo "eval_steps=${EVAL_STEPS}"
  echo "eval_num_envs=${EVAL_NUM_ENVS}"
  echo "eval_seeds=${EVAL_SEEDS}"
  echo "eval_timeout_sec=${EVAL_TIMEOUT_SEC}"
  echo "gpu=${GPU}"
  echo "gpu_util_max=${GPU_UTIL_MAX}"
  echo "gpu_mem_pct_max=${GPU_MEM_PCT_MAX}"
} > "${SUITE_DIR}/suite_manifest.env"

printf 'train_seed\trun_tag\tcheckpoint_kind\tcheckpoint\tsha256\talias_sha256\teval_summary\teval_status\n' \
  > "${SUITE_DIR}/checkpoint_eval_index.tsv"
printf 'train_seed\trun_tag\tcheckpoint_kind\teval_seed\tstatus\tsteps\tavg_reward\tavg_done_rate\tlatent_mse\tlatent_l1\taction_mse_to_teacher\tlog\n' \
  > "${SUITE_DIR}/per_eval_metrics.tsv"

checkpoint_path() {
  local run_tag="$1"
  local kind="$2"
  printf 'outputs/Dexh13HoraLightbulb_student_consistency_codrive/%s/stage2_consistency_nn/%s.ckpt' \
    "${run_tag}" "${kind}"
}

alias_hashes() {
  local run_tag="$1"
  local ckpt_dir="outputs/Dexh13HoraLightbulb_student_consistency_codrive/${run_tag}/stage2_consistency_nn"
  local aliases=(
    "${ckpt_dir}/model_best.ckpt"
    "${ckpt_dir}/model_best_eval.ckpt"
    "${ckpt_dir}/model_best_deploy.ckpt"
  )
  local path hashes=()
  for path in "${aliases[@]}"; do
    if [[ -f "${path}" ]]; then
      hashes+=("$(sha256sum "${path}" | awk '{print $1}')")
    else
      hashes+=("MISSING")
    fi
  done
  local out="${SUITE_DIR}/${run_tag}_selected_alias_sha256.txt"
  {
    printf 'model_best\t%s\n' "${hashes[0]}"
    printf 'model_best_eval\t%s\n' "${hashes[1]}"
    printf 'model_best_deploy\t%s\n' "${hashes[2]}"
  } > "${out}"
  printf '%s/%s/%s' "${hashes[0]}" "${hashes[1]}" "${hashes[2]}"
}

append_metrics() {
  local train_seed="$1"
  local run_tag="$2"
  local kind="$3"
  local summary="$4"

  if [[ ! -f "${summary}" ]]; then
    return 0
  fi

  awk -F'\t' -v train_seed="${train_seed}" -v run_tag="${run_tag}" -v kind="${kind}" '
    NR == 1 { next }
    {
      printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
        train_seed, run_tag, kind, $1, $2, $3, $4, $5, $6, $7, $8, $9
    }
  ' "${summary}" >> "${SUITE_DIR}/per_eval_metrics.tsv"
}

run_one_eval() {
  local train_seed="$1"
  local run_tag="$2"
  local kind="$3"
  local ckpt="$4"

  if [[ ! -f "${ckpt}" ]]; then
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${train_seed}" "${run_tag}" "${kind}" "${ckpt}" "MISSING" "" "" "missing" \
      >> "${SUITE_DIR}/checkpoint_eval_index.tsv"
    return 0
  fi

  local sha alias_sha eval_tag summary status
  sha="$(sha256sum "${ckpt}" | awk '{print $1}')"
  alias_sha=""
  if [[ "${kind}" == "model_best" ]]; then
    alias_sha="$(alias_hashes "${run_tag}")"
  fi
  eval_tag="${run_tag}_${kind}_eval${EVAL_STEPS}"
  summary="${EVAL_ROOT}/${SUITE_TAG}/${eval_tag}/eval_summary.tsv"

  set +e
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
  bash scripts/eval_consistency_codrive_checkpoint.sh
  status=$?
  set -e

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${train_seed}" "${run_tag}" "${kind}" "${ckpt}" "${sha}" "${alias_sha}" "${summary}" "${status}" \
    >> "${SUITE_DIR}/checkpoint_eval_index.tsv"
  append_metrics "${train_seed}" "${run_tag}" "${kind}" "${summary}"
}

read -r -a run_array <<< "${RUNS}"
read -r -a kind_array <<< "${CHECKPOINT_KINDS}"
for item in "${run_array[@]}"; do
  train_seed="${item%%:*}"
  run_tag="${item#*:}"
  for kind in "${kind_array[@]}"; do
    run_one_eval "${train_seed}" "${run_tag}" "${kind}" "$(checkpoint_path "${run_tag}" "${kind}")"
  done
done

awk -F'\t' '
  NR == 1 { next }
  $7 == "" { next }
  {
    key=$1 "\t" $2 "\t" $3
    n[key]++
    s[key]+=$7
    ss[key]+=$7*$7
    d[key]+=$8
    lm[key]+=$9
    l1[key]+=$10
    am[key]+=$11
  }
  END {
    print "train_seed\trun_tag\tcheckpoint_kind\tn\tmean_reward\tstd_reward\tdone_mean\tlatent_mse\tlatent_l1\taction_mse_to_teacher"
    for (key in n) {
      mean=s[key]/n[key]
      var=ss[key]/n[key]-mean*mean
      if (var < 0) var=0
      printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
        key, n[key], mean, sqrt(var), d[key]/n[key], lm[key]/n[key], l1[key]/n[key], am[key]/n[key]
    }
  }
' "${SUITE_DIR}/per_eval_metrics.tsv" > "${SUITE_DIR}/checkpoint_aggregate.tsv"

awk -F'\t' '
  NR == 1 { next }
  $7 == "" { next }
  {
    key=$3
    n[key]++
    s[key]+=$7
    ss[key]+=$7*$7
    d[key]+=$8
    lm[key]+=$9
    l1[key]+=$10
    am[key]+=$11
  }
  END {
    print "checkpoint_kind\tn\tmean_reward\tstd_reward\tdone_mean\tlatent_mse\tlatent_l1\taction_mse_to_teacher"
    for (key in n) {
      mean=s[key]/n[key]
      var=ss[key]/n[key]-mean*mean
      if (var < 0) var=0
      printf "%s\t%d\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
        key, n[key], mean, sqrt(var), d[key]/n[key], lm[key]/n[key], l1[key]/n[key], am[key]/n[key]
    }
  }
' "${SUITE_DIR}/per_eval_metrics.tsv" > "${SUITE_DIR}/kind_aggregate.tsv"

{
  echo "dirty_status_end<<EOF"
  git status --short --untracked-files=all 2>/dev/null || true
  echo "EOF"
} >> "${SUITE_DIR}/suite_manifest.env"

echo "suite_dir=${SUITE_DIR}"
echo "checkpoint_eval_index=${SUITE_DIR}/checkpoint_eval_index.tsv"
echo "per_eval_metrics=${SUITE_DIR}/per_eval_metrics.tsv"
echo "checkpoint_aggregate=${SUITE_DIR}/checkpoint_aggregate.tsv"
echo "kind_aggregate=${SUITE_DIR}/kind_aggregate.tsv"
