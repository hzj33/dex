#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   scripts/eval_screwdriver_student_robustness_multiseed.sh \
#     GPU_ID ALGO CHECKPOINT [STEPS] [EVAL_CACHE_PREFIX] [EVAL_SEEDS_CSV] [extra overrides...]
#
# Example:
#   scripts/eval_screwdriver_student_robustness_multiseed.sh \
#     0 DiffusionActionChunkStudent <ckpt> 256 actionchunk_ms "42,43,44"

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 GPU_ID ALGO CHECKPOINT [STEPS] [EVAL_CACHE_PREFIX] [EVAL_SEEDS_CSV] [extra overrides...]"
  exit 1
fi

GPU_ID=${1}
ALGO=${2}
CKPT=${3}
STEPS=${4:-256}
EVAL_CACHE_PREFIX=${5:-eval_multiseed}
EVAL_SEEDS_CSV=${6:-42,43,44}

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:6:$len}")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh"

if [[ ! -x "${BASE_SCRIPT}" ]]; then
  echo "Base eval script not found or not executable: ${BASE_SCRIPT}"
  exit 1
fi

IFS=',' read -r -a RAW_SEEDS <<< "${EVAL_SEEDS_CSV}"
SEEDS=()
for s in "${RAW_SEEDS[@]}"; do
  trimmed="$(echo "${s}" | xargs)"
  if [[ -n "${trimmed}" ]]; then
    SEEDS+=("${trimmed}")
  fi
done

if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "No valid eval seeds parsed from: ${EVAL_SEEDS_CSV}"
  exit 1
fi

REWARDS=()
DONE_RATES=()
USED_SEEDS=()

for seed in "${SEEDS[@]}"; do
  cache_name="${EVAL_CACHE_PREFIX}_s${seed}"
  tmp_log="$(mktemp)"
  echo "[multiseed] eval seed=${seed} cache=${cache_name}"

  set +e
  "${BASE_SCRIPT}" \
    "${GPU_ID}" "${seed}" "${ALGO}" "${CKPT}" "${STEPS}" "${cache_name}" \
    "${EXTRA_ARGS[@]}" 2>&1 | tee "${tmp_log}"
  rc=$?
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[multiseed] failed at seed=${seed} (exit=${rc})"
    rm -f "${tmp_log}"
    exit "${rc}"
  fi

  if command -v rg >/dev/null 2>&1; then
    summary_line="$(rg "EvalSummary steps=" "${tmp_log}" | tail -n 1 || true)"
  else
    summary_line="$(grep "EvalSummary steps=" "${tmp_log}" | tail -n 1 || true)"
  fi
  if [[ -z "${summary_line}" ]]; then
    echo "[multiseed] EvalSummary not found for seed=${seed}"
    rm -f "${tmp_log}"
    exit 1
  fi

  reward="$(echo "${summary_line}" | sed -E 's/.*avg_reward=([-0-9.eE+]+).*/\1/')"
  done_rate="$(echo "${summary_line}" | sed -E 's/.*avg_done_rate=([-0-9.eE+]+).*/\1/')"

  echo "[multiseed] parsed seed=${seed} avg_reward=${reward} avg_done_rate=${done_rate}"
  USED_SEEDS+=("${seed}")
  REWARDS+=("${reward}")
  DONE_RATES+=("${done_rate}")
  rm -f "${tmp_log}"
done

seeds_csv_out="$(IFS=,; echo "${USED_SEEDS[*]}")"
rewards_csv_out="$(IFS=,; echo "${REWARDS[*]}")"
dones_csv_out="$(IFS=,; echo "${DONE_RATES[*]}")"

python - "${seeds_csv_out}" "${rewards_csv_out}" "${dones_csv_out}" <<'PY'
import sys
import numpy as np

seeds = [x for x in sys.argv[1].split(",") if x]
rewards = np.array([float(x) for x in sys.argv[2].split(",") if x], dtype=np.float64)
dones = np.array([float(x) for x in sys.argv[3].split(",") if x], dtype=np.float64)

print("[multiseed] summary")
for s, r, d in zip(seeds, rewards, dones):
    print(f"  seed={s} avg_reward={r:.6f} avg_done_rate={d:.6f}")
print(
    "[multiseed] aggregate "
    f"reward_mean={rewards.mean():.6f} reward_std={rewards.std(ddof=0):.6f} "
    f"done_mean={dones.mean():.6f} done_std={dones.std(ddof=0):.6f}"
)
PY
