#!/usr/bin/env bash
set -euo pipefail

# Canonical 15-min acceptance run for robustness-oriented latent diffusion student.
# Usage:
#   scripts/screwdriver_student_diffusion_latent_robust_light_15min_docker.sh \
#     GPU_ID SEED TEACHER_CACHE [WINDOW_SEC] [DIFF_CACHE]

GPU_ID=${1:-0}
SEED=${2:-42}
TEACHER_CACHE=${3:?Missing TEACHER_CACHE, e.g. run_a}
WINDOW_SEC=${4:-900}
DIFF_CACHE=${5:-${TEACHER_CACHE}_latent_robust_light_seed${SEED}_${WINDOW_SEC}s}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

LOG_PATH="outputs/XHandHoraScrewDriver_student_diffusion_latent/${DIFF_CACHE}/train_${WINDOW_SEC}s.log"

mkdir -p "outputs/XHandHoraScrewDriver_student_diffusion_latent/${DIFF_CACHE}"

./docker-run-isaacgym.sh timeout "${WINDOW_SEC}" \
  scripts/screwdriver_student_diffusion_latent_robust_light.sh "${GPU_ID}" "${SEED}" "${DIFF_CACHE}" \
  "checkpoint=outputs/XHandHoraScrewDriver_teacher/${TEACHER_CACHE}/stage1_nn/best_reward_*.pth" \
  2>&1 | tee "${LOG_PATH}"

BEST=$(rg -o "Current Best: -?[0-9]+\\.?[0-9]*" "${LOG_PATH}" | awk '{print $3}' | sort -g | tail -n 1 || true)

echo "Diffusion latent robust-light 15-min acceptance finished."
echo "Teacher cache: ${TEACHER_CACHE}"
echo "Diffusion cache: ${DIFF_CACHE}"
echo "Window (sec): ${WINDOW_SEC}"
echo "Log: ${LOG_PATH}"
if [ -n "${BEST}" ]; then
  echo "Max Current Best: ${BEST}"
else
  echo "Max Current Best: N/A"
fi
