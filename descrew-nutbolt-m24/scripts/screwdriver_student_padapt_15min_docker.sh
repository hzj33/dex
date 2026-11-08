#!/usr/bin/env bash
set -euo pipefail

# Canonical 15-min acceptance run for current ProprioAdapt-style student.
# Usage:
#   scripts/screwdriver_student_padapt_15min_docker.sh GPU_ID SEED TEACHER_CACHE [WINDOW_SEC] [STUDENT_CACHE]
# Example:
#   scripts/screwdriver_student_padapt_15min_docker.sh 0 42 run_a 900 run_a_seed42_15min

GPU_ID=${1:-0}
SEED=${2:-42}
TEACHER_CACHE=${3:?Missing TEACHER_CACHE, e.g. run_a}
WINDOW_SEC=${4:-900}
STUDENT_CACHE=${5:-${TEACHER_CACHE}}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$HOME/Codefield/third_party/isaacgym_preview4}"
IMAGE="${DEXSCREW_IMAGE:-dexscrew:ig20-py38}"
ISAACGYM_BINDING="${ISAACGYM_DIR}/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so"

OUT_DIR="outputs/XHandHoraScrewDriver_student_padapt/${STUDENT_CACHE}"
LOG_PATH="${OUT_DIR}/train_${WINDOW_SEC}s.log"

mkdir -p "${PROJECT_DIR}/${OUT_DIR}"

if [[ ! -f "${ISAACGYM_BINDING}" ]]; then
  echo "Isaac Gym binding not found: ${ISAACGYM_BINDING}"
  echo "Please set ISAACGYM_DIR to IsaacGym Preview4 root."
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE}"
  echo "Build it with: docker build -t ${IMAGE} -f Dockerfile.isaacgym ."
  exit 1
fi

docker run --rm \
  --gpus "device=${GPU_ID}" \
  --runtime=nvidia \
  --user "$(id -u):$(id -g)" \
  --ipc=host \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_ID}" \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  -e ISAACGYM_PATH=/opt/isaacgym \
  -e PYTHONPATH=/opt/isaacgym/isaacgym/python \
  -v "${PROJECT_DIR}:/workspace/dexscrew-repro" \
  -v "${ISAACGYM_DIR}:/opt/isaacgym:ro" \
  -w /workspace/dexscrew-repro \
  "${IMAGE}" \
  bash -lc "
set -euo pipefail
python - <<'PY'
import isaacgym  # noqa: F401
import ninja  # noqa: F401
PY
set +e
timeout ${WINDOW_SEC} scripts/screwdriver_student_padapt.sh ${GPU_ID} ${SEED} ${STUDENT_CACHE} \
  \"checkpoint=outputs/XHandHoraScrewDriver_teacher/${TEACHER_CACHE}/stage1_nn/best_reward_*.pth\" \
  2>&1 | tee ${LOG_PATH}
RUN_RC=\${PIPESTATUS[0]}
set -e
if [ \${RUN_RC} -ne 0 ] && [ \${RUN_RC} -ne 124 ]; then
  exit \${RUN_RC}
fi
"

BEST=$(rg -o "Current Best: -?[0-9]+\\.?[0-9]*" "${PROJECT_DIR}/${LOG_PATH}" | awk '{print $3}' | sort -g | tail -n 1 || true)

echo "Student 15-min acceptance finished."
echo "Teacher cache: ${TEACHER_CACHE}"
echo "Student cache: ${STUDENT_CACHE}"
echo "Window (sec): ${WINDOW_SEC}"
echo "Log: ${LOG_PATH}"
if [ -n "${BEST}" ]; then
  echo "Max Current Best: ${BEST}"
else
  echo "Max Current Best: N/A"
fi
