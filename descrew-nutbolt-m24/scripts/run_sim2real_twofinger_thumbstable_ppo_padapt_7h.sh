#!/usr/bin/env bash
set -u

GPU="${1:-0}"
SEED="${2:-42}"
TEACHER_CACHE="${3:-sim2real_twofinger_thumbstable}"
STUDENT_CACHE="${4:-${TEACHER_CACHE}_padapt}"
PHASE_SEC="${PHASE_SEC:-12600}"
LOG_ROOT="outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_ROOT}/pipeline_${STAMP}.log"

mkdir -p "${LOG_ROOT}"
ln -sfn "pipeline_${STAMP}.log" "${LOG_ROOT}/latest.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

log() {
  echo "[$(date '+%F %T')] $*"
}

select_best_ckpt() {
  local ckpt_dir="$1"
  python - "$ckpt_dir" <<'PY'
import re
import sys
from pathlib import Path

ckpt_dir = Path(sys.argv[1])
paths = list(ckpt_dir.glob("best_reward_*.pth"))
if not paths:
    raise SystemExit(f"No best_reward_*.pth found in {ckpt_dir}")

def score(path: Path):
    m = re.search(r"best_reward_([-+]?[0-9]*\.?[0-9]+)\.pth$", path.name)
    if m:
        return (float(m.group(1)), path.stat().st_mtime)
    return (float("-inf"), path.stat().st_mtime)

print(max(paths, key=score))
PY
}

log "pipeline_start"
log "gpu=${GPU} seed=${SEED} teacher_cache=${TEACHER_CACHE} student_cache=${STUDENT_CACHE}"
log "phase_sec=${PHASE_SEC} total_nominal_sec=$((PHASE_SEC * 2))"
log "teacher_output=outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/${TEACHER_CACHE}"
log "student_output=outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/${STUDENT_CACHE}"

TEACHER_CMD=(
  ./docker-run-isaacgym.sh
  timeout "${PHASE_SEC}"
  scripts/run_with_cleanup.sh
  bash scripts/dexh13_lightbulb_teacher_sim2real_twofinger.sh
  "${GPU}" "${SEED}" "${TEACHER_CACHE}" True
  wandb_activate=True
  task.env.termination.log=True
)

log "teacher_command=${TEACHER_CMD[*]}"
"${TEACHER_CMD[@]}"
teacher_status=$?
log "teacher_exit_status=${teacher_status}"
if [[ "${teacher_status}" != "0" && "${teacher_status}" != "124" ]]; then
  log "teacher_failed_status=${teacher_status}"
  exit "${teacher_status}"
fi

TEACHER_CKPT_DIR="outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/${TEACHER_CACHE}/stage1_nn"
TEACHER_CKPT="$(select_best_ckpt "${TEACHER_CKPT_DIR}")"
log "selected_teacher_ckpt=${TEACHER_CKPT}"

STUDENT_CMD=(
  ./docker-run-isaacgym.sh
  timeout "${PHASE_SEC}"
  scripts/run_with_cleanup.sh
  bash scripts/dexh13_lightbulb_student_padapt_sim2real_twofinger.sh
  "${GPU}" "${SEED}" "${STUDENT_CACHE}" "${TEACHER_CKPT}"
  wandb_activate=True
  task.env.termination.log=True
)

log "student_command=${STUDENT_CMD[*]}"
"${STUDENT_CMD[@]}"
student_status=$?
log "student_exit_status=${student_status}"
if [[ "${student_status}" != "0" && "${student_status}" != "124" ]]; then
  log "student_failed_status=${student_status}"
  exit "${student_status}"
fi

log "student_checkpoints:"
find "outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/${STUDENT_CACHE}" \
  -maxdepth 3 -type f \( -name 'model_best.ckpt' -o -name '*.ckpt' -o -name 'events.out.tfevents*' \) \
  -printf '%p\n' | sort || true

log "pipeline_done"
