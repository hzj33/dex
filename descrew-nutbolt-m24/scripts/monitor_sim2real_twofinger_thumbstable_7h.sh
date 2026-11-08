#!/usr/bin/env bash
set -u

ROOT_DIR="${1:-outputs/sim2real_twofinger_thumbstable_ppo_padapt_7h}"
INTERVAL_SEC="${INTERVAL_SEC:-300}"
PIPE_PID_FILE="${ROOT_DIR}/pipeline.pid"
PIPE_LOG="${ROOT_DIR}/latest.log"

PIPE_PID=""
if [[ -f "${PIPE_PID_FILE}" ]]; then
  PIPE_PID="$(cat "${PIPE_PID_FILE}" 2>/dev/null || true)"
fi

while true; do
  echo "===== $(date '+%F %T') ====="

  if [[ -n "${PIPE_PID}" ]] && kill -0 "${PIPE_PID}" 2>/dev/null; then
    echo "pipeline_alive pid=${PIPE_PID}"
  else
    echo "pipeline_not_alive pid=${PIPE_PID:-unknown}"
  fi

  if grep -q "student_command=" "${PIPE_LOG}" 2>/dev/null; then
    echo "stage=student_or_done"
  else
    echo "stage=teacher"
  fi

  grep -E \
    "teacher_exit_status|selected_teacher_ckpt|student_command|student_exit_status|pipeline_done" \
    "${PIPE_LOG}" 2>/dev/null | tail -n 10 || true

  grep -E "Agent Steps:|mean_rewards:|Current Best:|save current best reward" \
    "${PIPE_LOG}" 2>/dev/null | tail -n 8 || true

  echo "teacher_best_ckpts:"
  find outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger/sim2real_twofinger_thumbstable/stage1_nn \
    -maxdepth 1 -type f -name "best_reward_*.pth" \
    -printf "%TY-%Tm-%Td %TH:%TM %p\n" 2>/dev/null | sort | tail -n 5 || true

  echo "student_ckpts:"
  find outputs/Dexh13HoraLightbulb_student_padapt_sim2real_twofinger/sim2real_twofinger_thumbstable_padapt \
    -maxdepth 3 -type f \( -name "model_best.ckpt" -o -name "*.ckpt" \) \
    -printf "%TY-%Tm-%Td %TH:%TM %p\n" 2>/dev/null | sort | tail -n 5 || true

  echo "gpu:"
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader 2>/dev/null || true
  nvidia-smi \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits 2>/dev/null || true

  recent_log="$(tail -n 120 "${PIPE_LOG}" 2>/dev/null || true)"
  if printf "%s\n" "${recent_log}" | grep -qiE \
    "Segmentation fault|PxgCudaDeviceMemoryAllocator fail|Traceback|FileNotFoundError|RuntimeError:|CUDA out of memory"; then
    echo "ALERT error_pattern_detected"
    printf "%s\n" "${recent_log}" | grep -iE \
      "Segmentation fault|PxgCudaDeviceMemoryAllocator fail|Traceback|FileNotFoundError|RuntimeError:|CUDA out of memory" \
      | tail -n 20 || true
  fi

  if [[ -z "${PIPE_PID}" ]] || ! kill -0 "${PIPE_PID}" 2>/dev/null; then
    echo "monitor_exit_pipeline_done_or_dead"
    exit 0
  fi

  sleep "${INTERVAL_SEC}"
done
