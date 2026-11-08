#!/usr/bin/env bash
set -u

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <command> [args...]" >&2
  echo "Example: $0 bash scripts/dexh13_lightbulb_teacher_middle_scalepos.sh 0 42 run True" >&2
  exit 2
fi

child_pid=""

cleanup_child() {
  local signal="${1:-TERM}"
  if [[ -z "${child_pid}" ]]; then
    return 0
  fi
  if ! kill -0 "${child_pid}" 2>/dev/null; then
    return 0
  fi

  echo
  echo "Stopping training process group ${child_pid} with SIG${signal}..."
  kill -s "${signal}" -- "-${child_pid}" 2>/dev/null || kill -s "${signal}" "${child_pid}" 2>/dev/null || true
}

on_interrupt() {
  trap - INT TERM HUP
  cleanup_child INT
  sleep 2
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    cleanup_child TERM
    sleep 2
  fi
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    cleanup_child KILL
  fi
  wait "${child_pid}" 2>/dev/null || true
  exit 130
}

trap on_interrupt INT TERM HUP

if command -v setsid >/dev/null 2>&1; then
  setsid "$@" &
else
  echo "Warning: setsid not found; Ctrl+C cleanup may not reach all descendants." >&2
  "$@" &
fi
child_pid=$!

wait "${child_pid}"
status=$?
child_pid=""
trap - INT TERM HUP
exit "${status}"
