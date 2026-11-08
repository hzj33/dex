#!/usr/bin/env bash

# Ensure current shell can import Isaac Gym with a compatible Python runtime.
# Returns 0 on success, non-zero on failure.
ensure_isaacgym_env() {
  local py_bin="${PYTHON_BIN:-python}"

  if "${py_bin}" - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info[:2] == (3, 8)
import isaacgym  # noqa: F401
PY
  then
    return 0
  fi

  local candidates=()
  if [[ -n "${ISAACGYM_DIR:-}" ]]; then
    candidates+=("${ISAACGYM_DIR}/isaacgym/python")
  fi
  candidates+=(
    "/opt/isaacgym/isaacgym/python"
    "$HOME/Codefield/third_party/isaacgym_preview4/isaacgym/python"
    "$HOME/isaacgym/isaacgym/python"
  )

  local p=""
  for p in "${candidates[@]}"; do
    if [[ -d "${p}" ]]; then
      export PYTHONPATH="${p}${PYTHONPATH:+:${PYTHONPATH}}"
      break
    fi
  done

  if "${py_bin}" - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info[:2] == (3, 8)
import isaacgym  # noqa: F401
PY
  then
    return 0
  fi

  local py_ver
  py_ver="$("${py_bin}" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
  echo "Isaac Gym environment check failed."
  echo "Current python: ${py_bin} (version ${py_ver})"
  echo "Need Python 3.8 + isaacgym importable."
  echo "Recommended:"
  echo "  1) ./docker-run-isaacgym.sh"
  echo "  2) then run this vis script inside that shell"
  return 1
}
