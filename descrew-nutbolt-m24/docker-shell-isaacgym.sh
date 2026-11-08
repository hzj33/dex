#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${DEXSCREW_CONTAINER_NAME:-dexscrew_isaacgym_shell}"

if docker ps --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Entering running IsaacGym container: ${CONTAINER_NAME}"
  exec docker exec -it "${CONTAINER_NAME}" bash
fi

if docker ps -a --format '{{.Names}}' | grep -Fxq "${CONTAINER_NAME}"; then
  echo "Removing stopped IsaacGym container with the same name: ${CONTAINER_NAME}"
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

echo "Starting IsaacGym shell container: ${CONTAINER_NAME}"
echo "Run training inside with: scripts/run_with_cleanup.sh bash scripts/<train_script>.sh ..."
DEXSCREW_CONTAINER_NAME="${CONTAINER_NAME}" exec "${SCRIPT_DIR}/docker-run-isaacgym.sh"
