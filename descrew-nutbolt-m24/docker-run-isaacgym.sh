#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR}}"

CONTAINER_ISAACGYM_BINDING="/opt/isaacgym/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so"
if [[ -f "/.dockerenv" ]] && [[ -f "${CONTAINER_ISAACGYM_BINDING}" ]]; then
  export ISAACGYM_DIR="${ISAACGYM_DIR:-/opt/isaacgym}"
  export ISAACGYM_PATH="${ISAACGYM_PATH:-/opt/isaacgym}"
  export PYTHONPATH="/opt/isaacgym/isaacgym/python${PYTHONPATH:+:${PYTHONPATH}}"
  if [[ $# -gt 0 ]]; then
    exec "$@"
  fi
  exec bash
fi

if [[ -z "${ISAACGYM_DIR:-}" ]]; then
  if [[ -d "$HOME/Codefield/third_party/isaacgym_preview4" ]]; then
    ISAACGYM_DIR="$HOME/Codefield/third_party/isaacgym_preview4"
  elif [[ -d "$HOME/isaacgym" ]]; then
    ISAACGYM_DIR="$HOME/isaacgym"
  else
    ISAACGYM_DIR="$HOME/isaacgym"
  fi
fi

ISAACGYM_BINDING="${ISAACGYM_DIR}/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so"
USER_ID="${USER_ID:-$(id -u)}"
GROUP_ID="${GROUP_ID:-$(id -g)}"
IMAGE="${DEXSCREW_IMAGE:-dexscrew:ig20-py38}"
CONTAINER_NAME="${DEXSCREW_CONTAINER_NAME:-}"

if [[ ! -d "${PROJECT_DIR}" ]]; then
  echo "PROJECT_DIR not found: ${PROJECT_DIR}"
  exit 1
fi
if [[ ! -f "${ISAACGYM_BINDING}" ]]; then
  echo "Isaac Gym binding not found: ${ISAACGYM_BINDING}"
  echo "Please set ISAACGYM_DIR correctly, e.g.:"
  echo "  export ISAACGYM_DIR=$HOME/Codefield/third_party/isaacgym_preview4"
  exit 1
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE}"
  echo "Build it with:"
  echo "  docker build -t ${IMAGE} -f Dockerfile.isaacgym ."
  exit 1
fi

if [[ $# -gt 0 ]]; then
  CONTAINER_CMD=("$@")
  DOCKER_TTY_FLAGS=()
else
  CONTAINER_CMD=(bash)
  DOCKER_TTY_FLAGS=(-it)
fi

PASSWD_GROUP_MOUNTS=()
if [[ -f /etc/passwd ]]; then
  PASSWD_GROUP_MOUNTS+=(-v /etc/passwd:/etc/passwd:ro)
fi
if [[ -f /etc/group ]]; then
  PASSWD_GROUP_MOUNTS+=(-v /etc/group:/etc/group:ro)
fi

WANDB_AUTH_MOUNTS=()
HOST_NETRC="${HOME}/.netrc"
HOST_WANDB_CONFIG_DIR="${HOME}/.config/wandb"
if [[ -f "${HOST_NETRC}" ]]; then
  WANDB_AUTH_MOUNTS+=(-v "${HOST_NETRC}:/tmp/.netrc:ro")
fi
if [[ -d "${HOST_WANDB_CONFIG_DIR}" ]]; then
  WANDB_AUTH_MOUNTS+=(-v "${HOST_WANDB_CONFIG_DIR}:/tmp/.config/wandb:ro")
fi

X11_ARGS=()
if [[ -n "${DISPLAY:-}" ]] && [[ -d /tmp/.X11-unix ]]; then
  HOST_XAUTHORITY="${XAUTHORITY:-}"
  if [[ -z "${HOST_XAUTHORITY}" ]] || [[ ! -f "${HOST_XAUTHORITY}" ]]; then
    for candidate in "/run/user/$(id -u)/gdm/Xauthority" "$HOME/.Xauthority"; do
      if [[ -f "${candidate}" ]]; then
        HOST_XAUTHORITY="${candidate}"
        break
      fi
    done
  fi

  X11_ARGS+=(-e "DISPLAY=${DISPLAY}")
  X11_ARGS+=(-e QT_X11_NO_MITSHM=1)
  X11_ARGS+=(-v /tmp/.X11-unix:/tmp/.X11-unix:rw)

  if [[ -n "${HOST_XAUTHORITY:-}" ]] && [[ -f "${HOST_XAUTHORITY}" ]]; then
    X11_ARGS+=(-e XAUTHORITY=/tmp/.docker.xauth)
    X11_ARGS+=(-v "${HOST_XAUTHORITY}:/tmp/.docker.xauth:ro")
  else
    echo "Warning: XAUTHORITY file not found; GUI viewer may fail."
    echo "Try exporting XAUTHORITY first, e.g.:"
    echo "  export XAUTHORITY=/run/user/$(id -u)/gdm/Xauthority"
  fi
fi

NAME_ARGS=()
if [[ -n "${CONTAINER_NAME}" ]]; then
  NAME_ARGS=(--name "${CONTAINER_NAME}")
fi

docker run --rm "${DOCKER_TTY_FLAGS[@]}" \
  "${NAME_ARGS[@]}" \
  --init \
  --gpus all \
  --runtime=nvidia \
  --user "${USER_ID}:${GROUP_ID}" \
  --ipc=host \
  --network host \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HOME=/tmp \
  -e XDG_CACHE_HOME=/tmp/.cache \
  -e TORCH_EXTENSIONS_DIR=/tmp/torch_extensions \
  -e ISAACGYM_PATH=/opt/isaacgym \
  -e PYTHONPATH=/opt/isaacgym/isaacgym/python \
  -v "${PROJECT_DIR}:/workspace/dexscrew-repro" \
  -v "${ISAACGYM_DIR}:/opt/isaacgym:ro" \
  "${PASSWD_GROUP_MOUNTS[@]}" \
  "${WANDB_AUTH_MOUNTS[@]}" \
  "${X11_ARGS[@]}" \
  -w /workspace/dexscrew-repro \
  "${IMAGE}" \
  "${CONTAINER_CMD[@]}"
