#!/usr/bin/env bash
set -euo pipefail

# Collect teacher rollout data in docker (stable Isaac Gym + Python 3.8 runtime).
# Usage:
#   scripts/collect_screwdriver_teacher_rollout_docker.sh GPU_ID SEED TEACHER_CACHE [STEPS] [TAG]

GPU_ID=${1:-0}
SEED=${2:-42}
TEACHER_CACHE=${3:?Missing TEACHER_CACHE, e.g. run_a}
STEPS=${4:-256}
TAG=${5:-run_a}
COLLECT_CACHE="${TEACHER_CACHE}_collect"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PROJECT_DIR="${PROJECT_DIR:-${REPO_DIR}}"
ISAACGYM_DIR="${ISAACGYM_DIR:-$HOME/Codefield/third_party/isaacgym_preview4}"
IMAGE="${DEXSCREW_IMAGE:-dexscrew:ig20-py38}"
ISAACGYM_BINDING="${ISAACGYM_DIR}/isaacgym/python/isaacgym/_bindings/linux-x86_64/gym_38.so"

OUT_FILE="outputs/teacher_rollouts/XHandHoraScrewDriver_teacher/${TEACHER_CACHE}/${TAG}_seed${SEED}_steps${STEPS}.pt"

array=("$@")
len=${#array[@]}
EXTRA_ARGS=("${array[@]:5:$len}")

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

mkdir -p "${PROJECT_DIR}/$(dirname "${OUT_FILE}")"

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
python train.py \
  task=XHandHoraScrewDriver \
  train.algo=PPO \
  headless=True \
  seed=${SEED} \
  sim_device=cuda:${GPU_ID} \
  rl_device=cuda:${GPU_ID} \
  graphics_device_id=7 \
  task.env.numEnvs=64 \
  test=True \
  wandb_activate=False \
  train.ppo.output_name=XHandHoraScrewDriver_teacher/${COLLECT_CACHE} \
  \"checkpoint=outputs/XHandHoraScrewDriver_teacher/${TEACHER_CACHE}/stage1_nn/best_reward_*.pth\" \
  task.env.randomization.randomizePDGains=False \
  task.env.randomization.action_noise_e_scale=0.0 \
  task.env.randomization.action_noise_t_scale=0.0 \
  task.env.randomization.obs_noise_e_scale=0.0 \
  task.env.randomization.obs_noise_t_scale=0.0 \
  task.env.randomization.noisy_rpy_scale=0.0 \
  task.env.randomization.noisy_pos_scale=0.0 \
  task.env.forceScale=0.0 \
  task.env.randomForceProbScalar=0.0 \
  +collect_rollout=True \
  +collect_steps=${STEPS} \
  +collect_save_point_cloud=True \
  +collect_out=${OUT_FILE} \
  ${EXTRA_ARGS[*]}
"

echo "Teacher rollout collection finished."
echo "Output: ${OUT_FILE}"
