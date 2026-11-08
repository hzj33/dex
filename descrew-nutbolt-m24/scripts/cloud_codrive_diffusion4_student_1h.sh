#!/usr/bin/env bash
set -euo pipefail

# SeeTaCloud entrypoint for four parallel CoDrive student distillations.
# All four runs use the same CoDrive PPO teacher and YAML base, with separate
# algorithm heads and output folders for direct visual/eval comparison.

ROOT="${ROOT:-/root/code/dexscrew-repro}"
cd "${ROOT}"

if [[ -f /root/miniconda3/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /root/miniconda3/etc/profile.d/conda.sh
elif [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate dexscrew-ig

export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

TASK="Dexh13HoraLightbulbSim2RealTwoFingerCoDrive"
SEED="${SEED:-42}"
GPU="${GPU:-0}"
WINDOW_SEC="${WINDOW_SEC:-3600}"
NUM_ENVS="${NUM_ENVS:-48}"
MINIBATCH="${MINIBATCH:-576}"
RUN_TAG="${RUN_TAG:-codrive_diffusion4_s${SEED}_${WINDOW_SEC}s_$(date +%Y%m%d_%H%M%S)}"
PIPE_ROOT="${PIPE_ROOT:-outputs/cloud_pipeline_codrive_diffusion4_1h}"
PIPE_DIR="${PIPE_ROOT}/${RUN_TAG}"
TEACHER_CKPT="${TEACHER_CKPT:-sim2real/codrive/best_reward_4159.37.pth}"

mkdir -p "${PIPE_DIR}/logs"
ln -sfn "${RUN_TAG}" "${PIPE_ROOT}/latest"
printf 'starting\n' > "${PIPE_DIR}/phase.txt"

if [[ ! -f "${TEACHER_CKPT}" ]]; then
  echo "[pipeline] missing teacher checkpoint: ${TEACHER_CKPT}" >&2
  printf 'failed\n' > "${PIPE_DIR}/phase.txt"
  exit 2
fi

cp "configs/task/${TASK}.yaml" "${PIPE_DIR}/${TASK}.task.yaml.used"
cp "configs/train/${TASK}.yaml" "${PIPE_DIR}/${TASK}.train.yaml.used"
cp "${TEACHER_CKPT}" "${PIPE_DIR}/teacher_best_reward_4159.37.pth.used"

common_args=(
  "task=${TASK}"
  headless=True
  "seed=${SEED}"
  "sim_device=cuda:${GPU}"
  "rl_device=cuda:${GPU}"
  graphics_device_id=7
  train.ppo.proprio_adapt=True
  task.env.termination.grace_steps=0
  task.env.termination.enable_finger_dist=True
  task.env.termination.enable_nut_stagnation=True
  task.env.termination.enable_no_contact=True
  task.env.termination.enable_screw_limit=True
  task.env.termination.log=True
  task.env.randomization.obs_noise_t_scale=0.01
  task.env.randomization.obs_noise_e_scale=0.02
  "task.env.numEnvs=${NUM_ENVS}"
  "train.ppo.minibatch_size=${MINIBATCH}"
  wandb_activate=False
  "checkpoint=${TEACHER_CKPT}"
)

algo_extra_args() {
  local label="$1"
  case "${label}" in
    diffusion_latent)
      printf '%s\n' \
        '+train.ppo.diffusion_steps=10' \
        '+train.ppo.diffusion_steps_infer=10' \
        '+train.ppo.diffusion_lr=3e-4' \
        '+train.ppo.diffusion_loss_coef=1.0' \
        '+train.ppo.bc_loss_coef=1.0'
      ;;
    consistency_latent)
      printf '%s\n' \
        '+train.ppo.consistency_lr=3e-4' \
        '+train.ppo.consistency_loss_coef=1.0' \
        '+train.ppo.consistency_boundary_coef=0.5' \
        '+train.ppo.consistency_num_scales=10' \
        '+train.ppo.bc_loss_coef=1.0'
      ;;
    flow_matching_latent)
      printf '%s\n' \
        '+train.ppo.flow_lr=3e-4' \
        '+train.ppo.flow_loss_coef=1.0' \
        '+train.ppo.flow_infer_steps=1' \
        '+train.ppo.bc_loss_coef=1.0'
      ;;
    diffusion_action_chunk)
      printf '%s\n' \
        '+train.ppo.action_chunk_len=8' \
        '+train.ppo.action_chunk_diffusion_steps=10' \
        '+train.ppo.action_chunk_diffusion_steps_infer=10' \
        '+train.ppo.action_chunk_diffusion_lr=3e-4' \
        '+train.ppo.action_chunk_diffusion_loss_coef=1.0' \
        '+train.ppo.action_chunk_first_action_bc_loss_coef=1.0' \
        '+train.ppo.action_chunk_bc_loss_coef=0.1' \
        '+train.ppo.action_chunk_stochastic_infer=False' \
        '+train.ppo.action_chunk_teacher_mix_steps=120000'
      ;;
    *)
      echo "unknown label: ${label}" >&2
      return 2
      ;;
  esac
}

algo_class() {
  case "$1" in
    diffusion_latent) echo "DiffusionLatentStudent" ;;
    consistency_latent) echo "ConsistencyLatentStudent" ;;
    flow_matching_latent) echo "FlowMatchingLatentStudent" ;;
    diffusion_action_chunk) echo "DiffusionActionChunkStudent" ;;
    *) return 2 ;;
  esac
}

output_name() {
  case "$1" in
    diffusion_latent) echo "Dexh13HoraLightbulb_student_diffusion_latent_codrive/${RUN_TAG}" ;;
    consistency_latent) echo "Dexh13HoraLightbulb_student_consistency_codrive/${RUN_TAG}" ;;
    flow_matching_latent) echo "Dexh13HoraLightbulb_student_flow_matching_codrive/${RUN_TAG}" ;;
    diffusion_action_chunk) echo "Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive/${RUN_TAG}" ;;
    *) return 2 ;;
  esac
}

nn_dir() {
  case "$1" in
    diffusion_latent) echo "stage2_diffusion_nn" ;;
    consistency_latent) echo "stage2_consistency_nn" ;;
    flow_matching_latent) echo "stage2_flow_nn" ;;
    diffusion_action_chunk) echo "stage2_diffusion_action_chunk_nn" ;;
    *) return 2 ;;
  esac
}

run_one() {
  local label="$1"
  local class output log status_file
  class="$(algo_class "${label}")"
  output="$(output_name "${label}")"
  log="${PIPE_DIR}/logs/${label}.log"
  status_file="${PIPE_DIR}/${label}.status"

  mapfile -t extra < <(algo_extra_args "${label}")
  cmd=(
    timeout "${WINDOW_SEC}"
    python train.py
    "${common_args[@]}"
    "train.algo=${class}"
    "train.ppo.output_name=${output}"
    "experiment=student_codrive_${label}"
    "${extra[@]}"
  )

  {
    echo "[${label}] start $(date -Iseconds)"
    echo "[${label}] algo=${class}"
    echo "[${label}] output=outputs/${output}"
    echo "[${label}] nn_dir=outputs/${output}/$(nn_dir "${label}")"
    printf '[%s] command:' "${label}"
    printf ' %q' "${cmd[@]}"
    printf '\n'

    set +e
    CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}"
    status=$?
    set -e

    echo "${status}" > "${status_file}"
    echo "[${label}] exit_status=${status}"
    echo "[${label}] end $(date -Iseconds)"
    find "outputs/${output}" -maxdepth 3 -type f \( -name '*.ckpt' -o -name '*.pth' -o -name 'config_*.yaml' \) \
      -printf '%TY-%Tm-%Td %TH:%TM:%TS %p %s bytes\n' 2>/dev/null | sort | tail -30 || true

    if [[ "${status}" -ne 0 && "${status}" -ne 124 ]]; then
      exit "${status}"
    fi
  } > "${log}" 2>&1
}

monitor_gpu() {
  while true; do
    printf '%s\t' "$(date -Iseconds)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits || true
    sleep 30
  done
}

{
  echo "[pipeline] start $(date -Iseconds)"
  echo "[pipeline] host=$(hostname) root=${ROOT}"
  echo "[pipeline] task=${TASK}"
  echo "[pipeline] teacher=${TEACHER_CKPT}"
  echo "[pipeline] run_tag=${RUN_TAG}"
  echo "[pipeline] window_sec=${WINDOW_SEC}"
  echo "[pipeline] resource=num_envs=${NUM_ENVS} minibatch=${MINIBATCH}"
  nvidia-smi --query-gpu=name,driver_version,memory.used,memory.total --format=csv,noheader || true
} | tee "${PIPE_DIR}/pipeline.log"

monitor_gpu > "${PIPE_DIR}/gpu_monitor.tsv" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT

printf 'running\n' > "${PIPE_DIR}/phase.txt"

labels=(diffusion_latent consistency_latent flow_matching_latent diffusion_action_chunk)
pids=()
for label in "${labels[@]}"; do
  run_one "${label}" &
  pids+=("$!")
  sleep 20
done

overall=0
for idx in "${!labels[@]}"; do
  label="${labels[$idx]}"
  pid="${pids[$idx]}"
  if wait "${pid}"; then
    echo "[pipeline] ${label} wait_ok" | tee -a "${PIPE_DIR}/pipeline.log"
  else
    status=$?
    echo "[pipeline] ${label} wait_failed=${status}" | tee -a "${PIPE_DIR}/pipeline.log"
    overall=1
  fi
done

kill "${monitor_pid}" 2>/dev/null || true

summary="${PIPE_DIR}/summary.tsv"
printf 'label\tstatus\tbest_reward\tcheckpoint_dir\toutput_dir\n' > "${summary}"
for label in "${labels[@]}"; do
  status="$(cat "${PIPE_DIR}/${label}.status" 2>/dev/null || echo missing)"
  log="${PIPE_DIR}/logs/${label}.log"
  best="$(tr '\r' '\n' < "${log}" 2>/dev/null | grep -a -o 'Current Best: [-0-9.]*' | tail -1 | awk '{print $3}' || true)"
  output="$(output_name "${label}")"
  checkpoint_dir="outputs/${output}/$(nn_dir "${label}")"
  printf '%s\t%s\t%s\t%s\toutputs/%s\n' "${label}" "${status}" "${best:-NA}" "${checkpoint_dir}" "${output}" >> "${summary}"
done

if [[ "${overall}" -eq 0 ]]; then
  printf 'done\n' > "${PIPE_DIR}/phase.txt"
else
  printf 'failed\n' > "${PIPE_DIR}/phase.txt"
fi

{
  echo "[pipeline] end $(date -Iseconds)"
  echo "[pipeline] summary=${summary}"
  cat "${summary}"
} | tee -a "${PIPE_DIR}/pipeline.log"

exit "${overall}"
