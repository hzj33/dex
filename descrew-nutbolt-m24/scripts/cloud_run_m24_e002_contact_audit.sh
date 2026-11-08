#!/usr/bin/env bash
set -euo pipefail
cd "${ROOT:-/root/code/dexscrew-repro}"
PIPE_DIR="${PIPE_DIR:-outputs/cloud_pipeline_m24_humantwist_e002_contacts_$(date +%Y%m%d_%H%M%S)}"
AUDIT_SOURCE="${AUDIT_SOURCE:-outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_axisfix_s42_20260908_113745/manifests/a0c_diagnostic.json}"
AUDIT_IDS="${AUDIT_IDS:-0 34 36 46 63 64}"
AUDIT_MODES="${AUDIT_MODES:-tensor_reference rigid_contacts}"
[[ ! -e "${PIPE_DIR}" ]] || { echo "Refusing to overwrite ${PIPE_DIR}" >&2; exit 2; }
mkdir -p "${PIPE_DIR}"/{commands,logs,status,manifests,source}
date --iso-8601=seconds > "${PIPE_DIR}/started_at.txt"
printf 'contact_audit_only\n' > "${PIPE_DIR}/phase.txt"
trap 'date --iso-8601=seconds > "${PIPE_DIR}/finished_at.txt"' EXIT
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dexscrew-ig
git rev-parse HEAD > "${PIPE_DIR}/source/git_commit.txt"
git status --short > "${PIPE_DIR}/source/git_status.txt"
for file in scripts/audit_m24_e002_contacts.py scripts/diagnose_m24_e002_a0c.py \
    scripts/m24_e002_a0c_diagnostics.py scripts/calibrate_m24_e002_a0.py \
    scripts/cloud_run_m24_e002_contact_audit.sh dexscrew/tasks/xhand_pasini.py \
    dexscrew/tasks/base/vec_task.py configs/task/XHandPasiniM24NutBoltE002.yaml \
    configs/task/XHandPasiniM24NutBolt.yaml configs/task/XHandPasiniNutBolt.yaml \
    assets/screw/m24hex/0000_m24x3_160.urdf; do
  cp "${file}" "${PIPE_DIR}/source/$(basename "${file}")"
  sha256sum "${file}" >> "${PIPE_DIR}/source/SHA256SUMS"
done
sha256sum "${AUDIT_SOURCE}" > "${PIPE_DIR}/source/input_SHA256SUMS"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_start.csv"
for candidate in ${AUDIT_IDS}; do
  for mode in ${AUDIT_MODES}; do
    gpu_pipeline=True
    [[ "${mode}" != rigid_contacts ]] || gpu_pipeline=False
    name="${mode}_candidate${candidate}"
    command=(timeout --signal=TERM --kill-after=30s 180 python -u scripts/audit_m24_e002_contacts.py
      task=XHandPasiniM24NutBoltE002 headless=True seed=42 task.env.numEnvs=1
      sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 num_threads=16
      "task.sim.use_gpu_pipeline=${gpu_pipeline}"
      task.env.e002_motion.torque_source=link_origin_proxy
      "+contact_audit_source=${AUDIT_SOURCE}" "+contact_candidate_id=${candidate}"
      "+contact_force_sensor_audit=${FORCE_SENSOR_AUDIT:-False}"
      "+contact_kinematic_audit=${KINEMATIC_AUDIT:-False}"
      "+contact_no_contact_control=${NO_CONTACT_CONTROL:-False}"
      "+contact_audit_mode=${mode}" "+contact_audit_out=${PIPE_DIR}/manifests/${name}.json")
    printf 'CUDA_VISIBLE_DEVICES=0 ' > "${PIPE_DIR}/commands/${name}.sh"
    printf '%q ' "${command[@]}" >> "${PIPE_DIR}/commands/${name}.sh"
    printf '\n' >> "${PIPE_DIR}/commands/${name}.sh"
    date --iso-8601=seconds > "${PIPE_DIR}/status/${name}.started_at"
    set +e
    CUDA_VISIBLE_DEVICES=0 "${command[@]}" > "${PIPE_DIR}/logs/${name}.log" 2>&1
    rc=$?
    set -e
    date --iso-8601=seconds > "${PIPE_DIR}/status/${name}.finished_at"
    printf '%s\n' "${rc}" > "${PIPE_DIR}/status/${name}.exit_status"
    if [[ "${rc}" != 0 ]]; then
      printf 'failed:%s:%s\n' "${name}" "${rc}" > "${PIPE_DIR}/phase.txt"
      exit "${rc}"
    fi
  done
done
printf 'completed:contact_audit_only\n' > "${PIPE_DIR}/phase.txt"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_end.csv"
