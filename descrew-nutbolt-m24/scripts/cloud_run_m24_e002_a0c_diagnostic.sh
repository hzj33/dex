#!/usr/bin/env bash
set -euo pipefail
# A0c measurement only. There is deliberately no downstream training command.
ROOT="${ROOT:-/root/code/dexscrew-repro}"
cd "${ROOT}"
PIPE_DIR="${PIPE_DIR:-outputs/cloud_pipeline_m24_humantwist_e002_a0c_diag_$(date +%Y%m%d_%H%M%S)}"
POSE_JSON="${POSE_JSON:-outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1b_s42_20260908_103643/manifests/a0_calibration.json}"
[[ ! -e "${PIPE_DIR}" ]] || { echo "Refusing to overwrite ${PIPE_DIR}" >&2; exit 2; }
mkdir -p "${PIPE_DIR}"/{commands,logs,status,manifests,source}
date --iso-8601=seconds > "${PIPE_DIR}/started_at.txt"
printf 'diagnostic_only\n' > "${PIPE_DIR}/phase.txt"
trap 'date --iso-8601=seconds > "${PIPE_DIR}/finished_at.txt"' EXIT
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dexscrew-ig
git rev-parse HEAD > "${PIPE_DIR}/source/git_commit.txt"
git status --short > "${PIPE_DIR}/source/git_status.txt"
for file in scripts/diagnose_m24_e002_a0c.py scripts/m24_e002_a0c_diagnostics.py \
    scripts/test_m24_e002_a0c_diagnostics.py scripts/test_m24_e002_reward_source.py \
    scripts/test_m24_e002_learning_contract.py scripts/m24_e002_evaluation_contract.py scripts/calibrate_m24_e002_a0.py \
    scripts/cloud_run_m24_e002_a0c_diagnostic.sh dexscrew/tasks/xhand_pasini.py dexscrew/tasks/contact_moments.py \
    configs/task/XHandPasiniM24NutBoltE002.yaml configs/task/XHandPasiniM24NutBolt.yaml \
    configs/task/XHandPasiniNutBolt.yaml; do
  cp "${file}" "${PIPE_DIR}/source/$(basename "${file}")"
  sha256sum "${file}" >> "${PIPE_DIR}/source/SHA256SUMS"
done
cp "${POSE_JSON}" "${PIPE_DIR}/manifests/source_pose.json"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_start.csv"
python scripts/test_m24_e002_a0.py > "${PIPE_DIR}/logs/a0_unit.log" 2>&1
python scripts/test_m24_e002_a0c_diagnostics.py > "${PIPE_DIR}/logs/diagnostic_unit.log" 2>&1
python scripts/test_m24_e002_reward_source.py > "${PIPE_DIR}/logs/reward_source_unit.log" 2>&1
python scripts/test_m24_e002_learning_contract.py > "${PIPE_DIR}/logs/learning_contract_unit.log" 2>&1
command=(timeout --signal=TERM --kill-after=30s 300 python -u scripts/diagnose_m24_e002_a0c.py
  task=XHandPasiniM24NutBoltE002 headless=True seed=42 "task.env.numEnvs=${NUM_ENVS:-67}"
  sim_device=cuda:0 rl_device=cuda:0 graphics_device_id=0 num_threads=16
  "task.env.e002_motion.torque_source=${TORQUE_SOURCE:-contact_point_normal}"
  "+diagnostic_pose_json=${POSE_JSON}"
  "+diagnostic_probe_steps=${PROBE_STEPS:-10}"
  "+diagnostic_out=${PIPE_DIR}/manifests/a0c_diagnostic.json")
if [[ -n "${CONTACT_BASIS_JSON:-}" ]]; then
  cp "${CONTACT_BASIS_JSON}" "${PIPE_DIR}/manifests/contact_basis.json"
  cp scripts/build_m24_contact_tangent_basis.py "${PIPE_DIR}/source/"
  sha256sum scripts/build_m24_contact_tangent_basis.py "${CONTACT_BASIS_JSON}" >> "${PIPE_DIR}/source/SHA256SUMS"
  command+=("+diagnostic_contact_basis_json=${PIPE_DIR}/manifests/contact_basis.json")
fi
if [[ -n "${DIAGNOSTIC_AMPLITUDES:-}" ]]; then
  command+=("+diagnostic_amplitudes=${DIAGNOSTIC_AMPLITUDES}")
fi
if [[ -n "${ACTION_BANK_JSON:-}" ]]; then
  cp "${ACTION_BANK_JSON}" "${PIPE_DIR}/manifests/action_bank.json"
  cp scripts/build_m24_contact_action_bank.py "${PIPE_DIR}/source/"
  sha256sum scripts/build_m24_contact_action_bank.py "${ACTION_BANK_JSON}" >> "${PIPE_DIR}/source/SHA256SUMS"
  command+=("+diagnostic_action_bank_json=${PIPE_DIR}/manifests/action_bank.json")
fi
printf 'CUDA_VISIBLE_DEVICES=0 ' > "${PIPE_DIR}/commands/a0c.sh"
printf '%q ' "${command[@]}" >> "${PIPE_DIR}/commands/a0c.sh"
printf '\n' >> "${PIPE_DIR}/commands/a0c.sh"
set +e
CUDA_VISIBLE_DEVICES=0 "${command[@]}" > "${PIPE_DIR}/logs/a0c.log" 2>&1
rc=$?
set -e
printf '%s\n' "${rc}" > "${PIPE_DIR}/status/a0c.exit_status"
if [[ "${rc}" == 0 ]]; then
  python - "${PIPE_DIR}/manifests/a0c_diagnostic.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
assert data['status'] == 'passed' and data['diagnostic_only']
assert data['selected']['passed']
PY
  printf 'passed:diagnostic_only\n' > "${PIPE_DIR}/phase.txt"
elif [[ "${rc}" == 4 ]]; then
  printf 'failed:a0c_gate\n' > "${PIPE_DIR}/phase.txt"
elif [[ "${rc}" == 124 ]]; then
  printf 'incomplete:diagnostic_timeout\n' > "${PIPE_DIR}/phase.txt"
else
  printf 'failed:runtime:%s\n' "${rc}" > "${PIPE_DIR}/phase.txt"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader > "${PIPE_DIR}/gpu_at_end.csv"
exit "${rc}"
