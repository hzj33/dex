#!/usr/bin/env bash
set -euo pipefail

# PLANS_v2 M1 baseline hardening pack:
# teacher PPO, current student (ProprioAdapt), pure BC
# unified protocol: nominal / light_v2 / hard
# seeds: default 42,43,44
#
# Usage:
#   scripts/eval_plansv2_m1_baseline_pack.sh [GPU_ID] [STEPS] [SEEDS_CSV]
#
# Example:
#   scripts/eval_plansv2_m1_baseline_pack.sh 0 256 42,43,44

GPU_ID=${1:-0}
STEPS=${2:-256}
SEEDS_CSV=${3:-42,43,44}
FROM_LOGS_ONLY=${PLANSV2_FROM_LOGS_ONLY:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
if [[ "${FROM_LOGS_ONLY}" != "1" ]]; then
  ensure_isaacgym_env
fi

OUT_DIR="outputs/robustness_eval/plansv2_m1"
mkdir -p "${OUT_DIR}"

TEACHER_CKPT="outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth"
PADAPT_CKPT="outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt"
PUREBC_CKPT="outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt"

if [[ "${FROM_LOGS_ONLY}" != "1" ]]; then
  if [[ ! -f "${TEACHER_CKPT}" ]]; then
    echo "Missing teacher ckpt: ${TEACHER_CKPT}"
    exit 1
  fi
  if [[ ! -f "${PADAPT_CKPT}" ]]; then
    echo "Missing padapt ckpt: ${PADAPT_CKPT}"
    exit 1
  fi
  if [[ ! -f "${PUREBC_CKPT}" ]]; then
    echo "Missing purebc ckpt: ${PUREBC_CKPT}"
    exit 1
  fi
fi

IFS=',' read -r -a RAW_SEEDS <<< "${SEEDS_CSV}"
SEEDS=()
for s in "${RAW_SEEDS[@]}"; do
  t="$(echo "${s}" | xargs)"
  if [[ -n "${t}" ]]; then
    SEEDS+=("${t}")
  fi
done
if [[ ${#SEEDS[@]} -eq 0 ]]; then
  echo "No valid seeds parsed from: ${SEEDS_CSV}"
  exit 1
fi

condition_extra_args() {
  local cond="$1"
  case "${cond}" in
    nominal)
      echo ""
      ;;
    light_v2)
      echo "task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2"
      ;;
    hard)
      echo "task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3"
      ;;
    *)
      echo "Unknown condition: ${cond}" >&2
      exit 1
      ;;
  esac
}

run_teacher_eval() {
  local seed="$1"
  local cond="$2"
  local log_file="$3"
  local extra
  extra="$(condition_extra_args "${cond}")"

  python train.py \
    task=XHandHoraScrewDriver \
    headless=True \
    seed="${seed}" \
    sim_device=cuda:"${GPU_ID}" \
    rl_device=cuda:"${GPU_ID}" \
    graphics_device_id=7 \
    test=True \
    +test_num_steps="${STEPS}" \
    train.algo=PPO \
    wandb_activate=False \
    train.ppo.output_name="XHandHoraScrewDriver_eval_robustness/plansv2_m1_teacher_${cond}_s${seed}" \
    "checkpoint=${TEACHER_CKPT}" \
    task.env.numEnvs=48 \
    task.env.reset_dist_threshold=0.15 \
    task.env.randomization.randomizePDGains=False \
    task.env.randomization.action_noise_e_scale=0.0 \
    task.env.randomization.action_noise_t_scale=0.0 \
    task.env.randomization.obs_noise_e_scale=0.0 \
    task.env.randomization.obs_noise_t_scale=0.0 \
    task.env.randomization.noisy_rpy_scale=0.0 \
    task.env.randomization.noisy_pos_scale=0.0 \
    task.env.forceScale=0.0 \
    task.env.randomForceProbScalar=0.0 \
    ${extra} \
    > "${log_file}" 2>&1
}

run_student_eval() {
  local algo="$1"
  local ckpt="$2"
  local seed="$3"
  local cond="$4"
  local log_file="$5"
  local extra
  extra="$(condition_extra_args "${cond}")"
  local cache="plansv2_m1_${algo}_${cond}_s${seed}"

  "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
    "${GPU_ID}" "${seed}" "${algo}" "${ckpt}" "${STEPS}" "${cache}" \
    ${extra} > "${log_file}" 2>&1
}

extract_eval_summary() {
  local log_file="$1"
  local summary
  if command -v rg >/dev/null 2>&1; then
    summary="$(rg "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
  else
    summary="$(grep "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
  fi
  if [[ -z "${summary}" ]]; then
    echo "Missing EvalSummary in ${log_file}" >&2
    exit 1
  fi
  local reward done
  reward="$(echo "${summary}" | sed -E 's/.*avg_reward=([-0-9.eE+]+).*/\1/')"
  done="$(echo "${summary}" | sed -E 's/.*avg_done_rate=([-0-9.eE+]+).*/\1/')"
  echo "${reward},${done}"
}

run_pack() {
  local algo_label="$1"
  local algo="$2"
  local ckpt="$3"

  for cond in nominal light_v2 hard; do
    for seed in "${SEEDS[@]}"; do
      local log_file="${OUT_DIR}/${algo_label}_${cond}_s${seed}.log"
      echo "[plansv2_m1] run ${algo_label} cond=${cond} seed=${seed}"
      if [[ "${algo_label}" == "teacher_ppo" ]]; then
        run_teacher_eval "${seed}" "${cond}" "${log_file}"
      else
        run_student_eval "${algo}" "${ckpt}" "${seed}" "${cond}" "${log_file}"
      fi
      local parsed
      parsed="$(extract_eval_summary "${log_file}")"
      echo "[plansv2_m1] done ${algo_label} cond=${cond} seed=${seed} => ${parsed}"
    done
  done
}

check_pack_logs() {
  local algo_label="$1"
  for cond in nominal light_v2 hard; do
    for seed in "${SEEDS[@]}"; do
      local log_file="${OUT_DIR}/${algo_label}_${cond}_s${seed}.log"
      if [[ ! -f "${log_file}" ]]; then
        echo "Missing expected log (from-logs mode): ${log_file}" >&2
        exit 1
      fi
      local parsed
      parsed="$(extract_eval_summary "${log_file}")"
      echo "[plansv2_m1] reuse ${algo_label} cond=${cond} seed=${seed} => ${parsed}"
    done
  done
}

if [[ "${FROM_LOGS_ONLY}" == "1" ]]; then
  echo "[plansv2_m1] from-logs mode enabled; skip eval runs."
  check_pack_logs "teacher_ppo"
  check_pack_logs "padapt"
  check_pack_logs "purebc"
else
  run_pack "teacher_ppo" "PPO" "${TEACHER_CKPT}"
  run_pack "padapt" "ProprioAdapt" "${PADAPT_CKPT}"
  run_pack "purebc" "PureBC" "${PUREBC_CKPT}"
fi

SUMMARY_MD="docs/plansv2_m1_baseline_pack.md"
SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
PLANSV2_M1_SEEDS="${SEEDS_JOINED}" PLANSV2_M1_STEPS="${STEPS}" python - <<'PY'
from pathlib import Path
import subprocess
import re
import statistics as st
import os

root = Path("outputs/robustness_eval/plansv2_m1")
summary_md = Path("docs/plansv2_m1_baseline_pack.md")

algos = ["teacher_ppo", "padapt", "purebc"]
conds = ["nominal", "light_v2", "hard"]
seeds = [x.strip() for x in os.environ["PLANSV2_M1_SEEDS"].split(",") if x.strip()]
steps = int(os.environ["PLANSV2_M1_STEPS"])

def parse_eval(log_path: Path):
    txt = log_path.read_text(errors="ignore")
    m = re.findall(r"EvalSummary steps=\d+ avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)", txt)
    if not m:
        raise RuntimeError(f"Missing EvalSummary in {log_path}")
    r, d = m[-1]
    return float(r), float(d)

rows = []
for algo in algos:
    for cond in conds:
        rewards = []
        dones = []
        for seed in seeds:
            log_file = root / f"{algo}_{cond}_s{seed}.log"
            r, d = parse_eval(log_file)
            rewards.append(r)
            dones.append(d)
        rows.append(
            {
                "algo": algo,
                "condition": cond,
                "reward_mean": st.mean(rewards),
                "reward_std": st.pstdev(rewards),
                "done_mean": st.mean(dones),
                "done_std": st.pstdev(dones),
            }
        )

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = []
lines.append("# PLANS_v2 M1 Baseline Pack (Multiseed)")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append(f"- run_id: `plansv2_m1_baseline_pack_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- config_snapshot:")
lines.append("  - `outputs/XHandHoraScrewDriver_teacher/run_a/config_031916_f5f9edb.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/config_032011_8bf90ec.yaml`")
lines.append("- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append("- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append(f"- seeds: `{','.join(seeds)}`")
lines.append("- eval_episodes_per_run: `N/A (fixed-step protocol)`")
lines.append(f"- eval_env_steps_per_run: `{steps}`")
lines.append(f"- protocol: `nominal + light_v2 + hard`")
lines.append("- primary_metrics:")
lines.append("  - `avg_reward` (main)")
lines.append("  - `avg_done_rate` (secondary)")
lines.append("- dispersion_metric: `std across seeds`")
lines.append(f"- artifacts_root: `outputs/robustness_eval/plansv2_m1/`")
lines.append(f"- table_source: `docs/plansv2_m1_baseline_pack.md`")
lines.append("- artifact_log_paths: `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`")
lines.append("- artifact_checkpoint_paths:")
lines.append("  - `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`")
lines.append("")
lines.append("## Aggregated Results")
lines.append("")
lines.append("| Algorithm | Condition | Reward Mean | Reward Std | Done Mean | Done Std |")
lines.append("|---|---|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['algo']} | {r['condition']} | "
        f"{r['reward_mean']:.6f} | {r['reward_std']:.6f} | "
        f"{r['done_mean']:.6f} | {r['done_std']:.6f} |"
    )
lines.append("")
lines.append("## Artifact Pointers")
lines.append("")
lines.append("- logs: `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`")
lines.append("- teacher_ckpt: `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`")
lines.append("- padapt_ckpt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`")
lines.append("- purebc_ckpt: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
lines.append("- support: M1 baseline pack now has multiseed, unified-protocol evidence for `teacher_ppo`, `padapt`, and `purebc`.")

summary_md.write_text("\n".join(lines) + "\n")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_m1] completed. Summary: ${SUMMARY_MD}"
