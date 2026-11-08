#!/usr/bin/env bash
set -euo pipefail

# PLANS_v2 M2 gate hardening:
# Evaluate latent representative under two inference modes:
#   - diffusion (default)
#   - decode_only (bypass diffusion sampling, use frozen adapt latent decode path)
# Unified protocol: nominal / light_v2 / hard, multiseed.
#
# Usage:
#   scripts/eval_plansv2_m2_gap_gate.sh [GPU_ID] [STEPS] [SEEDS_CSV] [CKPT]
#
# Example:
#   scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44 \
#     outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt

GPU_ID=${1:-0}
STEPS=${2:-256}
SEEDS_CSV=${3:-42,43,44}
CKPT=${4:-outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt}
FROM_LOGS_ONLY=${PLANSV2_FROM_LOGS_ONLY:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
if [[ "${FROM_LOGS_ONLY}" != "1" ]]; then
  ensure_isaacgym_env
fi

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing latent ckpt: ${CKPT}"
  exit 1
fi

OUT_DIR="outputs/robustness_eval/plansv2_m2_gap_gate"
mkdir -p "${OUT_DIR}"

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

mode_extra_args() {
  local mode="$1"
  case "${mode}" in
    diffusion)
      echo "+train.ppo.diffusion_eval_decode_only=False +train.ppo.diffusion_eval_report_recon=True"
      ;;
    decode_only)
      echo "+train.ppo.diffusion_eval_decode_only=True +train.ppo.diffusion_eval_report_recon=True"
      ;;
    *)
      echo "Unknown mode: ${mode}" >&2
      exit 1
      ;;
  esac
}

extract_required_summaries() {
  local log_file="$1"
  local eval_line recon_line
  if command -v rg >/dev/null 2>&1; then
    eval_line="$(rg "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    recon_line="$(rg "EvalReconSummary steps=" "${log_file}" | tail -n 1 || true)"
  else
    eval_line="$(grep "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    recon_line="$(grep "EvalReconSummary steps=" "${log_file}" | tail -n 1 || true)"
  fi
  if [[ -z "${eval_line}" ]]; then
    echo "Missing EvalSummary in ${log_file}" >&2
    exit 1
  fi
  if [[ -z "${recon_line}" ]]; then
    echo "Missing EvalReconSummary in ${log_file}" >&2
    exit 1
  fi
}

run_one() {
  local mode="$1"
  local cond="$2"
  local seed="$3"
  local log_file="${OUT_DIR}/${mode}_${cond}_s${seed}.log"
  local cache="plansv2_m2_${mode}_${cond}_s${seed}"
  local cond_extra mode_extra
  cond_extra="$(condition_extra_args "${cond}")"
  mode_extra="$(mode_extra_args "${mode}")"

  echo "[plansv2_m2] run mode=${mode} cond=${cond} seed=${seed}"
  "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
    "${GPU_ID}" "${seed}" DiffusionLatentStudent "${CKPT}" "${STEPS}" "${cache}" \
    ${cond_extra} ${mode_extra} > "${log_file}" 2>&1
  extract_required_summaries "${log_file}"
  echo "[plansv2_m2] done mode=${mode} cond=${cond} seed=${seed}"
}

for mode in diffusion decode_only; do
  for cond in nominal light_v2 hard; do
    for seed in "${SEEDS[@]}"; do
      if [[ "${FROM_LOGS_ONLY}" == "1" ]]; then
        log_file="${OUT_DIR}/${mode}_${cond}_s${seed}.log"
        if [[ ! -f "${log_file}" ]]; then
          echo "Missing expected log (from-logs mode): ${log_file}" >&2
          exit 1
        fi
        extract_required_summaries "${log_file}"
        echo "[plansv2_m2] reuse mode=${mode} cond=${cond} seed=${seed}"
      else
        run_one "${mode}" "${cond}" "${seed}"
      fi
    done
  done
done

SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
PLANSV2_M2_SEEDS="${SEEDS_JOINED}" PLANSV2_M2_STEPS="${STEPS}" PLANSV2_M2_CKPT="${CKPT}" python - <<'PY'
from pathlib import Path
import os
import re
import statistics as st
import subprocess

root = Path("outputs/robustness_eval/plansv2_m2_gap_gate")
summary_md = Path("docs/plansv2_m2_gap_gate.md")
modes = ["diffusion", "decode_only"]
conds = ["nominal", "light_v2", "hard"]
seeds = [x.strip() for x in os.environ["PLANSV2_M2_SEEDS"].split(",") if x.strip()]
steps = int(os.environ["PLANSV2_M2_STEPS"])
ckpt = os.environ["PLANSV2_M2_CKPT"]

def parse_log(path: Path):
    txt = path.read_text(errors="ignore")
    m_eval = re.findall(
        r"EvalSummary steps=\d+ avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)",
        txt,
    )
    m_recon = re.findall(
        r"EvalReconSummary steps=\d+ mode=([a-z_]+) latent_mse=([-0-9.eE+]+) latent_l1=([-0-9.eE+]+) action_mse_to_teacher=([-0-9.eE+]+)",
        txt,
    )
    if not m_eval:
        raise RuntimeError(f"Missing EvalSummary in {path}")
    if not m_recon:
        raise RuntimeError(f"Missing EvalReconSummary in {path}")
    reward, done = map(float, m_eval[-1])
    mode, latent_mse, latent_l1, action_mse = m_recon[-1]
    return {
        "reward": reward,
        "done": done,
        "mode_from_log": mode,
        "latent_mse": float(latent_mse),
        "latent_l1": float(latent_l1),
        "action_mse": float(action_mse),
    }

rows = []
for mode in modes:
    for cond in conds:
        rewards = []
        dones = []
        latent_mses = []
        latent_l1s = []
        action_mses = []
        for seed in seeds:
            log_path = root / f"{mode}_{cond}_s{seed}.log"
            out = parse_log(log_path)
            if out["mode_from_log"] != mode:
                raise RuntimeError(
                    f"Mode mismatch in {log_path}: expected {mode}, got {out['mode_from_log']}"
                )
            rewards.append(out["reward"])
            dones.append(out["done"])
            latent_mses.append(out["latent_mse"])
            latent_l1s.append(out["latent_l1"])
            action_mses.append(out["action_mse"])
        rows.append(
            {
                "mode": mode,
                "condition": cond,
                "reward_mean": st.mean(rewards),
                "reward_std": st.pstdev(rewards),
                "done_mean": st.mean(dones),
                "done_std": st.pstdev(dones),
                "latent_mse_mean": st.mean(latent_mses),
                "latent_mse_std": st.pstdev(latent_mses),
                "latent_l1_mean": st.mean(latent_l1s),
                "latent_l1_std": st.pstdev(latent_l1s),
                "action_mse_mean": st.mean(action_mses),
                "action_mse_std": st.pstdev(action_mses),
            }
        )

def find_row(mode, cond):
    for r in rows:
        if r["mode"] == mode and r["condition"] == cond:
            return r
    raise KeyError((mode, cond))

# Local gate decision heuristic for documentation:
# - recon_reported: all mode/condition rows have finite latent_mse/action_mse.
# - decode_runnable: all decode_only logs produced required summaries.
# - decode_stable: decode_only reward_mean > 0 under nominal/light_v2/hard.
recon_reported = all(
    (r["latent_mse_mean"] >= 0.0 and r["action_mse_mean"] >= 0.0) for r in rows
)
decode_stable = all(find_row("decode_only", c)["reward_mean"] > 0.0 for c in conds)
g2_local_pass = recon_reported and decode_stable
g2_local_decision = "PASS" if g2_local_pass else "FAIL"

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = []
lines.append("# PLANS_v2 M2 Latent Gap-Closing Gate Pack")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append("- run_id: `plansv2_m2_gap_gate_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- config_snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/config_032315_8bf90ec.yaml`")
lines.append("- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append("- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append(f"- representative_ckpt: `{ckpt}`")
lines.append(f"- seeds: `{','.join(seeds)}`")
lines.append("- eval_episodes_per_run: `N/A (fixed-step protocol)`")
lines.append(f"- eval_env_steps_per_run: `{steps}`")
lines.append("- protocol: `nominal + light_v2 + hard`")
lines.append("- modes: `diffusion`, `decode_only`")
lines.append("- primary_metrics:")
lines.append("  - `avg_reward` (main)")
lines.append("  - `avg_done_rate`, `latent_mse`, `latent_l1`, `action_mse_to_teacher`")
lines.append("- dispersion_metric: `std across seeds`")
lines.append("- artifacts_root: `outputs/robustness_eval/plansv2_m2_gap_gate/`")
lines.append("- table_source: `docs/plansv2_m2_gap_gate.md`")
lines.append("- artifact_log_paths: `outputs/robustness_eval/plansv2_m2_gap_gate/{mode}_{condition}_s{seed}.log`")
lines.append("- artifact_checkpoint_paths:")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("")
lines.append("## Aggregated Results")
lines.append("")
lines.append("| Mode | Condition | Reward Mean | Reward Std | Done Mean | Done Std | Latent MSE Mean | Latent L1 Mean | Action MSE Mean |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    lines.append(
        f"| {r['mode']} | {r['condition']} | "
        f"{r['reward_mean']:.6f} | {r['reward_std']:.6f} | "
        f"{r['done_mean']:.6f} | {r['done_std']:.6f} | "
        f"{r['latent_mse_mean']:.6f} | {r['latent_l1_mean']:.6f} | "
        f"{r['action_mse_mean']:.6f} |"
    )
lines.append("")
lines.append("## G2 Gate Decision (Local Execution Heuristic)")
lines.append("")
lines.append(f"- reconstruction_reported: `{recon_reported}`")
lines.append("- criteria: `EvalReconSummary` exists for every run, with finite latent/action errors.")
lines.append(f"- decode_only_stability: `{decode_stable}`")
lines.append("- criteria: decode-only aggregated reward is positive under nominal/light_v2/hard.")
lines.append(f"- local_G2_decision: `{g2_local_decision}`")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
if g2_local_pass:
    lines.append("- support: M2 gate evidence is sufficient locally (`PASS`) under the current representative ckpt and protocol.")
else:
    lines.append("- not support: M2 gate remains open (`FAIL`) under the current representative ckpt and protocol.")

summary_md.write_text("\n".join(lines) + "\n")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_m2] completed. Summary: docs/plansv2_m2_gap_gate.md"
