#!/usr/bin/env bash
set -euo pipefail

# PLANS_v2 M4 residual fallback gate (minimal executable pack):
# - explicit residual target path: latent residual around base student latent
# - residual magnitude and action-correction magnitude summary
# - nominal sanity eval (multiseed)
#
# Usage:
#   scripts/eval_plansv2_m4_residual_gate_nominal.sh [GPU_ID] [STEPS] [SEEDS_CSV] [CKPT] [TAG]
#
# Example:
#   scripts/eval_plansv2_m4_residual_gate_nominal.sh 0 256 42,43,44 \
#     outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt

GPU_ID=${1:-0}
STEPS=${2:-256}
SEEDS_CSV=${3:-42,43,44}
CKPT=${4:-outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt}
TAG=${5:-}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

source "${SCRIPT_DIR}/_ensure_isaacgym_env.sh"
ensure_isaacgym_env

if [[ ! -f "${CKPT}" ]]; then
  echo "Missing residual-base ckpt: ${CKPT}"
  exit 1
fi

if [[ -n "${TAG}" ]]; then
  OUT_DIR="outputs/robustness_eval/plansv2_m4_residual_gate_nominal_${TAG}"
  SUMMARY_MD="docs/plansv2_m4_residual_gate_nominal_${TAG}.md"
else
  OUT_DIR="outputs/robustness_eval/plansv2_m4_residual_gate_nominal"
  SUMMARY_MD="docs/plansv2_m4_residual_gate_nominal.md"
fi
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

extract_required_summaries() {
  local log_file="$1"
  local eval_line recon_line residual_line
  if command -v rg >/dev/null 2>&1; then
    eval_line="$(rg "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    recon_line="$(rg "EvalReconSummary steps=" "${log_file}" | tail -n 1 || true)"
    residual_line="$(rg "EvalResidualSummary steps=" "${log_file}" | tail -n 1 || true)"
  else
    eval_line="$(grep "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    recon_line="$(grep "EvalReconSummary steps=" "${log_file}" | tail -n 1 || true)"
    residual_line="$(grep "EvalResidualSummary steps=" "${log_file}" | tail -n 1 || true)"
  fi
  if [[ -z "${eval_line}" ]]; then
    echo "Missing EvalSummary in ${log_file}" >&2
    exit 1
  fi
  if [[ -z "${recon_line}" ]]; then
    echo "Missing EvalReconSummary in ${log_file}" >&2
    exit 1
  fi
  if [[ -z "${residual_line}" ]]; then
    echo "Missing EvalResidualSummary in ${log_file}" >&2
    exit 1
  fi
}

for seed in "${SEEDS[@]}"; do
  log_file="${OUT_DIR}/residual_nominal_s${seed}.log"
  cache="plansv2_m4_residual_nominal_s${seed}"
  echo "[plansv2_m4] run cond=nominal seed=${seed}"
  "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
    "${GPU_ID}" "${seed}" DiffusionLatentStudent "${CKPT}" "${STEPS}" "${cache}" \
    +train.ppo.diffusion_residual_base=True \
    +train.ppo.diffusion_eval_decode_only=False \
    +train.ppo.diffusion_eval_report_recon=True > "${log_file}" 2>&1
  extract_required_summaries "${log_file}"
  echo "[plansv2_m4] done cond=nominal seed=${seed}"
done

SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
PLANSV2_M4_SEEDS="${SEEDS_JOINED}" PLANSV2_M4_STEPS="${STEPS}" PLANSV2_M4_CKPT="${CKPT}" PLANSV2_M4_OUT_DIR="${OUT_DIR}" PLANSV2_M4_SUMMARY_MD="${SUMMARY_MD}" PLANSV2_M4_TAG="${TAG}" python - <<'PY'
from pathlib import Path
import os
import re
import statistics as st
import subprocess

root = Path(os.environ["PLANSV2_M4_OUT_DIR"])
summary_md = Path(os.environ["PLANSV2_M4_SUMMARY_MD"])
tag = os.environ.get("PLANSV2_M4_TAG", "")
seeds = [x.strip() for x in os.environ["PLANSV2_M4_SEEDS"].split(",") if x.strip()]
steps = int(os.environ["PLANSV2_M4_STEPS"])
ckpt = os.environ["PLANSV2_M4_CKPT"]

def parse_residual_summary_kv(path: Path):
    txt = path.read_text(errors="ignore")
    lines = re.findall(r"EvalResidualSummary[^\n]*", txt)
    if not lines:
        raise RuntimeError(f"Missing EvalResidualSummary in {path}")
    line = lines[-1]
    out = {}
    for key, val in re.findall(r"([a-z0-9_]+)=([A-Za-z_0-9.+-]+)", line):
        out[key] = val
    return out

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
    recon_mode, latent_mse, latent_l1, action_mse = m_recon[-1]
    residual_kv = parse_residual_summary_kv(path)
    residual_mode = residual_kv.get("mode", "")
    residual_abs = float(residual_kv["residual_abs_mean"])
    residual_l2 = float(residual_kv["residual_l2_mean"])
    residual_ratio = float(residual_kv["residual_to_target_ratio"])
    corr_abs = float(residual_kv["action_correction_abs_mean"])
    corr_l2 = float(residual_kv["action_correction_l2_mean"])
    base_action_mse = float(residual_kv["base_action_mse_to_teacher"])
    pred_residual_abs = (
        float(residual_kv["pred_residual_abs_mean"])
        if "pred_residual_abs_mean" in residual_kv
        else float("nan")
    )
    pred_to_target_ratio = (
        float(residual_kv["pred_to_target_ratio"])
        if "pred_to_target_ratio" in residual_kv
        else float("nan")
    )
    return {
        "reward": reward,
        "done": done,
        "recon_mode": recon_mode,
        "residual_mode": residual_mode,
        "latent_mse": float(latent_mse),
        "latent_l1": float(latent_l1),
        "action_mse": float(action_mse),
        "residual_abs": float(residual_abs),
        "residual_l2": float(residual_l2),
        "residual_ratio": float(residual_ratio),
        "corr_abs": float(corr_abs),
        "corr_l2": float(corr_l2),
        "base_action_mse": float(base_action_mse),
        "pred_residual_abs": float(pred_residual_abs),
        "pred_to_target_ratio": float(pred_to_target_ratio),
    }

rows = []
for seed in seeds:
    log_path = root / f"residual_nominal_s{seed}.log"
    x = parse_log(log_path)
    if x["recon_mode"] != "diffusion" or x["residual_mode"] != "diffusion":
        raise RuntimeError(
            f"Unexpected eval mode in {log_path}: recon={x['recon_mode']} residual={x['residual_mode']}"
        )
    rows.append({"seed": seed, **x})

def m(key):
    return st.mean([r[key] for r in rows])

def s(key):
    return st.pstdev([r[key] for r in rows])

reward_mean = m("reward")
reward_std = s("reward")
done_mean = m("done")
done_std = s("done")
latent_mse_mean = m("latent_mse")
latent_l1_mean = m("latent_l1")
action_mse_mean = m("action_mse")
residual_abs_mean = m("residual_abs")
residual_l2_mean = m("residual_l2")
residual_ratio_mean = m("residual_ratio")
corr_abs_mean = m("corr_abs")
corr_l2_mean = m("corr_l2")
base_action_mse_mean = m("base_action_mse")
pred_residual_abs_mean = m("pred_residual_abs")
pred_to_target_ratio_mean = m("pred_to_target_ratio")

sanity_reward_positive = reward_mean > 0.0
residual_nonzero = residual_ratio_mean > 0.02 and corr_abs_mean > 0.01
if pred_to_target_ratio_mean == pred_to_target_ratio_mean:
    residual_not_explosive = pred_to_target_ratio_mean < 1.2 and corr_abs_mean < 0.6
else:
    residual_not_explosive = corr_abs_mean < 0.6
g3_local_ready = sanity_reward_positive and residual_nonzero and residual_not_explosive
g3_local_decision = "PASS" if g3_local_ready else "FAIL"

ckpt_path = Path(ckpt)
run_dir = ckpt_path.parent.parent
config_candidates = sorted(run_dir.glob("config_*.yaml"))
config_snapshot = str(config_candidates[0]) if config_candidates else "N/A"
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = []
lines.append("# PLANS_v2 M4 Residual Fallback Gate (Nominal Minimal Pack)")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
tag_token = f"_{tag}" if tag else ""
lines.append(f"- run_id: `plansv2_m4_residual_gate_nominal{tag_token}_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append(f"- representative_ckpt: `{ckpt}`")
lines.append(f"- config_snapshot: `{config_snapshot}`")
lines.append(f"- seeds: `{','.join(seeds)}`")
lines.append(f"- eval_steps_per_run: `{steps}`")
lines.append("- protocol: `nominal`")
lines.append(f"- artifacts_root: `{root}`")
lines.append(f"- table_source: `{summary_md}`")
lines.append("")
lines.append("## Residual Target And Scaling (Code Path)")
lines.append("")
lines.append("- residual_target_definition: `target_x0 = (e_gt - base_latent) * diffusion_residual_target_scale`, where `base_latent = tanh(adapt_tconv(proprio_hist))`.")
lines.append("- decode_rule: `pred_latent = tanh((x0_pred / diffusion_residual_target_scale) + base_latent)` when `diffusion_residual_base=True`.")
lines.append("- action_output_scaling: actor output is clamped to `[-1, 1]` before env step.")
lines.append("")
lines.append("## Aggregated Nominal Results (Multiseed)")
lines.append("")
lines.append("| Metric | Mean | Std |")
lines.append("|---|---:|---:|")
lines.append(f"| avg_reward | {reward_mean:.6f} | {reward_std:.6f} |")
lines.append(f"| avg_done_rate | {done_mean:.6f} | {done_std:.6f} |")
lines.append(f"| latent_mse | {latent_mse_mean:.6f} | {s('latent_mse'):.6f} |")
lines.append(f"| latent_l1 | {latent_l1_mean:.6f} | {s('latent_l1'):.6f} |")
lines.append(f"| action_mse_to_teacher | {action_mse_mean:.6f} | {s('action_mse'):.6f} |")
lines.append(f"| residual_abs_mean | {residual_abs_mean:.6f} | {s('residual_abs'):.6f} |")
lines.append(f"| residual_l2_mean | {residual_l2_mean:.6f} | {s('residual_l2'):.6f} |")
lines.append(f"| residual_to_target_ratio | {residual_ratio_mean:.6f} | {s('residual_ratio'):.6f} |")
if pred_residual_abs_mean == pred_residual_abs_mean:
    lines.append(f"| pred_residual_abs_mean | {pred_residual_abs_mean:.6f} | {s('pred_residual_abs'):.6f} |")
if pred_to_target_ratio_mean == pred_to_target_ratio_mean:
    lines.append(f"| pred_to_target_ratio | {pred_to_target_ratio_mean:.6f} | {s('pred_to_target_ratio'):.6f} |")
lines.append(f"| action_correction_abs_mean | {corr_abs_mean:.6f} | {s('corr_abs'):.6f} |")
lines.append(f"| action_correction_l2_mean | {corr_l2_mean:.6f} | {s('corr_l2'):.6f} |")
lines.append(f"| base_action_mse_to_teacher | {base_action_mse_mean:.6f} | {s('base_action_mse'):.6f} |")
lines.append("")
lines.append("## G3 Readiness Check (Local Execution Heuristic)")
lines.append("")
lines.append(f"- sanity_reward_positive: `{sanity_reward_positive}`")
lines.append("- criteria: nominal aggregated reward is positive.")
lines.append(f"- residual_nonzero_signal: `{residual_nonzero}`")
lines.append("- criteria: residual_to_target_ratio > 0.02 and action_correction_abs_mean > 0.01.")
lines.append(f"- residual_not_explosive: `{residual_not_explosive}`")
if pred_to_target_ratio_mean == pred_to_target_ratio_mean:
    lines.append("- criteria: pred_to_target_ratio < 1.2 and action_correction_abs_mean < 0.6.")
else:
    lines.append("- criteria: action_correction_abs_mean < 0.6.")
lines.append(f"- local_g3_readiness_decision: `{g3_local_decision}`")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
if g3_local_ready:
    lines.append("- support: residual fallback branch is runnable with nontrivial correction signal under nominal protocol.")
else:
    lines.append("- not support: residual fallback branch is not yet stable/meaningful under nominal protocol.")

summary_md.write_text("\n".join(lines) + "\n")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_m4] completed. Summary: ${SUMMARY_MD}"
