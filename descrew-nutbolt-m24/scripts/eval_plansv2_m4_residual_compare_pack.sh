#!/usr/bin/env bash
set -euo pipefail

# PLANS_v2 M4 residual robustness extension:
# Compare under robust conditions (light_v2 + hard), multiseed:
#   - residual_unscaled (diffusion_residual_target_scale=1.0)
#   - residual_scale05 (diffusion_residual_target_scale=0.5)
#   - padapt baseline
#
# Usage:
#   scripts/eval_plansv2_m4_residual_compare_pack.sh [GPU_ID] [STEPS] [SEEDS_CSV]
#
# Example:
#   scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44

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

RESIDUAL_UNSCALED_CKPT="outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt"
RESIDUAL_SCALE05_CKPT="outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt"
PADAPT_CKPT="outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt"

for ckpt in "${RESIDUAL_UNSCALED_CKPT}" "${RESIDUAL_SCALE05_CKPT}" "${PADAPT_CKPT}"; do
  if [[ ! -f "${ckpt}" ]]; then
    echo "Missing ckpt: ${ckpt}" >&2
    exit 1
  fi
done

OUT_DIR="outputs/robustness_eval/plansv2_m4_residual_compare_pack"
SUMMARY_MD="docs/plansv2_m4_residual_compare_pack.md"
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
  echo "No valid seeds parsed from: ${SEEDS_CSV}" >&2
  exit 1
fi

condition_extra_args() {
  local cond="$1"
  case "${cond}" in
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

extract_required_summaries() {
  local variant="$1"
  local log_file="$2"
  local eval_line residual_line
  if command -v rg >/dev/null 2>&1; then
    eval_line="$(rg "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    residual_line="$(rg "EvalResidualSummary steps=" "${log_file}" | tail -n 1 || true)"
  else
    eval_line="$(grep "EvalSummary steps=" "${log_file}" | tail -n 1 || true)"
    residual_line="$(grep "EvalResidualSummary steps=" "${log_file}" | tail -n 1 || true)"
  fi
  if [[ -z "${eval_line}" ]]; then
    echo "Missing EvalSummary in ${log_file}" >&2
    exit 1
  fi
  if [[ "${variant}" != "padapt" && -z "${residual_line}" ]]; then
    echo "Missing EvalResidualSummary in ${log_file}" >&2
    exit 1
  fi
}

run_one() {
  local variant="$1"
  local cond="$2"
  local seed="$3"
  local cond_extra cache log_file
  cond_extra="$(condition_extra_args "${cond}")"
  cache="plansv2_m4_compare_${variant}_${cond}_s${seed}"
  log_file="${OUT_DIR}/${variant}_${cond}_s${seed}.log"

  echo "[plansv2_m4_compare] run variant=${variant} cond=${cond} seed=${seed}"
  if [[ "${variant}" == "padapt" ]]; then
    "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
      "${GPU_ID}" "${seed}" ProprioAdapt "${PADAPT_CKPT}" "${STEPS}" "${cache}" \
      ${cond_extra} > "${log_file}" 2>&1
  elif [[ "${variant}" == "residual_unscaled" ]]; then
    "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
      "${GPU_ID}" "${seed}" DiffusionLatentStudent "${RESIDUAL_UNSCALED_CKPT}" "${STEPS}" "${cache}" \
      ${cond_extra} \
      +train.ppo.diffusion_residual_base=True \
      +train.ppo.diffusion_residual_target_scale=1.0 \
      +train.ppo.diffusion_eval_decode_only=False \
      +train.ppo.diffusion_eval_report_recon=True > "${log_file}" 2>&1
  elif [[ "${variant}" == "residual_scale05" ]]; then
    "${SCRIPT_DIR}/eval_screwdriver_student_robustness.sh" \
      "${GPU_ID}" "${seed}" DiffusionLatentStudent "${RESIDUAL_SCALE05_CKPT}" "${STEPS}" "${cache}" \
      ${cond_extra} \
      +train.ppo.diffusion_residual_base=True \
      +train.ppo.diffusion_residual_target_scale=0.5 \
      +train.ppo.diffusion_eval_decode_only=False \
      +train.ppo.diffusion_eval_report_recon=True > "${log_file}" 2>&1
  else
    echo "Unknown variant: ${variant}" >&2
    exit 1
  fi
  extract_required_summaries "${variant}" "${log_file}"
  echo "[plansv2_m4_compare] done variant=${variant} cond=${cond} seed=${seed}"
}

for cond in light_v2 hard; do
  for seed in "${SEEDS[@]}"; do
    if [[ "${FROM_LOGS_ONLY}" == "1" ]]; then
      for variant in residual_unscaled residual_scale05 padapt; do
        log_file="${OUT_DIR}/${variant}_${cond}_s${seed}.log"
        if [[ ! -f "${log_file}" ]]; then
          echo "Missing expected log (from-logs mode): ${log_file}" >&2
          exit 1
        fi
        extract_required_summaries "${variant}" "${log_file}"
        echo "[plansv2_m4_compare] reuse variant=${variant} cond=${cond} seed=${seed}"
      done
    else
      run_one residual_unscaled "${cond}" "${seed}"
      run_one residual_scale05 "${cond}" "${seed}"
      run_one padapt "${cond}" "${seed}"
    fi
  done
done

SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
PLANSV2_M4CMP_OUT_DIR="${OUT_DIR}" PLANSV2_M4CMP_SUMMARY_MD="${SUMMARY_MD}" PLANSV2_M4CMP_SEEDS="${SEEDS_JOINED}" PLANSV2_M4CMP_STEPS="${STEPS}" python - <<'PY'
from pathlib import Path
import os
import re
import statistics as st
import subprocess

root = Path(os.environ["PLANSV2_M4CMP_OUT_DIR"])
summary_md = Path(os.environ["PLANSV2_M4CMP_SUMMARY_MD"])
seeds = [x.strip() for x in os.environ["PLANSV2_M4CMP_SEEDS"].split(",") if x.strip()]
steps = int(os.environ["PLANSV2_M4CMP_STEPS"])
variants = ["residual_unscaled", "residual_scale05", "padapt"]
conds = ["light_v2", "hard"]

def parse_eval(path: Path):
    txt = path.read_text(errors="ignore")
    m = re.findall(
        r"EvalSummary steps=\d+ avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)",
        txt,
    )
    if not m:
        raise RuntimeError(f"Missing EvalSummary in {path}")
    reward, done = m[-1]
    return float(reward), float(done)

def parse_residual(path: Path):
    txt = path.read_text(errors="ignore")
    lines = re.findall(r"EvalResidualSummary[^\n]*", txt)
    if not lines:
        raise RuntimeError(f"Missing EvalResidualSummary in {path}")
    kv = {}
    for key, val in re.findall(r"([a-z0-9_]+)=([A-Za-z_0-9.+-]+)", lines[-1]):
        kv[key] = val
    return {
        "action_correction_abs_mean": float(kv["action_correction_abs_mean"]),
        "pred_to_target_ratio": float(kv["pred_to_target_ratio"]) if "pred_to_target_ratio" in kv else float("nan"),
    }

rows = []
for variant in variants:
    for cond in conds:
        rewards = []
        dones = []
        corr_abs = []
        pred_ratio = []
        for seed in seeds:
            log_path = root / f"{variant}_{cond}_s{seed}.log"
            r, d = parse_eval(log_path)
            rewards.append(r)
            dones.append(d)
            if variant != "padapt":
                rr = parse_residual(log_path)
                corr_abs.append(rr["action_correction_abs_mean"])
                pred_ratio.append(rr["pred_to_target_ratio"])
        row = {
            "variant": variant,
            "condition": cond,
            "reward_mean": st.mean(rewards),
            "reward_std": st.pstdev(rewards),
            "done_mean": st.mean(dones),
            "done_std": st.pstdev(dones),
        }
        if variant != "padapt":
            row["corr_abs_mean"] = st.mean(corr_abs)
            row["corr_abs_std"] = st.pstdev(corr_abs)
            row["pred_ratio_mean"] = st.mean(pred_ratio)
            row["pred_ratio_std"] = st.pstdev(pred_ratio)
        rows.append(row)

def get_row(variant, cond):
    for r in rows:
        if r["variant"] == variant and r["condition"] == cond:
            return r
    raise KeyError((variant, cond))

cmp_rows = []
for cond in conds:
    ru = get_row("residual_unscaled", cond)["reward_mean"]
    rs = get_row("residual_scale05", cond)["reward_mean"]
    pd = get_row("padapt", cond)["reward_mean"]
    cmp_rows.append(
        {
            "condition": cond,
            "scale05_minus_unscaled": rs - ru,
            "scale05_minus_padapt": rs - pd,
            "unscaled_minus_padapt": ru - pd,
        }
    )

scale05_beats_unscaled_all = all(x["scale05_minus_unscaled"] > 0.0 for x in cmp_rows)
best_residual_beats_padapt_any = any(
    max(x["scale05_minus_padapt"], x["unscaled_minus_padapt"]) > 0.0 for x in cmp_rows
)

if scale05_beats_unscaled_all and best_residual_beats_padapt_any:
    local_m4_conclusion = "support_residual_robust_edge"
elif scale05_beats_unscaled_all:
    local_m4_conclusion = "partial_improvement_below_padapt"
else:
    local_m4_conclusion = "not_support_residual_robust_edge"

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = []
lines.append("# PLANS_v2 M4 Residual Robustness Compare Pack")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append("- run_id: `plansv2_m4_residual_compare_pack_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- config_snapshot:")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/config_032311_8bf90ec.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/config_032414_aabc11a.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`")
lines.append("- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append("- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`")
lines.append("- seeds: `" + ",".join(seeds) + "`")
lines.append("- eval_episodes_per_run: `N/A (fixed-step protocol)`")
lines.append(f"- eval_env_steps_per_run: `{steps}`")
lines.append("- protocol: `light_v2 + hard`")
lines.append("- variants: `residual_unscaled`, `residual_scale05`, `padapt`")
lines.append("- primary_metrics:")
lines.append("  - `avg_reward` (main compare axis)")
lines.append("  - `avg_done_rate`, `CorrAbs Mean`, `Pred/Target Ratio Mean`")
lines.append("- dispersion_metric: `std across seeds`")
lines.append("- ckpt_residual_unscaled: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("- ckpt_residual_scale05: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("- ckpt_padapt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`")
lines.append(f"- artifacts_root: `{root}`")
lines.append(f"- table_source: `{summary_md}`")
lines.append("- artifact_log_paths: `outputs/robustness_eval/plansv2_m4_residual_compare_pack/{variant}_{condition}_s{seed}.log`")
lines.append("- artifact_checkpoint_paths:")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`")
lines.append("")
lines.append("## Unified Robustness Table")
lines.append("")
lines.append("| Variant | Condition | Reward Mean | Reward Std | Done Mean | Done Std | CorrAbs Mean | Pred/Target Ratio Mean |")
lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
for r in rows:
    corr_abs = f"{r['corr_abs_mean']:.6f}" if "corr_abs_mean" in r else "N/A"
    pred_ratio = f"{r['pred_ratio_mean']:.6f}" if "pred_ratio_mean" in r else "N/A"
    lines.append(
        f"| {r['variant']} | {r['condition']} | {r['reward_mean']:.6f} | {r['reward_std']:.6f} | "
        f"{r['done_mean']:.6f} | {r['done_std']:.6f} | {corr_abs} | {pred_ratio} |"
    )
lines.append("")
lines.append("## Reward Delta Table")
lines.append("")
lines.append("| Condition | scale05 - unscaled | scale05 - padapt | unscaled - padapt |")
lines.append("|---|---:|---:|---:|")
for x in cmp_rows:
    lines.append(
        f"| {x['condition']} | {x['scale05_minus_unscaled']:.6f} | {x['scale05_minus_padapt']:.6f} | {x['unscaled_minus_padapt']:.6f} |"
    )
lines.append("")
lines.append("## Local M4 Conclusion")
lines.append("")
lines.append(f"- scale05_beats_unscaled_all_robust: `{scale05_beats_unscaled_all}`")
lines.append(f"- best_residual_beats_padapt_any_robust: `{best_residual_beats_padapt_any}`")
lines.append(f"- local_m4_conclusion: `{local_m4_conclusion}`")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
if local_m4_conclusion == "support_residual_robust_edge":
    lines.append("- support: residual fallback shows actionable robustness edge in the current compare pack.")
elif local_m4_conclusion == "partial_improvement_below_padapt":
    lines.append("- inconclusive: scale05 improves over unscaled residual but still does not beat padapt on robust conditions.")
else:
    lines.append("- not support: residual fallback does not show robustness edge under current checkpoints and protocol.")

summary_md.write_text("\n".join(lines) + "\n")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_m4_compare] completed. Summary: ${SUMMARY_MD}"
