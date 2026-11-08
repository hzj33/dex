#!/usr/bin/env bash
set -euo pipefail

# PLANS_v2 M3 minimum credible comparison package:
# Build a unified multiseed table for:
#   - latent diffusion (from M2 diffusion mode logs)
#   - padapt (from M1 logs)
#   - purebc (from M1 logs)
# Protocol: nominal / light_v2 / hard
#
# Usage:
#   scripts/eval_plansv2_m3_min_compare.sh [SEEDS_CSV]
#
# Example:
#   scripts/eval_plansv2_m3_min_compare.sh 42,43,44

SEEDS_CSV=${1:-42,43,44}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

M1_DIR="outputs/robustness_eval/plansv2_m1"
M2_DIR="outputs/robustness_eval/plansv2_m2_gap_gate"
SUMMARY_MD="docs/plansv2_m3_min_compare.md"

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

check_log_exists() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "Missing required log: ${path}" >&2
    exit 1
  fi
}

for cond in nominal light_v2 hard; do
  for seed in "${SEEDS[@]}"; do
    check_log_exists "${M2_DIR}/diffusion_${cond}_s${seed}.log"
    check_log_exists "${M1_DIR}/padapt_${cond}_s${seed}.log"
    check_log_exists "${M1_DIR}/purebc_${cond}_s${seed}.log"
  done
done

SEEDS_JOINED="$(IFS=,; echo "${SEEDS[*]}")"
PLANSV2_M3_SEEDS="${SEEDS_JOINED}" python - <<'PY'
from pathlib import Path
import os
import re
import statistics as st
import subprocess

seeds = [x.strip() for x in os.environ["PLANSV2_M3_SEEDS"].split(",") if x.strip()]
conds = ["nominal", "light_v2", "hard"]

m1 = Path("outputs/robustness_eval/plansv2_m1")
m2 = Path("outputs/robustness_eval/plansv2_m2_gap_gate")
out = Path("docs/plansv2_m3_min_compare.md")

algo_to_source = {
    "latent_diffusion": ("m2", "diffusion"),
    "padapt": ("m1", "padapt"),
    "purebc": ("m1", "purebc"),
}

def parse_eval(log_path: Path):
    txt = log_path.read_text(errors="ignore")
    m = re.findall(
        r"EvalSummary steps=\d+ avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)",
        txt,
    )
    if not m:
        raise RuntimeError(f"Missing EvalSummary in {log_path}")
    reward, done = m[-1]
    return float(reward), float(done)

rows = []
for algo, (src, prefix) in algo_to_source.items():
    base = m2 if src == "m2" else m1
    for cond in conds:
        rewards = []
        dones = []
        for seed in seeds:
            log_path = base / f"{prefix}_{cond}_s{seed}.log"
            r, d = parse_eval(log_path)
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

def row(algo, cond):
    for x in rows:
        if x["algo"] == algo and x["condition"] == cond:
            return x
    raise KeyError((algo, cond))

cmp_rows = []
for cond in conds:
    ld = row("latent_diffusion", cond)["reward_mean"]
    pd = row("padapt", cond)["reward_mean"]
    bc = row("purebc", cond)["reward_mean"]
    cmp_rows.append(
        {
            "condition": cond,
            "latent_minus_padapt": ld - pd,
            "latent_minus_purebc": ld - bc,
        }
    )

latent_beats_padapt = all(x["latent_minus_padapt"] > 0.0 for x in cmp_rows)
latent_beats_purebc_robust = all(
    x["latent_minus_purebc"] > 0.0 for x in cmp_rows if x["condition"] != "nominal"
)

if latent_beats_padapt and latent_beats_purebc_robust:
    m3_local_conclusion = "support_keep_latent_mainline"
    next_step_recommendation = "continue_latent_mainline_optimization"
else:
    m3_local_conclusion = "not_support_latent_mainline"
    next_step_recommendation = "prepare_m4_residual_fallback_gate"

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

lines = []
lines.append("# PLANS_v2 M3 Minimum Credible Comparison")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append("- run_id: `plansv2_m3_min_compare_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- config_snapshot:")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/config_032315_8bf90ec.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/config_032011_8bf90ec.yaml`")
lines.append("- dataset_version: `N/A (aggregated from online-eval logs; no rollout dataset used)`")
lines.append("- dataset_hash: `N/A (aggregated from online-eval logs; no rollout dataset used)`")
lines.append("- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append(f"- seeds: `{','.join(seeds)}`")
lines.append("- eval_episodes_per_run: `N/A (fixed-step protocol in upstream packs)`")
lines.append("- eval_env_steps_per_run: `256`")
lines.append("- protocol: `nominal + light_v2 + hard`")
lines.append("- upstream_evidence_m1: `docs/plansv2_m1_baseline_pack.md`")
lines.append("- upstream_evidence_m2: `docs/plansv2_m2_gap_gate.md`")
lines.append("- artifacts_root_m1: `outputs/robustness_eval/plansv2_m1/`")
lines.append("- artifacts_root_m2: `outputs/robustness_eval/plansv2_m2_gap_gate/`")
lines.append("- primary_metrics:")
lines.append("  - `avg_reward` (main compare axis)")
lines.append("  - `avg_done_rate` (secondary)")
lines.append("- dispersion_metric: `std across seeds`")
lines.append("- table_source: `docs/plansv2_m3_min_compare.md`")
lines.append("- artifact_log_paths:")
lines.append("  - `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`")
lines.append("  - `outputs/robustness_eval/plansv2_m2_gap_gate/{mode}_{condition}_s{seed}.log`")
lines.append("- artifact_checkpoint_paths:")
lines.append("  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`")
lines.append("  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`")
lines.append("")
lines.append("## Unified Comparison Table")
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
lines.append("## Reward Delta (latent diffusion as anchor)")
lines.append("")
lines.append("| Condition | latent - padapt | latent - purebc |")
lines.append("|---|---:|---:|")
for c in cmp_rows:
    lines.append(
        f"| {c['condition']} | {c['latent_minus_padapt']:.6f} | {c['latent_minus_purebc']:.6f} |"
    )
lines.append("")
lines.append("## Local M3 Decision")
lines.append("")
lines.append(f"- latent_beats_padapt_all_conditions: `{latent_beats_padapt}`")
lines.append(f"- latent_beats_purebc_on_robust_conditions: `{latent_beats_purebc_robust}`")
lines.append(f"- local_m3_conclusion: `{m3_local_conclusion}`")
lines.append(f"- local_next_step_recommendation: `{next_step_recommendation}`")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
if m3_local_conclusion == "support_keep_latent_mainline":
    lines.append("- support: latent diffusion shows consistent multiseed gains under current protocol.")
else:
    lines.append("- not support: latent diffusion does not show robust multiseed advantage over current student baselines, so M4 residual fallback preparation should be started.")

out.write_text("\n".join(lines) + "\n")
print(f"Wrote {out}")
PY

echo "[plansv2_m3] completed. Summary: ${SUMMARY_MD}"
