#!/usr/bin/env bash
set -euo pipefail

# Build thesis-facing result bundle from plansv2 paper data pack.
#
# Inputs:
#   - docs/data/plansv2_paper_agg_table.csv
#   - docs/data/plansv2_paper_delta_table.csv
#
# Outputs:
#   - docs/data/plansv2_thesis_main_table.csv
#   - docs/data/plansv2_thesis_negative_delta_table.csv
#   - docs/data/plansv2_thesis_tables.tex
#   - docs/plansv2_thesis_result_bundle.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

python - <<'PY'
from pathlib import Path
import csv
import subprocess

data_dir = Path("docs/data")
agg_csv = data_dir / "plansv2_paper_agg_table.csv"
delta_csv = data_dir / "plansv2_paper_delta_table.csv"
if not agg_csv.exists():
    raise RuntimeError(f"Missing input: {agg_csv}")
if not delta_csv.exists():
    raise RuntimeError(f"Missing input: {delta_csv}")

main_table_csv = data_dir / "plansv2_thesis_main_table.csv"
neg_table_csv = data_dir / "plansv2_thesis_negative_delta_table.csv"
latex_out = data_dir / "plansv2_thesis_tables.tex"
summary_md = Path("docs/plansv2_thesis_result_bundle.md")

agg_rows = list(csv.DictReader(agg_csv.open()))
delta_rows = list(csv.DictReader(delta_csv.open()))

def find_agg(source, variant, condition):
    for r in agg_rows:
        if r["source"] == source and r["variant"] == variant and r["condition"] == condition:
            return r
    raise KeyError((source, variant, condition))

def f6(x):
    return f"{float(x):.6f}"

def pm_md(mean, std):
    return f"{float(mean):.6f} +/- {float(std):.6f}"

def pm_tex(mean, std):
    return f"{float(mean):.6f} $\\\\pm$ {float(std):.6f}"

algorithms = [
    {
        "algorithm": "teacher_ppo",
        "nominal": ("m1_baseline", "teacher_ppo", "nominal"),
        "light_v2": ("m1_baseline", "teacher_ppo", "light_v2"),
        "hard": ("m1_baseline", "teacher_ppo", "hard"),
        "role": "upper_bound_teacher",
    },
    {
        "algorithm": "padapt",
        "nominal": ("m1_baseline", "padapt", "nominal"),
        "light_v2": ("m1_baseline", "padapt", "light_v2"),
        "hard": ("m1_baseline", "padapt", "hard"),
        "role": "selected_mainline_baseline",
    },
    {
        "algorithm": "purebc",
        "nominal": ("m1_baseline", "purebc", "nominal"),
        "light_v2": ("m1_baseline", "purebc", "light_v2"),
        "hard": ("m1_baseline", "purebc", "hard"),
        "role": "minimal_student_baseline",
    },
    {
        "algorithm": "latent_diffusion",
        "nominal": ("m2_gap", "diffusion", "nominal"),
        "light_v2": ("m2_gap", "diffusion", "light_v2"),
        "hard": ("m2_gap", "diffusion", "hard"),
        "role": "diffusion_candidate_latent",
    },
    {
        "algorithm": "residual_unscaled",
        "nominal": ("m4_nominal_unscaled", "residual_unscaled", "nominal"),
        "light_v2": ("m4_compare_robust", "residual_unscaled", "light_v2"),
        "hard": ("m4_compare_robust", "residual_unscaled", "hard"),
        "role": "diffusion_candidate_residual",
    },
    {
        "algorithm": "residual_scale05",
        "nominal": ("m4_nominal_scale05", "residual_scale05", "nominal"),
        "light_v2": ("m4_compare_robust", "residual_scale05", "light_v2"),
        "hard": ("m4_compare_robust", "residual_scale05", "hard"),
        "role": "diffusion_candidate_residual_scaled",
    },
]

main_rows = []
for a in algorithms:
    out = {
        "algorithm": a["algorithm"],
        "role": a["role"],
    }
    for cond in ["nominal", "light_v2", "hard"]:
        src, var, c = a[cond]
        r = find_agg(src, var, c)
        out[f"{cond}_reward_mean"] = f6(r["avg_reward_mean"])
        out[f"{cond}_reward_std"] = f6(r["avg_reward_std"])
        out[f"{cond}_reward_ci95"] = f6(r["avg_reward_ci95"])
        out[f"{cond}_reward_mean_pm_std"] = pm_md(r["avg_reward_mean"], r["avg_reward_std"])
        out[f"{cond}_reward_mean_pm_std_tex"] = pm_tex(r["avg_reward_mean"], r["avg_reward_std"])
        out[f"{cond}_done_mean"] = f6(r["avg_done_rate_mean"])
    main_rows.append(out)

main_fields = [
    "algorithm",
    "role",
    "nominal_reward_mean",
    "nominal_reward_std",
    "nominal_reward_ci95",
    "nominal_reward_mean_pm_std",
    "nominal_reward_mean_pm_std_tex",
    "nominal_done_mean",
    "light_v2_reward_mean",
    "light_v2_reward_std",
    "light_v2_reward_ci95",
    "light_v2_reward_mean_pm_std",
    "light_v2_reward_mean_pm_std_tex",
    "light_v2_done_mean",
    "hard_reward_mean",
    "hard_reward_std",
    "hard_reward_ci95",
    "hard_reward_mean_pm_std",
    "hard_reward_mean_pm_std_tex",
    "hard_done_mean",
]
with main_table_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=main_fields)
    w.writeheader()
    w.writerows(main_rows)

# Keep only thesis-facing negative-result deltas
neg_rows = []
for r in delta_rows:
    g = r["delta_group"]
    if g in {"m3_latent_vs_baselines", "m4_residual_compare"}:
        neg_rows.append(r)

neg_fields = ["delta_group", "condition", "metric", "lhs", "rhs", "delta_lhs_minus_rhs"]
with neg_table_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=neg_fields)
    w.writeheader()
    w.writerows(neg_rows)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

# Markdown summary
lines = []
lines.append("# PLANS_v2 Thesis Result Bundle")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append("- run_id: `plansv2_thesis_result_bundle_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- derived_from:")
lines.append("  - `docs/plansv2_paper_data_pack.md`")
lines.append("  - `docs/data/plansv2_paper_agg_table.csv`")
lines.append("  - `docs/data/plansv2_paper_delta_table.csv`")
lines.append("- outputs:")
lines.append(f"  - `{main_table_csv}`")
lines.append(f"  - `{neg_table_csv}`")
lines.append(f"  - `{latex_out}`")
lines.append("  - `docs/plansv2_thesis_result_bundle.md`")
lines.append("")
lines.append("## Main Table (Reward Mean ± Std)")
lines.append("")
lines.append("| Algorithm | Nominal | Light_v2 | Hard | Role |")
lines.append("|---|---:|---:|---:|---|")
for r in main_rows:
    lines.append(
        f"| {r['algorithm']} | {r['nominal_reward_mean_pm_std']} | "
        f"{r['light_v2_reward_mean_pm_std']} | {r['hard_reward_mean_pm_std']} | {r['role']} |"
    )
lines.append("")
lines.append("## Negative-Result Delta Table")
lines.append("")
lines.append("| Group | Condition | LHS - RHS | Delta (Reward) |")
lines.append("|---|---|---|---:|")
for r in neg_rows:
    lines.append(
        f"| {r['delta_group']} | {r['condition']} | {r['lhs']} - {r['rhs']} | {float(r['delta_lhs_minus_rhs']):.6f} |"
    )
lines.append("")
lines.append("## Thesis Narrative (Concise Draft)")
lines.append("")
lines.append("- 在统一 protocol（nominal/light_v2/hard，多 seed）下，`padapt` 在核心鲁棒条件上持续优于当前 diffusion 候选。")
lines.append("- `latent_diffusion` 在 nominal 与 `purebc` 可比，但在 `light_v2/hard` 下未形成相对 `padapt` 的优势。")
lines.append("- `residual` 路线（含 `scale05`）在局部稳定性指标上有可解释变化，但 robust reward 仍未超过 `padapt`。")
lines.append("- 因此本阶段采用 baseline-first closure：保留 diffusion 作为方法学探索与负/中性证据，主交付聚焦可复现基线结论。")
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
lines.append("- support: thesis-facing tables and concise narrative are now frozen from artifact-backed M1-M4 evidence.")
summary_md.write_text("\n".join(lines) + "\n")

# LaTeX snippets
def latex_escape(s: str):
    return s.replace("_", "\\_")

tex = []
tex.append("% Auto-generated by scripts/build_plansv2_thesis_result_bundle.sh")
tex.append("\\begin{table}[t]")
tex.append("\\centering")
tex.append("\\caption{Main comparison under unified protocol (reward mean $\\pm$ std).}")
tex.append("\\begin{tabular}{lccc}")
tex.append("\\hline")
tex.append("Algorithm & Nominal & Light\\_v2 & Hard \\\\")
tex.append("\\hline")
for r in main_rows:
    tex.append(
        f"{latex_escape(r['algorithm'])} & {r['nominal_reward_mean_pm_std_tex']} & "
        f"{r['light_v2_reward_mean_pm_std_tex']} & {r['hard_reward_mean_pm_std_tex']} \\\\"
    )
tex.append("\\hline")
tex.append("\\end{tabular}")
tex.append("\\label{tab:plansv2_main_compare}")
tex.append("\\end{table}")
tex.append("")
tex.append("\\begin{table}[t]")
tex.append("\\centering")
tex.append("\\caption{Negative/inconclusive diffusion deltas (reward).}")
tex.append("\\begin{tabular}{lllr}")
tex.append("\\hline")
tex.append("Group & Condition & LHS-RHS & Delta \\\\")
tex.append("\\hline")
for r in neg_rows:
    tex.append(
        f"{latex_escape(r['delta_group'])} & {latex_escape(r['condition'])} & "
        f"{latex_escape(r['lhs'])}-{latex_escape(r['rhs'])} & {float(r['delta_lhs_minus_rhs']):.6f} \\\\"
    )
tex.append("\\hline")
tex.append("\\end{tabular}")
tex.append("\\label{tab:plansv2_negative_deltas}")
tex.append("\\end{table}")
latex_out.write_text("\n".join(tex) + "\n")

print(f"Wrote {main_table_csv}")
print(f"Wrote {neg_table_csv}")
print(f"Wrote {latex_out}")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_thesis_result_bundle] completed."
