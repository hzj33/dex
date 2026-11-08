#!/usr/bin/env bash
set -euo pipefail

# Build thesis-ready data pack from existing PLANS_v2 artifacts (no new training).
#
# Outputs:
#   - docs/data/plansv2_paper_seed_table.csv
#   - docs/data/plansv2_paper_agg_table.csv
#   - docs/data/plansv2_paper_delta_table.csv
#   - docs/plansv2_paper_data_pack.md

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

python - <<'PY'
from pathlib import Path
import csv
import math
import re
import statistics as st
import subprocess

repo = Path(".")
data_dir = repo / "docs" / "data"
data_dir.mkdir(parents=True, exist_ok=True)

seed_csv = data_dir / "plansv2_paper_seed_table.csv"
agg_csv = data_dir / "plansv2_paper_agg_table.csv"
delta_csv = data_dir / "plansv2_paper_delta_table.csv"
summary_md = repo / "docs" / "plansv2_paper_data_pack.md"

dirs = {
    "m1_baseline": repo / "outputs" / "robustness_eval" / "plansv2_m1",
    "m2_gap": repo / "outputs" / "robustness_eval" / "plansv2_m2_gap_gate",
    "m4_nominal_unscaled": repo / "outputs" / "robustness_eval" / "plansv2_m4_residual_gate_nominal",
    "m4_nominal_scale05": repo / "outputs" / "robustness_eval" / "plansv2_m4_residual_gate_nominal_scale05",
    "m4_compare_robust": repo / "outputs" / "robustness_eval" / "plansv2_m4_residual_compare_pack",
}
for k, d in dirs.items():
    if not d.exists():
        raise RuntimeError(f"Missing required artifact dir for {k}: {d}")

eval_re = re.compile(
    r"EvalSummary steps=(\d+) avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)"
)
recon_re = re.compile(
    r"EvalReconSummary steps=(\d+) mode=([a-z_]+) "
    r"latent_mse=([-0-9.eE+]+) latent_l1=([-0-9.eE+]+) "
    r"action_mse_to_teacher=([-0-9.eE+]+)"
)

def parse_eval(path: Path):
    txt = path.read_text(errors="ignore")
    m = eval_re.findall(txt)
    if not m:
        raise RuntimeError(f"Missing EvalSummary in {path}")
    s, r, d = m[-1]
    return int(s), float(r), float(d)

def parse_recon(path: Path):
    txt = path.read_text(errors="ignore")
    m = recon_re.findall(txt)
    if not m:
        return None
    _, mode, latent_mse, latent_l1, action_mse = m[-1]
    return {
        "eval_mode": mode,
        "latent_mse": float(latent_mse),
        "latent_l1": float(latent_l1),
        "action_mse_to_teacher": float(action_mse),
    }

def parse_residual(path: Path):
    txt = path.read_text(errors="ignore")
    lines = re.findall(r"EvalResidualSummary[^\n]*", txt)
    if not lines:
        return None
    kv = {}
    for key, val in re.findall(r"([a-z0-9_]+)=([A-Za-z_0-9.+-]+)", lines[-1]):
        kv[key] = val

    def f(name):
        return float(kv[name]) if name in kv else float("nan")

    return {
        "residual_abs_mean": f("residual_abs_mean"),
        "residual_l2_mean": f("residual_l2_mean"),
        "residual_to_target_ratio": f("residual_to_target_ratio"),
        "pred_residual_abs_mean": f("pred_residual_abs_mean"),
        "pred_to_target_ratio": f("pred_to_target_ratio"),
        "action_correction_abs_mean": f("action_correction_abs_mean"),
        "action_correction_l2_mean": f("action_correction_l2_mean"),
        "base_action_mse_to_teacher": f("base_action_mse_to_teacher"),
    }

rows = []

def add_row(source, variant, condition, seed, log_path: Path):
    steps, reward, done = parse_eval(log_path)
    recon = parse_recon(log_path) or {}
    residual = parse_residual(log_path) or {}
    row = {
        "source": source,
        "variant": variant,
        "condition": condition,
        "seed": int(seed),
        "steps": steps,
        "avg_reward": reward,
        "avg_done_rate": done,
        "eval_mode": recon.get("eval_mode", ""),
        "latent_mse": recon.get("latent_mse", float("nan")),
        "latent_l1": recon.get("latent_l1", float("nan")),
        "action_mse_to_teacher": recon.get("action_mse_to_teacher", float("nan")),
        "residual_abs_mean": residual.get("residual_abs_mean", float("nan")),
        "residual_l2_mean": residual.get("residual_l2_mean", float("nan")),
        "residual_to_target_ratio": residual.get("residual_to_target_ratio", float("nan")),
        "pred_residual_abs_mean": residual.get("pred_residual_abs_mean", float("nan")),
        "pred_to_target_ratio": residual.get("pred_to_target_ratio", float("nan")),
        "action_correction_abs_mean": residual.get("action_correction_abs_mean", float("nan")),
        "action_correction_l2_mean": residual.get("action_correction_l2_mean", float("nan")),
        "base_action_mse_to_teacher": residual.get("base_action_mse_to_teacher", float("nan")),
        "log_path": str(log_path),
    }
    rows.append(row)

# M1 baseline
for p in sorted(dirs["m1_baseline"].glob("*.log")):
    m = re.match(r"(teacher_ppo|padapt|purebc)_(nominal|light_v2|hard)_s(\d+)\.log$", p.name)
    if m:
        add_row("m1_baseline", m.group(1), m.group(2), m.group(3), p)

# M2 latent gap pack
for p in sorted(dirs["m2_gap"].glob("*.log")):
    m = re.match(r"(diffusion|decode_only)_(nominal|light_v2|hard)_s(\d+)\.log$", p.name)
    if m:
        add_row("m2_gap", m.group(1), m.group(2), m.group(3), p)

# M4 nominal unscaled
for p in sorted(dirs["m4_nominal_unscaled"].glob("*.log")):
    m = re.match(r"residual_nominal_s(\d+)\.log$", p.name)
    if m:
        add_row("m4_nominal_unscaled", "residual_unscaled", "nominal", m.group(1), p)

# M4 nominal scale05
for p in sorted(dirs["m4_nominal_scale05"].glob("*.log")):
    m = re.match(r"residual_nominal_s(\d+)\.log$", p.name)
    if m:
        add_row("m4_nominal_scale05", "residual_scale05", "nominal", m.group(1), p)

# M4 robust compare pack
for p in sorted(dirs["m4_compare_robust"].glob("*.log")):
    m = re.match(r"(residual_unscaled|residual_scale05|padapt)_(light_v2|hard)_s(\d+)\.log$", p.name)
    if m:
        add_row("m4_compare_robust", m.group(1), m.group(2), m.group(3), p)

if not rows:
    raise RuntimeError("No rows parsed from PLANS_v2 artifact logs")

seed_fields = [
    "source", "variant", "condition", "seed", "steps",
    "avg_reward", "avg_done_rate",
    "eval_mode", "latent_mse", "latent_l1", "action_mse_to_teacher",
    "residual_abs_mean", "residual_l2_mean", "residual_to_target_ratio",
    "pred_residual_abs_mean", "pred_to_target_ratio",
    "action_correction_abs_mean", "action_correction_l2_mean",
    "base_action_mse_to_teacher", "log_path",
]

with seed_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=seed_fields)
    w.writeheader()
    for r in rows:
        rr = dict(r)
        for k in rr:
            v = rr[k]
            if isinstance(v, float) and math.isnan(v):
                rr[k] = ""
        w.writerow(rr)

def finite_vals(key, group):
    vals = [x[key] for x in group if isinstance(x[key], float) and not math.isnan(x[key])]
    return vals

agg_rows = []
group_keys = sorted({(r["source"], r["variant"], r["condition"]) for r in rows})
metric_keys = [
    "avg_reward", "avg_done_rate",
    "latent_mse", "latent_l1", "action_mse_to_teacher",
    "residual_abs_mean", "residual_l2_mean", "residual_to_target_ratio",
    "pred_residual_abs_mean", "pred_to_target_ratio",
    "action_correction_abs_mean", "action_correction_l2_mean",
    "base_action_mse_to_teacher",
]

for source, variant, condition in group_keys:
    g = [r for r in rows if (r["source"], r["variant"], r["condition"]) == (source, variant, condition)]
    seeds = sorted(r["seed"] for r in g)
    out = {
        "source": source,
        "variant": variant,
        "condition": condition,
        "n": len(g),
        "seeds": ",".join(map(str, seeds)),
    }
    for key in metric_keys:
        vals = finite_vals(key, g)
        if vals:
            mean = st.mean(vals)
            std = st.pstdev(vals)
            sem = std / math.sqrt(len(vals))
            ci95 = 1.96 * sem
            out[f"{key}_mean"] = mean
            out[f"{key}_std"] = std
            out[f"{key}_ci95"] = ci95
        else:
            out[f"{key}_mean"] = float("nan")
            out[f"{key}_std"] = float("nan")
            out[f"{key}_ci95"] = float("nan")
    agg_rows.append(out)

agg_fields = ["source", "variant", "condition", "n", "seeds"]
for key in metric_keys:
    agg_fields += [f"{key}_mean", f"{key}_std", f"{key}_ci95"]

with agg_csv.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=agg_fields)
    w.writeheader()
    for r in agg_rows:
        rr = dict(r)
        for k, v in rr.items():
            if isinstance(v, float) and math.isnan(v):
                rr[k] = ""
        w.writerow(rr)

def agg_get(source, variant, condition, metric):
    for r in agg_rows:
        if r["source"] == source and r["variant"] == variant and r["condition"] == condition:
            return r[f"{metric}_mean"]
    raise KeyError((source, variant, condition, metric))

delta_rows = []
for cond in ["nominal", "light_v2", "hard"]:
    # M3-style latent diffusion vs m1 baselines (using m2_gap diffusion + m1 baseline)
    if cond in ["nominal", "light_v2", "hard"]:
        try:
            latent = agg_get("m2_gap", "diffusion", cond, "avg_reward")
            padapt = agg_get("m1_baseline", "padapt", cond, "avg_reward")
            purebc = agg_get("m1_baseline", "purebc", cond, "avg_reward")
            delta_rows.append({
                "delta_group": "m3_latent_vs_baselines",
                "condition": cond,
                "metric": "avg_reward",
                "lhs": "latent_diffusion",
                "rhs": "padapt",
                "delta_lhs_minus_rhs": latent - padapt,
            })
            delta_rows.append({
                "delta_group": "m3_latent_vs_baselines",
                "condition": cond,
                "metric": "avg_reward",
                "lhs": "latent_diffusion",
                "rhs": "purebc",
                "delta_lhs_minus_rhs": latent - purebc,
            })
        except KeyError:
            pass

for cond in ["light_v2", "hard"]:
    # M4 robust compare deltas
    ru = agg_get("m4_compare_robust", "residual_unscaled", cond, "avg_reward")
    rs = agg_get("m4_compare_robust", "residual_scale05", cond, "avg_reward")
    pd = agg_get("m4_compare_robust", "padapt", cond, "avg_reward")
    delta_rows.append({
        "delta_group": "m4_residual_compare",
        "condition": cond,
        "metric": "avg_reward",
        "lhs": "residual_scale05",
        "rhs": "residual_unscaled",
        "delta_lhs_minus_rhs": rs - ru,
    })
    delta_rows.append({
        "delta_group": "m4_residual_compare",
        "condition": cond,
        "metric": "avg_reward",
        "lhs": "residual_scale05",
        "rhs": "padapt",
        "delta_lhs_minus_rhs": rs - pd,
    })
    delta_rows.append({
        "delta_group": "m4_residual_compare",
        "condition": cond,
        "metric": "avg_reward",
        "lhs": "residual_unscaled",
        "rhs": "padapt",
        "delta_lhs_minus_rhs": ru - pd,
    })

with delta_csv.open("w", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "delta_group",
            "condition",
            "metric",
            "lhs",
            "rhs",
            "delta_lhs_minus_rhs",
        ],
    )
    w.writeheader()
    for r in delta_rows:
        w.writerow(r)

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

def fmt(x):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "N/A"
    return f"{x:.6f}"

lines = []
lines.append("# PLANS_v2 Paper Data Pack")
lines.append("")
lines.append("## Evidence Block")
lines.append("")
lines.append("- run_id: `plansv2_paper_data_pack_2026-03-24`")
lines.append(f"- git_commit: `{commit}`")
lines.append("- sources:")
lines.append("  - `outputs/robustness_eval/plansv2_m1/`")
lines.append("  - `outputs/robustness_eval/plansv2_m2_gap_gate/`")
lines.append("  - `outputs/robustness_eval/plansv2_m4_residual_gate_nominal/`")
lines.append("  - `outputs/robustness_eval/plansv2_m4_residual_gate_nominal_scale05/`")
lines.append("  - `outputs/robustness_eval/plansv2_m4_residual_compare_pack/`")
lines.append("- data_tables:")
lines.append(f"  - `{seed_csv}`")
lines.append(f"  - `{agg_csv}`")
lines.append(f"  - `{delta_csv}`")
lines.append("- summary_doc: `docs/plansv2_paper_data_pack.md`")
lines.append("")
lines.append("## Coverage")
lines.append("")
lines.append(f"- total_seed_rows: `{len(rows)}`")
lines.append(f"- total_aggregated_groups: `{len(agg_rows)}`")
lines.append("- conditions_seen: `nominal, light_v2, hard`")
lines.append("- stages_seen: `M1, M2, M4`")
lines.append("")
lines.append("## Key Thesis Numbers (Reward Mean ± 95% CI)")
lines.append("")
lines.append("| Comparison | Condition | Value |")
lines.append("|---|---|---:|")
for cond in ["nominal", "light_v2", "hard"]:
    latent = agg_get("m2_gap", "diffusion", cond, "avg_reward")
    latent_ci = next(r for r in agg_rows if r["source"] == "m2_gap" and r["variant"] == "diffusion" and r["condition"] == cond)["avg_reward_ci95"]
    padapt = agg_get("m1_baseline", "padapt", cond, "avg_reward")
    padapt_ci = next(r for r in agg_rows if r["source"] == "m1_baseline" and r["variant"] == "padapt" and r["condition"] == cond)["avg_reward_ci95"]
    lines.append(f"| latent_diffusion | {cond} | {latent:.6f} ± {latent_ci:.6f} |")
    lines.append(f"| padapt | {cond} | {padapt:.6f} ± {padapt_ci:.6f} |")
for cond in ["light_v2", "hard"]:
    rs = agg_get("m4_compare_robust", "residual_scale05", cond, "avg_reward")
    rs_ci = next(r for r in agg_rows if r["source"] == "m4_compare_robust" and r["variant"] == "residual_scale05" and r["condition"] == cond)["avg_reward_ci95"]
    lines.append(f"| residual_scale05 | {cond} | {rs:.6f} ± {rs_ci:.6f} |")
lines.append("")
lines.append("## Key Deltas")
lines.append("")
lines.append("| Delta Group | Condition | LHS - RHS | Delta |")
lines.append("|---|---|---|---:|")
for d in delta_rows:
    if d["condition"] in ["light_v2", "hard"] or d["rhs"] in ["padapt", "purebc"]:
        lines.append(
            f"| {d['delta_group']} | {d['condition']} | {d['lhs']} - {d['rhs']} | {d['delta_lhs_minus_rhs']:.6f} |"
        )
lines.append("")
lines.append("## One-line Conclusion")
lines.append("")
lines.append("- support: this pack consolidates thesis-ready seed-level and aggregated evidence from M1-M4 without adding new training runs.")

summary_md.write_text("\n".join(lines) + "\n")
print(f"Wrote {seed_csv}")
print(f"Wrote {agg_csv}")
print(f"Wrote {delta_csv}")
print(f"Wrote {summary_md}")
PY

echo "[plansv2_paper_data_pack] completed."
