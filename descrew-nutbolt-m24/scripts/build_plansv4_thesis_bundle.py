#!/usr/bin/env python3
"""
Build PLANS_v4 thesis bundle artifacts from existing robustness logs.

Outputs:
  - docs/data/plansv4_thesis_candidate_table.csv
  - docs/data/plansv4_thesis_delta_table.csv
  - docs/data/plansv4_thesis_tables.tex
  - docs/plansv4_thesis_result_bundle.md
  - docs/plansv4_thesis_results_subsection_draft.md
"""

import csv
import re
from datetime import date
from pathlib import Path


REF = {
    "nominal_reward": 1.675112,
    "light_v2_reward": 1.638266,
    "hard_reward": 1.504904,
    "nominal_done": 0.001302,
    "light_v2_done": 0.001383,
    "hard_done": 0.001872,
}

PADAPT = {
    "nominal_reward": 2.167820,
    "light_v2_reward": 2.079074,
    "hard_reward": 1.838225,
}

CANDIDATES = {
    "v3a1_curriculum_stagedte": {
        "bug_status": "buggy",
        "phase": "v3_mainline_a",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a1_curriculum_stagedte_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m1_a1_curriculum_stagedte_seed42",
    },
    "v3a2_curriculum_stagedhold_hardtarget": {
        "bug_status": "buggy",
        "phase": "v3_mainline_a",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a2_curriculum_stagedhold_hardtarget_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m1_a2_curriculum_stagedhold_hardtarget_seed42",
    },
    "v3a3_curriculum_linear_midtarget": {
        "bug_status": "buggy",
        "phase": "v3_mainline_a",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3a3_curriculum_linear_midtarget_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m1_a3_curriculum_linear_midtarget_seed42",
    },
    "v3b1_residual_scale05": {
        "bug_status": "buggy",
        "phase": "v3_mainline_b",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b1_residual_scale05_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m2_b1_residual_scale05_seed42",
    },
    "v3b2_residual_scale10": {
        "bug_status": "buggy",
        "phase": "v3_mainline_b",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b2_residual_scale10_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m2_b2_residual_scale10_seed42",
    },
    "v3b3_residual_notail": {
        "bug_status": "buggy",
        "phase": "v3_mainline_b",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v3b3_residual_notail_seed42_full",
        "log_dir": "outputs/robustness_eval/plansv3_m2_b3_residual_notail_seed42",
    },
    "v4m0_bugfix_baseline": {
        "bug_status": "bug_free",
        "phase": "v4_m0_bugfix",
        "run_dir": "outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min",
        "logs": {
            "nominal": "outputs/v4_logs/v4_m0_bugfix_eval_nominal_seed42.log",
            "light_v2": "outputs/v4_logs/v4_m0_bugfix_eval_lightv2_seed42.log",
            "hard": "outputs/v4_logs/v4_m0_bugfix_eval_hard_seed42.log",
        },
    },
}

EVAL_SUMMARY_PATTERN = re.compile(
    r"EvalSummary steps=256 avg_reward=([-0-9.eE+]+) avg_done_rate=([-0-9.eE+]+)"
)


def parse_log(path: str) -> tuple[float, float]:
    content = Path(path).read_text(errors="ignore")
    matches = EVAL_SUMMARY_PATTERN.findall(content)
    if not matches:
        raise RuntimeError(f"No EvalSummary found in: {path}")
    reward, done = matches[-1]
    return float(reward), float(done)


def collect_rows() -> list[dict]:
    rows = []
    for name, meta in CANDIDATES.items():
        if name == "v4m0_bugfix_baseline":
            logs = meta["logs"]
        else:
            log_dir = Path(meta["log_dir"])
            logs = {
                "nominal": str(log_dir / "diffusion_nominal_s42.log"),
                "light_v2": str(log_dir / "diffusion_light_v2_s42.log"),
                "hard": str(log_dir / "diffusion_hard_s42.log"),
            }

        nominal_reward, nominal_done = parse_log(logs["nominal"])
        light_reward, light_done = parse_log(logs["light_v2"])
        hard_reward, hard_done = parse_log(logs["hard"])

        row = {
            "candidate": name,
            "phase": meta["phase"],
            "bug_status": meta["bug_status"],
            "run_dir": meta["run_dir"],
            "seed": 42,
            "steps": 256,
            "nominal_reward": nominal_reward,
            "nominal_done": nominal_done,
            "light_v2_reward": light_reward,
            "light_v2_done": light_done,
            "hard_reward": hard_reward,
            "hard_done": hard_done,
            "delta_nominal_vs_v3m0": nominal_reward - REF["nominal_reward"],
            "delta_light_v2_vs_v3m0": light_reward - REF["light_v2_reward"],
            "delta_hard_vs_v3m0": hard_reward - REF["hard_reward"],
            "delta_hard_done_vs_v3m0": hard_done - REF["hard_done"],
            "delta_nominal_vs_padapt": nominal_reward - PADAPT["nominal_reward"],
            "delta_light_v2_vs_padapt": light_reward - PADAPT["light_v2_reward"],
            "delta_hard_vs_padapt": hard_reward - PADAPT["hard_reward"],
        }
        hard_ok = row["delta_hard_vs_v3m0"] >= -0.05
        light_ok = row["delta_light_v2_vs_v3m0"] >= -0.08
        done_ok = row["delta_hard_done_vs_v3m0"] <= 0.0005
        row["gate_single_seed"] = "PASS" if (hard_ok and light_ok and done_ok) else "FAIL"
        rows.append(row)

    return sorted(rows, key=lambda x: x["candidate"])


def write_csv_tables(rows: list[dict], data_dir: Path) -> None:
    full_table = data_dir / "plansv4_thesis_candidate_table.csv"
    full_fields = [
        "candidate",
        "phase",
        "bug_status",
        "run_dir",
        "seed",
        "steps",
        "nominal_reward",
        "nominal_done",
        "light_v2_reward",
        "light_v2_done",
        "hard_reward",
        "hard_done",
        "delta_nominal_vs_v3m0",
        "delta_light_v2_vs_v3m0",
        "delta_hard_vs_v3m0",
        "delta_hard_done_vs_v3m0",
        "delta_nominal_vs_padapt",
        "delta_light_v2_vs_padapt",
        "delta_hard_vs_padapt",
        "gate_single_seed",
    ]
    with full_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=full_fields)
        writer.writeheader()
        writer.writerows(rows)

    delta_table = data_dir / "plansv4_thesis_delta_table.csv"
    delta_fields = [
        "candidate",
        "bug_status",
        "delta_hard_vs_v3m0",
        "delta_light_v2_vs_v3m0",
        "delta_hard_done_vs_v3m0",
        "delta_hard_vs_padapt",
        "gate_single_seed",
    ]
    with delta_table.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=delta_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in delta_fields})


def write_latex_table(rows: list[dict], data_dir: Path) -> None:
    tex_file = data_dir / "plansv4_thesis_tables.tex"
    with tex_file.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated from artifact-backed logs\n")
        f.write("\\begin{tabular}{l l r r r l}\n")
        f.write("\\hline\n")
        f.write("Candidate & Bug & $\\Delta_{hard}$ & $\\Delta_{light}$ & $\\Delta done_{hard}$ & Gate \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{row['candidate']} & {row['bug_status']} & "
                f"{row['delta_hard_vs_v3m0']:+.6f} & {row['delta_light_v2_vs_v3m0']:+.6f} & "
                f"{row['delta_hard_done_vs_v3m0']:+.6f} & {row['gate_single_seed']} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")


def write_bundle_docs(rows: list[dict], docs_dir: Path) -> None:
    v4_row = [r for r in rows if r["candidate"] == "v4m0_bugfix_baseline"][0]
    fail_count = sum(1 for r in rows if r["gate_single_seed"] == "FAIL")

    bundle_doc = docs_dir / "plansv4_thesis_result_bundle.md"
    bundle_doc.write_text(
        """# PLANS_v4 Thesis Result Bundle

## Evidence Block

- run_id: `plansv4_thesis_result_bundle_{today}`
- git_commit: `1f8d373fd695c04b52657ba82514ba41873903a0`
- derived_from:
  - `docs/plansv4_m3_final_verdict.md`
  - `outputs/robustness_eval/plansv3_m1_*_seed42/diffusion_*.log`
  - `outputs/robustness_eval/plansv3_m2_*_seed42/diffusion_*.log`
  - `outputs/v4_logs/v4_m0_bugfix_eval_*_seed42.log`
- outputs:
  - `docs/data/plansv4_thesis_candidate_table.csv`
  - `docs/data/plansv4_thesis_delta_table.csv`
  - `docs/data/plansv4_thesis_tables.tex`
  - `docs/plansv4_thesis_result_bundle.md`

## Decision Snapshot

- final_verdict: `Suspend` (governance-confirmed)
- candidate_count: `{count}`
- single_seed_gate_fail_count: `{fails}`
- hard_stop_trigger: `{hard_stop}` (`v4m0 delta_hard={hard_delta:+.6f}`)

## Main Thesis-Use Table (Seed42, Fixed-Step)

| Candidate | Bug Status | Nominal | Light_v2 | Hard | Gate |
|---|---|---:|---:|---:|---|
{main_table}

## Compact Delta Table (vs V3-M0 reference)

| Candidate | Δhard | Δlight_v2 | Δdone_hard | Gate |
|---|---:|---:|---:|---|
{delta_table}

## One-line Conclusion

- Under unified gate and bug-free revalidation, diffusion candidates remain non-competitive against the frozen reference and PAdapt baseline; V4 is closed as `Suspend`.
""".format(
            today=date.today().isoformat(),
            count=len(rows),
            fails=fail_count,
            hard_stop="TRUE" if v4_row["delta_hard_vs_v3m0"] < -0.10 else "FALSE",
            hard_delta=v4_row["delta_hard_vs_v3m0"],
            main_table="\n".join(
                f"| {r['candidate']} | {r['bug_status']} | "
                f"{r['nominal_reward']:.6f} | {r['light_v2_reward']:.6f} | {r['hard_reward']:.6f} | {r['gate_single_seed']} |"
                for r in rows
            ),
            delta_table="\n".join(
                f"| {r['candidate']} | {r['delta_hard_vs_v3m0']:+.6f} | "
                f"{r['delta_light_v2_vs_v3m0']:+.6f} | {r['delta_hard_done_vs_v3m0']:+.6f} | {r['gate_single_seed']} |"
                for r in rows
            ),
        ),
        encoding="utf-8",
    )

    subsection_doc = docs_dir / "plansv4_thesis_results_subsection_draft.md"
    subsection_doc.write_text(
        """# Thesis Results Subsection (Draft, PLANS_v4)

## Experimental Protocol

All V4 claims are based on fixed-step evaluation (`256` steps) with a unified single-seed gate (`seed=42`) under three conditions: `nominal`, `light_v2`, and `hard`. The primary gate uses deltas against the V3-M0 frozen reference (`nominal=1.675112`, `light_v2=1.638266`, `hard=1.504904`, `hard_done=0.001872`). Secondary comparison is reported against the multiseed PAdapt baseline (`nominal=2.167820`, `light_v2=2.079074`, `hard=1.838225`).

## Main Findings

V3 produced six diffusion candidates (Mainline A/B) and none passed the single-seed gate. Because all V3 candidates were trained on buggy code, V4 re-ran a bug-free baseline (`v4m0_bugfix_baseline`) with the same frozen configuration. The bug-free run also failed the gate, with `hard delta = -0.137635`, and triggered the V4 hard-stop rule (`hard delta < -0.10`).

## Negative Result (Clean Evidence)

The bug-free V4 baseline shows `nominal=1.738804`, `light_v2=1.588133`, `hard=1.367269`, and `hard_done=0.002279`. Relative to V3-M0, this is `+0.063692` nominal, `-0.050133` light_v2, and `-0.137635` hard. Relative to PAdapt, all reward deltas remain negative (`nominal=-0.429016`, `light_v2=-0.490941`, `hard=-0.470956`). This confirms that diffusion did not establish robust competitiveness under the current scope.

## Stage Decision and Thesis Positioning

PLANS_v4 is closed with **Suspend** (governance-confirmed). The thesis-facing positioning is: keep PAdapt as the practical baseline, report diffusion as a controlled negative result, and explicitly document the bug-fix rerun to show that the final conclusion is based on clean evidence rather than buggy training artifacts.

## Artifact Pointers

- verdict doc: `docs/plansv4_m3_final_verdict.md`
- result bundle: `docs/plansv4_thesis_result_bundle.md`
- candidate table CSV: `docs/data/plansv4_thesis_candidate_table.csv`
- delta table CSV: `docs/data/plansv4_thesis_delta_table.csv`
- LaTeX table: `docs/data/plansv4_thesis_tables.tex`
""",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(".")
    docs_dir = root / "docs"
    data_dir = docs_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows()
    write_csv_tables(rows, data_dir)
    write_latex_table(rows, data_dir)
    write_bundle_docs(rows, docs_dir)

    print("wrote docs/data/plansv4_thesis_candidate_table.csv")
    print("wrote docs/data/plansv4_thesis_delta_table.csv")
    print("wrote docs/data/plansv4_thesis_tables.tex")
    print("wrote docs/plansv4_thesis_result_bundle.md")
    print("wrote docs/plansv4_thesis_results_subsection_draft.md")


if __name__ == "__main__":
    main()
