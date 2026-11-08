# PLANS_v4 Thesis Result Bundle

## Evidence Block

- run_id: `plansv4_thesis_result_bundle_2026-04-15`
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
- candidate_count: `7`
- single_seed_gate_fail_count: `7`
- hard_stop_trigger: `TRUE` (`v4m0 delta_hard=-0.137635`)

## Main Thesis-Use Table (Seed42, Fixed-Step)

| Candidate | Bug Status | Nominal | Light_v2 | Hard | Gate |
|---|---|---:|---:|---:|---|
| v3a1_curriculum_stagedte | buggy | 1.866026 | 1.769389 | 1.352241 | FAIL |
| v3a2_curriculum_stagedhold_hardtarget | buggy | 1.493377 | 1.384558 | 1.189769 | FAIL |
| v3a3_curriculum_linear_midtarget | buggy | 1.645099 | 1.592684 | 1.453029 | FAIL |
| v3b1_residual_scale05 | buggy | 1.391040 | 1.247124 | 0.957792 | FAIL |
| v3b2_residual_scale10 | buggy | 1.332490 | 1.220360 | 1.152434 | FAIL |
| v3b3_residual_notail | buggy | 1.639847 | 1.789730 | 1.343203 | FAIL |
| v4m0_bugfix_baseline | bug_free | 1.738804 | 1.588133 | 1.367269 | FAIL |

## Compact Delta Table (vs V3-M0 reference)

| Candidate | Δhard | Δlight_v2 | Δdone_hard | Gate |
|---|---:|---:|---:|---|
| v3a1_curriculum_stagedte | -0.152663 | +0.131123 | +0.000569 | FAIL |
| v3a2_curriculum_stagedhold_hardtarget | -0.315135 | -0.253708 | +0.000488 | FAIL |
| v3a3_curriculum_linear_midtarget | -0.051875 | -0.045582 | +0.000244 | FAIL |
| v3b1_residual_scale05 | -0.547112 | -0.391142 | +0.000569 | FAIL |
| v3b2_residual_scale10 | -0.352470 | -0.417906 | +0.000732 | FAIL |
| v3b3_residual_notail | -0.161701 | +0.151464 | +0.000244 | FAIL |
| v4m0_bugfix_baseline | -0.137635 | -0.050133 | +0.000407 | FAIL |

## One-line Conclusion

- Under unified gate and bug-free revalidation, diffusion candidates remain non-competitive against the frozen reference and PAdapt baseline; V4 is closed as `Suspend`.
