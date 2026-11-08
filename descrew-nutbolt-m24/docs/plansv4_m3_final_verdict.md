# PLANS_v4 M3 Final Verdict

Date: 2026-04-15  

Stage: `V4-M3` (Final Verdict)  

Status: `completed`

## 1. Trigger and Scope

- Trigger condition: `V4-M0` bug-fixed baseline has `hard delta < -0.10` vs V3-M0 reference.
- Measured hard delta: `-0.137635`.
- Scope: summarize all V3+V4 candidates under unified single-seed gate (`seed=42`, `steps=256`, `nominal/light_v2/hard`).

## 2. Frozen References

- V3-M0 reference (primary gate): nominal `1.675112`, light_v2 `1.638266`, hard `1.504904`; hard done `0.001872`.
- PAdapt multiseed baseline (secondary metric): nominal `2.167820`, light_v2 `2.079074`, hard `1.838225`.

## 3. V3+V4 Candidate Summary (Single-Seed)

| candidate | bug_status | hard_delta_vs_v3m0 | light_delta_vs_v3m0 | hard_done_delta | gate |
|---|---|---:|---:|---:|---|
| v3a1_curriculum_stagedte | buggy | -0.152663 | +0.131123 | +0.000569 | FAIL |
| v3a2_curriculum_stagedhold_hardtarget | buggy | -0.315135 | -0.253708 | +0.000488 | FAIL |
| v3a3_curriculum_linear_midtarget | buggy | -0.051875 | -0.045582 | +0.000244 | FAIL |
| v3b1_residual_scale05 | buggy | -0.547112 | -0.391142 | +0.000569 | FAIL |
| v3b2_residual_scale10 | buggy | -0.352470 | -0.417906 | +0.000732 | FAIL |
| v3b3_residual_notail | buggy | -0.161701 | +0.151464 | +0.000244 | FAIL |
| v4m0_bugfix_baseline | bug_free | -0.137635 | -0.050133 | +0.000407 | FAIL |

## 4. V4-M0 Evidence Block

- run_id/output_path: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min`
- git commit hash: `1f8d373fd695c04b52657ba82514ba41873903a0`
- config snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min/config_041415_1f8d373.yaml`
- seed(s): `42`
- evaluation steps: `256`
- primary metrics:
  - nominal: reward `1.738804`, done `0.001465`
  - light_v2: reward `1.588133`, done `0.001790`
  - hard: reward `1.367269`, done `0.002279`
- delta vs V3-M0 (primary gate):
  - nominal reward `+0.063692`
  - light_v2 reward `-0.050133`
  - hard reward `-0.137635`
  - hard done `+0.000407`
- delta vs PAdapt (secondary metric):
  - nominal `-0.429016`
  - light_v2 `-0.490941`
  - hard `-0.470956`
- gate result: `FAIL`
- one-line conclusion: bug-fixed baseline does not pass single-seed gate and hits hard-stop threshold.
- best checkpoint: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_v4m0_bugfix_seed42_15min/stage2_diffusion_nn/model_best.ckpt` (sha1 `83bc317790da32d78273db13d9acb65da8249023`).

## 5. Final Verdict (三选一)

- Final verdict: **Suspend**
- Reason 1: no V3 candidate passed single-seed gate (all 6 fail, all on buggy code).
- Reason 2: bug-fixed V4-M0 still fails and directly triggers hard-stop (`hard delta < -0.10`).
- Reason 3: secondary metric vs PAdapt remains consistently negative across all conditions.

## 6. Thesis Presentation Recommendation

- Keep PAdapt as the practical student baseline for the main results table.
- Report diffusion latent as a controlled negative result: V3 (buggy phase) + V4 (bug-free confirmation).
- Explicitly include the bug-fix note and V4 clean-evidence rerun to show conclusion robustness.
- Put candidate-level hard-delta table in appendix to support reproducibility and stop-loss governance.

## 7. Local Next Step

- Freeze diffusion extension line under current scope and move effort to paper-ready evidence packaging/reporting.

## 8. Governance Confirmation

- Decision confirmation date: `2026-04-15`
- Confirmation source: user instruction in current session (`继续推进确认`)
- Confirmed action: adopt **Suspend** as the official V4 closure decision.
