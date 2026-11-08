# PLANS_v2 M5 Stage Convergence (Baseline-First Closure)

## Evidence Block (Decision Record)

- run_id: `plansv2_m5_baseline_closure_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- config_snapshot: `N/A (decision synthesis from M1-M4 evidence docs)`
- dataset_version: `N/A (decision synthesis; no new dataset consumed)`
- dataset_hash: `N/A (decision synthesis; no new dataset consumed)`
- seeds: `42,43,44` (inherited from M1-M4 packs)
- eval_episodes_per_run: `N/A (fixed-step upstream protocol)`
- eval_env_steps_per_run: `256` (inherited from M1-M4 packs)
- primary_metrics:
  - `avg_reward` under unified protocol (`nominal/light_v2/hard`)
  - `avg_done_rate` as secondary support metric
- dispersion_metric: `std across seeds` (inherited from M1-M4 packs)
- artifact_table_paths:
  - `docs/plansv2_m1_baseline_pack.md`
  - `docs/plansv2_m2_gap_gate.md`
  - `docs/plansv2_m3_min_compare.md`
  - `docs/plansv2_m4_residual_compare_pack.md`
- artifact_log_paths:
  - `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`
  - `outputs/robustness_eval/plansv2_m2_gap_gate/{mode}_{condition}_s{seed}.log`
  - `outputs/robustness_eval/plansv2_m4_residual_compare_pack/{variant}_{condition}_s{seed}.log`
- artifact_checkpoint_paths:
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

## Decision

- selected_path: `non_diffusion_baseline_closure`
- decision_date: `2026-03-24`
- decision_source: `user confirmed Option A (baseline-first convergence)`

## S3 Decision Refresh (2026-03-25)

- stage_gate_status_local:
  - `G1=PASS`
  - `G2=PASS (local)`
  - `G3=PASS (minimal)`
  - `G4=PASS (local)`
- m5_closure_status: `finalized_local_under_plans_v2`
- followup_execution_boundary:
  - keep follow-up work in low-cost supporting-axis checks only (`baseline robustness/reporting consolidation`),
  - do not reopen diffusion mainline expansion unless governance is explicitly reopened.

## Stage Conclusion

- final_stage_direction: keep `padapt` as current executable mainline baseline; stop generative mainline expansion in this stage.
- latent_diffusion_status: `not_support_mainline` (artifact-backed from M3).
- residual_diffusion_status: `not_support_robust_edge` (artifact-backed from M4 robust compare).
- action_diffusion_status: remains `exploratory / appendix / baseline` (not stage mainline).

## Evidence Pointers

- M1 baseline hardening:
  - `docs/plansv2_m1_baseline_pack.md`
- M2 latent gap-closing gate:
  - `docs/plansv2_m2_gap_gate.md`
- M3 latent minimum credible comparison:
  - `docs/plansv2_m3_min_compare.md`
- M4 residual nominal/stabilization:
  - `docs/plansv2_m4_residual_gate_nominal.md`
  - `docs/plansv2_m4_residual_gate_nominal_scale05.md`
- M4 residual robustness compare:
  - `docs/plansv2_m4_residual_compare_pack.md`
- Canonical acceptance index:
  - `docs/stage_acceptance_summary.md`

## Why Action Diffusion Is Not Mainline

- Under the current Plan v2 stage definition, action diffusion is explicitly downgraded to exploratory/appendix.
- In this stage we prioritized latent-first and residual-fallback lines, and both failed to show robust advantage over current baseline.
- Therefore action diffusion stays as comparative context, not a primary execution route.

## Artifact-Backed Final Claims

- claim_1: latent diffusion is not competitive enough to remain stage mainline.
  - support: `docs/plansv2_m3_min_compare.md` (`local_m3_conclusion=not_support_latent_mainline`).
- claim_2: residual fallback does not demonstrate robust edge over padapt under current checkpoints/protocol.
  - support: `docs/plansv2_m4_residual_compare_pack.md` (`local_m4_conclusion=not_support_residual_robust_edge`).
- claim_3: baseline path remains the most reliable continuation path for thesis delivery.
  - support: M1 baseline pack + M4 comparisons in `docs/stage_acceptance_summary.md`.

## Next Stage Single Supporting Axis

- supporting_axis_only: `baseline robustness/reporting consolidation`
- scope:
  - finalize thesis-ready tables and narrative around teacher/padapt/purebc (+ diffusion as negative/inconclusive evidence),
  - keep only low-cost reproducibility checks,
  - avoid new diffusion architecture expansion unless governance is re-opened.

## One-line Conclusion

- support: M5 closure selects baseline-first convergence; generative mainline expansion is closed for the current stage.
