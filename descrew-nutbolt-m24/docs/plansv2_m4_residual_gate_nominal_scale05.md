# PLANS_v2 M4 Residual Fallback Gate (Nominal Minimal Pack)

## Evidence Block

- run_id: `plansv2_m4_residual_gate_nominal_scale05_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- config_snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/config_032414_aabc11a.yaml`
- seeds: `42,43,44`
- eval_steps_per_run: `256`
- protocol: `nominal`
- artifacts_root: `outputs/robustness_eval/plansv2_m4_residual_gate_nominal_scale05`
- table_source: `docs/plansv2_m4_residual_gate_nominal_scale05.md`

## Residual Target And Scaling (Code Path)

- residual_target_definition: `target_x0 = (e_gt - base_latent) * diffusion_residual_target_scale`, where `base_latent = tanh(adapt_tconv(proprio_hist))`.
- decode_rule: `pred_latent = tanh((x0_pred / diffusion_residual_target_scale) + base_latent)` when `diffusion_residual_base=True`.
- action_output_scaling: actor output is clamped to `[-1, 1]` before env step.

## Aggregated Nominal Results (Multiseed)

| Metric | Mean | Std |
|---|---:|---:|
| avg_reward | 1.589266 | 0.063520 |
| avg_done_rate | 0.001410 | 0.000277 |
| latent_mse | 0.094090 | 0.002788 |
| latent_l1 | 0.199492 | 0.003170 |
| action_mse_to_teacher | 0.169102 | 0.008643 |
| residual_abs_mean | 0.250647 | 0.001743 |
| residual_l2_mean | 2.335321 | 0.010726 |
| residual_to_target_ratio | 1.049400 | 0.001607 |
| pred_residual_abs_mean | 0.093719 | 0.000807 |
| pred_to_target_ratio | 0.374187 | 0.003459 |
| action_correction_abs_mean | 0.171668 | 0.005947 |
| action_correction_l2_mean | 0.832797 | 0.023775 |
| base_action_mse_to_teacher | 0.273279 | 0.003731 |

## G3 Readiness Check (Local Execution Heuristic)

- sanity_reward_positive: `True`
- criteria: nominal aggregated reward is positive.
- residual_nonzero_signal: `True`
- criteria: residual_to_target_ratio > 0.02 and action_correction_abs_mean > 0.01.
- residual_not_explosive: `True`
- criteria: pred_to_target_ratio < 1.2 and action_correction_abs_mean < 0.6.
- local_g3_readiness_decision: `PASS`

## One-line Conclusion

- support: residual fallback branch is runnable with nontrivial correction signal under nominal protocol.
