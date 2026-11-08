# PLANS_v2 M4 Residual Fallback Gate (Nominal Minimal Pack)

## Evidence Block

- run_id: `plansv2_m4_residual_gate_nominal_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- config_snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/config_032311_8bf90ec.yaml`
- seeds: `42,43,44`
- eval_steps_per_run: `256`
- protocol: `nominal`
- artifacts_root: `outputs/robustness_eval/plansv2_m4_residual_gate_nominal/`
- table_source: `docs/plansv2_m4_residual_gate_nominal.md`

## Residual Target And Scaling (Code Path)

- residual_target_definition: `target_x0 = e_gt - base_latent`, where `base_latent = tanh(adapt_tconv(proprio_hist))`.
- decode_rule: `pred_latent = tanh(x0_pred + base_latent)` when `diffusion_residual_base=True`.
- action_output_scaling: actor output is clamped to `[-1, 1]` before env step.

## Aggregated Nominal Results (Multiseed)

| Metric | Mean | Std |
|---|---:|---:|
| avg_reward | 2.046098 | 0.226010 |
| avg_done_rate | 0.001194 | 0.000307 |
| latent_mse | 0.072796 | 0.003256 |
| latent_l1 | 0.170207 | 0.004061 |
| action_mse_to_teacher | 0.134181 | 0.005050 |
| residual_abs_mean | 0.247690 | 0.001460 |
| residual_l2_mean | 2.297879 | 0.015312 |
| residual_to_target_ratio | 1.050885 | 0.002266 |
| action_correction_abs_mean | 0.235764 | 0.007813 |
| action_correction_l2_mean | 1.144949 | 0.031846 |
| base_action_mse_to_teacher | 0.259222 | 0.006016 |

## G3 Readiness Check (Local Execution Heuristic)

- sanity_reward_positive: `True`
- criteria: nominal aggregated reward is positive.
- residual_nonzero_signal: `True`
- criteria: residual_to_target_ratio > 0.02 and action_correction_abs_mean > 0.01.
- residual_not_explosive: `False`
- criteria: residual_to_target_ratio < 1.0 and action_correction_abs_mean < 1.0.
- local_g3_readiness_decision: `FAIL`

## One-line Conclusion

- not support: residual fallback branch is not yet stable/meaningful under nominal protocol.
