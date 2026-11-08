# PLANS_v2 M4 Residual Robustness Compare Pack

## Evidence Block

- run_id: `plansv2_m4_residual_compare_pack_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- config_snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/config_032311_8bf90ec.yaml`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/config_032414_aabc11a.yaml`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`
- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`
- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`
- seeds: `42,43,44`
- eval_episodes_per_run: `N/A (fixed-step protocol)`
- eval_env_steps_per_run: `256`
- protocol: `light_v2 + hard`
- variants: `residual_unscaled`, `residual_scale05`, `padapt`
- primary_metrics:
  - `avg_reward` (main compare axis)
  - `avg_done_rate`, `CorrAbs Mean`, `Pred/Target Ratio Mean`
- dispersion_metric: `std across seeds`
- ckpt_residual_unscaled: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- ckpt_residual_scale05: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- ckpt_padapt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
- artifacts_root: `outputs/robustness_eval/plansv2_m4_residual_compare_pack`
- table_source: `docs/plansv2_m4_residual_compare_pack.md`
- artifact_log_paths: `outputs/robustness_eval/plansv2_m4_residual_compare_pack/{variant}_{condition}_s{seed}.log`
- artifact_checkpoint_paths:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`

## Unified Robustness Table

| Variant | Condition | Reward Mean | Reward Std | Done Mean | Done Std | CorrAbs Mean | Pred/Target Ratio Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| residual_unscaled | light_v2 | 1.767419 | 0.193093 | 0.001356 | 0.000138 | 0.237466 | 0.667432 |
| residual_unscaled | hard | 1.427755 | 0.125767 | 0.001845 | 0.000139 | 0.239002 | 0.665035 |
| residual_scale05 | light_v2 | 1.711370 | 0.046472 | 0.001546 | 0.000199 | 0.296494 | 0.778173 |
| residual_scale05 | hard | 1.548885 | 0.045004 | 0.001628 | 0.000305 | 0.301035 | 0.769964 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 | N/A | N/A |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 | N/A | N/A |

## Reward Delta Table

| Condition | scale05 - unscaled | scale05 - padapt | unscaled - padapt |
|---|---:|---:|---:|
| light_v2 | -0.056049 | -0.367704 | -0.311655 |
| hard | 0.121130 | -0.289340 | -0.410469 |

## Local M4 Conclusion

- scale05_beats_unscaled_all_robust: `False`
- best_residual_beats_padapt_any_robust: `False`
- local_m4_conclusion: `not_support_residual_robust_edge`

## One-line Conclusion

- not support: residual fallback does not show robustness edge under current checkpoints and protocol.
