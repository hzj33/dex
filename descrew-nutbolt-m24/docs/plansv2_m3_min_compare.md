# PLANS_v2 M3 Minimum Credible Comparison

## Evidence Block

- run_id: `plansv2_m3_min_compare_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- config_snapshot:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/config_032315_8bf90ec.yaml`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`
  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/config_032011_8bf90ec.yaml`
- dataset_version: `N/A (aggregated from online-eval logs; no rollout dataset used)`
- dataset_hash: `N/A (aggregated from online-eval logs; no rollout dataset used)`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- seeds: `42,43,44`
- eval_episodes_per_run: `N/A (fixed-step protocol in upstream packs)`
- eval_env_steps_per_run: `256`
- protocol: `nominal + light_v2 + hard`
- upstream_evidence_m1: `docs/plansv2_m1_baseline_pack.md`
- upstream_evidence_m2: `docs/plansv2_m2_gap_gate.md`
- artifacts_root_m1: `outputs/robustness_eval/plansv2_m1/`
- artifacts_root_m2: `outputs/robustness_eval/plansv2_m2_gap_gate/`
- primary_metrics:
  - `avg_reward` (main compare axis)
  - `avg_done_rate` (secondary)
- dispersion_metric: `std across seeds`
- table_source: `docs/plansv2_m3_min_compare.md`
- artifact_log_paths:
  - `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`
  - `outputs/robustness_eval/plansv2_m2_gap_gate/{mode}_{condition}_s{seed}.log`
- artifact_checkpoint_paths:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`

## Unified Comparison Table

| Algorithm | Condition | Reward Mean | Reward Std | Done Mean | Done Std |
|---|---|---:|---:|---:|---:|
| latent_diffusion | nominal | 2.062867 | 0.115423 | 0.001221 | 0.000133 |
| latent_diffusion | light_v2 | 1.788645 | 0.271742 | 0.001600 | 0.000595 |
| latent_diffusion | hard | 1.572475 | 0.192113 | 0.001845 | 0.000366 |
| padapt | nominal | 2.167820 | 0.182929 | 0.001302 | 0.000115 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 |
| purebc | nominal | 1.882908 | 0.437617 | 0.001302 | 0.000199 |
| purebc | light_v2 | 2.190508 | 0.123747 | 0.001329 | 0.000214 |
| purebc | hard | 1.849765 | 0.033950 | 0.001411 | 0.000307 |

## Reward Delta (latent diffusion as anchor)

| Condition | latent - padapt | latent - purebc |
|---|---:|---:|
| nominal | -0.104952 | 0.179959 |
| light_v2 | -0.290429 | -0.401863 |
| hard | -0.265750 | -0.277290 |

## Local M3 Decision

- latent_beats_padapt_all_conditions: `False`
- latent_beats_purebc_on_robust_conditions: `False`
- local_m3_conclusion: `not_support_latent_mainline`
- local_next_step_recommendation: `prepare_m4_residual_fallback_gate`

## One-line Conclusion

- not support: latent diffusion does not show robust multiseed advantage over current student baselines, so M4 residual fallback preparation should be started.
