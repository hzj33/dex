# PLANS_v2 M1 Baseline Pack (Multiseed)

## Evidence Block

- run_id: `plansv2_m1_baseline_pack_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- config_snapshot:
  - `outputs/XHandHoraScrewDriver_teacher/run_a/config_031916_f5f9edb.yaml`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/config_032010_8bf90ec.yaml`
  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/config_032011_8bf90ec.yaml`
- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`
- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`
- seeds: `42,43,44`
- eval_episodes_per_run: `N/A (fixed-step protocol)`
- eval_env_steps_per_run: `256`
- protocol: `nominal + light_v2 + hard`
- primary_metrics:
  - `avg_reward` (main)
  - `avg_done_rate` (secondary)
- dispersion_metric: `std across seeds`
- artifacts_root: `outputs/robustness_eval/plansv2_m1/`
- table_source: `docs/plansv2_m1_baseline_pack.md`
- artifact_log_paths: `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`
- artifact_checkpoint_paths:
  - `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`

## Aggregated Results

| Algorithm | Condition | Reward Mean | Reward Std | Done Mean | Done Std |
|---|---|---:|---:|---:|---:|
| teacher_ppo | nominal | 3.055567 | 0.133547 | 0.000733 | 0.000240 |
| teacher_ppo | light_v2 | 2.915410 | 0.157431 | 0.000787 | 0.000139 |
| teacher_ppo | hard | 2.762886 | 0.172094 | 0.000814 | 0.000239 |
| padapt | nominal | 2.167820 | 0.182929 | 0.001302 | 0.000115 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 |
| purebc | nominal | 1.882908 | 0.437617 | 0.001302 | 0.000199 |
| purebc | light_v2 | 2.190508 | 0.123747 | 0.001329 | 0.000214 |
| purebc | hard | 1.849765 | 0.033950 | 0.001411 | 0.000307 |

## Artifact Pointers

- logs: `outputs/robustness_eval/plansv2_m1/{algo}_{condition}_s{seed}.log`
- teacher_ckpt: `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
- padapt_ckpt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
- purebc_ckpt: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`

## One-line Conclusion

- support: M1 baseline pack now has multiseed, unified-protocol evidence for `teacher_ppo`, `padapt`, and `purebc`.
