# PLANS_v2 M2 Latent Gap-Closing Gate Pack

## Evidence Block

- run_id: `plansv2_m2_gap_gate_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- config_snapshot: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/config_032315_8bf90ec.yaml`
- dataset_version: `N/A (online environment evaluation; no rollout dataset used)`
- dataset_hash: `N/A (online environment evaluation; no rollout dataset used)`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- seeds: `42,43,44`
- eval_episodes_per_run: `N/A (fixed-step protocol)`
- eval_env_steps_per_run: `256`
- protocol: `nominal + light_v2 + hard`
- modes: `diffusion`, `decode_only`
- primary_metrics:
  - `avg_reward` (main)
  - `avg_done_rate`, `latent_mse`, `latent_l1`, `action_mse_to_teacher`
- dispersion_metric: `std across seeds`
- artifacts_root: `outputs/robustness_eval/plansv2_m2_gap_gate/`
- table_source: `docs/plansv2_m2_gap_gate.md`
- artifact_log_paths: `outputs/robustness_eval/plansv2_m2_gap_gate/{mode}_{condition}_s{seed}.log`
- artifact_checkpoint_paths:
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

## Aggregated Results

| Mode | Condition | Reward Mean | Reward Std | Done Mean | Done Std | Latent MSE Mean | Latent L1 Mean | Action MSE Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion | nominal | 2.062867 | 0.115423 | 0.001221 | 0.000133 | 0.080392 | 0.176390 | 0.146311 |
| diffusion | light_v2 | 1.788645 | 0.271742 | 0.001600 | 0.000595 | 0.080991 | 0.175863 | 0.151461 |
| diffusion | hard | 1.572475 | 0.192113 | 0.001845 | 0.000366 | 0.086572 | 0.179890 | 0.168924 |
| decode_only | nominal | 0.909722 | 0.138378 | 0.002089 | 0.000307 | 0.139231 | 0.250919 | 0.262430 |
| decode_only | light_v2 | 0.915083 | 0.074780 | 0.002143 | 0.000233 | 0.140678 | 0.252794 | 0.264398 |
| decode_only | hard | 0.700938 | 0.112248 | 0.002659 | 0.000378 | 0.144016 | 0.257412 | 0.267246 |

## G2 Gate Decision (Local Execution Heuristic)

- reconstruction_reported: `True`
- criteria: `EvalReconSummary` exists for every run, with finite latent/action errors.
- decode_only_stability: `True`
- criteria: decode-only aggregated reward is positive under nominal/light_v2/hard.
- local_G2_decision: `PASS`

## One-line Conclusion

- support: M2 gate evidence is sufficient locally (`PASS`) under the current representative ckpt and protocol.
