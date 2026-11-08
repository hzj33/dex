# PLANS_v2 Thesis Result Bundle

## Evidence Block

- run_id: `plansv2_thesis_result_bundle_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- derived_from:
  - `docs/plansv2_paper_data_pack.md`
  - `docs/data/plansv2_paper_agg_table.csv`
  - `docs/data/plansv2_paper_delta_table.csv`
- outputs:
  - `docs/data/plansv2_thesis_main_table.csv`
  - `docs/data/plansv2_thesis_negative_delta_table.csv`
  - `docs/data/plansv2_thesis_tables.tex`
  - `docs/plansv2_thesis_result_bundle.md`

## Main Table (Reward Mean ± Std)

| Algorithm | Nominal | Light_v2 | Hard | Role |
|---|---:|---:|---:|---|
| teacher_ppo | 3.055567 +/- 0.133547 | 2.915410 +/- 0.157431 | 2.762886 +/- 0.172094 | upper_bound_teacher |
| padapt | 2.167820 +/- 0.182929 | 2.079074 +/- 0.102648 | 1.838225 +/- 0.105458 | selected_mainline_baseline |
| purebc | 1.882908 +/- 0.437617 | 2.190508 +/- 0.123747 | 1.849765 +/- 0.033950 | minimal_student_baseline |
| latent_diffusion | 2.062867 +/- 0.115423 | 1.788645 +/- 0.271742 | 1.572475 +/- 0.192113 | diffusion_candidate_latent |
| residual_unscaled | 2.046098 +/- 0.226010 | 1.767419 +/- 0.193093 | 1.427755 +/- 0.125767 | diffusion_candidate_residual |
| residual_scale05 | 1.589266 +/- 0.063520 | 1.711370 +/- 0.046472 | 1.548885 +/- 0.045004 | diffusion_candidate_residual_scaled |

## Negative-Result Delta Table

| Group | Condition | LHS - RHS | Delta (Reward) |
|---|---|---|---:|
| m3_latent_vs_baselines | nominal | latent_diffusion - padapt | -0.104952 |
| m3_latent_vs_baselines | nominal | latent_diffusion - purebc | 0.179959 |
| m3_latent_vs_baselines | light_v2 | latent_diffusion - padapt | -0.290429 |
| m3_latent_vs_baselines | light_v2 | latent_diffusion - purebc | -0.401863 |
| m3_latent_vs_baselines | hard | latent_diffusion - padapt | -0.265750 |
| m3_latent_vs_baselines | hard | latent_diffusion - purebc | -0.277290 |
| m4_residual_compare | light_v2 | residual_scale05 - residual_unscaled | -0.056049 |
| m4_residual_compare | light_v2 | residual_scale05 - padapt | -0.367704 |
| m4_residual_compare | light_v2 | residual_unscaled - padapt | -0.311655 |
| m4_residual_compare | hard | residual_scale05 - residual_unscaled | 0.121130 |
| m4_residual_compare | hard | residual_scale05 - padapt | -0.289340 |
| m4_residual_compare | hard | residual_unscaled - padapt | -0.410469 |

## Thesis Narrative (Concise Draft)

- 在统一 protocol（nominal/light_v2/hard，多 seed）下，`padapt` 在核心鲁棒条件上持续优于当前 diffusion 候选。
- `latent_diffusion` 在 nominal 与 `purebc` 可比，但在 `light_v2/hard` 下未形成相对 `padapt` 的优势。
- `residual` 路线（含 `scale05`）在局部稳定性指标上有可解释变化，但 robust reward 仍未超过 `padapt`。
- 因此本阶段采用 baseline-first closure：保留 diffusion 作为方法学探索与负/中性证据，主交付聚焦可复现基线结论。

## One-line Conclusion

- support: thesis-facing tables and concise narrative are now frozen from artifact-backed M1-M4 evidence.
