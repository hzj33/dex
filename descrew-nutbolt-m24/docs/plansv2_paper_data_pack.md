# PLANS_v2 Paper Data Pack

## Evidence Block

- run_id: `plansv2_paper_data_pack_2026-03-24`
- git_commit: `aabc11a77d9c785a6c44beff32d8ee73e5774fa2`
- sources:
  - `outputs/robustness_eval/plansv2_m1/`
  - `outputs/robustness_eval/plansv2_m2_gap_gate/`
  - `outputs/robustness_eval/plansv2_m4_residual_gate_nominal/`
  - `outputs/robustness_eval/plansv2_m4_residual_gate_nominal_scale05/`
  - `outputs/robustness_eval/plansv2_m4_residual_compare_pack/`
- data_tables:
  - `docs/data/plansv2_paper_seed_table.csv`
  - `docs/data/plansv2_paper_agg_table.csv`
  - `docs/data/plansv2_paper_delta_table.csv`
- summary_doc: `docs/plansv2_paper_data_pack.md`

## Coverage

- total_seed_rows: `69`
- total_aggregated_groups: `23`
- conditions_seen: `nominal, light_v2, hard`
- stages_seen: `M1, M2, M4`

## Key Thesis Numbers (Reward Mean ± 95% CI)

| Comparison | Condition | Value |
|---|---|---:|
| latent_diffusion | nominal | 2.062867 ± 0.130614 |
| padapt | nominal | 2.167820 ± 0.207004 |
| latent_diffusion | light_v2 | 1.788645 ± 0.307505 |
| padapt | light_v2 | 2.079074 ± 0.116158 |
| latent_diffusion | hard | 1.572475 ± 0.217396 |
| padapt | hard | 1.838225 ± 0.119337 |
| residual_scale05 | light_v2 | 1.711370 ± 0.052588 |
| residual_scale05 | hard | 1.548885 ± 0.050926 |

## Key Deltas

| Delta Group | Condition | LHS - RHS | Delta |
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

## One-line Conclusion

- support: this pack consolidates thesis-ready seed-level and aggregated evidence from M1-M4 without adding new training runs.
