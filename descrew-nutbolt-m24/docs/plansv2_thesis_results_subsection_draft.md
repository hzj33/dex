# Thesis Results Subsection (Draft, PLANS_v2)

## Experimental Protocol

All comparisons use the same fixed-step evaluation protocol (`256` steps per run) and the same multiseed setting (`seed=42,43,44`). We report nominal and robustness conditions (`light_v2`, `hard`) under the Hora screwdriver task. The canonical evidence tables are generated from artifact-backed logs and aggregated in `docs/plansv2_thesis_result_bundle.md`.

## Main Results

The teacher policy remains the upper bound (`3.055567 +/- 0.133547` nominal; `2.915410 +/- 0.157431` light_v2; `2.762886 +/- 0.172094` hard). Among student-level candidates, `padapt` is the most stable mainline baseline (`2.167820 +/- 0.182929` nominal; `2.079074 +/- 0.102648` light_v2; `1.838225 +/- 0.105458` hard). `purebc` is competitive in parts of the protocol but shows larger variance and does not dominate `padapt` consistently.

For diffusion variants, `latent_diffusion` reaches reasonable nominal performance (`2.062867 +/- 0.115423`) but does not outperform `padapt` in robustness conditions (`light_v2: 1.788645 +/- 0.271742`, `hard: 1.572475 +/- 0.192113`). Residual variants also fail to establish robust superiority over `padapt` under the current checkpoints and budget (`residual_unscaled hard: 1.427755 +/- 0.125767`; `residual_scale05 hard: 1.548885 +/- 0.045004`; both below `padapt` hard `1.838225 +/- 0.105458`).

## Negative/Inconclusive Diffusion Evidence

Key deltas support the same conclusion. In M3, `latent_diffusion - padapt` is negative under all conditions (nominal `-0.104952`, light_v2 `-0.290429`, hard `-0.265750`). In M4 robust comparison, residual variants remain below `padapt` (`residual_scale05 - padapt`: light_v2 `-0.367704`, hard `-0.289340`; `residual_unscaled - padapt`: light_v2 `-0.311655`, hard `-0.410469`).

## Stage Decision And Thesis Positioning

Given artifact-backed M1-M4 evidence, this stage adopts baseline-first closure: `padapt` is kept as the executable mainline baseline, while diffusion routes are retained as method exploration and negative/inconclusive evidence. This positioning still contributes to the thesis by providing a reproducible comparison pipeline, explicit gate-based decision logic, and transparent negative-result reporting rather than unsupported performance claims.

## Artifact Pointers

- main bundle: `docs/plansv2_thesis_result_bundle.md`
- main table CSV: `docs/data/plansv2_thesis_main_table.csv`
- negative delta CSV: `docs/data/plansv2_thesis_negative_delta_table.csv`
- LaTeX tables: `docs/data/plansv2_thesis_tables.tex`
