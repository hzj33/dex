# Thesis Results Subsection (Draft, PLANS_v4)

## Experimental Protocol

All V4 claims are based on fixed-step evaluation (`256` steps) with a unified single-seed gate (`seed=42`) under three conditions: `nominal`, `light_v2`, and `hard`. The primary gate uses deltas against the V3-M0 frozen reference (`nominal=1.675112`, `light_v2=1.638266`, `hard=1.504904`, `hard_done=0.001872`). Secondary comparison is reported against the multiseed PAdapt baseline (`nominal=2.167820`, `light_v2=2.079074`, `hard=1.838225`).

## Main Findings

V3 produced six diffusion candidates (Mainline A/B) and none passed the single-seed gate. Because all V3 candidates were trained on buggy code, V4 re-ran a bug-free baseline (`v4m0_bugfix_baseline`) with the same frozen configuration. The bug-free run also failed the gate, with `hard delta = -0.137635`, and triggered the V4 hard-stop rule (`hard delta < -0.10`).

## Negative Result (Clean Evidence)

The bug-free V4 baseline shows `nominal=1.738804`, `light_v2=1.588133`, `hard=1.367269`, and `hard_done=0.002279`. Relative to V3-M0, this is `+0.063692` nominal, `-0.050133` light_v2, and `-0.137635` hard. Relative to PAdapt, all reward deltas remain negative (`nominal=-0.429016`, `light_v2=-0.490941`, `hard=-0.470956`). This confirms that diffusion did not establish robust competitiveness under the current scope.

## Stage Decision and Thesis Positioning

PLANS_v4 is closed with **Suspend** (governance-confirmed). The thesis-facing positioning is: keep PAdapt as the practical baseline, report diffusion as a controlled negative result, and explicitly document the bug-fix rerun to show that the final conclusion is based on clean evidence rather than buggy training artifacts.

## Artifact Pointers

- verdict doc: `docs/plansv4_m3_final_verdict.md`
- result bundle: `docs/plansv4_thesis_result_bundle.md`
- candidate table CSV: `docs/data/plansv4_thesis_candidate_table.csv`
- delta table CSV: `docs/data/plansv4_thesis_delta_table.csv`
- LaTeX table: `docs/data/plansv4_thesis_tables.tex`
