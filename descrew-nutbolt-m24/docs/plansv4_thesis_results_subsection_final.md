# Thesis Results Subsection (Final, PLANS_v4)

## Experimental Protocol

All PLANS_v4 conclusions use a fixed-step evaluation protocol (`256` steps) with a unified single-seed gate (`seed=42`) under three conditions: `nominal`, `light_v2`, and `hard`. The primary gate is defined by deltas against the V3-M0 frozen reference (`nominal=1.675112`, `light_v2=1.638266`, `hard=1.504904`, `hard_done=0.001872`). As a secondary competitiveness signal, we also report deltas against the multiseed PAdapt baseline (`nominal=2.167820`, `light_v2=2.079074`, `hard=1.838225`).

## Main Findings

V3 produced six diffusion candidates across Mainline A/B, and none passed the unified single-seed gate. Because these V3 runs were executed before the double-tanh bug fix, V4 re-ran a bug-free baseline (`v4m0_bugfix_baseline`) using the same frozen configuration. The bug-free run still failed the gate and triggered the V4 hard-stop rule with `hard delta = -0.137635` (threshold: `< -0.10`).

## Clean Negative Evidence

The bug-free V4 baseline achieved `nominal=1.738804`, `light_v2=1.588133`, `hard=1.367269`, and `hard_done=0.002279`. Relative to V3-M0, the reward deltas are `+0.063692` (nominal), `-0.050133` (light_v2), and `-0.137635` (hard). Relative to PAdapt, all reward deltas remain negative (`nominal=-0.429016`, `light_v2=-0.490941`, `hard=-0.470956`). Therefore, diffusion does not establish robust competitiveness within the current scope.

## Final Decision and Positioning

PLANS_v4 is closed with **Suspend** (governance-confirmed). The thesis-facing position is:

1. Keep PAdapt as the practical baseline for the main results.
2. Report diffusion as a controlled negative result (V3 buggy phase + V4 bug-free confirmation).
3. Explicitly include the bug-fix rerun to show that the final conclusion is based on clean evidence rather than artifact contamination.

## Reproducible Artifact Pointers

- Verdict: `docs/plansv4_m3_final_verdict.md`
- Thesis bundle: `docs/plansv4_thesis_result_bundle.md`
- Candidate table: `docs/data/plansv4_thesis_candidate_table.csv`
- Delta table: `docs/data/plansv4_thesis_delta_table.csv`
- LaTeX table body: `docs/data/plansv4_thesis_tables.tex`

