# PLANS_v5_5d Summary

Date: 2026-04-16  
Scope: summary-only document for next-step optimization alignment (no new train/eval run in this session).

## 1. Evidence Sources

- `PLANS_v5_5.md`
- `docs/plansv5_5_final_verdict.md`
- `docs/stage_acceptance_summary.md` (`PLANS_v5_5 Closure Snapshot`, historical `padapt/latent_diffusion` baselines)
- `docs/session_handoff_v2.md` (`v2-093`)

## 2. Current Position After V5.5 Accept

Accepted diffusion candidate:
- `ConsistencyLatentStudent`
- overrides:
  - `+train.ppo.consistency_boundary_coef=0.8`
  - `+train.ppo.consistency_num_scales=16`
  - `+train.ppo.bc_loss_coef=1.2`
- ckpt:
  - `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - sha1 `45d785763f385bfb0a5326866eb31c9cbcbb3215`

Unified comparison protocol: seeds `42/43/44`, `steps=256`, `nominal + light_v2 + hard`.

## 3. Side-by-Side Metrics (Multiseed Mean)

| Method | nominal reward / done | light_v2 reward / done | hard reward / done |
|---|---|---|---|
| consistency (V5.5 accepted) | `2.336284 / 0.000949` | `2.044725 / 0.001194` | `1.714471 / 0.001601` |
| latent_diffusion (V3 family reference) | `2.062867 / 0.001221` | `1.788645 / 0.001600` | `1.572475 / 0.001845` |
| padapt (mainline baseline) | `2.167820 / 0.001302` | `2.079074 / 0.001221` | `1.838225 / 0.001411` |

## 4. Delta Summary (Consistency V5.5 - Reference)

### 4.1 vs latent_diffusion

- nominal reward: `+0.273417`
- light_v2 reward: `+0.256080`
- hard reward: `+0.141996`
- hard done: `-0.000244` (better)

Conclusion: consistency is clearly ahead of previous latent reference on both reward and hard done.

### 4.2 vs padapt

- nominal reward: `+0.168464`
- light_v2 reward: `-0.034349`
- hard reward: `-0.123754`
- hard done: `+0.000190` (worse)

Conclusion: consistency is **not yet** fully ahead of padapt.  
Main gap is concentrated in robust condition (`hard`), with a small remaining gap in `light_v2`.

## 5. What V5.5 Solved and What Remains

Solved:
- Cleared V5.5 acceptance gates and removed V5-era hard_done bottleneck under plan-defined threshold.
- Established a stronger diffusion reference (`consistency_boundary_bc_tuned`) for follow-up work.

Not solved:
- Did not surpass `padapt` on all core robust metrics.
- `hard` condition remains the primary optimization bottleneck.

## 6. V5.5d-Oriented Optimization Focus (Summary-Level)

For the next iteration, optimization should be hard-focused and anti-regression constrained:

- Priority-1: recover `hard` reward while keeping `hard_done` near current accepted level.
- Priority-2: close the small `light_v2` reward gap without sacrificing `hard`.
- Constraint: keep nominal/light/hard done from regressing while pushing robust rewards upward.

Practical implication:
- Continue from V5.5 accepted configuration as the only reference branch.
- Use bounded local tuning (small coefficient moves) instead of architecture switching or major refactor.

## 7. Final Summary

- Relative to old latent diffusion reference: V5.5 consistency is comprehensively better.
- Relative to padapt: V5.5 consistency is competitive but not yet superior on robust metrics.
- Therefore, V5.5d is justified as a targeted robustness-closing phase, not a paradigm reset.
