# PLANS_v5_5 Final Verdict

Date: 2026-04-15  
Plan: `PLANS_v5_5.md`  
Decision: `Accept` (Consistency candidate C passes single-seed + multiseed acceptance)

## Scope and Execution

- Route: `ConsistencyLatentStudent` only.
- Budget used:
  - M1 train candidates: 3/3 (`A`, `B`, `C`)
  - M2 multiseed: top-1 candidate (`C`) on seeds `42/43/44`
- Unified eval protocol: `nominal + light_v2 + hard`, `steps=256`.

## M1 Candidate Outcomes (seed=42)

Primary reference (V3-M0 frozen): nominal `1.675112`, light_v2 `1.638266`, hard `1.504904`, hard_done ref `0.001872`.

Consistency baseline anchor: nominal `2.201198`, light_v2 `1.936892`, hard `1.714672`, hard_done `0.002686`.

| Candidate | key overrides | nominal | light_v2 | hard | hard_done | Primary gate | Anti-regression |
|---|---|---:|---:|---:|---:|---|---|
| A `action_l2_stable` | `action_l2=1e-3` | 2.052454 | 1.912768 | 1.531204 | 0.001383 | PASS | FAIL (`hard_reward_vs_base=-0.183468 < -0.15`) |
| B `anchor_l2_combo` | `action_l2=5e-4`, `base_action_anchor_coef=0.03` | 1.774299 | 1.534575 | 1.552297 | 0.001546 | FAIL (`light_v2`) | FAIL |
| C `boundary_bc_tuned` | `boundary=0.8`, `num_scales=16`, `bc=1.2` | 2.221939 | 1.902549 | 1.639399 | 0.002035 | PASS | PASS |

Key note:
- Candidate C is the only candidate that passes both primary gate and anti-regression guardrails.

## M2 Multiseed Validation (Candidate C)

Candidate ckpt:
- `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/stage2_consistency_nn/model_best.ckpt`
- sha1: `45d785763f385bfb0a5326866eb31c9cbcbb3215`

Per-seed reward/done (steps=256):

| Seed | nominal | light_v2 | hard |
|---|---|---|---|
| 42 | `2.221939 / 0.000732` | `1.902549 / 0.001546` | `1.639399 / 0.002035` |
| 43 | `2.477065 / 0.001139` | `2.103719 / 0.001058` | `1.820255 / 0.001221` |
| 44 | `2.309848 / 0.000977` | `2.127907 / 0.000977` | `1.683760 / 0.001546` |

Aggregate:
- nominal mean: `2.336284` (delta vs V3-M0: `+0.661172`)
- light_v2 mean: `2.044725` (delta vs V3-M0: `+0.406459`)
- hard mean: `1.714471` (delta vs V3-M0: `+0.209567`)
- hard done mean: `0.001601` (threshold: `<=0.002372`)

Acceptance checks:
- `hard_mean_delta >= 0`: PASS
- `light_v2_mean_delta >= 0`: PASS
- `nominal_mean_delta >= -0.10`: PASS
- `hard_done_mean <= 0.002372`: PASS

## Final Decision

`PLANS_v5_5` verdict: **Accept**.

Accepted candidate:
- `ConsistencyLatentStudent` with:
  - `+train.ppo.consistency_boundary_coef=0.8`
  - `+train.ppo.consistency_num_scales=16`
  - `+train.ppo.bc_loss_coef=1.2`

Practical conclusion:
- This candidate resolves the V5 hard_done bottleneck while preserving overall reward competitiveness.

## Recommended Next Step

- Promote candidate C as new consistency reference and run one reproducibility refresh pass (same config, fresh run id) to lock paper/report artifacts.
