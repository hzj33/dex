# PLANS_v5 M4 Final Verdict

Date: 2026-04-15  
Plan: `PLANS_v5.md`  
Decision: `Conclude` (no candidate passed single-seed gate)

## Scope Completion

- `V5-M0`: `ConsistencyLatentStudent` + `FlowMatchingLatentStudent` implemented, train/eval wiring completed, smoke verified.
- `V5-M1`: Consistency direction completed (`baseline`, `no_bc`, `tuned(step2)` + explicit eval-only 2-step validation).
- `V5-M2`: Flow direction completed to gate decision point (`flow_baseline`), follow-up candidates skipped per plan condition (candidate1 had no valid gate signal).
- `V5-M3`: skipped (precondition unmet: no candidate passed initial single-seed gate).
- `V5-M4`: final verdict consolidated in this document.

## Effective Candidate Results (seed=42, steps=256)

Primary reference (V3-M0 frozen): nominal `1.675112`, light_v2 `1.638266`, hard `1.504904`, hard_done `0.001872`.

| Candidate | nominal | light_v2 | hard | hard_done | delta_hard | delta_light_v2 | delta_done_hard | Gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| consistency_baseline | 2.201198 | 1.936892 | 1.714672 | 0.002686 | +0.209768 | +0.298626 | +0.000814 | FAIL (`hard_done`) |
| consistency_no_bc | 1.459489 | 1.346958 | 0.980692 | 0.002116 | -0.524212 | -0.291308 | +0.000244 | FAIL (`hard`,`light`) |
| consistency_infer2_evalonly | 1.896259 | 1.980752 | 1.520207 | 0.003337 | +0.015303 | +0.342486 | +0.001465 | FAIL (`hard_done`) |
| flow_baseline | 1.782656 | 1.587523 | 1.367413 | 0.002686 | -0.137491 | -0.050743 | +0.000814 | FAIL (`hard`,`hard_done`) |

## Key Evidence Notes

- `consistency_tuned_step2` retrain ckpt hash equals `consistency_baseline` hash:
  - baseline: `2209a84dc29d356c77275281085a30dfe8d36c91`
  - tuned_step2 retrain: `2209a84dc29d356c77275281085a30dfe8d36c91`
- Therefore, 2-step effect was validated with explicit eval override:
  - `+train.ppo.consistency_infer_steps=2`
  - outcome: light reward improved slightly, but hard reward dropped vs baseline and hard done regressed further.
- flow baseline hard delta is significantly negative (`-0.137491`), so flow candidate2/3 were not justified under the plan’s conditional expansion rule.

## Final Judgment

V5 introduced two new diffusion-family distillation forms, but **no candidate passed the V5 initial gate**.  
Hence within current scope the final verdict is:

1. `Accept`: NO
2. `Partial`: YES (consistency reward is stronger than prior DDPM family evidence)
3. `Conclude`: **YES (official execution conclusion for V5)** due to zero gate pass.

Practical execution decision: keep `PAdapt` as active student baseline/reference.

## Recommended Next Step

- Freeze V5 experimental expansion and move to reporting:
  - cite this file + `docs/session_handoff_v2.md` V5 entry
  - treat V5 as controlled negative/partial result evidence (reward gain without robustness done-rate compliance).
