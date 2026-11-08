# PLANS_v8 Final Verdict

Date: 2026-04-17  
Plan: `PLANS_v8.md`  
Status: `completed_conclude`

## Final State

- `V8-M0`: completed
- `V8-M1`: completed
- `V8-M2`: completed
- `V8-M2.5`: not triggered
- `V8-M3`: not triggered

## Locked Facts

- old `v5_m2_flow_baseline` remains numerically stable under the new M0 code path:
  - nominal `1.782656 / 0.001383`
  - light_v2 `1.587523 / 0.002116`
  - hard `1.367413 / 0.002686`
- `V8-M1 infer2` eval-only probe:
  - nominal `1.805502 / 0.001546`
  - light_v2 `1.681982 / 0.001953`
  - hard `1.428555 / 0.002604`
- `V8-M1 infer4` eval-only probe:
  - nominal `1.885794 / 0.001302`
  - light_v2 `1.640372 / 0.002523`
  - hard `1.189338 / 0.002930`
- interpretation:
  - `infer2` is the stronger overall multistep direction
  - `infer4` improves nominal only, but hurts robust conditions

## M2 Fresh Sweep

- `candidate 1` `v8_m2_flow_zeroinit_bc12_seed42_15min`:
  - nominal `0.992769 / 0.001872`
  - light_v2 `0.921589 / 0.002523`
  - hard `1.020256 / 0.002604`
  - rejected
- `candidate 2` `v8_m2_flow_align2_rollout_seed42_15min`:
  - nominal `1.082436 / 0.002035`
  - light_v2 `1.108353 / 0.002116`
  - hard `0.750190 / 0.003174`
  - rejected
- `candidate 4` `v8_m2_flow_align2_anchor_seed42_15min`:
  - nominal `1.000670 / 0.002035`
  - light_v2 `0.923009 / 0.002360`
  - hard `0.953265 / 0.002686`
  - rejected
- `candidate 5` `v8_m2_flow_align2_bcheavy_seed42_15min`:
  - nominal `1.124580 / 0.001872`
  - light_v2 `1.055061 / 0.002441`
  - hard `1.053348 / 0.002279`
  - rejected
- `candidate 3` `v8_m2_flow_align4_rollout_seed42_15min`:
  - nominal `1.114895 / 0.001790`
  - light_v2 `1.152107 / 0.001872`
  - hard `1.079496 / 0.002848`
  - best fresh candidate, but still rejected

## Final Verdict

- no seed42 candidate reached the V8 single-seed continue gate:
  - `nominal >= 1.95`
  - `light_v2 >= 1.70`
  - `hard >= 1.55`
  - `hard_done <= 0.002372`
- `V8-M2.5` 30min extension is not justified:
  - best fresh candidate `align4_rollout` fails every pre-threshold reward requirement
- `V8-M3` multiseed is not justified:
  - no seed42 candidate earned a continue-worthy signal
- conclusion:
  - `FlowMatchingLatentStudent` remains below the current flow baseline even after the V8 recovery sprint
  - `padapt` remains the mainline baseline
  - `V5.5 consistency_boundary_bc_tuned` remains the strongest accepted diffusion reference
  - `flow` is not continue-worthy under the current V8 scope
