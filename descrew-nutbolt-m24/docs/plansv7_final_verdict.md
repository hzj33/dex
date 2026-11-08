# PLANS_v7 Final Verdict

Date: 2026-04-16  
Plan: `PLANS_v7.md`  
Status: `completed_conclude`

## Current State

- `V7-M0`: completed
- `V7-M1`: completed
- `V7-M2`: completed
- `V7-M3`: not_triggered

## Locked Facts

- V7 corrected hard entry threshold:
  - `hard >= 1.733`
- Existing `v6_m3_ema_target` EMA-infer hard-only recheck:
  - reward `1.608308`
  - done `0.001628`
  - decision: fail V7 single-seed entry gate, do not promote directly
- `V7-M2` candidate A (`ema_obs_combo_seed42_15min`) full eval:
  - nominal `0.211057 / 0.001953`
  - light_v2 `0.296649 / 0.001709`
  - hard `0.355474 / 0.002279`
  - decision: fail by large margin; candidate D trigger condition not met
- `V7-M2` candidate B (`ema_target_seed42_30min`) full eval:
  - ckpt sha1 `1c7c75577496a00234b1007e89fb2715ea391d7c`
  - nominal `0.211057 / 0.001953`
  - light_v2 `0.296649 / 0.001709`
  - hard `0.355474 / 0.002279`
  - note: `model_best.ckpt` 与 candidate A 完全同 hash，30min EMA-only 未产生新的 best artifact
- `V7-M2` candidate C (`ema_alignfix_seed42_15min`) full eval:
  - ckpt sha1 `024409e9f765232775f33f139493e04c09ae29d2`
  - nominal `1.124038 / 0.002035`
  - light_v2 `1.177704 / 0.002279`
  - hard `1.135807 / 0.001953`
  - decision: train/infer align + EMA infer 修复后，训练信号有改善，但 deploy reward 仍明显低于 gate

## Final Verdict

- `PLANS_v7 = completed_conclude`
- `single-seed entry pass count = 0`
- `candidate D = not_triggered`
- `multiseed M3 = not_triggered`

## Why The Conclusion Holds

- M1 已补齐 EMA 推理语义，但已有最佳 `v6_m3_ema_target` 在 corrected hard gate 下仍只有 `1.608308`
- M2-A 与 M2-B 都停在同一个 best checkpoint，说明仅增加 EMA-only 训练预算没有带来新的可部署最优点
- M2-C 是本轮唯一明显不同的新 artifact，但三条件 reward 仍为：
  - nominal `1.124038`
  - light_v2 `1.177704`
  - hard `1.135807`
- 因此 A/B/C 全部未达到：
  - `hard >= 1.733`
  - `nominal >= 2.100`
  - `light_v2 >= 1.900`

## Baseline Positioning After V7

- keep_mainline_baseline: `padapt`
- strongest_accepted_consistency_reference: `V5.5 consistency_boundary_bc_tuned`
- V7_local_positioning: `bounded_followup_completed_but_no_single_seed_breakthrough`

## Governance Note

- `V6 Conclude` 不被推翻；`V7` 只是补齐剩余实现/评测风险并做有界后续优化。
- 在 `V7` 收口后，Consistency 仍未越过 hard single-seed 门槛，也未触发 multiseed primary accept。
