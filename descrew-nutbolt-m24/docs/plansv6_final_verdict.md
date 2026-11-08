# PLANS_v6 Final Verdict

Date: 2026-04-16  
Plan: `PLANS_v6.md`  
Decision: `Conclude` (no candidate reaches V6 hard-threshold for multiseed entry)

## Scope and Protocol

- Route: `ConsistencyLatentStudent` only.
- Baseline startpoint: V5.5 accepted candidate (`boundary=0.8, num_scales=16, bc=1.2`).
- Unified eval protocol: `seed=42`, `steps=256`, `nominal + light_v2 + hard`.
- V6 key single-seed stop threshold before multiseed (`M4`) entry:
  - `hard_reward >= 1.780` (`PAdapt hard mean - 1σ`)

## M0 Baseline Lock

- V5.5 accepted ckpt:
  - `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/stage2_consistency_nn/model_best.ckpt`
  - sha1: `45d785763f385bfb0a5326866eb31c9cbcbb3215`
- PAdapt target (from acceptance summary):
  - nominal `2.167820`, light_v2 `2.079074`, hard `1.838225`

## M1 Probes (seed=42)

| Probe | Key override | nominal (r/d) | light_v2 (r/d) | hard (r/d) | hard >= 1.780 |
|---|---|---|---|---|---|
| `capacity_boost` | `consistency_hidden_dim=512` | `1.451814 / 0.001383` | `1.668441 / 0.000977` | `1.255473 / 0.001628` | FAIL |
| `longer_train` | 30min budget | `2.064455 / 0.000977` | `1.858506 / 0.001546` | `1.415326 / 0.001953` | FAIL |
| `lr_schedule` | `consistency_lr=1e-4` | `2.291926 / 0.000814` | `2.055887 / 0.001058` | `1.335743 / 0.001628` | FAIL |

M1 decision:
- All probes fail hard-threshold and all have `hard_delta_vs_v55_seed42 < +0.02`.
- Triggered plan stop rule: skip `M2`, enter `M3`.

## M3 Code-Level Candidates (seed=42)

### Code changes introduced

- File: `dexscrew/algo/ppo/consistency_latent_student.py`
- Added options:
  - `consistency_obs_noise_curriculum*` (training-time obs-noise curriculum)
  - `consistency_train_align_infer` (train-time latent path aligned with infer-steps rollout)
  - `consistency_use_ema_target` + `consistency_ema_decay` (EMA target model for consistency low-step target)

### Candidate outcomes

| Candidate | Key idea | nominal (r/d) | light_v2 (r/d) | hard (r/d) | hard >= 1.780 |
|---|---|---|---|---|---|
| `obs_noise_curriculum` | direction A | `2.022649 / 0.001221` | `1.830568 / 0.001953` | `1.387970 / 0.002441` | FAIL |
| `infer2_align` | direction B | `1.980866 / 0.001058` | `1.786391 / 0.001546` | `1.622737 / 0.001302` | FAIL |
| `ema_target` | direction C | `1.736832 / 0.001383` | `2.077133 / 0.001058` | `1.723353 / 0.002116` | FAIL |

M3 decision:
- Best hard reward is `1.723353` (`ema_target`), still below `1.780`.
- Therefore all M3 directions fail V6 stop threshold.

## Final Decision

`PLANS_v6` verdict: **Conclude**.

- No candidate reaches the hard-threshold required to enter multiseed `M4`.
- V6 target “Consistency全面逼近并超越 PAdapt” is not achieved in this cycle.
- Mainline baseline remains `padapt`.
- Consistency remains a competitive secondary branch in selected conditions (e.g., nominal or light_v2 for some candidates), but robust hard performance is still short of replacement standard.

## Artifacts

- Train logs: `outputs/v6_logs/`
- Eval logs:
  - `outputs/robustness_eval/v6_m1_capacity_boost_seed42/`
  - `outputs/robustness_eval/v6_m1_longer_train_seed42/`
  - `outputs/robustness_eval/v6_m1_lr_schedule_seed42/`
  - `outputs/robustness_eval/v6_m3_obs_noise_curriculum_seed42/`
  - `outputs/robustness_eval/v6_m3_infer2_align_seed42/`
  - `outputs/robustness_eval/v6_m3_ema_target_seed42/`

