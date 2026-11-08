# Metric Card G1: Residual Consistency Latent Over PAdapt

Date: 2026-06-01

Repo and branch:

```text
/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro-public
branch: diffusion
```

## Scope

This card starts the first local diffusion optimization round defined in
`Goal/diffusion_goal_plan.md`.

Canonical task:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
```

Canonical PPO teacher:

```text
sim2real/codrive/best_reward_4159.37.pth
```

Compact PAdapt base bundle:

```text
sim2real/codrive/model_best_codrive.ckpt
```

The PPO teacher remains the distillation target and canonical benchmark. The
PAdapt bundle is used only as the residual base because residual-over-PAdapt is
invalid if `adapt_tconv` is random or restored only from the PPO teacher.

## Hypothesis

Current full-latent generative students spend capacity reconstructing a stable
latent that PAdapt already approximates. A consistency head should instead
learn a bounded residual around the PAdapt latent:

```text
z_base = tanh(adapt_tconv(proprio_hist))
r_target = (z_teacher - z_base) * residual_target_scale
r_pred = ConsistencyHead(proprio_hist, x_t, t)
z_student = z_base + residual_gate * r_pred / residual_target_scale
a_student = frozen_actor(obs, z_student)
```

Expected benefit:

- preserve PAdapt's stable action manifold;
- reduce full-latent decode mismatch;
- keep inference cheap at NFE <= 2;
- improve fixed-step eval over PAdapt and ordinary full-latent diffusion.

## Baseline Evidence

Known CoDrive 4159 evidence from branch handoff:

| Method | Train best | Eval256 reward | Eval256 done_rate |
| --- | ---: | ---: | ---: |
| Diffusion latent 1h | 4145.42 | 2.855633 | 0.000895 |
| Consistency latent 1h | 3837.96 | 5.155176 | 0.000000 |
| Flow matching 1h | 3975.03 | 4.827848 | 0.000081 |
| Diffusion latent continue-2h | 4246.32 | 2.942062 | 0.001465 |
| Consistency continue-2h | 4055.97 | 4.979207 | 0.000000 |
| Flow continue-2h | 4084.44 | 4.364139 | 0.000570 |

Working lesson:

```text
Training reward is not a reliable selector. Fixed-step eval is the selector.
```

PAdapt reference:

```text
PAdapt 30m scalar reward: 3448.38
PAdapt fixed-step handoff reference: about 4.72
```

## Implementation Plan

Add a new class rather than changing existing consistency behavior:

```text
ResidualConsistencyLatentStudent
```

Expected files:

```text
dexscrew/algo/ppo/residual_consistency_latent_student.py  # implementation class and residual training/eval logic
dexscrew/algo/student/residual_consistency_latent.py      # thin public import wrapper
dexscrew/algo/student/__init__.py
train.py
scripts/dexh13_lightbulb_student_residual_consistency_codrive.sh
```

Safety requirements:

- no changes to `dexscrew/dotpg/`;
- no changes to `docs/dotpg_*`;
- no changes to `sim2real/codrive/dotpg_bc5/`;
- no task reward, horizon, seed, or teacher checkpoint changes;
- student inference must not use privileged information;
- fail fast if a residual checkpoint restore lacks a trained PAdapt base.
- assert `consistency_residual_target_scale > 0.0` during initialization.
- `train.py` must only receive an additive import/dispatch symbol for the new
  class; existing algorithm dispatch behavior must remain unchanged.

First default flags:

```text
train.algo=ResidualConsistencyLatentStudent
train.ppo.proprio_adapt=True
checkpoint=sim2real/codrive/model_best_codrive.ckpt
+train.ppo.consistency_residual_target_scale=1.0
+train.ppo.consistency_residual_gate=0.25
+train.ppo.consistency_infer_steps=1
+train.ppo.consistency_stochastic_infer=False
+train.ppo.consistency_train_align_infer=False
+train.ppo.bc_loss_coef=1.0
+train.ppo.base_action_anchor_coef=0.05
```

## Metrics

Primary metric:

```text
fixed-step eval reward
```

Default local ladder:

| Stage | Protocol | Pass condition |
| --- | --- | --- |
| Static | compile and shell syntax | no syntax errors |
| Smoke | minimal agent steps or 16 to 64 fixed eval steps, seed 42, 4 envs | load, shape, no NaN/Inf, bounded GPU; no algorithm claim |
| Scout | short train, one seed, fixed eval256 | screen direction against PAdapt; tune or reject |
| Formal | 30m or larger train budget, train/eval seeds 42/43/44, fixed eval2048 when justified | only after scout improves |

Three-layer allocation adopted for local diffusion work:

- Smoke verifies code/config only. It cannot prove algorithm value.
- Scout is the first meaningful direction screen. It is longer than smoke but
  still cheap enough to discard or retune.
- Formal validation starts only after scout passes. It uses multi-seed training
  and unified fixed-step eval.

Secondary metrics:

- latent MSE/L1 to teacher latent;
- residual target norm and prediction norm;
- predicted residual/base latent norm ratio;
- action MSE to teacher;
- action MSE to PAdapt base;
- action delta, jerk, saturation ratio;
- done_rate;
- NFE and latency.

## Rejection Gates

Reject or revert this direction if:

- restore from `sim2real/codrive/model_best_codrive.ckpt` does not provide a valid
  trained `adapt_tconv` base;
- residual norm dominates base norm without eval improvement;
- action jitter, saturation, or done_rate clearly worsens;
- NFE 1 or 2 is unusable;
- candidate fixed-step reward is below PAdapt reference;
- any result depends on changed task reward, horizon, seeds, or teacher.

## Review Requirements

Before code changes:

- subagent algorithm/artifact audit must be incorporated;
- Claude Opus must review this card and the proposed implementation boundary.

Before accepting results:

- Claude Opus must review the result interpretation;
- update `Goal/diffusion_goal_plan.md` or a future handoff with artifact pointers.

## Implementation And Smoke Log

Implementation status:

```text
ResidualConsistencyLatentStudent added as an additive public entry.
```

Smoke commands were run locally with `task.env.numEnvs=4`, bounded timeouts, and
GPU utilization checked before launch.

Static checks:

```text
python compile: pass
bash -n residual script: pass
git diff --check: pass
DOTPG/private touched: no
```

Checkpoint audit:

```text
sim2real/codrive/best_reward_4159.37.pth adapt_tconv_keys=0
sim2real/codrive/model_best_codrive.ckpt adapt_tconv_keys=12
```

Train smoke:

```text
checkpoint=sim2real/codrive/model_best_codrive.ckpt
NUM_ENVS=4
MINIBATCH=12
STUDENT_MAX_AGENT_STEPS=16
result=exit 0
saved=outputs/Dexh13HoraLightbulb_student_residual_consistency_codrive/smoke_residual_consistency_g1/stage2_residual_consistency_nn/model_best.ckpt
```

Restore/test smoke:

```text
checkpoint=outputs/Dexh13HoraLightbulb_student_residual_consistency_codrive/smoke_residual_consistency_g1/stage2_residual_consistency_nn/model_best.ckpt
steps=16
avg_reward=4.695521
avg_done_rate=0.000000
latent_mse=0.062897
residual_mse=0.065716
action_mse_to_teacher=0.047855
delta_to_base_ratio=0.115267
action_mse_to_base=0.000821
saturation_ratio=0.320312
```

Post-review fix:

```text
restore_train/restore_test now use torch.load(..., map_location=self.device)
and avoid parent restore double-loading without map_location.
```

## G1 Scout Result

This run is a scout probe, not a smoke test.

Run:

```text
run_tag=g1_residual_consistency_s42_20260601_191111
base_ckpt=sim2real/codrive/model_best_codrive.ckpt
NUM_ENVS=16
MINIBATCH=192
STUDENT_MAX_AGENT_STEPS=200000
TRAIN_WINDOW_SEC=1800
EVAL_STEPS=256
EVAL_SEEDS=42
```

Artifacts:

```text
probe_dir=outputs/local_probe_residual_consistency_g1/g1_residual_consistency_s42_20260601_191111
checkpoint=outputs/Dexh13HoraLightbulb_student_residual_consistency_codrive/g1_residual_consistency_s42_20260601_191111/stage2_residual_consistency_nn/model_best.ckpt
eval_summary=outputs/local_probe_residual_consistency_g1/g1_residual_consistency_s42_20260601_191111/eval_summary.tsv
```

Train:

```text
exit_status=0
student_max_agent_steps=200000 reached
training_current_best=3798.00
fps_steady_state=about 246
```

Eval256 seed 42:

| Metric | Value |
| --- | ---: |
| avg_reward | 4.706028 |
| avg_done_rate | 0.000000 |
| latent_mse | 0.036799 |
| residual_mse | 0.064682 |
| action_mse_to_teacher | 0.045891 |
| delta_to_base_ratio | 0.170753 |
| action_mse_to_base | 0.010439 |
| saturation_ratio | 0.302887 |

Interpretation:

- The result is near but slightly below the PAdapt fixed-step reference of about
  `4.72`.
- It is below the existing Flow screen `4.827848` and far below the existing
  Consistency screen `5.155176`.
- `delta_to_base_ratio=0.170753` and low `action_mse_to_base` indicate the
  residual head stayed close to the PAdapt base.
- Training `Current Best=3798.00` is not accepted as the selector.

Claude Opus result cross-review:

```text
verdict=TUNE
do_not_promote_to_formal=true
main_risk=residual path is too close to base / likely over-constrained
recommended_next_probe=gate 0.25 -> 0.6, base_action_anchor_coef 0.05 -> 0.01
```

Decision:

- Do not run formal or multi-seed validation for this exact configuration.
- Run one more scout with a larger residual gate and weaker base anchor.
- If the next scout remains below `4.8`, reject this residual-over-PAdapt default
  and return the mainline to standalone consistency or a consistency-base
  residual variant.

## G1 Scout B Result

This run tested the Opus-suggested larger gate and weaker base anchor.

Run:

```text
run_tag=g1_residual_consistency_gate06_anchor001_s42_20260601_192855
base_ckpt=sim2real/codrive/model_best_codrive.ckpt
NUM_ENVS=16
MINIBATCH=192
STUDENT_MAX_AGENT_STEPS=200000
TRAIN_WINDOW_SEC=1800
RESIDUAL_GATE=0.6
BASE_ACTION_ANCHOR_COEF=0.01
EVAL_STEPS=256
EVAL_SEEDS=42
```

Artifacts:

```text
probe_dir=outputs/local_probe_residual_consistency_g1/g1_residual_consistency_gate06_anchor001_s42_20260601_192855
checkpoint=outputs/Dexh13HoraLightbulb_student_residual_consistency_codrive/g1_residual_consistency_gate06_anchor001_s42_20260601_192855/stage2_residual_consistency_nn/model_best.ckpt
eval_summary=outputs/local_probe_residual_consistency_g1/g1_residual_consistency_gate06_anchor001_s42_20260601_192855/eval_summary.tsv
```

Train:

```text
exit_status=0
student_max_agent_steps=200000 reached
training_current_best=3753.41
fps_steady_state=about 246
```

Eval256 seed 42:

| Metric | Value |
| --- | ---: |
| avg_reward | 4.607157 |
| avg_done_rate | 0.000000 |
| latent_mse | 0.038634 |
| residual_mse | 0.041733 |
| action_mse_to_teacher | 0.045579 |
| delta_to_base_ratio | 0.219738 |
| action_mse_to_base | 0.013250 |
| saturation_ratio | 0.306686 |

Scout comparison:

| Config | Train best | Eval256 reward | delta/base | action MSE to base | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| gate 0.25, anchor 0.05 | 3798.00 | 4.706028 | 0.170753 | 0.010439 | below PAdapt reference |
| gate 0.60, anchor 0.01 | 3753.41 | 4.607157 | 0.219738 | 0.013250 | worse reward |

Interpretation:

- Increasing the residual gate and relaxing the anchor made the residual path
  more active, but fixed-step reward became worse.
- Both scouts are below the PAdapt fixed-step reference of about `4.72`.
- Both scouts are below Flow `4.827848` and much lower than the existing
  Consistency screen `5.155176`.
- NFE=2 is not the next priority for this PAdapt-base residual path because the
  NFE=1 direction is already below the baseline gate.

Claude Opus result cross-review:

```text
verdict=REJECT residual-over-PAdapt default
formal_or_multiseed=false
nfe2_priority=low
recommended_next=validate standalone consistency first; consider consistency-base residual only after artifact audit
```

Decision update:

- Do not promote PAdapt-base `ResidualConsistencyLatentStudent` to formal or
  multi-seed validation.
- Downgrade the current PAdapt-base residual default from mainline to ablation.
- Next useful work should either:
  - formalize the existing standalone consistency result with artifact audit and
    multi-seed eval; or
  - design a consistency-base residual variant if the standalone consistency
    checkpoint is reusable as a stable base.
- Add clearer scout progress instrumentation before longer local runs, because
  the current `Agent Steps: 0000M` display is too coarse for sub-1M probes.
