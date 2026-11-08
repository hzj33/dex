# Metric Card G4B: Consistency Selector Metric Tuning

Date: 2026-06-02

Repo and branch:

```text
/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro-public
branch: diffusion
```

## Scope

Canonical task:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
```

Canonical teacher:

```text
sim2real/codrive/best_reward_4159.37.pth
```

This round only tunes checkpoint selection for `ConsistencyLatentStudent`.
It does not change task reward, horizon, reset logic, teacher checkpoint,
student architecture, DOTPG, or private repo flows.

## Starting Evidence

G4 showed that eval-select checkpoint preservation worked, but its metric was
badly calibrated:

| Checkpoint | Eval512 reward mean | action_mse_to_teacher |
| --- | ---: | ---: |
| `model_best_eval` / `model_best` | 4.457390 | 0.047801 |
| `model_last` | 4.933033 | 0.045391 |
| `model_best_train` | 4.909166 | 0.043669 |

G4 used:

```text
eval_select.num_steps=256
eval_select.done_penalty=2000.0
```

The internal score was:

```text
score = avg_reward - done_penalty * avg_done_rate
```

This made the tiny done-rate differences dominate raw reward and selected the
400k checkpoint despite lower raw internal reward.

## Hypothesis

A reward-first selector aligned with the external gate should choose a better
checkpoint than G4:

```text
eval_select.num_steps=512
eval_select.done_penalty=0.0
```

The final eval-select hook should also emit a `FINAL/student` row after the
training loop.

## Probe

Run one train seed first:

```text
GPU=0 SEED=43 RUN_TAG=g4b_consistency_seed43_evalselect512_reward_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g4b \
TRAIN_WINDOW_SEC=3600 STUDENT_MAX_AGENT_STEPS=450000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/run_consistency_g2_validate_codrive.sh \
  ++train.ppo.eval_select.enabled=True \
  ++train.ppo.eval_select.interval_agent_steps=100000 \
  ++train.ppo.eval_select.min_agent_steps=100000 \
  ++train.ppo.eval_select.num_steps=512 \
  ++train.ppo.eval_select.done_penalty=0.0 \
  ++train.ppo.eval_select.min_score_improvement=0.0 \
  ++train.ppo.eval_select.final_eval=True \
  ++train.ppo.eval_select.save_deploy_best=True
```

## Metrics

Primary:

```text
external eval512 reward mean over eval seeds 42,43,44
```

Secondary:

- selected checkpoint stem and hash;
- internal eval-select history, including final row;
- external eval512 for selected checkpoint;
- compare to `model_last` and `model_best_train` if selected checkpoint is weak;
- latent MSE/L1 and action MSE to teacher;
- GPU budget around `80%`, with no sustained `>85%`.

## Decision Rule

- PASS if selected `model_best_eval` reaches eval512 mean `>=5.1` and is not
  worse than `model_last`.
- TUNE if selected checkpoint improves over G4 but remains below the G3A
  recovered range.
- REJECT in-training selector if `model_last` or `model_best_train` is clearly
  stronger again; then use a post-training external eval512 selector wrapper.

Do not move to LR/EMA/NFE/residual until this selector decision is settled.

## Result

Run:

```text
g4b_consistency_seed43_evalselect512_reward_20260602_023047
```

Training budget:

```text
train seed: 43
max agent steps: 450k
num envs: 16
eval-select interval: 100k agent steps
eval-select num_steps: 512
eval-select done_penalty: 0.0
external eval: eval512 seeds 42,43,44
```

Internal eval-select history:

| Agent steps | Label | train_reward | avg_reward | avg_done_rate | score | saved_best |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 100k | `EVAL/student` | 3557.662965 | 4.740285 | 0.001831 | 4.740285 | 1 |
| 200k | `EVAL/student` | 3659.259795 | 5.155947 | 0.000854 | 5.155947 | 1 |
| 300k | `EVAL/student` | 3622.195985 | 5.190971 | 0.001831 | 5.190971 | 1 |
| 400k | `EVAL/student` | 3688.287552 | 5.269734 | 0.000122 | 5.269734 | 1 |
| 450k | `FINAL/student` | 3637.883280 | 4.959214 | 0.001587 | 4.959214 | 0 |

Checkpoint hashes:

| Checkpoint | SHA256 prefix | Note |
| --- | --- | --- |
| `model_best.ckpt` | `4318371a0fc5` | same as `model_best_eval` and `model_best_deploy` |
| `model_best_eval.ckpt` | `4318371a0fc5` | selected by eval-select at 400k |
| `model_best_deploy.ckpt` | `4318371a0fc5` | deploy alias republished from eval best |
| `model_best_train.ckpt` | `d97cbf759005` | scalar training-reward best |
| `model_last.ckpt` | `894ce60fac79` | final 450k checkpoint |

External eval512 aggregates:

| Checkpoint | seed42 | seed43 | seed44 | mean | std | done_mean | latent_mse | action_mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_best_eval` / `model_best` | 5.022744 | 5.181186 | 5.207341 | 5.137090 | 0.081557 | 0.000203 | 0.042821 | 0.042884 |
| `model_last` | 5.032464 | 5.111177 | 5.176339 | 5.106660 | 0.058824 | 0.000081 | 0.039430 | 0.036529 |
| `model_best_train` | 5.042362 | 5.255656 | 5.215462 | 5.171160 | 0.092540 | 0.000203 | 0.042360 | 0.039373 |

Decision:

```text
CONDITIONAL PASS: selector wiring validated under real training load; not a formal pass.
```

Interpretation:

- G4B fixed the obvious G4 metric failure: `done_penalty=0.0` and `512` internal
  eval steps selected a checkpoint with external eval512 mean `5.137090`, above
  `model_last` and well above the G4 selected checkpoint `4.457390`.
- The final eval-select wiring now works: the 450k `FINAL/student` row was emitted
  and did not overwrite the stronger 400k eval-selected checkpoint.
- The scalar training-reward checkpoint was still numerically higher on external
  eval512 (`5.171160`) than the selected checkpoint (`5.137090`). The gap is small
  relative to seed variation, but `model_best_train` was higher on all three
  eval seeds. The in-training selector is therefore not yet a final deploy
  policy.
- This is one train seed only. It validated selector wiring under a realistic
  training load; it did not prove that eval-select adds value over train-best.
  It is not a smoke test and not a formal three-train-seed validation.

Next step:

- Promote selector validation, not architecture changes.
- Run train seeds `42,43,44` with the G4B selector settings, then evaluate
  `model_best_eval`, `model_best_train`, and `model_last` with fixed eval seeds
  `42,43,44`.
- If train-best continues to beat in-training eval-select, use a post-training
  external eval512 selector wrapper for deploy aliases.
- Only after this selector decision is stable should we move to LR/EMA/NFE,
  robustness, or residual-consistency probes.

## Post-Result Claude Opus Review

```text
verdict=PASS on interpretation
wording_correction=selector wiring validated under real training load; it does
not yet prove eval-select adds value over train-best
next_step=multi-train-seed selector validation before external-selector switch
overclaim_risk=do not call G4B a selector win over train-best
```

Review guidance:

- The selected checkpoint beat `model_last`, so the G4B selector is not actively
  harmful in this seed.
- The `0.034070` reward gap between `model_best_train` and `model_best_eval` is
  smaller than the eval-seed standard deviations, so it is not enough evidence to
  reject in-training eval-select.
- Do not immediately replace eval-select with an external selector based on one
  train seed.
- In G4C, record whether the train-best checkpoint was internally close to the
  selected checkpoint. A top-K candidate policy may be enough if train-best is
  consistently a close second.

## Subagent Audit

```text
reviewer=Hypatia
verdict=supported
main_correction=avoid wording that G4B globally fixed selection; it fixed the
G4 done-penalty failure for this seed/scout
```

Audit notes:

- The conditional-pass wording is supported because the selected checkpoint beat
  `model_last` and recovered from the G4 failure.
- It is not a formal pass because this is one train seed, eval512 only, and
  `model_best_train` beat `model_best_eval` on all three eval seeds.
- Reproducibility note: G4B depends on the current uncommitted final-eval patch,
  so artifact provenance must include the dirty worktree state until the patch is
  committed or otherwise recorded.
