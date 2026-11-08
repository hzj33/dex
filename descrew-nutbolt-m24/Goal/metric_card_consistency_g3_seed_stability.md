# Metric Card G3: Consistency Train-Seed Stability

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

Canonical PPO teacher:

```text
sim2real/codrive/best_reward_4159.37.pth
```

This round diagnoses the G2 standalone consistency formal failure mode. It does
not change the task, teacher, horizon, eval seeds, private repo, or DOTPG flow.

## Starting Evidence

G2 formal run:

```text
formal_dir=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702
summary=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702/formal_eval_summary.tsv
NFE=1
```

Per-train-seed eval2048 means:

| Train seed | reward_mean | reward_std | done_mean | latent_mse | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.123429 | 0.084707 | 0.001017 | 0.032629 | 0.032020 |
| 43 | 4.730098 | 0.056040 | 0.001109 | 0.037348 | 0.038814 |
| 44 | 4.904273 | 0.088405 | 0.001078 | 0.040359 | 0.043973 |

Overall formal aggregate:

```text
reward_mean=4.919267
reward_std=0.178720
done_rate_mean=0.001068
```

Conclusion:

- G2 is `TUNE`, not stable formal `PASS`.
- Seed `43` is the weakest train seed despite having the highest scalar
  training best, so selector behavior and train-seed stability must be diagnosed
  before NFE or architecture changes.

Claude Opus formal review:

```text
verdict=TUNE
recommended_next=diagnose seed 43 before adding complexity
minimal_probe=seed43-only LR/EMA or selector probe around 450k steps per setting
```

Claude Opus G3 protocol review:

```text
verdict=PASS
blocking_issues=none
execution_order=run G3A first, alone
note=make model_best versus model_last comparison non-optional for at least G3A
```

Subagent read-only audit:

```text
agent=Euclid
verdict=selector is the safest first suspect
checkpoint_rule=model_best is selected by online training episode reward, while model_last is saved at the step cap
eval_select=existing EvalSelectMixin hook is available, but base CoDrive config does not enable it
caution=consistency_train_align_infer is diagnostic only because sampled latent uses no_grad and BC will not train through it
```

## Hypothesis

Standalone consistency is still the best active diffusion-family candidate, but
seed `43` falls into a weaker basin or selects a weaker checkpoint. A small
optimization or selector change should recover seed `43` without changing the
canonical protocol.

## Instrumentation Added

The consistency launch/eval scripts now expose and record these no-op-by-default
knobs:

```text
CONSISTENCY_LR
CONSISTENCY_LOSS_COEF
CONSISTENCY_BOUNDARY_COEF
CONSISTENCY_USE_EMA_TARGET
CONSISTENCY_EMA_DECAY
CONSISTENCY_INFER_USE_EMA
CONSISTENCY_TRAIN_ALIGN_INFER
BC_LOSS_COEF
BASE_ACTION_ANCHOR_COEF
CONSISTENCY_ACTION_L2_COEF
```

Default values preserve the G2 behavior.

## Metrics

Primary:

```text
fixed-step eval reward on train seed 43
```

Probe protocol:

```text
TRAIN_SEED=43
STUDENT_MAX_AGENT_STEPS=450000
TRAIN_WINDOW_SEC=2400
NUM_ENVS=16
MINIBATCH=192
EVAL_STEPS=512
EVAL_SEEDS="42 43 44"
CONSISTENCY_INFER_STEPS=1
```

Secondary:

- done_rate;
- latent MSE/L1;
- action MSE to teacher;
- model_best versus model_last when both exist;
- training `Current Best` versus fixed-step eval;
- local GPU budget, with target around `80%` and sustained `>85%` avoided.

Promotion threshold:

- seed `43` probe reward mean should recover toward `4.9+`;
- action MSE should not worsen relative to G2 seed43 formal mean `0.038814`;
- done_rate should remain near the G2 formal range;
- no protocol or privileged-input change.

## Probe Matrix

Run only one or two probes at a time, then update this card before continuing.

| Probe | Purpose | Key change |
| --- | --- | --- |
| G3A default rerun | estimate 450k seed43 variance and selector behavior | no algorithm change |
| G3B lower LR | test whether weaker seed is optimization noise | `CONSISTENCY_LR=1e-4` |
| G3C EMA | test whether target/inference smoothing stabilizes seed43 | `CONSISTENCY_USE_EMA_TARGET=True`, `CONSISTENCY_INFER_USE_EMA=True`, `CONSISTENCY_EMA_DECAY=0.995` |

No-retrain selector probe:

- Before promoting any tuning knob, evaluate G2 seed43 `model_best.ckpt` versus
  `model_last.ckpt` if both exist.
- This directly tests whether the weak G2 seed43 result is due to scalar
  training-reward checkpoint selection.

## Local Commands

Static:

```text
bash -n scripts/dexh13_lightbulb_student_consistency_codrive.sh scripts/run_consistency_g2_validate_codrive.sh scripts/eval_consistency_codrive_checkpoint.sh
git diff --check -- scripts/dexh13_lightbulb_student_consistency_codrive.sh scripts/run_consistency_g2_validate_codrive.sh scripts/eval_consistency_codrive_checkpoint.sh Goal/metric_card_consistency_g3_seed_stability.md
```

G3A default rerun:

```text
GPU=0 SEED=43 RUN_TAG=g3a_consistency_seed43_default_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g3 \
TRAIN_WINDOW_SEC=2400 STUDENT_MAX_AGENT_STEPS=450000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/run_consistency_g2_validate_codrive.sh
```

G3B lower LR:

```text
GPU=0 SEED=43 RUN_TAG=g3b_consistency_seed43_lr1e4_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g3 \
TRAIN_WINDOW_SEC=2400 STUDENT_MAX_AGENT_STEPS=450000 \
NUM_ENVS=16 MINIBATCH=192 \
CONSISTENCY_LR=1e-4 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/run_consistency_g2_validate_codrive.sh
```

G3C EMA:

```text
GPU=0 SEED=43 RUN_TAG=g3c_consistency_seed43_ema0995_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g3 \
TRAIN_WINDOW_SEC=2400 STUDENT_MAX_AGENT_STEPS=450000 \
NUM_ENVS=16 MINIBATCH=192 \
CONSISTENCY_USE_EMA_TARGET=True CONSISTENCY_INFER_USE_EMA=True CONSISTENCY_EMA_DECAY=0.995 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/run_consistency_g2_validate_codrive.sh
```

Optional selector check after each probe:

```text
CKPT=outputs/Dexh13HoraLightbulb_student_consistency_codrive/<run_tag>/stage2_consistency_nn/model_last.ckpt \
RUN_TAG=<run_tag>_modellast_eval512 \
EVAL_ROOT=outputs/local_eval_consistency_g3 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/eval_consistency_codrive_checkpoint.sh
```

Selector rule:

- G3A must evaluate both `model_best.ckpt` and `model_last.ckpt` if both exist.
- G3B/G3C should do the same if G3A shows a meaningful best/last gap.
- Do not run G3B or G3C until the no-retrain G2 seed43 selector check and the
  G3A best/last check are interpreted.

## Decision Rule

- If G3A is already near `4.9+`, the G2 seed43 formal result may be ordinary
  training variance; repeat or extend the default config before architecture
  changes.
- If G3B or G3C recovers seed43 while G3A does not, promote that knob to a
  three-train-seed formal rerun.
- If none recover seed43, move to selector instrumentation or
  consistency-base residual as the next controlled hypothesis.

## Results

### G3A Default Rerun

Run:

```text
run_tag=g3a_consistency_seed43_default_20260602_004958
train_seed=43
student_max_agent_steps=450000
train_window_sec=2400
num_envs=16
minibatch=192
consistency_lr=3e-4
NFE=1
```

Artifacts:

```text
probe_dir=outputs/local_probe_consistency_g3/g3a_consistency_seed43_default_20260602_004958
model_best=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g3a_consistency_seed43_default_20260602_004958/stage2_consistency_nn/model_best.ckpt
model_last=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g3a_consistency_seed43_default_20260602_004958/stage2_consistency_nn/model_last.ckpt
```

Train:

```text
exit_status=0
current_best=3833.17
gpu_budget=mostly 77-80 percent during training; eval had short 85-87 percent peaks
```

`model_best` eval512:

| Eval seed | reward | done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.147758 | 0.000000 | 0.046202 | 0.134473 | 0.046037 |
| 43 | 5.126883 | 0.000000 | 0.043431 | 0.130660 | 0.043142 |
| 44 | 5.254880 | 0.000122 | 0.046048 | 0.132187 | 0.041621 |

Aggregate:

```text
reward_mean=5.176507
reward_std=0.056070
done_mean=0.000041
latent_mse=0.045227
latent_l1=0.132440
action_mse_to_teacher=0.043600
```

`model_last` eval512:

| Eval seed | reward | done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.047732 | 0.000244 | 0.045722 | 0.132917 | 0.041453 |
| 43 | 5.136811 | 0.000366 | 0.043593 | 0.130155 | 0.038573 |
| 44 | 5.169687 | 0.000122 | 0.045930 | 0.132307 | 0.042244 |

Aggregate:

```text
reward_mean=5.118077
reward_std=0.051520
done_mean=0.000244
latent_mse=0.045082
latent_l1=0.131793
action_mse_to_teacher=0.040757
```

Interpretation:

- A same-seed default rerun at `450k` agent steps strongly recovers train seed
  `43`, so the G2 seed43 weakness is not a necessary architecture failure.
- `model_best` is slightly better than `model_last` within the 450k run, but
  both checkpoints are strong.
- Action MSE is higher than the original G2 seed43 formal `model_best` action
  MSE, so the best next step is selector diagnosis rather than declaring this
  shorter run accepted.

### No-Retrain G2 Seed43 Selector Probe

Original weak G2 checkpoint:

```text
run_tag=g2_consistency_formal_20260601_212702_train_s43
model_best=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_formal_20260601_212702_train_s43/stage2_consistency_nn/model_best.ckpt
model_last=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_formal_20260601_212702_train_s43/stage2_consistency_nn/model_last.ckpt
```

`model_best` eval512:

| Eval seed | reward | done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 4.427458 | 0.000244 | 0.041983 | 0.115638 | 0.044263 |
| 43 | 4.716400 | 0.000366 | 0.037080 | 0.109573 | 0.038736 |
| 44 | 4.922064 | 0.000122 | 0.034634 | 0.107471 | 0.035156 |

Aggregate:

```text
reward_mean=4.688641
reward_std=0.202874
done_mean=0.000244
latent_mse=0.037899
latent_l1=0.110894
action_mse_to_teacher=0.039385
```

`model_last` eval512:

| Eval seed | reward | done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.014484 | 0.000122 | 0.032341 | 0.102464 | 0.031591 |
| 43 | 5.293957 | 0.000000 | 0.027564 | 0.096752 | 0.027145 |
| 44 | 5.203867 | 0.000366 | 0.028218 | 0.098659 | 0.025337 |

Aggregate:

```text
reward_mean=5.170769
reward_std=0.116470
done_mean=0.000163
latent_mse=0.029374
latent_l1=0.099292
action_mse_to_teacher=0.028024
```

`model_last` eval2048:

| Eval seed | reward | done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 5.031105 | 0.001007 | 0.030490 | 0.100299 | 0.030568 |
| 43 | 5.151453 | 0.001068 | 0.030382 | 0.099503 | 0.030803 |
| 44 | 5.234457 | 0.001038 | 0.028911 | 0.098685 | 0.028151 |

Aggregate:

```text
reward_mean=5.139005
reward_std=0.083483
done_mean=0.001038
latent_mse=0.029928
latent_l1=0.099496
action_mse_to_teacher=0.029841
```

Interpretation:

- The G2 seed43 weak formal result is a checkpoint-selector failure.
- Under the same eval2048 protocol, G2 seed43 `model_last` recovers from
  `4.730098` to `5.139005`.
- The recovered `model_last` also improves reconstruction metrics relative to
  the weak `model_best`.
- Replacing only seed43 `model_best` rows with seed43 `model_last` rows would
  move the 9-row formal mean from `4.919267` to about `5.055569`.

Decision:

- Do not run G3B lower-LR or G3C EMA yet.
- Prioritize checkpoint selection instrumentation or enabling the existing
  eval-select hook for standalone consistency.
- The next formal consistency claim must compare `model_best_train`,
  `model_best_eval` or `model_last`, and must select by fixed-step eval rather
  than scalar online training reward.
