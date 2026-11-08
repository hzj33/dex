# Metric Card G4D: External Selector Formal Follow-Up

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

This round does not train and does not change algorithm code. It validates the
G4C deploy policy at a longer fixed-step horizon.

## Starting Evidence

G4C completed eval512 checkpoint comparison over:

```text
train seeds: 42,43,44
checkpoint kinds: model_best, model_best_train, model_last
eval seeds: 42,43,44
```

G4C fixed-kind aggregates:

| Checkpoint kind | n | Mean reward | Done mean |
| --- | ---: | ---: | ---: |
| `model_best` | 9 | 5.120598 | 0.000122 |
| `model_best_train` | 9 | 5.037163 | 0.000190 |
| `model_last` | 9 | 5.132772 | 0.000122 |

G4C external-selector candidates:

| Train seed | Candidate | Eval512 reward | Note |
| ---: | --- | ---: | --- |
| 42 | `model_last` | 5.150810 | best external eval512 checkpoint |
| 43 | `model_best_train` | 5.171160 | best external eval512 checkpoint |
| 44 | `model_best` | 5.140846 | tied with `model_last`; prefer alias clarity |

G4C decision:

```text
TUNE / use post-training external eval selector for deploy candidates
```

## Hypothesis

The external eval512 selector gives a better deploy-candidate policy than any
single fixed alias kind, but the claim must hold under the formal eval2048
protocol before further diffusion-family algorithm tuning.

## Protocol

Evaluate only existing checkpoints:

```text
eval steps: 2048
eval seeds: 42,43,44
eval num envs: 16
```

Candidate checkpoint map:

| Train seed | Checkpoint path |
| ---: | --- |
| 42 | `outputs/Dexh13HoraLightbulb_student_consistency_codrive/g4c_consistency_selector_20260602_032255_train_s42/stage2_consistency_nn/model_last.ckpt` |
| 43 | `outputs/Dexh13HoraLightbulb_student_consistency_codrive/g4c_consistency_trainonly_20260602_040200_s43/stage2_consistency_nn/model_best_train.ckpt` |
| 44 | `outputs/Dexh13HoraLightbulb_student_consistency_codrive/g4c_consistency_trainonly_20260602_043500_s44/stage2_consistency_nn/model_best.ckpt` |

Primary metric:

```text
formal eval2048 reward mean over train seeds 42,43,44 and eval seeds 42,43,44
```

Secondary metrics:

- reward std across the 9 rows;
- done rate;
- latent MSE/L1;
- action MSE to teacher;
- per-train-seed stability;
- whether eval512-selected candidates still beat the PAdapt fixed-step
  reference and historical consistency formal aggregate.

## Gate

PASS if:

- formal eval2048 aggregate is above the existing G2 formal consistency
  aggregate `4.919267`;
- no train seed falls below the PAdapt fixed-step reference;
- selected candidate policy does not introduce obvious done-rate or recon
  degradation.

TUNE if:

- aggregate improves but one train seed remains weak;
- eval512-selected candidates fail to transfer cleanly to eval2048;
- a fixed `model_last` policy looks equally good and simpler.

REJECT external selector if:

- eval2048 shows the selected candidates are worse than the existing formal
  consistency result or below PAdapt.

## Command Template

Run the three checkpoint evals serially:

```text
GPU=0 EVAL_STEPS=2048 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
EVAL_ROOT=outputs/local_eval_consistency_g4d_external_selector \
G4D_TAG=g4d_external_selector_$(date +%Y%m%d_%H%M%S) \
bash scripts/run_consistency_g4d_external_selector_eval.sh
```

## Review Requirements

- Use subagent result sanity checking before declaring `PASS`, `TUNE`, or
  `REJECT`.
- Use Claude Opus cross-review for the final interpretation.
- Do not start LR, EMA, NFE, robustness, or residual-consistency changes until
  this formal follow-up is classified.

## Results

Suite directory:

```text
outputs/local_probe_consistency_g4d_external_selector/g4d_external_selector_20260602_053228
```

Data completeness:

```text
per-eval rows: 9/9
status: all 0
steps: all 2048
```

Per-train-seed formal eval2048 aggregates:

| Train seed | Candidate | n | Mean reward | Reward std | Done mean | Latent MSE | Action MSE to teacher |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `model_last` | 3 | 5.063657 | 0.102057 | 0.001068 | 0.040193 | 0.045918 |
| 43 | `model_best_train` | 3 | 5.124141 | 0.070748 | 0.001079 | 0.042131 | 0.040266 |
| 44 | `model_best` | 3 | 5.078385 | 0.070459 | 0.001089 | 0.041352 | 0.040613 |

Overall aggregate:

| n | Mean reward | Reward std | Done mean | Latent MSE | Latent L1 | Action MSE to teacher |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9 | 5.088728 | 0.086362 | 0.001078 | 0.041225 | 0.120587 | 0.042266 |

Comparison:

- Above G2 formal consistency aggregate `4.919267` by about `+0.169461`.
- Above the PAdapt fixed-step reference of about `4.72` for every train seed.
- G4C eval512 external-selected aggregate was about `5.154272`, so eval2048
  introduces a formal-horizon gap of about `-0.065544`.
- `avg_done_rate` is a reset/termination rate in these eval summaries, not a
  task success rate. Lower is better; the observed `0.001078` is in the same
  range as the previous G2 formal value `0.001068`.

## Reviews

Subagent result audit:

```text
reviewer=Hypatia
verdict=PASS for G4D formal candidate validation
```

Hypatia notes:

- 9/9 eval rows are complete and successful.
- The external-selected candidate set is stable under eval2048.
- The eval512-to-eval2048 drop is real but modest, and does not invalidate the
  selector.
- Do not go back to a fixed alias policy. Keep post-training external eval as
  the deploy-candidate selector.

Claude Opus cross-review:

```text
initial review misread avg_done_rate as task completion rate
corrected review verdict=PASS after avg_done_rate was clarified as reset/termination rate
```

Corrected Opus notes:

- Reward `5.0887` exceeds both G2 formal consistency `4.919` and the PAdapt
  reference around `4.72`.
- Done mean `0.001078` is healthy low-termination behavior, not a failure
  signal.
- All 9 eval runs completed with stable cross-checkpoint performance.
- The best individual selected checkpoint is train seed `43`
  `model_best_train`, reward `5.124141`.

## Decision

```text
PASS selector formal validation / TUNE algorithm upper bound
```

Interpretation:

- The checkpoint-selection problem is no longer the main blocker for this
  consistency branch.
- The deploy policy should be: save candidate checkpoints during training, then
  choose the deploy candidate with post-training external fixed-step eval.
- Do not claim that in-training `model_best`, scalar `model_best_train`, or
  final `model_last` is a universally correct deploy alias.
- The formal reward is strong enough to proceed beyond selector diagnosis, but
  not strong enough to declare the diffusion-family optimization finished.

Next step:

- Use G4D as the current standalone-consistency formal anchor.
- Move the next metric card to either:
  - robustness/latency/export parity for the selected consistency candidates;
  - or a new consistency-base residual / loss-tuning scout aimed at lifting the
    formal eval2048 reward above the current `5.088728` anchor.
