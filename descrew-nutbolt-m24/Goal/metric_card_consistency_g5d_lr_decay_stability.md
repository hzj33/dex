# Metric Card G5D: Consistency Train-Aligned BC Stability Schedule

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

Evidence layer:

```text
Scout, not formal validation.
```

G5D follows the G5C seed-stability reject. It does not change task reward,
horizon, eval seeds, teacher, private repo, or DOTPG flow.

## Starting Evidence

G5C showed that the current train-aligned BC settings are not seed-stable:

| Train seed | Best checkpoint | Eval512 mean reward | Done mean | Latent MSE | Action MSE |
| --- | --- | ---: | ---: | ---: | ---: |
| `42` | `model_best_alias` | 4.960817 | 0.000041 | 0.071036 | 0.039160 |
| `44` | `model_best_alias` | 5.102461 | 0.000122 | 0.070952 | 0.037441 |

G5C internal eval-select peaked before the end:

| Train seed | 100k | 200k | 300k | 400k |
| --- | ---: | ---: | ---: | ---: |
| `42` | 4.811486 | 4.906337 | 4.931985 | 4.433161 |
| `44` | 4.842363 | 5.008591 | 5.172998 | 5.034783 |

## Hypothesis

The G5 train-aligned BC path may be useful, but the current schedule does not
hold the `300k` eval peak. A lower effective learning rate after about `250k`
should reduce late-training degradation and improve external eval stability.

## Candidate Change

One main change only:

```text
Add a configurable consistency LR schedule or run with a lower effective
consistency LR after about 250k agent steps.
```

Keep unchanged:

- task and teacher;
- eval seeds `42,43,44`;
- external selector over `model_best`, `model_best_train`, and `model_last`;
- `consistency_train_align_infer=True`;
- no privileged information at student inference.

## Local Protocol

Default first scout:

```text
TRAIN_SEEDS="44"
STUDENT_MAX_AGENT_STEPS=450000
EVAL_STEPS=512
EVAL_SEEDS="42 43 44"
NUM_ENVS=16
EVAL_NUM_ENVS=8
```

Use local GPU only. Training should remain near the `80%` utilization budget;
external eval uses `EVAL_NUM_ENVS=8` by default to reduce desktop impact.

## Gates

G5D directional PASS if:

- internal eval-select stays near or above `5.17` at `400k/450k`;
- external eval512 best checkpoint mean is at least `5.17`;
- action MSE does not regress versus G5C train seed `44` best alias
  (`0.037441`);
- done mean stays near `0.001` or lower.

G5D TUNE if:

- late-training degradation is reduced but external mean remains in the
  `5.05-5.17` range;
- reward improves but action MSE or latent MSE worsens clearly;
- selector alias and external selection disagree but a usable checkpoint is
  recovered.

G5D REJECT if:

- external best remains below `5.05`;
- `400k/450k` still drops sharply from the `300k` peak;
- action quality degrades enough to offset reward.

## Review Requirements

- Run static checks for any scheduler/config patch.
- Run one smoke before the 450k scout.
- Use Claude Opus review before accepting the result interpretation.
- Use subagent review if agent capacity is available.

## Implementation

Status: implemented as a default-off configuration path.

Touched paths:

```text
dexscrew/algo/ppo/consistency_latent_student.py
scripts/dexh13_lightbulb_student_consistency_codrive.sh
scripts/run_consistency_g2_validate_codrive.sh
scripts/run_consistency_g4c_selector_codrive.sh
```

New knobs:

```text
CONSISTENCY_LR_SCHEDULE=none|step|linear_decay|cosine_decay
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=<int>
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=<int>
CONSISTENCY_LR_FINAL_SCALE=<float>
```

Default behavior remains schedule `none` with final scale `1.0`.

Static checks passed:

```text
python -m py_compile dexscrew/algo/ppo/consistency_latent_student.py
bash -n scripts/dexh13_lightbulb_student_consistency_codrive.sh
bash -n scripts/run_consistency_g2_validate_codrive.sh
bash -n scripts/run_consistency_g4c_selector_codrive.sh
git diff --check -- dexscrew/algo/ppo/consistency_latent_student.py scripts/dexh13_lightbulb_student_consistency_codrive.sh scripts/run_consistency_g2_validate_codrive.sh scripts/run_consistency_g4c_selector_codrive.sh
```

## Smoke

Smoke run:

```text
g5d_lr_decay_smoke_20260602_092220
```

Artifact path:

```text
outputs/local_probe_consistency_g5d_smoke/g5d_lr_decay_smoke_20260602_092220
```

Smoke protocol:

```text
TRAIN_SEED=44
STUDENT_MAX_AGENT_STEPS=128
NUM_ENVS=4
EVAL_STEPS=8
EVAL_SEEDS=42
CONSISTENCY_TRAIN_ALIGN_INFER=True
CONSISTENCY_LR_SCHEDULE=linear_decay
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=32
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=96
CONSISTENCY_LR_FINAL_SCALE=0.1
```

Result:

```text
train exit_status=0
eval8 reward=0.847793
done=0.000000
latent_mse=0.191017
action_mse=0.120714
```

Interpretation:

- This is a code-path and config-wiring smoke only.
- The eval8 reward is not algorithm-quality evidence and must not be compared
  against G5C, PAdapt, Flow, or consistency baselines.
- The smoke verifies that Hydra overrides reach training and that the LR
  schedule is logged.

Observed LR log:

| Agent step | Logged consistency LR |
| ---: | ---: |
| 32 | 0.00030000 |
| 64 | 0.00018187 |
| 96 | 0.00004687 |
| 128 | 0.00003000 |

The `96`-step log is above the final LR because the LR update happens before
the batch increments `agent_steps`, while the progress line prints after the
batch. The final scale is reached on the next printed step. This is acceptable
for G5D; if exact boundary reporting becomes important, move LR update/logging
after the step counter update in a separate small patch.

## Three-Layer Training Allocation

G5D follows the local three-layer allocation:

- Smoke: tiny run to check code, config, logging, shape, restore, and summary
  emission. It cannot prove a configuration is better.
- Scout: nontrivial local train budget. For G5D, this is the `450k` seed-44 run
  with eval512 seeds `42,43,44`.
- Formal: multi-train-seed validation and eval2048 only after scout evidence is
  strong enough.

## Claude Opus Cross-Review

Status: `APPROVE`.

Review summary:

- The scheduler interpolation and cosine/linear decay implementation are
  correct for the intended default-off configuration path.
- Updating the optimizer LR before loss/backprop is appropriate for the current
  training step.
- The 128-step smoke is sufficient as a wiring check and the low eval8 reward is
  irrelevant for algorithm-quality interpretation.
- The proposed seed-44 `450k` scout with linear decay from `250k` to `450k` and
  final scale `0.1` is a reasonable low-cost probe.

Required scout metrics:

- eval-select rows at `100k` intervals;
- TensorBoard/logged `consistency_lr`;
- eval512 `action_mse_to_teacher`;
- `consistency_loss` and `bc_loss` trajectories;
- `consistency_grad_norm`.

## Scout Result

Run tag:

```text
g5d_lrdecay_s44_20260602_092610
```

Artifact roots:

```text
outputs/local_probe_consistency_g5d_lrdecay/g5d_lrdecay_s44_20260602_092610
outputs/local_eval_consistency_g5d_lrdecay/g5d_lrdecay_s44_20260602_092610
```

Protocol:

```text
TRAIN_SEEDS=44
STUDENT_MAX_AGENT_STEPS=450000
EVAL_STEPS=512
EVAL_SEEDS="42 43 44"
NUM_ENVS=16
EVAL_NUM_ENVS=8
CONSISTENCY_TRAIN_ALIGN_INFER=True
CONSISTENCY_LR_SCHEDULE=linear_decay
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=250000
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=450000
CONSISTENCY_LR_FINAL_SCALE=0.1
```

Training exit status:

```text
0
```

Internal eval-select:

| Agent steps | Eval512 reward | Done rate | Logged LR |
| ---: | ---: | ---: | ---: |
| 100k | 4.842363 | 0.001709 | 0.00030000 |
| 200k | 5.008591 | 0.000610 | 0.00030000 |
| 300k | 5.162218 | 0.001953 | 0.00023252 |
| 400k | 5.263977 | 0.000366 | 0.00009752 |
| final | 5.423057 | 0.001709 | near 0.00003000 |

Comparison to G5C train seed `44`:

| Agent steps | G5C reward | G5D reward | Delta |
| ---: | ---: | ---: | ---: |
| 100k | 4.842363 | 4.842363 | +0.000000 |
| 200k | 5.008591 | 5.008591 | +0.000000 |
| 300k | 5.172998 | 5.162218 | -0.010780 |
| 400k | 5.034783 | 5.263977 | +0.229194 |

The unchanged 100k/200k rows confirm that pre-decay behavior is reproducible.
The 400k row is the main directional evidence that LR decay mitigates the
late-training degradation observed in G5C.

External eval512 aggregate:

| Checkpoint kind | Mean reward | Reward std | Done mean | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_best_alias` | 5.348209 | 0.031296 | 0.000081 | 0.043561 | 0.023138 |
| `model_best_train` | 5.289931 | 0.072680 | 0.000081 | 0.049526 | 0.027875 |
| `model_last` | 5.348209 | 0.031296 | 0.000081 | 0.043561 | 0.023138 |

Best checkpoint hashes:

| Checkpoint kind | SHA256 |
| --- | --- |
| `model_best_alias` | `6689286487adf7e4fd20f95b24662e70ca709e6811d508ce87742c72b86e4967` |
| `model_best_train` | `e6a5f714ffacfbcf058cef7ed5a515aca961181f7ca35facab578fda5496ad51` |
| `model_last` | `93c7f534d7c76c95f80ba6ca6333360cbb8c4bdbc34dfd3808fb16ae9d905df0` |

Decision:

```text
SCOUT PASS / not formal
```

Rationale:

- The 400k eval-select value exceeds the G5C 400k value by `+0.229194`.
- External eval512 best mean reward `5.348209` exceeds the G5D pass gate
  `5.17`.
- Action MSE `0.023138` improves versus the G5C train seed `44` reference
  `0.037441`.
- Done mean `0.000081` stays comfortably below the `0.001` target.
- Evidence is still one train seed only, so it should not be promoted to a
  formal algorithm claim.

Resource note:

- Training with `NUM_ENVS=16` stayed near the local GPU budget.
- External eval with `EVAL_NUM_ENVS=8` produced short `86-88%` GPU-utilization
  peaks. Future local eval should default to `EVAL_NUM_ENVS=4` when desktop
  responsiveness matters.

Next action:

- Cross-review the result interpretation with Claude Opus.
- If approved, run a train-seed stability scout for seeds `42` and `43` using
  the same LR schedule before any formal eval2048 claim.

## Result Review

Claude Opus result-review verdict: `APPROVE`.

Review notes:

- `SCOUT PASS / not formal` is the right conclusion. The external eval512 mean
  reward clears the `5.17` gate with low cross-eval-seed standard deviation, but
  the run is still a single train seed.
- The internal eval-select comparison is valid because the `100k` and `200k`
  pre-decay rows exactly reproduce G5C seed `44`; this supports that the
  intervention starts at the intended point.
- Record that the `300k` G5D row is slightly below G5C (`-0.010780`) because
  LR decay has already started. Do not describe the whole curve as matching.
- `model_best_alias` and `model_last` have identical external eval summaries,
  so the run has not shown that early checkpoint selection is better than the
  final checkpoint.
- `model_best_train` lags `model_best_alias`/`model_last` by about `0.058`
  reward. Continue using post-training external eval for deploy selection.
- Done rate is excellent but not the differentiating signal; G5C was also low.
- Eval env count must be recorded whenever comparing local results. Lowering
  future local eval to `EVAL_NUM_ENVS=4` is acceptable for resource management,
  but comparisons should note the env-count change.

Subagent review:

```text
not executed
```

Reason:

```text
multi-agent spawn failed with "agent thread limit reached"
```
