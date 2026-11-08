# Metric Card G5E: LR-Decay Consistency Seed-Stability Scout

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

G5E follows the G5D seed-44 LR-decay scout pass. It tests whether the same
schedule survives additional train seeds before any formal eval2048 claim or
new architecture work.

## Starting Evidence

G5D seed `44` with linear LR decay from `250k` to `450k` produced:

| Checkpoint kind | Eval512 mean reward | Done mean | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: |
| `model_best_alias` | 5.348209 | 0.000081 | 0.043561 | 0.023138 |
| `model_best_train` | 5.289931 | 0.000081 | 0.049526 | 0.027875 |
| `model_last` | 5.348209 | 0.000081 | 0.043561 | 0.023138 |

G5D internal eval-select compared with G5C seed `44`:

| Agent steps | G5C | G5D | Delta |
| ---: | ---: | ---: | ---: |
| 100k | 4.842363 | 4.842363 | +0.000000 |
| 200k | 5.008591 | 5.008591 | +0.000000 |
| 300k | 5.172998 | 5.162218 | -0.010780 |
| 400k | 5.034783 | 5.263977 | +0.229194 |

Claude Opus reviewed the G5D result and approved the interpretation:

```text
SCOUT PASS / not formal
```

## Hypothesis

The LR decay schedule fixes a real late-training instability in the
train-aligned BC consistency path, not merely a lucky seed-44 outcome.

## Protocol

Run the same G5D schedule on train seeds `42` and `43`:

```text
TRAIN_SEEDS="42 43"
STUDENT_MAX_AGENT_STEPS=450000
EVAL_STEPS=512
EVAL_SEEDS="42 43 44"
NUM_ENVS=16
EVAL_NUM_ENVS=4
CONSISTENCY_TRAIN_ALIGN_INFER=True
CONSISTENCY_LR_SCHEDULE=linear_decay
CONSISTENCY_LR_DECAY_START_AGENT_STEPS=250000
CONSISTENCY_LR_DECAY_END_AGENT_STEPS=450000
CONSISTENCY_LR_FINAL_SCALE=0.1
```

Use local GPU only. `EVAL_NUM_ENVS=4` is used to keep eval peaks below the
desktop budget; record this when comparing against G5D seed `44`, which used
`EVAL_NUM_ENVS=8`.

## Gates

G5E directional PASS if:

- at least one additional train seed has external eval512 best mean reward
  `>=5.17`;
- action MSE does not regress versus G5C train seed `44` best alias
  (`0.037441`);
- done mean stays near `0.001` or lower;
- 400k/final eval-select does not repeat the G5C late-collapse pattern.

G5E strong PASS if:

- both train seeds `42` and `43` clear `>=5.17`;
- at least one clears `>=5.30`;
- best external action MSE stays `<=0.030`;
- `model_last` or eval-selected alias is stable enough to justify formal
  eval2048 planning.

G5E TUNE if:

- one seed clears gates while the other remains in the `5.05-5.17` band;
- reward improves but external selector disagreement or action MSE suggests a
  schedule tweak is needed;
- eval results are sensitive to checkpoint kind.

G5E REJECT if:

- both train seeds remain below `5.05`;
- late collapse remains visible at `400k/450k`;
- action MSE regresses above `0.037441` despite reward improvement.

## Review Requirements

- Confirm no reward/horizon/eval seed/teacher changes.
- Record local GPU peaks, especially with `EVAL_NUM_ENVS=4`.
- Use post-training external eval over `model_best`, `model_best_train`, and
  `model_last` for decision-making.
- Use Claude Opus cross-review before promoting to formal or changing
  architecture.
- Use subagent review if agent capacity is available.

## Decision

Pending.
