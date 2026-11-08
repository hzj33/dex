# Metric Card G5C: Consistency Train-Aligned BC Seed Stability

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

This round tests whether the G5 train-aligned action BC signal is stable across
train seeds. It does not change task reward, horizon, eval seeds, teacher,
private repo, or DOTPG flow.

## Starting Evidence

G5 seed43, `450k`:

| Checkpoint kind | Eval512 mean reward | Done mean | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: |
| `model_best` alias | 5.110040 | 0.000041 | 0.069509 | 0.037663 |
| `model_best_train` | 5.181952 | 0.000122 | 0.061416 | 0.033761 |
| `model_last` | 5.197552 | 0.000163 | 0.069851 | 0.035010 |

G5B seed43, `800k`:

| Checkpoint kind | Eval512 mean reward | Done mean | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: |
| `model_best` alias | 5.218559 | 0.000081 | 0.076395 | 0.027387 |
| `model_best_train` | 5.156018 | 0.000122 | 0.067694 | 0.030968 |
| `model_last` | 5.218559 | 0.000081 | 0.076395 | 0.027387 |

G5B decision:

```text
TUNE / near-pass
```

Review consensus:

- Do not promote to formal eval yet.
- Main unresolved risk is train-seed stability.
- A same-seed extension is informative but risks over-tuning train seed `43`.

## Hypothesis

If train-aligned action BC is a robust optimization direction, train seeds `42`
and `44` at the cheaper `450k` scout budget should stay near or above the G4C/G5
seed43 anchor band under the same external eval protocol.

## Protocol

Evidence layer:

```text
Scout, not formal validation.
```

This run is longer than a smoke test and is intended to expose whether the G5B
seed43 gain survives train-seed changes. It is still not a final algorithm
comparison. Promotion to formal validation requires useful external eval512
evidence first, followed by train seeds `42,43,44` and fixed-step eval2048 under
the shared protocol.

Use the existing G4C selector runner:

```text
GPU=0 TRAIN_SEEDS="42 44" \
SUITE_TAG=g5c_alignbc_s42_s44_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g5c_alignbc \
EVAL_ROOT=outputs/local_eval_consistency_g5c_alignbc \
TRAIN_WINDOW_SEC=4200 STUDENT_MAX_AGENT_STEPS=450000 \
STUDENT_PROGRESS_LOG_INTERVAL=100000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
EVAL_TIMEOUT_SEC=900 \
CONSISTENCY_TRAIN_ALIGN_INFER=True \
GPU_UTIL_MAX=85 GPU_MEM_PCT_MAX=85 \
bash scripts/run_consistency_g4c_selector_codrive.sh
```

Expected artifacts:

```text
outputs/local_probe_consistency_g5c_alignbc/<suite_tag>/suite_manifest.env
outputs/local_probe_consistency_g5c_alignbc/<suite_tag>/checkpoint_eval_index.tsv
outputs/local_probe_consistency_g5c_alignbc/<suite_tag>/eval_aggregate.tsv
outputs/local_eval_consistency_g5c_alignbc/<suite_tag>/*/eval_summary.tsv
```

## Gates

Seed-stability PASS if:

- both train seeds have an external-best eval512 mean at least `5.17`;
- at least one seed reaches or exceeds `5.20`;
- done mean remains near `0.001` or lower;
- action MSE does not regress versus G5/G5B;
- no seed falls near the PAdapt reference band.

Seed-stability TUNE if:

- one seed is strong but the other is only in the `5.05-5.17` range;
- reward is stable but latent MSE remains high;
- selector alias differs but external selection recovers a usable checkpoint.

Seed-stability REJECT if:

- either train seed falls below `5.05`;
- external-best reward regresses toward PAdapt/Flow;
- done/action quality worsens clearly.

## Next Decision

If G5C passes:

- choose whether to rerun seeds `42` and `44` at `800k`, or go directly to
  formal eval2048 with the external-selected candidates.

If G5C tunes:

- ablate `bc_loss_coef` or train-alignment strength before more long runs.

If G5C rejects:

- keep G5/G5B as seed43-only evidence;
- return to G4D standalone consistency as the current stable anchor.

## Results

Suite:

```text
outputs/local_probe_consistency_g5c_alignbc/g5c_alignbc_s42_s44_20260602_075709
outputs/local_eval_consistency_g5c_alignbc/g5c_alignbc_s42_s44_20260602_075709
```

Internal eval-select:

| Train seed | 100k | 200k | 300k | 400k | Selected best |
| --- | ---: | ---: | ---: | ---: | ---: |
| `42` | 4.811486 | 4.906337 | 4.931985 | 4.433161 | 4.931985 |
| `44` | 4.842363 | 5.008591 | 5.172998 | 5.034783 | 5.172998 |

External eval512 aggregate:

| Train seed | Checkpoint kind | Mean reward | Reward std | Done mean | Latent MSE | Action MSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `42` | `model_best_alias` | 4.960817 | 0.039809 | 0.000041 | 0.071036 | 0.039160 |
| `42` | `model_best_train` | 4.928148 | 0.112883 | 0.000203 | 0.087257 | 0.040332 |
| `42` | `model_last` | 4.839347 | 0.228928 | 0.000122 | 0.074092 | 0.042069 |
| `44` | `model_best_alias` | 5.102461 | 0.092682 | 0.000122 | 0.070952 | 0.037441 |
| `44` | `model_best_train` | 5.049980 | 0.082516 | 0.000163 | 0.077853 | 0.041135 |
| `44` | `model_last` | 4.984702 | 0.085569 | 0.000203 | 0.062614 | 0.036787 |

Checkpoint hashes:

| Train seed | Checkpoint kind | SHA256 |
| --- | --- | --- |
| `42` | `model_best_alias` | `d45acb35cf26cf9138c34cd8fe3bbd96fb39d999bf95ad27a354792a22e892b1` |
| `42` | `model_best_train` | `6e102d6929b2ac3a39c8908d23235e828bb94d992c1064aa3516db6d46ce1aec` |
| `42` | `model_last` | `f1b1c7b06220b3629bd794f7afd45a13e43a9ba10a694f1bf297f050579cb4b4` |
| `44` | `model_best_alias` | `c7231d945e3947b816dec5f4e73e855963ea2385b979eea0a24c09d1fd84f2ec` |
| `44` | `model_best_train` | `95967c36a4170787fd6d6065265d49b2b87394a513e394f018270afb927278c0` |
| `44` | `model_last` | `4a1fa7d11a8b6f7f2c8a93793c1ee7ab4d5624e9cb29e2782d8eac5ecae3154b` |

Resource note:

- Training ran as a single local GPU job and stayed around the `75-80%` GPU
  utilization band.
- External eval at `EVAL_NUM_ENVS=16` produced short `85-88%` utilization
  peaks. Use `EVAL_NUM_ENVS=8` for future local scouts when preserving desktop
  responsiveness is more important than eval speed.

## Result Review

Claude Opus review:

```text
Recommendation: REJECT.
```

Key points:

- Train seed `42` best external mean `4.960817` is below the `5.05` REJECT
  floor.
- Train seed `44` best external mean `5.102461` is moderate but below the
  `5.17` per-seed PASS target.
- Both train seeds peak around `300k` internal eval-select and soften or
  collapse afterward, so longer training alone is unlikely to fix the issue.
- Suggested next low-cost scout: keep train-aligned BC disabled or stabilized
  through schedule changes, for example lower LR or LR decay after roughly
  `250k`, and test whether the `300k` peak can be held through `400k+`.

Subagent review:

- Requested, but unavailable in this run because the existing subagent id could
  not be parsed by the current tool and the agent thread limit prevented
  spawning a new reviewer.

## Decision

```text
REJECT as a seed-stable improvement.
```

Rationale:

- G5C cannot pass because train seed `42` falls below the `5.05` reject floor
  and far below the `5.17` stability target.
- Train seed `44` gives a useful but non-decisive signal; its best external
  mean `5.102461` is below the G5/G5B seed43 band.
- The shared pattern is a `300k` internal peak followed by degradation, so the
  next experiment should target training stability rather than longer same-config
  training.

Next scout candidate:

- G5D: one-hypothesis stability scout for the train-aligned BC path.
- Use one train seed first, preferably `44`, because it shows the strongest
  local signal.
- Keep the same task, teacher, eval seeds, and external selector.
- Test either LR decay after about `250k` or a lower consistency LR.
- Success condition: hold reward near or above `5.17` at `400k/450k` and raise
  external eval512 mean without worsening action MSE.

## Review Notes

This G5C direction comes from the G5B result reviews:

- Hypatia: prefer train seeds `42` and `44` at the cheaper `450k` G5 budget
  before formal eval or more same-seed extension.
- Claude Opus: same-seed extension may still improve, but warned that G5B vs G5
  is confounded by longer training and remains single-seed.

Result review is required after the run:

- subagent result sanity review;
- Claude Opus result interpretation review;
- update `Goal/diffusion_goal_plan.md` and this metric card.
