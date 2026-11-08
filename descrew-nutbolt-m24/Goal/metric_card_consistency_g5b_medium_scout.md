# Metric Card G5B: Consistency Train-Aligned BC Medium Scout

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

This round continues G5 train-aligned action BC for standalone
`ConsistencyLatentStudent`. It does not change the PPO teacher, task reward,
horizon, eval seeds, reset logic, private repo, or DOTPG flow.

## Starting Evidence

G5 smoke verified the intended gradient path:

```text
train_align_latent_requires_grad/frame=1.0
consistency_grad_norm/frame=7.880384
```

G5 directional scout, train seed `43`, `450k` agent steps:

| Checkpoint kind | Eval512 mean reward | Done mean | Latent MSE | Action MSE to teacher |
| --- | ---: | ---: | ---: | ---: |
| `model_best` alias | 5.110040 | 0.000041 | 0.069509 | 0.037663 |
| `model_best_train` | 5.181952 | 0.000122 | 0.061416 | 0.033761 |
| `model_last` | 5.197552 | 0.000163 | 0.069851 | 0.035010 |

Relevant anchors:

| Anchor | Protocol | Reward |
| --- | --- | ---: |
| G4C seed43 `model_best_train` | eval512 seeds 42/43/44 | 5.171160 |
| G4D external-selected formal anchor | eval2048 train/eval seeds 42/43/44 | 5.088728 |

Review conclusion from G5:

```text
DIRECTIONAL PASS / TUNE
```

The mechanism works, but seed43 at `450k` is not formal evidence. The strongest
checkpoint was `model_last`, suggesting the run may not have plateaued.

## Hypothesis

If train-aligned action BC is genuinely useful, extending the same seed from
`450k` to `800k` agent steps should either:

- show a clearer reward separation from G4C/G5 seed43 anchors; or
- reveal that the G5 gain was a short-run/checkpoint-selection fluctuation.

This is a medium scout, not a new algorithmic change.

## Protocol

Use the existing G4C selector runner so checkpoint comparison remains consistent:

```text
GPU=0 TRAIN_SEEDS="43" \
SUITE_TAG=g5b_alignbc_s43_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g5b_alignbc \
EVAL_ROOT=outputs/local_eval_consistency_g5b_alignbc \
TRAIN_WINDOW_SEC=7200 STUDENT_MAX_AGENT_STEPS=800000 \
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
outputs/local_probe_consistency_g5b_alignbc/<suite_tag>/suite_manifest.env
outputs/local_probe_consistency_g5b_alignbc/<suite_tag>/checkpoint_eval_index.tsv
outputs/local_probe_consistency_g5b_alignbc/<suite_tag>/eval_aggregate.tsv
outputs/local_eval_consistency_g5b_alignbc/<suite_tag>/*/eval_summary.tsv
```

## Gates

Medium-scout PASS if:

- external best eval512 mean is at least `5.23`, roughly `+0.05` over the G4C
  seed43 `model_best_train` anchor;
- done mean remains near `0.001` or lower;
- action MSE does not degrade versus G5;
- latent MSE does not worsen enough to suggest decode mismatch.

Medium-scout TUNE if:

- external best remains in the `5.17-5.23` band;
- `model_last` stays best without plateau evidence;
- reward is stable but latent MSE remains clearly worse than G4D/G4C.

Medium-scout REJECT if:

- external best falls below `5.17`;
- fixed-step reward regresses toward the PAdapt reference;
- done/action quality worsens clearly.

## Next Decision

If G5B passes:

- expand train seeds `42` and `44` with the same settings;
- keep post-training external selection over `model_best`, `model_best_train`,
  and `model_last`;
- only then consider formal eval2048.

If G5B is flat:

- test seed stability at `450k` before adding new loss terms;
- consider lowering `bc_loss_coef` only after seed-stability evidence.

If G5B rejects:

- keep the G5 patch as an optional diagnostic flag;
- return to G4D standalone-consistency anchor and choose another metric card.

## Review Requirements

Before running:

- subagent preflight audit for protocol, teacher, and leakage boundaries;
- static checks on the modified consistency code and runner script.

Preflight review notes:

- Hypatia verdict: aligned and runnable as medium scout; prefer `800k` /
  `7200s` before `1M` to test separation without over-tuning one seed.
- Claude Opus verdict: proceed; flag train/eval seed overlap for seed `43`, and
  do not attribute any improvement solely to the G5 flag without a matched-step
  `consistency_train_align_infer=False` arm later.

After running:

- subagent result sanity review;
- Claude Opus result interpretation review;
- update `Goal/diffusion_goal_plan.md` and this metric card with artifacts and
  decision.

## Run Log

Run:

```text
suite_tag=g5b_alignbc_s43_20260602_064936
train_run=g5b_alignbc_s43_20260602_064936_train_s43
train_seed=43
student_max_agent_steps=800000
train_window_sec=7200
num_envs=16
eval_steps=512
eval_seeds=42,43,44
consistency_train_align_infer=True
```

Training exited with status `0`.

Internal eval-select history:

| agent_steps | avg_reward | avg_done_rate | best_score | note |
| ---: | ---: | ---: | ---: | --- |
| 100000 | 4.987697 | 0.001587 | 4.987697 | new best |
| 200000 | 5.146397 | 0.000488 | 5.146397 | new best |
| 300000 | 4.974154 | 0.001465 | 5.146397 | no save |
| 400000 | 5.002417 | 0.000854 | 5.146397 | no save |
| 500000 | 5.263258 | 0.001099 | 5.263258 | new best |
| 600000 | 5.188082 | 0.000977 | 5.263258 | no save |
| 700000 | 5.275939 | 0.001465 | 5.275939 | new best |
| 800000 | 5.286277 | 0.001465 | 5.286277 | new best |

Final training best reward:

```text
3610.48
```

External checkpoint comparison:

| Checkpoint kind | Eval512 mean reward | Reward std | Done mean | Latent MSE | Action MSE to teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_best` alias | 5.218559 | 0.085197 | 0.000081 | 0.076395 | 0.027387 |
| `model_best_train` | 5.156018 | 0.038688 | 0.000122 | 0.067694 | 0.030968 |
| `model_last` | 5.218559 | 0.085197 | 0.000081 | 0.076395 | 0.027387 |

Per-seed external best rows:

| Checkpoint kind | Eval seed | Reward | Done rate | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_best` alias | 42 | 5.106561 | 0.000122 | 0.077962 | 0.028320 |
| `model_best` alias | 43 | 5.313029 | 0.000122 | 0.074691 | 0.025618 |
| `model_best` alias | 44 | 5.236088 | 0.000000 | 0.076531 | 0.028222 |
| `model_last` | 42 | 5.106561 | 0.000122 | 0.077962 | 0.028320 |
| `model_last` | 43 | 5.313029 | 0.000122 | 0.074691 | 0.025618 |
| `model_last` | 44 | 5.236088 | 0.000000 | 0.076531 | 0.028222 |

Checkpoint hashes:

```text
model_best/model_best_eval/model_best_deploy:
326e5ae301d1bd48b31b22703bfb5c9b91bd0686f0601498db55a891e1992462

model_best_train:
276d43576dba6b75808f8d201a3474c661890dd13d55afa9d2145afd3a158398

model_last:
37249b6d7d8c8986e50a4562d08983c48eb74a7d4984af12b07a75ed3d37d47a
```

Artifact pointers:

```text
suite_dir=outputs/local_probe_consistency_g5b_alignbc/g5b_alignbc_s43_20260602_064936
eval_aggregate=outputs/local_probe_consistency_g5b_alignbc/g5b_alignbc_s43_20260602_064936/eval_aggregate.tsv
checkpoint_eval_index=outputs/local_probe_consistency_g5b_alignbc/g5b_alignbc_s43_20260602_064936/checkpoint_eval_index.tsv
```

## Result Reviews

Subagent sanity review:

```text
reviewer=Hypatia
verdict=TUNE at medium-scout level, near-pass but not PASS
```

Key notes:

- Data is complete: train exit `0`, all external eval rows status `0`.
- External best is `5.218559`, missing the `5.23` PASS gate by `0.011441`.
- Improvement over G5 450k `model_last` is only `+0.021007`; improvement over
  the G4C seed43 anchor `5.171160` is `+0.047399`, just below the intended
  `~+0.05` separation.
- Latent MSE worsens to `0.076395`, so the reward/action gain carries a
  decode/latent-quality caveat.
- `model_best` alias and `model_last` are eval-equivalent but not hash-identical.
- Recommended next step: run train seeds `42` and `44` at the cheaper `450k`
  G5 budget before formal eval or more same-seed extension.

Claude Opus result review:

```text
verdict=TUNE
```

Key notes:

- External best `5.218559` lands in the TUNE band and misses PASS by `0.011`.
- The final checkpoint is strongest by eval, so the same seed may not have
  plateaued at `800k`.
- Comparing G5B to G5 is confounded by longer training.
- Recommended next step from this review: extend the same seed to about `1.2M`
  before pivoting.

## Decision

Medium-scout decision:

```text
TUNE / near-pass
```

Reasoning:

- G5B improves over G5 450k and confirms that longer training with the
  train-aligned BC path can recover stronger fixed-step eval.
- It does not pass the predeclared `5.23` external-best threshold.
- It remains one train seed, and eval seed `43` overlaps the train seed.
- The latent MSE caveat got worse even though action MSE improved.

Next step:

```text
G5C seed-stability scout: train seeds 42 and 44, same G5 settings, 450k agent
steps, eval512 seeds 42/43/44, external selector over model_best/model_best_train/model_last.
```

Rationale:

- This follows the local workflow by checking train-seed stability before formal
  validation.
- It is cheaper and less likely to over-tune seed `43` than immediately
  extending the same seed to `1.2M`.
- If seeds `42` and `44` are stable and near the seed43 G5/G5B band, rerun the
  strongest setting at `800k` for those seeds or move to formal eval2048.
- If either seed is weak, tune `bc_loss_coef` or the train-alignment objective
  before spending more long-run compute.
