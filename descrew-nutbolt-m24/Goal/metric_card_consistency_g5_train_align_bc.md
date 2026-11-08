# Metric Card G5: Consistency Train-Aligned Action BC

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

This round optimizes only the standalone `ConsistencyLatentStudent` training
path. It does not change the task, reward, horizon, eval seeds, teacher
checkpoint, private repo, or DOTPG flow.

## Starting Evidence

Current anchor:

```text
G4D external-selected standalone consistency formal eval2048 mean reward=5.088728
done_mean=0.001078
```

Selector conclusion:

```text
PASS selector formal validation / TUNE algorithm upper bound
```

Residual evidence:

```text
PAdapt-base residual consistency G1 did not beat the PAdapt fixed-step reference.
Do not rerun that branch as the default next step.
```

Known code risk:

```text
Goal/metric_card_consistency_g3_seed_stability.md:
caution=consistency_train_align_infer is diagnostic only because sampled latent uses no_grad and BC will not train through it
```

This G5 card exists to test and resolve that specific diagnostic-only path. The
smoke result below verifies that the train-aligned sampled latent now carries
gradients when the explicit G5 flag is enabled.

Historical note:

```text
docs/plansv6_risk_optimize.md already listed train/infer alignment as an optional optimization axis.
Treat that as background evidence only, not as the current CoDrive mainline.
```

## Hypothesis

The consistency model is evaluated by sampling from the inference path:

```text
latent_0 = 0
latent = consistency_model(proprio_hist, latent_t, t)
a_student = frozen_actor(obs, latent)
```

But in training, action BC normally uses `pred_hi` from the random-time
consistency target rather than the actual inference rollout. The existing
`consistency_train_align_infer=True` switch attempts to use the inference path,
but that path currently calls a `@torch.no_grad()` sampling helper, so the BC
loss does not backprop through the inference-sampled latent.

Small fix:

```text
When consistency_train_align_infer=True during training, use a gradient-enabled
sampling helper for the current consistency model.
```

Expected benefit:

- reduce train/eval latent sampling mismatch;
- make action BC directly train the NFE=1 inference path;
- improve fixed-step eval without changing inference API or baseline defaults.

## Implementation Boundary

Allowed code change:

```text
dexscrew/algo/ppo/consistency_latent_student.py
```

Expected behavior:

- default `consistency_train_align_infer=False` remains byte-for-byte behaviorally
  unchanged for normal training;
- `sample_latent()` and eval inference remain `torch.no_grad`;
- only `sample_latent_train()` under `consistency_train_align_infer=True` uses
  a gradient-enabled current-model sampling path;
- no change to `dexscrew/dotpg/`, `docs/dotpg_*`, or private repo paths.

Risk controls:

- keep `CONSISTENCY_INFER_STEPS=1` for the first probe;
- do not combine with EMA, obs-noise curriculum, LR tuning, or residual changes
  in the first scout;
- record GPU usage and keep local runs serial.

## Protocol Ladder

### Smoke

Purpose:

```text
code path only; no algorithm claim
```

Command template:

```text
GPU=0 SEED=42 \
RUN_TAG=g5_alignbc_smoke_s42_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g5_alignbc_smoke \
TRAIN_WINDOW_SEC=300 STUDENT_MAX_AGENT_STEPS=16 \
NUM_ENVS=4 MINIBATCH=12 \
EVAL_STEPS=16 EVAL_NUM_ENVS=4 EVAL_SEEDS="42" \
CONSISTENCY_TRAIN_ALIGN_INFER=True \
GPU_UTIL_MAX=95 GPU_MEM_PCT_MAX=95 \
bash scripts/run_consistency_g2_validate_codrive.sh
```

Pass if:

- train exits `0` or controlled timeout after saving a checkpoint;
- eval emits `EvalSummary` and `EvalReconSummary`;
- no NaN/Inf or shape/load error;
- checkpoint can be restored.

### Scout

Purpose:

```text
single-train-seed direction screen, not a formal result
```

Use train seed `43` because previous 450k default consistency runs provide
direct selector/eval512 comparison points for this seed.

Command template:

```text
GPU=0 TRAIN_SEEDS="43" \
SUITE_TAG=g5_alignbc_s43_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g5_alignbc \
EVAL_ROOT=outputs/local_eval_consistency_g5_alignbc \
TRAIN_WINDOW_SEC=4200 STUDENT_MAX_AGENT_STEPS=450000 \
STUDENT_PROGRESS_LOG_INTERVAL=100000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
EVAL_TIMEOUT_SEC=900 \
CONSISTENCY_TRAIN_ALIGN_INFER=True \
GPU_UTIL_MAX=95 GPU_MEM_PCT_MAX=95 \
bash scripts/run_consistency_g4c_selector_codrive.sh
```

The scout reuses the G4C selector runner with only train seed `43`, so it
evaluates:

```text
model_best alias / model_best_eval
model_best_train
model_last
```

Comparison anchors:

| Anchor | Protocol | Reward |
| --- | --- | ---: |
| G4B seed43 `model_best` / eval-selected | eval512 seeds 42/43/44 | 5.137090 |
| G4B seed43 `model_best_train` | eval512 seeds 42/43/44 | 5.171160 |
| G4B seed43 `model_last` | eval512 seeds 42/43/44 | 5.106660 |
| G4D formal external-selected anchor | eval2048 all train/eval seeds | 5.088728 |

Scout PASS if:

- eval512 mean is at least `5.17`, matching or beating the best seed43 default
  scout anchor;
- done mean remains near `0.001` or lower;
- latent/action MSE do not clearly degrade versus G4B/G4D ranges.

Scout TUNE if:

- eval512 mean is between `5.05` and `5.17`;
- reward is competitive but recon/action quality worsens;
- training reaches high reward but external eval does not follow.

Scout REJECT if:

- eval512 mean falls below `5.05`;
- fixed-step reward drops near or below PAdapt reference;
- action MSE or termination clearly worsens.

### Formal

Only if scout passes:

```text
train seeds: 42,43,44
eval steps: 2048
eval seeds: 42,43,44
deploy candidate selected by post-training external eval
```

Formal target:

```text
beat G4D formal eval2048 reward 5.088728 without increasing done_mean or action jitter
```

## Review Requirements

Before code change:

- subagent read-only algorithm audit;
- Claude Opus direction review.

After smoke/scout:

- subagent result sanity check;
- Claude Opus result interpretation review;
- update `Goal/diffusion_goal_plan.md`.

## Preflight Reviews

Subagent algorithm audit:

```text
reviewer=Hypatia
verdict=prefer standalone consistency loss/alignment scout over new residual architecture
```

Key notes:

- Current `ResidualConsistencyLatentStudent` is PAdapt-base, not
  consistency-base.
- Directly loading a G4D consistency checkpoint into the residual runner would
  not make it a consistency-base residual.
- A new consistency-base residual would need a separate frozen base consistency
  model and is higher risk than this G5 training-path probe.

Claude Opus direction review:

```text
verdict=PASS for hypothesis and minimal patch
```

Key notes:

- The existing `@torch.no_grad()` sampling helper makes action BC a zero-gradient
  term for trainable consistency parameters when `consistency_train_align_infer`
  is enabled.
- The gradient-enabled train sampling path must use the online
  `consistency_model`, not the frozen EMA model.
- Keep the first probe at `CONSISTENCY_INFER_STEPS=1` and monitor grad norm.

## Implementation And Smoke Log

Implementation:

```text
dexscrew/algo/ppo/consistency_latent_student.py
```

Status:

- added a gradient-enabled train sampling helper;
- kept default `consistency_train_align_infer=False` behavior unchanged;
- added `consistency_train_align_use_ema`, default `False`;
- added TensorBoard diagnostics:
  - `train_align_infer/frame`
  - `train_align_use_ema/frame`
  - `train_align_latent_requires_grad/frame`
  - `consistency_grad_norm/frame`

Static checks:

```text
python -m py_compile dexscrew/algo/ppo/consistency_latent_student.py: pass
bash -n scripts/run_consistency_g2_validate_codrive.sh: pass
git diff --check: pass
```

Smoke run:

```text
run_tag=g5_alignbc_smoke_s42_20260602_055909
probe_dir=outputs/local_probe_consistency_g5_alignbc_smoke/g5_alignbc_smoke_s42_20260602_055909
checkpoint=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g5_alignbc_smoke_s42_20260602_055909/stage2_consistency_nn/model_best.ckpt
```

Smoke eval:

| seed | status | steps | avg_reward | avg_done_rate | latent_mse | action_mse_to_teacher |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0 | 16 | 3.897836 | 0.000000 | 0.169399 | 0.172865 |

Gradient sanity:

```text
train_align_infer/frame=1.0
train_align_use_ema/frame=0.0
train_align_latent_requires_grad/frame=1.0
consistency_grad_norm/frame=7.880384
```

Smoke decision:

```text
PASS code-path smoke; no algorithm-quality claim
```

## Scout Log

Scout run:

```text
suite_tag=g5_alignbc_s43_20260602_060054
train_run=g5_alignbc_s43_20260602_060054_train_s43
train_seed=43
student_max_agent_steps=450000
train_window_sec=4200
num_envs=16
eval_steps=512
eval_seeds=42,43,44
consistency_train_align_infer=True
```

Internal eval-select history:

| agent_steps | train_reward | avg_reward | avg_done_rate | saved_best |
| ---: | ---: | ---: | ---: | ---: |
| 100000 | 3275.069821 | 4.987697 | 0.001587 | 1 |
| 200000 | 3473.413043 | 5.146397 | 0.000488 | 1 |
| 300000 | 3498.467597 | 4.974154 | 0.001465 | 0 |
| 400000 | 3575.653824 | 5.002417 | 0.000854 | 0 |
| 450000 | 3535.069836 | 5.034296 | 0.001465 | 0 |

Post-training external checkpoint comparison:

| Checkpoint kind | Eval512 mean reward | Reward std | Done mean | Latent MSE | Action MSE to teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_best` alias | 5.110040 | 0.105208 | 0.000041 | 0.069509 | 0.037663 |
| `model_best_train` | 5.181952 | 0.048253 | 0.000122 | 0.061416 | 0.033761 |
| `model_last` | 5.197552 | 0.076452 | 0.000163 | 0.069851 | 0.035010 |

Per-seed summaries for the two external winners:

| Checkpoint kind | Eval seed | Reward | Done rate | Latent MSE | Action MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| `model_best_train` | 42 | 5.170581 | 0.000000 | 0.062511 | 0.036397 |
| `model_best_train` | 43 | 5.245909 | 0.000244 | 0.059189 | 0.031662 |
| `model_best_train` | 44 | 5.129367 | 0.000122 | 0.062549 | 0.033225 |
| `model_last` | 42 | 5.099684 | 0.000122 | 0.071442 | 0.039257 |
| `model_last` | 43 | 5.206692 | 0.000244 | 0.066795 | 0.029265 |
| `model_last` | 44 | 5.286281 | 0.000122 | 0.071316 | 0.036507 |

Artifact pointers:

```text
suite_dir=outputs/local_probe_consistency_g5_alignbc/g5_alignbc_s43_20260602_060054
eval_aggregate=outputs/local_probe_consistency_g5_alignbc/g5_alignbc_s43_20260602_060054/eval_aggregate.tsv
checkpoint_eval_index=outputs/local_probe_consistency_g5_alignbc/g5_alignbc_s43_20260602_060054/checkpoint_eval_index.tsv
```

## Result Reviews

Subagent sanity review:

```text
reviewer=Hypatia
verdict=directional scout pass through external checkpoint selection, not formal
```

Key notes:

- No obvious inference leakage. Train-aligned sampling uses `proprio_hist` and
  the online consistency model; privileged teacher state remains a training
  label source only.
- External `model_best_train` and `model_last` beat the seed43 G4C anchor
  `5.171160`, but the in-training `model_best` alias does not.
- Done rates are fine.
- Latent MSE is worse than the G4D/G4B range near `0.04`; action MSE is good, so
  this is a tuning signal rather than an immediate reject.
- Preferred next step from this review: train seeds `42` and `44` with the same
  budget and settings before BC-weight or anchor ablations.

Claude Opus result review:

```text
verdict=TUNE
```

Key notes:

- The mechanism is confirmed: action BC is alive for the consistency head in
  train-aligned mode.
- G5 `model_best_train` is only `+0.011` above the G4C seed43
  `model_best_train` anchor, which is marginal for a single train seed.
- `model_last > model_best_train > model_best_alias` means the internal selector
  remains imperfect.
- The `model_last` result suggests this train seed may not have plateaued at
  `450k` agent steps.
- Preferred next step from this review: run an `800k-1M` seed43 medium scout
  before expanding train seeds.

## Decision

Scout-level decision:

```text
DIRECTIONAL PASS for the G5 mechanism and external-selector scout;
TUNE before promotion to formal validation.
```

Reasoning:

- The code-path smoke passed and verified that the intended gradient path is
  active.
- The external best checkpoint reaches eval512 mean `5.197552`, above the
  `5.17` scout threshold and above the seed43 G4C anchor.
- The improvement is still single-train-seed evidence and is small enough that
  it should not be reported as a formal algorithm win.
- The in-training `model_best` alias is not the best checkpoint. Continue using
  post-training external selection over `model_best`, `model_best_train`, and
  `model_last`.
- Next work should be a medium-budget G5 validation, not a new architecture
  change.

Next recommended run:

```text
G5B medium scout: same G5 settings, train seed43, 800k-1M agent steps,
eval512 seeds 42,43,44, external selector over model_best/model_best_train/model_last.
```

Promotion rule:

- If G5B improves by at least about `+0.05` over the comparable G4C/G5 seed43
  anchors without worse done/action metrics, expand to train seeds `42` and
  `44`.
- If G5B is flat but stable, run train seeds `42` and `44` at the current `450k`
  scout budget to test stability.
- If latent MSE remains high or reward regresses, ablate action-BC strength
  before touching residual architecture.
