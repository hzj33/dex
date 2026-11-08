# Metric Card G4C: Consistency Selector Formal Validation

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

This round validates checkpoint selection for standalone
`ConsistencyLatentStudent`. It does not change architecture, loss, task reward,
horizon, reset logic, teacher checkpoint, DOTPG, or private repo flows.

## Starting Evidence

G4 proved the checkpoint aliasing hook worked mechanically but used an unsafe
selector metric:

```text
eval_select.num_steps=256
eval_select.done_penalty=2000.0
selected external eval512 mean=4.457390
```

G4B changed only selector metric and final-hook wiring:

```text
eval_select.num_steps=512
eval_select.done_penalty=0.0
train seed=43
max agent steps=450k
```

G4B external eval512 summary:

| Checkpoint | Mean reward | Done mean | action_mse_to_teacher |
| --- | ---: | ---: | ---: |
| `model_best_eval` / `model_best` | 5.137090 | 0.000203 | 0.042884 |
| `model_last` | 5.106660 | 0.000081 | 0.036529 |
| `model_best_train` | 5.171160 | 0.000203 | 0.039373 |

Interpretation:

```text
G4B is a meaningful selector scout, not a smoke test and not a formal pass.
```

## Hypothesis

A reward-first 512-step in-training selector is good enough to preserve strong
consistency checkpoints, but it still needs multi-train-seed validation. If
`model_best_train` or `model_last` repeatedly beats `model_best_eval`, then the
workflow should switch to a post-training external eval selector for deploy
aliases.

## Training Layers

Use the three-layer allocation:

| Layer | Budget | Purpose | Claim allowed |
| --- | --- | --- | --- |
| Smoke | tiny train/eval | code path, shape, checkpoint, summary emission | no algorithm claim |
| Scout | one train seed or short multi-seed, eval256/eval512 | reject or promote a direction | direction-level evidence |
| Formal | multi train seeds, longer horizon, eval2048 when justified | deployment/final comparison gate | formal result |

G4B was scout-level. G4C should not use a short run as final evidence.
Short training windows such as 30 minutes are only scout evidence unless they
are followed by the fixed-step multi-seed evaluation ladder. They are useful for
detecting whether a configuration is alive, but not for declaring that one
diffusion configuration is better than another.

## Probe A: Multi-Seed Selector Validation

Run train seeds `42,43,44` with the G4B selector settings:

```text
eval_select.enabled=True
eval_select.interval_agent_steps=100000
eval_select.min_agent_steps=100000
eval_select.num_steps=512
eval_select.done_penalty=0.0
eval_select.final_eval=True
eval_select.save_deploy_best=True
```

Default local budget:

```text
max agent steps: 450k per train seed
train timeout: 4200 seconds per train seed
num envs: 16
minibatch: 192
external eval: eval512 seeds 42,43,44
eval timeout: 900 seconds per eval seed
```

For each train seed, externally evaluate:

```text
stage2_consistency_nn/model_best.ckpt        # selected alias; hash-check against model_best_eval
stage2_consistency_nn/model_best_train.ckpt  # scalar training-reward best
stage2_consistency_nn/model_last.ckpt        # final checkpoint
```

`model_best_eval.ckpt`, `model_best.ckpt`, and `model_best_deploy.ckpt` must be
hash-compared for every train seed. If the hashes diverge, explicitly evaluate
`model_best_eval.ckpt` as a fourth entry and treat aliasing as failed.

Primary metric:

```text
external eval512 reward mean over eval seeds 42,43,44
```

Secondary metrics:

- done rate;
- latent MSE/L1;
- action MSE to teacher;
- checkpoint hash and selected step;
- internal eval-select rank or nearest-rank proxy for `model_best_train`;
- whether final eval-select row appears;
- train/eval exit status;
- dirty worktree provenance, because G4B/G4C depend on local uncommitted
  final-eval and progress-log workflow patches;
- GPU utilization / VRAM pressure around `80%`, with sustained `>85%` avoided.

## Gate After Probe A

PASS to formal if:

- selected `model_best_eval` checkpoints average near `5.1+`;
- selected checkpoints are not consistently worse than `model_best_train` or
  `model_last`;
- no train seed falls below the PAdapt fixed-step reference;
- final eval-select rows are emitted and do not corrupt aliases.

TUNE if:

- `model_best_eval` is sometimes good but frequently below train-best or last;
- the selected step is unstable while external eval can identify a better
  checkpoint.

REJECT in-training selector as deploy policy if:

- `model_best_train` or `model_last` clearly and repeatedly beats
  `model_best_eval`;
- then use a post-training external eval512 selector wrapper to choose deploy
  aliases.

If `model_best_train` repeatedly wins externally but was close to the selected
checkpoint internally, prefer a top-K candidate policy: save the top two or three
internal selector candidates and run external eval512 before assigning deploy
aliases. If train-best ranks poorly internally but wins externally, treat this as
a selector metric-alignment failure.

## Probe B: Formal Gate

Only if Probe A passes or an external selector policy is chosen:

```text
train seeds: 42,43,44
max agent steps: 900k or matched longer local budget
formal eval: eval2048 seeds 42,43,44
```

Formal comparison set:

- PAdapt fixed-step reference;
- existing consistency G2/G3/G4B results;
- flow matching reference;
- diffusion latent reference;
- teacher upper-bound reference if available.

Do not start LR, EMA, NFE, robustness, or residual-consistency changes until
G4C decides the selector policy.

## Claude And Subagent Review

Before running Probe A:

- Use a subagent for a read-only protocol/artifact audit.
- Use Claude Opus to review the metric card and overclaim risks.
- If the local `claude -p --model opus` call times out, record that failure and
  continue only after subagent review passes, then repeat Opus review after
  results when the CLI is responsive.

After results:

- Use a subagent for result-table sanity checking.
- Use Claude Opus to classify the decision as `PASS`, `TUNE`, or `REJECT`.

## Command Template

```text
GPU=0 \
TRAIN_SEEDS="42 43 44" \
SUITE_TAG=g4c_consistency_selector_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g4c \
EVAL_ROOT=outputs/local_eval_consistency_g4c \
TRAIN_WINDOW_SEC=4200 STUDENT_MAX_AGENT_STEPS=450000 \
STUDENT_PROGRESS_LOG_INTERVAL=100000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" EVAL_TIMEOUT_SEC=900 \
bash scripts/run_consistency_g4c_selector_codrive.sh
```

The runner internally calls `scripts/run_consistency_g2_validate_codrive.sh`
with the G4B selector overrides, then evaluates fallback checkpoints with
`scripts/eval_consistency_codrive_checkpoint.sh`.

Expected outputs:

```text
outputs/local_probe_consistency_g4c/<suite_tag>/suite_manifest.env
outputs/local_probe_consistency_g4c/<suite_tag>/checkpoint_eval_index.tsv
outputs/local_probe_consistency_g4c/<suite_tag>/eval_aggregate.tsv
outputs/local_probe_consistency_g4c/<suite_tag>/selected/<run_tag>/eval_summary.tsv
outputs/local_eval_consistency_g4c/<suite_tag>/<run_tag>_<checkpoint>_eval512/eval_summary.tsv
outputs/Dexh13HoraLightbulb_student_consistency_codrive/<run_tag>/eval_select/train_eval_history.tsv
```

## Preflight Notes

Subagent preflight verdict:

```text
reviewer=Hypatia
verdict=TUNE before launch
```

Claude Opus preflight status:

```text
claude -p --model opus timed out twice without output, including a 90-second
bounded retry. Do not count this as completed Opus review; repeat Opus review
after results or when the local CLI is responsive.
```

Actions taken before launch:

- Exact command values were fixed in this card and in
  `scripts/run_consistency_g4c_selector_codrive.sh`.
- The runner records dirty worktree provenance in `suite_manifest.env`.
- Fallback checkpoint evals are explicit for `model_best_train` and `model_last`;
  selected `model_best` is evaluated by the main validation script and
  hash-checked against `model_best_eval`.
- Aggregate fields are predefined in `eval_aggregate.tsv`.

## Probe A Results So Far

Decision:

```text
TUNE / external deploy selector
```

Subagent result audit:

```text
reviewer=Hypatia
verdict=TUNE / external deploy selector
```

Claude Opus result audit:

```text
verdict=TUNE
```

Supported claims:

- train-only/internal selector validation completed for train seeds `42,43,44`;
- all three selected checkpoints were strong under the internal eval-select
  metric;
- final-row handling worked: seeds `42` and `43` did not overwrite a better
  earlier checkpoint, while seed `44` saved the final row because it was best;
- aliasing worked for all seeds:
  `model_best == model_best_eval == model_best_deploy`.

Internal selector table:

| Train seed | Run tag | Selected step | Internal selected reward | Done | Final reward | Final saved |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 42 | `g4c_consistency_selector_20260602_032255_train_s42` | 400k | 5.144608 | 0.000610 | 4.899646 | 0 |
| 43 | `g4c_consistency_trainonly_20260602_040200_s43` | 400k | 5.269734 | 0.000122 | 4.959214 | 0 |
| 44 | `g4c_consistency_trainonly_20260602_043500_s44` | final / 450k | 5.250534 | 0.001465 | 5.250534 | 1 |

Internal selected aggregate:

```text
reward mean ~= 5.221625
done mean   ~= 0.000732
```

Alias hash prefixes:

| Train seed | `model_best == model_best_eval == model_best_deploy` |
| ---: | --- |
| 42 | `141eb7051732` |
| 43 | `4318371a0fc5` |
| 44 | `df90ce5a91f8` |

First completed external eval:

| Train seed | Checkpoint | Eval steps | Eval seeds | Mean reward | Std reward | Done mean | latent_mse | latent_l1 | action_mse_to_teacher |
| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | `model_best` selected alias | 512 | `42,43,44` | 5.083858 | 0.013228 | 0.000081 | 0.042133 | 0.123948 | 0.036857 |

Resource note:

- The original all-in-one G4C runner was paused during fallback external evals
  because local GPU utilization stayed around `85-87%`.
- Follow-up low-env resource probes at `EVAL_NUM_ENVS=12` and `8` still reached
  about `86%`.
- The `EVAL_NUM_ENVS=8` probe produced a log-only seed42 line
  (`avg_reward=5.228718`) but was killed before `eval_summary.tsv` was appended;
  do not use it as aggregate evidence.
- After the user moved private work to cloud, local GPU can be used more
  aggressively, but G4C should still remain serialized and provenance-logged.

Current interpretation:

- G4C has validated the internal selector/aliasing mechanics across train seeds.
- G4C has not yet validated deploy checkpoint policy.
- Do not claim that `model_best_eval` beats `model_best_train` or `model_last`.
- Do not start LR, EMA, NFE, robustness, or residual-consistency work until
  external checkpoint comparison is complete.

Required completion step:

Externally evaluate the existing three train runs without retraining:

```text
model_best       # selected alias
model_best_train # scalar training-reward best
model_last       # final checkpoint
```

for train seeds `42,43,44` under eval512 seeds `42,43,44`, then classify the
deploy policy as `PASS`, `TUNE`, or `REJECT`.

## Probe A Completion Results

Completion suite:

```text
outputs/local_probe_consistency_g4c_external/g4c_existing_checkpoint_eval_20260602_051014
```

Completeness:

- `27/27` per-eval rows are present:
  `3 train seeds * 3 checkpoint kinds * 3 eval seeds`.
- All rows have `status=0` and `steps=512`.
- All 9 checkpoint-level eval jobs have `eval_status=0`.
- `model_best`, `model_best_eval`, and `model_best_deploy` hashes match for
  train seeds `42,43,44`.
- Seed `44` has different hashes for `model_best` and `model_last`, but their
  eval512 metrics are identical. Treat them as eval-equivalent, not
  hash-identical.

Checkpoint-kind aggregate:

| Checkpoint kind | n | Mean reward | Std reward | Done mean | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_best` | 9 | 5.120598 | 0.061803 | 0.000122 | 0.043359 | 0.124445 | 0.041794 |
| `model_best_train` | 9 | 5.037163 | 0.154209 | 0.000190 | 0.045361 | 0.130432 | 0.043618 |
| `model_last` | 9 | 5.132772 | 0.065474 | 0.000122 | 0.041474 | 0.119785 | 0.041956 |

Per-train-seed external winner:

| Train seed | External eval512 winner | Mean reward | Done mean |
| ---: | --- | ---: | ---: |
| 42 | `model_last` | 5.150810 | 0.000203 |
| 43 | `model_best_train` | 5.171160 | 0.000203 |
| 44 | `model_best` / `model_last` eval-equivalent | 5.140846 | 0.000081 |

Post-training external selector aggregate:

```text
n=3
mean_reward=5.154272
std_reward=0.012615
done_mean=0.000162
latent_mse=0.042451
action_mse_to_teacher=0.042904
```

Interpretation:

- `model_best_train` is not safe as a default deploy alias. It wins seed `43`
  but is the weakest fixed kind overall and is especially poor on seed `44`.
- In-training `model_best` is no longer actively harmful, but it is not
  reliably the best deploy checkpoint.
- `model_last` has the best fixed-kind aggregate, but the margin over
  `model_best` is only about `0.012`, so do not claim a clear statistical win.
- The most defensible policy is a post-training external eval512 selector over
  `model_best`, `model_best_train`, and `model_last`.
- In-training eval-select should remain a candidate generator and aliasing
  mechanism, not the final deploy decision.

Decision:

```text
G4C = TUNE
deploy policy = post-training external eval selector
```

Current deploy-candidate assignment for this suite:

| Train seed | Candidate for formal follow-up |
| ---: | --- |
| 42 | `model_last` |
| 43 | `model_best_train` |
| 44 | `model_best` or `model_last`; prefer `model_best` for alias clarity |

Overclaims to avoid:

- Do not call G4C a formal algorithm PASS.
- Do not say in-training `model_best` is the best checkpoint.
- Do not say `model_last` clearly beats `model_best`; the fixed-kind advantage
  is small.
- Do not say seed `44` `model_best == model_last`; the hashes differ.
- Do not move to LR, EMA, NFE, robustness, or residual-consistency before the
  external-selector policy has a formal eval2048 follow-up.

Remaining provenance caveat:

```text
The public diffusion worktree does not currently contain an exact CoDrive PAdapt
fixed-step eval artifact for the handoff anchor. The active gate still uses the
documented PAdapt reference of about 4.72 until that artifact is recovered or
rerun. G4C is selector-policy validation; it is not the final baseline table.
```
