# Metric Card G4: Consistency Eval-Select Checkpointing

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

This round fixes checkpoint selection for standalone `ConsistencyLatentStudent`.
It does not change task reward, horizon, reset, eval seeds, teacher checkpoint,
private repo, or DOTPG.

## Starting Evidence

G2 formal seed43 originally failed because scalar training reward selected a bad
checkpoint:

| Checkpoint | Eval protocol | reward_mean | action_mse_to_teacher |
| --- | --- | ---: | ---: |
| G2 seed43 `model_best` | eval2048 seeds 42/43/44 | 4.730098 | 0.038814 |
| G2 seed43 `model_last` | eval2048 seeds 42/43/44 | 5.139005 | 0.029841 |

G3A seed43 default rerun at `450k` also recovered:

| Checkpoint | Eval protocol | reward_mean | action_mse_to_teacher |
| --- | --- | ---: | ---: |
| G3A `model_best` | eval512 seeds 42/43/44 | 5.176507 | 0.043600 |
| G3A `model_last` | eval512 seeds 42/43/44 | 5.118077 | 0.040757 |

Conclusion:

```text
The next bottleneck is checkpoint selection, not LR/EMA/architecture.
```

## Hypothesis

Enabling the existing `EvalSelectMixin` for consistency training will preserve a
fixed-step-eval-selected checkpoint and prevent scalar training reward from
overwriting the deploy/final alias with a weaker model.

## Existing Hook

`ConsistencyLatentStudent` already calls:

```text
_run_eval_select_if_due(
  "EVAL/student",
  train_reward=mean_rewards,
  eval_best_stem="model_best_eval",
  alias_stems=("model_best", "model_best_deploy"),
)
```

When eval-select is enabled, scalar training reward writes `model_best_train`
instead of `model_best`, and eval-select can alias `model_best` to
`model_best_eval`.

## Metrics

Primary:

```text
fixed-step eval reward of the selected checkpoint
```

Secondary:

- selected checkpoint stem: `model_best_eval`, `model_best_train`, or
  `model_last`;
- eval-select history reward, done_rate, and score;
- final external eval seeds `42,43,44`;
- latent MSE/L1 and action MSE to teacher;
- GPU budget, with training around `80%` and no sustained `>85%`.

## Probe

Run seed43 first:

```text
GPU=0 SEED=43 RUN_TAG=g4_consistency_seed43_evalselect_$(date +%Y%m%d_%H%M%S) \
PROBE_ROOT=outputs/local_probe_consistency_g4 \
TRAIN_WINDOW_SEC=2700 STUDENT_MAX_AGENT_STEPS=450000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/run_consistency_g2_validate_codrive.sh \
  ++train.ppo.eval_select.enabled=True \
  ++train.ppo.eval_select.interval_agent_steps=100000 \
  ++train.ppo.eval_select.min_agent_steps=100000 \
  ++train.ppo.eval_select.num_steps=256 \
  ++train.ppo.eval_select.done_penalty=2000.0 \
  ++train.ppo.eval_select.min_score_improvement=0.02 \
  ++train.ppo.eval_select.final_eval=True \
  ++train.ppo.eval_select.save_deploy_best=True
```

After the run, compare:

```text
stage2_consistency_nn/model_best.ckpt
stage2_consistency_nn/model_best_eval.ckpt
stage2_consistency_nn/model_best_train.ckpt
stage2_consistency_nn/model_last.ckpt
stage2_consistency_nn/model_best_deploy.ckpt
eval_select/train_eval_history.tsv
```

Mandatory external eval:

```text
CKPT=outputs/Dexh13HoraLightbulb_student_consistency_codrive/<run_tag>/stage2_consistency_nn/<checkpoint>.ckpt \
RUN_TAG=<run_tag>_<checkpoint>_eval512 \
EVAL_ROOT=outputs/local_eval_consistency_g4 \
EVAL_STEPS=512 EVAL_NUM_ENVS=16 EVAL_SEEDS="42 43 44" \
bash scripts/eval_consistency_codrive_checkpoint.sh
```

## Decision Rule

- PASS if eval-select `model_best` or `model_best_eval` lands near the recovered
  range: reward mean `>=5.1` on eval512 seeds `42,43,44`.
- TUNE if eval-select is valid but chooses a weaker checkpoint than `model_last`;
  adjust interval, min steps, or eval-select num steps.
- REJECT this hook only if it perturbs training or cannot preserve a better
  checkpoint than scalar training reward.

Do not run LR/EMA probes or residual variants until this selector path is
settled.

## Result

Run:

```text
g4_consistency_seed43_evalselect_20260602_014353
```

Training budget:

```text
train seed: 43
max agent steps: 450k
num envs: 16
eval-select interval: 100k agent steps
eval-select num_steps: 256
eval-select done_penalty: 2000.0
external eval: eval512 seeds 42,43,44
```

Internal eval-select history:

| Agent steps | train_reward | avg_reward | avg_done_rate | score | saved_best |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 100k | 3557.662965 | 4.588011 | 0.002930 | -1.271364 | 1 |
| 200k | 3635.556753 | 4.957870 | 0.002686 | -0.413224 | 1 |
| 300k | 3648.830180 | 4.943703 | 0.000977 | 2.990578 | 1 |
| 400k | 3639.567995 | 4.821063 | 0.000244 | 4.332782 | 1 |

Checkpoint hashes:

| Checkpoint | SHA256 prefix | Note |
| --- | --- | --- |
| `model_best.ckpt` | `01a7c2d379a8` | same as `model_best_eval` |
| `model_best_eval.ckpt` | `01a7c2d379a8` | selected by eval-select at 400k |
| `model_best_train.ckpt` | `e7a0ea834ef6` | scalar training-reward best |
| `model_last.ckpt` | `5616747f4a2` | final 450k checkpoint |

External eval512 results:

| Checkpoint | seed42 | seed43 | seed44 | mean | std | done_mean | latent_mse | action_mse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `model_best_eval` / `model_best` | 4.218256 | 4.420723 | 4.733190 | 4.457390 | 0.211814 | 0.000244 | 0.046628 | 0.047801 |
| `model_last` | 4.835059 | 5.046318 | 4.917721 | 4.933033 | 0.086923 | 0.000163 | 0.046214 | 0.045391 |
| `model_best_train` | 4.987658 | 4.822974 | 4.916865 | 4.909166 | 0.067452 | 0.000325 | 0.047362 | 0.043669 |

Decision:

```text
TUNE
```

Interpretation:

- The periodic eval-select hook is wired correctly: `model_best` and
  `model_best_eval` were aliased to the internally selected eval checkpoint,
  while `model_best_train` was preserved separately.
- The selected checkpoint failed the external fixed-step gate. Its eval512 mean
  was `4.457390`, below PAdapt reference and below `model_last`.
- The same run's `model_last` and `model_best_train` were both materially
  stronger than `model_best_eval`, which means the active failure is the
  selector metric or selector budget, not checkpoint preservation.
- This G4 run was weaker than the G3A seed43 default rerun
  (`model_best` eval512 mean `5.176507`), so it does not yet prove that
  eval-select can recover the best 5.1+ range. It does prove that the current
  `256`-step selector with a `2000` done penalty is unsafe.

Next selector probe:

- Keep architecture unchanged.
- Prefer a G4B selector-only probe before LR/EMA/NFE/residual:
  - `eval_select.num_steps=512`
  - `done_penalty=0` or at most a small tie-breaker such as `<=10`
  - compare reward-only score against reward-minus-small-done-penalty
  - keep `model_last` and `model_best_train` as mandatory fallbacks
- If G4B still fails to select a good checkpoint while late checkpoints are
  strong, use an external fixed-step selector wrapper instead of relying on
  short in-training eval-select.
- Include a final-eval-select wiring check. G4 configured `final_eval=True`,
  but the consistency trainer did not emit a 450k final eval-select row because
  `_run_final_eval_select` was not called after the training loop.

## Post-Result Claude Opus Review

```text
verdict=TUNE
unsupported_claims=none
main_issue=score = avg_reward - done_penalty * avg_done_rate made done_rate dominate checkpoint ranking
next_experiment=G4B with eval_select.num_steps=512 and done_penalty=0 or <=10
fallback=post-training external eval512 selector wrapper
```

Opus checked the internal selector scores:

| Agent steps | Formula check | Interpretation |
| ---: | --- | --- |
| 100k | `4.588011 - 2000 * 0.002930 = -1.271` | matches |
| 200k | `4.957870 - 2000 * 0.002686 = -0.413` | matches |
| 300k | `4.943703 - 2000 * 0.000977 = 2.991` | matches |
| 400k | `4.821063 - 2000 * 0.000244 = 4.333` | selected despite lower raw reward |

Review conclusion:

- The hook and aliasing behavior are correct.
- The selector chose the worst external checkpoint because `done_penalty=2000`
  overwhelmed raw reward differences.
- Architecture changes cannot fix a selector that discards the best available
  checkpoint.
- Do not overclaim G4, because the run itself remained weaker than G3A.

## Subagent Audit Correction

The subagent audit agreed with the G4 interpretation but narrowed one claim:

```text
periodic eval-select and aliasing worked; final eval-select was configured but
not executed for ConsistencyLatentStudent
```

Action taken after G4:

- `ConsistencyLatentStudent.train()` now calls `_run_final_eval_select(...)`
  after the training loop and republishes `model_best_deploy`.
- This is a no-op unless `train.ppo.eval_select.enabled=True` and
  `train.ppo.eval_select.final_eval=True`.
- G4B should verify that `eval_select/train_eval_history.tsv` contains a
  `FINAL/student` row at the end of training.

Smoke verification:

```text
run_tag=g4b_finaleval_wiring_smoke_20260602_022823
train seed=43
student_max_agent_steps=144
eval_select.num_steps=8
eval_select.done_penalty=0.0
```

Result:

- The smoke completed successfully.
- `eval_select/train_eval_history.tsv` contains periodic `EVAL/student` rows
  and a final `FINAL/student` row at agent step `144`.
- This smoke is only a wiring check; it is not an algorithm-quality result.

Code-review cross-check:

```text
reviewer=Claude Opus
verdict=PASS
notes=final eval-select placement mirrors PPO teacher; no-op semantics and
checkpoint aliasing are correct; only minor dead local assignment is harmless
```

## Pre-Result Claude Opus Review

```text
verdict=PASS
interpretation=selector failure is strongly justified
g4_priority=run eval-select before LR/EMA/NFE/residual
blocking_risks=none
notes=interval 100k is aggressive but acceptable for a probe; final external eval remains the gate
```

Review guidance:

- Eval-select must work across the full training horizon, because the G2
  failure may be a later-window selector problem.
- If `model_best_eval` reaches `>=5.1` on eval512, promote to three-train-seed
  formal with eval-select enabled.
- If `model_best_eval` is weak but `model_last` is strong, tune selector
  resolution, for example `num_steps=512` or `interval_agent_steps=50000`.
- If both are weak, extend the probe horizon before changing architecture.
