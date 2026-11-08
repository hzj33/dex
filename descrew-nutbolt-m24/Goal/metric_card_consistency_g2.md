# Metric Card G2: Standalone Consistency Validation

Date: 2026-06-01

Repo and branch:

```text
/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro-public
branch: diffusion
```

## Scope

This card starts the next step after G1 downgraded PAdapt-base residual
consistency to an ablation.

Canonical task:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
```

Canonical PPO teacher:

```text
sim2real/codrive/best_reward_4159.37.pth
```

This round validates standalone `ConsistencyLatentStudent` under the CoDrive
4159 teacher before any new residual-over-consistency design.

## Hypothesis

The existing best diffusion-family evidence is standalone consistency:

```text
Consistency latent 1h eval256 reward=5.155176
Flow matching 1h eval256 reward=4.827848
PAdapt fixed-step reference=about 4.72
```

Before optimizing a more complex residual design, the project needs a current,
local, artifact-indexed consistency reproduction path on the public `diffusion`
branch.

## Artifact Audit

Current public `sim2real/codrive/` artifacts:

```text
best_reward_4159.37.pth
model_best_codrive.ckpt
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml
dotpg_bc5/  # frozen baseline/reference only
```

The task/train config hashes match the live Hydra copies:

```text
task sha256=019876eaf3de4c8bbc2180886fd90ec406b753a8cdbfce72a198c31b790c6e62
train sha256=07f2903dd9aa511c44e358757a322cf361323b2876a5c2e15f507e627ce2a627
teacher sha256=2c3ff7411c7da40401866569116d98de73feded6f10f5431beac64f13ecd18ec
padapt sha256=c0f7a93f47dd2dd7204b798902f2897a2b4d0ce0cc5574c47fb62964d2556d74
```

Missing artifact:

```text
No standalone consistency CoDrive checkpoint is currently present in sim2real/codrive/.
```

Therefore G2 must generate or locate a consistency checkpoint before formal
multi-seed claims.

Subagent read-only audit:

```text
agent=Boyle
result=confirmed no local standalone consistency output/deploy artifact
teacher_restore=valid for training from scratch
teacher_restore_for_test=invalid because consistency_model is untrained/missing
dotpg_risk=only import/reference risk; train.algo=ConsistencyLatentStudent does not execute DOTPG
cloud_risk=old cloud scripts default ROOT may point at non-public worktree
```

## Implementation Boundary

Allowed changes:

- add CoDrive-specific consistency launcher scripts;
- add local GPU guard, manifest, and eval summary collection;
- add bounded `student_max_agent_steps` and exact-step progress logging to
  `ConsistencyLatentStudent`;
- use `torch.load(..., map_location=self.device)` in consistency restore paths.

Forbidden changes:

- no changes to task reward, horizon, reset, or eval seeds;
- no changes to the CoDrive 4159 teacher checkpoint;
- no changes to private repo;
- no changes to `dexscrew/dotpg/`, `docs/dotpg_*`, or `sim2real/codrive/dotpg_bc5/`.

## Metrics

Primary:

```text
fixed-step eval reward
```

Default G2 ladder:

| Stage | Protocol | Pass condition |
| --- | --- | --- |
| Static | compile and shell syntax | no syntax errors |
| Smoke | tiny local run, 4 envs, short fixed eval | load, train, restore, eval summary emitted |
| Scout | seed 42, 200k agent steps, eval256 | beat PAdapt reference and approach known consistency screen |
| Candidate | train seed 42, eval seeds 42/43/44, eval256 or eval512 | stable above Flow/PAdapt |
| Formal | train seeds 42/43/44, eval2048 | only after scout/candidate pass |

Secondary:

- latent MSE/L1 to teacher latent;
- action MSE to teacher action;
- done_rate;
- checkpoint selector behavior;
- NFE and local runtime;
- local GPU utilization and VRAM pressure.

## Local Commands

Static:

```text
bash -n scripts/dexh13_lightbulb_student_consistency_codrive.sh scripts/run_consistency_g2_validate_codrive.sh scripts/eval_consistency_codrive_checkpoint.sh scripts/run_consistency_g2_formal_codrive.sh
PYTHONDONTWRITEBYTECODE=1 python -m py_compile dexscrew/algo/ppo/consistency_latent_student.py
git diff --check -- dexscrew/algo/ppo/consistency_latent_student.py scripts/dexh13_lightbulb_student_consistency_codrive.sh scripts/run_consistency_g2_validate_codrive.sh scripts/eval_consistency_codrive_checkpoint.sh scripts/run_consistency_g2_formal_codrive.sh Goal/metric_card_consistency_g2.md
```

Smoke:

```text
GPU=0 SEED=42 \
TRAIN_WINDOW_SEC=600 STUDENT_MAX_AGENT_STEPS=384 \
NUM_ENVS=4 MINIBATCH=48 \
EVAL_STEPS=16 EVAL_NUM_ENVS=4 EVAL_SEEDS="42" \
bash scripts/run_consistency_g2_validate_codrive.sh
```

Scout:

```text
GPU=0 SEED=42 \
TRAIN_WINDOW_SEC=1800 STUDENT_MAX_AGENT_STEPS=200000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=256 EVAL_NUM_ENVS=16 EVAL_SEEDS="42" \
bash scripts/run_consistency_g2_validate_codrive.sh
```

Formal:

```text
GPU=0 \
BATCH_TAG=g2_consistency_formal_$(date +%Y%m%d_%H%M%S) \
TRAIN_SEEDS="42 43 44" \
EVAL_SEEDS="42 43 44" \
TRAIN_WINDOW_SEC=4200 STUDENT_MAX_AGENT_STEPS=900000 \
NUM_ENVS=16 MINIBATCH=192 \
EVAL_STEPS=2048 EVAL_NUM_ENVS=16 EVAL_TIMEOUT_SEC=2400 \
bash scripts/run_consistency_g2_formal_codrive.sh
```

## Decision Rules

Promote only if:

- scout fixed-step reward is above the PAdapt reference of about `4.72`;
- ideally it is near the known consistency screen `5.155176`;
- done_rate remains low;
- restore/eval emits the expected summary metrics;
- no protocol, teacher, task, or seed changes are involved.

Reject or tune if:

- scout is below PAdapt/Flow;
- checkpoint selector again disagrees strongly with fixed-step eval;
- action/latent diagnostics worsen without reward gain.

Current status:

```text
static=pass
smoke=pass
scout=pass
candidate_eval=pass
formal=tune
```

Claude Opus pre-smoke cross-review:

```text
verdict=TUNE
blocking_items=none
notes=workflow and CoDrive 4159 boundary are valid; map_location fix is appropriate
checks_before_smoke=confirm Hydra accepts ++train.ppo.student_max_agent_steps and keep smoke as code/config validation only
```

Local check:

```text
Hydra uses ++ overrides for new ppo keys, so no config-schema edit is required.
ConsistencyLatentStudent batch_size is num_actors; smoke MINIBATCH is non-critical for rollout shape.
```

## Smoke Result

Run:

```text
run_tag=g2_consistency_smoke_s42_20260601_195336
teacher_ckpt=sim2real/codrive/best_reward_4159.37.pth
NUM_ENVS=4
MINIBATCH=48
STUDENT_MAX_AGENT_STEPS=384
TRAIN_WINDOW_SEC=600
EVAL_STEPS=16
EVAL_SEEDS=42
```

Artifacts:

```text
probe_dir=outputs/local_probe_consistency_g2/g2_consistency_smoke_s42_20260601_195336
checkpoint=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_smoke_s42_20260601_195336/stage2_consistency_nn/model_best.ckpt
eval_summary=outputs/local_probe_consistency_g2/g2_consistency_smoke_s42_20260601_195336/eval_summary.tsv
```

Result:

| Metric | Value |
| --- | ---: |
| train_status | 0 |
| eval_status | 0 |
| eval_steps | 16 |
| avg_reward | 3.992155 |
| avg_done_rate | 0.000000 |
| latent_mse | 0.169495 |
| latent_l1 | 0.271324 |
| action_mse_to_teacher | 0.170243 |

Interpretation:

- Smoke passed: train, bounded stop, checkpoint save, restore, and fixed-step
  eval summary all work.
- Smoke reward is not algorithm evidence because the training budget is only
  `384` agent steps.
- Exact agent-step logging works for sub-1M local probes.

Next:

```text
Run G2 scout: seed 42, 200k agent steps, eval256.
```

## Scout Result

Run:

```text
run_tag=g2_consistency_scout_s42_20260601_195438
teacher_ckpt=sim2real/codrive/best_reward_4159.37.pth
NUM_ENVS=16
MINIBATCH=192
STUDENT_MAX_AGENT_STEPS=200000
TRAIN_WINDOW_SEC=1800
CONSISTENCY_INFER_STEPS=1
EVAL_STEPS=256
EVAL_SEEDS=42
```

Artifacts:

```text
probe_dir=outputs/local_probe_consistency_g2/g2_consistency_scout_s42_20260601_195438
checkpoint=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_scout_s42_20260601_195438/stage2_consistency_nn/model_best.ckpt
eval_summary=outputs/local_probe_consistency_g2/g2_consistency_scout_s42_20260601_195438/eval_summary.tsv
```

Train:

```text
exit_status=0
student_max_agent_steps=200000 reached
training_current_best=3609.46
fps_steady_state=about 252
```

Eval256 seed 42:

| Metric | Value |
| --- | ---: |
| avg_reward | 4.499055 |
| avg_done_rate | 0.000000 |
| latent_mse | 0.059613 |
| latent_l1 | 0.146053 |
| action_mse_to_teacher | 0.055103 |

Interpretation:

- The 200k scout did not pass the PAdapt fixed-step reference of about `4.72`.
- It is also below Flow `4.827848` and the historical Consistency 1h screen
  `5.155176`.
- This does not reject standalone consistency, because the historical best
  evidence used a longer 1h training window and higher train-best reward.
- Exact agent-step progress logging worked and should remain in place for longer
  local scouts.

Claude Opus result cross-review:

```text
verdict=TUNE
do_not_enter_formal=true
do_not_reject_consistency=true
recommended_next=1h scout before NFE=2 or multiseed formal
```

Decision update:

- Do not promote this 200k checkpoint to candidate or formal validation.
- Run a 1h local scout to match the historical consistency evidence scale before
  changing NFE or designing a consistency-base residual.
- Only after a useful 1h scout should G2 move to eval seeds `42,43,44` or
  formal train seeds.

## 1h Local Scout Result

Run:

```text
run_tag=g2_consistency_1h_s42_20260601_201217
teacher_ckpt=sim2real/codrive/best_reward_4159.37.pth
NUM_ENVS=16
MINIBATCH=192
STUDENT_MAX_AGENT_STEPS=900000
TRAIN_WINDOW_SEC=4200
CONSISTENCY_INFER_STEPS=1
EVAL_STEPS=256
EVAL_SEEDS=42
```

Artifacts:

```text
probe_dir=outputs/local_probe_consistency_g2/g2_consistency_1h_s42_20260601_201217
checkpoint=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_1h_s42_20260601_201217/stage2_consistency_nn/model_best.ckpt
eval_summary=outputs/local_probe_consistency_g2/g2_consistency_1h_s42_20260601_201217/eval_summary.tsv
```

Train:

```text
exit_status=0
student_max_agent_steps=900000 reached
training_current_best=3905.29
fps_steady_state=about 255
gpu_budget=within local 80 percent target; no >85 percent sustained overload observed
```

Eval256 seed 42:

| Metric | Value |
| --- | ---: |
| avg_reward | 4.972154 |
| avg_done_rate | 0.000000 |
| latent_mse | 0.041433 |
| latent_l1 | 0.111971 |
| action_mse_to_teacher | 0.043485 |

Comparison:

| Method / run | Eval256 reward | Note |
| --- | ---: | --- |
| PAdapt reference | ~4.72 | handoff anchor |
| Flow 1h | 4.827848 | historical screen |
| G2 consistency 200k | 4.499055 | local short scout |
| G2 consistency 1h | 4.972154 | local 16-env scout |
| Historical consistency 1h | 5.155176 | cloud/handoff screen |

Interpretation:

- The 1h local scout passes the PAdapt and Flow gates.
- It does not fully reproduce historical consistency `5.155176`, but protocol
  audit found important scale differences: historical cloud used `48` envs and
  `576` minibatch, while local scout used `16` envs and `192` minibatch.
- The result confirms that the 200k scout was too short to judge standalone
  consistency.
- This is a candidate-level result, not yet formal evidence, because it is still
  one train seed and one eval seed.

Claude Opus result cross-review:

```text
verdict=PASS
formal_now=false
recommended_next=multi-seed eval 42/43/44 on this checkpoint
priority=multi-seed eval > NFE=2 > selector instrumentation > consistency-base residual
```

Decision update:

- Promote standalone consistency from scout to candidate-eval stage.
- Evaluate the same checkpoint on seeds `42,43,44` with eval256 before changing
  NFE or architecture.
- Do not claim exact reproduction of historical `5.155176` until eval protocol
  details and scale differences are resolved.

## Candidate Eval Result

Run:

```text
run_tag=g2_consistency_1h_eval3_20260601_211543
checkpoint=outputs/Dexh13HoraLightbulb_student_consistency_codrive/g2_consistency_1h_s42_20260601_201217/stage2_consistency_nn/model_best.ckpt
EVAL_STEPS=256
EVAL_NUM_ENVS=16
EVAL_SEEDS="42 43 44"
```

Artifacts:

```text
eval_dir=outputs/local_eval_consistency_g2/g2_consistency_1h_eval3_20260601_211543
eval_summary=outputs/local_eval_consistency_g2/g2_consistency_1h_eval3_20260601_211543/eval_summary.tsv
```

Eval256:

| Eval seed | avg_reward | avg_done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 4.972154 | 0.000000 | 0.041433 | 0.111971 | 0.043485 |
| 43 | 5.328857 | 0.000000 | 0.030680 | 0.101958 | 0.029445 |
| 44 | 5.293205 | 0.000000 | 0.036612 | 0.108730 | 0.032891 |

Aggregate:

| Metric | Value |
| --- | ---: |
| reward_mean | 5.198072 |
| reward_std | 0.160410 |
| done_rate_mean | 0.000000 |
| latent_mse_mean | 0.036242 |
| latent_l1_mean | 0.107553 |
| action_mse_to_teacher_mean | 0.035274 |

Interpretation:

- Candidate eval passes the PAdapt fixed-step reference and Flow historical
  screen.
- The aggregate is slightly above the historical consistency eval256 screen
  `5.155176`, but this is not a strict reproduction because historical eval
  provenance and train scale differ.
- The large gap between the 200k scout and the 1h scout confirms that tiny
  training budgets are smoke/config checks, not reliable algorithm comparisons.
- Universal `done_rate=0` should be compared against teacher and PAdapt eval
  before treating it as either normal or suspicious.

Claude Opus result cross-review:

```text
verdict=PASS
enter_formal_train_seed_validation=true
notes=multi-seed eval is consistently above PAdapt/Flow; avoid strict historical reproduction claims
next_steps=formal train seeds 42/43/44; compare done_rate to teacher; optionally reproduce 48-env scale after formal pass
```

Decision update:

- Promote standalone consistency to formal train-seed validation.
- Next formal run should train seeds `42,43,44` under the same public CoDrive
  4159 teacher and evaluate each checkpoint with fixed-step eval seeds
  `42,43,44`.
- Keep `NFE=1` for the formal consistency run. Test `NFE=2` only after formal
  training-seed stability is known.

## Formal Train-Seed Validation Result

Run:

```text
batch_tag=g2_consistency_formal_20260601_212702
teacher_ckpt=sim2real/codrive/best_reward_4159.37.pth
TRAIN_SEEDS="42 43 44"
EVAL_SEEDS="42 43 44"
NUM_ENVS=16
MINIBATCH=192
STUDENT_MAX_AGENT_STEPS=900000
TRAIN_WINDOW_SEC=4200
CONSISTENCY_INFER_STEPS=1
EVAL_STEPS=2048
EVAL_NUM_ENVS=16
EVAL_TIMEOUT_SEC=2400
```

Artifacts:

```text
formal_dir=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702
summary=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702/formal_eval_summary.tsv
aggregate=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702/aggregate_stats.txt
train_runs=outputs/local_formal_consistency_g2/g2_consistency_formal_20260601_212702/train_runs.tsv
```

Train:

```text
all_train_status=0
all_eval_status=0
train_seed_42_current_best=3905.29
train_seed_43_current_best=3927.56
train_seed_44_current_best=3912.18
gpu_budget=within local target; no sustained >85 percent overload observed
```

Per-train-seed eval2048 aggregate:

| Train seed | Eval seeds | reward_mean | reward_std | done_mean | latent_mse | action_mse_to_teacher |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 42/43/44 | 5.123429 | 0.084707 | 0.001017 | 0.032629 | 0.032020 |
| 43 | 42/43/44 | 4.730098 | 0.056040 | 0.001109 | 0.037348 | 0.038814 |
| 44 | 42/43/44 | 4.904273 | 0.088405 | 0.001078 | 0.040359 | 0.043973 |

All formal eval2048 rows:

| Train seed | Eval seed | avg_reward | avg_done_rate | latent_mse | latent_l1 | action_mse_to_teacher |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 42 | 5.008217 | 0.001007 | 0.035323 | 0.105393 | 0.036004 |
| 42 | 43 | 5.209451 | 0.001038 | 0.031038 | 0.102292 | 0.029617 |
| 42 | 44 | 5.152619 | 0.001007 | 0.031526 | 0.102432 | 0.030440 |
| 43 | 42 | 4.677208 | 0.001068 | 0.038691 | 0.111764 | 0.040815 |
| 43 | 43 | 4.705429 | 0.001190 | 0.037290 | 0.109245 | 0.038956 |
| 43 | 44 | 4.807657 | 0.001068 | 0.036064 | 0.108511 | 0.036670 |
| 44 | 42 | 4.782201 | 0.001129 | 0.041527 | 0.112101 | 0.046587 |
| 44 | 43 | 4.941924 | 0.001038 | 0.039595 | 0.109109 | 0.041013 |
| 44 | 44 | 4.988695 | 0.001068 | 0.039954 | 0.108601 | 0.044318 |

Overall aggregate:

| Metric | Value |
| --- | ---: |
| valid_eval_rows | 9 |
| reward_mean | 4.919267 |
| reward_std | 0.178720 |
| done_rate_mean | 0.001068 |
| latent_mse_mean | 0.036779 |
| latent_l1_mean | 0.107716 |
| action_mse_to_teacher_mean | 0.038269 |

Interpretation:

- Formal validation shows real standalone-consistency value, but not stable
  formal acceptance.
- The overall mean beats PAdapt and Flow anchors, and train seed `42` matches
  the stronger historical consistency range.
- Train seed `43` is only barely above the PAdapt reference and below Flow,
  while train seed `44` is intermediate.
- The `seed42 -> seed43` reward gap is much larger than within-checkpoint eval
  seed noise, so this is a train-seed stability or selector problem rather than
  ordinary eval noise.
- Training `Current Best` is not a reliable selector here: seed `43` had the
  highest scalar training best but the weakest fixed-step eval.

Claude Opus formal result cross-review:

```text
verdict=TUNE
reason=aggregate beats PAdapt/Flow, but train-seed stability fails
do_not_jump_to_nfe2=true
do_not_jump_to_residual_over_consistency=true
recommended_next=diagnose seed 43 stability and selector behavior first
minimal_next=seed43-only LR/EMA or selector probe, about 450k steps per setting
```

Decision update:

- Do not claim a stable formal PASS for standalone consistency yet.
- Keep standalone consistency as the active best diffusion-family candidate.
- Start G3 as a train-seed stability and checkpoint-selector diagnosis before
  NFE, robustness, or new residual-over-consistency work.
