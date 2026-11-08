# Goal Mode Diffusion Optimization Plan

> Current Goal Mode entry for the public `diffusion` branch.
> `Goal/goal_overview.md` is retained as raw long-form background; this file is the execution-facing plan.

## 1. Current Branch And Canonical Scope

This plan applies to:

```text
repo: /home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro-public
branch: diffusion
remote base: origin/diffusion
```

The current `main` branch is an old baseline branch and is not the basis for this diffusion optimization round.

Canonical task:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
```

Canonical PPO teacher:

```text
sim2real/codrive/best_reward_4159.37.pth
```

Canonical frozen config copies:

```text
sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml
sim2real/codrive/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml
```

These frozen config copies have been verified in the diffusion branch history as byte-identical to the live Hydra copies:

```text
configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml
configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml
```

This CoDrive PPO teacher is the user-selected roughly 4000-reward baseline. It replaces the older thesis-formal teacher as the canonical teacher for this local diffusion optimization workflow.

## 2. Boundaries

Hard boundaries for this phase:

- Work in the public `diffusion` branch.
- Do not modify the private repo or private DOTPG workflow.
- Do not modify `dexscrew/dotpg/`, `docs/dotpg_*`, or `sim2real/codrive/dotpg_bc5/`.
- DOTPG is a frozen baseline/reference only, not the diffusion main line.
- CoDriveThesis is a separate formal/thesis reference and is not used as the teacher for this round.
- Old XHand Plan v2 results are historical evidence only and must not be mixed into current CoDrive diffusion claims.
- Do not change task reward, horizon, eval seeds, reset logic, or teacher checkpoint to manufacture gains.
- Student inference must not use privileged information.

## 3. Current Project Facts

The `origin/diffusion` branch contains the current DexH13 CoDrive diffusion stack:

- DexH13 CoDrive task/train configs and lightbulb assets.
- PPO teacher and PAdapt student pipeline.
- Diffusion-family student algorithms:
  - `DiffusionLatentStudent`
  - `ConsistencyLatentStudent`
  - `FlowMatchingLatentStudent`
  - `DiffusionActionChunkStudent`
- Classical student baselines:
  - PAdapt
  - PureBC
  - BC / LatentBC
  - DAgger
  - DOTPG frozen reference
- Deploy/eval packages under `sim2real/codrive/` and related deploy folders.

Known teacher facts from branch handoff:

- The CoDrive PPO run completed as a 2h training run.
- Best teacher reward: `4159.37`.
- The two-finger gate was stable.
- The teacher is usable as the fixed distillation teacher.
- Known limitations:
  - thumb tangent motion can still dominate index tangent motion;
  - the opposition-grip reward term was effectively dead in diagnostics;
  - visual checks remain important because scalar reward alone can miss co-drive behavior quality.

## 4. Existing CoDrive Diffusion Evidence

### 4.1 One-Hour Four-Way Diffusion Sweep

Teacher and task:

```text
teacher: sim2real/codrive/best_reward_4159.37.pth
task: Dexh13HoraLightbulbSim2RealTwoFingerCoDrive
seed: 42
```

| Method | Training-window best | Eval256 reward | Eval256 done_rate | Status |
| --- | ---: | ---: | ---: | --- |
| Diffusion latent | 4145.42 | 2.855633 | 0.000895 | useful but selector mismatch |
| Consistency latent | 3837.96 | 5.155176 | 0.000000 | best fixed-step screen |
| Flow matching latent | 3975.03 | 4.827848 | 0.000081 | strong fixed-step screen |
| Action chunk | 3623.38 | poor / noncompetitive | high | not main line |

### 4.2 Continue-2h From 1h

| Method | Continue-2h training best | Eval256 reward | Eval256 done_rate | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Diffusion latent | 4246.32 | 2.942062 | 0.001465 | slight eval improvement, still behind |
| Consistency latent | 4055.97 | 4.979207 | 0.000000 | still strong, but below 1h screen |
| Flow matching latent | 4084.44 | 4.364139 | 0.000570 | below 1h screen |
| Action chunk | 3620.43 | weak deploy-style result | high | exploratory only |

Working conclusion:

- Training reward is not a reliable checkpoint selector for diffusion-family students.
- Fixed-step eval currently supports the ranking:

```text
consistency latent > flow matching latent > ordinary diffusion latent >> action chunk
```

- Future optimization must select checkpoints by unified fixed-step eval and behavior diagnostics, not by training `Current Best` alone.

## 5. Baseline Framing

Required comparison set for future diffusion rounds:

- PPO teacher: `sim2real/codrive/best_reward_4159.37.pth`
- PAdapt: strong deterministic student baseline
- Consistency latent: current strongest diffusion-family fixed-step screen
- Flow matching latent: strong few-step baseline
- Diffusion latent: ordinary latent diffusion baseline
- Diffusion action chunk: action-space boundary / negative-control branch
- BC / DAgger / PureBC: classical imitation baselines
- DOTPG: frozen baseline/reference only

Known baseline anchors:

- PAdapt CoDrive 30m scalar reward: `3448.38`.
- PAdapt fixed-step reference is about `4.72` in handoff notes.
- DAgger pure-replay fixed-step candidate reached `4.528048`.
- DOTPG `dual_bc5` fixed-step deploy eval:
  - clean `4.496476`
  - light `3.824169`
  - hard `2.960864`

Baseline rule:

```text
Optimize only diffusion-family code and configs.
Compare against DOTPG and classical baselines.
Do not continue DOTPG optimization in this workflow.
```

## 6. First Optimization Mainline: Residual Latent

First priority:

```text
Residual latent over PAdapt / consistency
```

Goal:

Improve fixed-step eval, robustness, and latency for diffusion-family students under the CoDrive 4159 teacher, without retraining the teacher or changing the task definition.

Default hypothesis:

PAdapt and consistency already provide stable latent/action behavior. A generative head should learn a bounded correction instead of generating the entire latent from scratch.

Default structure:

```text
z_base = PAdapt(proprio_hist) or current best consistency latent
delta_z = GenerativeHead(z_base, proprio_hist, obs, t)
z_student = z_base + gate * delta_z
a_student = frozen_actor(obs, z_student)
```

Implementation defaults for the first probe:

- Use PAdapt latent as the first base unless an audit shows consistency latent is easier to reuse safely.
- Keep actor decode frozen.
- Start with a small gate, for example `0.1` or a learnable sigmoid initialized near zero.
- Support NFE `1,2,4,8,10`, but deployment target is `NFE <= 2`.
- Add explicit diagnostics:
  - `base_latent_norm`
  - `delta_latent_norm`
  - `gate_mean`
  - latent MSE/L1
  - teacher action MSE
  - base action anchor loss
  - action delta / jerk
  - saturation ratio

Default loss family:

```text
flow / consistency / diffusion residual loss
+ residual latent reconstruction loss
+ action BC loss
+ base action anchor loss
+ action smoothness or delta penalty
+ optional tail-action error penalty
```

Reject the direction if the first low-cost probe shows:

- residual norm dominates base latent norm;
- action quality becomes visibly jittery;
- eval reward improves only because of changed protocol;
- NFE `1` or `2` is unusable;
- fixed-step reward falls below PAdapt reference.

## 7. Metrics And Gates

Primary metric:

```text
fixed-step eval reward
```

Selector rule:

```text
Do not select by training Current Best alone.
Use fixed-step eval and behavior diagnostics for candidate ranking.
```

Default protocol ladder:

| Stage | Protocol | Purpose |
| --- | --- | --- |
| Smoke | tiny train/eval, usually one seed | check load, config wiring, TensorBoard/log fields, shape, NFE, no NaN/Inf; no algorithm claim |
| Scout | short but nontrivial train budget, one or a few train seeds, fixed eval256/eval512 | screen whether a direction is worth continuing; do not promote as a final result |
| Formal | 30m or larger train budget, train seeds 42/43/44, fixed eval2048 when justified | algorithm comparison only after scout evidence is useful |

Local training allocation uses three layers:

- Smoke verifies code and config only. It is intentionally too short to prove
  whether an algorithm is better.
- Scout is the first meaningful optimization screen. It should be long enough to
  expose large configuration differences, but cheap enough to tune or reject.
  A scout may justify more training, but it does not close an algorithm claim.
- Formal validation starts only after scout passes. It uses multi-seed training,
  unified eval, and robustness or latency checks as needed. Formal results must
  be compared against PAdapt, consistency, flow, and the relevant fixed
  reference baselines under the same protocol.

Current practical allocation:

- Smoke can be as small as tens to hundreds of agent steps plus tiny eval. It
  only checks loading, shapes, checkpoint restore, TensorBoard diagnostics, and
  summary emission. It never supports an algorithm-quality claim.
- Short scout can reject broken settings or obvious regressions. Runs such as
  `32k-200k` agent steps are useful for sign checks, but should not be used alone
  to compare close diffusion configurations.
- Directional scout is the first meaningful local optimization screen. The
  current default for consistency-family models is about `450k` agent steps at
  `16` envs, followed by fixed eval512 seeds `42,43,44` and post-training
  external selection over `model_best`, `model_best_train`, and `model_last`.
- Medium scout is used when the directional scout is promising but noisy, or
  when `model_last` suggests the run has not plateaued. The current target is
  `800k-1M` agent steps for the same train seed before multi-seed expansion.
- Formal validation should test train seeds `42,43,44` only after the scout tier
  gives a clear enough signal. Formal claims require fixed eval2048 seeds
  `42,43,44`; robustness, latency, and visual checks are added when deployment
  relevance matters.

Hard gates:

- Student inference uses no privileged information.
- No task reward / horizon / seed / reset manipulation.
- Main-line candidate must not fall below PAdapt fixed-step reference.
- `NFE <= 2` must be usable.
- Reward-only gains with worse action jitter or obvious visual co-drive regression are not accepted.

Recommended acceptance targets for the first residual-latent round:

- Minimum: beat PAdapt fixed-step reference and match or beat Flow eval screen.
- Strong: match or beat the best existing Consistency eval screen.
- Breakthrough: improve reward while keeping `NFE <= 2`, no action jitter regression, and better robustness than consistency/flow.

## 8. Local Workflow

Default workflow for every optimization round:

1. Write a metric card with one main hypothesis.
2. Audit current artifacts and baseline paths before code changes.
3. Check local GPU budget.
4. Use subagents for bounded support:
   - task/protocol and privileged-input audit
   - algorithm/loss implementation audit
   - artifact/checkpoint/eval audit
   - result interpretation audit
5. Use Claude Opus cross-review for important direction, code, and result decisions.
6. Make one small, reversible code or config change.
7. Run local smoke.
8. Run one or two low-cost probes.
9. Compare against the required baseline set.
10. Update handoff / artifact index / metric notes before the next round.

Local resource rule:

- Use local GPU by default.
- Target GPU utilization / VRAM pressure around `80%`.
- If sustained usage is above about `85%`, reduce `numEnvs`, batch size, parallelism, or defer training.
- Do not default to cloud training.
- Cloud expansion requires explicit user approval and a useful local probe.

Default runtime budget:

- Smoke: under 15 minutes.
- Low-cost probe: under 2 hours.
- Formal local expansion: only after candidate evidence justifies it.

Claude review mechanism:

```text
claude -p --model opus "<review prompt>"
```

The review should be non-mutating and should check:

- whether the hypothesis matches the evidence;
- whether the code change affects only diffusion-family logic;
- whether DOTPG/private flows remain untouched;
- whether the result interpretation overclaims.

## 9. Current Milestone Status

G1 PAdapt-base residual update:

- PAdapt-base residual consistency passed smoke and two scout runs, but did not
  beat the PAdapt fixed-step reference.
- Scout A: gate `0.25`, anchor `0.05`, eval256 seed42 `4.706028`.
- Scout B: gate `0.60`, anchor `0.01`, eval256 seed42 `4.607157`.
- This PAdapt-base residual default is downgraded to ablation, not promoted to
  formal validation.
- It remains a fallback/diagnostic branch, not the current mainline.

G2 standalone consistency update:

- Standalone consistency CoDrive validation path now exists for the 4159 teacher.
- Static checks passed and smoke passed.
- 200k local scout produced eval256 seed42 `4.499055`, below the PAdapt reference
  and below historical consistency evidence.
- The 1h local scout produced eval256 seed42 `4.972154` and passed PAdapt/Flow
  gates, while remaining below the historical consistency screen `5.155176`.
- Candidate eval on the same 1h checkpoint produced eval256 seeds `42,43,44`
  with rewards `4.972154`, `5.328857`, and `5.293205`.
- Candidate aggregate is reward mean `5.198072`, reward std `0.160410`,
  done_rate mean `0.000000`, latent_mse mean `0.036242`, and
  action_mse_to_teacher mean `0.035274`.
- This passes PAdapt/Flow and slightly exceeds the historical consistency screen,
  but should not be described as a strict reproduction because train scale and
  historical eval provenance differ.
- Formal train-seed validation then trained seeds `42,43,44`, each at `900k`
  agent steps / `4200s`, and evaluated each checkpoint with eval2048 seeds
  `42,43,44`.
- Formal per-train-seed means were:
  - seed `42`: reward `5.123429`, done `0.001017`
  - seed `43`: reward `4.730098`, done `0.001109`
  - seed `44`: reward `4.904273`, done `0.001078`
- Formal overall aggregate was reward mean `4.919267`, reward std `0.178720`,
  done_rate mean `0.001068`, latent_mse mean `0.036779`, and
  action_mse_to_teacher mean `0.038269`.
- Opus formal cross-review verdict is `TUNE`: aggregate beats PAdapt/Flow, but
  train-seed stability fails because seed `43` is only marginally above PAdapt
  and below Flow.

Decision:

- Keep standalone consistency as the active best diffusion-family candidate.
- Do not claim stable formal PASS yet.
- Do not jump to NFE=2, robustness, or consistency-base residual until the
  train-seed stability and checkpoint-selector issue is diagnosed.

G3 selector-diagnosis update:

- A seed43 default rerun truncated at `450k` agent steps recovered strongly:
  `model_best` eval512 mean `5.176507`, `model_last` eval512 mean `5.118077`.
- A no-retrain selector probe on the weak G2 formal seed43 run showed that the
  original `model_best` eval512 mean was only `4.688641`, while `model_last`
  eval512 mean was `5.170769`.
- Under the same formal eval2048 protocol, G2 seed43 `model_last` reached
  reward mean `5.139005`, compared with the original G2 seed43 `model_best`
  mean `4.730098`.
- This reinterprets the G2 formal instability primarily as checkpoint-selector
  failure, not as standalone consistency model incapacity.
- Replacing only seed43 `model_best` rows with `model_last` rows would move the
  9-row formal reward mean from `4.919267` to about `5.055569`.

G4 eval-select update:

- The existing eval-select hook was enabled on train seed `43` with `450k`
  agent steps, interval `100k`, internal eval `256` steps, and done penalty
  `2000.0`.
- The hook itself worked: `model_best` was aliased to `model_best_eval`, and
  `model_best_train` was preserved separately.
- Scope note: this only covered periodic eval-select and aliasing. G4 also
  configured `final_eval=True`, but `ConsistencyLatentStudent` did not call the
  final eval-select hook after its training loop, so no 450k final selector row
  was emitted.
- Internal eval-select picked the 400k checkpoint because the heavy done
  penalty made its score highest, despite lower raw eval reward.
- External eval512 seeds `42,43,44` showed:
  - `model_best_eval` / `model_best`: reward mean `4.457390`
  - `model_last`: reward mean `4.933033`
  - `model_best_train`: reward mean `4.909166`
- Decision: `TUNE`.
- Interpretation: the active issue remains checkpoint selection, but the first
  eval-select settings are unsafe. The current `256`-step selector with
  `2000.0` done penalty selected a low external-reward checkpoint. This result
  argues for selector-metric tuning or an external fixed-step selector wrapper,
  not for immediate architecture changes.
- This G4 run was also weaker than the G3A recovered seed43 run, so close
  configuration comparisons still require the scout/formal ladder rather than
  one short run.
- After this audit, `ConsistencyLatentStudent.train()` was minimally patched to
  call the existing final eval-select hook after training. This remains a no-op
  unless eval-select and `final_eval` are enabled.
- A tiny wiring smoke,
  `g4b_finaleval_wiring_smoke_20260602_022823`, confirmed that the final
  `FINAL/student` row is now written. This smoke is a code-path check only, not
  evidence about algorithm quality.
- Claude Opus reviewed the patch with verdict `PASS`.

G4B selector-metric update:

- A selector-only scout used train seed `43`, `450k` agent steps,
  `eval_select.num_steps=512`, `eval_select.done_penalty=0.0`, and external
  eval512 seeds `42,43,44`.
- Internal eval-select picked the 400k checkpoint with internal avg reward
  `5.269734`; the final 450k row was also emitted and scored `4.959214`, so it
  correctly did not overwrite the selected checkpoint.
- External eval512 showed:
  - `model_best_eval` / `model_best`: reward mean `5.137090`
  - `model_last`: reward mean `5.106660`
  - `model_best_train`: reward mean `5.171160`
- Decision: `CONDITIONAL PASS` for selector wiring under real training load, not
  a formal pass.
- Interpretation: the reward-first `512`-step selector fixed the obvious G4
  failure and recovered a `5.1+` checkpoint above `model_last`, but scalar
  training-best was still higher on all three eval seeds. This validates that
  eval-select is no longer actively harmful in this seed; it does not yet prove
  eval-select adds value over train-best.
- Next action is selector validation across train seeds `42,43,44`, with
  `model_best_eval`, `model_best_train`, and `model_last` all externally
  evaluated before choosing the deploy alias policy.

## 10. Immediate Next Metric Card

Title:

```text
G4C consistency selector formal validation
```

Hypothesis:

```text
Standalone consistency is strong enough at NFE=1, but scalar online training
reward, short eval-select, and last checkpoint can disagree. G4B shows that a
reward-first 512-step selector can recover a strong checkpoint on one train seed,
but the deploy alias policy must be validated across train seeds before further
algorithm changes.
```

Primary questions:

- Does the G4B selector setting hold across train seeds `42,43,44`?
- Should deploy aliases point to `model_best_eval`, `model_best_train`,
  `model_last`, or a post-training external eval winner?
- Is in-training eval-select sufficient, or should the workflow always run a
  post-training external eval512 selector before deploy/export?
- After selector validation, is there still enough unexplained gap to justify
  LR/EMA/NFE/residual probes?

Minimal next experiment:

- Keep architecture unchanged.
- Train seeds `42,43,44` with G4B selector settings:
  - `eval_select.num_steps=512`
  - `eval_select.done_penalty=0.0`
  - `eval_select.final_eval=True`
  - `eval_select.save_deploy_best=True`
- Keep the CoDrive 4159 teacher, task, eval seeds, NFE, and no-privileged-input
  boundary unchanged.
- Evaluate `model_best_eval`, `model_best_train`, and `model_last` for each
  train seed with fixed eval seeds `42,43,44`.
- Promote the selector only if selected checkpoints stay near the recovered
  `5.1+` range and are not consistently worse than the train-best or last
  checkpoints.
- Otherwise keep the best external-eval checkpoint as the deploy alias and treat
  in-training eval-select as a candidate generator rather than the final
  selector.

Deliverable:

- A new metric card with exact configs, GPU budget, checkpoint hashes, per-seed
  eval summaries, and Opus result review.
- A decision on deploy alias policy before any architecture or loss changes.

Current G4C status:

- `TUNE / external deploy selector`.
- Train-only/internal selector validation finished for train seeds `42,43,44`;
  internal selected rewards were `5.144608`, `5.269734`, and `5.250534`.
- The external checkpoint comparison also finished: 27/27 eval512 rows,
  all `status=0`.
- `model_best`, `model_best_eval`, and `model_best_deploy` are hash-identical
  for all three train seeds.
- Fixed-kind external eval512 aggregates:
  - `model_best`: mean reward `5.120598`
  - `model_best_train`: mean reward `5.037163`
  - `model_last`: mean reward `5.132772`
- Per-train-seed external winners:
  - seed `42`: `model_last`, reward `5.150810`
  - seed `43`: `model_best_train`, reward `5.171160`
  - seed `44`: `model_best` / `model_last` eval-equivalent, reward `5.140846`
- Current policy: in-training eval-select remains a candidate generator and
  aliasing mechanism, but deploy candidates should be chosen by post-training
  external eval512 over `model_best`, `model_best_train`, and `model_last`.
- G4D then evaluated the external-selected candidates under formal eval2048:
  - seed `42` `model_last`: reward mean `5.063657`, done `0.001068`
  - seed `43` `model_best_train`: reward mean `5.124141`, done `0.001079`
  - seed `44` `model_best`: reward mean `5.078385`, done `0.001089`
- G4D overall aggregate is reward mean `5.088728`, reward std `0.086362`,
  done mean `0.001078`, latent MSE `0.041225`, and action MSE to teacher
  `0.042266`.
- G4D is above the G2 formal consistency aggregate `4.919267` and above the
  PAdapt fixed-step reference for every train seed.
- G4D also shows a real but modest horizon gap from G4C eval512 selected
  aggregate `5.154272` to eval2048 aggregate `5.088728`.
- Decision: `PASS selector formal validation / TUNE algorithm upper bound`.
  The selector problem can be treated as controlled enough to move on, but the
  algorithm itself is not finished.

G5 train-aligned action BC update:

- A minimal `ConsistencyLatentStudent` training-path patch added a
  gradient-enabled `sample_latent_train()` path when
  `consistency_train_align_infer=True`.
- Default behavior remains unchanged because `consistency_train_align_infer`
  defaults to `False`.
- Smoke run `g5_alignbc_smoke_s42_20260602_055909` verified:
  - `train_align_latent_requires_grad/frame=1.0`
  - `consistency_grad_norm/frame=7.880384`
  - eval16 seed42 reward `3.897836`
  - decision: code-path smoke pass only, no algorithm-quality claim.
- Directional scout `g5_alignbc_s43_20260602_060054` trained only seed `43` for
  `450k` agent steps and externally evaluated eval512 seeds `42,43,44`.
- External checkpoint comparison:
  - `model_best` alias: reward mean `5.110040`, done `0.000041`
  - `model_best_train`: reward mean `5.181952`, done `0.000122`
  - `model_last`: reward mean `5.197552`, done `0.000163`
- This passes the numeric seed43 scout threshold through external checkpoint
  selection, but the gain over the comparable G4C seed43 anchor is still small
  and single-seed.
- The in-training `model_best` alias is not the best checkpoint, so the external
  selector policy remains required.
- Subagent review: directional scout pass via external selector, not formal;
  no obvious privileged-inference leakage; latent MSE is a tuning caveat.
- Claude Opus review: `TUNE`; mechanism works, but the result is marginal and
  `model_last` suggests the run may not have plateaued.
- Decision: `DIRECTIONAL PASS / TUNE`. Keep the patch behind the explicit flag
  and do not claim a formal algorithm win.

G5B medium-scout update:

- G5B extended the same train seed `43` to `800k` agent steps with the same
  train-aligned action BC settings.
- Internal eval-select history:
  - `100k`: reward `4.987697`, done `0.001587`
  - `200k`: reward `5.146397`, done `0.000488`
  - `300k`: reward `4.974154`, done `0.001465`
  - `400k`: reward `5.002417`, done `0.000854`
  - `500k`: reward `5.263258`, done `0.001099`
  - `600k`: reward `5.188082`, done `0.000977`
  - `700k`: reward `5.275939`, done `0.001465`
  - `800k`: reward `5.286277`, done `0.001465`
- External eval512 checkpoint comparison:
  - `model_best` alias: reward mean `5.218559`, done `0.000081`,
    latent MSE `0.076395`, action MSE `0.027387`
  - `model_best_train`: reward mean `5.156018`, done `0.000122`,
    latent MSE `0.067694`, action MSE `0.030968`
  - `model_last`: reward mean `5.218559`, done `0.000081`,
    latent MSE `0.076395`, action MSE `0.027387`
- G5B improves over the G5 450k external best `5.197552`, but misses the
  predeclared medium-scout PASS gate `5.23` by about `0.011`.
- Result reviews agree on `TUNE`: near-pass, not formal; single train seed and
  train/eval seed overlap remain overclaim risks.
- Latent MSE is worse than G4D/G4C/G5, so reward/action gains carry a
  decode/latent-quality caveat.
- Decision: `TUNE / near-pass`.

G5C train-seed stability scout:

- G5C tested the same train-aligned BC settings at `450k` agent steps for train
  seeds `42` and `44`, with external eval512 seeds `42,43,44`.
- Best external checkpoint per train seed:
  - train seed `42`: `model_best_alias`, reward mean `4.960817`, done
    `0.000041`, latent MSE `0.071036`, action MSE `0.039160`
  - train seed `44`: `model_best_alias`, reward mean `5.102461`, done
    `0.000122`, latent MSE `0.070952`, action MSE `0.037441`
- Internal eval-select showed a shared peak-then-soften pattern:
  - train seed `42`: `4.811486 -> 4.906337 -> 4.931985 -> 4.433161`
  - train seed `44`: `4.842363 -> 5.008591 -> 5.172998 -> 5.034783`
- Claude Opus review recommended `REJECT`: train seed `42` falls below the
  `5.05` reject floor, and train seed `44` remains below the `5.17` stability
  target despite a useful `300k` internal peak.
- Subagent result review was requested but unavailable because the existing
  subagent id could not be parsed and the agent thread limit prevented spawning
  a new reviewer.
- Decision: `REJECT as a seed-stable improvement / keep as instability
  evidence`.

Current deploy-candidate policy:

- Keep in-training eval-select as a candidate generator and aliasing mechanism.
- Use post-training external fixed-step eval over `model_best`,
  `model_best_train`, and `model_last` before assigning the deploy candidate.
- Do not claim any fixed alias kind is universally correct.

Next immediate direction:

- Use G4D as the standalone-consistency formal anchor.
- Treat G5/G5B as seed43-only train-alignment evidence, not as a formal win.
- Treat G5C as a seed-stability reject for the current train-aligned BC settings.
- Do not spend formal or `800k` multi-seed compute on the same G5C settings.
- Next run should be a G5D stability scout before new architecture work:
  - one main hypothesis only: stabilize the `300k` peak instead of extending the
    same unstable schedule;
  - start with train seed `44`, because it had the best G5C internal peak;
  - use either LR decay after about `250k` or a lower consistency LR;
  - keep task, teacher, eval seeds, and external selector unchanged;
  - use local GPU with `EVAL_NUM_ENVS=8` unless speed is more important than
    desktop responsiveness;
  - success requires holding reward near or above `5.17` at `400k/450k` and
    improving external eval512 without action-MSE regression.

G5D LR-schedule smoke:

- A default-off consistency LR schedule path has been added for the G5D
  hypothesis. Static checks passed.
- Smoke run `g5d_lr_decay_smoke_20260602_092220` used only `128` agent steps
  and eval8. It verified config wiring, exit status, and LR logging.
- Observed LR logs under a toy schedule were `0.00030000` at step `32`,
  `0.00018187` at step `64`, `0.00004687` at step `96`, and `0.00003000` at
  step `128`.
- This smoke is not reward evidence. The next meaningful test remains the G5D
  `450k` seed-44 scout with eval512 seeds `42,43,44`.

G5D LR-schedule scout:

- Run `g5d_lrdecay_s44_20260602_092610` trained seed `44` for `450k` agent
  steps with linear LR decay from `250k` to `450k`, final scale `0.1`.
- Internal eval-select reproduced the G5C pre-decay rows exactly at `100k`
  (`4.842363`) and `200k` (`5.008591`), then scored `5.162218` at `300k` and
  `5.263977` at `400k`.
- Compared with G5C seed `44`, the `300k` value was nearly preserved
  (`-0.010780`), while the `400k` value improved by `+0.229194`.
- Final in-training eval-select scored `5.423057`.
- External eval512 over seeds `42,43,44` selected `model_best_alias`/`model_last`
  with mean reward `5.348209`, done mean `0.000081`, latent MSE `0.043561`, and
  action MSE `0.023138`.
- Decision: `SCOUT PASS / not formal`. This supports LR decay as the next
  stability direction, but it remains one train seed and must be checked on
  train seeds `42` and `43` before any formal claim.
- Claude Opus result-review verdict: `APPROVE`; caveats are recorded in
  `Goal/metric_card_consistency_g5d_lr_decay_stability.md`.
- Local resource note: train `NUM_ENVS=16` stayed near budget, but external
  eval `EVAL_NUM_ENVS=8` produced short `86-88%` GPU-utilization peaks. Future
  local eval should use `EVAL_NUM_ENVS=4` when desktop responsiveness matters.

Metric card:

```text
Goal/metric_card_consistency_g3_seed_stability.md
Goal/metric_card_consistency_g4_eval_select.md
Goal/metric_card_consistency_g4b_selector_metric.md
Goal/metric_card_consistency_g4c_selector_formal.md
Goal/metric_card_consistency_g4d_external_selector_formal.md
Goal/metric_card_consistency_g5_train_align_bc.md
Goal/metric_card_consistency_g5b_medium_scout.md
Goal/metric_card_consistency_g5c_seed_stability.md
Goal/metric_card_consistency_g5d_lr_decay_stability.md
Goal/metric_card_consistency_g5e_lr_decay_seed_stability.md
```

## 11. Goal Mode Starting Prompt

Use this prompt for the next Goal Mode run:

```text
You are working in /home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro-public on branch diffusion.

Use Goal/diffusion_goal_plan.md as the current execution plan.
Use Goal/goal_overview.md only as raw background.

Canonical task:
Dexh13HoraLightbulbSim2RealTwoFingerCoDrive

Canonical teacher:
sim2real/codrive/best_reward_4159.37.pth

Do not use CoDriveThesis as the teacher for this round.
Do not touch private repo or DOTPG optimization flow.
DOTPG is a frozen baseline/reference only.

Main objective:
Optimize diffusion-family student algorithms under the CoDrive 4159 teacher.

Current priority:
G4D external-selector formal follow-up is complete and is the current
standalone-consistency formal anchor: eval2048 overall reward 5.088728.
G5 train-aligned action BC then passed smoke and produced a directional seed43
scout signal: external eval512 model_last mean reward 5.197552 and
model_best_train mean reward 5.181952. Treat G5 as DIRECTIONAL PASS / TUNE, not
as a formal win. G5B extended seed43 to 800k and reached external best
eval512 mean reward 5.218559, which is TUNE / near-pass but still below the
5.23 medium-scout PASS gate. G5C tested train seeds 42 and 44 at 450k and
rejected the same settings as a seed-stable improvement: train42 best external
mean reward was 4.960817 and train44 best external mean reward was 5.102461.
G5D added a default-off consistency LR decay schedule and passed the seed44
stability scout: internal eval-select improved the 400k row from G5C's 5.034783
to 5.263977, and external eval512 selected model_best_alias/model_last with
mean reward 5.348209, done mean 0.000081, and action MSE 0.023138. Treat this
as SCOUT PASS, not formal.

Deploy candidate policy:
Keep in-training eval-select as candidate generation, but choose deploy
checkpoints by post-training external fixed-step eval over model_best,
model_best_train, and model_last. Do not assume one fixed alias is always best.

Next design branch:
Before new architecture work, run a train-seed stability scout for G5D on train
seeds 42 and 43 using the same LR schedule. Keep eval512 seeds 42/43/44 and the
external selector over model_best/model_best_train/model_last. Use
EVAL_NUM_ENVS=4 locally when desktop responsiveness matters. If at least one
additional train seed clears the G5D gates without action-MSE regression, then
promote to a formal multi-seed decision; otherwise tune the LR decay schedule
instead of adding architecture complexity.

Primary metric:
fixed-step eval reward, selected by unified eval rather than training Current Best.

Hard constraints:
No privileged info at student inference.
No task reward / horizon / eval seed manipulation.
Main-line candidates must not fall below PAdapt fixed-step reference.
Default deployment target is NFE <= 2.
Local GPU target is around 80% utilization / VRAM pressure.
Use subagents for bounded audits.
Use Claude Opus cross-review for important direction, code, and result decisions.

Before code changes:
write the metric card, audit artifacts, check GPU budget, and identify exact baseline checkpoint paths.
```
