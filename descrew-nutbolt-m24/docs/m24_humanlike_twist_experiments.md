# M24 Human-Like Twist Experiment Ledger

## 2026-09-10 -- Cycle learning and validated release-state seeds; full goal still incomplete

- Target: learned drive -> release -> return -> regrasp -> repeat. Implemented opt-in `XHandPasiniM24NutBoltE002Cycle`,101Dphase observation, measured event transitions, global high-water across cycles, separate phase failures and fixed first-episode cycle evaluator. Original E002 remains available. Report: `docs/m24_e002_cycle_learning_2026-09-10.md`.
- Cloud remains shutdown after prior no-inventory power-on failure; user-authorized local RTX3060fallback used. **All local training/eval/record/replay jobs are finished; no GPU compute process remains.** Final provider status verified `shutdown`, request `67efaba785cfcdad4900eb02d1b68baa`. These updates are local-only and pending cloud sync. No30minute or student run.
- Three separately bounded comparisons, each fresh passed A0c ->60s smoke ->300s PPO ->best/last512-env evaluation: `outputs/local_pipeline_m24_e002_cycle_20260910_probe3072`, `...cycle_lr1e4_20260910_probe3072`, `...relativeforce_20260910_probe3072`. Each phase records exact command/timestamps/checkpoints; training exit124 expected, all six evals exit0 and finite. Goal incomplete in every JSON.
- First cycle run: last505drive/0release,median56.99deg; reward-best97drive/0release,median1.33deg,allworkspace failures. InitialKL4.335against0.02 prompted an isolated initialLR0.005->0.0001comparison. That reduces firstKL to0.0107; best/last512drive/0release,median23.25/58.52deg,all release timeouts. Late stochastic angle-jump failures persist.
- Third comparison changes only release potential to recorded-entry-relative force; keeps0.5scale and all actual event criteria. Supported by successful-trajectory counterfactual (early reward0.00356->0.12659,total0.48949->0.5). Best512drive/0release,median24.89deg; last0drive/0release,median2.99deg,512drive timeouts. This did not solve learning and is retained as an opt-in comparison, not promoted as a solution. No autonomous release/return/regrasp/repeated cycle has been demonstrated.
- Physical action diagnostics:55low-amplitude candidates identify outward unloading; directed8candidate test yields6true pair releases retaining23–24deg. Video `outputs/local_pipeline_m24_e002_cycle_20260910/release_reference_video/drive_then_scripted_release.gif` records one47frame/20Hz episode: learned drive plus SCRIPTED release, finalforces0/0N,net23.796deg. Inspected loaded/released frames. This is feasibility evidence only.
- New `replay_m24_e002_release_states.py` reconstructs5recorded intermediate states and continues recorded actions:5/5release. Early states have transientforce differences up to4.12N; near-release states40/43have maximum angle error0.00649/0deg and force error0.00990/0.02688N. Seed archive and replay results are under the video and `release_state_replay/` directories. Archive itself retains unvalidated provenance; separate replay JSON supplies validation evidence. Solver caches are not serialized.
- Actual tensor tests, relative unloading/force-oscillation/reset tests, existing learning and A0c contract checks, syntax checks all pass. Normalization preparation changes only constant phase-bit statistics, not modelweights. Recorder/export/source snapshot fixes are included; train/task config snapshots now use distinct names. Full source and consolidated results: `outputs/local_pipeline_m24_e002_cycle_20260910/final_source` and `session_results.json`.
- Preserve the original stable-drive default: `outputs/XHandPasiniM24NutBolt_teacher/local_pipeline_m24_e002_contactprogress_20260910_retry3072_ppo/stage1_nn/last.pth` (SHA `ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`). Do not automatically select the newest cycle checkpoint.
- **Single recommended next step:** implement an opt-in DemoStart-style release-state reset curriculum using validated states40/43 first, within existing PPO. Preserve physically verified target/history/cycle state reconstruction and no free angle credit; gradually move to earlier states. Evaluate full behavior from original candidate743 with no scripted actions or demonstration resets. Primary-source basis and exact seed/replay paths are in the report. Do not repeat the failed full-start settings unchanged.

## 2026-09-10 -- Revised gates produced stable learned20degree drive locally

- User authorized relaxing normal-moment/mature-success gates and local training when cloud has no resources. AutoDL power-on failed with no inventory (`InternalError`, request `5dc32034b9c3ed3712ae56d689677ce8`); instance remains shutdown. This entry is local-only until cloud becomes reachable. No local/cloud training or evaluation job remains active.
- Report: `docs/m24_e002_gate_revision_2026-09-10.md`. Active `acceptance_mode=contact_progress` uses nut-only contact and actual new rotation for drive progress/milestones. Normal moment/balance remain diagnostics. Early90%/95% thresholds and exact50ms synchrony no longer have veto power. All prior conflicting plan restrictions are superseded by the September10 amendment.
- Actual reward CPU-tensor tests, A0c false-pass tests and existing expression checks pass. Fresh local candidate743 A0c explicitly passed, candidate22,17 contact/direction steps,7.362deg new motion under pair contact. SHA `e67e3626ae141bf083241edfa67bc959016970b2ae6ee5b11a005fede8a58d49`.
- Local RTX3060/12GiB; existing IsaacGym Docker, torch2.3.1+cu121. Smoke60s ended expected124. A launcher falsely matched historical diff text containing Traceback; fixed anchored matching, reused the verified smoke with identical A0c/starting weights. Original evidence retained in `outputs/local_pipeline_m24_e002_contactprogress_20260910_probe3072`.
- PPO pipeline `outputs/local_pipeline_m24_e002_contactprogress_20260910_retry3072`:3072envs/6144minibatch,15:46:39–15:51:46 +08,300s timeout plus7s completion grace,expected124.43completed updates, counter1,622,016 includes initial-batch offset. Warm-started September8 best policy/normalization with fresh optimizer. Sampled peak GPU4724MiB, stable5.2–5.5k environment steps/s.
- **Matched local revised-contract evaluations:** old starting policy0/512 hold20deg for4pair-contact steps,0successes,510workspace+2contact losses. Latest policy512/512 hold20deg,159/512 meet25–35deg drive band+4step hold,353drive deadlines,0workspace/contact-loss/angle-jump terminations; median24.64deg.0/512 reaches30deg. Reward-best has median9.30deg and0successes, so select `last.pth` by behavior.
- Selected directory `outputs/XHandPasiniM24NutBolt_teacher/local_pipeline_m24_e002_contactprogress_20260910_retry3072_ppo/stage1_nn`; `last.pth` SHA `ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`; `best_reward_5.44.pth` SHA `d91fcd9a39c0e7ce3bb388f1f1a6cc0f03e7101bfd8c4831671a597f948e50dc`. Full source/commands/checkpoints/TensorBoard/evaluation traces retained locally.
- A0c/analysis pipeline `outputs/local_pipeline_m24_e002_contactprogress_20260910`; `results_summary.json` and `policy_video/learned_drive.gif` document this result. One actual81frame/20fps first episode reaches23.67deg then holds until deadline. Inspected rendered initial/held frames; no resets are concatenated as cycles.
- Known applied nut-torque diagnostic is now complete:12cases with/without hand,200physics samples each. No-contact0.05/0.1Nm yields20.03/66.58deg; endpoint velocity integrals stay near0 even without finger contact. The grasp resists0.1Nm (2.02deg). This invalidates using endpoint qdot alone as a motion veto, not a proof of complete physical fidelity. Artificial nut torque is diagnostic only; production physics/actuation unchanged.
- Remaining limits: full release/return/regrasp/repetition is not trained; stochastic exploration still exhibits frequent large-angle terminations; normal-force median11.68/28.90N and maxima15.81/33.90N are not load-optimized. Full goal remains incomplete; drive success is not cycle success.
- **Single next action:** add and test release/return/regrasp learning using the already demonstrated approximately20degree stable stroke. Preserve net rotation across cycles; do not wait for30degree/90% maturity, and do not count phase timeouts as success. No automatic30minute run or student stage.


Status: active  
Start date: 2026-09-03  
Working branch: `feat/m24-humanlike-twist`

## 2026-09-08 -- Nut-contact reward comparison completed; strict goal unmet

- User explicitly authorized the pending reward-multiplier exception. This comparison is completed; do NOT ask for its approval again. All training was cloud-only. Goal remains uncompleted; no B/C, student or30-minute training was launched.
- Report: `docs/m24_e002_nutcontact_comparison_2026-09-08.md`.
- One production change: `forward_source=nut_contact_quality`, positive high-water progress weighted by minimum opposing NUT-specific contact confidence. Legacy pair-quality mode retained. All weights and strict gates unchanged. Nut-only force prevents bolt-contact credit. Actual-expression and A0 checks pass.
- Pipeline: `outputs/cloud_pipeline_m24_e002_nutcontact_20260908_175230`. A0c explicitly passed4/27, selected19; physical traces match previous A0c exactly. SHA256 `6e344b6ccf47fa9331a8b3a7d7af2887ee43a99b9512dad293ae8bbe21b5e6b7`. Candidate19 identical-trajectory reward−0.15029 ->+9.23656, not a learned pass.
- Smoke17:54:05–17:55:06+08, expected124. PPO12288/24576,17:55:06–18:00:13+08, timeout300 plus7s completed-update grace, expected124. Best update10; last update11/counter1,769,472 (initial-batch offset included). Checkpoints are inference snapshots, not optimizer state.
- Best/last fixed512-env,100-step first-episode evaluation both0/512 success and0/512 reached30deg. Median net26.0711deg /18.3390deg; mostly monotonic, but stalls around3.7deg under increasing thumb pressure then loses contact. Best469/512 and last512/512 terminate on workspace. Strict credited ratios41.2%/50.45%. `phase.txt` / `acceptance.json`: `rejected:fixed_step_numeric_gate`. No headed pass.
- Checkpoint directory: `outputs/XHandPasiniM24NutBolt_teacher/m24_e002_nutcontact_s42_20260908_175230_probe_12288_24576/stage1_nn`. Best SHA `b1e8123447fc31340373a93af8ac93cc7d7e988a6b67e8a4594b91bfbad8ef31`; last SHA `9859db2503758d11b0c92e0c1f56c27ce65270d00e89d177d528a67356fadd65`. Files are synced and hashes match.
- Frozen-policy target-clamp and6-step startup-settling diagnostics both fail to solve behavior; neither is applied to production. Contact-point gait checks do not establish a direction-only reward fix. `gait_audit` first attempt failed a tensor-view indexing check; corrected diagnostic is `gait_audit_retry`.
- New reference substep audit:800 physics samples still gate0 everywhere. Angle31.54448deg but sampled qdot integral0.79903deg; diagnostic velocity-iterations1 gives30.02368deg vs0.00296deg integral, still gate0. Thumb normal moment negative throughout baseline. Full friction remains unmeasured. Do not mistake geometric reachability or endpoint velocity for validated torque-driven motion. PhysX internal position/velocity iteration semantics may explain differences; no simulator bug or all-motion artifact is established. No physics setting was changed in production.
- Current cloud: no tmux/train/eval/probe job; GPU idle. Source, commands, logs, timings, plots, full raw diagnostics and checkpoints preserved.
- Single next action: diagnostic-only known nut torque, with and without finger contact, to distinguish actual angle, sampled velocity and constraint response before further reward design. This next diagnostic is NOT yet executed. No repeated PPO until an evidenced fix and fresh A0c; strict gates and weights remain pinned.

## 2026-09-08 reset-history correction and reachability result

- Report: `docs/m24_e002_reset_and_reachability_2026-09-08.md`.
- E002 reset-history order fixed; real cold/subset test and unchanged A0c pass.
  One new cloud smoke + five-minute PPO completed, bestupdate11/lastupdate12
  preserved; both still0/512 strict successes. No reward/gate change in this cycle.
- Same-pose diagnostic contact-force/PD reference reaches31.54degrees with
  80/80 nut-only pair contact and normal gate0. Removing either tangential
  command reduces angle to~0.001degrees. Command targets are not measured
  wrenches; this supports cooperative reachability, not an RL policy pass.
- CPU reference did not recover friction: extra force was thumb-bolt contact.
  Friction fields stay0; clean resultants reconstruct from normals alone.
- Pipeline `outputs/cloud_pipeline_m24_e002_reset_history_20260908_171000` has
  raw records, ablations, checkpoints and figure, synced locally/cloud.
- Literature-backed contact-progress comparison is prepared but unapplied,
  awaiting the requested explicit TRAINING-gate exception. Strict acceptance
  and all weights stay fixed. No blind repeat, longer training, B/C or student.

## 2026-09-08 reward audit: A0c passed, corrected PPO still rejected

- Report: `docs/m24_e002_reward_audit_2026-09-08.md`.
- Changed one invalid-forward penalty source, preserving positive torque-gated
  reward, all coefficients and thresholds. Physical A0c traces unchanged;
  4/27 candidates passed. Offline candidate-19 motion component -9.5371 ->
  -0.15029, which is not a learning result.
- Pipeline `outputs/cloud_pipeline_m24_e002_reward_contactdeficit_20260908_161444`:
  initial smoke config-resolution failure preserved; corrected `a1_retry`
  smoke and five-minute PPO exited expected124. Best epoch1 and final epoch11
  were both saved and evaluated: 0/512 successes each, near-zero angles,
  thumb-contact-loss terminations step15 / step10, gate0, credited progress0.
- Run rejected. Full artifacts and both checkpoints synced and hash verified.
  No 30-minute, B/C or student; cloud idle. User goal remains unmet.
- Read-only first-observation audit proves stale/zero actor history assembled
  before reset history initialization; same-state refill alters first actions.
  Next single fix: E002 reset-history order, then A0c and reevaluate existing
  checkpoints before another PPO. No second task change made in this cycle.

## 2026-09-08 result: corrected A0c passed; five-minute E002-A1b rejected

- Report and all artifact paths: `docs/m24_e002_cloud_recovery_2026-09-08.md`.
- Corrected actual normal-contact application points after two-env validation;
  preserved reward weights and all thresholds. CPU tensor access / GPU PhysX
  exposes contact records, with CUDA PPO observation copies. Full friction is
  explicitly unmeasured. Contact-aware actions and late increment taper give
  A0c `passed`: 4/27 candidates, selected sign -1, streak 7, net 0.12568 rad.
- A0c accepted-window progress is only 0.000760 rad, so it is not a useful
  sustained-stroke result. Full signed per-step traces and correlated top-k
  remain authoritative; do not use isolated maxima to infer a behavior pass.
- Cloud pipeline `outputs/cloud_pipeline_m24_e002_passed_a1_20260908_154343`:
  smoke timeout 60 and PPO timeout 300 completed expected 124, 12288/24576,
  ~5.8k actor steps/s. Best reward -1.06909 is epoch 1 / counter 294,912;
  no final last.pth. Checkpoint and exact commands/configs/statuses are synced.
- Selected best checkpoint fixed evaluation: 512 envs / 100 steps, success
  0/512, contact loss 512/512 at steps 13–14, median net 1.633e-7 rad, zero
  credited steps, no hard-speed terminations. Numeric gate rejected; headed
  acceptance not performed under the immediate hard-gate stop rule.
- No 30-minute training, E002-B/C or student; cloud is idle. User motion goal
  remains unmet. Next experiment: compare early PPO/zero/A0c action traces on
  cloud and isolate thumb loss before another bounded change, with weights
  and pair threshold fixed.


## 1. Objective

Optimize the M24 nut-bolt policy so that index and thumb perform a repeatable
human-like cycle:

1. both fingers establish contact;
2. both fingers contribute positive torque during the drive stroke;
3. both fingers unload and release within a short synchronized window;
4. both fingers return without dragging or reversing the nut;
5. both fingers regrasp for the next drive stroke.

The immediate stage evaluates the motion reward in simulation. Simulated
contact-load reward tuning and real DexH13 motor-current work are deferred until
the two-finger drive/release behavior is accepted.

## 2. Fixed constraints

- The bolt base remains fixed in every training and evaluation variant. Do not
  use a floating, compliant, or weakly constrained base as part of this task.
- Base stability is evaluated through reaction-force and reaction-moment
  proxies while the base remains fixed. In particular, track lateral wrench
  that would tend to tip an equivalent unfixed assembly.
- The active path remains `XHandPasiniM24NutBolt -> PPO teacher -> evaluation`.
  Retrain or distill a student only after teacher behavior passes the
  current-stage motion gates below.
- Only index and thumb may act in the motion experiments. Middle and ring use
  the explicit mask `action_mask_indices=[4,5,6,7,8,9,10,11]`.
- Preserve existing train, evaluation, export, and deployment entrypoints.
- Each experiment changes one primary hypothesis whenever practical. Any
  coupled changes must be identified explicitly.
- A higher episodic reward is not sufficient evidence of improvement.

## 3. Deferred DexH13 current-signal contract

User decision on 2026-09-03: do not implement, log, calibrate, or use motor
current during the immediate reward-training stage. Motor current is not an
acceptance gate for the E002 motion stages.

The DexH13 API can provide motor current through
`getMotorActualCurrent()`. For the first optimization stage, current remains
outside the policy observation and is used for:

- timestamped telemetry and sim-to-real calibration;
- a deployment-side safety governor;
- stopping further closing-target accumulation when current is excessive;
- controlled retreat or motor disable at a hard threshold;
- offline comparison against simulated joint torque and fingertip contact
  force.

The existing `r3_s5` TorchScript observation contract is frozen, so motor
current cannot be appended to that policy at deployment without changing the
input shape and retraining/exporting a compatible policy.

A later current-aware policy is technically possible, but it is a separate
experiment. It requires a calibrated simulation proxy for current, appropriate
noise/delay/randomization, observation normalization, teacher/student training,
and a new export contract. Until those prerequisites exist, current must not be
treated as an unmodeled policy input.

## 4. Reward design versions

All reward terms must be normalized before their scales are compared. Record
both raw and scaled means for every enabled term.

| Version | Rotation credit | Pair behavior | Load behavior | Phase behavior | Status |
| --- | --- | --- | --- | --- | --- |
| `R0` | Positive nut velocity directly earns reward | Mean thumb/index proximity; no proof both contribute | Global joint torque/work penalties only | None | Existing baseline |
| `R1-motion` | Positive velocity in `drive`, gated by pair torque | Pair contact/torque proxy | Diagnostics only | Timeouts incorrectly allowed phase progress | **Rejected: oscillation and continuous-contact loopholes** |
| `R2A-monotonic` | Only new unwrapped high-water angle earns reward; reverse travel is always penalized | Pair gate applies only to new progress | Diagnostics only | Drive-only; phase observation slots reserved | Next experiment |
| `R2B-release` | Accepted R2A progress reward | Same as R2A | Diagnostics only | Strict unload/off-dwell/return; every timeout is failure | After R2A passes |
| `R2C-cycle` | Accepted monotonic progress plus valid-cycle completion | Same as R2A | Diagnostics only | Repeated strict drive/release/return/regrasp | After R2B passes |
| `R3-load` | Same as accepted R2C | Same as accepted R2C | Add soft overload/impulse penalties | Same as accepted R2C | Deferred until motion passes |

### 4.1 Planned pair-drive term

For finger `i` in `{index, thumb}`, define its signed torque about the nut axis:

`tau_i = dot(cross(p_i - p_nut, F_i_on_nut), nut_axis)`

The E002 candidate is:

`r_pair_drive = new_high_water_progress * min(g_index, g_thumb)`

where `g_i` combines pair-specific contact confidence with normalized positive
`tau_i`. A finger merely touching the nut while the other finger does all the
work must not satisfy this gate.

Use a dead-band imbalance penalty on normalized torque contribution instead of
forcing equal raw contact forces. Different finger lever arms can require
different forces while still producing a balanced pair drive.

### 4.2 Planned fixed-base stability proxy

Because the base must remain fixed, do not use base displacement as the
stability signal. Record and, after instrumentation is validated, penalize:

- lateral resultant contact force on the nut/bolt assembly;
- bending moment about the two horizontal base axes;
- single-finger drive while nut angular velocity is positive;
- short high-impulse impacts that are hidden by policy-step averages.

Normalize the wrench penalty by nut size and a calibrated reference load so its
scale is comparable across domain-randomized environments.

### 4.3 Planned contact-load term

For each active finger, use pair-specific normal contact force where available:

`p_over = sum(relu(F_normal_i - F_soft)^2)`

Track a separate hard threshold `F_hard`. `contact_force_max` used to saturate a
contact-confidence weight is not a physical force cap and must not be reported
as one.

Do not assign final values to `F_soft` or `F_hard` until simulated force/torque
has been aligned with low-speed DexH13 motor-current measurements. Penalizing
all force can encourage slipping or fast impacts; penalize force above the
minimum useful contact band instead.

### 4.4 Planned phase behavior

The phase reward requires state/history; an instantaneous contact reward cannot
distinguish intentional synchronized release from failed contact.

- `grasp`: both contacts rise above an on-threshold with hysteresis.
- `drive`: both signed torque contributions are positive.
- `unload`: after a target stroke angle or joint-margin trigger, both contacts
  fall below an off-threshold within the configured release window.
- `return`: both fingers remain unloaded and nut reverse motion is penalized.
- `regrasp`: both contacts recover before another drive reward is available.

Record the phase trigger and transition reason for every rollout used in an
acceptance decision.

### 4.5 E001 failure-derived hard rules

- Never reward gross positive angular velocity. Reward only new unwrapped
  high-water progress and separately penalize every reverse increment.
- The existing M24 asset has a revolute nut joint only. Report equivalent M24x3
  virtual screw advance as
  `advance_mm = net_unwrapped_angle * 3 / (2*pi)`. The rendered nut will not
  physically translate along the bolt unless the asset is redesigned.
- A drive timeout is failure, not permission to enter unload.
- An unload timeout without both fingers off is failure, not permission to
  enter return.
- A return with either finger still in contact is failure and cannot transition
  to regrasp.
- A release lag is defined only for a valid double-release event. Missing
  releases must be counted separately and must never default to zero lag.
- Static contact cannot earn an accumulating proximity reward. Approach
  shaping must be potential-based and stop once contact is established.
- No checkpoint may be selected from episodic return alone.

## 5. Required metrics

### 5.1 Simulation behavior

| Metric | Definition |
| --- | --- |
| Nut progress | Unwrapped final net angle, new-high-water angle, gross absolute travel, and reverse travel per episode/cycle |
| Net/gross efficiency | `max(final_angle-start_angle, 0) / sum(abs(delta_angle))`; exposes back-and-forth reward exploits |
| Virtual screw advance | M24x3 equivalent net/high-water advance in millimetres using `3 mm` per revolution |
| Pair-drive ratio | Fraction of positive-rotation steps where both fingers exceed the minimum positive torque contribution |
| Single-finger drive ratio | Fraction of positive-rotation steps where exactly one finger contributes above threshold |
| Contact overlap | Fraction of drive steps with both pair-specific contacts active |
| Torque share | Per-finger positive axial torque and normalized imbalance |
| Valid release rate | Fraction of unload attempts where both contacts turn off before timeout |
| Release lag | Absolute time between contact-off events, computed only for valid double releases |
| Phase failure rate | Drive timeout, unload timeout, contact-during-return, and invalid-regrasp rates reported separately |
| Return drag | Contact and reverse nut motion during the return phase |
| Force distribution | Per finger mean, P50, P95, P99, maximum, and time above `F_soft` |
| Contact impulse | Per finger and combined impulse per drive stroke |
| Fixed-base wrench | Lateral force and horizontal bending-moment mean/P95/P99/max |
| Smoothness | Action rate, joint acceleration, jerk, saturation, and direction reversals |

### 5.2 Real DexH13 deployment — deferred

Do not block or delay E002 motion implementation/training on this subsection.
It is retained only as the later sim-to-real record contract.

Every controlled real run must preserve:

- policy name/hash and matching init pose/action mask;
- configured and measured policy rate;
- measured joint position, previous/next target, and action;
- per-motor actual current from `getMotorActualCurrent()` with synchronized
  timestamps;
- target-position error and consecutive steps at its guard limit;
- per-motor current mean/P95/P99/max and time above soft limit;
- soft-limit intervention, retreat, hard-stop, communication fault, and E-stop
  event counts;
- video path and a short manual behavior verdict.

Motor-current thresholds are `TBD` until a separate low-speed calibration is
recorded. No reward experiment authorizes raising a deployment safety limit.

## 6. Provisional acceptance gates

These are initial engineering gates and may be tightened after the baseline is
instrumented. Any change must be recorded as a ledger event rather than edited
silently.

- E002-A final net angle is positive in at least `95%` of evaluation
  environments;
- E002-A net/gross travel efficiency: at least `0.80`;
- E002-A reverse/gross travel: at most `0.10`;
- pair-drive ratio: at least `90%`;
- single-finger drive ratio: at most `5%`;
- E002-B/C valid double-release rate: at least `95%`;
- median valid-release lag: at most `150 ms`;
- P95 valid-release lag: at most `250 ms`;
- drive/unload timeout success count: exactly zero;
- no sustained nut reverse motion during any phase;
- fixed-base lateral wrench and bending moment are recorded as diagnostics but
  are not pass/fail gates for E002 motion training;
- headed video must visibly show repeatable two-finger drive and synchronized
  release for multiple cycles.

Motor current, real deployment, and force-overload thresholds are not part of
the E002 motion acceptance decision.

## 7. Experiment protocol

Before starting a run:

1. assign the next experiment ID;
2. state one primary hypothesis;
3. freeze the task YAML or record all Hydra overrides;
4. record branch, commit, dirty-tree state, seed, checkpoint, environment count,
   minibatch size, duration/step limit, and exact command;
5. identify the baseline and the metrics required for the decision;
6. verify no existing failed setting is being repeated without a fix
   hypothesis.

After a run:

1. record timestamps and exit status;
2. preserve logs, resolved config, reward-term summaries, selected checkpoints,
   and videos;
3. fill both quantitative and manual behavior results;
4. label the verdict `accept`, `reject`, `inconclusive`, or `invalid`;
5. state why, including reward exploits or safety events;
6. name exactly one recommended next experiment.

Recommended naming:

`m24_humantwist_<experiment-id>_<reward-version>_s<seed>_<date>`

## 8. Experiment index

| ID | Date | Reward | Policy/checkpoint | Primary change | Sim result | Real result | Verdict | Next |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `E000` | 2026-08-12 | `R0` | `r3_s5.pt`, SHA256 `dca0d430...561d56` | Existing deployed baseline; no controlled change | Existing task can rotate the nut, but the present logs lack pair torque, contact force, fixed-base wrench, and phase metrics | Video shows alternating thumb/index single-finger flicking; current was not logged | `baseline-only` | `E001`: implement and train `R1-motion` |
| `E001` | 2026-09-03 | `R1-motion` | `best_reward_549.29.pth` | Two-finger torque-gated velocity reward plus permissive phase machine | Forward/reverse travel cancel; false phase cycles; static contact exploit | Not in scope | **`reject`** | `E002-A`: learn monotonic two-finger drive only |
| `E002-A` | 2026-09-04 | `R2A-monotonic` | Transient smoke/probe checkpoints only; seed 42 | High-water net progress, global reverse penalty, no accumulating static-contact reward | Simulator contact/torque A0 gate failed: no valid pair drive, immediate contact-loss terminals | Not in scope | **`invalid`** | Controlled-action contact/torque sign trace; no extension |
| `E002-A0b/A0c` | 2026-09-08 | `R2A-contact-recovery` | No checkpoint | Fixed reset noise, automatic stable-pair pose search, action/sign calibration, physical-contact lifecycle, torque balance, invalid single-finger progress penalty | A0b found reset candidate 743; A0c failed `no_pair_positive_drive` over 1024 candidates (best pair streak 0); failed report lacks correlated per-candidate traces | Not in scope | **`reject`** | Instrument a diagnostic-only A0c trace from candidate 743; do not launch PPO |
| `E002-B` | Blocked on E002-A | `R2B-release` | Resume accepted E002-A | Strict double release and off-dwell return | Pending | Not in scope | `blocked` | Start only after E002-A passes |
| `E002-C` | Blocked on E002-B | `R2C-cycle` | Resume accepted E002-B | Repeated valid full cycles | Pending | Not in scope | `blocked` | Start only after E002-B passes |

The detailed E002 implementation and execution contract was revised after the
2026-09-04 literature review and local observation/termination audit. Use
`docs/m24_e002_training_plan_2026-09-04.md`; it takes precedence over the
abbreviated E002 records below where they differ.

### 2026-09-07 cloud execution availability

The requested A0b/A0c -> A1b execution did not start.  After explicit power-on
confirmation, AutoDL rejected instance `pro-786095bb6e03` with
`InternalError` because the requested compute specification had no inventory
(request `9c090abb63e5d5c549304c3aaf15fe27`); status remained `shutdown`
(request `073cc034045395ef1d68f66bd8b40b1f`).  Consequently commit `694a1b0`
was not synced, the mandatory live cloud handoff could not be read, and no
calibration JSON, simulator log, checkpoint, PPO run, or headed video exists
for this attempt.  Verdict: **`infrastructure-blocked`**.  Retry the exact
gated A1b launch only after cloud inventory returns; no reward gate is relaxed.

### 2026-09-08 E002-A0b/A0c cloud calibration

After inventory returned, the user-authorized `694a1b0` changed-file set was
synced to the idle cloud branch (cloud synchronization commit `973fa42`).  Two
pre-measurement type defects—NumPy joint limits used in Torch calibration and
reset-noise paths—were fixed as `339fad1`/`f8c9c3b` locally and
`16dcfa7`/`8c6e96a` on cloud; deterministic E002 tests then passed.  No reward
gate, force/current setting, action mask, or stage definition changed.

The final gated pipeline is
`outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1b_s42_20260908_103643`.
Its deterministic test passed, while calibration exited `4` with
`failed:no_pair_positive_drive`; the pipeline phase is `failed:a0_calibration`.
The JSON selected stable-pair reset candidate `743` from `1024` pose/action
candidates, with 16-DOF reset
`[0.214112,1.067002,0.241880,0.439259,0,0,0,0,0,0,0,0,-0.126119,1.425131,0.741017,0.472675]`.
A0b necessarily met its required ten-step stable-contact gate to enter A0c,
but A0c had `best_probe_streak=0` against the required `5`, best directed net
angle `0.0551694781 rad`, and best mean pair gate `0.0539684556`.  It therefore
selected no `force_sign` or two-finger drive action.  No smoke, PPO, checkpoint,
fixed-step metrics (`contact_acquired`, pair/single-finger ratios, net/gross,
reverse/gross, success), or headed video exists.  The required response was
observed: terminate before PPO without weakening the gate.  Full JSON/logs and
commands were synchronized locally under the same pipeline path.

Post-run audit qualification: the reported best angle and best mean pair gate
are independent maxima and may belong to different action/sign candidates.
The failure artifact does not contain candidate actions, correlated top-k
metrics, per-step contact/torque/gate traces, or per-condition pass counts.
Consequently the evidence proves that the full conjunction failed, but does
not yet distinguish wrong action basis, contact loss, insufficient/bad-sign
torque, an overly strong balance gate, or resultant-force contamination. The
next experiment is instrumentation-only on the recorded candidate-743 reset;
it must not simultaneously lower gates or alter PPO reward weights.

## 9. Detailed experiment records

### E000 — Existing R3/S5 deployment baseline

#### Hypothesis

None. This is historical evidence used to define the failure mode, not a
controlled optimization experiment.

#### Configuration and artifacts

- deployment policy: `outputs/local_m24_deployment/policies/r3_s5.pt`;
- policy SHA256: `dca0d430a423b78259d54d599a8ef7e242b8e9699de5851a660bea8854561d56`;
- recorded run: `outputs/m24_dexh13_runs/20260812T080118_347962Z`;
- attached video: `/home/vision/拧螺母.mp4`;
- hardware action scale recorded by the run: `0.02`;
- measured policy rate: approximately `20 Hz`;
- current telemetry: unavailable;
- matched pair-contact/torque telemetry: unavailable.

#### Observed behavior

- the nut rotates successfully;
- thumb and index alternate single-finger flicks instead of producing a
  sustained pair drive;
- the fingers do not consistently unload and release together;
- the motion can create unbalanced lateral load even though the simulated base
  is fixed;
- the real hand appears to apply excessive load, but the run cannot quantify it
  because motor current was not recorded.

#### Verdict

`baseline-only`. The evidence establishes the optimization target but is not
sufficient to choose force thresholds or attribute the behavior to a single
reward term.

#### Historical next experiment (superseded)

This E000-era recommendation was to run E001. E001 has since been rejected;
the active recommendation is E002-A below. Motor-current work and contact-force
overload reward remain outside the current motion experiment.

### E001 — R1-motion two-finger drive and synchronized release

#### Status

`rejected` after headed inspection and post-hoc metric audit on 2026-09-03.

#### Live execution updates

- local source commit: `3a203bf2764c0aaa255e717e72d9d7da6d12cfe1`;
  cloud execution commit: `8d729a217049b11272df5611ec11ff5d5eeecf2a`.
  The cloud repository has divergent historical ancestry, so only the three
  reviewed R1 files were synchronized; their R1 content is identical.
- cloud instance: AutoDL Pro `pro-786095bb6e03`, NVIDIA GeForce RTX 4090 with
  49140 MiB VRAM; no pre-existing compute job; 25 GiB free filesystem space.
- tmux: `m24_humantwist_e001_s42`; pipeline:
  `outputs/cloud_pipeline_m24_humantwist_e001_m24_humantwist_e001_r1motion_s42_20260903_160559`.
- one-environment GPU smoke: started `2026-09-03T16:05:59+08:00`, ended
  `2026-09-03T16:07:00+08:00`, exit `124` (`completed:expected_timeout`), no
  traceback/OOM/PhysX marker, finite episode rewards, best checkpoint
  `..._smoke_1env/stage1_nn/best_reward_6.98.pth`.
- the ordered `12288 environments / 24576 minibatch` capacity probe is active
  at this record. Initial observed VRAM is approximately 12443 MiB; do not
  choose a lower capacity unless its completion shows allocation or stability
  evidence.
- update `2026-09-03T16:10:00+08:00`: the 12288/24576 probe ended at its
  expected 180-second timeout (`124`), reached about 30.1k FPS, produced finite
  rewards (best `39.37`), and had no allocation/PhysX/NaN marker. It selected
  the formal capacity without fallback. The single formal `timeout 1800` PPO
  phase is active from scratch; its early reward rose `0.49 -> 1.14` and VRAM
  is about 12.5 GiB.
- final `2026-09-03T16:40:01+08:00`: formal phase ended at its expected
  1800-second timeout (`124`) after about 51M agent steps. It remained finite,
  reached about 28.7k aggregate FPS near the end, and had no traceback, CUDA
  OOM, PhysX error, illegal-memory-access, or NaN marker. The selected teacher
  is `best_reward_549.29.pth`, SHA-256
  `eb3a3f9f3cece152ad15f2b28ae3dca316e7ed04757a2011a4fe86c38a88585b`.
- local synchronization verified the same SHA-256 at
  `outputs/XHandPasiniM24NutBolt_teacher/m24_humantwist_e001_r1motion_s42_20260903_160559_formal/stage1_nn/best_reward_549.29.pth`.
  The complete cloud pipeline (commands, resolved configs, manifests, statuses,
  and smoke/probe/formal logs) is preserved locally under
  `outputs/cloud_pipeline_m24_humantwist_e001_m24_humantwist_e001_r1motion_s42_20260903_160559/`.

#### Final behavior and metric verdict

- User-headed inspection: the nut only oscillates back and forth, makes no net
  advance, and both fingers remain attached instead of releasing.
- Training-final last-20 TensorBoard means:
  - positive angular travel proxy: `0.4801`;
  - negative angular travel proxy: `0.4904`;
  - signed balance: approximately `-0.0103`, so gross motion cancels;
  - rotation credit still positive: `0.1546`;
  - contact overlap: `0.9670`;
  - return-drag ratio: `0.1434`;
  - pair-drive ratio: `0.7958`, below the `0.90` gate;
  - single-finger-drive ratio: `0.1743`, above the `0.05` gate;
  - mean transition count: `74.2`, despite no visible valid releases.
- The reported `release_lag ~= 0.003 s` is invalid evidence because missing
  releases retain the zero-initialized lag and are averaged as if successful.
- `reverse_penalty` is computed in `compute_reward` but is not passed into
  `compute_hand_reward`, making reverse motion free outside the weak return-only
  term.
- `drive_max_steps` forces drive into unload without progress; unload timeout
  forces return without release; persistent pair contact then returns to grasp.
- The global per-step proximity reward continues paying the policy to remain in
  static contact.

#### Final verdict

**`reject`**. `best_reward_549.29.pth` is preserved only as a documented
negative example and must not be resumed, distilled, deployed, or used as a
behavioral baseline.

#### Single recommended next experiment

Implement and run only `E002-A / R2A-monotonic`. Do not reintroduce release
phases until monotonic net progress passes its numeric and visual gates.

#### Primary hypothesis

Crediting rotation only when both index and thumb make positive axial torque,
combined with an explicit unload/return state and synchronized contact-off
reward, will replace alternating single-finger flicks with a repeatable
two-finger drive-and-release cycle.

#### Fixed configuration

- task: `XHandPasiniM24NutBolt`;
- PPO teacher trained from scratch with seed `42`;
- bolt base fixed;
- current M24 asset, saved init pose, joint limits, 20 Hz policy rate, and
  200 Hz physics unchanged;
- active actions: index `0:4` and thumb `12:16` only;
- masked actions: middle/ring indices `[4,5,6,7,8,9,10,11]`;
- no motor-current observation/logging/control work;
- no contact-force overload or impulse penalty in the reward;
- contact forces, per-finger torque, release timing, and fixed-base wrench may
  be added as diagnostics.

#### Required implementation behavior

- derive index/thumb contact confidence using fingertip force plus distance to
  the nut; record raw ingredients separately;
- compute each finger's signed torque about the nut axis;
- use the weaker positive per-finger contribution to gate positive rotation;
- expose pair-drive ratio, single-finger-drive ratio, per-finger signed torque,
  contact state, current phase, transition count/reason, release lag, return
  drag, and nut progress in rollout extras;
- implement phase hysteresis so contact-force noise does not rapidly toggle
  states;
- do not give positive drive reward during unload or return;
- penalize one-finger-still-contacting asymmetry during unload and nut reversal
  during return;
- require both fingers to re-establish contact before the next drive phase.

#### Original training protocol (historical)

1. Use AutoDL Pro instance `pro-786095bb6e03`. If it is off, follow the named
   `autodl-pro-manager` skill and obtain explicit user confirmation before
   powering it on. Do not record connection secrets in this repository.
2. Once reachable, read cloud-side
   `/root/code/dexscrew-repro/docs/cloud_session_handoff.md` before syncing or
   changing anything. Verify the actual GPU/VRAM, disk, repo branch/status,
   active `tmux` sessions, and running train/eval processes.
3. Sync or otherwise make the current `feat/m24-humanlike-twist` branch/commit
   available in `/root/code/dexscrew-repro`, preserving unrelated cloud edits.
4. Run deterministic reward/phase math tests where one finger, both fingers,
   wrong-direction torque, and asynchronous release are exercised explicitly.
5. Run one-environment GPU smoke validation for finite states/rewards and phase
   transitions.
6. If at least 24 GB VRAM is usable, probe `12288/24576`, then back off to
   `8192/16384` and `6144/12288` only with allocation/startup evidence.
7. Create a reproducible E001 cloud launcher under
   `outputs/cloud_pipeline_m24_humantwist_e001_*`; wrap the 30-minute PPO phase
   in `timeout 1800` and launch it in a named `tmux` session.
8. Treat timeout exit status `124` as expected, preserve the best checkpoint,
   and run normal-real-time headed visualization where the environment allows.
9. Fill this record with the exact code commit, resolved reward scales, command,
   output paths, metrics, video verdict, and next experiment.
10. Update both `docs/session_handoff_v2.md` and cloud-side
    `docs/cloud_session_handoff.md` with the active/no-active tmux state, exact
    pipeline/checkpoint paths, result summary, invalid results, and one next
    cloud action.

#### Original acceptance decision (failed)

The first run passes only if it visibly produces repeated two-finger drive,
both fingers unload within the configured release window, return does not drag
the nut backward, and the quantitative pair/single-finger/release metrics meet
the provisional motion gates. Episodic return alone cannot pass E001.

### E002-A — Monotonic two-finger drive

#### Primary hypothesis

R1 learned oscillation because it paid every positive velocity sample while
reverse samples were nearly free. Paying only new unwrapped high-water angle,
penalizing all reverse travel, and removing static-contact income will make net
advance the easiest way to earn return.

#### Reward contract

For unwrapped nut angle `theta`, maintain episode high-water `h`:

- `delta_forward = max(theta_t - h_(t-1), 0)`;
- `h_t = max(h_(t-1), theta_t)`;
- `delta_reverse = max(theta_(t-1) - theta_t, 0)`;
- `gross_travel += abs(theta_t - theta_(t-1))`;
- `net_travel = theta_t - theta_start`;
- `virtual_advance_mm = net_travel * 3 / (2*pi)`.

The only progress term is
`w_forward * delta_forward * pair_gate`. Every reverse step receives
`-w_reverse * delta_reverse`, with `w_reverse >= w_forward`. The existing
constant proximity reward is disabled. If approach help is needed, use only a
bounded potential difference such as improvement in pair distance/contact
confidence; holding the same contact cannot accumulate it.

Reserve fixed phase-observation slots in the policy input from this stage
onward and set them to `drive` for E002-A. This preserves network input shape
when E002-B is introduced.

#### Mandatory diagnostics and selector

- unwrapped start/final/high-water angle;
- forward-high-water, reverse, gross, and net travel;
- net/gross efficiency and reverse/gross fraction;
- final/high-water virtual M24x3 advance in millimetres;
- fraction of environments with positive final net angle;
- gated and ungated progress, pair-drive and single-finger-drive ratios;
- per-finger torque/contact values and action saturation.

Do not select `best_reward_*.pth` by filename. Evaluate fixed candidate
checkpoints under the same fixed-step protocol and select only a checkpoint
that passes all E002-A acceptance gates.

#### Bounded execution gate

1. Unit-test forward, reverse, oscillation, and new-high-water trajectories.
   A `+a, -a, +a, -a` trace must earn no repeated high-water progress and must
   have negative total progress reward after reverse penalty.
2. Unit-test that static double contact earns zero approach/progress reward.
3. Run one-environment GPU smoke and inspect raw unwrapped-angle traces.
4. Run a cloud `timeout 300` probe at the already validated capacity, then a
   fixed-step numeric evaluation and headed playback.
5. Reject without extension if net/gross `<0.80`, reverse/gross `>0.10`, fewer
   than `95%` of environments end with positive net angle, pair-drive `<0.90`,
   or visible oscillation remains.
6. Only after passing may E002-A be extended to `timeout 1800`.

#### 2026-09-04 implementation and bounded-cloud result

**Hypothesis.** New high-water-only angle progress plus a global reverse cost
would make monotonic, two-finger pair drive the only positive-reward route from
the saved nominal grasp.

**Configuration.** Local commits `3e9d014d74402863866dd5d580e95bfee92bbcbf`
and `33a23589d4a17525995c7894f4fec698022e4fab`; cloud commits
`3945540b351f6605a885ddc8f7787ebf994c9358` and
`5ef588ea667f9bf328fd759ef593fc8e28704441`.  The opt-in E002 config keeps the
fixed M24 revolute asset, saved pose, index/thumb action mask, 20 Hz policy and
200 Hz physics, disables E001/legacy shaping and randomization, and changes the
actor input only for E002 to 101 values: 96 joint/target-history values plus
the fixed `drive` command/elapsed tail.  The reward pays only new capped
high-water progress behind the weaker index/thumb contact-and-positive-torque
gate; reverse increments are globally penalized.  Drive loss of valid pair
contact after an eight-step grace period, deadline, workspace escape, and
collision-like steps terminate explicitly.

**Preflight and invalid startup.**

- `python3 scripts/test_m24_e002_a0.py` passed the deterministic
  high-water/reset/oscillation traces; source compilation, config checks, and
  launcher syntax checks passed locally.
- The first pipeline
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1_s42_20260904_150357`
  is excluded from all results: Hydra could not compose an E002 train config,
  so no simulator or GPU result exists.  Adding
  `configs/train/XHandPasiniM24NutBoltE002.yaml` and resolving the config on
  cloud fixed this startup-only issue.

**Valid bounded execution.**

- Pipeline:
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1_s42_20260904_150834`;
  tmux `m24_e002_a1_s42`; start/end
  `2026-09-04T15:08:34+08:00` / `2026-09-04T15:14:35+08:00`.
- A0 status was `passed`.  The 1-environment 60-second smoke and
  12288-environment / 24576-minibatch 300-second capacity probe both exited
  via their expected `timeout` status `124`, without traceback, CUDA OOM,
  PhysX, illegal-memory, or NaN marker.  The probe reached about 39.8k FPS and
  approximately 11.8 GiB VRAM use, so capacity is available but is not an
  acceptance metric.
- The selected-by-launcher probe artifact is
  `..._probe_12288_24576/stage1_nn/best_reward_-2.00.pth`; this filename is
  **not** a policy selection.  Its local SHA-256 is
  `bb5be140f10f0fe9e01fa0fdfeff70cc67bca8e00a30cd7f54f67b5b36a94a37`.
  The smoke counterpart is SHA-256
  `880db2ade085a112b48c3265342d14397b80b84017bd05accb029254e1053436`.
  Full pipeline/logs/configs plus both checkpoints were synchronized locally.

**Behavioral diagnostics.**  The real-simulator A0 gate failed, so no
fixed-step evaluation, headed playback, or `timeout 1800` extension was
authorized.  Over all probe TensorBoard samples, pair-drive ratio and success
were exactly zero; pair gate was at most `3.57e-05`; final index/thumb contact
occupancy was `0.00724` / `0.000163`; index signed torque remained negative
(`-0.00187` to `-6.71e-06`) while thumb torque was near zero and occasionally
negative.  Contact-loss termination reached `0.01489` per sampled update.  In
the independent 1-env smoke, contact-loss termination reached `1` and success
remained zero.  Mean return therefore remained around `-2` (probe final
`-2.0041`), consistent with nearly universal terminal failure rather than
useful exploration.

**Verdict: `invalid`.**  High-water mathematics and resource stability are
implemented, but the physical index/thumb contact source, fingertip mapping,
or force/axis sign required by E002-A0 is unproven and the direct-drive
termination makes the task unlearnable.  This is not evidence against the
monotonic-progress hypothesis itself and is not a rejected teacher checkpoint;
it is a failed instrumentation gate.

**Single recommended next experiment.**  Run a one-environment controlled
positive/negative action trace that records raw fingertip force, distance,
torque vector, axis projection, contact state, and nut-angle response.  Apply
only the resulting demonstrated mapping/sign or reset-grasp fix, then repeat
the deterministic check, 1-env smoke, and one 300-second probe.  Do not start
E002-B/C, a student, or a 30-minute A extension first.

**2026-09-04 controlled A0 trace follow-up.**  Added
`scripts/trace_m24_e002_a0.py` (local commits
`19e1195450ec91bfc7d75fe9f25493451d135d2b` and
`0cdc2decb29f7fe5dd96241603c43aaa5f47deac`) and executed the finite,
one-environment `timeout 180` trace at
`outputs/cloud_pipeline_m24_humantwist_e002_a0trace_s42_20260904_152300`
(exit `0`, 107 samples, local copy preserved).  It explicitly swept zero,
individual index/thumb, same-direction pair, and opposed pair action
increments.  The base saved pose has distance-only proximity (index about
`0.0333 m`, thumb about `0.0432 m`) and exactly zero contact force under zero
action, hence is not a stable double-contact reset.  `pair_plus` created
negative nut motion (`-0.351 rad`) with opposite projected torques (index
`+0.091`, thumb `-0.431`); `thumb_plus` created positive motion (`+0.351 rad`)
without a valid index contribution; opposed candidates lost a finger or ended
by workspace escape.  No candidate made both finger contact hysteresis and
positive torque valid together.  This proves the immediate blocker is missing
stable pair contact at reset, not a reason to flip signs, soften the pair gate,
or extend PPO.

**Revised single recommended next experiment.**  Use the existing Pasini
pose-tuning workflow to record a fixed M24 reset where index and thumb both
remain force-and-distance contact-on for the A0 pre-drive window.  Then rerun
the controlled trace to determine any remaining torque mapping/sign correction
before another smoke/probe.  No current transient checkpoint is a selector.

#### 2026-09-08 candidate-743 A0c diagnostic and one axis fix

Both cloud diagnostic JSONs explicitly failed. Full evidence and protocol:
`docs/m24_e002_a0c_diagnostic_2026-09-08.md`.

- Baseline: `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_diag_s42_20260908_113348`.
- One-fix rerun: `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_axisfix_s42_20260908_113745`.
- Each bank has 67 fixed-pose candidates: zero, 48 signed single-joint and 18
  Jacobian-derived index/thumb tangent proposals, amplitudes .025/.05/.10.
  Each run saved 2144 finite policy-step records and had zero internal resets.
- Stable holds 67/67; pair contact 637/670 probe steps; both torque threshold,
  balance-weighted pair gate and full conjunction pass counts 0 under both
  signs. Rerun best net angle 8.28396e-7 rad, accepted streak 0/5.
- Sole behavioral correction: E002 `rotation_axis=-z`, matching the actual
  URDF nut joint's positive DOF axis. No reward scale, force sign, gate,
  pose, action amplitude/duration or physical friction parameter was tuned.
- No calibrated sign/action, checkpoint, smoke, PPO or headed video exists.
  All training is cloud-only under the user's new persistent convention.

**Verdict:** diagnostics and one mapping repair completed; A0c remains failed.
**Single recommended next experiment:** cloud one-environment nut-specific
contact-point/normal/wrench diagnostics to verify the link-origin torque proxy
before another bounded fix. No further speculative action/sign/weight change,
smoke/PPO, 30-minute extension or E002-B/C is allowed from these JSONs.

#### 2026-09-08 candidate-743 contact-point audit

**Result:** the torque source is not yet a validated full contact wrench.
Pipeline `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_contactbank_20260908_140118`
completed twelve bounded one-environment cloud runs: zero, three single-joint
actions and two coupled tangent actions, each with GPU tensor reference and
CPU tensor contact reading while retaining GPU PhysX. There are 3840 physics
samples, 1920 with 23088 raw contact records; all twelve exit statuses are 0.

The index link-origin proxy sign differs from contact-point normal torque on
600/600 probe samples. Normal force reconstruction agrees with the reported
tensor within 3.48e-6 N. Actual hand/object shape friction is 1.0; zero raw
friction fields do not establish absent physical friction. Both signs still
have zero simultaneous normal-torque/balance passes. Full details, correlated
candidate rows and paired-path differences are in
`docs/m24_e002_contact_audit_2026-09-08.md`.

No task behavior, weight, threshold, pose or physics setting changed. No new
A0c or PPO was launched; the prior axis-fixed A0c remains failed. Audit JSON
is explicitly `a0c_status=not_evaluated`, `training_eligible=false`.

**Single recommended next experiment:** validate a per-finger total
contact-wrench source in one cloud environment, including tangential forces
and separation from joint reaction, before one source correction and A0c
rerun. Do not replace total torque with normal-only torque or guessed offsets.

### E002-B — Strict double release and off-contact return

Start from an accepted E002-A checkpoint with the same observation dimension.

- Enter unload only after a configurable amount of new-high-water drive
  progress. A drive timeout terminates the episode as failure.
- Unload succeeds only after index and thumb are both off for at least two
  consecutive policy steps. Timeout terminates as failure.
- During return, either finger contacting the nut is a failure; require an
  off-contact dwell and return-pose criterion before regrasp.
- Record `valid_release_count`, `drive_timeout_count`,
  `unload_timeout_count`, `contact_during_return_count`, and
  `invalid_regrasp_count` separately.
- Compute release lag only over valid release events and publish the valid
  sample count beside every mean/percentile.
- Apply the same 5-minute numeric/visual gate before a longer run.

### E002-C — Repeated complete cycles

Start only from an accepted E002-B checkpoint.

- Award a cycle bonus only for the ordered sequence: new monotonic progress,
  valid double release, off-contact return, and valid double regrasp.
- Preserve the E002-A high-water and global reverse penalties through every
  phase, so cyclic motion cannot erase net advance.
- Acceptance requires at least three visibly valid cycles, no timeout promoted
  to success, no persistent contact shortcut, positive virtual advance, and no
  reverse oscillation.
- Only an accepted E002-C checkpoint may proceed to load/force optimization or
  student training.

## 10. New experiment template

Copy this section for every new run.

### EXXX — Short title

#### Hypothesis

- Primary hypothesis:
- Why the previous experiment was insufficient:

#### Configuration

- branch / commit / dirty state:
- task and reward version:
- changed reward terms and exact scales:
- unchanged reward terms and exact scales:
- seed(s):
- environment count / minibatch:
- starting checkpoint:
- duration or step limit:
- exact command:
- output/log/config/video paths:

#### Simulation results

- reward and nut progress:
- pair-drive / single-finger-drive ratios:
- index/thumb torque contribution:
- contact overlap / release lag / return drag:
- force distribution / impulse:
- fixed-base wrench:
- smoothness:
- resets, numerical health, FPS, and resource use:

#### Real results

- policy hash / init pose / action mask:
- actual-current distribution by motor:
- tracking error and safety interventions:
- completed cycles / failures:
- manual video verdict:

#### Interpretation

- Did the primary hypothesis pass?
- Was a reward exploit observed?
- Were results comparable with the baseline?
- Safety concerns:

#### Verdict

`accept | reject | inconclusive | invalid`

#### Single recommended next experiment

-
