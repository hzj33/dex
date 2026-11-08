# E002 autonomous release curriculum, 2026-09-10


## 2026-09-10 -- Release subgoal achieved from original743 with one PPO

- **Current result:** selected LAST from `_context1536` completes learned drive512/512 and actual two-finger release488/512 (95.3125%) from original candidate743, with normal task inputs and no demonstration resets, scripted actions or policy composition. All accepted releases follow a measured drive event, retain at least27.489degrees (median27.822), and have both contacts off/forces<0.15N for two endpoints; final force0/0N. Initial-hand-reference error0. No workspace/jump termination.24release timeouts remain. Return/regrasp/repetition is NOT learned.
- Checkpoint `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/selected_teacher/learned_drive_release.pth`,SHA256 `aa0e55b06ca7fd864b2ccf717bd73bd95c173e9eface25be93c3328450b9f471`; original training file is `_context1536_ppo/stage1_nn/last.pth`. Reward-best18.50only20/512releases; LAST is deliberately selected by behavior. Both models and source are preserved,downloaded,hash-verified.
- Four separately bounded cloud PPO runs this session (each300s plus bounded completion grace),with separate60s smokes: `_near6144`14updates,`_value1536`52,`_early1536`61,`_context1536`57. Last run1536envs/3072minibatch,seeds18/28/36,demofraction0.5,initialLR0.0001. Reward weights/actual release criteria unchanged. No local training,30minute run or student. All final-run phases exited expected124(training)/0(evaluation).
- Key corrections: reset the articulated nut's position/velocity; restore saved critic value-normalization statistics; preserve canonical episode initial-hand-pose context during intermediate-state resets. The last mismatch was introduced by this session's new demo helper and is now fixed. CPU actual-method/helper tests and physical cloud preflights pass. Earlier context-override512/512result remains diagnostic-only and is not the selected solution.
- Actual **single-episode** playback verified at20Hz: `outputs/local_pipeline_m24_e002_releasecurriculum_20260910/learned_release_video/learned_drive_then_release.gif`,35frames,net27.749degrees,finalforces0/0N. Inspected initial,drive-complete and released frames. Thumb opens before index by0.60s; this is not perfectly simultaneous release or a repeated cycle. Cloud graphics initialization crashed; the identical SHA-verified model was recorded locally with test=True/inference only. Video/trace/plot are mirrored to cloud.
- Authoritative acceptance: root pipeline `release_acceptance.json` (`release_subgoal_completed=true`, `full_cycle_goal_completed=false`). Final source: `final_source/SHA256SUMS`. Report: `docs/m24_e002_release_curriculum_2026-09-10.md`. Source,commands,evaluations and best/last checkpoints preserved locally and on cloud. The second run's best-eval shutdown crash has clean separate retries in `retry_completion.json`.
- Current execution: cloud training/evaluation complete; local visualization complete. Server remains ON. Final process/GPU state is recorded in `final_execution_state.json`.
- **Single recommended next step:** use the selected single-PPO checkpoint to collect actual released-state seeds, then learn return while keeping both fingers off and preserving net nut rotation; evaluate from original743 and retain this drive→release checkpoint as the baseline. Do not treat release as completion of return,regrasp or repeated cycles.


Current subgoal: learn autonomous release, preserving actual net nut rotation.
Full release/return/regrasp/repetition remains the larger objective. Cloud capacity
returned: AutoDL power-on succeeded, request8e99b32b5905767ea80b8847264534df;
RTX4090/49140MiB initially idle. This session trains on cloud only. The current
cloud handoff was read before code transfer; cloud originals are backed up under
`outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/source_before`.

## Evidence and implementation

[DemoStart, III-A](https://arxiv.org/html/2409.06613v1) uses intermediate
simulation demonstration states to change the RL initialization distribution,
then evaluates from the target initial distribution. Our selected-state release
curriculum is a local adaptation within the existing PPO teacher. It is not the
paper's full auto-curriculum or zero-variance filtering implementation. No
scripted action enters learned policy rollouts or their PPO action buffer.

Previous full-start5minute comparisons learned no release. Scripted outward
motions release both fingers while retaining23–24degrees. Five recorded states
can be restored and continued physically; near-release states40/43have maximum
angle errors0.00649/0deg and force errors0.00990/0.02688N. Hypothesis: these initial
states expose successful release outcomes to PPO; gradually earlier starts can
teach the missing unload action. Policy and zero-action baselines distinguish
learning from passive release due to saved velocity/PDtargets.

New `e002_demo_resets.py` caches recorded DOF positions/velocities, PDtargets,
actor root states, history and cycle state. It offsets roots for each environment,
uses both articulated actors in the DOF setter, preserves angle high-water and
potential, clears imported event counters, and starts a fresh release clock.
The existing101Dactor and action interface remain. `XHandPasiniM24NutBoltE002Release`
is opt-in. Actual release with retained net angle terminates the release subtask
with code6, distinct from full-cycle success5. Reward scales, physical contact
thresholds and the normal-moment diagnostic convention are unchanged.

## Confirmed pre-existing nut reset defect

`reset_idx` only wrote hand positions/velocities and submitted hand actors to
`set_dof_state_tensor_indexed`. The M24 nut belongs to the object actor; its
position and velocity therefore survived episode resets. Physical4env test:
reset env0kept0.4rad/0.5rad/s, env2kept0.3rad/0.3rad/s. Unreset environments were
unchanged. `reset_before.json` explicitly fails (`reset_max_abs=0.5`).

The E002-only fix zeros the nut DOF and includes hand plus object actor indices
when submitting the reset. The same cloud test now passes: reset error0 and
untouched change0. This defect confounds earlier repeated-episode training and
may explain much of its late angle-jump behavior. It does not invalidate the
recorded first-episode deterministic evaluations. Do not attribute earlier
training degradation solely to reward/learning-rate choices or claim this one
reset test proves a trained behavior.

## Checks and execution

- Actual demo-helper CPU test passes: subset isolation, environment offsets,
  complete DOF actor indices, history preservation and no imported event credit.
- Actual cloud pre/post nut reset checks confirm the correction.
- Initial demo baseline crashed at the native API: an inline int32conversion
  created a temporary tensor passed to `gymtorch.unwrap_tensor`. A named tensor
  now keeps that actor-index buffer alive through the setter. Failedlog preserved;
  physical baseline retry completed successfully for both action modes. Each had0/64releases and64release timeouts, retaining23.796degrees; initialization alone does not release.
- Baselines precede fresh A0c, smoke60s, PPO300s and independent demo-start / original743
  best-and-last evaluations. Each training phase has its own timeout;124is expected.
- Root cloud pipeline: `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910`.
  Local evidence directory: `outputs/local_pipeline_m24_e002_releasecurriculum_20260910`.

The release subgoal is now achieved at the fixed original pose as reported above; full return/regrasp/repetition remains incomplete.

## 2026-09-10 -- Near-release first run complete; corrected value normalization continuation

- First cloud near-state run completed: fresh passed A0c,smoke60s/PPO300s both124,6144envs/12288minibatch. Best/last512-env evaluations each0release from demo seeds40/43; original743 each512drive/0release,median23.707/22.943degrees. Pressure traces show continued squeezing, not unloading (last thumb32–38N). All evaluations finite; no deterministic jump/workspace failure. Original drive remains preserved; release is NOT learned.
- New bounded fix: `PPO.restore_train` loaded the critic weights but omitted saved `value_mean_std`, so rollout values used different units after resume. Restore that statistic when present; legacy checkpoints without it still load. Actual-method CPU test using the saved checkpoint: max denormalized value error1.455477->0. This proves the restore defect, not its share of the learning failure. Reward weights unchanged.
- Cloud continuation: tmux `m24_release_value`,driver `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/learn_release_value_restore.sh`,fresh A0c under `a0c_value_restore`,results suffix `_value1536`. Start the first run's last.pth,keep seeds40/43,LR0.0001.1536envs/3072minibatch increases completed updates in a5minute budget; this is not a one-factor reward ablation. Each smoke/PPO phase independently bounded60/300seconds.
- Single next step: inspect this continuation's best/last demo/original evaluations and advance the initialization curriculum only from measured release progress. Preserve original stable-drive checkpoint. Cloud is ON; no local training,30minute run or student.


## 2026-09-10 -- Learned near-release512/512; earlier/full-start mixture now training

- Corrected-value continuation `_value1536`: smoke60s/PPO300s exit124. Last checkpoint deterministic512/512 releases from seeds40/43 (256each),all code6,median23.791degrees; evaluator exit0,finite. Original743 last0drive/0release,median17.974degrees,all drive deadlines. This is learned near-release, NOT a full original-start solution.
- Reward-best near-state JSON also contains512releases, but process crashed with SIGSEGV after writing/printing the report, during shutdown. Overall workflow accurately remains `failed:execution`; best result requires a clean retry and best-original eval was not run. Preserve failure evidence. Continued from independently validated LAST, not the runtime-failed best evaluation.
- Earlier-state preflight128:72demo starts (18:30,28:23,36:19) plus56originalstarts. Zero-action0release; policy releases19/19from36,0from18/28 and0originaldrive. Both modes exit0,finite. This supports advancing the curriculum into early unloading while mixing original drive practice.
- Cloud tmux `m24_release_early`,driver root `learn_release_early.sh`; third independently bounded smoke60s/PPO300s,1536envs,3072minibatch,seeds18/28/36,demofraction0.5,LR0.0001. Same reward weights, force thresholds and validated source/A0c. Output suffix `_early1536`. Cloud is on; local machines only development/analysis.
- Single next action: finish mixed training, evaluate actual original-start drive→release, retry the second run's best eval separately, and record learned motion without scripted actions. Full return/regrasp/repetition remains unachieved.


## 2026-09-10 -- Confirmed curriculum context mismatch; corrected short continuation

- Third `_early1536` run completed all phases: smoke/PPO124; best/last demo and original evaluations exit0. Last mixture264/264demonstrated states release (18:67,28:108,36:89); all248originalstarts drive but fail release. Dedicated original512/512drive,0release,median40.183degrees. This is not an original-start solution.
- Two learned-policy inference diagnostic (original stable drive + latest release) also0/512release,retaining23.319degrees. First wrapper attempt failed CPU/CUDA mask device mismatch before evaluation; fixed selector mask device, retained failed log, retry exit0. This is optional inference only, not PPO redesign/student training.
- **Causal one-feature diagnostic:** keep both models,physical scene,phase events and thresholds unchanged; feed only the release expert the demonstrated step18 hand-initial-pose reference. Result512/512drive+release,median23.305degrees,exit0. No scripted joint actions, but the feature override is DIAGNOSTIC ONLY, not accepted as the completed goal. Artifacts `learned_skill_chain_retry/` vs `hand_reference_diagnostic/`.
- Root cause in this session's new demo helper: it overwrote privileged `hand_joint_pos` with the intermediate physical pose. Base reset writes that field only once, before noise, so it represents nominal episode initial pose; normal full-start rollouts keep743 there. Fix removes the overwrite, preserving canonical task context while restoring actual physical DOFs/history. New actual-helper context regression fails before,passes after; cloud test passes. No reward/threshold change.
- Fourth bounded continuation: tmux `m24_release_context`,root `learn_release_context.sh`,fresh A0c `a0c_context`,same1536envs,seeds18/28/36,fraction0.5,LR0.0001,starting early-last. Preflight explicitly reports canonical initial-hand-reference error; smoke60s/PPO300s then best/last evaluations. Output suffix `_context1536`.
- The second run's best evaluation shutdown crash was retried cleanly: best-near512release and best-original0release, both exit0. `retry_completion.json` preserves original failed workflow status plus successful retries.
- Single next action: verify context-consistent learned release from the real original start; record it without any policy-input override. Full release/return/regrasp/repetition is not yet achieved. Cloud ON; no local training or automatic30minute/student run.


The optional two-policy diagnostic follows the general option/closed-loop-skill
composition framework of [Sutton, Precup & Singh (1999)](https://www.sciencedirect.com/science/article/pii/S0004370299000521).
Our measured-stage selector is hand specified; no learned high-level selector,
intra-option algorithm or consolidated single network is claimed.
