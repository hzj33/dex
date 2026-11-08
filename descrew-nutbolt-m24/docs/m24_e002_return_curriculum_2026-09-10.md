# E002 learned return continuation

## 2026-09-10 -- Learned original-start return verified; productive regrasp remains

- **Return achieved:** selected LAST from`_currentpath1536` completes drive/release/return512/512from original candidate743 with one PPO,normal inputs,no demo resets or action/policy-input overrides. All512return trajectories stay off-contact throughout,retain at least25.7498degrees (median26.1594),and end within2.4081degrees of the initial radial sectors. All four final-run evaluations exit0/finite. This is a fixed-pose result,not broad pose robustness.
- The policy then makes pair-contact events512/512,but recontact drifts back to26.7664degrees median sector error. Every second drive times out (0second-drive milestones,0two-cycle completions). **Do not equate the permissive pair-contact/regrasp event with productive regrasp or a completed repeated cycle.** This is the next concrete learning problem; return itself is now verified.
- Selected checkpoint`outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/selected_teacher/learned_drive_release_return.pth`,SHA256`7bba93e1ada571995b5db6f452530d625b264b6771c6c62e90c6496b0799b4d3`. Originalfile`...currentpath1536_ppo/stage1_nn/last.pth`. Baseline drive/release model preserved. Git milestone before this continuation:`f91ec27`; subsequent return work is not included in that commit.
- Four completed cloud PPO comparisons,each separately bounded300s with own60s smoke:near1536,progress1536,earlier1536,currentpath1536. All completed phases training124/eval0. One additional farther1536was intentionally stopped after its60s smoke and2PPOupdates when preflight showed its proposed states already solved; excluded from completed comparisons and preserved with stop evidence. No local training,long-run launch or student. Final curriculum current-path9/12/15,fraction0.5,1536/3072,LR0.0001. Reward weights/physical thresholds unchanged; freshA0c source remains matching.
- Actual local inference playback verified at20Hz with the SHA-matched model.41frame single episode:driveevent13,release23,return30(retaintheta26.407degrees,error1.729degrees,force0/0),recontact40(error27.002degrees,force3.326/2.055N). Inspected frames0/13/23/30/40and angle/force/error plot. Video`outputs/local_pipeline_m24_e002_returncurriculum_20260910/learned_first_cycle_video/learned_return_then_recontact.gif` is one episode; GIF looping is not learned repetition.31frame return-only clip is also saved. Full tensor states at return/recontact are available in these traces for follow-up diagnostics.
- Authoritative acceptance:`outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/return_acceptance.json`,return_subgoal_completed=true,full_cycle_goal_completed=false,repeatable_regrasp_completed=false. Report:`docs/m24_e002_return_curriculum_2026-09-10.md`. Cloud training/eval and local recording are complete; server remains ON. Artifacts/checkpoints/source are archived locally and on cloud; final_execution_state.json records the final process state.
- **Single recommended next step:** use the verified returned state to diagnose/reward a regrasp that stays near the returned start sector and leaves travel for the next stroke; then validate a second effective drive. Do not add a mature-success-rate veto or accept contact alone as repeatability. Preserve the learned return baseline.



The baseline single PPO learns drive and release from original candidate743 (488/512 releases) but never returns. Cloud single-environment state capture now measures a35.575degree maximum finger-sector error at actual release, with27.742degrees net rotation. Continuing that policy recontacts the nut and ends in a return timeout. The baseline checkpoint remains unchanged.

DemoStart Section III-A (https://arxiv.org/html/2409.06613v1) saves complete intermediate simulation states to vary start-state difficulty and evaluates on the original target distribution. Our local adaptation uses a fixed measured released-state archive with the existing PPO and rewards, rather than implementing its automatic zero-variance filtering or copying demonstration actions into PPO experience.

Testable hypothesis: supplying physically reconstructed released states gives PPO direct return-phase practice, while mixing original743 starts helps retain drive and release. Success must occur under learned policy actions from original743 without demonstration resets or policy-input overrides. Physical action-bank results are feasibility evidence only. Return means two fingers off the nut for the established dwell and within the existing5degree sector tolerance, with retained net progress; a return-only termination is code7, never full-cycle5.

Implementation adds opt-in phase2 demo resets, clears imported events, preserves nominal episode hand reference, and records actor/DOF/target/history/cycle state before observation roll. PhysX solver caches remain unserialized, so physical preflights are required. Existing reward weights/force thresholds unchanged. Exact commands and results live in `outputs/cloud_pipeline_m24_e002_returncurriculum_20260910`.

Git milestone before continuation: `f91ec27`. Return changes are subsequent work.

## Physical feasibility and initial baselines

Cloud65candidate action diagnostic:2true returns. Candidate62 (`sector_0.40_out0.2`) completes in28control steps (1.40s), at4.523degrees sector error,27.742degrees net nut rotation; both contacts remain off and force0/0N throughout. Candidate58 at0.20amplitude also returns. Returning directly to nominal joints reaches2.16degrees but still presses the nut, so it correctly does not count. This distinguishes actual free return from a pose-only shortcut.

Original learned-policy/zero baselines from a50%mixture of real released step34 and original743 are finite but both0returns. Policy retains56/56drive and51/56release for original starts;72/72demonstrated starts fail return. Zero-action states do not earn a free return. Canonical initial-hand-reference error0. Saved phase normalization variances are approximately0.42–0.66, so no normalization reset is needed.

First training curriculum uses physically recorded diagnostic trajectory steps15,18,21 (errors15.73,11.11,7.47degrees), mixed50%with original starts. These are controlled-trajectory initial states, not learned return demonstrations. Their exact archive is SHA256pinned, all are off-contact phase2/event0, and reset clears prior earned events. Policy/zero physical preflights precede the bounded run.

## 2026-09-10 -- Learned return endpoint; earlier-state continuation active

- First return run `_near1536` completed: smoke/PPO124; all4best/last curriculum/original evaluations exit0. LAST learns return89/89fromseed21; seeds15(67envs)/18(108envs)0returns. Bothbest/last retain original512/512drive+release but0return. The endpoint succeeds with0/0Ncontact force; this is learned partial return, not the original-start chain.
- LAST SHA256`b18400d6f30072c94e2613f516c5b1511e9048637ed141abf061631938630a44`. Root source/commands/checkpoints preserved. First-run failed seeds approach9.54/6.22degree minimum error then recontact; seed21stays off and reaches4.59degrees. No reward/threshold change is indicated by these data.
- Continue progressively earlier states18/19/20/21,50%originalstarts,from this measured LAST; archiveSHA`25c94bec57223b4625a6fc9b3b34a8c7b073a7990882fb4a0e54999bfbda8c29`. Active tmux `m24_return_progress`; driver root`learn_progress_return.sh`; output`_progress1536`. Physical policy preflight then independent60s smoke/300s PPO,1536envs/3072minibatch,LR0.0001. Existing passedA0c remains source-identical. No local/long/student training.
- Single next action: assess this continuation's curriculum and original743 return traces, including recontact during return. Full return/regrasp/repetition remain incomplete.


## 2026-09-10 -- Learned 11degree return segment; third bounded continuation

- Second `_progress1536` run complete: smoke/PPO124 and all4evaluations0. Bothbest/last264/264demonstrated starts18/19/20/21return; original512/512drive+release,0return. All264return trajectories stay off contact throughout,retain27.742degrees,finish at<=2.575degrees sector error. This is autonomous PPO from intermediate states, not a full original-start return.
- LAST SHA256`1b8147979949fb886683f9982022b7c00a94058e6659abe7134d3ba0a4bb9ed4`,downloaded and verified. Diagnostics: `progress_last_return_analysis.json`. Baseline and all prior models preserved.
- Third continuation moves to controlled-trajectory states12/15/18 (~20/15.7/11.1degreeerrors),50%originalstarts,archiveSHA`67cc1a816431aa7a6f460ffdc11f4d552269892bc95f94b90a5fefa96e077d0a`. Active tmux `m24_return_earlier`,driver root`learn_earlier_return.sh`,output`_earlier1536`; preflight then separate60s smoke/300s PPO at1536/3072,LR0.0001. Same verifiedA0c/source,weights and thresholds unchanged.
- Single next action: finish this run and inspect return from earlier/original states; do not confuse successful short return segments with the full drive-release-return chain or repeated cycles.


## 2026-09-10 -- Learned 20degree return segment; fourth bounded continuation

- Third `_earlier1536` complete: smoke/PPO124,all4evals0. Best/last264/264demonstrated starts12/15/18return; original512/512drive+release,0return. LAST SHA256`7b4e95ef4af911a3d95cf8a3ff7019ba4d9627babf6cb4bc9d36578faf61912a`,downloaded/hashverified. All264return trajectories stay off throughout. Return competence now covers the20degree-error intermediate state,not yet actual release at~35degrees.
- Fourth continuation uses earlier controlled states6/9/12 (~27.8/24/20degreeerrors),50%originalstarts; archiveSHA`d39c689cde6e14b3e0c1d1353e3acb91cf17dde3745f86e4cd2ab6057df23de3`. Active tmux`m24_return_farther`,driver root`learn_farther_return.sh`,output`_farther1536`. Each phase is independently bounded: physical preflight180s,smoke60s,PPO300s,then best/last curriculum/original evaluations. Same1536/3072,LR0.0001,verifiedA0c,unchanged reward weights and thresholds.
- Single next action: evaluate this earlier return segment and original start; keep advancing only from measured learning progress. Full original drive-release-return chain and regrasp/repetition remain incomplete.


## 2026-09-10 -- Discarded already-solved curriculum; diagnose actual release distribution

- Fourth `_farther1536` preflight showed the third model already returns72/72from6/9/12,so these proposed states do not test the remaining full-start gap. Stopped that owned pipeline after its60s smoke and2completed PPO updates; this interrupted run is NOT an accepted5minute comparison. `workflow_status.json=stopped:curriculum_already_mastered`,exact prior phase/pid evidence in`intentional_stop.json`; interrupted checkpoints preserved but not selected. No residual GPU task after stop. Continue from third-run LAST, not the interrupted model.
- Original third-model trace starts return at32.17degrees,improves to24.42then recontacts/fails; old demonstrated states at27.8degrees already return. Angle alone therefore does not characterize the remaining distribution gap. Need actual current released pose/targets/history,not another solved old-state run.
- Captured current model's actual original743release: step26,33.354degree sector error,27.888degree nut angle,force0/0. Archive`current_return_seed_candidates.json`,SHA256`9f52ee811aa427cf7ca4dc797773c9ab35910cbe904783780278982d13758270`; complete source trace saved,normal single-PPO inputs, no controlled actions.
- Active tmux`m24_return_current_diagnostic`,driver root`diagnose_current_context.sh`: compare old states0/3/6/34,policy/zero from current actual release,and65controlled current-state return candidates. Eachphase180s,diagnostic only; no PPO currently running.
- Single next action: identify a learnable bridge from the actual current release distribution and validate its state reset,then bounded PPO with original-start evaluation. Reward weights/thresholds unchanged; full return/regrasp/repetition incomplete.


The current-state65candidate diagnostic yields3endpoint returns,but two recontact during the motion. Onlycandidate62(`sector_0.40_out0.2`) stays force0/0and off throughout. The next bridge archive uses its actual steps0/3/6/9 plus the current policy's real released step26. Old-state preflight0/3/34fail while6succeeds; current actual release policy/zero both0/128returns. These distinguish unfinished early control from merely using a5degree endpoint threshold; no threshold change is needed.

## 2026-09-10 -- Current-release trajectory boundary identified; bounded adaptation active

- Current-release controlled return has3endpoint-success candidates, but only62stays off throughout; use62for curriculum,not61/57which recontact. Old starts0/3/34fail,6passes; actual current released26 policy/zero both0/128returns. Current-path early0/3/6/9andactual26allfail under PPO,despite old-path27.8degree state succeeding.
- Boundary preflight128 on current path9/12/15/18/21/24/27/30:9/12fail,15andlater allreturn. Canonicalreferenceerror0,finite. This supplies a measured successful/failed starting-state boundary,not an endpoint-threshold problem. `current_boundary_seed_candidates.json` SHA256`064ce5287ad23824b094cd2d136809b101e0fd27beb005ae0f97102dd212d4d3`.
- Active cloud tmux`m24_return_currentpath`,driver root`learn_currentpath_return.sh`,output`_currentpath1536`. Start third-run LAST; curriculum current-path9/12/15 plus50%original743,physical preflight then separate60s smoke/300s PPO,1536/3072,LR0.0001. Same validatedA0c/source,reward weights and thresholds. Earlier proposedbridge[0,3,6,9,26]was only evaluated (allfail),not trained; earlier all-solved farther run remains intentionally interrupted and excluded.
- Single next action: finish this current-state adaptation and evaluate the original chain; if needed extend towards current actual released26using the measured bridge. Full original return/regrasp/repetition remain incomplete.

