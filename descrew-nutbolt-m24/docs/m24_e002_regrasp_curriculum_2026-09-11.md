# E002 productive regrasp continuation (2026-09-11)

## 2026-09-11 -- Fixed-pose learned repetition achieved and visually verified

- **Current result:** selected BEST completes5continuous drive/release/off-contact-return/regrasp cycles in128/128originalcandidate743episodes. Net rotation148.542–150.743degrees,median149.747; minimum measured contact-supported drive stroke20.527degrees. Every return trajectory stays off-contact throughout. One PPO,normal inputs,no demonstrated resets/action overrides/policy composition in acceptance.
- Separate512env two-cycle evaluation:501/512complete (97.85%),also with both returns off throughout. Remaining11failures:8first-drive timeouts,3regrasp timeouts. The128and512runs have different reset batches; do not claim universal100%robustness. LAST comparison:464/512two cycles,109/128five cycles. Select BEST by repeated-cycle behavior.
- Selected checkpoint`outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911/selected_teacher/learned_repeated_twist.pth`,SHA256`7742e40ddf1d55a48167f011307c77c5ca943306a7754298e433f7bdbc586c7c`. Source training file`...closure1536_ppo/stage1_nn/best_reward_54.97.pth`. Previous learned-return/release models preserved. Only one new300scloud PPO with separate60s smoke (both124),1536/3072,LR0.0001; **reward weights and phase gates unchanged**. No local training,30minute run,student or deployment.
- Native cloud camera verified selected BEST in a197frame/20Hzsingle episode (9.85s),20ordered phase events,five cycles,149.331degrees net. Inspected all20event keyframes and angle/force/sector plot. Actual video`outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911/best_five_cycle_video/learned_five_cycles.mp4`; GIFalsoavailable. No episode concatenation; GIF looping is only playback.
- Execution receipts: initial post-JSON native shutdown segfaults and one missing-conda retry are preserved. With conda activation and task-local NVIDIA EGL Vulkan ICD, all4BEST/LAST original/curriculum retries and bothfive-cycle evaluations plusnative recordings exit0/finite. Clean receipt`...closure1536/retry_completion_v3.json`,longer receiptroot`longer_evaluation_status.json`. Originalfailedworkflow remainshistorical; authoritative acceptanceisroot`repetition_acceptance.json`,status`passed:fixed_pose_learned_repeated_cycle`,full_cycle_goal_completed=truewithinthisfixed-pose scope.
- Local/cloud source,checkpoint,seed/eval/recording artifacts are preserved withSHA256. Git remains`f91ec27`plus uncommitted return/regrasp continuation; no new commit/push this session. Cloud work complete; instance remainsON,idle after final verification. LocalGPUvisualization isblockedbykernel580.173.02/userspace580.178.04mismatch; no workstation driver changes/reboot. CloudnativeEGLrecording is the validated route.
- **Single recommended next step:** preserve this learned-five-cycle model and expand initial-state/seed robustness around the measured11/512failures. Do not restart solved drive/release/return curricula. More than5cycles,full160mmtravel,randompose/domainrobustness and realhand transfer remain unverified.


Target: preserve the learned drive/release/return baseline and learn regrasp that permits another effective drive from original candidate743. Cloud instance powered on by explicit user authorization; previous selected checkpoint SHA256 `7bba93e1ada571995b5db6f452530d625b264b6771c6c62e90c6496b0799b4d3` verified. No new reward weight or physical gate changes.

## Evidence and hypothesis

Baseline original-start512 evaluation had512drive/release/return,512pair recontacts but0second drives. Pair recontact median sector error26.766degrees undoes the return. Current regrasp phase rewards contact quality and permits pair-contact transitions, without considering available next-stroke travel. This is a hypothesis about the learning signal, not evidence that a stricter transition gate is needed.

New controlled cloud diagnostic starts from actual learned return frame30 (26.407degrees net,1.729degrees sector error,zero forces) and compares66candidates: unchanged PPO,zero,48single-joint actions at0.05/0.1/0.2,12radial closure/sector-anchor actions at0.05/0.1/0.2/0.4,and4nominal-target closures. On pair recontact, the same unchanged PPO supplies all subsequent drive actions.8candidates achieve the next20degree drive,9recontact; bestcandidate60 (`close_0.20_anchor3.0`) recontacts at0.252degree sector error and reaches46.980degrees total. This is controlled-action feasibility, NOT learned regrasp or full-goal acceptance. No controlled actions will enter PPO rollouts.

The restored-state policy baseline recontacts at6.046degrees and still misses the next drive (final37.092degrees). Its trajectory differs from the uninterrupted original-start baseline, so only original-start evaluations may establish learned-chain success. Zero action fails regrasp with26.407degrees unchanged. Complete simulation state restoration does not serialize PhysX solver caches.

[DemoStart III-A](https://arxiv.org/html/2409.06613v1) motivates intermediate-state initialization and testing on original starts. Local adaptation: use full measured state seeds along successful controlled regrasp trajectories, select a measured learning boundary, mix50%original743resets, and keep PPO/reward coefficients unchanged. This is not the paper's complete automatic zero-variance-filter curriculum. [Tac2Motion II-B](https://arxiv.org/html/2509.17812v1) motivates contact quality,release and contact-weighted rotation; it does not justify importing its virtual torque or forcing an unvalidated normal-moment gate here.

Testable hypothesis: direct practice from regrasp states with remaining next-drive travel lets PPO learn the missing closure while downstream drive reward supplies task relevance. Evaluate next actual contact-supported20degree stroke, not mere recontact counts. If this fails, compare a smooth sector-aware regrasp potential using these measured states; do not arbitrarily tune weights or add a mature-success-rate veto.

## Changes and verification

- Opt-in phase3 full-state reset, preserving nominal initial-hand context and global paid/high-water; imported event counts cleared. Release/return entrypoints preserved.
- New Regrasp task/train configs, controlled diagnostic, bounded probe option `--regrasp-preflight`.
- Actual-helper CPU tests pass for release/return/regrasp: subset/world-offset isolation, complete actor reset indices, history/context, and no imported events/progress credit. Python syntax and diff whitespace checks pass.
- Pipeline root `outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911`; exact bounded commands/logs and source/checkpoint/seed hashes preserved. Fresh source A0c and128env boundary/zero physical preflights precede any training.

Single next action: identify mixed-success regrasp states, then one60s smoke/300s cloud PPO continuation and original512 evaluation. Repeated screwing remains incomplete.

## Boundary preflight and first bounded continuation

FreshA0c explicitly passed. Policy128env preflight:seed0=19/19next drives,seed26=16/16;6/12/18/21/24/actualreturned30allfail next drive. All recontact; zero fromactual30=0/128recontacts and0drives,canonicalcontext error0. All exits0/finite.

Sector angle is not sufficient by itself: seed0 succeeds after8.61degree recontact error, while seed24 fails after3.49degrees. Joint pose, targets, contact and history matter. Do not add a5degree regrasp veto based on this proxy. Use real next-drive outcome and the full measured state distribution.

First run `_closure1536` uses all8measured seeds0/6/12/18/21/24/26/30,50%original743,LR0.0001,1536/3072,starting the selected return model. Smoke60s exits124; PPO300s starts2026-09-11T10:01:45+08:00. Initial training~4500FPS,~3.2GBVRAM,CPUcontact collection bottleneck. Await original512 behavior; mean reward alone cannot establish the goal.

## 2026-09-11 -- Learned two-cycle breakthrough; longer inference verification

- One60s smoke and300s cloud PPO completed (both124),unchanged reward weights/gates,1536/3072,LR0.0001,regrasp curriculum50%original743. LAST original512:512second effective drives,464two complete cycles; BEST original512:501two complete cycles,8first-drive timeouts,3regrasp timeouts. BEST SHA256 `7742e40ddf1d55a48167f011307c77c5ca943306a7754298e433f7bdbc586c7c`; LAST `3d3efe9a8d054f613f34d90cb215c4142d3746fa09b94dac921b885dd02d0cd3`. Prior learned-return baseline preserved.
- Original numeric evaluations initially wrote finite JSON then native shutdown segfault(-11). First retry also missed conda activation (libpython missing),preserved as failed. Activateconda and explicitly select NVIDIA EGL Vulkan ICD per official headless-library guidance; retry3 original/curriculum evaluations now exit0. All failed attempts remain in logs/retry records; do not silently label the original failed pipeline clean.
- Local visualization blocked by hostkernel580.173.02/userspace580.178.04 mismatch and inactive persistenced; no workstation restart or driver changes. Cloud native-camera LAST recording exits0 with the explicit EGLICD. Actual68frame/20Hzuninterrupted episode events1/2/3/4/1/2/3/4,final60.922degrees. Both returns physically off-contact. Keyframes0/13/24/28/34/42/54/61/67visually inspected. GIFloop is playback,not evidence of extra cycles.
- Active next step is inference only:128env five-cycle comparison for BEST/LAST and native-cloud BEST five-cycle recording;tmux`m24_regrasp_longer_eval`,driver root`evaluate_longer.py`,each180s. No additional PPO. Basic learned repetition now demonstrated; maximum sustainable repetition/remaining failure distribution still being measured.


### Cloud headless rendering environment

The existing`/etc/vulkan/icd.d/nvidia_icd.json`points toGLX. This session used a task-localICDpointing to the already-installed`/usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0`and`VK_ICD_FILENAMES`; no system driver/package installation. [NVIDIA installed-components documentation](https://download.nvidia.com/XFree86/Linux-x86_64/460.91.03/README/installedcomponents.html) recommends the EGLlibrary for Vulkan environments without X11clientlibraries. [Khronos Vulkan loader documentation](https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderDriverInterface.md) documents manifest overrides. Exact commands and manifest are archived under the pipeline root. Future cloud inference/recording should sourceconda and reuse the explicitEGLICD. The lower-level native shutdown fault was not independently debugged; clean retries establish the accepted results, not a proven C++rootcause.
