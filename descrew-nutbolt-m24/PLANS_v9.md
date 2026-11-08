# PLANS_v9

## Title
- `dexh13_hora` + `lightbulb` + core student algo packaging

## Status
- `completed_smoke_accept`
- started_on: `2026-04-17`
- completed_on: `2026-04-17`
- final_decision: `accept_smoke_matrix_and_close_v9`

## Scope
- 新 hand 家族：
  - `Dexh13HoraLightbulb`
- 新 object 家族：
  - `XHandHoraLightbulb`
  - `Dexh13HoraLightbulb`
- core student algo packaging:
  - `ProprioAdapt`
  - `PureBC`
  - `DiffusionLatentStudent`
  - `ConsistencyLatentStudent`
  - `FlowMatchingLatentStudent`

## Goals
- 保持旧 `XHandHoraScrewDriver` / `XHandPasini*` 路线可用
- 为 `xhand + lightbulb` 与 `dexh13 + lightbulb` 提供独立 task/config/script 入口
- 将 core student family 通过 `dexscrew/algo/student/` 做统一导出
- 在进入长训练前完成：
  - env reset smoke
  - core student ctor smoke
  - 一个旧 baseline regression ctor smoke

## Implemented
- task / asset wiring:
  - `dexscrew/tasks/dexh13_hora.py`
  - `dexscrew/tasks/__init__.py`
  - `configs/task/XHandHoraLightbulb.yaml`
  - `configs/task/Dexh13HoraLightbulb.yaml`
  - `configs/train/XHandHoraLightbulb.yaml`
  - `configs/train/Dexh13HoraLightbulb.yaml`
  - `assets/screw/lightbulb/0000_lightbulb.urdf`
- shared task compatibility layer:
  - `dexscrew/tasks/xhand_hora.py`
  - added optional support for:
    - `env.apply_action_mask`
    - `env.action_mask_indices`
    - `env.asset.fingertipBodies`
    - `env.asset.handInitPose`
    - `env.asset.handRootPos`
    - `env.asset.handRootQuat`
    - `env.asset.handRootRPY`
    - `env.asset.handRootPosNoise`
    - `env.asset.handRootPosZScaleComp`
    - `env.asset.dofLowerLimits`
    - `env.asset.dofUpperLimits`
    - `env.asset.dofEffortLimits`
    - `env.asset.dofVelocityLimits`
    - `env.object.init_pos`
    - `env.object.init_pos_noise`
- student packaging:
  - `dexscrew/algo/student/__init__.py`
  - thin re-export modules for the 5 core student families
  - `train.py` / `student_eval.py` switched to the new import surface
  - `student_eval.py` no longer hardcodes `student_dim=24`
- scripts:
  - teacher:
    - `scripts/xhand_lightbulb_teacher.sh`
    - `scripts/dexh13_lightbulb_teacher.sh`
  - teacher vis:
    - `scripts/vis_xhand_lightbulb_teacher.sh`
    - `scripts/vis_dexh13_lightbulb_teacher.sh`
  - student core:
    - `scripts/xhand_lightbulb_student_padapt.sh`
    - `scripts/xhand_lightbulb_student_purebc.sh`
    - `scripts/xhand_lightbulb_student_diffusion_latent.sh`
    - `scripts/xhand_lightbulb_student_consistency.sh`
    - `scripts/xhand_lightbulb_student_flow_matching.sh`
    - `scripts/dexh13_lightbulb_student_padapt.sh`
    - `scripts/dexh13_lightbulb_student_purebc.sh`
    - `scripts/dexh13_lightbulb_student_diffusion_latent.sh`
    - `scripts/dexh13_lightbulb_student_consistency.sh`
    - `scripts/dexh13_lightbulb_student_flow_matching.sh`

## Validation completed
- source syntax compile:
  - `train.py`
  - `student_eval.py`
  - `dexscrew/tasks/xhand_hora.py`
  - `dexscrew/tasks/dexh13_hora.py`
  - `dexscrew/tasks/__init__.py`
  - `dexscrew/algo/student/*`
- shell syntax:
  - all new `lightbulb` scripts pass `bash -n`
- docker Isaac Gym runtime smoke:
  - `XHandHoraLightbulb` env instantiation + `reset()` passes
  - `Dexh13HoraLightbulb` env instantiation + `reset()` passes
  - `XHandHoraLightbulb` core student ctor smoke passes for all 5 algos
  - `Dexh13HoraLightbulb` core student ctor smoke passes for all 5 algos
  - old-path regression:
    - `XHandHoraScrewDriver + ProprioAdapt` env + ctor + reset still passes
- viewer / teacher smoke progress:
  - `xhand_hora.py` empty-reset assignment bug fixed:
    - `self.nut_dof_vel_cf[at_reset_env_ids]` now guards empty index sets
    - this unblocked `1-env` PPO rollout on both `XHandHoraLightbulb` and `Dexh13HoraLightbulb`
  - `XHandHoraLightbulb` teacher smoke:
    - `3 min / 32 env / minibatch 384` passes
    - reward trend improves from roughly `-655` to `-46.5`
    - accepted as current workable lightbulb teacher-start route
  - `Dexh13HoraLightbulb` teacher smoke:
    - default `3 min / 32 env / minibatch 384` fails
    - reward degrades from roughly `-5.6k` to `-8.5k`
    - two root-position-only probes also fail to stabilize:
      - candidate A `handRootPos=[0.14, 0.102, 0.137]`
      - candidate B `handRootPos=[0.14, 0.092, 0.127]`
- geometry reset probes:
  - `XHandHoraLightbulb` fingertip->nut distance:
    - `min=0.0635`, `mean=0.0859`
  - `Dexh13HoraLightbulb` default:
    - `min=0.1040`, `mean=0.1151`
  - `Dexh13HoraLightbulb` candidate A:
    - `min=0.0786`, `mean=0.0874`
  - `Dexh13HoraLightbulb` candidate B:
    - `min=0.0717`, `mean=0.0795`
  - interpretation:
    - reducing initial distance alone is not sufficient for DexH13;
    - the more aggressive candidate made training materially worse despite better proximity
- XHand student smoke progress:
  - student shell scripts now force `train.ppo.minibatch_size=576` for `task.env.numEnvs=48`
  - `XHandHoraLightbulb + ProprioAdapt` train smoke passes:
    - output: `outputs/XHandHoraLightbulb_student_padapt/v9_xhand_padapt_smoke/stage2_nn/model_best.ckpt`
  - `XHandHoraLightbulb + ProprioAdapt` eval smoke passes through official `train.py test=True` path:
    - `EvalSummary steps=64 avg_reward=-5.448445 avg_done_rate=0.015625`
  - remaining 4 XHand core student smokes also pass through official `train.py test=True` path:
    - `PureBC`: `avg_reward=-5.379851`, `avg_done_rate=0.015625`
    - `DiffusionLatentStudent`: `avg_reward=-5.226565`, `avg_done_rate=0.015625`
      - `EvalReconSummary latent_mse=0.128224 latent_l1=0.306090 action_mse_to_teacher=0.006538`
    - `ConsistencyLatentStudent`: `avg_reward=-5.652328`, `avg_done_rate=0.015625`
      - `EvalReconSummary latent_mse=0.122866 latent_l1=0.294118 action_mse_to_teacher=0.005339`
    - `FlowMatchingLatentStudent`: `avg_reward=-5.256454`, `avg_done_rate=0.015625`
      - `EvalReconSummary latent_mse=0.133663 latent_l1=0.307018 action_mse_to_teacher=0.006894`
- DexH13 smoke recovery:
  - `Dexh13HoraLightbulb` reward shaping defaults updated:
    - `pose_diff_penalty_scale: -0.01`
    - `torque_penalty_scale: -0.5`
    - `work_penalty_scale: -0.001`
    - `rotate_penalty_scale: -0.2`
  - rationale:
    - default DexH13 lightbulb teacher smoke failed mainly because preload contact produced overly large controller penalties (`work_done`, `torques`), not because the object/task wiring itself was broken
  - recovered teacher smoke:
    - `2.2 min / 32 env / minibatch 384` stays in the `-705 ~ -769` band
    - accepted as the current DexH13 lightbulb smoke teacher start point
- common student-model fix:
  - `dexscrew/algo/models/models.py`
    - stage2 `TemporalConv` no longer hardcodes `temporal_fusing_input_dim=24`
    - now uses `train.ppo.proprio_dim`
  - `dexscrew/algo/ppo/padapt.py`
    - passes `proprio_dim` into `ActorCritic`
  - impact:
    - fixes `Dexh13HoraLightbulb` student smoke crash:
      - `RuntimeError: mat1 and mat2 shapes cannot be multiplied (... 32 and 24 ...)`
    - preserves `XHand` behavior because `XHand` still resolves to `proprio_dim=24`
- DexH13 student smoke progress:
  - `ProprioAdapt`: `EvalSummary steps=64 avg_reward=-9.051320 avg_done_rate=0.015625`
  - `PureBC`: `EvalSummary steps=64 avg_reward=-9.049654 avg_done_rate=0.015625`
  - `DiffusionLatentStudent`: `EvalSummary steps=64 avg_reward=-9.163227 avg_done_rate=0.015625`
    - `EvalReconSummary latent_mse=0.111387 latent_l1=0.279704 action_mse_to_teacher=0.001462`
  - `ConsistencyLatentStudent`: `EvalSummary steps=64 avg_reward=-9.187637 avg_done_rate=0.015625`
    - `EvalReconSummary latent_mse=0.102859 latent_l1=0.261397 action_mse_to_teacher=0.001217`
  - `FlowMatchingLatentStudent`: `EvalSummary steps=64 avg_reward=-9.202663 avg_done_rate=0.015625`
    - `EvalReconSummary latent_mse=0.113891 latent_l1=0.277006 action_mse_to_teacher=0.001581`
- post-fix regression spot-check:
  - `XHandHoraLightbulb + ProprioAdapt` 16-step eval still passes after the common `proprio_dim` fix:
    - `EvalSummary steps=16 avg_reward=-7.725555 avg_done_rate=0.000000`

## Important implementation note
- user-added `assets/lightbulb/0000_lightbulb.urdf` referenced missing STL meshes under `assets/object_sim/lightbulb/`
- runtime V9 route therefore uses:
  - `assets/screw/lightbulb/0000_lightbulb.urdf`
- this new asset keeps the intended `3-link / 1-DOF screw-like` contract, but uses self-contained primitive geometry so smoke and training entrypoints are not blocked by missing meshes

## Remaining work
- `PLANS_v9` 范围内的 smoke acceptance 已完成
- 若继续推进，只剩 plan 外工作：
  - 更长 teacher/student 训练
  - `nominal/light/hard` 正式比较矩阵
  - 排名变化是否稳定于 longer-run / multiseed

## Current acceptance view
- `engineering_wiring_accept = yes`
- `runtime_env_reset_accept = yes`
- `core_student_ctor_accept = yes`
- `teacher_smoke_accept = yes`
  - `XHandHoraLightbulb = yes`
  - `Dexh13HoraLightbulb = yes`
- `student_train_test_smoke_accept = yes`
  - `XHandHoraLightbulb + 5 core students = yes`
  - `Dexh13HoraLightbulb + 5 core students = yes`
- `ranking_comparison_accept = smoke_protocol_yes`
  - current comparable protocol = `teacher smoke + student 64-step no-noise eval`

## Single next step
- `PLANS_v9` 已收口。
- 如果要继续，下一步应新开“longer-run comparison”计划，而不是继续在 V9 smoke 里重复试验。
