# PLANS_v8

## Status

- status: `completed_conclude`
- proposed_on: 2026-04-17
- activated_on: 2026-04-17
- completed_on: 2026-04-17
- based_on: `PLANS_v7.md (completed_conclude)`
- execution_intent: `flow_matching_recovery_sprint`
- final_decision: `conclude_flow_not_continue_worthy_under_v8_scope`

---

## 0. 背景与定位

`PLANS_v8` 不推翻 `V6/V7` 的结论。  
本轮只针对 `FlowMatchingLatentStudent` 做一次有界 recovery sprint，目标不是直接替代 `padapt`，而是先把 flow 修到“有资格继续推进”的状态。

主线 baseline 保持：

- `padapt`

diffusion 内部参考保持：

- `V5.5 consistency_boundary_bc_tuned`

---

## 1. 已知起点

当前唯一正式 flow 候选：

- `outputs/XHandHoraScrewDriver_student_flow_matching/v5_m2_flow_baseline_seed42_15min/stage2_flow_nn/model_best.ckpt`

其统一协议下指标（`seed=42`, `steps=256`, `nominal + light_v2 + hard`）为：

- nominal `1.782656`
- light_v2 `1.587523`
- hard `1.367413`
- hard_done `0.002686`

已知缺口：

- 训练默认 `x1 ~ N(0, I)`，推理默认却从 `zero latent` 开始
- BC 当前只监督解析式 `pred_latent = x_t - t * v_pred`
- 尚未补齐 `sa_mean_std` / `agent_steps` 的保存恢复闭环

---

## 2. 里程碑

### V8-M0：Flow 工程闭环补丁

目标：

- 统一 flow latent rollout 语义
- 增加 train-time rollout BC 对齐能力
- 补齐 save/restore 工程闭环

实现文件：

- `dexscrew/algo/ppo/flow_matching_latent_student.py`

完成条件：

- [x] `_init_latent(batch_size, mode)`
- [x] `_sample_latent_rollout(proprio_hist, infer_steps=None, init_mode=None)`
- [x] `sample_latent()` 统一走 rollout helper
- [x] 新增 `flow_train_init_mode`
- [x] 新增 `flow_train_align_infer`
- [x] 新增 `flow_rollout_bc_coef`
- [x] `restore_train()` 恢复 `sa_mean_std`
- [x] `restore_train()` 恢复 `agent_steps`
- [x] `restore_test()` 缺失 `sa_mean_std` 时显式 warning
- [x] `save()` 增加 `agent_steps`

兼容性结果：

- 旧 `v5_m2_flow_baseline` 在新代码下默认 1-step eval 与历史完全一致：
  - nominal `1.782656 / 0.001383`
  - light_v2 `1.587523 / 0.002116`
  - hard `1.367413 / 0.002686`

### V8-M1：现有 flow artifact 的 multistep eval-only probe

目标：

- 补齐 `V5` 未执行的 `flow_multistep` 假设

固定 artifact：

- `v5_m2_flow_baseline_seed42_15min/stage2_flow_nn/model_best.ckpt`

固定 probe：

1. `v8_m1_flow_eval_infer2_seed42`
2. `v8_m1_flow_eval_infer4_seed42`

规则：

- 两个 probe 都跑 `nominal + light_v2 + hard`
- 只补事实，不作为正式 accept 候选

当前结果：

- `infer2`:
  - nominal `1.805502 / 0.001546`
  - light_v2 `1.681982 / 0.001953`
  - hard `1.428555 / 0.002604`
- `infer4`:
  - nominal `1.885794 / 0.001302`
  - light_v2 `1.640372 / 0.002523`
  - hard `1.189338 / 0.002930`

结论：

- `infer2` 是整体更强的 multistep 方向：
  - `light_v2`、`hard` 都较 baseline 改善
- `infer4` 只在 nominal 上更强，但鲁棒条件明显退化
- 后续 M2 虽按固定 run list 执行，但解释优先级以 `infer2` 为主

### V8-M2：bounded fresh sweep（5 个 seed42 候选）

统一前提：

- fresh train
- teacher ckpt 起训
- `seed=42`
- 统一 eval：`nominal + light_v2 + hard`, `steps=256`
- 默认：
  - `flow_loss_coef=1.0`
  - `bc_loss_coef=1.2`
  - `flow_stochastic_infer=False`

候选：

1. `v8_m2_flow_zeroinit_bc12_seed42_15min`
2. `v8_m2_flow_align2_rollout_seed42_15min`
3. `v8_m2_flow_align4_rollout_seed42_15min`
4. `v8_m2_flow_align2_anchor_seed42_15min`
5. `v8_m2_flow_align2_bcheavy_seed42_15min`

当前状态：

- [x] candidate 1 train/eval
- [x] candidate 2 train/eval
- [x] candidate 3 train/eval
- [x] candidate 4 train/eval
- [x] candidate 5 train/eval

#### Candidate 1: `v8_m2_flow_zeroinit_bc12_seed42_15min`

- run_dir: `outputs/XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_zeroinit_bc12_seed42_15min`
- config: `config_041617_1f8d373.yaml`
- ckpt sha1: `40c10286d6f25dd8e80aff1a3eeb8324adc8a89b`
- training:
  - teacher fresh-start restore emitted expected compatibility warnings for missing `sa_mean_std` / `agent_steps`
  - `Current Best` stayed pinned at `-1.62` through ~`4M` agent steps, so the run was early-stopped as an obvious failure
- eval:
  - nominal `0.992769 / 0.001872`
  - light_v2 `0.921589 / 0.002523`
  - hard `1.020256 / 0.002604`
- recon:
  - nominal `latent_mse=0.136123 latent_l1=0.249300 action_mse_to_teacher=0.227103`
  - light_v2 `latent_mse=0.135628 latent_l1=0.248594 action_mse_to_teacher=0.222130`
  - hard `latent_mse=0.135610 latent_l1=0.248958 action_mse_to_teacher=0.219368`
- relative to current flow baseline:
  - nominal reward `-0.789887`, done `+0.000489`
  - light_v2 reward `-0.665934`, done `+0.000407`
  - hard reward `-0.347157`, done `-0.000082`
- conclusion:
  - `zero-init + bc1.2` with 1-step rollout is much worse than the existing flow baseline and is rejected

#### Candidate 2: `v8_m2_flow_align2_rollout_seed42_15min`

- run_dir: `outputs/XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_align2_rollout_seed42_15min`
- config: `config_041617_1f8d373.yaml`
- ckpt sha1: `9e4b34ef18de0df0648acec5c12719f0e39d1175`
- training:
  - `infer2 + rollout_bc` run stayed pinned at `Current Best = -1.62` through ~`2M` agent steps
  - `model_best.ckpt` timestamp never advanced beyond the initial save, so the run was early-stopped and treated as a failed training trajectory
- eval:
  - nominal `1.082436 / 0.002035`
  - light_v2 `1.108353 / 0.002116`
  - hard `0.750190 / 0.003174`
- recon:
  - nominal `latent_mse=0.138329 latent_l1=0.251547 action_mse_to_teacher=0.223302`
  - light_v2 `latent_mse=0.136343 latent_l1=0.249738 action_mse_to_teacher=0.222772`
  - hard `latent_mse=0.137999 latent_l1=0.250759 action_mse_to_teacher=0.230356`
- relative to current flow baseline:
  - nominal reward `-0.700220`, done `+0.000652`
  - light_v2 reward `-0.479170`, done `+0.000000`
  - hard reward `-0.617223`, done `+0.000488`
- conclusion:
  - adding rollout BC on top of `infer2` did not recover training and substantially underperformed the existing flow baseline

#### Candidate 5: `v8_m2_flow_align2_bcheavy_seed42_15min`

- run_dir: `outputs/XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_align2_bcheavy_seed42_15min`
- config: `config_041618_1f8d373.yaml`
- ckpt sha1: `d928a041e9c5171124b333b175d5794c9da6ec0c`
- training:
  - `infer2 + rollout_bc` with `flow_loss=0.8`, `bc_loss=1.6` still stayed pinned at `Current Best = -1.62`
  - no `model_best` refresh by ~`1M` agent steps, so the run was early-stopped as another failed training trajectory
- eval:
  - nominal `1.124580 / 0.001872`
  - light_v2 `1.055061 / 0.002441`
  - hard `1.053348 / 0.002279`
- recon:
  - nominal `latent_mse=0.138899 latent_l1=0.251806 action_mse_to_teacher=0.227683`
  - light_v2 `latent_mse=0.136222 latent_l1=0.249287 action_mse_to_teacher=0.221649`
  - hard `latent_mse=0.137980 latent_l1=0.251982 action_mse_to_teacher=0.220854`
- relative to current flow baseline:
  - nominal reward `-0.658076`, done `+0.000489`
  - light_v2 reward `-0.532462`, done `+0.000325`
  - hard reward `-0.314065`, done `-0.000407`
- conclusion:
  - heavier BC slightly improves nominal vs candidate 2, but it still fails badly on all three conditions and remains far below the current flow baseline

#### Candidate 4: `v8_m2_flow_align2_anchor_seed42_15min`

- run_dir: `outputs/XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_align2_anchor_seed42_15min`
- config: `config_041618_1f8d373.yaml`
- ckpt sha1: `cd9f477ba4bd6bbee2974c9d1b3031562096c2c2`
- training:
  - `infer2 + rollout_bc + base_action_anchor` still stayed pinned at `Current Best = -1.62`
  - no best refresh by ~`1M` agent steps, so the run was early-stopped as another failed training trajectory
- eval:
  - nominal `1.000670 / 0.002035`
  - light_v2 `0.923009 / 0.002360`
  - hard `0.953265 / 0.002686`
- recon:
  - nominal `latent_mse=0.137646 latent_l1=0.251132 action_mse_to_teacher=0.226906`
  - light_v2 `latent_mse=0.137913 latent_l1=0.250770 action_mse_to_teacher=0.226544`
  - hard `latent_mse=0.134574 latent_l1=0.247927 action_mse_to_teacher=0.226123`
- relative to current flow baseline:
  - nominal reward `-0.781986`, done `+0.000652`
  - light_v2 reward `-0.664514`, done `+0.000244`
  - hard reward `-0.414148`, done `+0.000000`
- conclusion:
  - adding a small action anchor does not rescue the `infer2 + rollout` family; this branch is now effectively swept out

#### Candidate 3: `v8_m2_flow_align4_rollout_seed42_15min`

- run_dir: `outputs/XHandHoraScrewDriver_student_flow_matching/v8_m2_flow_align4_rollout_seed42_15min`
- config: `config_041618_1f8d373.yaml`
- ckpt sha1: `70d0ed774910581006041918bf8aaf8321c7aec7`
- training:
  - `infer4 + rollout_bc` also stayed pinned at `Current Best = -1.62`
  - no best refresh by ~`1M` agent steps, so the run was early-stopped as another failed training trajectory
- eval:
  - nominal `1.114895 / 0.001790`
  - light_v2 `1.152107 / 0.001872`
  - hard `1.079496 / 0.002848`
- recon:
  - nominal `latent_mse=0.135902 latent_l1=0.249294 action_mse_to_teacher=0.220026`
  - light_v2 `latent_mse=0.135799 latent_l1=0.249863 action_mse_to_teacher=0.222044`
  - hard `latent_mse=0.135067 latent_l1=0.249105 action_mse_to_teacher=0.225638`
- relative to current flow baseline:
  - nominal reward `-0.667761`, done `+0.000407`
  - light_v2 reward `-0.435416`, done `-0.000244`
  - hard reward `-0.287917`, done `+0.000162`
- conclusion:
  - `infer4 + rollout` is the strongest fresh seed42 candidate in V8, but it still remains clearly below the current flow baseline and far below the V8 continue gate

### V8-M2.5：best seed42 候选扩时

仅当 best seed42 候选同时满足：

- `hard_reward >= 1.48`
- `light_v2_reward >= 1.66`
- `nominal_reward >= 1.90`
- `hard_done <= 0.00245`

才执行：

- `v8_m2_best_seed42_30min`

结果：

- `not_triggered`
- best fresh candidate = `v8_m2_flow_align4_rollout_seed42_15min`
- but it fails the required pre-threshold on every reward metric:
  - nominal `1.114895 < 1.90`
  - light_v2 `1.152107 < 1.66`
  - hard `1.079496 < 1.48`
  - hard_done `0.002848 > 0.00245`

### V8-M3：multiseed top-1 only

仅当某个 seed42 候选满足 single-seed continue gate 时触发：

- 复用 seed42
- 新跑 seeds `43,44`
- 不开第二个 multiseed 候选

结果：

- `not_triggered`
- reason:
  - no seed42 candidate reached the single-seed continue gate
  - best fresh candidate hard reward `1.079496` is far below `1.55`

---

## 3. 验收标准

### 3.1 Single-seed continue gate

必须同时满足：

- `nominal_reward >= 1.95`
- `light_v2_reward >= 1.70`
- `hard_reward >= 1.55`
- `hard_done <= 0.002372`

### 3.2 Multiseed continue accept

必须同时满足：

- `nominal_mean >= 1.90`
- `light_v2_mean >= 1.65`
- `hard_mean >= 1.50`
- `hard_done_mean <= 0.002372`

### 3.3 Stretch accept

若达到下列更强水平，则记为 stretch：

- `nominal >= 2.10`
- `light_v2 >= 1.85`
- `hard >= 1.64`
- `hard_done <= 0.002035`

### 3.4 V8 conclude

满足任一则收口：

- `M1 + M2` 后仍无 single-seed continue gate pass
- `M2.5` 预门槛未满足
- `M3` 启动后未达到 multiseed continue accept

V8 最终结论：

- `M1` 虽表明 `infer2` 比 `infer4` 更有 multistep 信号，但 `M2` 的 5 个 fresh seed42 候选全部失败
- `infer2` family (`align2_rollout / anchor / bcheavy`) 全部复现相同的 failed-training pattern：
  - `Current Best` 持续卡在 `-1.62`
  - deploy reward 也显著低于 current flow baseline
- `infer4_rollout` 作为最强 fresh 候选，仍然只有：
  - nominal `1.114895`
  - light_v2 `1.152107`
  - hard `1.079496`
- 因此：
  - `M2.5` 不触发
  - `M3` 不触发
  - `PLANS_v8` 记为 `completed_conclude`

---

## 4. 记录规范

每个已执行候选必须记录：

- run_dir
- ckpt sha1
- config snapshot
- train/eval logs
- `EvalSummary`
- `EvalReconSummary`
- 相对以下 reference 的 delta：
  - current flow baseline
  - `V5.5 accepted consistency`
  - `padapt`
