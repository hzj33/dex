# PLANS_v7

## Status

- status: `completed_conclude`
- proposed_on: 2026-04-16
- activated_on: 2026-04-16
- completed_on: 2026-04-16
- final_decision: `conclude_no_single_seed_breakthrough_under_v7_scope`
- based_on: `PLANS_v6.md (completed_conclude)`
- execution_intent: `consistency_close_hard_gap_toward_padapt`

---

## 0. 背景与定位

`PLANS_v7` 不推翻 `V6 Conclude`。  
它是在 V6 结论成立的前提下，完成两件事：

1. 关闭 V6 剩余工程/实现风险；
2. 在严格有界预算内，继续尝试让 `ConsistencyLatentStudent` 先跨过单种子 hard 门槛，再决定是否进入 multiseed。

主线 baseline 仍为 `padapt`，`consistency` 继续只在当前 student 路线内局部优化。

---

## 1. V7 目标

### 1.1 主目标

先拿到一个 **可信的 single-seed 过线候选**：

- `hard_reward >= 1.733`
- `nominal_reward >= 2.100`
- `light_v2_reward >= 1.900`
- `hard_done <= 0.002372`

若达到，再进入 multiseed 验证。

### 1.2 修正后的门槛口径

`padapt hard mean = 1.838225`  
`padapt hard std = 0.105458`

因此修正后的 1σ hard 门槛为：

`1.838225 - 0.105458 = 1.732767`

V7 中统一记为：

- `hard >= 1.733`

Done 容差沿用：

- `nominal_done <= 0.001420`
- `light_v2_done <= 0.001400`
- `hard_done <= 0.001550`

---

## 2. 范围与边界

### 2.1 纳入范围

- 仅 `ConsistencyLatentStudent`
- 仅局部 student 头部/推理路径/保存恢复修补
- 仅现有 eval 协议：
  - `seed=42`
  - `steps=256`
  - `nominal + light_v2 + hard`
- fresh train 一律从 teacher ckpt 起训

### 2.2 不纳入范围

- 不改 teacher
- 不改 frozen backbone
- 不新增 student family
- 不改 eval 条件定义
- 不做数据策略/训练分布大改

---

## 3. 里程碑

### V7-M0：风险收口补丁

目标：关闭 V6 剩余实现风险，并补齐 EMA 推理语义。

完成条件：

- [x] 抽出 `_sample_latent_with_model(model, proprio_hist)`
- [x] `sample_latent()` 统一走 helper
- [x] `sample_latent_train()` 在 `train_align_infer=True` 且有 EMA 时走 EMA latent rollout
- [x] 新增 `consistency_infer_use_ema`
- [x] checkpoint 保存 `agent_steps`
- [x] `restore_train()` 恢复 `agent_steps`
- [x] `restore_train()` 恢复 `sa_mean_std`，缺失时显式 warning
- [x] `restore_test()` 在 `sa_mean_std` 缺失时显式 warning

实现文件：

- `dexscrew/algo/ppo/consistency_latent_student.py`

---

### V7-M1：现有 EMA artifact 最小复核

目标：先判断已有 `v6_m3_ema_target` 是否在 **EMA 推理语义** 下可直接晋级。

目标 artifact：

- `outputs/XHandHoraScrewDriver_student_consistency/v6_m3_ema_target_seed42_15min/stage2_consistency_nn/model_best.ckpt`

执行规则：

1. 先跑 `hard-only`
2. 使用：
   - `+train.ppo.consistency_use_ema_target=True`
   - `+train.ppo.consistency_infer_use_ema=True`
3. 若 `hard >= 1.733` 且 `hard_done <= 0.002372`，再补跑 nominal / light_v2
4. 若未过 hard 门槛，则不再补 nominal / light_v2，直接进入 M2

当前结果：

- [x] hard-only EMA eval 已完成
- result:
  - `hard_reward = 1.608308`
  - `hard_done = 0.001628`
- decision:
  - 未达到 `hard >= 1.733`
  - 不晋级 multiseed
  - 进入 M2 fresh-train sweep

eval artifact：

- `outputs/XHandHoraScrewDriver_eval_robustness/v7_m1_ema_target_seed42_hard_emaeval_fix/`

---

### V7-M2：single-seed bounded sweep

预算：最多 4 个候选（6-run 扩展版中的 single-seed 部分）

统一起点：

- 基于 `V5.5 accepted config`
- fresh train from teacher ckpt

统一 teacher ckpt：

- `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`

#### 候选 A：`ema_obs_combo_seed42_15min`

- `+train.ppo.consistency_boundary_coef=0.8`
- `+train.ppo.consistency_num_scales=16`
- `+train.ppo.bc_loss_coef=1.2`
- `+train.ppo.consistency_use_ema_target=True`
- `+train.ppo.consistency_ema_decay=0.995`
- `+train.ppo.consistency_infer_use_ema=True`
- `+train.ppo.consistency_obs_noise_curriculum=True`
- `+train.ppo.consistency_obs_noise_curriculum_mode=linear`
- `+train.ppo.consistency_obs_noise_curriculum_start=0`
- `+train.ppo.consistency_obs_noise_curriculum_steps=800000`
- `+train.ppo.consistency_obs_noise_e_target=0.05`
- `+train.ppo.consistency_obs_noise_t_target=0.025`

状态：

- [x] train
- [x] nominal eval
- [x] light_v2 eval
- [x] hard eval

结果：

- nominal:
  - `reward = 0.211057`
  - `done = 0.001953`
- light_v2:
  - `reward = 0.296649`
  - `done = 0.001709`
- hard:
  - `reward = 0.355474`
  - `done = 0.002279`

结论：

- 明确未达到 V7 single-seed gate
- 也未满足候选 D 触发条件（`hard >= 1.700` 且 `nominal >= 2.100`）
- 本方向淘汰，进入候选 B

run dir：

- `outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_obs_combo_seed42_15min/`

#### 候选 B：`ema_target_seed42_30min`

- V5.5 accepted config
- `+train.ppo.consistency_use_ema_target=True`
- `+train.ppo.consistency_ema_decay=0.995`
- `+train.ppo.consistency_infer_use_ema=True`
- 30min budget via external timeout

状态：

- [x] train_finished
- [x] nominal eval
- [x] light_v2 eval
- [x] hard eval

结果：

- ckpt sha1:
  - `1c7c75577496a00234b1007e89fb2715ea391d7c`
- note:
  - `model_best.ckpt` 与候选 A 完全同 hash，说明 30min EMA-only run 未产生新的 best artifact
- nominal:
  - `reward = 0.211057`
  - `done = 0.001953`
- light_v2:
  - `reward = 0.296649`
  - `done = 0.001709`
- hard:
  - `reward = 0.355474`
  - `done = 0.002279`

结论：

- 与候选 A 数值一致，未提供新增证据价值
- 明确未达到 V7 single-seed gate
- 转入候选 C

run dir：

- `outputs/XHandHoraScrewDriver_student_consistency/v7_m2_ema_target_seed42_30min/`

#### 候选 C：`ema_alignfix_seed42_15min`

- V5.5 accepted config
- `+train.ppo.consistency_use_ema_target=True`
- `+train.ppo.consistency_ema_decay=0.995`
- `+train.ppo.consistency_infer_use_ema=True`
- `+train.ppo.consistency_infer_steps=2`
- `+train.ppo.consistency_train_align_infer=True`

状态：

- [x] train_finished
- [x] nominal eval
- [x] light_v2 eval
- [x] hard eval

结果：

- ckpt sha1:
  - `024409e9f765232775f33f139493e04c09ae29d2`
- nominal:
  - `reward = 1.124038`
  - `done = 0.002035`
- light_v2:
  - `reward = 1.177704`
  - `done = 0.002279`
- hard:
  - `reward = 1.135807`
  - `done = 0.001953`

结论：

- 训练信号明显优于 A/B，但三条件 reward 仍整体低于 single-seed gate
- `nominal < 2.100` 且 `hard < 1.700`，因此候选 D 不触发
- M2 无 single-seed 过线候选

#### 候选 D：`ema_obs_combo_seed42_30min`

仅在 A/B/C 均未过 single-seed gate，且最佳候选满足：

- `hard >= 1.700`
- `hard_done <= 0.002372`
- `nominal >= 2.100`

时才执行。

状态：

- [x] not_triggered

原因：

- A/B 均远低于门槛
- C 为本轮最佳新 artifact，但仍不满足：
  - `hard >= 1.700`
  - `nominal >= 2.100`

---

### V7-M3：multiseed top-1 only

仅当 M2 中出现 single-seed entry pass 候选时触发。

规则：

- 只选 top-1
- seed42 复用已完成 run
- 新跑 seeds `43,44`
- 配置/timeout/eval mode 必须完全一致

状态：

- [x] not_triggered

原因：

- M2 无 single-seed entry pass 候选

---

## 4. 验收标准

### 4.1 Single-Seed Entry Gate

候选进入 multiseed 前必须同时满足：

- `hard_reward >= 1.733`
- `nominal_reward >= 2.100`
- `light_v2_reward >= 1.900`
- `hard_done <= 0.002372`

### 4.2 V7 Primary Accept

top-1 multiseed 候选必须同时满足：

- `hard_mean >= 1.733`
- `nominal_mean >= 2.100`
- `light_v2_mean >= 2.000`
- `nominal_done_mean <= 0.001420`
- `light_v2_done_mean <= 0.001400`
- `hard_done_mean <= 0.001550`

### 4.3 Stretch Accept

若 multiseed 同时满足：

- `nominal_mean >= 2.168`
- `light_v2_mean >= 2.079`
- `hard_mean >= 1.838`
- done 三项全部不差于 `PAdapt + 1σ`

则记为 stretch success。

### 4.4 V7 Conclude

满足任一则收口：

- EMA 重评估 + A/B/C 后仍无 single-seed entry pass
- D 触发后仍未过 single-seed entry gate
- 或 multiseed 未达到 V7 primary accept

当前执行结果：

- 已满足第一条：
  - `M1 EMA recheck` 失败
  - `M2 A/B/C` 全部未过 single-seed gate
  - `D` 未触发
  - `M3` 未触发

最终本地结论：

- `PLANS_v7 = completed_conclude`
- 在当前 bounded scope 内，`ConsistencyLatentStudent` 仍未拿到可进入 multiseed 的 single-seed hard 过线候选
- `padapt` 继续保持主线 baseline 地位

---

## 5. 记录规范

每个已执行候选必须记录：

- train run_dir
- ckpt sha1
- 三条件 eval logs
- 与 `V5.5 accepted consistency`、`padapt` 的 delta
- 是否通过：
  - single-seed gate
  - anti-regression
  - multiseed primary accept / stretch accept
