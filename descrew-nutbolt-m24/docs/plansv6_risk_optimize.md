# PLANS_v6 Risk & Optimization Summary

Date: 2026-04-16  
Based on: Code review + evaluation audit of `PLANS_v6` execution  
Verdict: `support_current_conclusion` — V6 Conclude 结论成立，但存在若干残余风险与可选优化方向

---

## Part 1: 已验证无问题（初始疑虑已排除）

以下问题在审阅中被提出，但经过最小复核后确认**不影响当前结论**：

### R-0a: EMA 恢复路径（已排除）

- 疑虑：`restore_train` / `restore_test` 中若 checkpoint 不含 `consistency_ema_model`，会从 `consistency_model` 初始化，可能导致 EMA 状态丢失。
- 验证：
  ```
  python3 -c "import torch; ckpt=torch.load('...v6_m3_ema_target.../model_best.ckpt', map_location='cpu'); print(list(ckpt.keys()))"
  # 输出包含 consistency_ema_model
  ```
- 结论：`ema_target` 候选的 checkpoint 正确保存了 EMA 权重，恢复路径无问题。

### R-0b: Obs Noise Curriculum 污染 Eval（已排除）

- 疑虑：训练时 curriculum 将 `self.env.random_obs_noise_e_scale` ramp 到 target，eval 时未重置，可能导致 nominal eval 在错误 noise level 下进行。
- 验证：
  - eval 脚本 `scripts/eval_screwdriver_student_robustness.sh` 通过 Hydra config 显式覆盖 `task.env.randomization.obs_noise_e_scale=0.0`（nominal）
  - eval log 确认：nominal 条件下 `obs_noise_e_scale=0.0`，hard 条件下 `obs_noise_e_scale=0.05`
  - eval 使用独立进程 + 新 env 实例，训练时的 Python 对象状态不传递
- 结论：eval 协议正确，`obs_noise_curriculum` 候选结果有效。

---

## Part 2: 残余风险（低影响，当前结论不受影响）

### R-1: Train/Infer Alignment 设计次优

- **文件/行号：** [consistency_latent_student.py:427-428](../dexscrew/algo/ppo/consistency_latent_student.py#L427-L428)
- **问题描述：**
  当 `train_align_infer=True` 时，BC loss 使用 `sample_latent_train()` 产生的 latent，但：
  - `sample_latent_train()` 与 `sample_latent()` 代码完全相同（均调用 `self.consistency_model`，非 EMA）
  - Consistency loss 的 `pred_lo_target` 使用 EMA model（若启用），BC loss 的 latent 来源不一致
  - 训练时 BC loss 优化的 latent 路径与 consistency loss 优化的路径存在分歧
- **当前影响：**
  - `infer2_align` 候选（V6-M3 direction B）的训练动态可能次优
  - 结果 hard `1.622737` 仍然有效，但该方向的潜力可能未被充分挖掘
- **严重度：** Medium（设计次优，非实现 bug）
- **是否影响 V6 结论：** 否（结果有效，只是可能低估了该方向的上限）

### R-2: Checkpoint 未保存 Agent Steps

- **文件/行号：** [consistency_latent_student.py:551-567](../dexscrew/algo/ppo/consistency_latent_student.py#L551-L567)
- **问题描述：**
  `save()` 中未保存 `self.agent_steps`。若 resume 训练，`agent_steps` 从 0 重新计数，导致 obs_noise_curriculum 的 progress 计算错误（`L188-189`）。
- **当前影响：**
  - V6 所有候选均为 fresh train，未触发 resume，当前无影响
  - 若未来需要 resume 含 curriculum 的训练，会导致 curriculum 重置
- **严重度：** Low（当前无影响）

### R-3: Restore 路径 Train/Test 不对称

- **文件/行号：** [consistency_latent_student.py:504-549](../dexscrew/algo/ppo/consistency_latent_student.py#L504-L549)
- **问题描述：**
  - `restore_train` 使用 `strict=False` 加载 model，并有详细的 EMA fallback 逻辑
  - `restore_test` 同样 `strict=False`，EMA fallback 逻辑相同，但缺少 optimizer state 加载（合理）
  - 两者行为基本一致，但若 checkpoint 格式不一致（如旧版 checkpoint 无 `sa_mean_std`），`restore_test` 会静默跳过，可能导致 normalization 状态不一致
- **当前影响：** 低，V6 所有候选均使用同版本 checkpoint
- **严重度：** Low

### R-4: sample_latent_train 与 sample_latent 完全重复

- **文件/行号：** [consistency_latent_student.py:274-284](../dexscrew/algo/ppo/consistency_latent_student.py#L274-L284)
- **问题描述：** 两个方法代码完全相同，`sample_latent_train` 存在的意义是为了未来可能的差异化（如 train 时用 EMA），但当前未实现
- **当前影响：** 无，代码冗余
- **严重度：** Low（代码质量问题）

---

## Part 3: 可选优化方向（若用户决定继续探索）

以下方向**不是当前结论的必要修正**，而是在 V6 Conclude 基础上，若治理层决定继续探索时的优先候选。

每个方向均标注：预期收益、实现成本、风险。

---

### O-1: 修复 Train/Infer Alignment（对应 R-1）

- **方向：** 让 BC loss 使用与 consistency loss 一致的 latent 来源
- **具体改动：**
  ```python
  # 当前（次优）
  if self.train_align_infer:
      pred_latent = self.sample_latent_train(input_dict["proprio_hist"])
  
  # 改进：BC loss 使用 EMA model 的多步推理（若启用 EMA）
  if self.train_align_infer:
      with torch.no_grad():
          infer_model = self.consistency_ema_model if self.consistency_ema_model is not None else self.consistency_model
          pred_latent = self._sample_latent_with_model(infer_model, input_dict["proprio_hist"])
  ```
- **预期收益：** `infer2_align` 方向的 hard reward 可能从 `1.622737` 提升，但幅度不确定
- **实现成本：** ~10 行改动，低风险
- **前提：** 需要新增 `_sample_latent_with_model(model, proprio_hist)` 辅助方法
- **注意：** 改动后需重新训练 + 评测，不能复用旧 checkpoint

---

### O-2: 组合最佳 M3 方向（ema_target + obs_noise_curriculum）

- **方向：** 同时启用 EMA target 和 obs noise curriculum
- **具体 config：**
  ```
  +train.ppo.consistency_use_ema_target=True
  +train.ppo.consistency_ema_decay=0.995
  +train.ppo.consistency_obs_noise_curriculum=True
  +train.ppo.consistency_obs_noise_e_target=0.05
  +train.ppo.consistency_obs_noise_t_target=0.025
  ```
- **预期收益：**
  - EMA target 提供训练稳定性（hard `1.723353`）
  - Obs noise curriculum 提供 hard 条件鲁棒性（hard `1.387970`，但方向有意义）
  - 组合可能产生协同效应
- **实现成本：** 零代码改动，纯 config 组合，1 次训练 run
- **风险：** 两个方向在 V6 中单独均未达标，组合不保证超越 1.780；可能相互干扰
- **建议：** 若继续探索，这是成本最低的第一步

---

### O-3: 延长训练时长 + EMA Target 组合

- **方向：** `ema_target` 基础上延长训练到 30min 或 1h
- **具体 config：**
  ```
  +train.ppo.consistency_use_ema_target=True
  +train.ppo.consistency_ema_decay=0.995
  max_training_time=1800  # 30min
  ```
- **预期收益：**
  - M1 `longer_train` 显示 30min 相比 15min 有改善（hard `1.415326` vs V5.5 baseline `1.714471`，但 longer_train 从 V5.5 config 出发）
  - `ema_target` 在 15min 已达 `1.723353`，延长训练可能进一步收敛
- **实现成本：** 零代码改动，1 次训练 run（30min 预算）
- **风险：** M1 `longer_train` 结果显示延长训练对 hard 改善有限（`1.415326`），但 `ema_target` 起点更高，收益可能更大
- **建议：** 若 O-2 失败，这是第二优先候选

---

### O-4: 修复 Checkpoint 保存 Agent Steps（对应 R-2）

- **方向：** 在 `save()` 中加入 `agent_steps`，支持正确 resume
- **具体改动：**
  ```python
  weights = {
      "model": self.model.state_dict(),
      "consistency_model": self.consistency_model.state_dict(),
      "consistency_optim": self.optim.state_dict(),
      "agent_steps": self.agent_steps,  # 新增
  }
  ```
  同时在 `restore_train` 中恢复：
  ```python
  if "agent_steps" in checkpoint:
      self.agent_steps = checkpoint["agent_steps"]
  ```
- **预期收益：** 支持含 curriculum 的训练正确 resume
- **实现成本：** ~4 行改动，低风险
- **建议：** 若未来有 resume 需求，应在下一个训练 run 前修复

---

## Part 4: 优先级矩阵

| 编号 | 方向 | 预期 hard 收益 | 实现成本 | 推荐优先级 |
|------|------|---------------|----------|-----------|
| O-2 | ema_target + obs_noise_curriculum 组合 | 未知，可能 +0.03~+0.08 | 极低（纯 config） | ⭐⭐⭐ 最高 |
| O-3 | ema_target + 30min 训练 | 可能 +0.02~+0.05 | 低（1 run，30min） | ⭐⭐ 次高 |
| O-1 | 修复 train/infer alignment | 未知，可能 +0.02~+0.06 | 低（~10 行代码 + 1 run） | ⭐⭐ 次高 |
| O-4 | 修复 checkpoint agent_steps | 无直接 reward 收益 | 极低（~4 行代码） | ⭐ 工程修复 |

---

## Part 5: 决策建议

### 若接受 V6 Conclude（推荐）

- 当前结论成立，无需任何修复
- 保留 `padapt` 为主线，`consistency` 作为次优备选与论文负结果对照
- 可选：执行 O-4（工程修复，不影响结论）

### 若决定继续探索（需新计划授权）

建议执行顺序：
1. **O-2**（零代码成本，1 run）→ 若 hard >= 1.780，进入 multiseed
2. **O-3**（零代码成本，1 run，30min 预算）→ 若 O-2 失败
3. **O-1**（需代码改动，1 run）→ 若 O-2/O-3 均失败

若 O-1/O-2/O-3 全部失败（hard 仍 < 1.780），则确认 V6 Conclude 为最终结论，不再继续。

**注意：** 继续探索需要在新计划中明确授权，不应在 V6 边界内执行。

---

## 附：V6 M3 候选结果汇总（供参考）

| Candidate | nominal | light_v2 | hard | hard >= 1.780 | 结果可信度 |
|-----------|---------|----------|------|---------------|-----------|
| `obs_noise_curriculum` | 2.022649 | 1.830568 | 1.387970 | FAIL | ✅ 高 |
| `infer2_align` | 1.980866 | 1.786391 | 1.622737 | FAIL | ✅ 高（设计次优但结果有效） |
| `ema_target` | 1.736832 | 2.077133 | 1.723353 | FAIL | ✅ 高 |
| PAdapt reference | 2.167820 | 2.079074 | 1.838225 | — | — |
