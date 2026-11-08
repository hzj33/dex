# PLANS_v4

## Status

- status: `completed_suspend`
- activated_on: 2026-04-14
- completed_on: 2026-04-15
- final_decision: `Suspend` (governance-confirmed)
- supersedes_for_execution: `PLANS_v3.md`
- depends_on: bug fixes in `dexscrew/algo/ppo/diffusion_latent_student.py`

---

## 0. 生效说明

V3 共 6 个候选全部在 single-seed gate 被 reject，
两条主线均触发止损冻结。但代码审查发现 double-tanh bug
存在于所有 V3 候选中，V3 负结果可能受 bug 污染。
V4 唯一目的：在 bug-free 代码上用最小计算预算
确认 diffusion student 是否可行，或给出 clean 终止证据。

---

## 1. 为什么需要 V4

### 1.1 V3 终态

Mainline A（curriculum-based）3 个候选全部 reject：
- v3a1: delta_hard = -0.153
- v3a2: delta_hard = -0.315
- v3a3: delta_hard = -0.052（差 0.002 未过 gate）

Mainline B（residual-based）3 个候选全部 reject：
- v3b1: delta_hard = -0.547
- v3b2: delta_hard = -0.353
- v3b3: delta_hard = -0.162

两条主线均触发止损冻结，V3 正式耗尽。

### 1.2 发现的代码 bug

<!-- CHUNK_PLACEHOLDER_1 -->

**BUG（确认，HIGH）: double-tanh 压缩 latent 分布**

位置：`diffusion_latent_student.py:698` + `models.py:129`

数据流：
1. `models.py:129`: `extrin_gt = torch.tanh(extrin_gt)` — e_gt 已是 tanh 后值
2. `diffusion_latent_student.py:681`: `target_latent = e_gt.detach()` — target 已含 tanh
3. `diffusion_latent_student.py:698`: `pred_latent = torch.tanh(x0_pred)` — 对输出再做 tanh
4. `diffusion_latent_student.py:277`: `return torch.tanh(x)` — inference 同样 double-tanh

效果：tanh(tanh(x)) 严重压缩分布尾部，降低 latent 表达力。
对 hard/light_v2 条件下需要更大 latent 修正的场景尤其有害。

修复（已完成）：移除 line 698 和 line 277 的 tanh。

**可疑点（LOW）: teacher_mu 未在生成时 clamp**

位置：`diffusion_latent_student.py:666`

line 704 和 line 720 都使用了 clamped 版本，实际影响较小。
为一致性已同时修复：添加 `torch.clamp(..., -1.0, 1.0)`。

### 1.3 V4 核心判断

V3 的 6 个负结果全部在 buggy code 上产生，不能作为 clean evidence。
V4 必须在 bug-free code 上重新建立 baseline 并做最小验证。
如果 bug fix 本身不能带来显著改善，则 diffusion 扩张应终止。

---

## 2. V4 总目标

在 bug-free 代码上回答一个问题：
**修复 bug 后的 diffusion student 能否在统一 gate 下追平或超过 V3-M0 reference？**

1. 修复 bug，建立 bug-free baseline（Phase 0）
2. 对 bug-fixed baseline 做 targeted ablation（Phase 1）
3. 如果有信号，跑正式候选过 gate（Phase 2）
4. 如果失败，给出 bug-free 终止证据（Phase 3）

---

## 3. V4 范围与边界

### 3.1 纳入范围

- double-tanh bug 修复 + teacher_mu clamp 修复
- Bug-fixed baseline 重建（同 V3-M0 frozen config）
- Targeted ablation：tail_coef=0, inference steps, 有限 unfreeze
- 最多 3 个正式候选（如果 Phase 1 有信号）
- 统一评测协议（沿用 V3）

### 3.2 不纳入范围

- 全新架构改动（新 diffusion head 等）
- 新 loss 类型引入
- Curriculum / residual 方向重新探索（V3 已充分验证）
- 大规模 hyperparameter sweep
- 工程重构

---

<!-- CHUNK_PLACEHOLDER_2 -->

## 4. V4 决策指标与 Gate

### 4.1 统一评测协议（保持）

- 条件：`nominal`, `light_v2`, `hard`
- steps：`256`
- 默认先 `seed=42`，通过初筛后再 `seed=42,43,44`
- 评测脚本：`scripts/eval_screwdriver_student_robustness.sh`

### 4.2 双 Reference 体系

**Primary gate（决定 pass/fail）**: delta vs V3-M0 frozen reference

V3-M0 Reference metrics（seed 42, buggy code）：
- nominal: reward 1.675, done 0.001302
- light_v2: reward 1.638, done 0.001383
- hard: reward 1.505, done 0.001872

**Secondary metric（信息性）**: delta vs PAdapt multiseed baseline

PAdapt Reference metrics（multiseed mean）：
- nominal: 2.168 ± 0.183
- light_v2: 2.079 ± 0.103
- hard: 1.838 ± 0.105

Secondary metric 不作为 gate 判定依据，但记录在 evidence block 中。

### 4.3 初筛 Gate（单 seed）

候选进入多 seed 的必要条件：
1. `delta_hard >= -0.05`（vs V3-M0 reference）
2. `delta_light_v2 >= -0.08`（vs V3-M0 reference）
3. `delta_done_hard <= +0.0005`

### 4.4 接受 Gate（多 seed）

候选可替换 reference 的必要条件：
1. `hard_mean_delta >= 0`
2. `light_v2_mean_delta >= 0`
3. `nominal_mean_delta >= -0.10`

### 4.5 止损规则

- Phase 0 无止损（必须完成）
- Phase 1 最多 4 个 ablation 实验
- Phase 2 最多 3 个正式候选；连续 3 个初筛全失败则触发终止
- Bug-fixed baseline 的 hard delta 相对 V3-M0 恶化超过 0.10 → 立即终止
- 总计算预算：最多 8 次训练 run

---

<!-- CHUNK_PLACEHOLDER_3 -->

## 5. V4 里程碑

### V4-M0：Bug Fix + Bug-Fixed Baseline

完成条件：
- [x] BUG 修复：移除 `diffusion_latent_student.py:698` 的 `torch.tanh(x0_pred)`
- [x] BUG 修复：移除 `diffusion_latent_student.py:277` 的 `torch.tanh(x)`
- [x] 一致性修复：`diffusion_latent_student.py:666` 添加 `torch.clamp(..., -1.0, 1.0)`
- [ ] 修复后代码通过 syntax check
- [ ] 用 V3-M0 frozen config 重新训练（seed=42, 15min）
- [ ] 对 bug-fixed checkpoint 跑 nominal/light_v2/hard 单 seed 评测
- [ ] 记录 bug-fixed baseline metrics 与 V3-M0 reference 和 PAdapt 的 delta
- [ ] 产出 evidence block

训练命令：
```bash
# 与 V3-M0 frozen config 完全一致，仅代码不同
python train.py task=XHandHoraScrewDriver \
  train=XHandHoraScrewDriverStudentDiffusionLatent \
  +train.ppo.diffusion_teacher_delta_tail_coef=0.2 \
  +train.ppo.diffusion_teacher_delta_tail_threshold=0.15 \
  +train.ppo.diffusion_teacher_delta_tail_selective=True \
  +train.ppo.diffusion_base_action_anchor_coef=0.03 \
  train.ppo.max_training_time=900 \
  seed=42
```

关键判断点：
- hard delta(vs V3-M0) >= 0 → 直接进入 V4-M2 多 seed 验证
- hard delta(vs V3-M0) in [-0.05, 0) → 进入 V4-M1 做 ablation
- hard delta(vs V3-M0) < -0.05 → 仍进入 V4-M1（bug fix 可能改变 optimal 区域）
- hard delta(vs V3-M0) < -0.10（严重恶化）→ 直接进入 V4-M3 终止

---

### V4-M1：Targeted Ablation（最多 4 个实验）

前提：V4-M0 完成，且未触发直接进入 V4-M2 或 V4-M3 的条件。

<!-- CHUNK_PLACEHOLDER_4 -->

**Ablation 1: teacher_delta_tail_coef = 0**
- 假设 H2：teacher_delta_tail loss 在 bug-fixed code 上可能有害
- Config override：`+train.ppo.diffusion_teacher_delta_tail_coef=0.0`
- 其余参数与 V4-M0 baseline 一致
- 评测：单 seed gate

**Ablation 2: diffusion_steps = 20**
- 假设 H3：10-step DDPM inference 不足
- Config override：`+train.ppo.diffusion_steps=20 +train.ppo.diffusion_steps_infer=20`
- 其余参数与 V4-M0 baseline 一致
- 评测：单 seed gate

**Ablation 3: 组合（tail_coef=0 + steps=20）**
- 仅在 Ablation 1 或 2 中至少一个有改善信号时执行
- 如果两个都无改善信号，跳过此 ablation，直接尝试 Ablation 4
- Config override：两者组合

**Ablation 4: 有限 unfreeze backbone（最后手段）**
- 仅在 Ablation 1-3 全部未通过初筛 gate 时执行
- 解冻 `adapt_tconv` 的最后一层参数
- Config override：`+train.ppo.diffusion_student_trainable_param_patterns=["adapt_tconv.low_dim_proj"]`
- 基于 V4-M0 baseline config（或 Ablation 1-3 中最佳配置）
- 评测：单 seed gate
- 如果仍失败 → 直接进入 V4-M3

完成条件：
- [ ] 完成 1-4 个 ablation 的训练 + 单 seed 评测
- [ ] 至少 1 个 ablation 通过初筛 gate → 进入 V4-M2
- [ ] 或全部 ablation 未通过初筛 gate → 进入 V4-M3

---

### V4-M2：多 Seed 验证

前提：V4-M0 或 V4-M1 中至少一个候选通过单 seed 初筛 gate。

执行：
- [ ] 对所有通过初筛 gate 的候选，跑 seed=42,43,44 多 seed 评测
- [ ] 按接受 Gate（4.4）判定
- [ ] 同时记录 vs PAdapt 的 delta（判断最终竞争力）
- [ ] 如果有候选通过接受 Gate → 记录为 V4 accepted candidate
- [ ] 如果全部未通过 → 进入 V4-M3

完成条件：
- [ ] 多 seed 评测完成
- [ ] 产出 evidence block（含 per-seed metrics + aggregate）
- [ ] 明确结论：accept 或 reject

---

### V4-M3：Final Verdict

触发条件（任一）：
- V4-M0 bug-fixed baseline 严重恶化（hard delta < -0.10）
- V4-M1 全部 ablation 未通过初筛 gate
- V4-M2 全部候选未通过接受 gate
- 用户选择终止

完成条件：
- [ ] 汇总 V3 + V4 全部实验证据
- [ ] 产出 final verdict evidence block
- [ ] 明确三选一结论：
  1. **Accept**: diffusion student 通过 gate，替换 reference
  2. **Suspend**: diffusion 扩张暂停，保留 PAdapt baseline，
     diffusion 结果作为论文负结果/对照
  3. **Inconclusive**: 需要超出 V4 scope 的改动才能继续
     （如全面 unfreeze backbone），记录为 future work

---

<!-- CHUNK_PLACEHOLDER_5 -->

## 6. 证据与文档要求

### 6.1 Evidence Block 格式（沿用 V2/V3）

每个实验必须记录：
- run ID / output path
- git commit hash（bug fix commit）
- config snapshot（关键 override 列表）
- seed(s)
- evaluation steps
- primary metrics: nominal_reward, light_v2_reward, hard_reward, done_rate per condition
- delta vs V3-M0 reference（primary gate）
- delta vs PAdapt baseline（secondary metric）
- gate 判定结果：pass / fail
- 一句话结论

### 6.2 Bug Fix 验证记录

V4-M0 必须额外记录：
- bug fix 的 exact diff（3 处修改）
- syntax check 结果
- 修复前后 reference config 的 metrics 对比表

### 6.3 Final Verdict 记录

V4-M3 必须产出：
- V3 + V4 全部候选的汇总表
  （candidate name, hard_delta, gate result, bug status）
- 明确的三选一结论及理由
- 对论文的建议（如何呈现 diffusion 结果）

---

## 7. 退出条件

当满足任一条件时结束 V4：

1. V4-M2 产生通过接受 Gate 的候选 → 结论：Accept
2. V4-M3 触发且 final verdict 完成 → 结论：Suspend 或 Inconclusive
3. 用户选择终止 → 记录当前状态，进入 V4-M3

---

## 8. 硬止损规则

1. V4 总计算预算：最多 8 次训练 run（1 baseline + 4 ablation + 3 candidate）
2. Phase 0 必须完成，不可跳过
3. Phase 1 中 Ablation 1+2 都无改善信号 → Ablation 3 跳过，直接尝试 Ablation 4
4. Ablation 4（unfreeze）失败 → 直接进入 V4-M3
5. Phase 2 中连续 3 个候选初筛 gate 全失败 → 立即触发 V4-M3
6. Bug-fixed baseline 如果 hard delta < -0.10（vs V3-M0）→ 立即触发 V4-M3
7. V4 不允许回退到 V3 的 curriculum 或 residual 方向
8. V4 不允许引入 V3 scope 之外的全新架构方法

---

## 9. 关键文件索引

| 文件 | 用途 |
|------|------|
| `dexscrew/algo/ppo/diffusion_latent_student.py` | bug 修复 + 训练/推理路径 |
| `dexscrew/algo/models/models.py` | tanh 来源确认（line 129） |
| `scripts/screwdriver_student_diffusion_latent.sh` | 训练脚本 |
| `scripts/eval_screwdriver_student_robustness.sh` | 单 seed 评测脚本 |
| `scripts/eval_screwdriver_student_robustness_multiseed.sh` | 多 seed 评测脚本 |
| `docs/plansv3_m0_freeze.md` | V3 冻结参考 checkpoint + metrics |
| `claudediffusion.md` | 诊断报告 |
