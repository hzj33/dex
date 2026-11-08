# PLANS_v6

## Status

- status: `completed_conclude`
- proposed_on: 2026-04-16
- activated_on: 2026-04-16
- completed_on: 2026-04-16
- final_decision: `conclude_consistency_not_surpass_padapt`
- based_on: `PLANS_v5_5.md (completed_accept)`
- execution_intent: `consistency_surpass_padapt`

---

## 0. 背景与现状

V5.5 的 `boundary_bc_tuned` 候选通过了全部 gate，
Consistency 首次成为可接受的 diffusion student。
现在目标从"通过 gate"升级为"全面超越 PAdapt baseline"。

### 当前最佳 Consistency（multiseed mean, seeds 42/43/44）

| 条件 | Reward | Done |
|------|--------|------|
| nominal | 2.336284 | 0.000949 |
| light_v2 | 2.044725 | 0.001194 |
| hard | 1.714471 | 0.001601 |

### PAdapt Baseline（multiseed mean）

| 条件 | Reward | Done |
|------|--------|------|
| nominal | 2.167820 ± 0.183 | 0.001302 ± 0.000115 |
| light_v2 | 2.079074 ± 0.103 | 0.001221 ± 0.000176 |
| hard | 1.838225 ± 0.105 | 0.001411 ± 0.000138 |

### 差距分析

| 指标 | Delta (Consistency - PAdapt) | 状态 |
|------|------------------------------|------|
| nominal reward | **+0.168** | 已超越 |
| nominal done | **-0.000353** | 已超越 |
| light_v2 reward | **-0.034** | 接近，差距小 |
| light_v2 done | **-0.000027** | 基本持平 |
| hard reward | **-0.124** | 主要差距 |
| hard done | **+0.000190** | 需改善 |

**结论**：nominal 已赢，light_v2 接近持平，hard 是唯一显著短板。

### V5.5 Accepted Config（作为 V6 起点）

```
consistency_boundary_coef=0.8
consistency_num_scales=16
bc_loss_coef=1.2
consistency_loss_coef=1.0（默认）
consistency_hidden_dim=256（默认）
consistency_t_dim=64（默认）
consistency_lr=3e-4（默认）
base_action_anchor_coef=0.0（默认）
consistency_action_l2_coef=0.0（默认）
consistency_infer_steps=1（默认）
```

---

## 1. V6 目标

在 V5.5 accepted config 基础上，使 Consistency student 的 multiseed 均值
在全部 3 个条件上达到或超过 PAdapt baseline。

**量化目标**（multiseed mean, seeds 42/43/44）：

| 指标 | 目标值 | 当前值 | 需改善 |
|------|--------|--------|--------|
| nominal reward | >= 2.168 | 2.336 | 已达标 |
| light_v2 reward | >= 2.079 | 2.045 | +0.034 |
| hard reward | >= 1.838 | 1.714 | +0.124 |
| hard done | <= 0.001411 | 0.001601 | -0.000190 |
| light_v2 done | <= 0.001221 | 0.001194 | 已达标 |
| nominal done | <= 0.001302 | 0.000949 | 已达标 |

**核心挑战**：hard reward 差距 0.124 是最大瓶颈。

---

## 2. 验收标准

### 2.1 最终验收（Accept）

候选必须同时满足以下全部条件（multiseed mean）：

**Reward 超越 PAdapt**：
- `nominal_mean >= 2.168`（PAdapt nominal mean）
- `light_v2_mean >= 2.079`（PAdapt light_v2 mean）
- `hard_mean >= 1.838`（PAdapt hard mean）

**Done 不劣于 PAdapt**：
- `hard_done_mean <= 0.001550`（PAdapt hard_done + 1σ 容差）
- `light_v2_done_mean <= 0.001400`（PAdapt light_v2_done + 1σ 容差）
- `nominal_done_mean <= 0.001420`（PAdapt nominal_done + 1σ 容差）

**同时满足 V3-M0 primary gate**（向后兼容）：
- `delta_hard >= -0.05`（vs V3-M0）
- `delta_light_v2 >= -0.08`（vs V3-M0）
- `delta_done_hard <= +0.0005`（vs V3-M0）

### 2.2 部分验收（Partial Accept）

如果无法全面超越，但满足以下条件，记录为 Partial：
- 至少 2/3 条件的 reward 超越 PAdapt
- 剩余 1 条件的 reward delta 在 PAdapt 1σ 范围内
- 全部 done 指标不劣于 PAdapt + 1σ

### 2.3 Anti-regression（vs V5.5 accepted）

任何候选不得在已达标指标上显著退化：
- `nominal_reward >= 2.100`（不低于 PAdapt 水平）
- `light_v2_reward >= 1.900`（V5.5 accepted - 0.145）
- `hard_done <= 0.002372`（V3-M0 gate 绝对阈值）

---

## 3. 优化方向分析

### 3.1 为什么 hard 条件差距最大？

Hard 条件施加了更强的扰动：
- `obs_noise_e_scale=0.05`（nominal 无噪声）
- `obs_noise_t_scale=0.025`
- `forceScale=1.5`（nominal 为 1.0）
- `randomForceProbScalar=0.3`

Consistency 在 hard 下的问题：
1. **latent 预测在强噪声下不够鲁棒** — 单步 x0 预测对输入噪声敏感
2. **网络容量可能不足** — 2 层 256-dim MLP 可能无法学到足够复杂的映射
3. **训练时未见过 hard 级别的噪声** — 训练数据来自 nominal 环境
4. **bc_loss 在 hard 条件下的 teacher action 本身可能有偏差**

### 3.2 可优化的维度

| 维度 | 当前值 | 优化方向 | 预期影响 | 风险 |
|------|--------|----------|----------|------|
| 网络容量 | 256-dim, 2层 | 增大 hidden_dim 或加深 | hard reward ↑ | 过拟合 |
| 训练时长 | 15min | 延长到 30min/1h | 全指标 ↑ | 收益递减 |
| 学习率 | 3e-4 | 调低到 1e-4 | 稳定性 ↑ | 收敛慢 |
| num_scales | 16 | 增加到 24/32 | 一致性 ↑ | 计算量 |
| boundary_coef | 0.8 | 微调 0.6-1.0 | latent 稳定性 | trade-off |
| bc_loss_coef | 1.2 | 微调 1.0-1.5 | action 精度 | trade-off |
| action_l2_coef | 0.0 | 小量 1e-4~5e-4 | done ↓ | reward ↓ |
| 推理步数 | 1 | 2 步 | latent 精度 ↑ | done 可能 ↑ |
| 训练噪声注入 | 无 | 加入 obs noise curriculum | hard 鲁棒性 ↑ | 实现改动 |

---

## 4. 范围与边界

### 4.1 纳入范围

- 仅 `ConsistencyLatentStudent` 路线
- 超参调优（loss 权重、网络容量、训练时长、学习率）
- 轻量代码改动（如训练时噪声注入，需可回滚）
- 统一评测协议不变

### 4.2 不纳入范围

- 不改 teacher / frozen backbone
- 不引入新 student 类型
- 不修改 eval 协议或 gate 定义
- 不做大规模架构重写

---

## 5. 里程碑

### V6-M0：基线锁定

完成条件：
- [ ] 锁定 V5.5 accepted config + checkpoint 作为 V6 baseline
- [ ] 确认 PAdapt multiseed metrics 作为超越目标
- [ ] 确认训练/评测脚本可用

V6 Baseline（V5.5 accepted, multiseed mean）：
- nominal: 2.336 / done 0.000949
- light_v2: 2.045 / done 0.001194
- hard: 1.714 / done 0.001601

---

### V6-M1：低成本探针（最多 3 个，seed=42，15min）

目标：快速定位最有效的优化维度，不做大改动。

**探针 1: capacity_boost**
- `+train.ppo.consistency_hidden_dim=512`
- 其余沿用 V5.5 accepted config
- 假设：网络容量不足是 hard 条件下的瓶颈
- 预期：hard reward ↑，其他指标不退化

**探针 2: longer_train**
- 训练时长从 15min 延长到 30min（`max_training_time=1800`）
- 其余沿用 V5.5 accepted config
- 假设：15min 训练不充分，更长训练可改善所有指标
- 预期：全指标小幅提升

**探针 3: lr_schedule**
- `+train.ppo.consistency_lr=1e-4`
- 其余沿用 V5.5 accepted config
- 假设：较低学习率可提高收敛精度
- 预期：latent MSE ↓，hard reward ↑

每个探针完成条件：
- [ ] 训练完成
- [ ] 三条件 eval（seed=42）
- [ ] 记录 vs V6 baseline 和 vs PAdapt 的 delta

判断规则：
- 选出 hard reward 改善最大且不触发 anti-regression 的维度
- 如果多个维度有效，M2 中组合使用
- 如果全部无效（hard reward delta < +0.02），直接进入 M3 代码改动

---

### V6-M2：组合优化候选（最多 3 个，seed=42）

基于 M1 结果，组合最有效的维度。

**候选策略**（根据 M1 结果动态决定，以下为预设模板）：

**候选 A: best_combo**
- 组合 M1 中 hard reward 改善最大的 2-3 个维度
- 例如：hidden_dim=512 + 30min 训练 + lr=1e-4

**候选 B: best_combo + fine_tune**
- 在候选 A 基础上微调 loss 权重
- 例如：bc_loss_coef=1.3 或 boundary_coef=0.9

**候选 C: best_combo + action_reg**
- 在候选 A 基础上加入轻量 action 正则
- `consistency_action_l2_coef=1e-4`（比 V5.5 候选 A 的 1e-3 更保守）
- 目标：在不损失 reward 的前提下进一步降低 hard_done

每个候选完成条件：
- [ ] 训练完成
- [ ] 三条件 eval（seed=42）
- [ ] 计算 vs PAdapt delta
- [ ] 检查 anti-regression

止损：3 个候选的 hard reward 均未达到 1.780（PAdapt - 1σ）→ 进入 M3

---

### V6-M3：代码级优化（可选，仅在 M1/M2 不足时触发）

如果纯超参调优无法弥合 hard reward 差距，尝试轻量代码改动。

**方向 A: 训练时噪声注入**
- 在训练 loop 中对 `input_dict["obs"]` 注入与 hard 条件同级的噪声
- 复用 DiffusionLatentStudent 中已有的 `obs_noise_curriculum` 机制
- 让 consistency model 在训练时就见过 hard 级别的扰动
- 代码改动量：~20 行（从 diffusion_latent_student.py 移植）

**方向 B: 2-step 推理 + 训练对齐**
- 推理改为 2 步，同时在训练时也用 2-step consistency loss
- 当前 2-step 推理使 done 恶化，可能是因为训练时只用 1-step
- 代码改动量：~30 行

**方向 C: EMA teacher**
- 启用 `consistency_ema_decay`（当前 reserved 未实现）
- 用 EMA 版本的 consistency model 作为 pred_lo_target 的来源
- 提高训练稳定性，可能改善 hard 条件表现
- 代码改动量：~40 行

每个方向最多 1 个候选（seed=42），止损同 M2。

---

### V6-M4：Multiseed 验证

前提：M1-M3 中至少一个候选的 single-seed 结果满足：
- hard reward >= 1.780（PAdapt hard - 1σ）
- 全部 anti-regression 通过

执行：
- [ ] top-1 候选跑 seed=42,43,44
- [ ] 计算 multiseed mean
- [ ] 按验收标准（Section 2）判定

---

### V6-M5：Final Verdict

三选一：
1. **Accept**: multiseed 全面超越 PAdapt → Consistency 替代 PAdapt 成为主 student
2. **Partial Accept**: 2/3 条件超越，1 条件在 1σ 内 → 记录为竞争性替代方案
3. **Conclude**: 无法超越 → Consistency 确认为次优 student，PAdapt 保持主线

输出：
- `docs/plansv6_final_verdict.md`
- 更新 `docs/stage_acceptance_summary.md`
- 更新 `docs/session_handoff_v2.md`
- 论文对比表更新

---

## 6. 止损规则

1. 总训练预算：最多 12 次 run（M1: 3 + M2: 3 + M3: 3 + multiseed: 3）
2. M1 全部探针 hard reward delta < +0.02 → 跳过 M2，直接进 M3
3. M2 全部候选 hard reward < 1.780 → 进入 M3
4. M3 全部方向 hard reward < 1.780 → 进入 M5 Conclude
5. 任何候选 nominal reward < 2.100 → 判定 anti-regression fail
6. 任何候选 hard_done > 0.002372 → 判定 V3-M0 gate fail

---

## 7. 执行节奏

- M1 探针可并行训练（3 个独立 config）
- M2 候选依赖 M1 结果，串行决策
- M3 仅在 M2 不足时触发
- M4 multiseed 仅对 top-1 候选执行
- 连续推进，同一里程碑内不做逐候选停顿

---

## 8. 记录规范

每个候选必须记录：
- train run_dir + checkpoint hash
- 三条件 eval metrics（reward + done）
- vs V6 baseline delta
- vs PAdapt delta
- vs V3-M0 delta（primary gate）
- 关键参数覆盖项

---

## 9. 关键文件索引

| 文件 | 用途 |
|------|------|
| `dexscrew/algo/ppo/consistency_latent_student.py` | Consistency 实现 |
| `dexscrew/algo/ppo/padapt.py` | PAdapt baseline |
| `dexscrew/algo/models/models.py` | 共享 backbone |
| `train.py` | 训练入口 |
| `student_eval.py` | 评测入口 |
| `scripts/eval_screwdriver_student_robustness.sh` | 评测脚本 |
| `docs/plansv5_5_final_verdict.md` | V5.5 结果 |
| `docs/stage_acceptance_summary.md` | 全局 metrics |

---

## 10. 激活前检查清单

- [ ] 用户确认激活
- [ ] V5.5 accepted checkpoint 可用
- [ ] PAdapt multiseed metrics 确认
- [ ] GPU/脚本/容器可用
