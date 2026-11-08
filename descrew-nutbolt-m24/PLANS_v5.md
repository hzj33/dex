# PLANS_v5

## Status

- status: `completed_conclude`
- activated_on: 2026-04-15
- completed_on: 2026-04-15
- final_decision: `conclude_no_single_seed_gate_pass`
- supersedes_for_execution: `PLANS_v4.md`
- 前置结论: V4 Suspend — latent DDPM 在 bug-free 代码上仍不过 gate

---

## 0. 背景与动机

V2-V4 共验证了 7 个 latent DDPM 候选（6 buggy + 1 bug-fixed），
全部在 single-seed hard gate 失败。架构分析表明失败根因不是超参或 bug，
而是 DDPM 迭代去噪在 8 维确定性 latent 空间上的结构性不匹配：
- 迭代去噪引入累积误差
- 噪声预测（predict eps）是间接目标，不如直接预测 latent
- 10 步推理循环在低维空间无优势

V5 引入两种新的蒸馏形式，验证 diffusion 家族方法在灵巧手
低维蒸馏场景下的适用性边界：

| 方法 | 核心思路 | 推理步数 | 训练目标 |
|------|----------|----------|----------|
| DDPM (V2-V4) | 预测噪声，迭代去噪 | 10 步 | eps prediction |
| **Consistency** | 单步映射到 x0 | **1 步** | consistency loss |
| **Flow Matching** | ODE 速度场 | **1 步** | velocity matching |

三者共享同一 frozen backbone + 相同 eval 协议，
构成完整的 diffusion 家族对比实验。

---

## 1. V5 总目标

1. 实现 `ConsistencyLatentStudent` — 单步 consistency distillation
2. 实现 `FlowMatchingLatentStudent` — 单步 ODE flow matching
3. 两者均继承 `ProprioAdapt`，复用 frozen backbone + eval 协议
4. 与 PAdapt / DDPM 在统一 gate 下对比
5. 给出 diffusion 家族在低维灵巧手蒸馏中的完整结论

---

## 2. 范围与边界

### 2.1 纳入范围

- 两个新 student 类的实现（继承 ProprioAdapt）
- train.py / student_eval.py 注册
- 统一评测（nominal / light_v2 / hard, 256 steps）
- 与 V2-V4 历史数据的对比分析

### 2.2 不纳入范围

- 修改 frozen backbone 或 teacher
- 修改 PAdapt 基线
- 修改现有 DiffusionLatentStudent
- 大规模超参搜索（每个方向最多 3 个候选）

---

## 3. 新 Student 架构设计

### 3.1 共享设计

两个新 student 均：
- 继承 `ProprioAdapt`（与 DiffusionLatentStudent 同级）
- 冻结 backbone（`self.model.parameters()` 全部 freeze）
- 仅训练各自的 head 网络
- 复用 `LatentDiffusionHead` 的网络结构（hist_encoder + denoiser）
  但修改输出语义（不再预测 noise）
- 使用连续时间嵌入替代离散 `nn.Embedding`
- 推理时单步前向，无迭代循环
- 共享 loss 辅助项：bc_loss + base_action_anchor_loss

### 3.2 ConsistencyLatentStudent

**文件**: `dexscrew/algo/ppo/consistency_latent_student.py`

**核心思路**: 训练网络 f(x_t, t, c) 满足 consistency 性质：
对任意 t, t'，f(x_t, t, c) = f(x_{t'}, t', c) = x_0

**网络**: `ConsistencyHead`
- 输入: proprio_hist (conditioning), x_t (noisy latent), t (continuous)
- 输出: 直接预测 x_0（不是 noise）
- 时间嵌入: sinusoidal positional encoding（连续 t ∈ [0, 1]）
- 其余结构与 LatentDiffusionHead 相同

**训练 loss**:
```
consistency_loss = ||f(x_{t_n}, t_n, c) - sg(f(x_{t_{n+1}}, t_{n+1}, c))||^2
```
其中 sg = stop_gradient，(t_n, t_{n+1}) 是相邻时间步对。
加上 boundary condition: f(x_0, 0, c) = x_0

**辅助 loss**: bc_loss（student action vs teacher action）

**推理**: 单步 — `latent = f(x_T, T, c)` 其中 x_T ~ N(0, I) 或 zeros

**Config 前缀**: `consistency_*`

### 3.3 FlowMatchingLatentStudent

**文件**: `dexscrew/algo/ppo/flow_matching_latent_student.py`

**核心思路**: 学习速度场 v(x_t, t, c)，
使得 ODE dx/dt = v 将噪声 x_1 传输到目标 x_0

**网络**: `FlowMatchingHead`
- 输入: proprio_hist (conditioning), x_t (interpolated), t (continuous)
- 输出: 预测速度 v（与 latent 同维度）
- 时间嵌入: sinusoidal positional encoding（连续 t ∈ [0, 1]）
- 其余结构与 LatentDiffusionHead 相同

**训练 loss** (Conditional Flow Matching):
```
x_t = (1 - t) * x_0 + t * x_1    # 线性插值，x_1 ~ N(0, I)
v_target = x_1 - x_0              # 目标速度（常数）
flow_loss = ||v_pred - v_target||^2
```

**辅助 loss**: bc_loss（student action vs teacher action）

**推理**: 单步 Euler — `latent = x_1 - v(x_1, 1, c)`
其中 x_1 ~ N(0, I) 或 zeros（deterministic mode）

**Config 前缀**: `flow_*`

---

## 4. 决策指标与 Gate（沿用 V3/V4）

### 4.1 双 Reference 体系

**Primary gate**: delta vs V3-M0 frozen reference（seed 42）
- nominal: 1.675, light_v2: 1.638, hard: 1.505

**Secondary metric**: delta vs PAdapt multiseed baseline
- nominal: 2.168, light_v2: 2.079, hard: 1.838

### 4.2 初筛 Gate（单 seed）

1. `delta_hard >= -0.05`
2. `delta_light_v2 >= -0.08`
3. `delta_done_hard <= +0.0005`

### 4.3 接受 Gate（多 seed 42/43/44）

1. `hard_mean_delta >= 0`
2. `light_v2_mean_delta >= 0`
3. `nominal_mean_delta >= -0.10`

---

## 5. V5 里程碑

### V5-M0：实现 + 冒烟测试

完成条件：
- [ ] 实现 `ConsistencyLatentStudent` 类
- [ ] 实现 `FlowMatchingLatentStudent` 类
- [ ] 在 `train.py` 和 `student_eval.py` 中注册
- [ ] 两者均通过冒烟测试（短时间训练 + 推理不报错）
- [ ] 确认 checkpoint save/load 正常

### V5-M1：Consistency 方向（最多 3 个候选）

**候选 1: consistency_baseline**
- 默认 config：consistency_loss_coef=1.0, bc_loss_coef=1.0
- 训练 15min, seed=42
- 单 seed 评测

**候选 2: consistency_no_bc**（如果候选 1 有信号）
- bc_loss_coef=0.0，纯 consistency loss
- 验证 bc_loss 是否必要

**候选 3: consistency_tuned**（如果候选 1 或 2 有信号）
- 基于最佳候选微调 loss 权重
- 或尝试 2-step 推理（而非 1-step）

止损：3 个候选全部初筛 fail → Consistency 方向冻结

### V5-M2：Flow Matching 方向（最多 3 个候选）

**候选 1: flow_baseline**
- 默认 config：flow_loss_coef=1.0, bc_loss_coef=1.0
- 单步 Euler 推理
- 训练 15min, seed=42

**候选 2: flow_multistep**（如果候选 1 有信号）
- 2-4 步 Euler 推理（训练不变）
- 验证多步推理是否改善

**候选 3: flow_tuned**（如果候选 1 或 2 有信号）
- 基于最佳候选微调 loss 权重

止损：3 个候选全部初筛 fail → Flow Matching 方向冻结

### V5-M3：多 Seed 验证

前提：V5-M1 或 V5-M2 中至少一个候选通过初筛 gate。
- 对通过初筛的候选跑 seed=42,43,44
- 按接受 Gate 判定

### V5-M4：Final Verdict — Diffusion 家族完整结论

汇总 V2-V5 全部实验，产出论文级对比表：

| 方法 | 版本 | 候选数 | 最佳 hard delta | Gate |
|------|------|--------|-----------------|------|
| DDPM latent | V2-V4 | 7 | -0.052 (v3a3) | FAIL |
| Consistency | V5 | ≤3 | TBD | TBD |
| Flow Matching | V5 | ≤3 | TBD | TBD |
| PAdapt | baseline | - | +0.333 | PASS |

三选一结论：
1. **Accept**: 某方法通过 gate → 替换 reference
2. **Partial**: 某方法显著优于 DDPM 但未过 gate → 记录为改进方向
3. **Conclude**: 全部失败 → diffusion 家族在低维蒸馏中不适用，
   PAdapt 确认为最优 student，论文给出完整负结果分析

---

## 6. 实现细节

### 6.1 新文件清单

| 文件 | 说明 |
|------|------|
| `dexscrew/algo/ppo/consistency_latent_student.py` | ~300 行 |
| `dexscrew/algo/ppo/flow_matching_latent_student.py` | ~250 行 |

### 6.2 需修改的文件

| 文件 | 修改内容 |
|------|----------|
| `train.py` | 添加 2 行 import |
| `student_eval.py` | 添加 2 行 import + isinstance 分支 |

### 6.3 网络复用

两个新 Head 均复用 LatentDiffusionHead 的结构：
```python
hist_encoder: Linear(proprio_hist_dim * proprio_dim → 256) → ELU
              → Linear(256 → 256) → ELU
t_embed: SinusoidalPosEmbed(64)  # 替代 nn.Embedding
denoiser: Linear(256 + latent_dim + 64 → 256) → ELU
          → Linear(256 → 256) → ELU → Linear(256 → latent_dim)
```

唯一区别：
- Consistency: 输出语义 = 预测 x_0
- Flow Matching: 输出语义 = 预测速度 v

### 6.4 Config 参数

**Consistency**:
- `consistency_loss_coef` (default 1.0)
- `consistency_boundary_coef` (default 0.5)
- `consistency_num_scales` (default 10)
- `consistency_ema_decay` (default 0.999)

**Flow Matching**:
- `flow_loss_coef` (default 1.0)
- `flow_sigma_min` (default 1e-4)
- `flow_infer_steps` (default 1)

---

## 7. 止损规则

1. 总计算预算：最多 10 次训练 run
2. 每个方向最多 3 个候选，连续 3 个初筛 fail → 方向冻结
3. 两个方向都冻结 → 直接进入 V5-M4
4. V5 不允许修改 frozen backbone
5. V5 不允许回退到 DDPM 方向

---

## 8. 与历史计划的关系

```
V2: 建立 evidence 框架 + 首次 DDPM 验证 → 负结果
V3: DDPM 超参/curriculum/residual 穷举 → 6/6 fail, 止损
V4: Bug fix + 重验证 → 仍然 fail, Suspend
V5: 换形式（Consistency + Flow Matching）→ 最终结论
```

V5 是 diffusion 探索的最后一个 plan。
无论结果如何，V5-M4 将给出 diffusion 家族的完整结论。

---

## 9. 退出条件

1. V5-M3 产生通过接受 Gate 的候选 → Accept
2. V5-M4 完成 final verdict → Conclude 或 Partial
3. 用户选择终止 → 记录当前状态，进入 V5-M4
