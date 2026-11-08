# PLANS_v3.md

## Status

- status: `active`
- activated_on: `2026-04-14`
- activated_by: `user confirmation in chat`
- supersedes_for_execution: `PLANS_v2.md` (optimization stage only; historical evidence docs remain valid)

## 0. 生效说明

本文件是当前 active plan。  
目的：在 `Plan v2` 多轮近参考微调持续被 reject 后，给出“可执行、可止损、可转向”的下一阶段路线。  

---

## 1. 为什么需要 v3

### 1.1 当前诊断（来自 v2 后段执行证据）

- 近参考单变量 full-budget 探针多数能恢复高训练信号（`Current Best` 高），但在统一 gate 下仍失败。  
- 常见模式：`nominal` 有提升，但 `light_v2/hard` 回退，最终被 reject。  
- 说明问题不再主要是“训练跑不起来”，而是“优化方向与 robust 目标不匹配”。  

### 1.2 v3 核心判断

继续在 v2 末段同类微调空间内加密搜索，边际收益很低。  
v3 需要从“局部调参”切到“假设驱动的小方向切换”，并附带硬止损。

---

## 2. v3 总目标

在不做大重构的前提下，找到一个能在统一协议下通过 keep/drop gate 的候选，或者用可追溯证据确认 diffusion 进一步扩张应暂停。

具体目标：

1. 优先提升 `hard` 与 `light_v2`，不再以 `nominal` 提升作为主要推进信号。  
2. 用更早期的多种子与硬门限，减少“单 seed 假阳性/假阴性”。  
3. 控制计算预算，避免重复 reject 循环。  

---

## 3. v3 范围与边界

### 3.1 纳入范围

- 仍在当前仓库、当前 Hora teacher-student 主链路内。  
- 允许的改动类型：
  - 训练分布/课程策略（已有参数与轻量开关）
  - loss 权重与选择策略（不改大架构）
  - residual fallback 的小规模再验证
  - 评测与证据流程强化

### 3.2 不纳入范围

- 新的大模型架构分支（如大规模 action diffusion 重构、offline RL 新管线）。  
- 跨仓库重依赖引入。  
- 以工程重构为目标的大改。  

---

## 4. v3 决策指标与 Gate（更新版）

## 4.1 统一评测协议（保持）

- 条件：`nominal`, `light_v2`, `hard`
- steps：`256`
- 默认先 `seed=42`，通过初筛后再 `seed=42,43,44`

### 4.2 初筛 Gate（单 seed）

候选进入多 seed 的必要条件：

1. `hard` 不低于 reference 超过 `0.05`（即 `delta_hard >= -0.05`）  
2. `light_v2` 不低于 reference 超过 `0.08`（即 `delta_light >= -0.08`）  
3. 不出现明显 done-rate 恶化（`delta_done_hard <= +0.0005`）

未满足则直接 reject，不进多 seed。

### 4.3 接受 Gate（多 seed）

候选可替换 reference 的必要条件：

1. `hard_mean_delta >= 0`
2. `light_v2_mean_delta >= 0`
3. `nominal_mean_delta >= -0.10`
4. 关键对齐指标不出现系统性恶化（`action_mse_to_teacher` 不显著上升）

### 4.4 止损规则

- 同一方向最多 `3` 个候选。  
- 连续 `3` 个候选在初筛 Gate 全失败，则该方向冻结。  
- 两个方向都冻结后，触发 v3 收敛评审（进入“暂停 diffusion 扩张”候选结论）。

---

## 5. v3 主线结构

### Mainline A：Robust-First Latent Tuning（首选）

目标：优先修复 `hard/light_v2`，降低“nominal 独好”模式。

执行策略：

1. 先做“训练分布与课程”类候选（不改核心结构）。
2. 再做“loss 组合”类候选（权重/选择策略）。
3. 每个候选必须走 `single-seed gate -> multiseed gate`。

### Mainline B：Residual-Corrective Re-entry（保底）

触发条件：

- Mainline A 被止损冻结，且没有可接受候选。

执行策略：

1. 只做小规模 residual 稳定化回归验证。  
2. 仍按 hard-first gate 判定。  
3. 不扩展成大架构研究。

---

## 6. v3 里程碑

### V3-M0：治理切换准备

完成条件：

- v3 文档确认为执行主计划（或明确为试运行计划）。  
- reference checkpoint 与统一评测脚本冻结一次。  

### V3-M1：方向 A 首批候选（最多 3 个）

完成条件：

- 完成 3 个以内候选的 full-budget + single-seed gate。  
- 至少 1 个候选进入 multiseed，或方向 A 触发止损冻结。  

### V3-M2：方向 A 多 seed 决策

完成条件：

- 对入围候选完成 `42/43/44` 对比。  
- 明确是否产生可接受替代候选。  

### V3-M3：方向 B（若触发）

完成条件：

- residual 小规模回归验证完成（最多 3 候选）。  
- 给出是否继续 diffusion 扩张的证据结论。  

### V3-M4：阶段收敛

完成条件：

- 三选一结论明确：
  1. 发现可接受 diffusion 候选并替换 reference  
  2. 暂不替换，保留 baseline-first  
  3. 暂停 diffusion 扩张，转论文收敛/对照结论

---

## 7. v3 首批实验包（建议）

说明：以下是“方向级别提案”，执行时按当前代码可用开关细化为具体命令。

### Pack A（训练分布/课程）

候选上限 3：

1. 启用 obs-noise curriculum（已有 `diffusion_obs_noise_curriculum*` 参数）  
2. curriculum 模式改为 staged/staged_hold，目标对齐 `light_v2` 噪声档  
3. 在不改变评测协议前提下，做一次更平滑的 train-time 扰动过渡

### Pack B（loss 组合）

候选上限 3：

1. 围绕当前 reference 的 `bc_loss_coef` 对称探针（先 `0.95`）  
2. tail 相关项只做“单变量且双侧”探针，不再单边盲扫  
3. 所有候选必须先过 single-seed hard-first gate

---

## 8. 证据与文档要求

沿用 v2 evidence block，额外增加：

1. 每个候选必须记录“所属方向/假设编号”。  
2. 记录是否触发方向止损计数。  
3. 在 `docs/session_handoff_v2.md` 明确写出：
   - 为什么继续该方向
   - 为什么冻结该方向

---

## 9. 退出条件

当满足任一条件时结束 v3：

1. 产生通过接受 Gate 的新候选。  
2. Mainline A/B 均触发止损冻结且无可接受候选。  
3. 用户选择停止算法扩张，进入论文收敛阶段。
