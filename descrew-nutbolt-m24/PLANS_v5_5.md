# PLANS_v5_5

## Status

- status: `completed_accept`
- proposed_on: 2026-04-15
- activated_on: 2026-04-15
- completed_on: 2026-04-15
- final_decision: `accept_consistency_boundary_bc_tuned`
- based_on: `PLANS_v5.md (completed_conclude)`
- execution_intent: `consistency_hard_done_targeted_optimization`
- supersedes_for_execution: `PLANS_v5.md`

---

## 0. 背景

`PLANS_v5` 已完成并收敛为 `Conclude`。  
当前 diffusion 家族里，Consistency 的 reward 最强，但卡在 hard done gate。

已知最佳候选（seed=42, steps=256）：
- consistency_baseline: nominal `2.201198`, light_v2 `1.936892`, hard `1.714672`, hard_done `0.002686`
- 相对 V3-M0（hard done `0.001872`）：`delta_done_hard=+0.000814`（超门槛 `+0.0005`）

结论：离单 seed 初筛 gate 只差一项（hard done），具备小范围继续优化价值。

**关键数据点**：
- consistency_baseline 的 reward 三项均为正 delta（nominal +0.526, light_v2 +0.299, hard +0.210）
- 唯一失败项是 hard_done，说明 latent 质量已足够，问题在 action 稳定性
- consistency_no_bc 的 hard_done 反而更低（0.002116），说明 bc_loss 权重影响 done 行为
- 2-step 推理使 hard_done 恶化到 0.003337，排除推理步数作为优化方向

---

## 1. V5.5 目标

1. 在不改 teacher/backbone 的前提下，针对 Consistency 做小范围稳定性优化。
2. 以最小训练预算优先修复 `hard_done` 超标问题。
3. 若出现单 seed gate 通过候选，进入 multiseed 验证；否则快速止损并收敛。

---

## 2. 范围与边界

### 2.1 纳入范围

- 仅 `ConsistencyLatentStudent` 路线。
- 仅小规模参数优化与轻量 student-side 稳定性改动（可回滚）。
- 统一评测协议：`nominal + light_v2 + hard`, `steps=256`。

### 2.2 不纳入范围

- 不改 teacher / frozen backbone / PAdapt baseline。
- 不做大重构、不引入跨仓库依赖。
- 不重启 DDPM 或 action-chunk 主线。
- 不增加推理步数（V5 已验证 2-step 使 hard_done 恶化）。

---

## 3. 核心假设（面向 hard_done）

- H1: `action_l2` 约束缺失（当前 coef=0.0），策略在 hard 扰动下动作幅度偏大，触发更多 done。
  - 证据：consistency_baseline 的 action_mse=0.144926，高于 PAdapt 水平
- H2: `boundary_coef=0.5` 偏低，latent 在 hard 条件下偏离 teacher 过多，间接导致动作不稳定。
  - 证据：latent_mse=0.091366，有改善空间
- H3: `base_action_anchor_coef=0.0`，缺少对 base policy 的锚定，hard 扰动下 student 动作漂移无约束。
  - 证据：V3/V4 中 anchor_coef=0.03 的配置在 reward 上表现合理

---

## 4. 指标与 Gate

沿用 `PLANS_v5` gate（主判定不变）：

### 4.1 Primary gate（vs V3-M0 frozen）

- `delta_hard >= -0.05`
- `delta_light_v2 >= -0.08`
- `delta_done_hard <= +0.0005`

V3-M0 Reference（seed 42）：
- nominal: 1.675112, light_v2: 1.638266, hard: 1.504904
- hard_done: 0.001872

**hard_done 绝对阈值**: `<= 0.002372`（即 0.001872 + 0.0005）

### 4.2 Secondary metric（报告层）

- delta vs `PAdapt`（nominal/light_v2/hard）

### 4.3 Anti-regression Guardrails（vs consistency_baseline）

基线锚点（Consistency baseline, seed42, steps256）：
- reward: nominal `2.201198`, light_v2 `1.936892`, hard `1.714672`
- done: nominal `0.000651`, light_v2 `0.001628`, hard `0.002686`

候选要被判定为”可接受改进”，除 Primary gate 外，还需满足：

1. reward 防退化（vs consistency_baseline）
   - `delta_hard_reward >= -0.15`（允许小幅 reward 换 done 改善）
   - `delta_light_v2_reward >= -0.10`
   - `delta_nominal_reward >= -0.25`

2. done 防退化（非 hard 条件不能恶化）
   - `done_nominal <= 0.001151`（baseline + 0.0005）
   - `done_light_v2 <= 0.002128`（baseline + 0.0005）

3. hard_done 突破要求（核心）
   - 必须满足 Primary gate 的 `delta_done_hard <= +0.0005`
   - 等价于绝对阈值：`hard_done <= 0.002372`

说明：
- 若 hard_done 改善但 reward 明显退化（触发第1条），仍判 FAIL。
- 若 reward 提升但 hard_done 未达阈值，仍判 FAIL。
- reward 防退化阈值比原 draft 略宽松（-0.15 vs -0.10），因为核心目标是降 done，
  适度 reward 让步是可接受的 trade-off。

---

## 5. 里程碑

### V5.5-M0：基线锁定与运行协议校验

完成条件：
- [ ] 锁定 baseline artifacts（run dir / ckpt hash / eval logs）
- [ ] 复核 hard_done 超标幅度（`+0.000814`）
- [ ] 确认统一脚本与覆盖参数生效（避免无效候选）

---

### V5.5-M1：训练候选（最多 3 个，seed=42，15min）

> 注：移除了原 draft 的 eval-only 探针阶段。
> 理由：V5 已验证 infer_steps=2 使 hard_done 恶化，
> stochastic_infer 默认已是 False，eval-only 探针无新信息可提供。
> 直接进入训练候选，节省时间。

只允许小改动，优先”降 done 不伤 reward”。

**候选 A: action_l2_stable**（验证 H1）
- `+train.ppo.consistency_action_l2_coef=1e-3`
- `+train.ppo.bc_loss_coef=1.0`（保持不变）
- 目标：直接压动作幅度，降低 done。最小改动，最低风险。

**候选 B: anchor_l2_combo**（验证 H1 + H3）
- `+train.ppo.consistency_action_l2_coef=5e-4`
- `+train.ppo.base_action_anchor_coef=0.03`
- `+train.ppo.bc_loss_coef=1.0`（保持不变）
- 目标：action L2 + base policy 锚定双管齐下。

**候选 C: boundary_bc_tuned**（验证 H2）
- `+train.ppo.consistency_boundary_coef=0.8`
- `+train.ppo.consistency_num_scales=16`
- `+train.ppo.bc_loss_coef=1.2`
- 目标：提高 latent 稳定性 + 加强行为监督。

**候选优先级**: A > B > C
- A 是最小改动（单变量），如果 A 通过 gate 则 B/C 可跳过
- B 是组合策略，如果 A 不够则 B 补充锚定
- C 改动最大（3 个参数），作为兜底

每个候选完成条件：
- [ ] train 完成（无 error）
- [ ] 三条件 eval 完成
- [ ] 计算 primary gate
- [ ] 计算 anti-regression guardrails（4.3 全项）
- [ ] 记录 ckpt hash + eval 日志路径

止损规则：
- 3 个训练候选均 gate fail → 结束 V5.5，进入 M3 Conclude。
- 即使 primary gate 通过，只要 4.3 任一项 fail，仍不进入 M2。
- 如果候选 A 直接通过 gate + anti-regression，可跳过 B/C 直接进 M2。

---

### V5.5-M2：Multiseed 验证（仅在 M1 有初筛通过时触发）

- top-1 候选跑 `seed=42,43,44`
- 接受 gate（沿用 V5）：
  - `hard_mean_delta >= 0`
  - `light_v2_mean_delta >= 0`
  - `nominal_mean_delta >= -0.10`
- 同时检查 multiseed hard_done 均值是否 <= 0.002372

---

### V5.5-M3：Final Verdict

三选一：
1. `Accept`: 通过 multiseed gate + 全部 anti-regression 要求
2. `Partial`: 单 seed hard_done 达阈值但 multiseed 未通过，或存在轻微 anti-regression 触发
3. `Conclude`: 无候选通过初筛，或均触发防退化约束

输出：
- `docs/plansv5_5_final_verdict.md`
- 更新 `docs/stage_acceptance_summary.md`
- 更新 `docs/session_handoff_v2.md`

---

## 6. 执行预算与节奏

- 训练预算上限：`<= 3` runs（候选 A/B/C）
- 评测：每候选 3 条件 (`nominal/light_v2/hard`)
- 多 seed：最多 1 候选 × 3 seeds = 3 eval runs
- 连续推进：同一里程碑内不做逐候选停顿确认
- 如果候选 A 直接通过，总预算仅 1 train + 3 eval + 3 multiseed eval

---

## 7. 复现与记录规范

每个候选必须记录：
- train run_dir
- `model_best.ckpt` hash
- 三条件 eval log 路径
- 关键参数覆盖项
- 与 V3-M0 reference 的 delta（reward + done）
- 与 consistency_baseline 的 delta（anti-regression）
- 与 PAdapt 的 delta（secondary metric）

统一写入：
- `docs/session_handoff_v2.md`（会话级）
- `docs/plansv5_5_final_verdict.md`（结论级）

---

## 8. 激活前检查清单

- [ ] 用户/治理确认激活 `PLANS_v5_5`
- [ ] baseline reference 与 gate 口径冻结
- [ ] GPU/脚本/容器可用
- [ ] 明确本轮只做 Consistency，不扩展到新主线
