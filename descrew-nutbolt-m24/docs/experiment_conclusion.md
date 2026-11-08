# 实验结论（当前阶段）

## 1. 结论目标
本结论文件用于固化本阶段最关键的机制定位结果，避免后续会话重复回溯。

当前要回答的问题：
- `adapt_tconv` 是否是 diffusion 特性？
- 为什么 `adapt_tconv-only` 比放开更多 MLP 参数效果更好？
- 主线接下来是否需要新增第 5 套算法？

## 2. 关键定义澄清
- `adapt_tconv` 不是 diffusion 的特性。
- `adapt_tconv` 属于当前 `ProprioAdapt` student 的时序适配模块（从 `proprio_hist` 预测 teacher latent）。
- diffusion 路线在本项目里是独立 student 实现（如 `DiffusionLatentStudent`、`DiffusionActionChunkStudent`），不是由 `adapt_tconv` 定义。

## 3. 对照实验设置（统一口径）
- teacher checkpoint: `run_a`
- seed: `42`
- window: `15min`（`timeout 900`）
- task/主路径：`XHandHoraScrewDriver`
- 指标：训练日志中的 `Max Current Best`

## 4. 定位结果（P2/M2）
| 训练范围 | Max Current Best |
|---|---:|
| `adapt_tconv`（baseline） | `1496.08` |
| `adapt_tconv + mu` | `69.11` |
| `adapt_tconv + actor_mlp` | `7.06` |
| `adapt_tconv + actor_mlp + mu` | `32.31` |

数据来源：
- `docs/stage_acceptance_summary.md`
- `outputs/XHandHoraScrewDriver_student_padapt_trainrange/*/train_900s.log`

## 5. 当前可落地结论
1. 放开 `actor_mlp` 是主要退化源（退化幅度最大）。
2. 仅放开 `mu` 也会退化，但程度小于放开 `actor_mlp`。
3. 现阶段应保持 `adapt_tconv-only` 作为稳定 baseline，不把 train-range ablation 纳入默认训练路径。

说明：上述第 1/2 点属于从同口径对照结果得出的工程结论，不等同于完整机理证明。

## 6. 主线决策（面向 PLANS）
- 不新增第 5 套算法作为当前阶段必做项。
- 继续沿既有 diffusion 主线推进：在已跑通 diffusion 基础上进入 `P5`（效率或鲁棒性二选一优化轴）。
- 若后续出现持续阻塞，按 `PLANS` 保底策略处理（latent/residual 路线），而不是盲目扩展算法数量。

## 7. 下一步执行建议（单一推荐）
进入 diffusion 的 `P5` 贡献轴，优先做“鲁棒性最小评测包”：
- 对 `ProprioAdapt` baseline 与 diffusion 主实现做同口径扰动评测（噪声/随机化强度小步扫描）。
- 先做最小可复现 1 组对照，再决定是否扩展到完整鲁棒性矩阵。

## 8. 最新进展补充（2026-03-21）
在不改算法结构、仅调 action-chunk diffusion 训练超参数的前提下，15min 同口径结果出现连续提升：

| 配置 | Max Current Best |
|---|---:|
| action-chunk 修复版 | `1203.64` |
| tune-v1 (`teacher_mix=300k`, `first_action_bc=2.0`, `chunk_bc=0.2`) | `1280.19` |
| tune-v2 (`teacher_mix=500k`, `first_action_bc=3.0`, `chunk_bc=0.2`) | `1502.21` |

阶段结论更新：
- 已达到“出现较好结果”条件（`1502.21` 超过 `ProprioAdapt=1496.08`）。
- 后续优先从“继续提分”转到“贡献轴验证”（鲁棒性或效率），避免无限调参。

## 9. P5 鲁棒性最小评测补充
固定步评测（`steps=256`）后，出现了和训练曲线不同的关键信号：

- `ProprioAdapt`：nominal `avg_reward=1.918988`
- `DiffusionLatentStudent`：nominal `avg_reward=1.489716`
- `DiffusionActionChunk` 修复版：nominal `avg_reward=-1.873523`
- `DiffusionActionChunk` tune-v2：nominal `avg_reward=-1.248181`

解释与决策：
1. action-chunk 的训练分数提升是真实存在的，但纯 student 固定步评测仍未达到可替代 baseline 的水平。
2. action-chunk tune-v2 相比修复版明显更好（负回报绝对值降低），说明方向有效但未收敛。
3. 当前“稳态 diffusion 路径”应优先落在 latent diffusion；action-chunk 继续作为主线探索分支推进偏差修复。

## 10. P5 主线优化结论（latent 训练侧扰动注入）
针对“latent 在 hard 扰动下明显弱于 ProprioAdapt”的现象，我们做了一个最小优化验证：

- 不改模型结构，不新增算法；
- 仅在 latent 训练阶段注入轻外力扰动（`forceScale=0.5`, `randomForceProbScalar=0.1`）；
- 再用同一 `steps=256` 口径评测 nominal/light/hard。

结果（与 latent baseline 对比）：
- nominal：`1.489716 -> 1.432408`（小幅回落）
- light：`1.327045 -> 1.630288`（显著提升）
- hard：`0.950702 -> 1.344396`（显著提升）
- done_rate 在三档均改善。

阶段性结论：
1. 这支持了一个项目内可解释机制：latent 的差距并非纯推理问题，而是“训练分布中扰动覆盖不足”。
2. 在灵巧手接触任务里，外力/接触扰动暴露对 diffusion student 的恢复动作与时序稳定性很关键。
3. 当前主线应继续沿 latent 做“受控鲁棒性优化”，目标是缩小与 ProprioAdapt 的 hard 档差距，同时约束 nominal 不明显退化。
