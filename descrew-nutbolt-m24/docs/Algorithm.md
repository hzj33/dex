# Algorithm.md（中文）

## 1. 文档目的与范围
本文档用于说明当前仓库在 student 阶段的 4 种算法实现、关键差异、验收指标口径，以及它们在 `XHandHoraScrewDriver` 场景下的表现原因分析，供后续论文撰写直接引用。

当前 4 种 student 算法为：
- `ProprioAdapt`（当前强基线）
- `PureBC`（纯 BC 对照）
- `DiffusionLatentStudent`（latent diffusion 路线）
- `DiffusionActionChunkStudent`（action-chunk diffusion 主线）


## 2. 统一实验前提（四者共享）
四种算法均在同一 teacher-student 主路径下比较：

`XHandHoraScrewDriver -> PPO teacher -> student distillation -> 统一评测`

共享前提：
- 同一 teacher checkpoint 来源：`outputs/XHandHoraScrewDriver_teacher/<cache>/stage1_nn/best_reward_*.pth`
- 同一任务与动作维度：Hora screwdriver，动作维度 12
- 同一训练入口：`train.py` + `train.algo=<AlgoName>`
- 同一 student 输入主干：`obs + proprio_hist`（point cloud 路径保持兼容）


## 3. 四种算法实现细节

### 3.1 ProprioAdapt（当前强基线）
**代码**：`dexscrew/algo/ppo/padapt.py`  
**脚本**：`scripts/screwdriver_student_padapt.sh`

核心机制：
- 用 `adapt_tconv` 从 `proprio_hist` 预测 teacher latent。
- 用 teacher privileged latent 作为监督目标（latent distillation）。
- 同时用 teacher action 做 action BC 监督。
- 主干网络基本冻结，重点训练适配模块。

目标函数：
- `L = L_latent + L_bc`

特点：
- 工程稳定，推理代价低。
- 在当前仓库语境中是“强 student baseline”，不是“弱 BC”。


### 3.2 PureBC（纯 BC 对照）
**代码**：`dexscrew/algo/ppo/pure_bc.py`  
**脚本**：`scripts/screwdriver_student_purebc.sh`

核心机制：
- 保持与 ProprioAdapt 兼容的训练框架与参数冻结策略。
- 去掉 latent distillation，只保留动作监督。

目标函数：
- `L = L_bc`

价值：
- 用于隔离“latent distillation”在当前任务中究竟贡献了多少。
- 文件独立实现，不覆盖 `padapt.py`。


### 3.3 DiffusionLatentStudent（latent diffusion）
**代码**：`dexscrew/algo/ppo/diffusion_latent_student.py`  
**脚本**：`scripts/screwdriver_student_diffusion_latent.sh`

核心机制：
- 将“确定性 latent 回归”替换为“条件扩散 latent 生成”。
- 条件输入为 `proprio_hist`，目标为 teacher latent 分布。
- 采样得到 latent 后再走 actor 输出动作；仍保留 action BC 约束。

目标函数：
- `L = w_diff * L_diffusion_latent + w_bc * L_bc`

特点：
- 比 action diffusion 改动更小、集成风险更低。
- 可表达 latent 多模态，不直接改动作接口。


### 3.4 DiffusionActionChunkStudent（action-chunk diffusion）
**代码**：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`  
**脚本**：
- `scripts/screwdriver_student_diffusion_action_chunk.sh`
- `scripts/screwdriver_student_diffusion_action_chunk_15min_docker.sh`

核心机制：
- 学习短时域 action chunk 分布（例如 `H=8`）。
- 条件输入：当前 `obs + proprio_hist`。
- 目标：teacher 动作片段 `a[t:t+H-1]`。
- 执行：receding horizon（只执行 chunk 第一步，下个时刻重采样）。

当前稳定化改动（已落地）：
- 默认确定性去噪执行（关闭执行时随机噪声）。
- 增强首步动作监督权重（first-action BC）。
- 保留弱全 chunk BC 约束。
- 可选 teacher-mix warmup，减轻冷启动抖动。
- 增加 obs/action 维度断言，防止静默错配。
- 新增可选 rollout 预训练入口（`rollout.pt` -> action-chunk diffusion head），用于先离线预热再在线蒸馏。

目标函数（当前实现）：
- `L = w_diff * L_diff_chunk + w_fa * L_first_action_bc + w_chunk * L_chunk_bc`


## 4. 关键差异总表
| 维度 | ProprioAdapt | PureBC | DiffusionLatentStudent | DiffusionActionChunkStudent |
|---|---|---|---|---|
| 监督目标 | teacher latent + teacher action | teacher action | teacher latent 分布 + teacher action | teacher action chunk 分布 |
| 输出粒度 | 单步动作 | 单步动作 | 单步动作（经 latent 采样） | 短时序动作块（执行首步） |
| 核心新增模块 | 适配器训练 | 无新增，仅损失删减 | latent diffusion head | action-chunk diffusion head |
| 推理复杂度 | 低 | 低 | 中 | 中-高 |
| 多模态建模能力 | 弱-中 | 弱 | 中 | 强 |
| 时序一致性潜力 | 中 | 低 | 中 | 高 |
| 工程稳定性 | 高 | 高 | 中高 | 中（对训练细节敏感） |


## 5. 本阶段统一验收指标（奖励之外）
为对齐 `PLANS.md`，建议统一看以下指标束：
- 回报类：
  - `Current Best`
  - `episode_rewards/step`
- 轨迹/终止类：
  - `episode_lengths/step`
  - `done_rate/frame`
- 优化健康类：
  - `total_loss/frame`
  - `latent_loss/frame` / `bc_loss/frame`
  - `diffusion_loss/frame`
  - `first_action_bc_loss/frame` / `chunk_bc_loss/frame`
- 稳定性类：
  - `Last FPS` 统计
  - 错误关键词扫描（`Traceback` / `RuntimeError` / `NaN` 等）
- 产物完整性：
  - `model_best.ckpt` 是否存在且可追溯

仓库已补充支持：
- student 训练中增加 `done_rate` 与 env 数值字段自动记录（新跑次生效）。
- 统一汇总脚本：`scripts/summarize_student_acceptance.py`


## 6. 当前阶段结果快照（15min 验收）
基于当前阶段验收汇总（同任务、同 teacher、同 15min 预算），`Max Current Best` 参考如下：

| 算法 | Max Current Best | 现阶段结论 |
|---|---:|---|
| ProprioAdapt | 1496.08 | 强基线，稳定可靠 |
| PureBC | 1495.38 | 与 ProprioAdapt 接近，说明短窗下 BC 已较强 |
| DiffusionLatentStudent | 1860.15 | 当前最好，扩散收益已可见 |
| DiffusionActionChunkStudent（修复版） | 1203.64 | 已脱离“卡 0”，但尚未追平前两类 |

说明：
- 该表用于阶段性对比，不代表最终论文结论（仍需多种子与更长预算复验）。
- Action-chunk 初版曾出现长期 0 回报，修复后已有明显抬升，表明路径可行但训练稳定性仍是核心挑战。


## 7. 场景化优劣势与效果原因分析（重点）

### 7.1 ProprioAdapt vs PureBC：为什么在当前结果里很接近
观察到短时验收中二者数值接近，常见原因有：
- teacher action 本身已很强，BC 监督可快速逼近主行为模式；
- 当前 15min 窗口偏短，latent 蒸馏优势可能尚未完全显现；
- 在当前随机化/噪声设定下，动作监督信号密度较高，弱化了 latent 项的边际收益。

结论：
- PureBC 不是更强方法，而是说明“当前窗口下 BC 很能打”。
- 这正好支持你论文里“当前 student 是强基线，不是弱基线”的叙述。


### 7.2 DiffusionLatentStudent：为何当前更容易出成绩
已观察到它在当前设置下可较快拉升，原因通常是：
- 它仍复用成熟 actor 接口，不改动作执行语义；
- 扩散只放在 latent 层，表达能力提升但控制链路改动较小；
- 同时保留 action BC，对策略输出有“收敛锚点”。

结论：
- 这是当前阶段很实用的 diffusion 路线：收益可见、工程风险可控。


### 7.3 DiffusionActionChunkStudent：为何更难、但更有潜力
该路线理论上最贴近 thesis 主线（多模态动作 + 时序一致性），但实现上更敏感：
- 训练目标从单步变成短时序，采样误差会被执行闭环放大；
- 若执行端带随机采样噪声，早期会出现“会训不会控”的卡死现象；
- 若首步动作监督不够，receding horizon 的执行质量会劣化。

这也是我们看到“初版卡 0、修复后显著改善”的主要原因。

结论：
- 在本任务里 action-chunk diffusion 不是“不可行”，而是“需要稳定化设计”；
- 一旦稳定，可直接对应你要写的时序一致性与恢复动作价值点。


### 7.4 四种算法在本任务下的优劣势总表
| 算法 | 主要优势 | 主要劣势 | 在 screwdriver 场景下的表现原因 |
|---|---|---|---|
| ProprioAdapt | 工程成熟、收敛稳、推理便宜；teacher-student 链路已验证 | 多模态表达能力有限；对突发恢复动作建模较弱 | 任务已有较强 teacher 且观测链路固定，latent+BC 足够支撑主行为，故表现稳定 |
| PureBC | 实现最简、训练最快、对照价值高 | 缺少 latent 对齐先验，泛化与鲁棒潜力受限 | 短时间预算内 teacher action 监督信号密度高，能快速拟合主策略，因此分数接近 ProprioAdapt |
| DiffusionLatentStudent | 在不改动作接口前提下提升分布表达；兼顾稳定与收益 | 采样与训练成本高于确定性方法；调参更复杂 | 扩散放在 latent 层，既获得表达增益又避免动作执行语义大改，因此在当前阶段最容易“又稳又涨” |
| DiffusionActionChunkStudent | 最符合 thesis 主线：动作多模态 + 时序一致性 + 恢复动作潜力 | 训练/执行耦合强，早期易不稳定；对损失配比和推理噪声敏感 | 接触密集、时序误差会被闭环放大；若首步约束不足或执行噪声过大，会出现“会拟合但不会控”，修复后才逐步回升 |


## 8. 当前阶段的论文写作建议
建议按以下结构组织“方法对比”章节：
1. 先明确 ProprioAdapt 是强基线（不是弱 BC）。
2. 用 PureBC 解释当前 student 的组成贡献边界。
3. 用 DiffusionLatentStudent 证明 diffusion 在当前架构下可行且有效。
4. 用 DiffusionActionChunkStudent 说明主线价值与工程难点：
   - 失败模式（初版卡 0）
   - 稳定化策略（确定性执行、首步损失、teacher mix）
   - 改进后趋势（由 0 拉升到可用区间）


## 9. 统一命令参考
- ProprioAdapt 15min  
  `scripts/screwdriver_student_padapt_15min_docker.sh 0 42 run_a 900 <cache>`

- PureBC 15min  
  `scripts/screwdriver_student_purebc_15min_docker.sh 0 42 run_a 900 <cache>`

- DiffusionLatent 15min  
  `scripts/screwdriver_student_diffusion_latent_15min_docker.sh 0 42 run_a 900 <cache>`

- DiffusionActionChunk 15min  
  `scripts/screwdriver_student_diffusion_action_chunk_15min_docker.sh 0 42 run_a 900 <cache>`
