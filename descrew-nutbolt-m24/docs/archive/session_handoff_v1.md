# Session Handoff Summary（2026-03-20）

## 1. 当前阶段一句话
当前主线已从“仅能跑 teacher/student”推进到“基线可比 + diffusion 可跑 + teacher rollout 可采集且可被 diffusion 消费 + P2 train-range ablation 已具备可跑入口”的状态，正处于 `PLANS.md` 的 `P2/M2` 与 `P4/M4` 并行收敛阶段。

## 2. rollout 的意义（你问的核心）
`rollout` 在这里不是替代原 teacher->student 两阶段流程，而是补齐“数据闭环”能力：

- 原流程：teacher 训练出 `.pth`，student 在线蒸馏 teacher。
- rollout 新增能力：把 teacher 在环境中的时序行为导出成可复用数据（`.pt`），用于：
  - 离线预训练/热启动 diffusion student（减少冷启动不稳定）
  - 固定数据口径做可复现实验（同一数据、不同算法）
  - 后续效率/鲁棒性研究的统一输入基线

简化理解：`.pth` 是“模型参数快照”，rollout `.pt` 是“可学习的行为轨迹数据”。

## 3. 已完成内容（按 PLANS 里程碑）
### M0 主路径冻结
- Hora canonical path、teacher/student/vis 入口已固定并可运行。
- 可视化入口的参数可靠性问题已修复（teacher vis deterministic 覆盖不再丢参）。

### M1 baseline 定位与闭环
- 已形成 4 组 student 可比较实现与脚本：
  - `ProprioAdapt`
  - `PureBC`
  - `DiffusionLatentStudent`
  - `DiffusionActionChunkStudent`
- 统一验收汇总已落地：
  - `docs/stage_acceptance_summary.md`
  - `scripts/summarize_student_acceptance.py`
- 当前 student 的方法学定位已写入：
  - `docs/Algorithm.md`

### M2 机制拆分对照（部分完成）
- 已有 `PureBC`（去除 latent distillation）这条拆分对照，可用于回答“current student 比 pure BC 多了什么”。
- `adapter training range ablation` 最小实现已落地（默认语义不变，ablation 脚本中开启扩大训练范围）。
- 首轮 15min（`adapt_tconv+actor_mlp+mu`）已完成，`Max Current Best=32.31`，相对 baseline 显著退化。
- 窄范围 15min 已完成：
  - `adapt_tconv+mu`：`Max Current Best=69.11`
  - `adapt_tconv+actor_mlp`：`Max Current Best=7.06`
- 当前可得最小结论：放开 `actor_mlp` 是主要退化源，`mu` 放开也会退化但程度较轻。

### M3 teacher 数据接口最小可用（已完成）
- teacher rollout 采集已支持：
  - `scripts/collect_screwdriver_teacher_rollout.sh`
  - `scripts/collect_screwdriver_teacher_rollout_docker.sh`
- 导出字段包含：`obs/proprio_hist/priv_info/actions/rewards/dones/done_rate_per_step/point_cloud_info`（可开关）。

### M3 消费侧最小对接（本次新增，已完成）
- Action-chunk diffusion 已支持可选 rollout 预训练入口（默认关闭，不影响原训练语义）：
  - `+train.ppo.rollout_pretrain_path`
  - `+train.ppo.rollout_pretrain_updates`
  - `+train.ppo.rollout_pretrain_batch_size`
  - `+train.ppo.rollout_pretrain_log_interval`
- 新增脚本：
  - `scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh`

### M4 diffusion 首次跑通
- `DiffusionLatentStudent` 与 `DiffusionActionChunkStudent` 都已可训练与评测，action-chunk 已完成修复并可持续训练。
- `DiffusionActionChunkStudent` 已完成两轮 15min 调参并提升：
  - tune-v1：`1203.64 -> 1280.19`
  - tune-v2：`1280.19 -> 1502.21`（超过 `ProprioAdapt=1496.08`）

## 4. 本次关键代码落点
- `dexscrew/algo/ppo/padapt.py`
  - 新增 `train.ppo.student_trainable_param_patterns` 解析逻辑
  - 默认仍为 `[adapt_tconv]`，保持原 ProprioAdapt 训练语义
  - 新增 trainable pattern / trainable params 启动日志，便于验收
- `configs/train/XHandHoraScrewDriver.yaml`
  - 新增默认项：`student_trainable_param_patterns: [adapt_tconv]`
- `scripts/screwdriver_student_padapt_trainrange.sh`
  - 提供 adapter training range ablation 的主入口（默认使用 `[adapt_tconv,actor_mlp,mu]`）
- `scripts/screwdriver_student_padapt_trainrange_15min_docker.sh`
  - 提供 train-range ablation 的 15 分钟容器验收入口
- `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - 新增 rollout 数据集构造函数 `build_action_chunk_rollout_dataset(...)`
  - 新增可选 rollout 预训练流程 `_run_rollout_pretrain_if_enabled()`
  - 训练循环中接入“预训练后再在线蒸馏”逻辑
- `scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh`
  - 一键触发 rollout 预训练 + 原 action-chunk 训练入口
- `docs/stage_closure.md`
  - 记录了 P3 消费侧已落地
- `docs/Algorithm.md`
  - 补充 action-chunk 的 rollout 预训练能力说明
- `docs/experiment_conclusion.md`
  - 固化本阶段机制定位结论（`adapt_tconv` 非 diffusion 特性、train-range 退化来源、后续主线决策）
- `docs/robustness_eval_summary.md`
  - 新增 P5 最小鲁棒性评测包结果（nominal/perturb 对照）
- 本轮以“实验调参”推进为主，无新增算法文件与核心代码改动（保持主线路径稳定）。

## 5. 已验证事项（可复现）
- 脚本语法检查：
  - `bash -n scripts/screwdriver_student_padapt_trainrange.sh`
  - `bash -n scripts/screwdriver_student_padapt_trainrange_15min_docker.sh`
- 参数透传检查（python shim）：
  - `scripts/screwdriver_student_padapt_trainrange.sh` 仅触发一次 `python train.py`
  - 包含关键 override：`+train.ppo.student_trainable_param_patterns=[adapt_tconv,actor_mlp,mu]`
- 容器内 smoke（120s 截断）：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_padapt_trainrange.sh 0 42 run_a task.env.numEnvs=8`
  - 关键日志已出现：`ProprioAdapt trainable patterns: ['adapt_tconv', 'actor_mlp', 'mu'] | trainable params: ...`
- 15 分钟验收已完成：
  - `scripts/screwdriver_student_padapt_trainrange_15min_docker.sh 0 42 run_a 900 run_a_trainrange_seed42_15min`
  - 结果：`Max Current Best: 32.31`
  - 汇总已更新：`docs/stage_acceptance_summary.md` 增加 `padapt_trainrange` 行
- 两条窄范围 15 分钟验收已完成：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_padapt_trainrange.sh ... +train.ppo.student_trainable_param_patterns=[adapt_tconv,mu]`
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_padapt_trainrange.sh ... +train.ppo.student_trainable_param_patterns=[adapt_tconv,actor_mlp]`
  - 结果分别为：`69.11` 与 `7.06`
  - 汇总已更新：`docs/stage_acceptance_summary.md` 增加 `padapt_trainrange_adapt_mu` 与 `padapt_trainrange_adapt_actor` 行
- diffusion action-chunk 调参 15 分钟验收（同口径）：
  - tune-v1：`train.ppo.action_chunk_teacher_mix_steps=300000`, `train.ppo.action_chunk_first_action_bc_loss_coef=2.0`, `train.ppo.action_chunk_bc_loss_coef=0.2`
    - 结果：`Max Current Best=1280.19`
  - tune-v2：`train.ppo.action_chunk_teacher_mix_steps=500000`, `train.ppo.action_chunk_first_action_bc_loss_coef=3.0`, `train.ppo.action_chunk_bc_loss_coef=0.2`
    - 结果：`Max Current Best=1502.21`
  - 统一汇总已更新：`docs/stage_acceptance_summary.md` 增加 `diffusion_action_chunk_tune_v1/v2` 行
- P5 鲁棒性最小评测包（256 steps）：
  - `ProprioAdapt`：nominal `avg_reward=1.918988`
  - `DiffusionLatentStudent`：nominal `avg_reward=1.489716`
  - `DiffusionActionChunk` 修复版：nominal `avg_reward=-1.873523`
  - `DiffusionActionChunk` tune-v2：nominal `avg_reward=-1.248181`
  - 结果文档：`docs/robustness_eval_summary.md`
- 语法检查：
  - `bash -n scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh`
- rollout 数据构造函数 shape 验证（容器内）：
  - `./docker-run-isaacgym.sh python -c "... build_action_chunk_rollout_dataset ..."`
- teacher rollout 采集 smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/collect_screwdriver_teacher_rollout.sh 0 42 run_a 12 pretrain_smoke task.env.numEnvs=8`
- rollout 预训练触发 smoke（关键日志已出现）：
  - `Rollout pretrain start | ...`
  - `Rollout pretrain | 1/2 | ...`
  - `Rollout pretrain | 2/2 | ...`
- 事件文件中已存在 `rollout_pretrain/*` 标量 tag（已通过字符串检索确认）。

## 6. 当前风险与注意
- 若 rollout 步数 `< action_chunk_len`，会触发显式报错（预期行为）：
  - 例如 `chunk_len=8` 时，rollout 至少需要 `>=8` 步。
- 本地 Python 环境可能缺少 `tensorboard`/`tensorboardX`，建议验证以容器环境为准。
- 当前 `AGENTS.md`、`PLANS.md`、多脚本仍有未提交改动，后续合并前需做一次有序提交整理。
- train-range 对照显示：`actor_mlp` 放开会导致最强退化，当前不宜作为主线默认设置。
- action-chunk tune-v2 结果受较长 teacher mixing 影响，后续需补一次固定测试口径评估，确认是否存在“混合执行抬高训练分数”的偏差。
- 已补固定步评测后，action-chunk 仍存在“训练高分 vs 纯评测低分”偏差；当前需避免只用训练曲线下结论。

## 7. 下一步建议（单一推荐）
进入 `P5` 鲁棒性轴的下一步：先以 `DiffusionLatentStudent` 作为稳态 diffusion 路径做贡献验证，再并行保留 action-chunk 偏差修复。  
原因：action-chunk 虽有训练分数提升，但纯 student 评测仍不稳；latent 已体现更稳的正回报行为。

## 8. 新会话快速上手命令
```bash
# 1) 进入项目
cd ~/Codefield/py/dexscrew-repro

# 2) 快速看阶段状态
sed -n '1,220p' docs/session_handoff.md
sed -n '1,220p' docs/stage_closure.md
sed -n '1,220p' docs/stage_acceptance_summary.md

# 3) 采集 teacher rollout（示例）
./docker-run-isaacgym.sh timeout 180 scripts/collect_screwdriver_teacher_rollout.sh 0 42 run_a 128 handoff

# 4) 用 rollout 预训练 action-chunk diffusion（示例）
./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh \
  0 42 run_a outputs/teacher_rollouts/XHandHoraScrewDriver_teacher/run_a/handoff_seed42_steps128.pt \
  run_a_action_chunk_rollout_pretrain 2000

# 5) 跑 train-range ablation 15min（示例）
scripts/screwdriver_student_padapt_trainrange_15min_docker.sh 0 42 run_a 900 run_a_trainrange_seed42_15min

# 6) action-chunk diffusion tune-v2（示例）
./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh \
  0 42 run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min \
  "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" \
  train.ppo.action_chunk_teacher_mix_steps=500000 \
  train.ppo.action_chunk_first_action_bc_loss_coef=3.0 \
  train.ppo.action_chunk_bc_loss_coef=0.2
```

## 9. 2026-03-21 追加推进（P5 最小里程碑）
### 目标子项
- 在不改训练代码的前提下，定位 action-chunk “训练高分 vs 纯评测偏低”是否由推理配置引起。
- 同时把“会话启动必须读交接状态”的动作规则固化到治理文件，降低后续会话重复试错。

### 本次改动
- `AGENTS.md`
  - 在 `Session Bootstrap` 中补充开工前检查清单：
    - 先确认 handoff 的“single recommended next step”
    - 避免重复跑已失败设置（除非有明确修复假设）
    - 首条执行更新需声明当前 milestone/subgoal
- `docs/robustness_eval_summary.md`
  - 新增“推理侧 512-step 定位”小节，记录 3 组 inference 对照结果与结论

### 本次验证（可复现）
- `bash -n scripts/eval_screwdriver_student_robustness.sh`
- 三组 action-chunk tune-v2 固定步评测（nominal, 512 steps）：
  - deterministic + `diffusion_steps_infer=10`：
    - `EvalSummary steps=512 avg_reward=-1.626525 avg_done_rate=0.013184`
  - stochastic + `diffusion_steps_infer=10`：
    - `EvalSummary steps=512 avg_reward=-1.588469 avg_done_rate=0.013509`
  - deterministic + `diffusion_steps_infer=20`：
    - `EvalSummary steps=512 avg_reward=-1.626525 avg_done_rate=0.013184`

### 当前结论
- 仅靠推理侧开关（stochastic / infer steps）无法实质修复 action-chunk 纯评测负回报。
- 当前更像训练目标与 teacher mixing 的对齐问题，而非简单采样步数问题。

### 单一推荐下一步
- 按 `P5` 先走稳态路径：以 `DiffusionLatentStudent` 进入鲁棒性贡献验证（nominal/perturb 扩展与对比），action-chunk 保留并行修复但不作为当前唯一主线。

## 10. 2026-03-21 继续推进（P5 v2：latent 主线稳定性）
### 目标子项
- 按 handoff 推荐路线，先不继续纠缠 action-chunk，补齐 `DiffusionLatentStudent` 的多扰动强度验证，并与 `ProprioAdapt` 同口径对照。

### 本次验证（可复现）
- 命令入口统一为：`scripts/eval_screwdriver_student_robustness.sh`
- 评测档位：`nominal / light / hard`（`steps=256`, `seed=42`）
- ckpt：
  - latent：`outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - padapt：`outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
- 关键结果：
  - `DiffusionLatentStudent`
    - nominal：`avg_reward=1.489716`, `avg_done_rate=0.001872`
    - light：`avg_reward=1.327045`, `avg_done_rate=0.002116`
    - hard：`avg_reward=0.950702`, `avg_done_rate=0.002523`
  - `ProprioAdapt`
    - nominal：`avg_reward=1.918988`, `avg_done_rate=0.001383`
    - light：`avg_reward=2.063784`, `avg_done_rate=0.001058`
    - hard：`avg_reward=1.826708`, `avg_done_rate=0.001546`

### 本次结论
- latent diffusion 在三档扰动下保持正回报，已满足“稳态 diffusion 主线可持续推进”的条件。
- 但相对强 baseline（ProprioAdapt）仍有显著差距，因此当前定位应是“可比较可优化主线”，不是“已替代 baseline”。

### 单一推荐下一步
- 进入 `P5` 的下一小步：围绕 latent 路线做一次“轻量训练侧鲁棒性增强”实验（先从 `light` 档随机化注入开始），并复用本次三档评测口径做闭环对比。

## 11. 2026-03-21 继续推进（P5 v3：latent 训练侧鲁棒性优化）
### 目标子项
- 在不新增算法的前提下，验证“训练分布扰动覆盖不足”是否是 latent 路线鲁棒性差距的关键原因。

### 本次代码与脚本改动
- 新增：
  - `scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - `scripts/screwdriver_student_diffusion_latent_robust_light_15min_docker.sh`
- 作用：
  - 固化“训练侧轻扰动注入”的可复现入口（默认注入 `forceScale=0.5`, `randomForceProbScalar=0.1`）
  - 不影响原有 latent baseline 训练脚本与语义

### 本次验证（可复现）
- 语法检查：
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light_15min_docker.sh`
- 训练 run（15min 窗口）：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent.sh 0 42 run_a_latent_robust_light_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - 产物：`outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- 固定步评测（steps=256）：
  - nominal：`avg_reward=1.432408`, `avg_done_rate=0.001709`
  - light：`avg_reward=1.630288`, `avg_done_rate=0.001383`
  - hard：`avg_reward=1.344396`, `avg_done_rate=0.002035`

### 与 latent baseline 的关键对比
- baseline latent：
  - nominal `1.489716 / 0.001872`
  - light `1.327045 / 0.002116`
  - hard `0.950702 / 0.002523`
- robust-light latent：
  - nominal 轻微回落（`-0.0573`）
  - light 显著提升（`+0.3032`）
  - hard 显著提升（`+0.3937`）

### 本次结论
- “训练侧扰动覆盖不足”是 latent 鲁棒性差距的真实原因之一；通过小幅训练注入即可获得可观改进。
- 该方向符合当前主线约束：不新增算法、改动可控、收益可验证。

### 单一推荐下一步
- 继续 `P5`：以 latent 为主线，做“扰动强度小网格 + 名义性能约束”的 2~3 点扫描（例如 `forceScale`/`randomForceProbScalar`），目标是在不明显牺牲 nominal 的前提下继续拉高 hard 档表现。

## 12. 2026-03-21 继续推进（P5 v4：关键性能优化定位）
### 目标子项
- 按 v3 推荐，完成 latent 扰动强度小网格扫描，检查是否有比 `0.5/0.1` 更优的稳定点。
- 同时修复一个会影响后续优化效率的工程瓶颈：DiffusionLatentStudent 续训能力。

### 本次代码改动（最小且可逆）
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `restore_train(...)` 支持从 diffusion ckpt 恢复 `diffusion_model`（兼容 teacher ckpt fallback）。
  - `save(...)` 新增保存 `diffusion_optim`，为后续真正续训提供状态恢复。
- 影响：
  - 不改变现有 teacher->student 冷启动语义。
  - 新增“可从已有 diffusion ckpt 接力优化”的能力，便于做课程式/分阶段调优。

### 本次实验（可复现）
- 新增 2 个 15min 训练点（seed=42）：
  - `run_a_latent_robust_fs04_p008_seed42_15min`（`forceScale=0.4`, `randomForceProbScalar=0.08`）
    - `Max Current Best=1831.22`
  - `run_a_latent_robust_fs06_p012_seed42_15min`（`forceScale=0.6`, `randomForceProbScalar=0.12`）
    - `Max Current Best=1862.06`
- 复核已有点：
  - `run_a_latent_robust_mid_seed42_15min`（`0.8/0.15`）
    - `Max Current Best=1797.54`
- 统一评测口径（`scripts/eval_screwdriver_student_robustness.sh`, `steps=256`）：
  - `0.5/0.1`（robust-light）：
    - nominal `1.432408 / 0.001709`
    - light `1.630288 / 0.001383`
    - hard `1.344396 / 0.002035`
  - `0.8/0.15`（robust-mid）：
    - nominal `1.292951 / 0.001953`
    - light `1.427850 / 0.002279`
    - hard `1.230681 / 0.002523`
  - `0.4/0.08`：
    - nominal `1.051928 / 0.001628`
    - light `1.134281 / 0.001628`
    - hard `0.950787 / 0.002686`
  - `0.6/0.12`：
    - nominal `1.500453 / 0.001953`
    - light `1.477706 / 0.001628`
    - hard `1.229945 / 0.002035`
- 推理步数复核（robust-light ckpt）：
  - `+train.ppo.diffusion_steps_infer=20` 在 nominal/hard 与 infer=10 一致，无可见收益。

### 本次结论
- 训练峰值 `Current Best` 和固定步评测并不总一致，当前不能只看训练曲线下结论。
- 静态扰动小网格下，`0.5/0.1` 仍是当前最稳妥的综合点；`0.6/0.12` 仅在 nominal 更高，但 light/hard 回落。
- 关键工程增益已落地：latent 续训能力可用，为下一步“分阶段优化”打开路径。

### 单一推荐下一步
- 进入“分阶段续训”验证：先用 `0.5/0.1` 训练一个短窗口，再以该 diffusion ckpt 续训一个更短窗口（同点位或轻微降扰动），验证是否能在不牺牲 hard 的前提下回补 nominal。

## 13. 2026-03-21 追加（P5 v6：PureBC 同口径补测）
### 目标子项
- 回答“PureBC 是否已测，以及在 nominal/light/hard 下表现如何”，补齐统一对照。

### 本次验证（可复现）
- ckpt：`outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`
- 脚本：`scripts/eval_screwdriver_student_robustness.sh`
- 评测设置：`steps=256, seed=42`
- 结果：
  - nominal：`avg_reward=1.267421`, `avg_done_rate=0.001546`
  - light：`avg_reward=2.144827`, `avg_done_rate=0.001465`
  - hard：`avg_reward=1.807907`, `avg_done_rate=0.001628`

### 本次结论
- `PureBC` 已完成训练验收与鲁棒性三档补测；后续可以作为和 `ProprioAdapt`、`DiffusionLatentStudent` 的稳定对照项。

## 14. 2026-03-21 追加（P5 v7：四算法统一三挡复核）
### 目标子项
- 回答“是否四种算法都完成三挡测试，以及是否 adapt 更优”。
- 统一 light/hard 定义后重跑缺失项，避免口径混用。

### 本次验证（可复现）
- 脚本：`scripts/eval_screwdriver_student_robustness.sh`
- 统一三挡定义（`steps=256`, `seed=42`）：
  - nominal：默认 deterministic
  - light：`obs_noise_e=0.02`, `obs_noise_t=0.01`, `forceScale=0.5`, `randomForceProbScalar=0.1`
  - hard：`obs_noise_e=0.05`, `obs_noise_t=0.025`, `forceScale=1.5`, `randomForceProbScalar=0.3`
- 结果（`avg_reward / avg_done_rate`）：
  - ProprioAdapt：nominal `1.918988 / 0.001383`，light `1.958849 / 0.001546`，hard `1.826708 / 0.001546`
  - PureBC：nominal `1.267421 / 0.001546`，light `2.144827 / 0.001465`，hard `1.807907 / 0.001628`
  - DiffusionLatent（robust-light）：nominal `1.432408 / 0.001709`，light `1.556369 / 0.001221`，hard `1.344396 / 0.002035`
  - DiffusionActionChunk（tune-v2）：nominal `-1.248181 / 0.013184`，light `-0.937148 / 0.012533`，hard `-0.768358 / 0.013916`

### 本次结论
- 四算法三挡 reward + done_rate 已齐全。
- 当前综合表现看：`ProprioAdapt` 仍是最稳的强 baseline；`DiffusionLatent` 可用但未持平；`ActionChunk` 仍明显落后。
- 当前未单独输出任务级“成功率（success rate）”，若需要论文图表口径，下一步应补成功事件统计并统一导出。

## 15. 2026-03-21 追加（P5 v8：ProprioAdapt 1h 验证）
### 目标子项
- 验证“15min 是否低估 adapt”，并明确延长训练对三挡扰动的影响方向。

### 本次执行（可复现）
- 训练：
  - `scripts/screwdriver_student_padapt_15min_docker.sh 0 42 run_a 3600 run_a_seed42_1h`
  - 产物：`outputs/XHandHoraScrewDriver_student_padapt/run_a_seed42_1h/stage2_nn/model_best.ckpt`
  - `Max Current Best=1608.43`
- 评测（同口径，steps=256）：
  - nominal：`avg_reward=2.077178`, `avg_done_rate=0.000977`
  - light：`avg_reward=2.065615`, `avg_done_rate=0.000977`
  - hard：`avg_reward=1.733165`, `avg_done_rate=0.001465`

### 与 15min 对比（`run_a`）
- nominal：`1.918988 -> 2.077178`（上升）
- light：`1.958849 -> 2.065615`（上升）
- hard：`1.826708 -> 1.733165`（下降）

### 本次结论
- “15min 太短”这个假设对 nominal/light 成立。
- 但长训并不保证强扰动泛化提升，hard 回落提示需要把训练目标或模型选择纳入“鲁棒性约束”。

## 16. 2026-03-21 追加（P5 v9：DiffusionLatent robust-light 1h 验证）
### 目标子项
- 回答“DiffusionLatent 从 15min 拉到 1h，是否差距很大；是否主要是拟合时长问题”。

### 本次执行（可复现）
- 训练：
  - `./docker-run-isaacgym.sh timeout 3600 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_robust_light_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth"`
  - 结果：`Max Current Best=1848.04`
  - 产物：`outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_1h/stage2_diffusion_nn/model_best.ckpt`
- 评测（统一口径，`scripts/eval_screwdriver_student_robustness.sh`, `steps=256`, `seed=42`）：
  - nominal：`avg_reward=1.610548`, `avg_done_rate=0.001546`
  - light：`avg_reward=1.432580`, `avg_done_rate=0.001872`
  - hard：`avg_reward=1.248908`, `avg_done_rate=0.002360`

### 与 15min robust-light 对比
- nominal：`1.432408 -> 1.610548`（`+0.178140`）
- light：`1.556369 -> 1.432580`（`-0.123789`）
- hard：`1.344396 -> 1.248908`（`-0.095488`）

### 本次结论
- 你的理解大方向是对的，但要加一个关键限定：
  - 如果 1h 相比 15min 显著提升，说明确实有“拟合时长不足”成分（本次在 nominal 成立）。
  - 若 1h 提升不大或出现回落，不能简单判定“算法无问题”，更可能是目标函数/模型选择偏向名义性能，导致鲁棒性不随时长同步提升（本次 light/hard 即如此）。
- 现阶段可将 `15min` 作为快速蒸馏可行性筛选窗口；论文主结论仍应以更长训练 + 统一鲁棒评测给出。

## 17. 2026-03-21 追加（P5 v10：DiffusionActionChunk tune-v2 1h 验证）
### 目标子项
- 回答“action-chunk 表现差是否仅是 15min 欠拟合”。

### 本次执行（可复现）
- 训练：
  - `./docker-run-isaacgym.sh timeout 3600 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=3.0 train.ppo.action_chunk_bc_loss_coef=0.2`
  - `Max Current Best=1502.21`
- 与 15min tune-v2 对比（`run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min`）：
  - 训练峰值同值：`1502.21`
  - `model_best.ckpt` 哈希完全一致：
    - `3e360f545ced53aaf39c6cc4998fb7e600bf80a2c55257a07b4fb0128e688bc3`
- nominal 评测（steps=256）：
  - `avg_reward=-1.248181`, `avg_done_rate=0.013184`
  - 与 15min 一致。

### 本次结论
- 当前 action-chunk tune-v2 的瓶颈不再主要是“训练时长不够”；延长到 1h 未带来更优 checkpoint。
- 下一步应把主精力放在算法/训练目标对齐排查，而不是继续单纯拉长时长。

## 18. 2026-03-21 追加（P5 v11：ActionChunk 训练-评测口径对齐修复）
### 目标子项
- 高效定位 action-chunk 的“训练好看但纯 student 评测差”问题，并做最小修复。

### 本次代码改动（最小且可逆）
- `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - 新增 `teacher_mix_ratio` 日志。
  - 新增周期性 `model_last.ckpt` 保存（默认 `action_chunk_ckpt_interval_steps=50000`）。
  - 新增 student 对齐选模：
    - `model_best_student_reward.ckpt`（warmup 后按 reward）；
    - `model_best_student.ckpt`（warmup 后按首动作 `student_action_mse`）。
  - 训练日志新增 `Best Student Reward / Best Student MSE`。
- `scripts/vis_screwdriver_student_diffusion_action_chunk.sh`
  - 默认 checkpoint 选择改为：
    - `model_best_student_reward.ckpt` -> `model_best_student.ckpt` -> `model_last.ckpt` -> `model_best.ckpt`

### 本次验证（可复现）
- 诊断 A（300s，`teacher_mix_steps=20000`）：
  - run: `run_a_action_chunk_aligndiag_mix20k_seed42_300s`
  - nominal（steps=256）：
    - `model_best`: `-1.951475 / 0.010173`
    - `model_best_student_reward`: `-2.500227 / 0.010579`
    - `model_best_student`(MSE): `-0.626234 / 0.010905`
    - `model_last`: `-0.740303 / 0.010905`
- 诊断 B（主线 15min，`mix500k + fa3 + cb0.2`）：
  - run: `run_a_action_chunk_tune_mix500k_fa3_cb02_alignmse_seed42_15min`
  - `model_best` 与历史 tune-v2 哈希一致（`3e360f...`），确认旧 best 仍是早期锁定点。
  - nominal（steps=256）：
    - `model_best`: `-1.248181 / 0.013184`
    - `model_best_student_reward`: `-0.676029 / 0.011963`
    - `model_best_student`(MSE): `-0.812928 / 0.009847`
    - `model_last`: `-0.466151 / 0.011475`

### 本次结论
- 核心问题已定位：`model_best` 选模口径和 pure-student 评测口径不一致（teacher-mix 阶段锁定 best）。
- 修复后无需改主算法结构即可显著改善 action-chunk 评测结果（虽然仍未转正）。
- “只拉长时长”不是当前最高收益路径；应继续做对齐与目标函数优化。

### 单一推荐下一步
- 在当前修复版本上做一个 15min 小网格：
  - 固定 `mix500k`，扫描 `first_action_bc_loss_coef`（例如 `3/5/8`）与 `chunk_bc_loss_coef`（例如 `0.2/0.1/0.05`），
  - 统一以 `model_best_student_reward` 与 `model_last` 做 nominal/light/hard 三档评测，优先寻找“转正 nominal”组合。

## 19. 2026-03-21 追加（P5 v12：ActionChunk 修复后快速扫描与候选确认）
### 目标子项
- 按 v11 推荐，快速扫描 `fa/cb` 组合，找到比旧 tune-v2 更接近转正的配置。

### 本次执行（可复现）
- 扫描统一设置：
  - `teacher_mix_steps=500000`
  - `action_chunk_model_selection_warmup_steps=0`
  - 训练窗口：300s，seed=42
  - teacher：`outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth`
- 扫描点及 nominal（steps=256）结果：
  - A `fa5/cb0.2`（`run_a_action_chunk_scan_fa5_cb02_mix500k_w0_seed42_300s`）
    - `model_best` / `model_best_student_reward`: `-0.497100 / 0.013835`
    - `model_last`: `-0.623638 / 0.010417`
  - B `fa8/cb0.2`（`run_a_action_chunk_scan_fa8_cb02_mix500k_w0_seed42_300s`）
    - `model_last` 最优：`-0.682759 / 0.011475`
    - `model_best`: `-3.641524 / 0.012288`
  - C `fa5/cb0.1`（`run_a_action_chunk_scan_fa5_cb01_mix500k_w0_seed42_300s`）
    - `model_best_student`(MSE) 最优：`-0.589909 / 0.009847`
    - `model_best`: `-2.000882 / 0.011637`
- 选定候选：`fa5/cb0.2`

### 候选 15min 验证
- run：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_15min`
- 训练：
  - `Max Current Best=1536.97`
  - `Best Student MSE=0.2761`
- nominal（steps=256）：
  - `model_best`: `-0.497100 / 0.013835`
  - `model_best_student_reward`: `-0.497100 / 0.013835`
  - `model_best_student`: `-0.741821 / 0.010417`
  - `model_last`: `-0.047760 / 0.008626`（当前最好）
- light/hard（均用 `model_last`）：
  - light：`-0.061166 / 0.009847`
  - hard：`-0.255275 / 0.008789`

### 本次结论
- 修复后的 action-chunk 已从“明显负回报”提升到“接近转正”区间。
- `model_last` 在当前主线设置下优于各类 `model_best*`，说明“后期 student-only 收敛”是当前关键收益来源。
- `fa5/cb0.2` 是当前最优候选点。

### 单一推荐下一步
- 对 `fa5/cb0.2 + mix500k + warmup0` 做 1h 验证，并在 `model_last` 与 `model_best_student_reward` 上统一跑 nominal/light/hard，确认是否可稳定转正。

## 20. 2026-03-21 追加（P5 v13：ActionChunk 候选 1h 验证完成）
### 目标子项
- 完成 v12 推荐的 1h 验证，确认 `fa5/cb0.2` 是否可稳定转正。

### 本次执行（可复现）
- 训练：
  - run：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_1h`
  - 命令：
    - `./docker-run-isaacgym.sh timeout 3600 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
  - 训练结果：
    - `Max Current Best=1536.97`
    - `Best Student MSE(末值)=0.2761`
- 评测（统一口径，steps=256, seed=42）：
  - `model_best_student_reward`：
    - nominal：`-0.497100 / 0.013835`
    - light：`-0.415648 / 0.014079`
    - hard：`-0.215941 / 0.013672`
  - `model_last`：
    - nominal：`0.121916 / 0.007975`
    - light：`0.435321 / 0.007568`
    - hard：`0.233560 / 0.007813`

### 本次结论
- action-chunk 在当前修复和参数下已经达到三挡正回报（基于 `model_last`）。
- `model_last` 持续优于 `model_best_student_reward`，当前导出/可视化应优先使用 `model_last`。
- 这是从“训练-评测口径错位”修复到“可用候选”闭环的关键里程碑。

### 单一推荐下一步
- 进入多 seed 复核（建议 seed `42/43/44`，先 15min，再挑 1 个 seed 拉 1h），验证该候选点是否统计稳健；确认后再进入论文正式对比图表输出。

## 21. 2026-03-22 追加（P5 v14：ActionChunk 多 seed 复核完成）
### 目标子项
- 按 v13 推荐，完成 `seed 42/43/44` 的统一三挡复核，检查 `fa5/cb0.2 + mix500k + warmup0` 是否具备统计稳健性。

### 本次执行（可复现）
- 补齐训练（seed44, 15min）：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 44 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed44_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
- 统一评测口径（均用 `model_last.ckpt`, steps=256）：
  - nominal：默认 deterministic
  - light：`obs_noise_e=0.02`, `obs_noise_t=0.01`, `forceScale=0.5`, `randomForceProbScalar=0.1`
  - hard：`obs_noise_e=0.05`, `obs_noise_t=0.025`, `forceScale=1.5`, `randomForceProbScalar=0.3`
- 结果（`avg_reward / avg_done_rate`）：
  - seed42：nominal `0.121916 / 0.007975`，light `0.435321 / 0.007568`，hard `0.233560 / 0.007813`
  - seed43：nominal `-0.616286 / 0.007650`，light `-0.206074 / 0.007487`，hard `-0.825969 / 0.009277`
  - seed44：nominal `-0.111548 / 0.007975`，light `-0.041217 / 0.008870`，hard `-0.240999 / 0.008789`

### 本次结论
- 当前 action-chunk 候选存在明显 seed 方差，`seed42` 的正回报结论不能直接外推为稳态结论。
- 三 seed reward 均值（nominal/light/hard）分别为 `-0.201973 / 0.062677 / -0.277803`，说明主线仍需“先稳态、后冲高”。
- 现阶段可保留该路线为“有效候选”，但论文主结论应以 `ProprioAdapt` 与 `DiffusionLatent` 稳态对照为主，action-chunk 作为改进中分支更合适。

### 单一推荐下一步
- 做一轮“稳定性优先”的最小改动实验：固定当前参数，仅将训练时长统一拉到 `1h`（seed `43/44`），验证负回报是否主要由欠拟合导致；若 1h 仍不稳，再进入算法细节排查（先排对齐/归一化，再排模型结构）。

## 22. 2026-03-22 追加（P5 v15：先做 15min 方差定位，不上 1h）
### 目标子项
- 按用户要求，先完成 15min 排查，确认“不同 seed 差异大”是否是算法实现错误，避免直接上 1h 扩时长。

### 本次执行（可复现）
- 先停掉已启动的 `seed43_1h` 任务，回到 15min 主线。
- 完成交叉评测矩阵（`mix500k`, `fa5/cb0.2`, `model_last`, nominal, steps=256）：
  - train seed42 -> eval seed42/43/44：`0.121916 / -0.343631 / 0.023255`
  - train seed43 -> eval seed42/43/44：`0.068391 / -0.616286 / 0.110339`
  - train seed44 -> eval seed42/43/44：`-0.092808 / 0.055781 / -0.111548`
- 追加 15min 对照实验（只改 `mix_steps`）：
  - run：`run_a_action_chunk_tune_fa5_cb02_mix300k_w0_seed43_15min`
  - 训练结果：`Current Best=1725.38`, `Best Student MSE=0.3250`
  - `model_last` 评测（eval seed 42/43/44）：`-0.163048 / 0.113329 / -0.367999`
- 新增多 seed 评测脚本：
  - `scripts/eval_screwdriver_student_robustness_multiseed.sh`
  - 语法检查：`bash -n scripts/eval_screwdriver_student_robustness_multiseed.sh`
  - docker smoke（单 seed）：`./docker-run-isaacgym.sh timeout 300 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionActionChunkStudent <ckpt> 64 tmp_multiseed_smoke "42"`
  - 脚本可自动解析 `EvalSummary` 并输出 `reward/done_rate` 的 mean/std 汇总。

### 本次结论
- 关键发现：同一 ckpt 在不同 eval seed 下可出现正负翻转，评测方差很大；不能用单一 eval seed 判定算法成败。
- `mix500k -> mix300k` 在 15min/seed43 下没有实质提升（均值几乎不变），说明当前问题不是单靠 teacher-mix 时长就能修复。
- 现阶段不支持“实现明显写错”这一判断；更像评测口径噪声大 + policy 仍在临界区。

### 单一推荐下一步
- 固化“多 eval seed 均值”作为 action-chunk 的验收口径（至少 eval seed `42/43/44`），然后再做小步参数优化；先优化均值和方差，再讨论是否上 1h。

## 23. 2026-03-22 追加（P5 v16：ActionChunk deterministic 实现级修复）
### 目标子项
- 继续排查“不同 seed 差异大”的根因，确认是否为 action-chunk 模块实现问题，而非环境噪声或训练时长单因素。

### 本次代码改动（最小且可逆）
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 位置：`sample_action_chunk(...)`
- 改动：
  - 修复前：即使 `action_chunk_stochastic_infer=False`，也总是 `torch.randn(...)` 初始化 `x_T`。
  - 修复后：仅 stochastic 模式使用随机初始化；deterministic 模式改为 `x_T=0` 固定初始化。
- 影响：
  - deterministic 评测语义与配置对齐；
  - 不改 teacher/student 架构，不改训练入口，不改 checkpoint 格式。

### 本次验证（可复现）
- ActionChunk 修复后同 ckpt 回归（multiseed512）：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionActionChunkStudent outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed43_15min/stage2_diffusion_action_chunk_nn/model_last.ckpt 512 p5v16_actionchunk_determinfix_multiseed512 "42,43,44"`
  - 结果：
    - seed42：`0.289716 / 0.009318`
    - seed43：`0.079925 / 0.007853`
    - seed44：`0.486741 / 0.007894`
    - aggregate：`reward_mean=0.285461`, `reward_std=0.166109`
- 修复前同口径基线（同 ckpt，历史记录）：
  - aggregate：`reward_mean=-0.140979`, `reward_std=0.205605`
- 同轮横向对照（multiseed512）：
  - ProprioAdapt：`2.400372 ± 0.151554`
  - DiffusionLatent(robust-light)：`1.742653 ± 0.104761`
  - PureBC：`2.116345 ± 0.263339`

### 本次结论
- 已定位到 action-chunk 的实现级一致性问题：deterministic 推理路径此前仍被随机初始化污染。
- 最小修复后，action-chunk 从“负均值”变为“正均值”，并且 seed 方差下降，说明此前结论里一部分“算法不稳”来自推理实现问题。
- 当前 action-chunk 绝对性能仍未追平强 baseline（ProprioAdapt/PureBC），但主阻塞已从“评测口径/实现不一致”转为“训练与目标优化问题”。

### 单一推荐下一步
- 在该修复基础上执行一轮最小 15min 网格（保持 `mix500k`、`warmup0`，扫描 `fa/cb` 2~3 点），统一用 `multiseed(42/43/44)` 的 `reward_mean` 作为唯一选型指标，再决定是否进入 1h 验证。

## 24. 2026-03-22 追加（P5 v17：修复版 15min 重训复核）
### 目标子项
- 验证 `P5 v16` 的 deterministic 修复是否不仅改善旧 checkpoint 评测，也能在新训练中稳定带来更好的 pure-student 表现。

### 本次执行（可复现）
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
- 训练结果：
  - `Current Best=1692.80`
  - `Best Student MSE=0.1589`
- 部署评测（`multiseed512`, nominal）：
  - `model_last.ckpt`
    - seed42：`-0.510649 / 0.011230`
    - seed43：`-0.084439 / 0.010905`
    - seed44：`-0.332757 / 0.011190`
    - aggregate：`reward_mean=-0.309282`, `reward_std=0.174790`
  - `model_best_student.ckpt`
    - aggregate：`reward_mean=-1.339584`, `reward_std=0.165684`
  - `model_best_student_reward.ckpt`
    - aggregate：`reward_mean=-1.409180`, `reward_std=0.224958`

### 本次结论
- 这轮结果非常关键：action-chunk 现在的主问题已经不是“deterministic 实现错误”，而是“训练高分与 pure-student 部署高分没有稳定对齐”。
- 修复后的旧 checkpoint 可以评测转正，但新训练并没有自动继承这个收益；反而出现了更高 `Current Best` 对应更差 pure-student multiseed 的情况。
- 当前三类 checkpoint 里依然是 `model_last` 最接近可用，但其表现也未稳定为正，说明不能继续只靠训练曲线或 mixed reward 选型。

### 单一推荐下一步
- 下一步优先级应从“继续扫超参”切到“补部署对齐指标”：
  - 在训练中周期性跑一个小型 pure-student eval（固定 deterministic + 低成本 steps），
  - 用这个部署对齐指标选 checkpoint，
  - 再在该口径下做小网格调参。

## 25. 2026-03-22 追加（P5 v18：训练内 pure-student 选模对齐验证）
### 目标子项
- 在不改 teacher-student 主结构的前提下，给 action-chunk 加一条最小的 training-time deployment-aligned selector，验证 “reward 选模错位” 是否是当前主瓶颈。

### 本次代码改动（最小且可逆）
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 改动：
  - 新增 pure-student phase meter：`student_mean_eps_reward / student_step_reward / student_tracking_active`
  - 仅当 `teacher_mix_ratio <= 0` 后才开始统计；
  - 只统计 pure-student phase 之后重新开始的 episode；
  - `model_best_student_reward` 改为基于该 meter 保存。
- 行为影响：
  - `Best Student Reward` 不再混入 teacher-mix 阶段 reward；
  - 原 `model_last` / `model_best_student` / `model_best` 机制保持不变。

### 本次验证（可复现）
- 轻量 smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 smoke_actionchunk_studenteval_align_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=5000 ++train.ppo.action_chunk_model_selection_warmup_steps=5000 ++train.ppo.action_chunk_ckpt_interval_steps=20000 ++task.env.numEnvs=24`
  - 结果：训练正常；`Best Student Reward` 在 pure-student phase 后从 `N/A` 变为数值，说明新链路已接通。
- 15min 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
  - 训练结果：
    - `Current Best=1166.49`
    - `Best Student Reward=50.85`
    - `Best Student MSE=0.3953`
- 单 seed nominal 评测（seed42, 512 steps）：
  - `model_last.ckpt`：`avg_reward=-0.388507`, `avg_done_rate=0.012248`
  - `model_best_student_reward.ckpt`：`avg_reward=-0.899636`, `avg_done_rate=0.014689`
  - `model_best_student.ckpt`：`avg_reward=-0.253256`, `avg_done_rate=0.013550`

### 本次结论
- 这轮已经能确认两件事：
  1. 训练内 reward 选模的统计口径现在更干净了，机制层问题被修掉了。
  2. 但在当前 15min 窗口下，pure-student episode reward 仍然不是一个足够强的 selector；`model_best_student_reward` 反而比 `model_last` 和 `model_best_student` 更差。
- 当前这组三个 ckpt 的最小排序是：
  - 最好：`model_best_student`
  - 次之：`model_last`
  - 最差：`model_best_student_reward`
- 因此 action-chunk 现在的下一阻塞不是“reward meter 接错”，而是“短窗 reward 仍不足以稳定代表部署质量”。

### 单一推荐下一步
- 不再继续围绕 `student_reward_meter` 小修小补；下一步应把 `model_best_student` 作为当前默认导出候选，并尝试一个更直接的 deterministic deployment proxy（比 episodic reward 更贴近最终评测）。

## 26. 2026-03-22 追加（P5 v19：ActionChunk len=4 试探 + Latent deterministic 迁移）
### 目标子项
- 继续沿 action-chunk 主线做最小改进验证，判断“缩短预测时域”是否能提升部署表现。
- 同时把 action-chunk 已验证有效的 deterministic 推理经验迁移到 latent diffusion，检查是否存在共享随机源问题。

### 本次代码改动（最小且可逆）
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 改动：
  - 新增 `diffusion_stochastic_infer`（默认 `False`）；
  - 当 stochastic 关闭时，latent diffusion 推理使用固定 `x_T=0` 初始化，并关闭中间步随机噪声。
- 影响：
  - latent 评测路径现在也具备真正 deterministic 语义；
  - 不改训练入口，不改 checkpoint 格式。

### 本次执行与验证（可复现）
- 语法/编译检查：
  - `bash -n scripts/eval_screwdriver_student_robustness.sh`
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`
- ActionChunk `len=4` 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" ++train.ppo.action_chunk_len=4 train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- ActionChunk `len=4` 单 seed nominal（seed42, 512 steps）：
  - 注意：评测时必须加 `++train.ppo.action_chunk_len=4`，否则会按默认 `len=8` 构图并报 shape mismatch。
  - `model_best_student.ckpt`：`avg_reward=-0.893365`, `avg_done_rate=0.014689`
  - `model_last.ckpt`：`avg_reward=-0.471724`, `avg_done_rate=0.014242`
  - `model_best_student_reward.ckpt`：`avg_reward=-0.328458`, `avg_done_rate=0.014974`
- Latent diffusion 对照（同一 ckpt，seed42, 512 steps）：
  - deterministic 默认：`avg_reward=1.682415`, `avg_done_rate=0.001058`
  - stochastic 显式开启：`avg_reward=1.626398`, `avg_done_rate=0.001302`

### 本次结论
- `chunk_len=4` 没有把 action-chunk 的绝对表现拉起来，当前仍未追上 `len=8` 的最佳部署候选（`len=8 model_best_student = -0.253256`）。
- 但 `len=4` 的 selector 排序变成了：
  - 最好：`model_best_student_reward`
  - 次之：`model_last`
  - 最差：`model_best_student`
- 这说明“缩短时域”对缓解 selector 失配可能有帮助，但它不是主解。
- latent diffusion 也存在同类推理随机源问题；迁移 deterministic 修复后，在当前 seed42/512 下 deterministic 略优于 stochastic，因此这条经验值得保留。

### 单一推荐下一步
- 对 action-chunk，不再继续盲目扫 `chunk_len`；下一步应实现一个低成本 deterministic deploy probe，并在训练中周期性运行它来选 ckpt。当前 action-chunk 的参考候选仍保留 `len=8 / model_best_student`，latent 则默认保留 deterministic 推理路径。

## 27. 2026-03-23 追加（P5 v20：ActionChunk 训练内 deterministic deploy probe）
### 目标子项
- 沿上一轮 handoff 的单一建议推进：实现一个训练内、低成本、deterministic 的 deploy proxy，用来替代当前不稳定的 action-chunk 选模信号。

### 本次代码改动（最小且可逆）
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 改动摘要：
  - 新增 `pure_student_window_len`
    - 仅允许完全位于 pure-student 阶段的连续窗口进入 probe，避免 mixed-policy 历史污染；
  - 新增固定 deploy probe buffer
    - 收集 pure-student 访问分布下的固定 `obs / proprio_hist / target_chunk`；
    - buffer 填满后冻结，训练期间周期性重复评估；
  - 新增 `model_best_deploy_probe`
    - 以 deterministic 推理在固定 probe buffer 上计算 first-action MSE；
    - 若优于历史最佳，则保存 `model_best_deploy_probe.ckpt`。

### 本次验证（可复现）
- 语法/编译：
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_action_chunk_student.py', cfile='/tmp/diffusion_action_chunk_student.pyc', doraise=True) ... PY`
- 3min smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 smoke_actionchunk_deploy_probe_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=5000 ++train.ppo.action_chunk_model_selection_warmup_steps=5000 ++train.ppo.action_chunk_ckpt_interval_steps=2000 ++train.ppo.action_chunk_deploy_probe_size=64 ++train.ppo.action_chunk_deploy_probe_batch_size=32 ++train.ppo.action_chunk_deploy_probe_interval_steps=2000 ++task.env.numEnvs=24`
- smoke 关键结果：
  - 训练日志已稳定打印：`Best Deploy Probe: 0.4382`
  - 输出目录已生成：
    - `model_best_deploy_probe.ckpt`
    - `model_best.ckpt`
    - `model_best_student.ckpt`
    - `model_best_student_reward.ckpt`
    - `model_last.ckpt`

### 本次结论
- 这轮已经把“更直接的 deterministic deployment proxy”从想法变成可运行实现。
- 该 proxy 不是完整 rollout reward，但它具备三点当前最需要的性质：
  - deterministic
  - 固定条件集
  - pure-student 分布对齐
- 因此它比当前 action-chunk 的两条旧 selector 更适合进入下一轮正式 15min 比较：
  - `student_action_mse`：太在线、太局部；
  - `student_eval_episode_reward`：太稀疏、太噪声。

### 单一推荐下一步
- 直接用当前最强 action 配置 `alignsel_v1 len8` 跑一轮正式 15min（保留新 deploy probe），然后统一评测：
  - `model_best_student`
  - `model_best_student_reward`
  - `model_best_deploy_probe`
- 目标不是再看训练曲线，而是验证 `model_best_deploy_probe` 是否第一次能在 nominal 评测里超过现有 `model_best_student`。

## 28. 2026-03-23 追加（P5 v21：deploy probe 正式 15min 结论）
### 目标子项
- 用正式 15min 训练验证：`model_best_deploy_probe` 是否真的比现有 `model_best_student` 更接近最终 nominal 部署评测。

### 本次执行（可复现）
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 ++train.ppo.action_chunk_deploy_probe_size=512 ++train.ppo.action_chunk_deploy_probe_batch_size=256 ++train.ppo.action_chunk_deploy_probe_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- 训练窗口末端观测：
  - `Current Best=1166.49`
  - `Best Student Reward=50.85`
  - `Best Student MSE=0.3953`
  - `Best Deploy Probe=0.4237`
- 单 seed nominal 评测（seed42, 512 steps）：
  - `model_best_student.ckpt`：`avg_reward=-0.253256`, `avg_done_rate=0.013550`
  - `model_last.ckpt`：`avg_reward=-0.388507`, `avg_done_rate=0.012248`
  - `model_best_deploy_probe.ckpt`：`avg_reward=-0.590261`, `avg_done_rate=0.014242`
  - `model_best_student_reward.ckpt`：`avg_reward=-0.899636`, `avg_done_rate=0.014689`

### 本次结论
- 这轮已经可以明确否掉当前这版 deploy probe：
  - 它确实更 deterministic、也更稳定；
  - 但它没有选出更好的环境部署 ckpt，反而比 `model_best_student` 更差。
- 当前 action-chunk 的 selector 排序重新稳定为：
  - 最好：`model_best_student`
  - 次之：`model_last`
  - 再次：`model_best_deploy_probe`
  - 最差：`model_best_student_reward`
- 所以 action-chunk 当前的主要问题已经不再像“选模口径小 bug”，而更像 teacher imitation proxy 和最终 contact-rich reward 之间存在结构性错位。

### 单一推荐下一步
- 不再优先继续修 action-chunk selector；论文主线应回到 latent diffusion 作为 diffusion 主实现，action-chunk 作为失败模式/分析分支保留。若后续还救 action，应直接尝试更高一级改法，例如真正的小 rollout probe 或 residual-style diffusion，而不是继续堆 MSE 类 proxy。

## 29. 2026-03-23 追加（P5 v22：Latent 主线回归验证）
### 目标子项
- 按上一轮 handoff 的单一建议，回到 latent diffusion 主线，确认当前最强 latent ckpt 在 deterministic 口径下是否稳定可作为后续论文主代表。

### 本次执行（可复现）
- 候选单 seed 复核（seed42, nominal, 512 steps）：
  - `run_a_seed42_15min/model_best.ckpt`：`avg_reward=1.682415`, `avg_done_rate=0.001058`
  - `run_a_latent_robust_light_seed42_15min/model_best.ckpt`：`avg_reward=1.780555`, `avg_done_rate=0.001058`
  - `run_a_latent_robust_light_seed42_1h/model_best.ckpt`：`avg_reward=1.773617`, `avg_done_rate=0.000936`
- 正式多 seed 评测：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_det_multiseed512 "42,43,44"`
- multiseed512 结果：
  - seed42：`1.780555 / 0.001058`
  - seed43：`2.146149 / 0.000936`
  - seed44：`2.322781 / 0.000895`
  - aggregate：`reward_mean=2.083162`, `reward_std=0.225799`, `done_mean=0.000963`, `done_std=0.000069`

### 本次结论
- 当前 diffusion 主线代表应更新为：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- 这版 latent 已明显强于旧的 `run_a_seed42_15min` 代表，也基本追近 `PureBC`，说明当前 diffusion 主线是可用且稳定的。
- 我还补了统一 `seed42 / 256 steps` 的三挡评测：
  - nominal：`1.623421 / 0.001383`
  - light：`1.661758 / 0.001872`
  - hard：`1.412261 / 0.001953`
- 我继续补了 deterministic 的 `light / hard` 多 seed 统计（steps=512）：
  - light：`2.020480 ± 0.087508`
  - hard：`1.582382 ± 0.210612`
- 这说明 latent 主线在 nominal 和 light 档都已经相当稳，当前剩余差距主要集中在 hard 扰动，而不是整体训练完全不到位。

### 单一推荐下一步
- 以 `run_a_latent_robust_light_seed42_15min/model_best.ckpt` 为当前 diffusion 主代表，下一步优先做一次“面向 hard 档的 latent 定向增强”小实验，而不是泛化地继续拉长训练时长。

## 30. 2026-03-23 追加（P5 v23：Latent hard 档原因定位）
### 目标子项
- 找到 latent diffusion 在 hard 档表现和方差不佳的更直接原因，并验证当前最小可行增强方向是否真的有效。

### 本次执行（可复现）
- 先做代码级最小改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 增加 `train.ppo.diffusion_residual_base`，让 diffusion 只学习相对 `adapt_tconv` base latent 的 residual。
- 验证：
  - `py_compile` 通过
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过
  - 3 分钟 smoke 成功产出 `model_best.ckpt`
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_residual_base_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_residual_base=True`
  - 训练峰值：`Current Best=1772.90`
- 评测（seed42, steps=512）：
  - nominal：`avg_reward=1.531935`, `avg_done_rate=0.001099`
  - hard：`avg_reward=1.237797`, `avg_done_rate=0.001994`
- 与当前主代表 `run_a_latent_robust_light_seed42_15min/model_best.ckpt` 对比：
  - nominal：`1.780555 -> 1.531935`
  - hard：`1.394853 -> 1.237797`
- 然后继续做两条“分布失配”快速排查：
  - full hard-match 15min：
    - `task.env.randomization.obs_noise_e_scale=0.05`
    - `task.env.randomization.obs_noise_t_scale=0.025`
    - `task.env.forceScale=1.5`
    - `task.env.randomForceProbScalar=0.3`
    - 结果：训练吞吐约 `375 FPS`，`Current Best` 长时间只在 `0.00~17.05`
  - obs-noise hard only 15min：
    - 仅把 `obs_noise_e/t` 提到 hard，force 保持 `0.5/0.1`
    - 结果：训练同样长时间停留 `Current Best=0.00`
- 两条失败 run 都已及时停止，避免继续浪费 GPU。

### 本次结论
- 这轮已经把 hard 档主因进一步收敛到了：
  - 不是简单的 force 不够强；
  - 更像高 observation noise 一上来就会把 15min latent 蒸馏打穿。
- `residual-base` 方向当前不能作为有效增强：
  - 它会抬高训练期 `Current Best`；
  - 但真实 nominal 和 hard 评测都比现有主代表更差。
- “静态更强扰动训练”这一类思路，目前也已被进一步否掉：
  - full hard 不行；
  - 仅 hard obs-noise 也不行。

### 单一推荐下一步
- 下一步应实现一个最小 `obs-noise curriculum`（例如 warmup 或分段提升），先解决“高观测噪声直接压穿短预算蒸馏”的可学性问题；在此之前，latent 主代表继续保持为 `run_a_latent_robust_light_seed42_15min/model_best.ckpt`。

## 31. 2026-03-23 追加（P5 v24：obs-noise curriculum v1 与 1h hard 复核）
### 目标子项
- 在不改 latent 主结构的前提下，验证最小 `obs-noise curriculum` 是否能改善 hard 档；同时确认 hard 短板里到底有多少“训练预算不足”成分。

### 本次执行（可复现）
- 代码改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 加入可选 `obs-noise curriculum`
  - 新增配置：
    - `diffusion_obs_noise_curriculum`
    - `diffusion_obs_noise_curriculum_start`
    - `diffusion_obs_noise_curriculum_steps`
    - `diffusion_obs_noise_e_target`
    - `diffusion_obs_noise_t_target`
- 验证：
  - `py_compile` 通过（字节码输出到 `/tmp`）
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过
  - 3 分钟 smoke：
    - `Current Best=275.45`
    - 说明 curriculum 能避免“训练从一开始就被 hard obs-noise 压穿”
- 正式 15min curriculum：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_curr_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1432.96`
  - 评测（seed42, 512 steps）：
    - nominal：`1.673794 / 0.000854`
    - hard：`1.121324 / 0.001587`
- 然后补做 `run_a_latent_robust_light_seed42_1h` 的 hard 复核：
  - 单 seed hard：`1.629211 / 0.001587`
  - multiseed hard（42/43/44）：
    - seed42：`1.629211 / 0.001587`
    - seed43：`1.512188 / 0.001058`
    - seed44：`1.410248 / 0.001180`
    - aggregate：`reward_mean=1.517216`, `reward_std=0.089462`

### 本次结论
- `curriculum v1` 机械上是成功的：
  - 它把“完全学不动”变成了“可以正常训练”
  - 但 hard 评测没有改善，反而低于当前 15min 主代表
- `1h robust-light` 说明 hard 确实含有训练预算成分：
  - seed42 hard 从 `1.394853` 提到 `1.629211`
  - 但 multiseed 平均没有超过 15min 主代表 `1.582382`
- 所以当前 hard 问题的最新判断是：
  - 既不是纯结构性不可学；
  - 也不是单纯多训一会儿就自然解决；
  - 更像“预算 + seed 交互 + 课程设计”共同作用

### 单一推荐下一步
- 下一步不要继续用 `curriculum v1` 的线性 schedule，改做更保守的两阶段/分段课程：
  - 前段保持现有 robust-light；
  - 中后段只先抬 `obs_noise_t`；
  - 最后再抬 `obs_noise_e`；
  并用 2-3 个 seed 的小样本 hard 评测及时验收，避免再被单 seed 假提升误导。

## 32. 2026-03-23 追加（P5 v25：staged-te curriculum 15min 验收）
### 目标子项
- 按“所有新策略先跑满 15min 再决定是否继续修改”的规则，验证更保守的 `staged_te` 课程是否优于线性 curriculum v1。

### 本次执行（可复现）
- 代码改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 为 `obs-noise curriculum` 增加：
    - `diffusion_obs_noise_curriculum_mode`
    - `diffusion_obs_noise_curriculum_t_phase_ratio`
  - 新模式 `staged_te`：
    - 前段先抬 `obs_noise_t`
    - 后段再抬 `obs_noise_e`
- 验证：
  - `py_compile` 通过
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_stage_te_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.65 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1611.54`
- 评测（seed42, 512 steps）：
  - nominal：`1.580571 / 0.001058`
  - hard：`1.206818 / 0.001221`

### 本次结论
- `staged_te` 比线性 curriculum v1 是进步的：
  - 训练峰值更高
  - hard 也更高
- 但它仍未超过当前 15min 主代表 `run_a_latent_robust_light_seed42_15min/model_best.ckpt`：
  - mainline nominal：`1.780555`
  - mainline hard：`1.394853`
- 所以这轮策略结论是：
  - 方向比线性 curriculum 更对；
  - 但 15min 验收仍不通过，不能升成新主线。

### 单一推荐下一步
- 如果继续做 curriculum，下一轮要在 `staged_te` 基础上再加一个“前段纯 light 保持期”，而不是继续改回线性 schedule；同时继续坚持每个新策略都必须先完成 15min 验证后再继续改。

## 33. 2026-03-23 追加（P5 v26：staged-hold-te curriculum 15min 验收）
### 目标子项
- 继续按 15min 先验收原则，验证更保守的三阶段课程 `hold_light -> raise_t -> raise_e` 是否能在 hard 上进一步改善且不明显伤 nominal。

### 本次执行（可复现）
- 代码改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 新增：
    - `diffusion_obs_noise_curriculum_hold_ratio`
  - 新模式：`diffusion_obs_noise_curriculum_mode=staged_hold_te`
- 验证：
  - `py_compile` 通过
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.25 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.7 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1651.39`
- 评测（seed42, 512 steps）：
  - nominal：`1.396033 / 0.001424`
  - hard：`1.261073 / 0.001546`

### 本次结论
- 这轮依然没有通过当前 15min 验收：
  - nominal 和 hard 都没超过现有主代表 `run_a_latent_robust_light_seed42_15min`
- 但它也提供了新的定位信息：
  - 相比 `staged_te`，hard 继续改善了
  - 但 nominal 下降更明显
- 当前 hard curriculum 的问题已经更具体了：
  - 不是“完全没方向”
  - 而是“hard 提升与 nominal 保持”之间存在显著 trade-off

### 单一推荐下一步
- 如果继续试 curriculum，下一轮不要再增加 hold 强度，而应改成：
  - 缩短 hold；
  - 保留 `t -> e` 分阶段；
  - 同时降低 `e` 的最终目标或把 `e` 进入时机继续后移，
  依然先跑满 15min 再决定是否继续。

## 34. 2026-03-23 追加（P5 v27：staged-hold-te(e035) 15min 验收）
### 目标子项
- 在不改代码结构的前提下，继续做最小策略调参，验证更低的 `e_target` 是否能改善 current curriculum 的 nominal/hard trade-off。

### 本次执行（可复现）
- 本轮没有改代码，只改策略配置：
  - `mode=staged_hold_te`
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.80`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.035`
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_e035_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.80 +train.ppo.diffusion_obs_noise_e_target=0.035 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1689.30`
- 评测（seed42, 512 steps）：
  - nominal：`1.484588 / 0.001058`
  - hard：`1.316540 / 0.001831`

### 本次结论
- 这轮是目前 curriculum 线里最好的一个配置：
  - 相比上一版 `staged_hold_te`
    - nominal：`1.396033 -> 1.484588`
    - hard：`1.261073 -> 1.316540`
- 但它仍然没有超过当前主代表：
  - mainline nominal：`1.780555`
  - mainline hard：`1.394853`
- 所以现在最重要的判断是：
  - 课程方向本身不是错的；
  - 降低 `e_target` 确实在改善 trade-off；
  - 但离“通过 15min 验收”还有一段距离。

### 单一推荐下一步
- 如果继续推进，下一轮优先继续沿“更弱 `e`、更强调 `t`”做最小调参，例如把 `e_target` 再降到 `0.03` 或把 `e` 进入时机进一步后移；仍然保持先跑满 15min、再做 nominal+hard 验收的节奏。

## 35. 2026-03-23 追加（P5 v28：staged-hold-te(e03) 15min 验收）
### 目标子项
- 继续沿“更弱 `e`、更强调 `t`”方向做最小调参，验证把 `obs_noise_e_target` 从 `0.035` 进一步降到 `0.03` 是否还能继续改善 nominal/hard trade-off。

### 本次执行（可复现）
- 本轮没有改代码，只改策略配置：
  - `mode=staged_hold_te`
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.80`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.03`
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_e03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.80 +train.ppo.diffusion_obs_noise_e_target=0.03 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1689.30`
- 评测（seed42, 512 steps）：
  - nominal：`1.484588 / 0.001058`
  - hard：`1.316540 / 0.001831`

### 本次结论
- 把 `e_target` 从 `0.035` 再降到 `0.03` 没有带来进一步收益：
  - nominal 与 hard 都与上一轮 `e035` 相同
- 这说明当前“单纯继续压低 `e_target`”这条线大概率已经碰到平台期
- 当前 curriculum 线最有价值的候选仍然是：
  - `run_a_latent_obs_hold_te_e035_seed42_15min`
- 但主代表依然不变：
  - `run_a_latent_robust_light_seed42_15min/model_best.ckpt`

### 单一推荐下一步
- 如果继续推进，下一轮不要再继续单独减小 `e_target`，而应优先测试“进一步后移 `e` 的进入时机”或先补 `e035` 的 multiseed hard 复核，再决定是否继续改课程。

## 36. 2026-03-24 追加（P5 v29：staged-hold-te(t90,e035) 15min 验收）
### 目标子项
- 在不继续压低 `e_target` 的前提下，验证“进一步后移 `e` 的进入时机”是否能改善 `staged_hold_te(e035)` 的 nominal/hard trade-off。

### 本次执行（可复现）
- 本轮没有改代码，只改策略配置：
  - `mode=staged_hold_te`
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.90`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.035`
- 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_t90_e035_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.90 +train.ppo.diffusion_obs_noise_e_target=0.035 +train.ppo.diffusion_obs_noise_t_target=0.025`
  - 训练峰值：`Current Best=1866.55`
- 评测（seed42, 512 steps）：
  - nominal：`1.388510 / 0.000651`
  - hard：`1.195002 / 0.001587`

### 本次结论
- 单纯把 `e` 再往后推并没有带来收益，反而比 `staged_hold_te(e035)` 更差：
  - nominal：`1.484588 -> 1.388510`
  - hard：`1.316540 -> 1.195002`
- 这说明当前课程线的主要瓶颈并不是“`e` 进入得还不够晚”，至少在这条单轴调参上已经不成立。
- 当前最有价值的 curriculum 候选仍然是：
  - `run_a_latent_obs_hold_te_e035_seed42_15min`
- 当前 diffusion 主代表仍不变：
  - `run_a_latent_robust_light_seed42_15min/model_best.ckpt`

### 单一推荐下一步
- 如果继续推进，下一轮不要再在 `staged_hold_te` 上继续做单轴 `t/e` 时序微调，而应优先补 `run_a_latent_obs_hold_te_e035_seed42_15min` 的 multiseed hard 复核，确认它的提升是不是稳定现象，再决定是否需要换更高一级的鲁棒性策略。

## 37. 2026-03-24 追加（P5 v30：staged-hold-te(e035) hard multiseed 复核）
### 目标子项
- 对当前最有价值的 curriculum 候选 `run_a_latent_obs_hold_te_e035_seed42_15min` 做 `hard multiseed512` 复核，判断它在强扰动下的改善是否稳定，而不是单 seed 偶然。

### 本次执行（可复现）
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e035_hard_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
- 结果：
  - seed42：`1.286080 / 0.001506`
  - seed43：`1.512002 / 0.000854`
  - seed44：`1.318451 / 0.001261`
  - aggregate：`reward_mean=1.372178`, `reward_std=0.099750`, `done_mean=0.001207`, `done_std=0.000269`

### 本次结论
- `e035` 的 hard 改善不具备足够竞争力：
  - 它低于当前主代表 `run_a_latent_robust_light_seed42_15min` 的 hard multiseed512
  - 主代表 hard multiseed512：`1.582382 ± 0.210612`
  - `e035` hard multiseed512：`1.372178 ± 0.099750`
- 这说明当前 curriculum 线即便在最优候选上，也还没有形成稳定收益。
- 到这一步可以更明确地说：
  - `staged_hold_te` 系列目前只是“分析性尝试”，不是可升级主线的候选。

### 单一推荐下一步
- 如果继续推进 latent 主线，下一轮不要再优先投入 `obs-noise curriculum` 微调，而应转向更高一级的鲁棒性策略，或先回到当前主代表 `run_a_latent_robust_light_seed42_15min` 做非 curriculum 方向的增强。

## 38. 2026-03-24 追加（P5 v31：latent joint-decoder 非 curriculum 尝试失败）
### 目标子项
- 按 `P5` 非 curriculum 鲁棒性策略方向，验证“轻度解冻 student decoder”能否改善 latent diffusion 在强扰动下的部署表现。

### 本次执行（可复现）
- 代码改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 增加可选参数 `train.ppo.diffusion_student_trainable_param_patterns`
  - 允许 diffusion 训练时额外解冻匹配到的 student 参数，并并入同一个 Adam 优化器
- 语法验证：
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_joint_decoder_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_student_trainable_param_patterns=[actor_mlp,mu]`
  - 训练窗口内 `Current Best` 长时间贴地，最终只到 `8.03`

### 本次结论
- 这条“联合解冻 actor_mlp + mu”分支明显破坏了 latent diffusion 训练稳定性。
- 失败原因更像是 student decoder 和 diffusion head 同步共训带来的优化耦合，而不是评测口径问题。
- 因此这条线应被归档为失败分支，不再作为当前 latent 主线候选。

### 单一推荐下一步
- 不再继续 joint-decoder 共训；回到 frozen decoder 主干，尝试更保守的非 curriculum 监督增强。

## 39. 2026-03-24 追加（P5 v32：latent_recon 非 curriculum 增强成功）
### 目标子项
- 在保持 decoder 冻结、保持 `robust_light` 基线不变的前提下，验证给 diffusion latent 增加直接 `x0/latent` 重建监督，是否能带来更稳定的 nominal + hard 收益。

### 本次执行（可复现）
- 代码改动：
  - 在 `dexscrew/algo/ppo/diffusion_latent_student.py` 增加可选参数 `train.ppo.diffusion_latent_recon_coef`
  - 训练时新增 `latent_recon_loss = MSE(tanh(x0_pred), teacher_latent)`，并写入 tensorboard
- 语法验证：
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`
- 正式 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5`
  - 训练峰值：`Current Best=1786.12`
- 最小单 seed 验收（seed42, 512 steps）：
  - nominal：
    - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_nominal_seed42`
    - `avg_reward=2.163711`, `avg_done_rate=0.000854`
  - hard：
    - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_hard_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
    - `avg_reward=1.467671`, `avg_done_rate=0.001546`
- multiseed512 复核：
  - nominal：
    - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_nominal_ms512 '42,43,44'`
    - aggregate：`reward_mean=2.272142`, `reward_std=0.078250`, `done_mean=0.000786`, `done_std=0.000051`
  - hard：
    - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_hard_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
    - aggregate：`reward_mean=1.650344`, `reward_std=0.149604`, `done_mean=0.001261`, `done_std=0.000202`

### 本次结论
- 这次是当前 latent 主线的实质性升级，不是单 seed 偶然：
  - 旧主代表 nominal multiseed512：`2.083162 ± 0.225799`
  - 新 `latent_recon05` nominal multiseed512：`2.272142 ± 0.078250`
  - 旧主代表 hard multiseed512：`1.582382 ± 0.210612`
  - 新 `latent_recon05` hard multiseed512：`1.650344 ± 0.149604`
- 相比之前的 curriculum 微调，这条线同时提高了 nominal 和 hard，且方差更低。
- 因此当前 `DiffusionLatentStudent` 的主代表应升级为：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

### 单一推荐下一步
- 以 `latent_recon05` 为新的 diffusion latent 主代表，下一轮优先围绕它补 `light multiseed512`，确认这次升级在统一三挡扰动口径下是否全面成立。

## 40. 2026-03-24 追加（P5 v33：latent_recon05 light multiseed512 复核）
### 目标子项
- 按上一轮 handoff 推荐，补齐 `latent_recon05` 在 `light` 档的 multiseed512 结果，确认这次升级是否在统一三挡扰动口径下全面成立。

### 本次执行（可复现）
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_light_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
- 结果：
  - seed42：`1.731917 / 0.001383`
  - seed43：`2.009683 / 0.001017`
  - seed44：`1.980781 / 0.000977`
  - aggregate：`reward_mean=1.907460`, `reward_std=0.124687`, `done_mean=0.001126`, `done_std=0.000183`

### 本次结论
- `latent_recon05` 并没有形成“三挡统一升级”：
  - 旧主代表 `run_a_latent_robust_light_seed42_15min`
    - light multiseed512：`2.020480 ± 0.087508`
  - 新候选 `run_a_latent_recon05_seed42_15min`
    - light multiseed512：`1.907460 ± 0.124687`
- 所以当前更准确的判断是：
  - `latent_recon05` 明确提升了 `nominal + hard`
  - 但它牺牲了一部分 `light`
  - 这是一条“鲁棒性 trade-off 更偏 hard”的增强，而不是无条件全面更优

### 单一推荐下一步
- 保持 `latent_recon` 这条线继续推进，但下一轮不要直接加大监督，而应优先做一个更小的 `diffusion_latent_recon_coef` 回扫（例如 `0.3`），目标是在保住 `hard` 改善的同时把 `light` 拉回到旧主代表附近。

## 41. 2026-03-24 追加（P5 v34：latent_recon 小系数回扫失败）
### 目标子项
- 按上一轮 handoff 推荐，验证把 `diffusion_latent_recon_coef` 从 `0.5` 下调到更小系数后，是否能在保持 `nominal + hard` 收益的同时恢复 `light`。

### 本次执行（可复现）
- 正式 15min：
  - `recon03`
    - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.3`
    - 训练峰值：`Current Best=1759.20`
  - `recon04`
    - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon04_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.4`
    - 训练峰值：`Current Best=1876.57`
- 单 seed 三挡验收（seed42, 512 steps，串行评测）：
  - `recon03`
    - nominal：`avg_reward=1.764286`, `avg_done_rate=0.001139`
    - light：`avg_reward=1.604022`, `avg_done_rate=0.001546`
    - hard：`avg_reward=1.307460`, `avg_done_rate=0.001546`
  - `recon04`
    - nominal：`avg_reward=1.728607`, `avg_done_rate=0.001099`
    - light：`avg_reward=1.565717`, `avg_done_rate=0.001343`
    - hard：`avg_reward=1.260885`, `avg_done_rate=0.001587`

### 本次结论
- 这条“小系数回扫恢复 light”的假设已被当前 15min 口径否定：
  - `recon03` 和 `recon04` 都没有超过 `latent_recon05`
  - 而且两者在 `nominal / light / hard` 三挡下都弱于 `recon05`
- 训练峰值更高并不代表最终部署更好：
  - `recon04` 的 `Current Best=1876.57` 高于 `recon05=1786.12`
  - 但最终三挡评测全部更差
- 因此当前 `latent_recon` 线的最优代表仍然保持为：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

### 单一推荐下一步
- 暂停继续向下扫 `diffusion_latent_recon_coef`；保留 `latent_recon05` 主代表，转做更高信息量的 `light gap` 诊断或更结构化的增强，而不是继续在 `0.3/0.4` 这段系数上盲试。

## 42. 2026-03-24 追加（P5 v35：student rollout 诊断打通 + light 对比口径纠正）
### 目标子项
- 把 `DiffusionLatentStudent` 的 rollout 诊断链路打通，并纠正 `latent_recon05` 与 `robust_light` 在 `light` 档上的历史对比口径不一致问题。

### 本次执行（可复现）
- 代码改动：
  - `dexscrew/algo/ppo/padapt.py`
    - 为 student 分支新增 `collect_rollout(...)`
    - rollout 现在可导出 `extras` 标量时间序列（如 `rotation_reward`、`screw/angular_velocity` 等）
  - `dexscrew/algo/ppo/ppo.py`
    - teacher/PPO 分支的 `collect_rollout(...)` 同步支持导出 `extras`
  - `scripts/analyze_rollout_diagnostics.py`
    - 新增最小分析脚本，可对两份 rollout `.pt` 进行 `early/mid/late` 对照
- 语法验证：
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/padapt.py', ...) ... PY`
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/ppo.py', ...) ... PY`
  - `python - <<'PY' ... py_compile.compile('scripts/analyze_rollout_diagnostics.py', ...) ... PY`
- seed42 `light` rollout 采样（新口径 `0.03 / 0.015 + force 1.0 / prob 0.2`）：
  - `robust_light`：
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_robust_light_light_seed42_steps512.pt`
  - `latent_recon05`：
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_recon05_light_seed42_steps512.pt`
  - 对照分析：
    - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_robust_light_light_seed42_steps512.pt outputs/diagnostics/latent_recon05_light_seed42_steps512.pt`
- 关键纠偏评测：
  - 之前文档把 `latent_recon05 light`（`0.03 / 0.015 + 1.0 / 0.2`）和旧 `robust_light light`（`0.02 / 0.01 + 0.5 / 0.1`）直接比较，口径不一致。
  - 已补做对齐后的 `robust_light light_v2 multiseed512`：
    - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_light_v2_multiseed512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
    - 结果：
      - seed42：`1.873305 / 0.001302`
      - seed43：`1.885252 / 0.001017`
      - seed44：`2.048657 / 0.000854`
      - aggregate：`reward_mean=1.935738`, `reward_std=0.079995`, `done_mean=0.001058`, `done_std=0.000185`

### 本次结论
- 之前“`latent_recon05` 在 light 明显更差”的说法过强，主要因为比较口径不一致。
- 在对齐后的 `light_v2` 口径下：
  - `robust_light`：`1.935738 ± 0.079995`
  - `latent_recon05`：`1.907460 ± 0.124687`
  - 均值差只有 `0.028278`
- 这说明当前问题更像是：
  - `recon05` 在 nominal / hard 上的收益是真实的
  - `light` 上并不存在“大幅均值塌陷”
  - 更接近“轻扰动下的 seed 稳定性/方差问题”
- seed42 的 rollout 质性诊断也支持“阶段性 trade-off”而非全程退化：
  - `recon05` 在中段 `rotation_reward / angular_velocity / positive_vel_ratio` 更强
  - 但后段 `reward_per_step / angular_position` 反而弱于 `robust_light`

### 单一推荐下一步
- 不再把 `light` 当成 `latent_recon05` 的明显短板；下一轮应利用新 rollout 诊断链路，对 `seed42` 与一个回落 seed（优先 `seed44`）做同口径 `light_v2` 阶段对照，定位差异究竟来自后段保持、接触恢复，还是 seed 触发的阶段切换不稳定。

## 43. 2026-03-24 追加（P5 v36：light_v2 seed44 阶段诊断）
### 目标子项
- 按上一轮 handoff 推荐，用 student rollout 诊断链路对 `seed42` 与回落更明显的 `seed44` 做同口径 `light_v2` 阶段对照，判断 `latent_recon05` 的波动到底来自中后段保持、接触恢复，还是 seed 触发的轨迹分叉。

### 本次执行（可复现）
- 采样 `seed44` 的 `light_v2` rollout（`obs_noise=0.03/0.015`, `force=1.0`, `prob=0.2`）：
  - `robust_light`
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... seed=44 ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_robust_light_light_seed44_steps512.pt`
    - `mean_reward=1.0075`, `mean_done_rate=0.0018`
  - `latent_recon05`
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... seed=44 ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_recon05_light_seed44_steps512.pt`
    - `mean_reward=0.8812`, `mean_done_rate=0.0016`
- 三组对照分析：
  - 同 seed44 的模型对照：
    - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_robust_light_light_seed44_steps512.pt outputs/diagnostics/latent_recon05_light_seed44_steps512.pt`
  - `latent_recon05` 的 `seed42 -> seed44`：
    - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_recon05_light_seed42_steps512.pt outputs/diagnostics/latent_recon05_light_seed44_steps512.pt`
  - `robust_light` 的 `seed42 -> seed44`：
    - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_robust_light_light_seed42_steps512.pt outputs/diagnostics/latent_robust_light_light_seed44_steps512.pt`

### 本次结论
- `robust_light` 自身也有 eval-seed 波动，但它在 `seed44` 反而更强：
  - `seed42 -> seed44` 的 `reward_per_step`：`0.899332 -> 1.007452`
  - 提升主要集中在中段：`0.832410 -> 1.082685`
- `latent_recon05` 在 `seed44` 出现的是“中段发力失败”，不是单纯后段崩：
  - `seed42 -> seed44` 的 `reward_per_step`：
    - early：`0.843274 -> 0.867773`（略升）
    - mid：`0.966963 -> 0.869660`（明显下降）
    - late：`0.913008 -> 0.906077`（基本持平）
- 同一 `seed44` 下，`latent_recon05` 相比 `robust_light` 的主要差距也集中在中段：
  - `reward_per_step`：mid `0.869660 vs 1.082685`
  - `rotation_reward`：mid `0.397807 vs 0.433682`
  - 同时 `pose_diff_penalty` 更高：mid `0.453893 vs 0.433671`
  - `torques` 更高：mid `0.351320 vs 0.318314`
- 这说明当前 `recon05` 的 `light_v2` 波动更像是：
  - 某些 seed 下会进入一种“中段动作更激进但奖励对齐更差”的模式
  - 它并不表现为 done 变差，反而 done_rate 更低
  - 更像 reward-shaping 对齐问题或接触/姿态代价过高，而不是简单“任务失败”

### 单一推荐下一步
- 下一轮不要再优先做通用超参扫描；应直接围绕 `latent_recon05` 做一个最小的“reward-aligned 中段稳态增强”试探，优先抑制中段的 `pose_diff_penalty / torques` 抬升，而不是继续改 `recon_coef`。

## 44. 2026-03-24 追加（P5 v37：reward-aligned 最小试探失败归档）
### 目标子项
- 按上一轮 handoff 推荐，围绕 `latent_recon05` 做两条最小的 reward-aligned 中段稳态增强试探，先看是否能在不大改结构的前提下修复 `light_v2` 下的中段对齐问题。

### 本次执行（可复现）
- 试探 A：增大 teacher action BC 权重
  - 15min 训练：
    - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_bc15_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 train.ppo.bc_loss_coef=1.5`
    - 峰值：`Current Best=1846.98`
  - `light_v2` 定向验收：
    - seed42：`1.601600 / 0.001628`
    - seed44：`1.863312 / 0.001221`
- 试探 B：新增 base-student 动作锚定正则
  - 代码改动：
    - `dexscrew/algo/ppo/diffusion_latent_student.py`
      - 新增 `train.ppo.diffusion_base_action_anchor_coef`
      - 训练时可选加入 `base_action_anchor_loss`，约束 diffusion student 动作不要偏离 frozen `adapt_tconv` student 动作太多
  - 语法与 smoke：
    - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
    - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_baseanchor03_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_base_action_anchor_coef=0.3 task.env.numEnvs=8`
  - 15min 训练：
    - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_baseanchor03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_base_action_anchor_coef=0.3`
    - 峰值：`Current Best=1821.35`
  - `light_v2` multiseed 定向验收：
    - seed42：`1.570475 / 0.001180`
    - seed43：`1.891393 / 0.000936`
    - seed44：`2.040849 / 0.000854`
    - aggregate：`reward_mean=1.834239`, `reward_std=0.196244`, `done_mean=0.000990`

### 本次结论
- `bc15` 可以直接判为失败：
  - 虽然训练峰值更高，但 `light_v2` 的 seed42 和 seed44 都低于 `latent_recon05` 原版（`1.731917 / 1.980781`）。
- `base-action anchor` 不能直接升级为主线：
  - 它只明显帮到了 `seed44`（`2.040849`，接近 `robust_light seed44=2.048657`）
  - 但 `seed42` 明显回退，三 seed 均值仍低于当前主代表 `latent_recon05` 的 `1.907460 ± 0.124687`
- 因此当前可收束的判断是：
  - “纯加 teacher BC” 不是正确方向
  - “适度向 base student 动作靠拢” 可能含有局部正信号，但当前实现过于粗糙，只是把收益转移到个别 seed，而没有提升整体稳定性

### 单一推荐下一步
- 下一轮不要继续盲扫 `bc_loss_coef` 或 `base_action_anchor_coef`；应直接用 rollout 诊断链路，对 `latent_recon05` 与 `baseanchor03` 在 `seed42/44 light_v2` 的中段行为做对照，确认 `seed44` 的改善是否真的来自 `pose_diff_penalty / torques` 回落，再据此决定是否值得做更直接的 torque-aware / action-magnitude regularizer。

## 45. 2026-03-24 追加（P5 v38：新阶段计划草案）
### 目标子项
- 在当前 `PLANS.md` 基本完成的前提下，起草一份新的执行计划草案，把后续主线明确收缩到“`latent diffusion` 追平/逼近 `adapt`”。

### 本次改动
- 新增：
  - `plan2.md`
- 计划草案要点：
  - 冻结 `latent_recon05` 为当前 diffusion-latent 主代表
  - 明确后续唯一核心问题是“为什么 latent 还没追平 adapt”
  - 把工作拆成：
    - 参考基线冻结
    - gap 诊断
    - 最小结构增强
    - 鲁棒性收敛
    - 论文可写性收敛
  - 明确升级规则：
    - 必须 15min + 统一口径 + multiseed
    - 训练峰值不得单独作为升级依据
  - 当前 immediate next step 固定为：
    - 先用 rollout diagnostics 比较 `latent_recon05` 与 `baseanchor03` 在 `seed42/44 light_v2` 的中段差异

### 本次验证
- `sed -n '1,260p' PLANS.md`
- `tail -n 120 docs/session_handoff.md`
- `tail -n 120 docs/stage_acceptance_summary.md`
- 新计划未改动任何现有训练/评测语义，仅作为下一阶段草案文件落地

### 本次结论
- 当前旧 `PLANS.md` 的“主线搭建/跑通”型目标基本已完成，后续更适合进入“定向优化 + 论文收敛”阶段。
- `plan2.md` 可以作为下一版计划的基础骨架，由你后续按需要继续裁剪和改写。

### 单一推荐下一步
- 先审阅并修改 `plan2.md`，把你想保留或删掉的研究目标定下来；之后再按新计划推进 `latent diffusion` 主线优化。

## 46. 2026-03-24 追加（执行总结会话：Plan v1 GPT 交接文档）
### 目标子项
- 产出一份面向 GPT 治理更新的 Plan v1 阶段总结（非研究扩展、非大规模代码实现）。

### 本次改动
- 新增：
  - `plan_summary_v1.md`
- 行为影响：
  - 不改训练/评测代码路径；
  - 提供结构化事实总结，区分“已完成”“已验证”“未验证/风险”和“下一阶段建议”，用于下一轮 GPT 更新 `AGENTS.md / PLANS.md`。

### 本次验证（只读核对）
- 已核对治理与阶段文档：
  - `AGENTS.md`
  - `PLANS.md`
  - `docs/session_handoff.md`
  - `docs/stage_acceptance_summary.md`
  - `plan2.md`
- 已核对活动代码与脚本入口：
  - `train.py`, `student_eval.py`
  - `dexscrew/algo/ppo/padapt.py`
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
  - `dexscrew/algo/ppo/diffusion_action_chunk_student.py`
  - `dexscrew/algo/ppo/ppo.py`
  - `scripts/screwdriver_student_*`, `scripts/collect_screwdriver_teacher_rollout*.sh`, `scripts/eval_screwdriver_student_robustness*.sh`
- 关键结果：
  - `plan_summary_v1.md` 已按 8 段固定结构落地；
  - 已明确当前证据主要来自文档与代码状态；
  - 当前工作区无 `outputs/` 目录，无法在本会话直接复核历史产物文件。

### 当前阻塞/风险
- 多份文档对同一指标存在口径/数值不一致（如 latent 峰值写法）；
- 当前会话无法直接访问历史 `outputs/...` 实物产物，证据可追溯性依赖文档记录；
- `plan2.md` 仍是草案，尚未转化为正式治理文件。

### 单一推荐下一步
- 由 GPT 先基于 `plan_summary_v1.md` 决定是否正式切换到 `plan2` 导向，并给出下一版 `PLANS.md` 的硬性验收口径（特别是 multiseed 与三挡扰动统一口径）。


## 47. 2026-03-24 追加（执行总结会话：plan_summary_v1 产出）
### 目标子项
- 完成 Plan v1 的结构化 handoff 文档，供 GPT 下一轮治理更新使用。

### 本次改动
- 新增：
  - `plan_summary_v1.md`
- 影响：
  - 不改变训练/评测/导出代码路径，仅新增总结文档。

### 本次验证
- 只读核对已执行：
  - `AGENTS.md`
  - `PLANS.md`
  - `docs/session_handoff.md`
  - `docs/stage_acceptance_summary.md`
  - `docs/stage_closure.md`
  - `docs/robustness_eval_summary.md`
  - `train.py`
  - `student_eval.py`
  - `dexscrew/algo/ppo/{ppo,padapt,diffusion_latent_student,diffusion_action_chunk_student}.py`
  - `scripts/` 关键训练/评测/rollout 脚本
- 快照结论：
  - `plan_summary_v1.md` 已按 8 段固定结构落地；
  - 当前仓库不含 `outputs/` 训练工件，历史结果复核主要依赖文档证据。

### 当前阻塞/风险
- 主线叙事存在文档层面的张力（PLANS 的 action-chunk-first 与近期 handoff 的 latent-mainline 倾向）。
- 证据可追溯性受限于工件未随仓库快照提供。

### 单一推荐下一步
- 由 GPT 先基于 `plan_summary_v1.md` 明确下一版 `PLANS.md` 的唯一主线与硬性证据规范（multiseed + 三挡口径 + 工件清单）。

## 48. 2026-03-24 追加（P5 v39：Plan v1 治理/证据汇总交接）
### 目标子项
- 面向 GPT 下一轮治理更新，完成一次“只读型”Plan v1 执行总结：对齐 AGENTS/PLANS、当前代码路径、文档证据与风险点，产出结构化交接文档。

### 本次改动
- 新增：
  - `plan_summary_v1.md`
- 影响：
  - 不改训练/评测代码语义；仅新增一份面向 GPT 的阶段总结工件，明确“已完成/已验证/未验证/风险/下一阶段方向”。

### 本次验证（可复现）
- `Get-Content AGENTS.md`
- `Get-Content -Encoding UTF8 PLANS.md`
- `Get-Content -Encoding UTF8 docs/session_handoff.md`
- `Get-Content -Encoding UTF8 docs/stage_acceptance_summary.md`
- `Get-Content -Encoding UTF8 train.py`
- `Get-Content -Encoding UTF8 dexscrew/algo/ppo/padapt.py`
- `Get-Content -Encoding UTF8 dexscrew/algo/ppo/diffusion_latent_student.py`
- `Get-Content -Encoding UTF8 dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- `rg -n "collect_rollout|rollout_pretrain|train.algo|DiffusionLatentStudent|DiffusionActionChunkStudent" dexscrew/algo/ppo scripts train.py student_eval.py`

关键结果：
- 代码路径层面：canonical Hora teacher-student + diffusion 路径与评测脚本齐全。
- 证据层面：当前仓库快照无 `outputs/` 目录，实验结论主要依赖文档记录而非本地工件复核。
- 风险层面：发现一个需治理确认的语义风险（`collect_rollout` 在 diffusion student 下是否真正走 diffusion 采样路径）。

### 当前阻塞/风险
- `PLANS.md`（action-chunk-first）与 `plan2.md`/最新 handoff（latent-first）存在主线叙事漂移。
- 本地缺少运行工件，导致部分“已完成”结论只能做文档级信任，不能工件级复核。
- rollout diagnostics 语义一致性需要在下一阶段先明确后再作为关键决策依据。

### 单一推荐下一步
- 先由 GPT 基于 `plan_summary_v1.md` 对 AGENTS/PLANS 做一次治理收敛（主线与验收证据标准统一），再进入下一轮执行。


## 46. 2026-03-24 追加（P5 v39：baseanchor 诊断完成 + action_l2 首轮失败）
### 目标子项
- 按 `plan2.md` 的 immediate next step，先完成 `latent_recon05` 与 `baseanchor03` 在 `seed42/44 + light_v2` 下的阶段诊断。
- 基于诊断结果，推进一个更直接的最小 latent 改动，而不是继续盲扫旧系数。

### 本次执行（可复现）
- 补齐 `baseanchor03` rollout：
  - seed42：
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... test=True seed=42 ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_baseanchor03_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_baseanchor03_light_seed42_steps512.pt`
    - `mean_reward=0.9031`, `mean_done_rate=0.0023`
  - seed44：
    - `./docker-run-isaacgym.sh timeout 1800 python train.py ... test=True seed=44 ... checkpoint=outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_baseanchor03_seed42_15min/stage2_diffusion_nn/model_best.ckpt ... +collect_rollout=True +collect_steps=512 +collect_save_point_cloud=False +collect_out=outputs/diagnostics/latent_baseanchor03_light_seed44_steps512.pt`
    - `mean_reward=0.8226`, `mean_done_rate=0.0021`
- 三组诊断对照：
  - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_recon05_light_seed42_steps512.pt outputs/diagnostics/latent_baseanchor03_light_seed42_steps512.pt`
  - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_recon05_light_seed44_steps512.pt outputs/diagnostics/latent_baseanchor03_light_seed44_steps512.pt`
  - `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_baseanchor03_light_seed42_steps512.pt outputs/diagnostics/latent_baseanchor03_light_seed44_steps512.pt`
- 基于诊断后的新改动：
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
    - 新增 `train.ppo.diffusion_action_l2_coef`
    - 新增 `action_l2_loss = mean(student_mu^2)` 的默认关闭正则
- 语法与 smoke：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_actionl2_001_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_action_l2_coef=0.01 task.env.numEnvs=8`
- 15min 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_actionl2_001_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_action_l2_coef=0.01`
  - 峰值：`Current Best=1824.54`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_actionl2_001_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_actionl2_001_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - 结果：`avg_reward=1.532299`, `avg_done_rate=0.001261`

### 本次结论
- `baseanchor03` 的阶段诊断说明：
  - seed42 下它不是简单“更稳”，而是把中段推进压弱了：
    - `reward_per_step mid: 0.966963 -> 0.908913`
    - `rotation_reward mid: 0.377728 -> 0.350038`
    - 但 late 反而更高：`0.913008 -> 0.971481`
  - seed44 下它确实降低了部分代价项：
    - `pose_diff_penalty mid: 0.453893 -> 0.418299`
    - `torques mid: 0.351320 -> 0.342736`
    - 但同时把中后段推进一起压低了：
      - `rotation_reward mid: 0.397807 -> 0.367355`
      - `angular_position late: 2.957677 -> 2.612854`
- 这说明 `baseanchor03` 的核心问题不是“方向完全错”，而是：
  - 它属于过粗的全局动作收缩
  - 能压一点代价，但会连推进能力一起压掉
- `action_l2=0.01` 也得到了同样方向的负结果：
  - 虽然训练峰值到 `1824.54`
  - 但 `light_v2 seed42/512 = 1.532299`
  - 低于 `latent_recon05` 原版 `1.731917`
  - 也低于 `baseanchor03 seed42 = 1.570475`

### 单一推荐下一步
- 下一轮不要再做“全局压动作幅度/全局向 base 靠拢”这类静态正则；如果继续优化 latent，应改做更有方向性的最小改动，例如只约束大幅动作尾部、只约束 student-teacher 偏差过大的样本，或进入阶段性论文收敛而不是继续扫通用正则。

## 47. 2026-03-24 追加（P5 v40：teacher-delta tail 首轮 15min 验收）
### 目标子项
- 按 v39 的 handoff 推荐，把 latent 正则从“全局静态收缩”进一步缩到更有方向性的 tail-only 约束，验证它是否比 `baseanchor03 / action_l2` 更接近正确方向。

### 本次改动
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_coef`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_threshold`
  - 新增
    - `teacher_delta_tail = relu(abs(student_mu - teacher_mu) - threshold)`
    - `teacher_delta_tail_loss = mean(teacher_delta_tail^2)`
  - 新增日志项：`teacher_delta_tail_loss`

### 本次执行（可复现）
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 task.env.numEnvs=8`
  - 结果：训练能正常启动并持续输出，无新的实现级报错
- 15min 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_t025_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25`
  - 峰值：`Current Best=1863.07`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_t025_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_t025_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - 结果：`avg_reward=1.602075`, `avg_done_rate=0.001506`

### 本次结论
- 这条更有方向性的 tail-only 正则，确实比前两条更粗的静态正则好一些：
  - `teacher_delta_tail seed42=1.602075`
  - `baseanchor03 seed42=1.570475`
  - `action_l2=0.01 seed42=1.532299`
- 但它仍未超过当前主代表 `latent_recon05` 原版：
  - `latent_recon05 seed42=1.731917`
- 因此当前可收束的判断是：
  - “只约束大偏差尾部”比“整体压动作/整体向 base 靠拢”更接近正确方向
  - 但当前实现仍然过于静态，它没有区分阶段，也没有区分哪些高偏差样本是真正有害的
  - 所以这轮不能升级主线，只能作为下一轮更细化定向正则的依据

### 当前风险 / 剩余问题
- `teacher_delta_tail` 仍然出现“训练峰值更高，但部署评测未同步提升”的熟悉现象：
  - `Current Best=1863.07` 高于 `latent_recon05=1786.12`
  - 但 `light_v2 seed42/512` 仍然更低
- 说明只靠训练峰值不能判断这类正则是否真的改善了 latent 部署行为

### 单一推荐下一步
- 下一轮如果继续优化 latent，不要再做新的全局静态正则；应改成更细的 selective regularizer，例如只对超过阈值的高偏差样本加权，或按 rollout 诊断聚焦中段问题窗口做定向约束。

## 48. 2026-03-24 追加（P5 v41：teacher-delta tail selective 首轮失败）
### 目标子项
- 在 `teacher_delta_tail` 已经比全局静态正则更接近正确方向的基础上，继续把作用范围缩到“只对真正激活的高偏差样本生效”，验证是否能减少普通样本被过度约束。

### 本次改动
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_selective`
  - 当开启时，仅对 `teacher_delta_tail > 0` 的样本计算 `teacher_delta_tail_loss`
  - 新增日志项：`teacher_delta_tail_active_ratio`

### 本次执行（可复现）
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_sel_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_selective=True task.env.numEnvs=8`
  - 结果：训练能正常启动并持续输出，无新的实现级报错
- 15min 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_t025_sel_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_selective=True`
  - 峰值：`Current Best=1941.87`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_t025_sel_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_t025_sel_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - 结果：`avg_reward=1.471573`, `avg_done_rate=0.001302`

### 本次结论
- 这轮可以明确判失败：
  - `teacher_delta_tail_selective seed42=1.471573`
  - 低于上一版 `teacher_delta_tail=1.602075`
  - 也明显低于主代表 `latent_recon05=1.731917`
- 这说明“只对激活样本计入正则”虽然继续抬高了训练峰值：
  - `Current Best=1941.87`
  - 但把训练目标进一步推离了最终部署评测
- 当前可收束的判断是：
  - 静态正则线已经被我们缩到比较细了，但效果仍然没有回到主代表之上
  - 下一步如果还沿正则走，不能再做新的静态 gating 变体，而要改成带阶段语义的约束；否则更适合暂停这条线

### 当前风险 / 剩余问题
- `Current Best` 再次明显高于历史版本，但部署分数进一步下降，说明“训练峰值 vs 最终评测”的错位在这类正则上非常严重
- 如果继续盲试静态正则，会很容易重复高训练分、低部署分的模式

### 单一推荐下一步
- 下一轮不要再继续改静态正则形式；应转去做阶段感知的最小约束，例如基于 rollout 诊断只针对中段窗口施加约束，或者先暂停正则线，回到 rollout 行为差异分析上再决定下一步。

## 49. 2026-03-24 追加（P5 v42：mid-stage teacher-delta tail 出现正信号）
### 目标子项
- 按 v41 的 handoff 推荐，停止继续尝试静态正则，转向更贴近 rollout 诊断的阶段感知版本，只在 episode 中段施加 `teacher_delta_tail`。

### 本次改动
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_mid_only`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_progress_start`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_progress_end`
  - 当开启时，仅在 `step_length / max_episode_length` 落入指定窗口时施加 `teacher_delta_tail_loss`

### 本次执行（可复现）
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_mid_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.25 +train.ppo.diffusion_teacher_delta_tail_progress_end=0.75 task.env.numEnvs=8`
  - 结果：训练能正常启动，无新的实现级报错
- 15min 正式训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_mid2575_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.25 +train.ppo.diffusion_teacher_delta_tail_progress_end=0.75`
  - 峰值：`Current Best=1814.62`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_mid2575_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_mid2575_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - 结果：`avg_reward=1.662729`, `avg_done_rate=0.001139`

### 本次结论
- 这是当前正则线里第一次比较明确的部署侧正信号：
  - `teacher_delta_tail_selective=1.471573`
  - `teacher_delta_tail=1.602075`
  - `teacher_delta_tail_mid_only=1.662729`
- 它还没有超过当前主代表 `latent_recon05=1.731917`，但已经证明：
  - “阶段感知”比“静态正则/静态 gating”更接近正确方向
  - 问题确实更像集中在某个行为阶段，而不是所有样本都需要同样的约束
- 所以这轮不升级主线，但可以把 `mid-stage teacher_delta_tail` 视为目前最值得继续收敛的一条正则支线

### 当前风险 / 剩余问题
- 训练峰值 `1814.62` 仍然没有转化为超过主代表的部署收益
- 当前只验证了 `seed42 light_v2`，还没做多 seed
- 如果窗口选得不对，仍可能出现“训练更高、部署一般”的错位

### 单一推荐下一步
- 下一轮优先继续沿 `mid-stage teacher_delta_tail` 做小范围收敛，不再回到静态正则：
  - 首选更窄的中段窗口，或
  - 保持窗口不变、下调 `tail_coef`
  - 并继续用同一口径 `15min + light_v2 seed42/512` 做验收

## 50. 2026-03-24 追加（PLANS_v2 对齐执行：M1 证据硬化微里程碑）
### 目标子项
- 按 `PLANS_v2` 的 M1（Evidence hardening baseline pack）先做最小闭环：
  - 补齐 teacher 在统一 fixed-step 口径下的可复核评测证据；
  - 修复 `stage_acceptance_summary` 的已知 artifact pointer 断链；
  - 保持改动局部、可逆，不改主训练路径。

### 本次改动
- `dexscrew/algo/ppo/ppo.py`
  - 新增 `test_num_steps` 支持（读取 `+test_num_steps`）。
  - `PPO.test()` 新增固定步评测统计并输出：
    - `EvalSummary steps=... avg_reward=... avg_done_rate=...`
  - 当未设置 `test_num_steps` 时，保留原 viewer 连续运行逻辑不变。
- `docs/stage_acceptance_summary.md`
  - 新增 `teacher_ppo` 行与 teacher 评测 artifact 指针。
  - 修复 `diffusion_latent` event 路径（`1774284474` -> `1774281454`）。
  - 补充 teacher nominal/light_v2 的固定步评测结果说明。

### 本次执行（可复现）
- 语法检查：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile('dexscrew/algo/ppo/ppo.py', 'exec') ... PY`
- teacher nominal（fixed-step）：
  - `./docker-run-isaacgym.sh timeout 1800 python train.py ... train.algo=PPO ... +test_num_steps=256 ... task.env.forceScale=0.0 task.env.randomForceProbScalar=0.0 > outputs/robustness_eval/teacher_nominal.log 2>&1`
  - 结果：`EvalSummary steps=256 avg_reward=2.918504 avg_done_rate=0.000407`
- teacher light_v2（fixed-step）：
  - `./docker-run-isaacgym.sh timeout 1800 python train.py ... train.algo=PPO ... +test_num_steps=256 ... task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2 > outputs/robustness_eval/teacher_light_v2.log 2>&1`
  - 结果：`EvalSummary steps=256 avg_reward=2.729195 avg_done_rate=0.000732`

### 本次结论
- 这轮完成了一个可落地的 M1 证据硬化子步：
  - teacher 已能用与 student 一致的 fixed-step `EvalSummary` 口径输出结果；
  - 现有文档中的一个关键 artifact pointer 已修复；
  - baseline pack 从“无 teacher 同口径评测”推进到“有可追溯 teacher nominal/light_v2 证据”。

### 当前风险 / 剩余问题
- `PLANS_v2` 的完整 evidence block 仍未全量落地（commit/config/dataset hash/eval episodes/dispersion 尚未统一模板化覆盖全部关键实验）。
- 当前 teacher 仍是单 seed（42），尚未进入 multiseed 聚合。
- `AGENTS.md` 中对计划文件的引用仍写 `PLANS.md`，与当前 `PLANS_v2` 主治理存在文本漂移（本轮未改治理文件）。

### 单一推荐下一步
- 继续 M1：先把 `teacher/current student/purebc` 三者做同口径 `nominal + light_v2 + hard` 的最小 multiseed（至少 3 seed）聚合表，并用统一 evidence block 模板落到 `docs/stage_acceptance_summary.md`。
