# 阶段性收尾（当前快照）

## 1. 四算法状态
- `ProprioAdapt`：已完成基准跑（15min）
- `PureBC`：已完成基准跑（15min）
- `DiffusionLatentStudent`：已完成基准跑（15min）
- `DiffusionActionChunkStudent`：已完成两轮 15min 调参并持续提升：
  - 原修复版：`1203.64`
  - tune-v1（mix=300k, first_action_bc=2.0, chunk_bc=0.2）：`1280.19`
  - tune-v2（mix=500k, first_action_bc=3.0, chunk_bc=0.2）：`1502.21`（已超过 `ProprioAdapt=1496.08`）
- `ProprioAdapt train-range ablation`：已完成 15min 验收，当前结果显著退化（`Max Current Best=32.31`）
- `ProprioAdapt train-range（adapt_tconv+mu）`：15min 已完成，仍显著退化（`Max Current Best=69.11`）
- `ProprioAdapt train-range（adapt_tconv+actor_mlp）`：15min 已完成，退化最明显（`Max Current Best=7.06`）

## 2. 当前统一结果（已完成跑次）
详见：
- `docs/stage_acceptance_summary.md`
- `docs/robustness_eval_summary.md`

当前可用对比（`Max Current Best`）：
- ProprioAdapt：`1496.08`
- PureBC：`1495.38`
- DiffusionLatent：`1860.15`
- DiffusionActionChunk（修复版 15min）：`1203.64`
- DiffusionActionChunk tune-v1（15min）：`1280.19`
- DiffusionActionChunk tune-v2（15min）：`1502.21`
- ProprioAdapt train-range（`adapt_tconv+actor_mlp+mu`）：`32.31`
- ProprioAdapt train-range（`adapt_tconv+mu`）：`69.11`
- ProprioAdapt train-range（`adapt_tconv+actor_mlp`）：`7.06`

P5 最小鲁棒性评测（256 steps）结论摘要：
- `ProprioAdapt`：`avg_reward≈1.92`（nominal）
- `DiffusionLatentStudent`：`avg_reward≈1.49`（nominal）
- `DiffusionActionChunk tune-v2`：`avg_reward≈-1.25`（nominal）
- 说明：action-chunk 训练分数已提升，但纯 student 固定步评测仍未达到可替代水平。

## 3. 验收口径补强（奖励之外）
已补内容：
- 统一指标汇总脚本：`scripts/summarize_student_acceptance.py`
- student 训练额外记录：
  - `done_rate/frame`
  - env 数值字段自动记录（若 env info 提供）

说明：
- 旧跑次（补强前）没有 `done_rate` 等新标量，因此在汇总中会显示 `N/A`。

## 4. 对 PLANS.md 的阶段对齐
当前处于：
- `P4 / M4`（diffusion student 已形成可比较且可优化结果）  
- `P2 / M2`（adapter training range ablation 已完成首轮定位）
- `P5`（鲁棒性最小评测包已启动并产出首轮结果）

下一步优先顺序（建议）：
1. `P2/M2` 结论可先冻结：当前主退化源来自放开 `actor_mlp`（`mu` 放开也有退化但相对轻）。
2. 训练主线保持 `adapt_tconv`-only，不建议把 train-range ablation 直接纳入默认路径。
3. `P5` 优先走“鲁棒性轴”：
   - 以 latent diffusion 作为当前稳态 diffusion 路径推进贡献验证
   - action-chunk 保留为探索分支，继续修复“训练高分/纯评测低分”偏差
4. 后续再进入 `P5` 另一轴（采样效率）：
   - 采样效率
   - 鲁棒性

## 5. 新增：P3 最小数据接口已落地
已新增 teacher rollout 采集入口（最小可用版）：
- `train.py` 的 `test + collect_rollout` 模式
- `scripts/collect_screwdriver_teacher_rollout.sh`（本地 py3.8+isaacgym 场景）
- `scripts/collect_screwdriver_teacher_rollout_docker.sh`（推荐，容器稳态场景）

当前接口可稳定导出以下字段（`torch.save(.pt)`）：
- `obs`
- `proprio_hist`
- `priv_info`
- `actions`
- `rewards`
- `dones`
- `done_rate_per_step`
- `point_cloud_info`（可开关）

参考命令：
- `scripts/collect_screwdriver_teacher_rollout.sh 0 42 run_a 256 run_a_collect`

说明：
- 采集脚本默认沿用 Hora canonical path 且附带 deterministic eval 覆盖项，便于对齐复现口径。

## 6. 新增：P3 消费侧最小对接已落地（action-chunk diffusion）
已新增 rollout -> diffusion student 的最小消费链路：
- `DiffusionActionChunkStudent` 支持可选 rollout 预训练（默认关闭，不影响现有训练）
  - `+train.ppo.rollout_pretrain_path=<rollout.pt>`
  - `+train.ppo.rollout_pretrain_updates=<N>`
  - `+train.ppo.rollout_pretrain_batch_size=<B>`
- 辅助脚本：
  - `scripts/screwdriver_student_diffusion_action_chunk_rollout_pretrain.sh`

当前语义：
- 先用 rollout 数据做离线预训练若干步（学习 action chunk 分布）
- 再进入原有在线 teacher-student 训练循环
- 不改变原主路径接口与默认命令

## 7. 新增：P2 adapter training range ablation（最小实现）
已落地最小可运行实现（默认语义不变，仅在 ablation 脚本中开启）：
- `dexscrew/algo/ppo/padapt.py`
  - 新增 `train.ppo.student_trainable_param_patterns` 配置入口
  - 默认值仍为 `[adapt_tconv]`（保持当前 baseline 语义）
  - 可通过子串匹配扩展训练范围（如 `actor_mlp`、`mu`）
- `configs/train/XHandHoraScrewDriver.yaml`
  - 新增默认项：`student_trainable_param_patterns: [adapt_tconv]`
- 新增脚本：
  - `scripts/screwdriver_student_padapt_trainrange.sh`
  - `scripts/screwdriver_student_padapt_trainrange_15min_docker.sh`

已完成最小 smoke：
- 新脚本语法检查通过：`bash -n ...`
- 容器内启动训练通过（`timeout 120` 截断），并出现关键日志：
  - `ProprioAdapt trainable patterns: ['adapt_tconv', 'actor_mlp', 'mu'] | trainable params: ...`
