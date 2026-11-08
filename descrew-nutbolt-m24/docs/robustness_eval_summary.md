# Robustness Eval Summary（P5 最小评测包）

## 1. 评测口径
- 脚本：`scripts/eval_screwdriver_student_robustness.sh`
- task：`XHandHoraScrewDriver`
- rollout steps：`256`
- seed：`42`
- 指标：
  - `avg_reward`（步均奖励）
  - `avg_done_rate`（步均 done 比例）

扰动组覆盖项：
- `task.env.randomization.obs_noise_e_scale=0.03`
- `task.env.randomization.obs_noise_t_scale=0.015`
- `task.env.forceScale=1.0`
- `task.env.randomForceProbScalar=0.2`

## 2. 结果汇总
| Policy / Checkpoint | Nominal avg_reward | Nominal done_rate | Perturb avg_reward | Perturb done_rate |
|---|---:|---:|---:|---:|
| ProprioAdapt (`run_a`) | `1.918988` | `0.001383` | `1.967401` | `0.001465` |
| DiffusionLatentStudent (`run_a_seed42_15min`) | `1.489716` | `0.001872` | `1.382724` | `0.002360` |
| DiffusionActionChunkStudent 修复版 (`run_a_action_chunk_fix_seed42_15min`) | `-1.873523` | `0.011149` | `-1.628801` | `0.010742` |
| DiffusionActionChunkStudent tune-v2 (`mix500k_fa3_cb0.2`) | `-1.248181` | `0.013184` | `-0.960685` | `0.012614` |

## 3. 当前结论
1. action-chunk tune-v2 在训练曲线里显著提升（`Max Current Best=1502.21`），但纯 student 固定步评测仍为负回报。
2. action-chunk tune-v2 相比修复版有实质改善（`-1.87 -> -1.25` nominal），但还未达到可替代 baseline 的水平。
3. latent diffusion 在该最小评测包下表现稳定为正回报，且 done_rate 明显低于 action-chunk。

## 4. 建议
- 若目标是当前阶段“稳态 diffusion 贡献”，优先用 latent diffusion 进入鲁棒性/效率贡献轴。
- action-chunk 继续保留为主线探索，但要把“训练混合执行指标”和“纯 student 评测指标”拆开记录，避免误判。

## 5. 本轮追加定位（推理侧，512 steps，nominal）
目的：确认 action-chunk 的纯评测偏差是否主要来自推理配置。

命令（同一 ckpt：`run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min`）：

- 默认确定性推理：
  - `+train.ppo.action_chunk_stochastic_infer=False`
  - `+train.ppo.action_chunk_diffusion_steps_infer=10`
- 随机采样推理：
  - `+train.ppo.action_chunk_stochastic_infer=True`
  - `+train.ppo.action_chunk_diffusion_steps_infer=10`
- 提高去噪步数：
  - `+train.ppo.action_chunk_stochastic_infer=False`
  - `+train.ppo.action_chunk_diffusion_steps_infer=20`

结果：

| Inference Setup | avg_reward | avg_done_rate |
|---|---:|---:|
| deterministic, steps=10 | `-1.626525` | `0.013184` |
| stochastic, steps=10 | `-1.588469` | `0.013509` |
| deterministic, steps=20 | `-1.626525` | `0.013184` |

结论：

1. 仅改推理侧配置无法实质修复负回报问题（提升幅度很小，且 done_rate 无明显改善）。
2. `diffusion_steps_infer: 10 -> 20` 在该实现/配置下未带来可见收益，提示当前瓶颈更可能在训练目标或 teacher mixing 对齐，而非简单采样步数不足。

## 6. P5 v2 扩展评测（主线稳定性，256 steps）
目的：按 handoff 建议，把 `DiffusionLatentStudent` 作为稳态 diffusion 路径，和 `ProprioAdapt` 在同一扰动强度下做直接对照。

评测设置：

- nominal（脚本默认）
- light:
  - `task.env.randomization.obs_noise_e_scale=0.01`
  - `task.env.randomization.obs_noise_t_scale=0.005`
  - `task.env.forceScale=0.5`
  - `task.env.randomForceProbScalar=0.1`
- hard:
  - `task.env.randomization.obs_noise_e_scale=0.05`
  - `task.env.randomization.obs_noise_t_scale=0.025`
  - `task.env.forceScale=1.5`
  - `task.env.randomForceProbScalar=0.3`

结果：

| Policy | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| ProprioAdapt (`run_a`) | `1.918988 / 0.001383` | `2.063784 / 0.001058` | `1.826708 / 0.001546` |
| DiffusionLatentStudent (`run_a_seed42_15min`) | `1.489716 / 0.001872` | `1.327045 / 0.002116` | `0.950702 / 0.002523` |

结论：

1. `DiffusionLatentStudent` 在 nominal/light/hard 三档下均保持正回报，说明作为当前 diffusion 主线是稳定可用的。
2. 与 `ProprioAdapt` 相比，latent 路径在强扰动下存在明显性能差距（奖励更低、done_rate 更高），当前阶段更适合作为“可运行可比较”的 diffusion 主线，而非直接替代强 baseline。
3. 基于这轮结果，下一步应进入“latent 路线的小步优化与鲁棒性贡献验证”，继续保持 action-chunk 为并行修复线。

## 7. P5 v3 优化验证（latent 训练侧轻扰动注入）
目的：验证“训练分布缺少外力扰动暴露”是否是 latent 路线在强扰动下掉点的关键原因。

优化动作（仅训练侧，不改算法结构）：
- 在 latent 训练阶段注入轻外力扰动：
  - `task.env.forceScale=0.5`
  - `task.env.randomForceProbScalar=0.1`
- 其余核心设置保持不变，得到 ckpt：
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

训练侧信号（TensorBoard）：
- baseline latent：`episode_rewards/step max=1860.1531`
- robust-light latent：`episode_rewards/step max=1836.4569`
- 说明：训练峰值略降，但不代表鲁棒性必然变差，需要看固定步评测。

固定步评测对照（steps=256）：

| Latent Variant | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| baseline latent | `1.489716 / 0.001872` | `1.327045 / 0.002116` | `0.950702 / 0.002523` |
| robust-light latent | `1.432408 / 0.001709` | `1.630288 / 0.001383` | `1.344396 / 0.002035` |

变化（robust-light - baseline）：
- nominal：reward `-0.057308`，done_rate `-0.000163`
- light：reward `+0.303243`，done_rate `-0.000733`
- hard：reward `+0.393694`，done_rate `-0.000488`

结论：
1. 假设被支持：latent 的主要短板之一确实是“训练扰动覆盖不足”，而不是仅靠推理侧可解决的问题。
2. 轻扰动注入明显改善 light/hard 档鲁棒性，同时 nominal 奖励仅小幅回落（且 done_rate 仍改善）。
3. 这是一条对毕设主线友好的优化方向：无需新增第 5 套算法，即可让 diffusion 路线更接近强 baseline。

## 8. P5 v4 小网格扫描（围绕 robust-light）
目的：在不改算法结构的前提下，继续搜索训练侧扰动强度甜点，验证是否存在优于 `forceScale=0.5, randomForceProbScalar=0.1` 的稳定点。

训练（统一 15min, seed=42）：

| Variant | 训练参数（forceScale / randomForceProbScalar） | Max Current Best |
|---|---:|---:|
| robust-light（已有） | `0.5 / 0.1` | `1836.46` |
| robust-mid（已有） | `0.8 / 0.15` | `1797.54` |
| fs04_p008（本轮新增） | `0.4 / 0.08` | `1831.22` |
| fs06_p012（本轮新增） | `0.6 / 0.12` | `1862.06` |

固定步评测（steps=256）：

| Variant | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| robust-light (`0.5/0.1`) | `1.432408 / 0.001709` | `1.630288 / 0.001383` | `1.344396 / 0.002035` |
| robust-mid (`0.8/0.15`) | `1.292951 / 0.001953` | `1.427850 / 0.002279` | `1.230681 / 0.002523` |
| fs04_p008 (`0.4/0.08`) | `1.051928 / 0.001628` | `1.134281 / 0.001628` | `0.950787 / 0.002686` |
| fs06_p012 (`0.6/0.12`) | `1.500453 / 0.001953` | `1.477706 / 0.001628` | `1.229945 / 0.002035` |

补充检查（推理步数）：
- 在 robust-light ckpt 上将 `diffusion_steps_infer` 从 `10` 提到 `20`：
  - nominal：`1.432408 / 0.001709`
  - hard：`1.344396 / 0.002035`
- 与 `infer=10` 一致，无实质收益。

结论：
1. 训练峰值 `Current Best` 与固定步评测并不一致，不能单独作为优化依据（`fs06_p012` 训练峰值最高，但鲁棒性不如 robust-light）。
2. 就当前三档综合表现看，`robust-light (0.5/0.1)` 仍是最稳妥主线点位。
3. 下一步更应关注“训练流程与模型选择策略”优化，而不仅是继续抬高静态扰动强度。

## 9. PureBC 同口径补测（P5 v6）
目的：补齐 `PureBC` 在最新 nominal/light/hard 口径下的对照结果，便于和 `ProprioAdapt`、`DiffusionLatentStudent` 直接比较。

设置（steps=256, seed=42）：
- nominal：脚本默认 deterministic eval（无额外噪声/外力）
- light：
  - `task.env.randomization.obs_noise_e_scale=0.02`
  - `task.env.randomization.obs_noise_t_scale=0.01`
  - `task.env.forceScale=0.5`
  - `task.env.randomForceProbScalar=0.1`
- hard：
  - `task.env.randomization.obs_noise_e_scale=0.05`
  - `task.env.randomization.obs_noise_t_scale=0.025`
  - `task.env.forceScale=1.5`
  - `task.env.randomForceProbScalar=0.3`

结果（ckpt: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`）：

| Policy | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| PureBC | `1.267421 / 0.001546` | `2.144827 / 0.001465` | `1.807907 / 0.001628` |

## 10. 四算法统一三挡复核（P5 v7）
目的：消除不同轮次 light/hard 定义差异，给出四种 student 在同一口径下的直接对比。

统一设置（steps=256, seed=42）：
- nominal：脚本默认 deterministic eval
- light：`obs_noise_e=0.02`, `obs_noise_t=0.01`, `forceScale=0.5`, `randomForceProbScalar=0.1`
- hard：`obs_noise_e=0.05`, `obs_noise_t=0.025`, `forceScale=1.5`, `randomForceProbScalar=0.3`

结果：

| Algo | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| ProprioAdapt | `1.918988 / 0.001383` | `1.958849 / 0.001546` | `1.826708 / 0.001546` |
| PureBC | `1.267421 / 0.001546` | `2.144827 / 0.001465` | `1.807907 / 0.001628` |
| DiffusionLatentStudent（robust-light） | `1.432408 / 0.001709` | `1.556369 / 0.001221` | `1.344396 / 0.002035` |
| DiffusionActionChunkStudent（tune-v2） | `-1.248181 / 0.013184` | `-0.937148 / 0.012533` | `-0.768358 / 0.013916` |

说明：
- 当前“成功率”尚未单独统计为任务成功事件比例，现阶段统一使用 `avg_reward + avg_done_rate` 做对照。

## 11. ProprioAdapt 1h 验证（P5 v8）
目的：验证“15min 是否低估 adapt”，并检查延长训练是否在三挡上都收益一致。

训练：
- 命令：`scripts/screwdriver_student_padapt_15min_docker.sh 0 42 run_a 3600 run_a_seed42_1h`
- 结果：`Max Current Best=1608.43`

同口径评测（steps=256）：
- checkpoint：`outputs/XHandHoraScrewDriver_student_padapt/run_a_seed42_1h/stage2_nn/model_best.ckpt`
- nominal：`avg_reward=2.077178`, `avg_done_rate=0.000977`
- light：`avg_reward=2.065615`, `avg_done_rate=0.000977`
- hard：`avg_reward=1.733165`, `avg_done_rate=0.001465`

与 15min（`run_a`）对比：
- nominal：`1.918988 -> 2.077178`（提升）
- light：`1.958849 -> 2.065615`（提升）
- hard：`1.826708 -> 1.733165`（下降）

结论：
1. `15min` 确实低估了 adapt 在 nominal/light 的表现。
2. 但延长训练并非全域单调提升：hard 档出现回落，提示存在“训练后期对强扰动泛化下降”的风险。

## 12. DiffusionLatent robust-light 1h 验证（P5 v9）
目的：验证 `15min` 对 diffusion-latent 是否同样存在明显低估，并判断差距是否主要来自拟合时长。

训练：
- 命令：`timeout 3600 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_robust_light_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth"`
- 结果：`Max Current Best=1848.04`
- 产物：`outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_1h/stage2_diffusion_nn/model_best.ckpt`

同口径评测（steps=256, seed=42）：
- nominal：`avg_reward=1.610548`, `avg_done_rate=0.001546`
- light：`avg_reward=1.432580`, `avg_done_rate=0.001872`
- hard：`avg_reward=1.248908`, `avg_done_rate=0.002360`

与 15min robust-light（`run_a_latent_robust_light_seed42_15min`）对比：
- nominal：`1.432408 -> 1.610548`（`+0.178140`）
- light：`1.556369 -> 1.432580`（`-0.123789`）
- hard：`1.344396 -> 1.248908`（`-0.095488`）

结论：
1. diffusion-latent 的确存在“时长敏感性”：nominal 明显提升，说明 15min 对名义表现有低估。
2. 但延长至 1h 后 light/hard 回落，说明当前训练目标仍偏向名义拟合，鲁棒性没有随时长同步提升。
3. 因此“15min 足以证明是否能蒸馏”这个判断对**基础可行性**成立；但若要证明**鲁棒性价值**，仍需继续优化训练策略而非仅延时长。

## 13. DiffusionActionChunk tune-v2 1h 验证（P5 v10）
目的：排查 action-chunk 低表现是否仅因 `15min` 欠拟合，验证“延长训练时长”是否能显著提升。

训练：
- 命令：
  - `timeout 3600 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=3.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- 结果：
  - `Max Current Best=1502.21`
  - 与 15min tune-v2（`run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min`）相同。

关键核对：
- `model_best.ckpt` 哈希一致（15min vs 1h）：
  - `3e360f545ced53aaf39c6cc4998fb7e600bf80a2c55257a07b4fb0128e688bc3`
- nominal 固定步评测（steps=256）：
  - 1h：`avg_reward=-1.248181`, `avg_done_rate=0.013184`
  - 与 15min 结果一致。

结论：
1. 对当前 action-chunk tune-v2 实现，延长训练到 1h 并未带来可观收益。
2. 当前差距不再主要由“训练时长不足”解释，下一步应优先做算法/训练目标层面的排查与改进。

## 14. ActionChunk 对齐排查与修复（P5 v11）
目的：定位“训练表现与纯 student 评测不一致”的对齐问题，并做最小可验证修复。

### 发现的关键错位
- 原实现 `model_best.ckpt` 按训练期 `mean_eps_reward` 选模。
- 但 action-chunk 使用 `teacher_mix_steps` 时，训练执行动作为 teacher/student 混合；评测时是纯 student。
- 导致：训练 best 可能在 teacher 占比仍高阶段被锁定，和纯 student 评测口径不一致。

### 代码修复（最小改动）
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 新增：
  - `teacher_mix_ratio` 日志；
  - 周期性 `model_last.ckpt`（默认每 `action_chunk_ckpt_interval_steps=50000`）；
  - student 对齐选模：
    - `model_best_student_reward.ckpt`：warmup 后按 reward 选；
    - `model_best_student.ckpt`：warmup 后按 `student_action_mse` 选（首动作模仿对齐）。
- 文件：`scripts/vis_screwdriver_student_diffusion_action_chunk.sh`
  - 默认 checkpoint 优先级改为：
    - `model_best_student_reward.ckpt`
    - `model_best_student.ckpt`
    - `model_last.ckpt`
    - `model_best.ckpt`

### 诊断实验 A（300s, mix=20k）
- run: `run_a_action_chunk_aligndiag_mix20k_seed42_300s`
- nominal（steps=256）：
  - `model_best`: `-1.951475 / 0.010173`
  - `model_best_student_reward`: `-2.500227 / 0.010579`
  - `model_best_student`(MSE): `-0.626234 / 0.010905`
  - `model_last`: `-0.740303 / 0.010905`

### 诊断实验 B（主线参数，15min, mix=500k）
- run: `run_a_action_chunk_tune_mix500k_fa3_cb02_alignmse_seed42_15min`
- `model_best` 与历史 tune-v2 `model_best` 哈希相同：
  - `3e360f545ced53aaf39c6cc4998fb7e600bf80a2c55257a07b4fb0128e688bc3`
- nominal（steps=256）：
  - `model_best`: `-1.248181 / 0.013184`
  - `model_best_student_reward`: `-0.676029 / 0.011963`
  - `model_best_student`(MSE): `-0.812928 / 0.009847`
  - `model_last`: `-0.466151 / 0.011475`

结论：
1. 已确认主问题是“checkpoint 选模口径错位”，不是单纯训练时长不足。
2. 在不改核心算法结构下，仅修复选模与保存策略即可显著改善 action-chunk 评测表现（仍未转正，但明显优于旧 `model_best`）。
3. 下一步应在该修复基础上继续做 first-action 对齐强化与 teacher-mix 日程优化，而不是继续无条件拉长训练时长。

## 15. ActionChunk 对齐后参数扫描与新候选（P5 v12）
目的：在修复后的选模机制上快速搜索有效参数方向，优先提升 pure-student 评测分数。

扫描设置（均为 300s，seed=42，teacher=`run_a`）：
- 固定：`teacher_mix_steps=500000`, `action_chunk_model_selection_warmup_steps=0`
- 评测口径：nominal, `steps=256`
- 点位：
  - A: `first_action_bc_loss_coef=5.0`, `chunk_bc_loss_coef=0.2`
  - B: `first_action_bc_loss_coef=8.0`, `chunk_bc_loss_coef=0.2`
  - C: `first_action_bc_loss_coef=5.0`, `chunk_bc_loss_coef=0.1`

结果（各点取该点内最优 checkpoint 的 nominal）：
- A (`fa5/cb0.2`)：
  - `model_best` / `model_best_student_reward`：`avg_reward=-0.497100`, `avg_done_rate=0.013835`
  - `model_last`：`-0.623638 / 0.010417`
- B (`fa8/cb0.2`)：
  - `model_last` 最优：`avg_reward=-0.682759`, `avg_done_rate=0.011475`
  - `model_best`：`-3.641524 / 0.012288`
- C (`fa5/cb0.1`)：
  - `model_best_student`(MSE) 最优：`avg_reward=-0.589909`, `avg_done_rate=0.009847`
  - `model_best`：`-2.000882 / 0.011637`

选定候选：`fa5/cb0.2`

候选 15min 验证（`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_15min`）：
- 训练：
  - `Max Current Best=1536.97`
  - `Best Student MSE=0.2761`（末值）
- nominal（steps=256）：
  - `model_best`: `-0.497100 / 0.013835`
  - `model_best_student_reward`: `-0.497100 / 0.013835`
  - `model_best_student`: `-0.741821 / 0.010417`
  - `model_last`: `-0.047760 / 0.008626`  ← 当前最优
- light/hard（均用 `model_last`）：
  - light：`-0.061166 / 0.009847`
  - hard：`-0.255275 / 0.008789`

与旧 action-chunk tune-v2（`-1.248181 / -0.937148 / -0.768358`）相比：
- nominal：显著改善到接近转正（`-0.047760`）
- light/hard：同样大幅改善。

结论：
1. 修复选模口径后，参数搜索已经找到可显著提升的稳定方向（`fa5/cb0.2`）。
2. 当前 action-chunk 仍略低于 0，但已从“明显负回报”进入“接近可用”区间。
3. 下一步应对该候选做 1h 验证，检查是否可稳定转正并保持 light/hard 改善。

## 16. ActionChunk 候选 1h 验证（P5 v13）
目的：验证 `fa5/cb0.2 + mix500k + warmup0` 在 1h 下是否稳定转正，并复核三挡表现。

训练：
- run：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_1h`
- 命令：
  - `timeout 3600 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed42_1h "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
- 结果：
  - `Max Current Best=1536.97`
  - `Best Student MSE(末值)=0.2761`

同口径评测（steps=256, seed=42）：
- `model_best_student_reward`：
  - nominal：`-0.497100 / 0.013835`
  - light：`-0.415648 / 0.014079`
  - hard：`-0.215941 / 0.013672`
- `model_last`（当前最佳）：
  - nominal：`0.121916 / 0.007975`
  - light：`0.435321 / 0.007568`
  - hard：`0.233560 / 0.007813`

对比提升（按 `model_last`）：
- 相比旧 action-chunk tune-v2（15min）：
  - nominal：`-1.248181 -> 0.121916`（`+1.370097`）
  - light：`-0.937148 -> 0.435321`（`+1.372469`）
  - hard：`-0.768358 -> 0.233560`（`+1.001918`）
- 相比本候选 15min（`model_last`）：
  - nominal：`-0.047760 -> 0.121916`（`+0.169676`）
  - light：`-0.061166 -> 0.435321`（`+0.496487`）
  - hard：`-0.255275 -> 0.233560`（`+0.488835`）

结论：
1. action-chunk 在当前修复与参数下已实现三挡正回报，主线可继续保留。
2. `model_last` 明显优于 `model_best_student_reward`，说明该设置下后期 student-only 收敛更关键。
3. 下一步应做多 seed 复核，确认统计稳健后再进入和 Adapt/PureBC 的正式论文对比口径。

## 17. ActionChunk 候选多 seed 复核（P5 v14）
目的：完成 `fa5/cb0.2 + mix500k + warmup0` 的最小统计稳健性检查，验证是否能跨随机种子稳定保持正回报。

训练补齐（seed=44, 15min）：
- run：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed44_15min`
- 命令：
  - `timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 44 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed44_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
- 评测 checkpoint：`model_last.ckpt`

同口径评测（steps=256，light=`0.02/0.01/0.5/0.1`，hard=`0.05/0.025/1.5/0.3`）：

| Seed | Nominal avg_reward / done_rate | Light avg_reward / done_rate | Hard avg_reward / done_rate |
|---|---:|---:|---:|
| 42 | `0.121916 / 0.007975` | `0.435321 / 0.007568` | `0.233560 / 0.007813` |
| 43 | `-0.616286 / 0.007650` | `-0.206074 / 0.007487` | `-0.825969 / 0.009277` |
| 44 | `-0.111548 / 0.007975` | `-0.041217 / 0.008870` | `-0.240999 / 0.008789` |

三 seed 汇总（reward）：
- nominal：mean `-0.201973`，std `0.308078`
- light：mean `0.062677`，std `0.271959`
- hard：mean `-0.277803`，std `0.433333`

结论：
1. 当前 action-chunk 候选存在明显 seed 方差，`seed42` 可转正但 `seed43/44` 未稳定复现。
2. 现阶段不宜将该配置直接作为“已稳态可替代 Adapt”的主结论，只能作为“有潜力但稳定性不足”的探索结果。
3. 下一步应优先做“稳定性导向”优化（先稳住多 seed，再继续追单 seed 峰值）。

## 18. ActionChunk 评测方差定位（P5 v15）
目的：确认“不同 seed 差异大”主要来自训练 seed 还是评测 seed，并验证 `mix_steps` 调整是否能在 15min 内显著改善。

### 18.1 交叉评测矩阵（`mix500k`, `fa5/cb0.2`, `model_last`, nominal, steps=256）
行：训练 ckpt seed；列：eval seed

| train \ eval | 42 | 43 | 44 | 行均值 |
|---|---:|---:|---:|---:|
| 42 | `0.121916` | `-0.343631` | `0.023255` | `-0.066153` |
| 43 | `0.068391` | `-0.616286` | `0.110339` | `-0.145852` |
| 44 | `-0.092808` | `0.055781` | `-0.111548` | `-0.049525` |

列均值（按 eval seed）：
- eval seed 42：`0.032500`
- eval seed 43：`-0.301379`
- eval seed 44：`0.007349`

观察：
1. 同一个 ckpt 在不同 eval seed 下可“正负翻转”（例如 train seed43：`0.068391 / -0.616286 / 0.110339`）。
2. `eval seed=43` 对三组 ckpt 都更差，说明评测方差显著，且评测 seed 本身会主导结论。

### 18.2 15min 对照：仅改 `teacher_mix_steps`（500k -> 300k）
设置：
- run: `run_a_action_chunk_tune_fa5_cb02_mix300k_w0_seed43_15min`
- 固定：`first_action_bc_loss_coef=5.0`, `chunk_bc_loss_coef=0.2`, `warmup=0`
- 训练结果：`Current Best=1725.38`, `Best Student MSE=0.3250`

同口径评测（`model_last`, nominal, steps=256, eval seed 42/43/44）：
- `-0.163048 / 0.113329 / -0.367999`
- 均值：`-0.139239`（旧 `mix500k` 同 train seed43 为 `-0.145852`）

结论：
1. 将 `mix_steps` 从 `500k` 改到 `300k` 在该 seed 下未带来实质改善（均值几乎不变）。
2. 当前“不同 seed 差异大”不应直接归因于单一训练 seed 失败，更像是：
   - 评测方差较高（尤其特定 eval seed 更苛刻）；
   - policy 仍处于“临界区”（不同初始化条件下正负波动）。
3. 下一步应改为“多 eval seed 均值口径”驱动优化，而不是单点 seed 判定算法优劣。

## 19. ActionChunk deterministic 推理修复（P5 v16）
目的：验证 action-chunk 的 seed 敏感是否来自实现问题，并做最小可逆修复。

代码修复：
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 函数：`sample_action_chunk(...)`
- 改动：当 `action_chunk_stochastic_infer=False` 时，不再用随机 `torch.randn` 初始化 `x_T`，改为固定 `0` 初始化；仅在 stochastic 模式下使用随机初始化。

修复前（同一 ckpt，同口径 multiseed512）：
- ckpt：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed43_15min/model_last.ckpt`
- 结果：
  - seed42：`-0.127865 / 0.009318`
  - seed43：`-0.399093 / 0.008057`
  - seed44：`0.104021 / 0.007731`
  - aggregate：`reward_mean=-0.140979`, `reward_std=0.205605`

修复后（同一 ckpt，同口径 multiseed512）：
- 命令：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionActionChunkStudent outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_seed43_15min/stage2_diffusion_action_chunk_nn/model_last.ckpt 512 p5v16_actionchunk_determinfix_multiseed512 "42,43,44"`
- 结果：
  - seed42：`0.289716 / 0.009318`
  - seed43：`0.079925 / 0.007853`
  - seed44：`0.486741 / 0.007894`
  - aggregate：`reward_mean=0.285461`, `reward_std=0.166109`

横向对照（multiseed512，seed=42/43/44）：
- ProprioAdapt：`reward_mean=2.400372`, `reward_std=0.151554`
- DiffusionLatent(robust-light)：`reward_mean=1.742653`, `reward_std=0.104761`
- PureBC：`reward_mean=2.116345`, `reward_std=0.263339`
- ActionChunk（修复后）：`reward_mean=0.285461`, `reward_std=0.166109`

结论：
1. 这次定位确认存在实现级不一致：原 deterministic 路径并不真正 deterministic。
2. 最小修复后，action-chunk 同一 ckpt 的三 seed 评测由负均值转为正均值，且方差下降。
3. 目前 action-chunk 仍落后于强 baseline，但“评测不稳”这一关键阻塞已显著缓解，后续可继续做训练侧优化而非重复排查环境/口径。

## 20. ActionChunk 修复版重新训练复核（P5 v17）
目的：确认 deterministic 推理修复不仅能改善旧 checkpoint 评测，还能否在新一轮 15min 训练中稳定转化为更好的 pure-student 表现。

训练设置：
- run：`run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min`
- 命令：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=500000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2 ++train.ppo.action_chunk_model_selection_warmup_steps=0`
- 训练结果：
  - `Current Best=1692.80`
  - `Best Student MSE=0.1589`

部署评测（multiseed512，nominal）：
- `model_last.ckpt`
  - seed42：`-0.510649 / 0.011230`
  - seed43：`-0.084439 / 0.010905`
  - seed44：`-0.332757 / 0.011190`
  - aggregate：`reward_mean=-0.309282`, `reward_std=0.174790`
- `model_best_student.ckpt`
  - aggregate：`reward_mean=-1.339584`, `reward_std=0.165684`
- `model_best_student_reward.ckpt`
  - aggregate：`reward_mean=-1.409180`, `reward_std=0.224958`

与上一轮“旧 ckpt 修复后回归”对比：
- 旧 ckpt（seed43 run, model_last）修复后：`reward_mean=0.285461`, `reward_std=0.166109`
- 本轮新训练（seed42 run, model_last）：`reward_mean=-0.309282`, `reward_std=0.174790`

结论：
1. deterministic 推理修复是真实有效的，但它没有自动解决“训练峰值更高 -> pure-student 部署更好”这个对齐问题。
2. 本轮出现了更强的训练分数（`1692.80`），却对应更差的部署评测，说明 action-chunk 当前的主要瓶颈已转向训练目标/选模口径失配。
3. 在现阶段，`model_last` 仍是三类 checkpoint 中最接近可用的导出物，但单靠继续看 `Current Best` 调参已经不够可靠。

## 21. ActionChunk 训练内 pure-student 选模对齐（P5 v18）
目的：把 `model_best_student_reward` 从 mixed-policy reward 中解耦出来，只统计 teacher-mix 结束后的 pure-student episode，再验证这种“更干净的 reward 选模”是否真的更接近部署结果。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 新增：
  - `student_eval_phase_started`
  - `student_mean_eps_reward / student_mean_eps_length`
  - `student_step_reward / student_step_length`
  - `student_tracking_active`
- 逻辑：
  - 仅当 `teacher_mix_ratio <= 0` 时启动 pure-student meter；
  - 仅统计 pure-student phase 后重新开始的 episode；
  - `model_best_student_reward` 改为基于该 meter 选模，而不再直接复用训练主 reward meter。

轻量验证：
- 3min smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 smoke_actionchunk_studenteval_align_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=5000 ++train.ppo.action_chunk_model_selection_warmup_steps=5000 ++train.ppo.action_chunk_ckpt_interval_steps=20000 ++task.env.numEnvs=24`
- 结果：
  - 训练可正常进入循环；
  - `Best Student Reward` 会在 pure-student phase 后从 `N/A` 变为数值，说明新链路已接通。

15min 正式训练：
- run：`run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min`
- 命令：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- 训练结果：
  - `Current Best=1166.49`
  - `Best Student Reward=50.85`
  - `Best Student MSE=0.3953`

单 seed nominal 评测（seed=42, steps=512）：
- `model_last.ckpt`
  - `avg_reward=-0.388507`, `avg_done_rate=0.012248`
- `model_best_student_reward.ckpt`
  - `avg_reward=-0.899636`, `avg_done_rate=0.014689`
- `model_best_student.ckpt`
  - `avg_reward=-0.253256`, `avg_done_rate=0.013550`

结论：
1. 这次修复证明了：`model_best_student_reward` 之前确实有 phase 污染风险；现在统计口径更干净了。
2. 但在当前 15min 窗口内，**更干净的 reward 选模并没有变成更好的 selector**：
   - `model_best_student_reward` 最差；
   - `model_best_student` 最好；
   - `model_last` 居中。
3. 因此当前主瓶颈已经不是“reward meter 接错”，而是 pure-student episode reward 在短窗内依然太稀疏、太噪声，尚不足以稳定代表最终部署质量。
4. 下一步应优先考虑更直接的 deterministic deployment proxy，或暂时以 `model_best_student` 作为 action-chunk 的默认导出候选。

## 22. ActionChunk 缩短时域尝试 + Latent deterministic 推理迁移（P5 v19）
目的：
- 验证 `action_chunk_len=8 -> 4` 是否能降低时序建模难度，缓解 action-chunk 的“训练分数高、部署分数低”问题。
- 将 action-chunk 已验证有效的 deterministic 推理对齐经验，迁移到 latent diffusion，检查是否也是共享收益点。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 改动：
  - 新增 `diffusion_stochastic_infer` 开关（默认 `False`）；
  - 当 stochastic 关闭时，latent diffusion 推理改为固定 `x_T=0` 初始化，且中间步不再加随机噪声。

ActionChunk `len=4` 训练：
- run：`run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min`
- 命令：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" ++train.ppo.action_chunk_len=4 train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- 评测注意：
  - `len=4` checkpoint 在评测时必须显式追加 `++train.ppo.action_chunk_len=4`，否则会按默认 `len=8` 构图并触发 shape mismatch。

ActionChunk `len=4` 单 seed nominal 评测（seed=42, steps=512）：
- `model_best_student.ckpt`
  - `avg_reward=-0.893365`, `avg_done_rate=0.014689`
- `model_last.ckpt`
  - `avg_reward=-0.471724`, `avg_done_rate=0.014242`
- `model_best_student_reward.ckpt`
  - `avg_reward=-0.328458`, `avg_done_rate=0.014974`

与 `len=8` 对照（`run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min`）：
- `len=8 model_best_student.ckpt`
  - `avg_reward=-0.253256`, `avg_done_rate=0.013550`
- `len=8 model_last.ckpt`
  - `avg_reward=-0.388507`, `avg_done_rate=0.012248`
- `len=8 model_best_student_reward.ckpt`
  - `avg_reward=-0.899636`, `avg_done_rate=0.014689`

Latent diffusion deterministic/stochastic 对照（seed=42, steps=512）：
- deterministic（默认，`diffusion_stochastic_infer=False`）：
  - `avg_reward=1.682415`, `avg_done_rate=0.001058`
- stochastic（显式 `++train.ppo.diffusion_stochastic_infer=True`）：
  - `avg_reward=1.626398`, `avg_done_rate=0.001302`

结论：
1. 单纯把 `chunk_len` 从 `8` 缩到 `4`，**没有把 action-chunk 的绝对部署性能拉起来**；当前最好 ckpt 仍是负回报，且不如 `len=8` 的 `model_best_student (-0.253256)`。
2. 但 `len=4` 的 selector 排序发生了变化：
   - `model_best_student_reward (-0.328458)` 反而优于 `model_last (-0.471724)` 与 `model_best_student (-0.893365)`；
   - 说明“缩短预测时域”可能在一定程度上减轻了 reward-selector 与部署结果的失配，但还不足以构成完整解法。
3. latent diffusion 的 deterministic 推理迁移是有效且低风险的：
   - 在当前 ckpt / seed42 / 512steps 下，deterministic 略优于 stochastic；
   - 说明 action-chunk 里发现的“推理随机源对评测口径有污染”并非孤例。
4. 下一步不应继续沿“盲目缩短 chunk”反复试，而应转向更直接的 training-time deploy proxy，使 action-chunk 的 checkpoint 选择尽量贴近最终 deterministic 评测。

## 23. ActionChunk 训练内 deterministic deploy probe（P5 v20）
目的：
- 不再继续依赖在线单步 `student_action_mse` 或稀疏 episode reward 选模；
- 在不引入第二套环境、不改 teacher-student 主结构的前提下，增加一个更稳定的 training-time deterministic deploy proxy。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_action_chunk_student.py`
- 新增逻辑：
  - `pure_student_window_len`
    - 只统计完全处于 pure-student 阶段的连续窗口，避免 mixed-policy 历史污染 probe 条件集；
  - 固定 deploy probe buffer
    - 从 pure-student 阶段的真实访问分布中收集固定 `obs / proprio_hist / target_chunk`；
    - buffer 填满后冻结，后续训练用同一组条件做 deterministic 评测；
  - `model_best_deploy_probe`
    - 周期性在固定 buffer 上以 deterministic 推理评估 first-action MSE；
    - 若优于历史最佳则保存 `model_best_deploy_probe.ckpt`。

新增配置项（默认均为保守值）：
- `action_chunk_deploy_probe_size`
- `action_chunk_deploy_probe_batch_size`
- `action_chunk_deploy_probe_interval_steps`

验证：
- 语法/编译：
  - `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_action_chunk_student.py', cfile='/tmp/diffusion_action_chunk_student.pyc', doraise=True) ... PY`
- 3min smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 smoke_actionchunk_deploy_probe_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=5000 ++train.ppo.action_chunk_model_selection_warmup_steps=5000 ++train.ppo.action_chunk_ckpt_interval_steps=2000 ++train.ppo.action_chunk_deploy_probe_size=64 ++train.ppo.action_chunk_deploy_probe_batch_size=32 ++train.ppo.action_chunk_deploy_probe_interval_steps=2000 ++task.env.numEnvs=24`
- smoke 结果：
  - 训练日志已出现 `Best Deploy Probe: 0.4382`
  - 已落出 `model_best_deploy_probe.ckpt`
  - 同时仍保留原有 `model_best / model_best_student / model_best_student_reward / model_last`

结论：
1. 这次新增的是一个真正“可运行的训练内 deploy proxy”，不是概念草图。
2. 它仍然不是完整 rollout reward，但相比：
   - 在线单步 `student_action_mse`
   - 稀疏且高噪声的 `student_eval_episode_reward`
   更稳定，也更接近 deterministic 部署语义。
3. 下一步不该继续先扫超参，而应先用当前 `alignsel_v1 len8` 口径跑一次正式 15min，对比：
   - `model_best_student`
   - `model_best_student_reward`
   - `model_best_deploy_probe`
   看 deploy probe 是否真的更接近最终 nominal 评测。

## 24. ActionChunk deploy probe 正式 15min 对照（P5 v21）
目的：
- 在与 `alignsel_v1 len8` 同口径的正式 15min 训练中，验证新加的 `model_best_deploy_probe` 是否比旧 selector 更接近最终 nominal 部署表现。

训练：
- run：`run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min`
- 命令：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_action_chunk.sh 0 42 run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" train.ppo.action_chunk_teacher_mix_steps=120000 ++train.ppo.action_chunk_model_selection_warmup_steps=120000 ++train.ppo.action_chunk_ckpt_interval_steps=50000 ++train.ppo.action_chunk_deploy_probe_size=512 ++train.ppo.action_chunk_deploy_probe_batch_size=256 ++train.ppo.action_chunk_deploy_probe_interval_steps=50000 train.ppo.action_chunk_first_action_bc_loss_coef=5.0 train.ppo.action_chunk_bc_loss_coef=0.2`
- 训练窗口末端观测：
  - `Current Best=1166.49`
  - `Best Student Reward=50.85`
  - `Best Student MSE=0.3953`
  - `Best Deploy Probe=0.4237`

单 seed nominal 评测（seed=42, steps=512）：
- `model_best_student.ckpt`
  - `avg_reward=-0.253256`, `avg_done_rate=0.013550`
- `model_last.ckpt`
  - `avg_reward=-0.388507`, `avg_done_rate=0.012248`
- `model_best_deploy_probe.ckpt`
  - `avg_reward=-0.590261`, `avg_done_rate=0.014242`
- `model_best_student_reward.ckpt`
  - `avg_reward=-0.899636`, `avg_done_rate=0.014689`

结论：
1. 当前这版 deploy probe **没有选出更好的部署 ckpt**。
2. 本轮排序反而是：
   - `model_best_student`
   - `model_last`
   - `model_best_deploy_probe`
   - `model_best_student_reward`
3. 这说明“固定 pure-student 条件集上的 deterministic first-action MSE”虽然比在线 MSE 更稳定，但仍不足以代表最终环境回报。
4. action-chunk 当前的主问题已进一步收敛到：
   - 不是简单的随机性 bug；
   - 也不只是 checkpoint selector 形式不对；
   - 更像 teacher imitation 目标与最终 contact-rich 部署 reward 之间仍存在结构性错位。

下一步建议：
1. 不要继续在 action-chunk 上反复堆 selector 小修补。
2. 若还继续救 action，下一层级应是真正的小 rollout probe 或 residual-style 改写，而不是再做 MSE 类 proxy。
3. 论文主线角度，当前 diffusion 主实现应继续以 latent diffusion 为主，action-chunk 作为“失败模式与分析分支”更合适。

## 25. Latent 主线回归验证（P5 v22）
目的：
- 按 handoff 建议回归 latent diffusion 主线，先确认当前最强 latent ckpt 在 deterministic 口径下是否真的稳定，再决定是否有必要继续强化训练。

候选筛查（seed=42, steps=512, nominal deterministic）：
- `run_a_seed42_15min/model_best.ckpt`
  - `avg_reward=1.682415`, `avg_done_rate=0.001058`
- `run_a_latent_robust_light_seed42_15min/model_best.ckpt`
  - `avg_reward=1.780555`, `avg_done_rate=0.001058`
- `run_a_latent_robust_light_seed42_1h/model_best.ckpt`
  - `avg_reward=1.773617`, `avg_done_rate=0.000936`

正式多 seed 复核（`run_a_latent_robust_light_seed42_15min/model_best.ckpt`）：
- 命令：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_det_multiseed512 "42,43,44"`
- 结果：
  - seed42：`1.780555 / 0.001058`
  - seed43：`2.146149 / 0.000936`
  - seed44：`2.322781 / 0.000895`
  - aggregate：`reward_mean=2.083162`, `reward_std=0.225799`, `done_mean=0.000963`, `done_std=0.000069`

light 多 seed 复核（same ckpt, steps=512）：
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_light_multiseed512 "42,43,44" task.env.randomization.obs_noise_e_scale=0.02 task.env.randomization.obs_noise_t_scale=0.01 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
- 结果：
  - seed42：`1.904964 / 0.001221`
  - seed43：`2.039789 / 0.000814`
  - seed44：`2.116688 / 0.000732`
  - aggregate：`reward_mean=2.020480`, `reward_std=0.087508`, `done_mean=0.000922`, `done_std=0.000214`

hard 多 seed 复核（same ckpt, steps=512）：
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_hard_multiseed512 "42,43,44" task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
- 结果：
  - seed42：`1.394853 / 0.001546`
  - seed43：`1.876549 / 0.000854`
  - seed44：`1.475745 / 0.001343`
  - aggregate：`reward_mean=1.582382`, `reward_std=0.210612`, `done_mean=0.001248`, `done_std=0.000290`

统一三挡补测（seed=42, steps=256）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 900 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 256 latent_robust_light_15min_nominal256_seed42`
  - `avg_reward=1.623421`, `avg_done_rate=0.001383`
- light：
  - `./docker-run-isaacgym.sh timeout 900 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 256 latent_robust_light_15min_light256_seed42 task.env.randomization.obs_noise_e_scale=0.02 task.env.randomization.obs_noise_t_scale=0.01 task.env.forceScale=0.5 task.env.randomForceProbScalar=0.1`
  - `avg_reward=1.661758`, `avg_done_rate=0.001872`
- hard：
  - `./docker-run-isaacgym.sh timeout 900 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 256 latent_robust_light_15min_hard256_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.412261`, `avg_done_rate=0.001953`

与现有主基线对照（同为 multiseed512）：
- `ProprioAdapt`：`2.400372 ± 0.151554`
- `PureBC`：`2.116345 ± 0.263339`
- `DiffusionLatent(旧 robust-light 统计)`：`1.742653 ± 0.104761`
- `DiffusionLatent(本轮 robust-light 15min det)`：`2.083162 ± 0.225799`

结论：
1. latent diffusion 主线已通过这轮回归验证，不是“只在 seed42 单点偶然偏高”。
2. 当前最强 diffusion ckpt 应更新为 `run_a_latent_robust_light_seed42_15min/model_best.ckpt`，而不是旧的 `run_a_seed42_15min/model_best.ckpt`。
3. 这版 latent 已基本追近 `PureBC`，但仍明显落后于 `ProprioAdapt`；当前更合理的动作不是立刻重训，而是先把这版 ckpt 固化为 diffusion 主代表。
4. 在统一三挡口径下，这版 latent 的 nominal/light/hard 全部保持正回报；而且多 seed 下 `light` 非常稳，说明它在轻扰动条件下已经具备较好的鲁棒性。
5. 当前 latent 与 baseline 的主要剩余差距，更像集中在 `hard` 档而不是 nominal 或 light 档。这说明下一步优化方向应从“泛化强化训练”收紧为“强扰动鲁棒性定向增强”。

下一步建议：
1. 将 `run_a_latent_robust_light_seed42_15min/model_best.ckpt` 固化为当前 diffusion 主代表。
2. 以 `hard` 档为主要短板，优先设计一次面向强扰动的 latent 定向增强，而不是笼统延长训练时长。
3. 只有当 hard 档也出现明确改善后，再决定是否值得把该配方扩展成更长训练预算。

## 26. Latent hard 档问题定位追加（P5 v23）
目的：
- 进一步定位 latent diffusion 在 hard 档下方差和掉分的主要原因，并用最小改动验证“residual anchor”与“更贴近 hard 的训练分布”是否能带来实质改善。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 新增可选开关：`train.ppo.diffusion_residual_base`
- 实现方式：
  - 以冻结 backbone 的 `adapt_tconv(proprio_hist)` 作为 base latent；
  - diffusion 只学习 residual `delta`，训练目标从 `e_gt` 改为 `e_gt - base_latent`；
  - 推理时输出改为 `tanh(base_latent + delta)`。

编译/冒烟验证：
- `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', doraise=True) ... PY`
  - 结果：`py_compile ok`
- `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
  - 结果：通过
- 3 分钟 smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 smoke_latent_residual_base_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_residual_base=True task.env.numEnvs=24`
  - 结果：训练链路无崩溃，产出 `model_best.ckpt`

正式 15min：residual-base latent
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_residual_base_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_residual_base=True`
- 训练窗口末端观测：
  - `Current Best=1772.90`
- 单 seed 评测（seed42, steps=512）：
  - nominal：
    - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_residual_base_nominal512_seed42 +train.ppo.diffusion_residual_base=True`
    - `avg_reward=1.531935`, `avg_done_rate=0.001099`
  - hard：
    - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_residual_base_hard512_seed42 +train.ppo.diffusion_residual_base=True task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
    - `avg_reward=1.237797`, `avg_done_rate=0.001994`

对照结论：
- 对比当前主代表 `run_a_latent_robust_light_seed42_15min/model_best.ckpt` 的 seed42/512：
  - nominal：`1.780555 -> 1.531935`
  - hard：`1.394853 -> 1.237797`
- 说明 residual-base 虽然把训练期 `Current Best` 抬得更高，但没有转化成真实部署收益，当前不能作为 latent 主线增强方案。

进一步分布定位（静态 hard 训练是否能直接补上 hard？）：
- full hard-match 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_robust_hardmatch_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" task.env.randomization.obs_noise_t_scale=0.025 task.env.randomization.obs_noise_e_scale=0.05 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - 观测：训练吞吐下降到约 `375 FPS`，`Current Best` 长时间停留在 `0.00~17.05`
  - 结论：直接用 hard 分布静态训练会把 15min student 蒸馏几乎完全压穿
- obs-noise hard only 15min（force 保持 `0.5/0.1`）：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obsnoise_hard_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" task.env.randomization.obs_noise_t_scale=0.025 task.env.randomization.obs_noise_e_scale=0.05`
  - 观测：训练同样长时间停留 `Current Best=0.00`
  - 结论：当前 hard 档的更直接主因已收敛到高 observation noise；即使不把 force 一起拉满，只把 obs 噪声提到 hard 级别，也足以让短预算 latent 蒸馏学不动

阶段结论：
1. `hard` 档差距目前更像是“高观测噪声下的训练可学性问题”，而不是单纯 force 不够强。
2. `residual-base` 不是本阶段的有效解法；它提高了训练峰值，但降低了 nominal 和 hard 真实评测。
3. “静态更强扰动训练”已经被这轮进一步否掉：无论 full hard 还是仅 obs-noise hard，都会把 15min 训练显著压穿。

下一步建议：
1. 暂不继续堆静态 hard 配置。
2. 下一轮应该实现一个最小的 hard curriculum，优先对 `obs_noise_e/t` 做 warmup 或分段提升，再看 hard 是否提升且 nominal 不回退。
3. 在 curriculum 之前，当前 latent 主代表仍保持为 `run_a_latent_robust_light_seed42_15min/model_best.ckpt`。

## 27. Latent obs-noise curriculum v1 与 1h hard 复核（P5 v24）
目的：
- 验证最小 `obs-noise curriculum` 是否能避免 hard 噪声直接压穿训练，并判断 hard 短板是否主要来自训练预算不足。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 新增可选配置：
  - `train.ppo.diffusion_obs_noise_curriculum`
  - `train.ppo.diffusion_obs_noise_curriculum_start`
  - `train.ppo.diffusion_obs_noise_curriculum_steps`
  - `train.ppo.diffusion_obs_noise_e_target`
  - `train.ppo.diffusion_obs_noise_t_target`
- 实现方式：
  - 训练循环中按 `agent_steps` 线性更新 `env.random_obs_noise_e_scale / t_scale`
  - 默认关闭，不影响现有 latent 主线

验证：
- 语法：
  - `python - <<'PY' ... py_compile.compile(..., cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`
  - `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
- 3 分钟 smoke：
  - `./docker-run-isaacgym.sh timeout 180 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 smoke_latent_obs_curr_seed42_3min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_steps=200000 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025 task.env.numEnvs=24`
  - 结果：未像静态 hard 一样卡死，`Current Best` 抬到 `275.45`

正式 15min：curriculum v1
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_curr_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1432.96`
- 单 seed 评测（seed42, steps=512）：
  - nominal：`avg_reward=1.673794`, `avg_done_rate=0.000854`
  - hard：`avg_reward=1.121324`, `avg_done_rate=0.001587`

curriculum v1 结论：
1. 这版 curriculum 的收益是“把训练从完全压穿，拉回到可学习状态”。
2. 但它没有把 hard 评测拉上来，反而低于当前 15min 主代表：
   - current mainline hard：`1.394853`
   - curriculum v1 hard：`1.121324`
3. 说明“线性从 light 拉到 hard obs-noise”还不是合适的课程设计。

训练预算复核：`run_a_latent_robust_light_seed42_1h`
- 单 seed hard（seed42, steps=512）：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_1h/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_1h_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.629211`, `avg_done_rate=0.001587`
- multiseed hard（steps=512）：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_1h/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_1h_hard_multiseed512 "42,43,44" task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - seed42：`1.629211 / 0.001587`
  - seed43：`1.512188 / 0.001058`
  - seed44：`1.410248 / 0.001180`
  - aggregate：`reward_mean=1.517216`, `reward_std=0.089462`, `done_mean=0.001275`, `done_std=0.000226`

与当前 15min 主代表 hard multiseed 对照：
- 15min robust-light：
  - `reward_mean=1.582382`, `reward_std=0.210612`
- 1h robust-light：
  - `reward_mean=1.517216`, `reward_std=0.089462`

阶段结论：
1. hard 短板并非“纯粹训练不够久”：
   - 延长到 1h 可以改善 seed42 单点 hard（`1.629211 > 1.394853`）；
   - 但 multiseed 平均并没有超过 15min。
2. 训练预算对 hard 有影响，但不是唯一主因；更像存在明显的 seed/轨迹分布交互。
3. 1h 的价值主要体现在“降低 hard 方差”，而不是“稳定提升 hard 均值”。
4. curriculum v1 与 1h 复核合起来说明：
   - hard 的确有可学习成分；
   - 但当前还没找到能把这种可学习成分稳定转化为 multiseed 均值提升的训练配方。

下一步建议：
1. 不再继续使用 `curriculum v1` 原样线性 schedule。
2. 下一轮应改成更保守的两阶段/分段 schedule，例如：
   - 先长时间保持 light；
   - 后段只抬 `obs_noise_t`；
   - 最后再抬 `obs_noise_e`。
3. 如果继续做长训练，也应配合多 seed 小样本验证，否则容易被单 seed 假提升误导。

## 28. Latent staged-te curriculum 15min 验收（P5 v25）
目的：
- 严格按“新策略先跑满 15min 再判断是否继续修改”的原则，验证更保守的两阶段 obs-noise curriculum 是否优于上一版线性 curriculum。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 在已有 `obs-noise curriculum` 基础上新增：
  - `diffusion_obs_noise_curriculum_mode`
  - `diffusion_obs_noise_curriculum_t_phase_ratio`
- 新模式 `staged_te`：
  - 前段只把 `obs_noise_t` 从 base 拉到 target，`obs_noise_e` 保持 base；
  - 后段固定 `obs_noise_t=target`，再把 `obs_noise_e` 拉到 target。

验证：
- `py_compile` 通过
- `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_stage_te_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.65 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1611.54`

评测（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_stage_te_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_stage_te_nominal512_seed42`
  - `avg_reward=1.580571`, `avg_done_rate=0.001058`
- hard：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_stage_te_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_stage_te_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.206818`, `avg_done_rate=0.001221`

与已有代表对照：
- 当前 15min 主代表 `run_a_latent_robust_light_seed42_15min`
  - nominal：`1.780555`
  - hard：`1.394853`
- 线性 curriculum v1
  - nominal：`1.673794`
  - hard：`1.121324`
- staged-te curriculum
  - nominal：`1.580571`
  - hard：`1.206818`

阶段结论：
1. `staged_te` 比线性 curriculum v1 更合理：
   - 训练峰值更高（`1611.54 > 1432.96`）
   - hard 也更高（`1.206818 > 1.121324`）
2. 但它仍未通过当前 15min 验收门槛：
   - nominal 和 hard 都没有超过现有 15min 主代表
3. 因此这轮策略应视为“有方向感但未达标”，不能升级为新主线。

下一步建议：
1. 继续保留 `run_a_latent_robust_light_seed42_15min/model_best.ckpt` 作为当前 diffusion 主代表。
2. 如果继续尝试 hard curriculum，下一轮应更保守：
   - 增加前段纯 light 保持期；
   - 再进入 `t -> e` 两阶段提升；
   - 并继续坚持“每个新策略必须先完成 15min 验证再继续修改”。

## 29. Latent staged-hold-te curriculum 15min 验收（P5 v26）
目的：
- 在 `staged_te` 基础上进一步保守化，增加前段纯 light 保持期，验证三阶段 `hold_light -> raise_t -> raise_e` 是否能通过 15min 验收。

代码改动：
- 文件：`dexscrew/algo/ppo/diffusion_latent_student.py`
- 新增配置：
  - `diffusion_obs_noise_curriculum_hold_ratio`
- 新模式：
  - `diffusion_obs_noise_curriculum_mode=staged_hold_te`
  - `hold` 期间维持当前 robust-light 噪声；
  - 中段只拉升 `obs_noise_t`；
  - 后段再拉升 `obs_noise_e`。

验证：
- `py_compile` 通过
- `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh` 通过

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.25 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.7 +train.ppo.diffusion_obs_noise_e_target=0.05 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1651.39`

评测（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_nominal512_seed42`
  - `avg_reward=1.396033`, `avg_done_rate=0.001424`
- hard：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.261073`, `avg_done_rate=0.001546`

与已有代表对照：
- 当前 15min 主代表 `run_a_latent_robust_light_seed42_15min`
  - nominal：`1.780555`
  - hard：`1.394853`
- `staged_te`
  - nominal：`1.580571`
  - hard：`1.206818`
- `staged_hold_te`
  - nominal：`1.396033`
  - hard：`1.261073`

阶段结论：
1. `staged_hold_te` 的 hard 再次提升，说明“先 hold 再进课程”方向是对的。
2. 但它的 nominal 明显下滑，导致这轮仍未通过 15min 验收。
3. 相比 `staged_te`：
   - hard 更好（`1.261073 > 1.206818`）
   - nominal 更差（`1.396033 < 1.580571`）
4. 这说明目前课程设计已经显现出更明确的 trade-off：
   - 越偏保守的 hard 课程，越容易牺牲 nominal 主表现。

下一步建议：
1. 继续保持当前主代表不变：`run_a_latent_robust_light_seed42_15min/model_best.ckpt`
2. 如果还继续试 curriculum，下一轮不要再一味加长 hold，而应尝试：
   - 缩短 hold 比例；
   - 保留 `t -> e` 分阶段；
   - 并降低 `e` 最终目标或延后 `e` 进入时机。
3. 继续坚持：每个新策略必须先完成 15min 训练与 seed42 nominal+hard 验收，再决定是否继续修改。

## 30. Latent staged-hold-te(e035) 15min 验收（P5 v27）
目的：
- 在三阶段课程框架不变的前提下，测试“更低的 `e` 最终目标”能否同时缓解 nominal 回退并继续抬升 hard。

本轮策略：
- 模式：`staged_hold_te`
- 配置：
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.80`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.035`

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_e035_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.80 +train.ppo.diffusion_obs_noise_e_target=0.035 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1689.30`

评测（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e035_nominal512_seed42`
  - `avg_reward=1.484588`, `avg_done_rate=0.001058`
- hard：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e035_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.316540`, `avg_done_rate=0.001831`

与前两版课程对照：
- `staged_te`
  - nominal：`1.580571`
  - hard：`1.206818`
- `staged_hold_te`
  - nominal：`1.396033`
  - hard：`1.261073`
- `staged_hold_te(e035)`
  - nominal：`1.484588`
  - hard：`1.316540`

阶段结论：
1. 这轮是目前 curriculum 线里最有价值的一次微调：
   - nominal 比上一版 `staged_hold_te` 明显恢复
   - hard 也继续提升
2. 但它仍未超过当前主代表 `run_a_latent_robust_light_seed42_15min`：
   - mainline nominal：`1.780555`
   - mainline hard：`1.394853`
3. 当前可以更明确地说：
   - 课程方向是有效的；
   - 降低 `e_target` 能改善 nominal/hard 的 trade-off；
   - 但这条线还没到“通过 15min 验收”的程度。

下一步建议：
1. 继续保持当前主代表不变。
2. 若继续试课程，优先沿“更弱 `e`、更强调 `t`”这条线走，而不是继续增大 hold。
3. 下一轮优先尝试：
   - 更低 `e_target`（如 `0.03`）
   - 或进一步延后 `e` 进入时机，
   依然保持每个新策略先跑满 15min 再判断。

## 31. Latent staged-hold-te(e03) 15min 验收（P5 v28）
目的：
- 验证在 `staged_hold_te` 框架下，把 `obs_noise_e_target` 从 `0.035` 继续降到 `0.03` 是否还能进一步改善 nominal/hard trade-off。

本轮策略：
- 模式：`staged_hold_te`
- 配置：
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.80`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.03`

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_e03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.80 +train.ppo.diffusion_obs_noise_e_target=0.03 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1689.30`

评测（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e03_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e03_nominal512_seed42`
  - `avg_reward=1.484588`, `avg_done_rate=0.001058`
- hard：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e03_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e03_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.316540`, `avg_done_rate=0.001831`

与上一版课程对照：
- `staged_hold_te(e035)`
  - nominal：`1.484588`
  - hard：`1.316540`
- `staged_hold_te(e03)`
  - nominal：`1.484588`
  - hard：`1.316540`

阶段结论：
1. 继续降低 `e_target` 没有带来额外收益，说明这一维度大概率已碰到平台期。
2. 当前课程线最有价值的候选仍然是 `run_a_latent_obs_hold_te_e035_seed42_15min`，但它仍未超过主代表 `run_a_latent_robust_light_seed42_15min`。
3. 下一步不应继续单独压低 `e_target`，而应转向：
   - 进一步延后 `e` 的进入时机；
   - 或先做 `e035` 的 multiseed hard 复核，再决定是否继续改课程。

## 32. Latent staged-hold-te(t90,e035) 15min 验收（P5 v29）
目的：
- 在不继续压低 `e_target` 的前提下，验证“进一步后移 `e` 的进入时机”是否能改善当前最佳 curriculum 候选 `staged_hold_te(e035)` 的 nominal/hard trade-off。

本轮策略：
- 模式：`staged_hold_te`
- 配置：
  - `hold_ratio=0.10`
  - `t_phase_ratio=0.90`
  - `obs_noise_t_target=0.025`
  - `obs_noise_e_target=0.035`

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_obs_hold_te_t90_e035_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_obs_noise_curriculum=True +train.ppo.diffusion_obs_noise_curriculum_mode=staged_hold_te +train.ppo.diffusion_obs_noise_curriculum_steps=250000 +train.ppo.diffusion_obs_noise_curriculum_hold_ratio=0.10 +train.ppo.diffusion_obs_noise_curriculum_t_phase_ratio=0.90 +train.ppo.diffusion_obs_noise_e_target=0.035 +train.ppo.diffusion_obs_noise_t_target=0.025`
- 训练窗口末端观测：
  - `Current Best=1866.55`

评测（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_t90_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_t90_e035_nominal512_seed42`
  - `avg_reward=1.388510`, `avg_done_rate=0.000651`
- hard：
  - `./docker-run-isaacgym.sh timeout 1200 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_t90_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_t90_e035_hard512_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.195002`, `avg_done_rate=0.001587`

与已有课程候选对照：
- `staged_hold_te(e035)`
  - nominal：`1.484588`
  - hard：`1.316540`
- `staged_hold_te(t90,e035)`
  - nominal：`1.388510`
  - hard：`1.195002`

阶段结论：
1. 单纯把 `e` 再进一步后移并没有改善 trade-off，反而 nominal 和 hard 都退步了。
2. 这说明当前课程线的主要瓶颈并不是“`e` 进入得还不够晚”。
3. 当前最有价值的 curriculum 候选仍然是 `run_a_latent_obs_hold_te_e035_seed42_15min`。
4. 若继续推进，优先级不应再放在 `staged_hold_te` 的单轴 `t/e` 时序微调，而应先验证 `e035` 的 multiseed hard 稳定性。

## 33. Latent staged-hold-te(e035) hard multiseed 复核（P5 v30）
目的：
- 对当前课程线里最有价值的候选 `run_a_latent_obs_hold_te_e035_seed42_15min` 做 `hard multiseed512` 复核，判断其提升是否稳定，而不是单 seed 偶然。

正式评测：
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_obs_hold_te_e035_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_obs_hold_te_e035_hard_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`

结果：
- seed42：
  - `avg_reward=1.286080`, `avg_done_rate=0.001506`
- seed43：
  - `avg_reward=1.512002`, `avg_done_rate=0.000854`
- seed44：
  - `avg_reward=1.318451`, `avg_done_rate=0.001261`
- aggregate：
  - `reward_mean=1.372178`, `reward_std=0.099750`
  - `done_mean=0.001207`, `done_std=0.000269`

与当前主代表对照：
- 当前主代表 `run_a_latent_robust_light_seed42_15min`
  - hard multiseed512：`1.582382 ± 0.210612`
- curriculum 候选 `staged_hold_te(e035)`
  - hard multiseed512：`1.372178 ± 0.099750`

阶段结论：
1. `e035` 在 hard 档并没有稳定逼近主代表，均值差距依然明显。
2. 它的方差更小，但“更稳地更差”并不足以支持把 curriculum 线升级成主线增强方向。
3. 到这一步可以阶段性收束当前 `obs-noise curriculum` 线：
   - 它提供了有价值的失败模式分析；
   - 但还没有拿到可接受的 hard 改善收益。
4. 后续 latent 主线若继续优化，应优先转向非 curriculum 的更高一级鲁棒性策略，而不是继续在 `staged_hold_te` 上微调。

## 34. Latent joint-decoder 非 curriculum 尝试失败（P5 v31）
目的：
- 验证在 latent diffusion 训练中，轻度解冻 student decoder（`actor_mlp + mu`）是否能改善当前 hard gap。

代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增可选参数 `train.ppo.diffusion_student_trainable_param_patterns`
  - 支持把匹配到的 student 参数重新设为 `requires_grad=True` 并纳入 diffusion optimizer

验证：
- `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
- `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_joint_decoder_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_student_trainable_param_patterns=[actor_mlp,mu]`
- 训练窗口内表现：
  - `Current Best=8.03`

阶段结论：
1. 联合解冻 `actor_mlp + mu` 会显著破坏 latent diffusion 训练稳定性。
2. 这条线不值得继续消耗预算，后续不应作为当前主线增强方向。

## 35. Latent latent_recon 非 curriculum 增强成功（P5 v32）
目的：
- 在不改 decoder 冻结形态、不引入 curriculum 的前提下，验证给 diffusion latent 增加直接 `x0/latent` 重建监督，能否同时改善 nominal 和 hard。

代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增可选参数 `train.ppo.diffusion_latent_recon_coef`
  - 训练时增加 `latent_recon_loss = MSE(tanh(x0_pred), teacher_latent)`
  - `loss = diffusion_loss + bc_loss + latent_recon_coef * latent_recon_loss`
  - 新增 tensorboard 指标 `latent_recon_loss`

验证：
- `bash -n scripts/screwdriver_student_diffusion_latent_robust_light.sh`
- `python - <<'PY' ... py_compile.compile('dexscrew/algo/ppo/diffusion_latent_student.py', cfile='/tmp/diffusion_latent_student.pyc', doraise=True) ... PY`

正式 15min：
- 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5`
- 训练峰值：
  - `Current Best=1786.12`

最小单 seed 验收（seed42, steps=512）：
- nominal：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_nominal_seed42`
  - `avg_reward=2.163711`, `avg_done_rate=0.000854`
- hard：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_hard_seed42 task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - `avg_reward=1.467671`, `avg_done_rate=0.001546`

multiseed512 复核：
- nominal：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_nominal_ms512 '42,43,44'`
  - seed42：`2.163711 / 0.000854`
  - seed43：`2.307207 / 0.000773`
  - seed44：`2.345508 / 0.000732`
  - aggregate：`reward_mean=2.272142`, `reward_std=0.078250`, `done_mean=0.000786`, `done_std=0.000051`
- hard：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_hard_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.05 task.env.randomization.obs_noise_t_scale=0.025 task.env.forceScale=1.5 task.env.randomForceProbScalar=0.3`
  - seed42：`1.467671 / 0.001546`
  - seed43：`1.649242 / 0.001099`
  - seed44：`1.834120 / 0.001139`
  - aggregate：`reward_mean=1.650344`, `reward_std=0.149604`, `done_mean=0.001261`, `done_std=0.000202`

与旧主代表对照：
- 旧主代表 `run_a_latent_robust_light_seed42_15min`
  - nominal multiseed512：`2.083162 ± 0.225799`
  - hard multiseed512：`1.582382 ± 0.210612`
- 新候选 `run_a_latent_recon05_seed42_15min`
  - nominal multiseed512：`2.272142 ± 0.078250`
  - hard multiseed512：`1.650344 ± 0.149604`

阶段结论：
1. `latent_recon05` 是当前 latent 主线的实质性升级，而不是单 seed 偶然。
2. 它同时提高了 nominal 和 hard，且在两档下都表现出更低方差。
3. 当前 `DiffusionLatentStudent` 的主代表应切换为：
   - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`

## 36. Latent latent_recon05 light multiseed512 复核（P5 v33）
目的：
- 补齐 `latent_recon05` 在 `light` 档的 multiseed512 结果，确认它是否在三挡统一评测口径下全面优于旧主代表。

正式评测：
- 命令：
  - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_light_ms512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`

结果：
- seed42：
  - `avg_reward=1.731917`, `avg_done_rate=0.001383`
- seed43：
  - `avg_reward=2.009683`, `avg_done_rate=0.001017`
- seed44：
  - `avg_reward=1.980781`, `avg_done_rate=0.000977`
- aggregate：
  - `reward_mean=1.907460`, `reward_std=0.124687`
  - `done_mean=0.001126`, `done_std=0.000183`

与旧主代表对照：
- 旧主代表 `run_a_latent_robust_light_seed42_15min`
  - light multiseed512：`2.020480 ± 0.087508`
- 新候选 `run_a_latent_recon05_seed42_15min`
  - light multiseed512：`1.907460 ± 0.124687`

阶段结论：
1. `latent_recon05` 不是“三挡都更好”的全面升级。
2. 它当前的真实画像应改写为：
   - nominal：更好
   - hard：更好
   - light：更差
3. 所以这条增强更像是把 latent 主线往“强扰动鲁棒性”方向推，而不是无代价普适提升。
4. 下一步不应继续盲目加大 `latent_recon_coef`，而应优先做更小系数回扫，寻找 `light / hard` 更平衡的点。

## 37. Latent latent_recon 小系数回扫失败（P5 v34）
目的：
- 验证把 `diffusion_latent_recon_coef` 从 `0.5` 下调到 `0.3 / 0.4` 后，是否能在保住 `nominal + hard` 收益的同时恢复 `light`。

正式 15min：
- `recon03`
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.3`
  - 训练峰值：`Current Best=1759.20`
- `recon04`
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon04_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.4`
  - 训练峰值：`Current Best=1876.57`

最小单 seed 三挡验收（seed42, steps=512）：
- `recon03`
  - nominal：
    - `avg_reward=1.764286`, `avg_done_rate=0.001139`
  - light：
    - `avg_reward=1.604022`, `avg_done_rate=0.001546`
  - hard：
    - `avg_reward=1.307460`, `avg_done_rate=0.001546`
- `recon04`
  - nominal：
    - `avg_reward=1.728607`, `avg_done_rate=0.001099`
  - light：
    - `avg_reward=1.565717`, `avg_done_rate=0.001343`
  - hard：
    - `avg_reward=1.260885`, `avg_done_rate=0.001587`

与当前主代表 `latent_recon05` 对照：
- `latent_recon05`（seed42, 512）
  - nominal：`2.163711`
  - light：`1.731917`
  - hard：`1.467671`
- `latent_recon03`
  - nominal：`1.764286`
  - light：`1.604022`
  - hard：`1.307460`
- `latent_recon04`
  - nominal：`1.728607`
  - light：`1.565717`
  - hard：`1.260885`

阶段结论：
1. 继续下调 `diffusion_latent_recon_coef` 并不能恢复 `light`，反而会把 `nominal / hard` 一起拉低。
2. `recon04` 训练峰值虽然更高，但最终部署三挡结果仍全面弱于 `recon05`，再次说明不能只看 `Current Best`。
3. 当前 `latent_recon` 主代表保持不变，仍为 `run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`。
4. 下一步不应继续在 `0.3 / 0.4` 区间盲扫，而应转向更有信息量的 `light gap` 诊断或结构性改动。

## 38. Latent light 对比口径纠正 + rollout 诊断链路打通（P5 v35）
目的：
- 修正 `latent_recon05` 与 `robust_light` 的 `light` 对比口径不一致问题，并为 student 分支补通可复用 rollout 诊断出口。

代码与工具：
- `dexscrew/algo/ppo/padapt.py`
  - 新增 student 版 `collect_rollout(...)`
  - rollout `.pt` 新增 `extras` 标量时间序列导出
- `dexscrew/algo/ppo/ppo.py`
  - teacher/PPO 版 `collect_rollout(...)` 同步支持 `extras`
- `scripts/analyze_rollout_diagnostics.py`
  - 可对比两份 rollout 的 `early/mid/late` 指标差异

纠偏评测：
- 对齐后的 `robust_light light_v2 multiseed512`（与 `latent_recon05 light` 使用同一扰动口径）：
  - 命令：
    - `./docker-run-isaacgym.sh timeout 2400 scripts/eval_screwdriver_student_robustness_multiseed.sh 0 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_robust_light_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_robust_light_15min_light_v2_multiseed512 '42,43,44' task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - 结果：
    - seed42：`1.873305 / 0.001302`
    - seed43：`1.885252 / 0.001017`
    - seed44：`2.048657 / 0.000854`
    - aggregate：`reward_mean=1.935738`, `reward_std=0.079995`, `done_mean=0.001058`, `done_std=0.000185`
- 已有 `latent_recon05 light multiseed512`（同口径）：
  - seed42：`1.731917 / 0.001383`
  - seed43：`2.009683 / 0.001017`
  - seed44：`1.980781 / 0.000977`
  - aggregate：`reward_mean=1.907460`, `reward_std=0.124687`, `done_mean=0.001126`, `done_std=0.000183`

rollout 质性诊断（seed42, steps=512, `light_v2`）：
- `python scripts/analyze_rollout_diagnostics.py outputs/diagnostics/latent_robust_light_light_seed42_steps512.pt outputs/diagnostics/latent_recon05_light_seed42_steps512.pt`
- 关键现象：
  - `recon05` 的 `reward_per_step` 呈现“中段更强、后段略弱”的阶段性 trade-off
  - 中段：
    - `rotation_reward` 更高（`0.377728 vs 0.342521`）
    - `screw/angular_velocity` 更高（`0.278265 vs 0.265611`）
    - `screw/positive_vel_ratio` 更高（`0.597222 vs 0.573465`）
  - 后段：
    - `reward_per_step` 更低（`0.913008 vs 1.021700`）
    - `screw/angular_position` 更低（`2.857802 vs 3.046834`）

阶段结论：
1. 历史上“`latent_recon05` 的 light 明显更差”这一说法被高估了，根源是比较口径不一致。
2. 在对齐后的 `light_v2` 口径下，`recon05` 与 `robust_light` 的均值差只有 `0.028278`，更像轻扰动下的 seed 稳定性差异，而不是大幅均值塌陷。
3. `recon05` 的 nominal + hard 收益依然成立，因此当前主代表不应因为旧版 `light` 对比而轻易回退。
4. 下一步应把注意力放在 `light_v2` 下的 seed 方差与阶段性 trade-off，而不是继续回扫 `latent_recon_coef`。

## 39. Latent light_v2 seed44 阶段诊断（P5 v36）
目的：
- 进一步判断 `latent_recon05` 在 `light_v2` 下的波动是后段保持问题，还是某些 eval seed 会在中段进入不同轨迹模式。

rollout 采样（seed44, steps=512, `light_v2`）：
- `robust_light`
  - `mean_reward=1.0075`, `mean_done_rate=0.0018`
- `latent_recon05`
  - `mean_reward=0.8812`, `mean_done_rate=0.0016`

关键对照：
1. 同一 `seed44` 下 `robust_light -> recon05`
- `reward_per_step`
  - early：`0.920506 -> 0.867773`
  - mid：`1.082685 -> 0.869660`
  - late：`1.018658 -> 0.906077`
- `rotation_reward`
  - early：`0.416165 -> 0.395524`
  - mid：`0.433682 -> 0.397807`
  - late：`0.425019 -> 0.370437`
- `pose_diff_penalty`
  - early：`0.384914 -> 0.401137`
  - mid：`0.433671 -> 0.453893`
  - late：`0.439408 -> 0.453329`
- `torques`
  - early：`0.356672 -> 0.355693`
  - mid：`0.318314 -> 0.351320`
  - late：`0.327741 -> 0.317508`

2. `latent_recon05` 的 `seed42 -> seed44`
- `reward_per_step`
  - early：`0.843274 -> 0.867773`
  - mid：`0.966963 -> 0.869660`
  - late：`0.913008 -> 0.906077`
- `rotation_reward`
  - early：`0.375052 -> 0.395524`
  - mid：`0.377728 -> 0.397807`
  - late：`0.366978 -> 0.370437`
- `pose_diff_penalty`
  - early：`0.384656 -> 0.401137`
  - mid：`0.443243 -> 0.453893`
  - late：`0.424187 -> 0.453329`
- `torques`
  - early：`0.342386 -> 0.355693`
  - mid：`0.305638 -> 0.351320`
  - late：`0.317554 -> 0.317508`

3. `robust_light` 的 `seed42 -> seed44`
- `reward_per_step`
  - early：`0.843559 -> 0.920506`
  - mid：`0.832410 -> 1.082685`
  - late：`1.021700 -> 1.018658`

阶段结论：
1. `robust_light` 自身也有 eval-seed 波动，但它在 `seed44` 反而更强，尤其是中段明显抬升。
2. `latent_recon05` 的回落不是“全程更差”，而是主要卡在中段 reward 对齐：
   - mid 段 `rotation_reward` 没有跟上 `robust_light`
   - 同时 `pose_diff_penalty` 和 `torques` 更高
3. `latent_recon05` 的某些 seed 更像进入了“动作更激进、螺钉角速度和角位置不低，但 reward 对齐更差”的模式。
4. 因此下一步应优先做 reward-aligned 的中段稳态增强，而不是继续做泛化超参扫或单纯回扫 `recon_coef`。

## 40. Latent reward-aligned 最小试探：`bc15` 与 `base-action anchor`
目的：
- 在不改 teacher/student 主路径的前提下，做两个最小的 reward-aligned 修复试探，验证能否缓解 `latent_recon05` 在 `light_v2` 下的 seed 级中段对齐问题。

### 40.1 更强 teacher BC：`run_a_latent_recon05_bc15_seed42_15min`
训练：
- 命令：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_bc15_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 train.ppo.bc_loss_coef=1.5`
- 峰值：
  - `Current Best=1846.98`

`light_v2` 定向验收（512 steps）：
- seed42：
  - `avg_reward=1.601600`, `avg_done_rate=0.001628`
- seed44：
  - `avg_reward=1.863312`, `avg_done_rate=0.001221`

结论：
- 训练峰值继续升高，但 `light_v2` 的两个已验收 seed 都低于 `latent_recon05` 原版：
  - seed42：`1.731917 -> 1.601600`
  - seed44：`1.980781 -> 1.863312`
- 因此“单纯增大 teacher action BC”可直接判为失败路线。

### 40.2 Base-student 动作锚定：`run_a_latent_recon05_baseanchor03_seed42_15min`
代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_base_action_anchor_coef`
  - 当系数大于 0 时，训练中额外加入：
    - `base_action_anchor_loss = || student_mu - base_student_mu ||^2`
  - 其中 `base_student_mu` 来自 frozen `adapt_tconv` latent 经过现有 actor head 的输出

验证：
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_baseanchor03_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_base_action_anchor_coef=0.3 task.env.numEnvs=8`
- 15min 训练：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_baseanchor03_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_base_action_anchor_coef=0.3`
  - `Current Best=1821.35`

`light_v2` multiseed 定向验收（512 steps）：
- seed42：
  - `avg_reward=1.570475`, `avg_done_rate=0.001180`
- seed43：
  - `avg_reward=1.891393`, `avg_done_rate=0.000936`
- seed44：
  - `avg_reward=2.040849`, `avg_done_rate=0.000854`
- aggregate：
  - `reward_mean=1.834239`, `reward_std=0.196244`
  - `done_mean=0.000990`

对比当前 `latent_recon05` 主代表的 `light_v2 multiseed512`：
- `latent_recon05`：`1.907460 ± 0.124687`
- `baseanchor03`：`1.834239 ± 0.196244`

阶段结论：
1. `base-action anchor` 不是完全无效，它对 `seed44` 有明显帮助：
   - `1.980781 -> 2.040849`
   - 已非常接近 `robust_light seed44=2.048657`
2. 但它同时显著伤害了 `seed42`，导致三 seed 均值和方差都劣于 `latent_recon05` 原版。
3. 因此当前更合理的解释是：
   - “向 base student 动作靠拢”可能确实碰到了问题的一部分
   - 但现在这版全局、静态的动作锚定过粗，只是把收益集中到个别 seed，而没有带来稳定的整体增益
4. 下一步不应继续盲扫 `bc_loss_coef / base_action_anchor_coef`，而应先用 rollout 诊断明确 `baseanchor03` 的 `seed44` 提升是否真的来自中段 `pose_diff_penalty / torques` 回落，再决定是否值得实现更直接的 torque-aware / action-magnitude regularizer。

## 41. Latent 诊断推进：`baseanchor03` 阶段对照与 `action_l2=0.01`
目的：
- 先完成 `latent_recon05 vs baseanchor03` 在 `seed42/44 + light_v2` 下的阶段诊断，再据此验证一个更直接、更轻量的全局动作幅度正则。

### 41.1 `baseanchor03` rollout 阶段诊断
新增 rollout：
- seed42：
  - `outputs/diagnostics/latent_baseanchor03_light_seed42_steps512.pt`
  - `mean_reward=0.9031`, `mean_done_rate=0.0023`
- seed44：
  - `outputs/diagnostics/latent_baseanchor03_light_seed44_steps512.pt`
  - `mean_reward=0.8226`, `mean_done_rate=0.0021`

关键对照 A：`latent_recon05 -> baseanchor03`（seed42）
- `reward_per_step`
  - early：`0.843274 -> 0.828503`
  - mid：`0.966963 -> 0.908913`
  - late：`0.913008 -> 0.971481`
- `rotation_reward`
  - early：`0.375052 -> 0.377986`
  - mid：`0.377728 -> 0.350038`
  - late：`0.366978 -> 0.388252`
- `pose_diff_penalty`
  - early：`0.384656 -> 0.381631`
  - mid：`0.443243 -> 0.436809`
  - late：`0.424187 -> 0.412276`

结论：
- `baseanchor03` 在 seed42 下不是单纯更稳，而是明显牺牲了中段推进，换来后段略好。
- 它的代价压制是有效的，但过粗，压掉了中段有效动作。

关键对照 B：`latent_recon05 -> baseanchor03`（seed44）
- `reward_per_step`
  - early：`0.867773 -> 0.761453`
  - mid：`0.869660 -> 0.838943`
  - late：`0.906077 -> 0.867086`
- `rotation_reward`
  - early：`0.395524 -> 0.368766`
  - mid：`0.397807 -> 0.367355`
  - late：`0.370437 -> 0.377346`
- `pose_diff_penalty`
  - early：`0.401137 -> 0.399395`
  - mid：`0.453893 -> 0.418299`
  - late：`0.453329 -> 0.423362`
- `torques`
  - early：`0.355693 -> 0.367208`
  - mid：`0.351320 -> 0.342736`
  - late：`0.317508 -> 0.333064`
- `angular_position`
  - early：`1.772533 -> 1.656514`
  - mid：`2.835959 -> 2.633136`
  - late：`2.957677 -> 2.612854`

结论：
- `baseanchor03` 在 seed44 下确实降低了部分中后段代价项，但同时把推进幅度也压下去了。
- 说明“全局向 base 动作靠拢”并不是真正需要的机制；它更像过强的静态收缩。

### 41.2 更直接的全局动作幅度正则：`action_l2=0.01`
代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_action_l2_coef`
  - 新增 `action_l2_loss = mean(student_mu^2)`，默认关闭

验证：
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_actionl2_001_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_action_l2_coef=0.01 task.env.numEnvs=8`
- 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_actionl2_001_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_action_l2_coef=0.01`
  - `Current Best=1824.54`
- 最小验收：
  - `light_v2 seed42 / 512 = 1.532299`, `done_rate=0.001261`

对比：
- `latent_recon05` 原版 seed42：`1.731917`
- `baseanchor03` seed42：`1.570475`
- `action_l2=0.01` seed42：`1.532299`

阶段结论：
1. 这轮结果把“全局静态收缩”这类方向又收窄了一步：
   - `baseanchor03` 不够好
   - `action_l2=0.01` 也不够好
2. 两者共同说明：
   - 问题不是简单“动作太大，需要整体压小”
   - 更像“只在某些阶段/某些样本上存在过激动作或 reward alignment 偏差”
3. 因此下一步若继续推进 latent，不应再做全局动作抑制，而应转向：
   - 更有方向性的 tail-only / thresholded regularizer
   - 或者只对 student-teacher 偏差过大的样本加额外约束

## 42. Latent 定向正则推进：`teacher_delta_tail`
目的：
- 按上一轮结论，把“全局静态收缩”进一步收窄为更有方向性的 tail-only 正则，只惩罚 student 动作相对 teacher 动作偏差过大的尾部。

代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_coef`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_threshold`
  - 新增
    - `teacher_delta_tail = relu(abs(student_mu - teacher_mu) - threshold)`
    - `teacher_delta_tail_loss = mean(teacher_delta_tail^2)`
  - 训练日志新增 `teacher_delta_tail_loss`

验证：
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 task.env.numEnvs=8`
  - 结果：训练能正常启动并持续输出，无新的实现级报错
- 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_t025_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25`
  - `Current Best=1863.07`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_t025_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_t025_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - `light_v2 seed42 / 512 = 1.602075`, `done_rate=0.001506`

对比：
- `latent_recon05` 原版 seed42：`1.731917`
- `baseanchor03` seed42：`1.570475`
- `action_l2=0.01` seed42：`1.532299`
- `teacher_delta_tail` seed42：`1.602075`

阶段结论：
1. `teacher_delta_tail` 比之前两条更粗的全局静态正则略好：
   - `1.602075 > 1.570475 > 1.532299`
2. 但它仍然没有超过当前主代表 `latent_recon05` 原版：
   - `1.602075 < 1.731917`
3. 这说明：
   - “只约束大偏差尾部”比“整体压动作/整体向 base 靠拢”更接近正确方向
   - 但当前实现仍然过于静态，它没有区分阶段，也没有区分哪些样本的偏差真正有害
4. 因此这条线暂不升级主线，下一步更适合继续缩小作用范围：
   - 只对高偏差样本加权
   - 或只在 rollout 诊断指向的问题阶段做定向约束

## 43. Latent 定向正则推进：`teacher_delta_tail_selective`
目的：
- 在 `teacher_delta_tail` 已证明“方向比全局静态正则更接近正确”之后，继续缩小作用范围，验证“只对真正激活的高偏差样本计入正则”是否能避免普通样本被过度约束。

代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_selective`
  - 当开启时，仅对 `teacher_delta_tail > 0` 的样本计算 `teacher_delta_tail_loss`
  - 新增日志：`teacher_delta_tail_active_ratio`

验证：
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_sel_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_selective=True task.env.numEnvs=8`
  - 结果：训练能正常启动并持续输出，无新的实现级报错
- 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_t025_sel_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_selective=True`
  - `Current Best=1941.87`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_t025_sel_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_t025_sel_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - `light_v2 seed42 / 512 = 1.471573`, `done_rate=0.001302`

对比：
- `latent_recon05` 原版 seed42：`1.731917`
- `teacher_delta_tail` seed42：`1.602075`
- `teacher_delta_tail_selective` seed42：`1.471573`

阶段结论：
1. 这轮结果把“只对激活样本计入尾部正则”先判为失败：
   - 虽然训练峰值继续升高到 `1941.87`
   - 但部署评测反而退到 `1.471573`
2. 这说明当前 selective 版本把训练目标进一步推离了最终部署目标：
   - 只保留激活样本的 tail loss，反而可能让正则过于稀疏、过于尖锐
   - 训练端更容易抬高 `Current Best`
   - 但没有转化成 `light_v2` 的稳定收益
3. 因此当前可收束的判断是：
   - 继续做“样本级静态 gating”并不是正确下一步
   - 如果还要沿正则线推进，应转向更有阶段语义的约束，而不是继续改静态样本选择逻辑

## 44. Latent 阶段感知正则：`teacher_delta_tail_mid_only`
目的：
- 在静态 `teacher_delta_tail` 和 `teacher_delta_tail_selective` 都没把部署评测拉回主代表之上后，转向更贴近 rollout 诊断的阶段感知版本，只在 episode 中段施加 tail regularizer。

代码改动：
- `dexscrew/algo/ppo/diffusion_latent_student.py`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_mid_only`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_progress_start`
  - 新增 `train.ppo.diffusion_teacher_delta_tail_progress_end`
  - 当开启时，仅在 `step_length / max_episode_length` 落在指定窗口内时施加 `teacher_delta_tail_loss`

验证：
- 语法：
  - `PYTHONDONTWRITEBYTECODE=1 python - <<'PY' ... compile(..., 'exec') ... PY`
- smoke：
  - `./docker-run-isaacgym.sh timeout 120 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail_mid_smoke "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.25 +train.ppo.diffusion_teacher_delta_tail_progress_end=0.75 task.env.numEnvs=8`
  - 结果：训练能正常启动，无新的实现级报错
- 15min：
  - `./docker-run-isaacgym.sh timeout 900 scripts/screwdriver_student_diffusion_latent_robust_light.sh 0 42 run_a_latent_recon05_deltatail05_mid2575_seed42_15min "checkpoint=outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_*.pth" +train.ppo.diffusion_latent_recon_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_coef=0.5 +train.ppo.diffusion_teacher_delta_tail_threshold=0.25 +train.ppo.diffusion_teacher_delta_tail_mid_only=True +train.ppo.diffusion_teacher_delta_tail_progress_start=0.25 +train.ppo.diffusion_teacher_delta_tail_progress_end=0.75`
  - `Current Best=1814.62`
- 最小验收：
  - `./docker-run-isaacgym.sh timeout 1800 scripts/eval_screwdriver_student_robustness.sh 0 42 DiffusionLatentStudent outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_deltatail05_mid2575_seed42_15min/stage2_diffusion_nn/model_best.ckpt 512 latent_recon05_deltatail05_mid2575_light_seed42 task.env.randomization.obs_noise_e_scale=0.03 task.env.randomization.obs_noise_t_scale=0.015 task.env.forceScale=1.0 task.env.randomForceProbScalar=0.2`
  - `light_v2 seed42 / 512 = 1.662729`, `done_rate=0.001139`

对比：
- `latent_recon05` 原版 seed42：`1.731917`
- `teacher_delta_tail` seed42：`1.602075`
- `teacher_delta_tail_selective` seed42：`1.471573`
- `teacher_delta_tail_mid_only(0.25-0.75)` seed42：`1.662729`

阶段结论：
1. 这是当前正则线里第一次明确的部署侧回升：
   - `1.471573 -> 1.662729`
   - 也超过了静态 tail 的 `1.602075`
2. 说明“阶段感知”比“静态样本 gating”更接近正确方向。
3. 但它仍未超过当前主代表 `latent_recon05`：
   - `1.662729 < 1.731917`
4. 因此当前最合理的判断是：
   - 中段窗口约束是有效线索，但窗口和强度还没调到最佳
   - 后续应优先沿这条线做小范围收敛，而不是回到静态正则
