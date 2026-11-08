# 项目算法总览与 Diffusion / Consistency / Flow 实现说明

本文档不是再写一遍计划，而是把当前仓库里的核心算法放到同一张图里讲清楚：

- 每个算法在项目里的定位是什么
- 它的输入、输出分别是什么
- 核心网络有多少层、每层在做什么
- 它在本项目里的实际表现如何
- 为什么最终主线还是 `padapt`，而不是某个 diffusion 变体

当前结论以仓库现状为准：

- 主线 baseline：`padapt`
- 最强 accepted diffusion reference：`V5.5 consistency_boundary_bc_tuned`
- `V6 / V7` consistency 扩展已收口，未超过 `padapt`
- `V8` flow matching recovery sprint 已收口，不具备继续推进资格

## 1. 整体工程链路

本项目的 canonical pipeline 是：

`XHandHoraScrewDriver -> PPO teacher -> student distillation family -> unified evaluation`

这里的 student family 不是第二阶段 RL policy，而是各种蒸馏/模仿学生：

1. `ProprioAdapt (padapt)`
2. `PureBC`
3. `DiffusionLatentStudent`
4. `ConsistencyLatentStudent`
5. `FlowMatchingLatentStudent`
6. `DiffusionActionChunkStudent`（历史 exploratory 分支）

统一评测协议长期固定为：

- `nominal`
- `light_v2`
- `hard`
- 固定 `steps=256`

## 2. 一句话定位：每个算法在项目里扮演什么角色

| 算法 | 项目定位 | 当前状态 | 一句话判断 |
|---|---|---|---|
| `teacher_ppo` | 上限参考 / 监督来源 | 稳定保留 | 不是 student 对手，而是 teacher ceiling |
| `padapt` | 主线 student baseline | 当前主线 | 最稳、最均衡、hard 最强 |
| `purebc` | 最简行为克隆对照组 | 保留 | 简单但 surprisingly strong，尤其 `light_v2/hard` reward |
| `diffusion_latent` | 最早的 latent diffusion 主线尝试 | 历史部分有效，但未升主线 | nominal 有信号，robust 不够 |
| `consistency_latent` | diffusion 家族里最成功的一条 | accepted secondary reference | nominal 很强，light 也接近，但 hard 仍没过 `padapt` |
| `flow_matching_latent` | V8 冲刺的 flow-native 恢复路线 | concluded | 连原 flow baseline 都没追回，不具备继续资格 |
| `diffusion_action_chunk` | 直接建模短时动作块的 exploratory 分支 | 冻结 | 工程上很复杂，部署侧收益不成立 |

## 3. 先看公共骨架：所有 student 基本都复用了什么

核心共享模型在：

- [models.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/models/models.py:1)
- [block.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/models/block.py:1)

### 3.1 共享输入张量

按当前默认配置，student 相关算法最常见的输入有：

1. `obs`
- 当前时刻观测向量
- 进入 actor 解码器

2. `proprio_hist`
- 形状语义：`[B, T, 24]`
- 其中 `T` 是历史长度，当前配置 `proprio_len=30`
- 这是 student 最关键的可部署输入

3. `priv_info`
- 只在训练监督中使用
- 用来生成 teacher latent target
- 部署时 student 不能依赖它

4. `point_cloud_info`
- 点云缓冲区
- 在当前 student 路线上，它更像 teacher latent target 的组成部分
- 不是 student 部署时直接拼给 actor 的原始输入

### 3.2 共享输出张量

最重要的输出有两个：

1. `latent / extrin`
- teacher 侧：来自 `env_mlp(priv_info)`，再加上点云特征
- student 侧：来自 `adapt_tconv(proprio_hist)` 或 diffusion/consistency/flow 头部预测

2. `mu`
- 最终动作均值
- 部署时真正送进环境的是 `clamp(mu, -1, 1)`

### 3.3 ActorCritic 骨架

当前保存下来的主线 config 使用的是：

- actor MLP: `[512, 256, 128]`
- privileged MLP: `[256, 128, 8]`
- point MLP: `[32, 32, 32]`

对应代码在 [models.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/models/models.py:24)。

可以把公共骨架拆成 5 个部分：

#### A. `env_mlp`

作用：
- 把 `priv_info` 编码成 teacher latent 的 privileged 部分

按当前 config 的层数：
- `Linear(priv_info_dim -> 256)`
- `ELU`
- `Linear(256 -> 128)`
- `ELU`
- `Linear(128 -> 8)`

说明：
- 最后一层不带激活，后面统一再 `tanh`
- 它输出的是 teacher latent 的 8 维 privileged core

#### B. `point_mlp`

作用：
- 把每个点云点的 3D 坐标编码成特征，再做 max-pool 得到全局点云摘要

按当前 config 的层数：
- `Linear(3 -> 32)`
- `ELU`
- `Linear(32 -> 32)`
- `ELU`
- `Linear(32 -> 32)`
- `ELU`

后处理：
- 对点维度做 `torch.max(..., dim=1)`
- 得到 32 维点云摘要

#### C. `adapt_tconv`

作用：
- 把 `proprio_hist` 压成 student latent
- 它是 `padapt / purebc / diffusion family` 的 student 侧共享入口

实现见 [TemporalConv](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/models/block.py:12)。

结构可以拆成 3 段：

1. `channel_transform`
- `Linear(24 -> 32)`
- `ReLU`
- `Linear(32 -> 32)`
- `ReLU`
- 作用：先把每个时间步的本体感觉从 24 维投到更适合时序卷积的通道维

2. `temporal_aggregation`
- `Conv1d(32 -> 32, kernel=9, stride=2)`
- `ReLU`
- `Conv1d(32 -> 32, kernel=5, stride=1)`
- `ReLU`
- `Conv1d(32 -> 32, kernel=5, stride=1)`
- `ReLU`
- 作用：把 30 帧历史压成一个低时序分辨率表示，抽取时间模式

3. `low_dim_proj`
- `Linear(32*3 -> output_dim)`
- 作用：把卷积后的时序特征展平后投成最终 latent

当前默认 `use_point_cloud_info=True` 时：
- `output_dim = 8 + 32 = 40`

这点非常关键：
- student latent 不是纯 8 维
- 它要同时拟合 teacher 的 privileged latent `8` 维部分
- 还要拟合点云摘要 `32` 维部分
- 所以当前 student latent 的默认维度是 `40`

#### D. `actor_mlp`

作用：
- 把 `obs + latent` 解码成动作特征

按当前 config 的层数：
- `Linear(policy_input_dim -> 512)`
- `ELU`
- `Linear(512 -> 256)`
- `ELU`
- `Linear(256 -> 128)`
- `ELU`

这里 `policy_input_dim` 取决于：
- `obs_dim`
- 加上的 latent 维度（当前 student 路线通常是 40）

#### E. 输出头

作用：
- `mu`: 动作均值头
- `value`: PPO teacher 的价值头
- `sigma`: PPO teacher 的可学习对数方差参数

层数：
- `mu`: `Linear(128 -> action_dim)`
- `value`: `Linear(128 -> 1)`
- `sigma`: 独立参数向量，不是 MLP

## 4. Teacher PPO

核心代码：
- [ppo.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/ppo.py:1)

### 4.1 定位

`teacher_ppo` 是：

- 上限参考
- 蒸馏监督源
- 不是 student 分支中的候选算法

### 4.2 输入与输出

输入：
- `obs`
- `priv_info`
- 可选 `point_cloud_info`

输出：
- `mu`
- `value`
- `sigma`
- 训练时还会输出 PPO 所需的 log-prob / entropy 等

### 4.3 结构与层数

它复用前面那套 `ActorCritic` 骨架：

- `env_mlp`
- `point_mlp`
- `actor_mlp`
- `mu/value/sigma`

区别在于：
- teacher 训练的是整套 PPO
- student 训练通常只动局部模块或额外 head

### 4.4 项目内表现

它是 ceiling，不和 student 争主线。

当前 canonical multiseed 指标：
- nominal `3.055567`
- light_v2 `2.915410`
- hard `2.762886`

### 4.5 优劣势

优势：
- 上限高
- 稳定
- 是所有 student 的监督来源

劣势：
- 依赖 privileged information
- 不是最终 student 部署形态

## 5. ProprioAdapt (`padapt`)

核心代码：
- [padapt.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/padapt.py:1)

### 5.1 定位

`padapt` 是当前项目的主线 student baseline。

它的思想最直接：

- 不去做生成建模
- 不去多步采样
- 直接用 `proprio_hist -> latent`
- 再通过冻结/共享的 actor 解码成动作

### 5.2 输入与输出

训练输入：
- `obs`
- `proprio_hist`
- `priv_info`
- `point_cloud_info`

部署输入：
- `obs`
- `proprio_hist`
- `point_cloud_info` 仅用于保持接口一致，actor 真正依赖的是 student latent

训练监督目标：
- `e_gt`：teacher latent

输出：
- `mu`：最终动作
- `extrin`：student latent
- `extrin_gt`：teacher latent

### 5.3 核心前向

可以写成：

1. `z_student = tanh(adapt_tconv(proprio_hist))`
2. `z_teacher = tanh(env_mlp(priv_info))`
3. 如果开点云：
   - `pcs = point_mlp(point_cloud_info) -> maxpool -> 32 维`
   - `z_teacher = concat(z_teacher, pcs)`，变成 40 维
4. `a_student = mu(actor_mlp(concat(obs, z_student)))`

### 5.4 训练什么

`padapt` 默认只训练：
- `adapt_tconv`

也就是：
- actor 解码器基本不动
- student 只学“如何从 proprio history 预测 teacher latent”

这也是它稳定的关键原因之一。

### 5.5 项目内表现

当前 canonical multiseed：
- nominal `2.167820`
- light_v2 `2.079074`
- hard `1.838225`
- done: `0.001302 / 0.001221 / 0.001411`

### 5.6 优势

1. 路径最短
- `proprio_hist -> latent -> action`
- train/test 几乎没有路径错位

2. 训练对象小
- 默认只训 `adapt_tconv`
- 工程风险低

3. hard 最稳
- 当前所有 student 系列里，它仍是 hard 条件主线 baseline

### 5.7 劣势

1. 表达能力受限
- latent 是单次 deterministic 回归
- 没有显式建模一对多或多模态结构

2. 能力上限受 frozen decoder 限制
- 它更多是在拟合已有 teacher latent 语义
- 不太容易靠 student 自己“生成出更强策略”

## 6. PureBC

核心代码：
- [pure_bc.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/pure_bc.py:1)

### 6.1 定位

`PureBC` 是最简 student baseline。

它和 `padapt` 的骨架几乎一样，但去掉了 latent distillation 损失，只保留动作级 BC。

### 6.2 输入与输出

输入：
- 与 `padapt` 相同

输出：
- `mu`

监督：
- teacher action `teacher_mu`

### 6.3 结构与层数

网络结构和 `padapt` 一样：

- `adapt_tconv`
- `actor_mlp`
- `mu`

区别不在结构，而在 loss：

- `padapt` 更强调 latent 对齐
- `purebc` 只看动作是否接近 teacher

### 6.4 训练目标

核心损失只有：

- `L_bc = MSE(mu_student, mu_teacher)`

### 6.5 项目内表现

当前 canonical multiseed：
- nominal `1.882908`
- light_v2 `2.190508`
- hard `1.849765`
- done: `0.001302 / 0.001329 / 0.001411`

这个结果很有意思：

- nominal 不如 `padapt`
- `light_v2` reward 反而比 `padapt` 高
- `hard` reward 也略高于 `padapt`
- 但 nominal 波动更大，稳定性不如 `padapt`

### 6.6 优势

1. 最简单
- 目标单纯
- debug 成本低

2. 动作对齐直接
- 不需要额外讨论 latent 误差怎样传到 action

3. 在本项目里 surprisingly strong
- 特别是 `light_v2/hard` reward 不弱

### 6.7 劣势

1. nominal 稳定性不足
- multiseed 方差大

2. 可解释性较弱
- 它没有显式约束 latent 结构
- 更像“把动作硬拟合出来”

## 7. DiffusionLatentStudent

核心代码：
- [diffusion_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/diffusion_latent_student.py:1)

### 7.1 定位

这是最早的 latent diffusion 主线尝试。

它不直接扩散动作，而是：

- 扩散 teacher latent
- 再把生成的 latent 丢给原 actor 解码

这是一个典型的“保留 actor decoder，换上游 latent 生成器”的路线。

### 7.2 输入与输出

训练输入：
- `proprio_hist`
- `obs`
- `priv_info`
- `point_cloud_info`
- 以及 diffusion 训练额外需要的：
  - `x_t`
  - `t`
  - `noise`

部署输入：
- `proprio_hist`
- `obs`

中间输出：
- `eps_pred`：预测噪声
- `x0_pred / pred_latent`

最终输出：
- `mu`

### 7.3 头部网络结构

`LatentDiffusionHead` 的结构很清晰：

1. `hist_encoder`
- `Linear(T*24 -> hidden_dim)`
- `ELU`
- `Linear(hidden_dim -> hidden_dim)`
- `ELU`
- 作用：把 `proprio_hist` 展平后编码成条件特征

2. `t_embed`
- `Embedding(num_steps, t_dim)`
- 作用：给扩散时间步一个离散可学习嵌入

3. `denoiser`
- `Linear(hidden_dim + latent_dim + t_dim -> hidden_dim)`
- `ELU`
- `Linear(hidden_dim -> hidden_dim)`
- `ELU`
- `Linear(hidden_dim -> latent_dim)`
- 作用：预测当前 `x_t` 上的噪声

### 7.4 算法逻辑

训练时：

1. teacher latent 作为 `x0`
2. 前向加噪得到 `x_t`
3. 预测噪声 `eps_pred`
4. 再由 `x_t` 和 `eps_pred` 还原出 `pred_latent`
5. `pred_latent` 送入 actor 解码动作

推理时：

1. 从 `0` 或高斯噪声 latent 开始
2. 迭代 `diffusion_steps_infer` 步去噪
3. 得到最终 latent
4. 解码成动作

### 7.5 训练目标

这条线的 loss 最丰富，主要包括：

1. `L_diff`
- 噪声预测损失

2. `L_bc`
- student action 对齐 teacher action

3. `L_recon`
- latent 重建损失

4. `L_anchor`
- 把 student action 锚在 base student action 附近

5. `L_l2`
- action 正则

6. `L_tail`
- 对 teacher-action 偏差尾部做额外惩罚

### 7.6 项目内表现

它的历史定位是：
- 有局部 signal
- 但 robust 条件没有足够强到替代 `padapt`

早期 canonical multiseed 对比里：
- nominal `2.062867`
- light_v2 `1.788645`
- hard `1.572475`

后续也出现过更强的单支 `latent_recon05` deploy 候选，但整体仍没有稳定翻盘 `padapt`，所以没有升格成主线。

### 7.7 优势

1. 表达力强
- 能显式建模 latent 分布

2. 可以叠很多正则和辅助约束
- 灵活度高

3. 对研究写作友好
- 算法形态完整，分析维度多

### 7.8 劣势

1. 推理慢
- 需要多步采样

2. train/infer mismatch 风险高
- 训练看的是单步噪声预测
- 部署真正执行的是多步采样后的 latent

3. hard 条件增益不够稳定
- 本项目最终没有出现“robust 明确压过 `padapt`”的证据

## 8. ConsistencyLatentStudent

核心代码：
- [consistency_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/consistency_latent_student.py:1)

### 8.1 定位

这是 diffusion 家族里在本项目表现最好的一条。

它可以理解成：

- 保留 latent 生成思想
- 但不再走完整 diffusion 反向链
- 改成 consistency-style 的“单步或少步直接映射到 x0”

它是当前最强 accepted diffusion reference。

### 8.2 输入与输出

训练输入：
- `proprio_hist`
- `obs`
- `priv_info`
- `point_cloud_info`
- `x_hi / x_lo`
- 连续时间 `t_hi / t_lo`

部署输入：
- `proprio_hist`
- `obs`

中间输出：
- `pred_hi`
- `pred_lo_target`
- `pred_latent`

最终输出：
- `mu`

### 8.3 头部网络结构

`ConsistencyHead` 和 diffusion head 结构很像，但时间编码不同：

1. `hist_encoder`
- 2 层 `Linear + ELU`
- 把 `proprio_hist` 展平编码成条件特征

2. `t_embed`
- `SinusoidalPosEmbed`
- 不是离散 embedding，而是连续时间位置编码

3. `denoiser`
- 3 层线性网络
- 输入是 `hist_feat + x_t + t_feat`
- 输出直接是一个更接近 `x0` 的 latent

### 8.4 算法逻辑

训练时：

1. 采样同一目标 latent 对应的两个噪声级别 `x_hi` 和 `x_lo`
2. 当前模型预测 `pred_hi`
3. EMA target 或当前模型预测 `pred_lo_target`
4. 用二者的一致性误差做 `consistency_loss`
5. 再用 `t=0` 的 `boundary_loss` 约束模型在边界点恢复原 latent
6. 最后把预测 latent 或 rollout latent 解码成动作做 BC

推理时：

1. 从 `0` 或高斯噪声 latent 开始
2. 执行 `consistency_infer_steps` 次映射
3. 得到最终 latent

和 diffusion 相比：
- 采样链更短
- 更适合控制任务部署

### 8.5 训练目标

主要损失是：

1. `consistency_loss`
- 高噪声和低噪声两点的预测要一致

2. `boundary_loss`
- `t=0` 处要回到目标 latent

3. `bc_loss`
- 动作对齐 teacher

4. 可选：
- `base_action_anchor_loss`
- `action_l2_loss`

### 8.6 项目内表现

`V5.5 accepted` 候选 `consistency_boundary_bc_tuned` 的 multiseed 结果是：

- nominal `2.336284`
- light_v2 `2.044725`
- hard `1.714471`
- done mean:
  - nominal `0.000949`
  - light_v2 `0.001194`
  - hard `0.001601`

这说明：

1. nominal 很强
- 明显高于 `padapt`

2. light_v2 很接近 `padapt`
- reward 略低，但 done 很接近甚至更好

3. hard 仍不如 `padapt`
- 这也是它没能升级为主线 baseline 的根本原因

后续 `V6 / V7` 已把这条线继续往上推过：
- obs noise curriculum
- infer/train align
- EMA target recheck

但都没有把 hard 真正推过 `padapt` 的 gate。

### 8.7 优势

1. diffusion 家族里最贴近控制部署的一条
- 少步推理
- 路径更短

2. nominal 提升非常明显
- 说明它确实学到了比纯 deterministic latent 更丰富的结构

3. 工程风险比完整 diffusion 小
- train/test 对齐更容易做

### 8.8 劣势

1. hard 条件始终差一口气
- 这是决定主线资格的关键短板

2. 继续加工程修补收益递减
- `V6 / V7` 已证明这一点

3. 仍然需要额外 head 和额外推理过程
- 比 `padapt` 更复杂

## 9. FlowMatchingLatentStudent

核心代码：
- [flow_matching_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/flow_matching_latent_student.py:1)

### 9.1 定位

这是 `V8` 冲刺的 flow-native 路线。

它的目标不是直接超越 `padapt`，而是先证明：
- flow matching 至少值得继续推进

最终结论是否定的：
- `V8` 已收口
- flow 在当前 scope 下不具备继续资格

### 9.2 输入与输出

训练输入：
- `proprio_hist`
- `obs`
- `priv_info`
- `point_cloud_info`
- 以及 flow 训练特有的：
  - `x_t`
  - `t`
  - `x1`

部署输入：
- `proprio_hist`
- `obs`

中间输出：
- `v_pred`：预测速度场
- `pred_latent` 或 rollout latent

最终输出：
- `mu`

### 9.3 头部网络结构

`FlowMatchingHead` 和 consistency head 基本同型：

1. `hist_encoder`
- 2 层 `Linear + ELU`

2. `t_embed`
- `SinusoidalPosEmbed`

3. `denoiser`
- 3 层线性网络
- 输出不是噪声，而是 velocity field `v_pred`

### 9.4 算法逻辑

训练时：

1. 目标 latent 记为 `x0`
2. 采一个噪声端点 `x1`
3. 在线段上采样：
   - `x_t = (1 - t) * x0 + t * x1`
4. 目标速度：
   - `v_target = x1 - x0`
5. 模型预测：
   - `v_pred = F(proprio_hist, x_t, t)`
6. 再从解析式恢复近似 latent：
   - `pred_latent = x_t - t * v_pred`
7. 用 `pred_latent` 解码动作做 BC
8. 如果开 `flow_train_align_infer`，还会额外跑真实 rollout latent 再做 rollout BC

推理时：

1. 从 `0` 或高斯 latent 开始
2. 做若干步 Euler rollout：
   - `latent = latent - dt * v_pred`
3. 最后得到部署 latent

### 9.5 训练目标

主要包括：

1. `flow_loss`
- 速度场回归

2. `bc_loss`
- 解析式恢复 latent 解码出的动作对齐 teacher

3. `rollout_bc_loss`
- 可选，对真实 rollout latent 的动作做额外 BC

4. 可选：
- `base_action_anchor_loss`
- `action_l2_loss`

### 9.6 项目内表现

当前旧 flow baseline：
- nominal `1.782656`
- light_v2 `1.587523`
- hard `1.367413`

`V8` 先做了最小恢复补丁，再做：
- infer2 / infer4 eval-only probe
- 5 个 fresh seed42 候选 sweep

最终最好 fresh 候选 `align4_rollout` 也只有：
- nominal `1.114895`
- light_v2 `1.152107`
- hard `1.079496`

也就是说：
- 不只是没追上 `padapt`
- 连旧的 flow baseline 都没追回来

### 9.7 优势

1. 理论上训练目标干净
- 直接学速度场
- 不需要完整 diffusion 噪声日程

2. 推理链可以比 diffusion 更短

3. 结构上和 consistency 接近
- 工程上容易共享经验

### 9.8 劣势

1. train/infer alignment 仍然难
- 训练里常用的是解析式恢复 latent
- 部署里真正执行的是 rollout latent

2. 这个任务上 rollout 路径不稳
- infer2 略有信号
- infer4 更差

3. 在本项目里没有形成 continue-worthy 结果
- `V8` 已证据化收口

## 10. DiffusionActionChunkStudent

核心代码：
- [diffusion_action_chunk_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/diffusion_action_chunk_student.py:1)

### 10.1 定位

这条线不再生成 latent，而是直接生成一个短 horizon 的动作块。

这是一个更激进的 exploratory 分支：

- 不再严格复用“先 latent 再 decode”的主设计
- 而是试图直接学未来若干步动作序列

### 10.2 输入与输出

输入：
- `obs`
- `proprio_hist`
- 训练时额外需要 teacher rollout dataset

输出：
- 一个长度为 `chunk_len` 的动作块
- 部署时按 receding horizon 只执行第一个动作，再滚动重算

### 10.3 头部结构

`ActionChunkDiffusionHead` 有 3 段：

1. `cond_encoder`
- 输入是 `obs + flatten(proprio_hist)`
- 2 层 `Linear + ELU`

2. `t_embed`
- 离散 diffusion step embedding

3. `denoiser`
- 3 层线性网络
- 输出维度是 `chunk_len * action_dim`

如果 `chunk_len=8`，那输出其实是一个扁平化的 8 步动作块。

### 10.4 项目内表现

这条线在工程上做了很多工作：
- teacher mix
- deploy probe
- selector
- rollout pretrain

但最终结论并不乐观：
- 训练 current best 可能变高
- 部署侧 selector 与真实 reward 对齐很差
- 某些变体 nominal 甚至会变成负回报

因此它被保留为历史 exploratory branch，而不是当前主推进方向。

### 10.5 优势

1. 理论上能表达短时动作相关性

2. 适合研究“单步 action 不够时，chunk 是否能补”

### 10.6 劣势

1. 最复杂
- 数据组织、训练、部署、selector 都复杂

2. train/deploy gap 最大
- chunk 级损失不等于第一步执行 reward

3. 本项目里没有形成稳定收益

## 11. 结果对比：它们在这个项目里谁更强

### 11.1 当前最重要的几条结果

统一对比最值得看的 5 个代表：

| 算法 | nominal | light_v2 | hard | 结论 |
|---|---:|---:|---:|---|
| `padapt` | 2.167820 | 2.079074 | 1.838225 | 当前主线 baseline |
| `purebc` | 1.882908 | 2.190508 | 1.849765 | 简单但强，尤其 robust reward 不差 |
| `diffusion_latent` | 2.062867 | 1.788645 | 1.572475 | 有 signal，但 robust 不足 |
| `consistency V5.5 accepted` | 2.336284 | 2.044725 | 1.714471 | diffusion 家族最强 reference |
| `flow baseline` | 1.782656 | 1.587523 | 1.367413 | 明显落后于主线 |

### 11.2 从这些结果能读出什么

1. `padapt` 是最均衡的
- 它不是每个单项都最强
- 但 hard 最可靠，这在本项目里权重最高

2. `purebc` 比直觉里更强
- 特别是 `light_v2/hard reward`
- 说明这个任务里 teacher action imitation 本身就能带来很强基线

3. `consistency` 是 diffusion 家族里最成功的
- nominal 明显最强
- light 基本能打
- 但 hard 还是差 `padapt` 一截

4. `flow` 目前没有继续价值
- `V8` 已经把“是否值得继续”这个问题回答掉了

## 12. 为什么最终主线不是 diffusion，而还是 `padapt`

这是本项目最重要的经验结论。

### 12.1 因为这个任务更奖励“部署路径短、错位少”的算法

`padapt` 的部署路径是：

`proprio_hist -> adapt_tconv -> actor -> action`

而 diffusion / flow 往往多了一层：

- 训练目标和部署 rollout 不完全一致
- latent 生成误差要再经过 actor 解码才会体现在 reward 上

这类接触任务对误差非常敏感，所以一旦多步生成链不稳，hard 条件就先掉。

### 12.2 因为本项目的 frozen actor decoder 很强，也很“挑 latent”

actor 已经是一个很强的解码器：
- 给它一个对的 latent，它能做得很好
- 但 latent 只要有一点系统性偏移，hard reward 就会受影响

这使得：
- `padapt` 这种直接回归 teacher latent 的路线更稳
- 生成式路线虽然 nominal 常常有亮点，但 robust 转化更难

### 12.3 因为 hard 条件是最终 gate

在本项目里，决定主线资格的不是：
- nominal 漂不漂亮

而是：
- `hard reward`
- `hard done`

`consistency` 的 nominal 很漂亮，但 hard 还是没越过 `padapt`。
这就是它最终只能成为 accepted secondary reference，而不是 mainline baseline 的原因。

## 13. 当前建议的算法定位

如果现在要写论文、做汇报、或者继续组织代码，推荐这样定位：

1. `teacher_ppo`
- ceiling / supervisor

2. `padapt`
- current mainline baseline
- 最推荐保留为主对照

3. `purebc`
- simple but strong imitation control baseline
- 非常适合拿来证明“复杂方法不一定天然更强”

4. `diffusion_latent`
- 早期主要生成式路线
- 有研究意义，但最终未赢下主线资格

5. `consistency_latent`
- strongest accepted diffusion reference
- 代表“生成式 student 在本项目中最接近成功的一条线”

6. `flow_matching_latent`
- V8 recovery sprint 后 conclude
- 目前不建议继续在同一 scope 内投入

7. `diffusion_action_chunk`
- historical exploratory appendix branch
- 更适合放进“负结果与工程经验”而不是主结果

## 14. 入口文件速查

### 14.1 训练/评测入口

- [train.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/train.py:1)
- [student_eval.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/student_eval.py:1)

### 14.2 核心算法实现

- [ppo.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/ppo.py:1)
- [padapt.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/padapt.py:1)
- [pure_bc.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/pure_bc.py:1)
- [diffusion_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/diffusion_latent_student.py:1)
- [consistency_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/consistency_latent_student.py:1)
- [flow_matching_latent_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/flow_matching_latent_student.py:1)
- [diffusion_action_chunk_student.py](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/dexscrew/algo/ppo/diffusion_action_chunk_student.py:1)

### 14.3 结果与结论文档

- [stage_acceptance_summary.md](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/docs/stage_acceptance_summary.md:1)
- [plansv5_5_final_verdict.md](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/docs/plansv5_5_final_verdict.md:1)
- [plansv8_final_verdict.md](/home/wbz-ubuntu22-pc/Codefield/py/dexscrew-repro/docs/plansv8_final_verdict.md:1)

## 15. 最后一句话总结

如果只用一句话概括当前仓库的算法格局：

`padapt` 依然是最稳的主线 student；`purebc` 是简单但强的对照；`consistency` 是最成功的 diffusion 参考；`flow` 和 `action-chunk` 都已经在当前 scope 下证据化收口。
