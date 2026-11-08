# 四种 Diffusion / 生成式 Student 蒸馏算法说明

本文参考 `output_docs/algo_adapt.md` 的组织方式，系统总结当前仓库中四类 diffusion / 生成式 student 的数学形式、训练目标、代码实现和推理逻辑。

这里的四类算法是：

```text
1. DiffusionLatentStudent        : latent DDPM
2. ConsistencyLatentStudent      : consistency latent model
3. FlowMatchingLatentStudent     : flow-matching latent ODE model
4. DiffusionActionChunkStudent   : action-chunk DDPM
```

它们都不是第二阶段 RL。它们和 PADAPT 一样，属于 teacher-student imitation/distillation 路线：teacher 是已经训练好的 PPO policy；student 在仿真 rollout 中查询 teacher supervision，用监督学习目标训练生成式模块。

## 1. 总体位置

项目当前 teacher-student 结构可以抽象成：

```text
PPO teacher:
  obs + priv_info + point_cloud -> teacher latent z_T -> teacher action a_T

PADAPT student:
  obs + proprio_hist -> deterministic latent z_S -> student action a_S

Diffusion/生成式 student:
  obs + proprio_hist -> generated latent/action -> student action a_S
```

四种 diffusion 的核心差别在于生成对象不同：

| 算法 | 生成对象 | 条件输入 | 推理输出 | 执行动作 |
|---|---|---|---|---|
| `DiffusionLatentStudent` | teacher latent `z_T` | `proprio_hist` | `z_S` | `actor(obs, z_S)` |
| `ConsistencyLatentStudent` | teacher latent `z_T` | `proprio_hist` | `z_S` | `actor(obs, z_S)` |
| `FlowMatchingLatentStudent` | teacher latent `z_T` | `proprio_hist` | `z_S` | `actor(obs, z_S)` |
| `DiffusionActionChunkStudent` | teacher action chunk `a_{t:t+K-1}` | `obs + proprio_hist` | action chunk | first action |

前三种算法在 latent 空间生成。它们继承 PPO teacher 的 actor head，只替换 `adapt_tconv(proprio_hist)` 这一条 deterministic latent 路径。

第四种算法直接在 action 空间建模短时动作序列，不再先生成 latent。

## 2. 共同基础

### 2.1 代码入口

训练入口仍然是：

```text
train.py
```

算法类：

```text
dexscrew/algo/ppo/diffusion_latent_student.py
dexscrew/algo/ppo/consistency_latent_student.py
dexscrew/algo/ppo/flow_matching_latent_student.py
dexscrew/algo/ppo/diffusion_action_chunk_student.py
```

前三个 latent 算法也通过 `dexscrew/algo/student/` 重新导出：

```text
dexscrew/algo/student/diffusion_latent.py
dexscrew/algo/student/consistency_latent.py
dexscrew/algo/student/flow_matching_latent.py
```

在命令行里通过 `train.algo=<ClassName>` 选择：

```bash
python train.py task=XHandHoraScrewDriver train.algo=DiffusionLatentStudent ...
python train.py task=XHandHoraScrewDriver train.algo=ConsistencyLatentStudent ...
python train.py task=XHandHoraScrewDriver train.algo=FlowMatchingLatentStudent ...
python train.py task=XHandHoraScrewDriver train.algo=DiffusionActionChunkStudent ...
```

### 2.2 共同 teacher/student 张量

沿用 `algo_adapt.md` 中 PADAPT 的符号。

对 DexH13：

```text
numActions = 16
single proprio frame = concat(q_t, target_t)  # 32 dim
obs_t = last 3 frames flattened               # 96 dim
proprio_hist_t = last 30 frames               # [30, 32]
```

记：

```text
o_t   : normalized obs
h_t   : normalized proprio_hist
p_t   : normalized priv_info
c_t   : normalized point cloud
z_T   : teacher latent
z_B   : deterministic PADAPT base latent, tanh(adapt_tconv(h_t))
a_T   : teacher action mean after actor head
a_S   : student action
pi(.) : frozen actor head = mu(actor_mlp(concat(obs, latent)))
```

Teacher latent 和 teacher action：

```text
z_T = teacher_latent(o_t, p_t, c_t)
a_T = pi(o_t, z_T)
```

代码中 teacher latent 来自：

```python
_, _, _, _, e_gt = self.model._actor_critic(input_dict)
teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
teacher_x = self.model.actor_mlp(teacher_obs_input)
teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)
```

这里的 `e_gt` 就是本文的 `z_T`。

### 2.3 共同训练范式

这四类算法都继承自 `ProprioAdapt`：

```python
class DiffusionLatentStudent(ProprioAdapt)
class ConsistencyLatentStudent(ProprioAdapt)
class FlowMatchingLatentStudent(ProprioAdapt)
class DiffusionActionChunkStudent(ProprioAdapt)
```

共同流程是：

1. 构建 student-shaped `ActorCritic`。
2. 从 PPO teacher checkpoint 加载 `model` 和 normalizers。
3. 冻结 teacher/student backbone。
4. 新增一个生成式 head。
5. 用 student 当前 rollout 状态在线查询 teacher。
6. 用监督损失更新生成式 head。
7. 用 student action 推进环境。
8. reward 主要用于选择 checkpoint，不参与反向传播。

因此它们是 DAgger 风格的在线蒸馏，而不是 PPO/RL。

## 3. 共同网络骨架

### 3.1 Frozen ActorCritic

四个算法都复用：

```text
dexscrew/algo/models/models.py
```

核心模块：

```text
env_mlp      : priv_info -> privileged latent
point_mlp    : point cloud -> point-cloud latent
adapt_tconv  : proprio_hist -> deterministic student latent
actor_mlp    : concat(obs, latent) -> actor feature
mu           : actor feature -> action mean
```

前三个 latent 算法冻结这些模块，然后训练一个新的 latent generator。

Action-chunk diffusion 同样冻结 `ActorCritic`，但只用它在线产生 teacher action chunk 标签。

### 3.2 Normalization

训练和测试都依赖 checkpoint 中保存的统计量：

```text
running_mean_std       : obs normalization
sa_mean_std            : proprio_hist normalization
priv_mean_std          : priv_info normalization, teacher query/eval 需要
point_cloud_mean_std   : point cloud normalization, teacher query/eval 需要
```

注意：部署 student 时不需要 `priv_info` 和 `point_cloud_info` 作为输入，但训练/eval 代码仍需要这些统计去复现 teacher latent/action 监督信号。

## 4. 算法一：DiffusionLatentStudent

### 4.1 代码位置

```text
dexscrew/algo/ppo/diffusion_latent_student.py
```

核心类：

```python
class LatentDiffusionHead(nn.Module)
class DiffusionLatentStudent(ProprioAdapt)
```

输出目录：

```text
stage2_diffusion_nn/
stage2_diffusion_tb/
```

### 4.2 建模目标

该算法在 teacher latent 空间做 DDPM。

目标是学习：

```text
p_phi(z_T | h_t)
```

推理时先生成 student latent：

```text
z_S ~ p_phi(. | h_t)
a_S = pi(o_t, z_S)
```

也就是说，它不直接生成 action，而是生成一个能被 frozen teacher actor head 理解的 latent。

### 4.3 Diffusion head 结构

`LatentDiffusionHead`：

```text
hist_encoder:
  flatten(proprio_hist) -> Linear -> ELU -> Linear -> ELU

t_embed:
  integer diffusion step t -> Embedding

denoiser:
  concat(hist_feat, x_t, t_embed) -> MLP -> predicted noise epsilon_hat
```

公式：

```text
g_phi(h_t, x_t, t) = epsilon_hat
```

代码：

```python
hist_feat = self.hist_encoder(proprio_hist.reshape(B, -1))
t_feat = self.t_embed(t)
denoise_in = torch.cat([hist_feat, x_t, t_feat], dim=-1)
eps_pred = self.denoiser(denoise_in)
```

### 4.4 前向扩散过程

代码中使用线性 beta schedule：

```python
betas = linspace(beta_start, beta_end, diffusion_steps)
alphas = 1 - betas
alpha_bars = cumprod(alphas)
```

前向加噪：

```text
epsilon ~ N(0, I)
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

代码：

```python
def _q_sample(self, x0, t, noise):
    s1 = sqrt_alpha_bars[t]
    s2 = sqrt_one_minus_alpha_bars[t]
    return s1 * x0 + s2 * noise
```

默认 `x_0 = z_T`。

### 4.5 Residual latent 变体

`DiffusionLatentStudent` 还有一个重要变体：

```yaml
train.ppo.diffusion_residual_base: True
train.ppo.diffusion_residual_target_scale: <scale>
```

此时不是直接 diffusion `z_T`，而是 diffusion PADAPT base latent 的修正量：

```text
z_B = tanh(adapt_tconv(h_t))
r_T = z_T - z_B
x_0 = scale * r_T
```

推理还原：

```text
z_S = z_B + x_0_hat / scale
```

代码：

```python
residual_base_latent = self._get_base_latent(input_dict["proprio_hist"])
target_x0 = target_latent - residual_base_latent
target_x0 = target_x0 * self.residual_target_scale
...
x0_pred = x0_pred / self.residual_target_scale
pred_latent = x0_pred + residual_base_latent
```

这不是独立的第五种算法，而是 latent DDPM 的 residual 模式。

### 4.6 训练损失

训练时：

```text
t ~ Uniform({0, ..., T-1})
epsilon ~ N(0, I)
x_t = q(x_t | x_0)
epsilon_hat = g_phi(h_t, x_t, t)
```

#### 4.6.1 Noise prediction loss

```text
L_diff = mean(||epsilon_hat - epsilon||_2^2)
```

代码：

```python
eps_pred = self.diffusion_model(input_dict["proprio_hist"], x_t, t)
diffusion_loss = ((eps_pred - noise) ** 2).mean()
```

#### 4.6.2 Latent reconstruction loss

先从 `x_t` 和 `epsilon_hat` 反推 `x_0_hat`：

```text
x_0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t)
```

代码：

```python
x0_pred = self._predict_x0(x_t, t, eps_pred)
latent_recon_loss = ((pred_latent - target_latent) ** 2).mean()
```

损失：

```text
L_latent = mean(||z_hat - z_T||_2^2)
```

#### 4.6.3 Behavior cloning loss

用生成 latent 走 frozen actor head：

```text
a_S = pi(o_t, z_hat)
a_T = pi(o_t, z_T)
```

动作 BC：

```text
L_BC = mean(sum_i (a_S[i] - a_T[i])^2)
```

代码：

```python
student_obs_input = torch.cat([input_dict["obs"], pred_latent], dim=-1)
student_mu = self.model.mu(self.model.actor_mlp(student_obs_input))
bc_loss = torch.sum(
    self.recon_criterion(student_mu_clamped, teacher_mu_clamped),
    dim=-1,
).mean()
```

#### 4.6.4 可选 regularization

代码中还支持：

```text
L_anchor       : student action 接近 PADAPT base action
L_action_l2    : action L2
L_tail         : 只惩罚 student/teacher action 差距超过阈值的 tail error
```

对应配置：

```yaml
diffusion_base_action_anchor_coef
diffusion_action_l2_coef
diffusion_teacher_delta_tail_coef
diffusion_teacher_delta_tail_threshold
diffusion_teacher_delta_tail_selective
diffusion_teacher_delta_tail_mid_only
```

总损失：

```text
L =
  c_diff   * L_diff
+ c_bc     * L_BC
+ c_latent * L_latent
+ c_anchor * L_anchor
+ c_l2     * L_action_l2
+ c_tail   * L_tail
```

代码：

```python
loss = (
    diffusion_loss_coef * diffusion_loss
    + bc_loss_coef * bc_loss
    + latent_recon_coef_cur * latent_recon_loss
    + base_action_anchor_coef * base_action_anchor_loss
    + action_l2_coef * action_l2_loss
    + teacher_delta_tail_coef * teacher_delta_tail_loss
)
```

### 4.7 训练 rollout 里的动作

训练中环境使用：

```text
mu_env = clamp(a_S, -1, 1)
```

其中 `a_S` 来自当前 batch 中随机 timestep 的 `x_0_hat`，不是完整 reverse diffusion chain。

代码：

```python
mu_env = torch.clamp(student_mu.detach(), -1.0, 1.0)
obs_dict, r, done, info = self.env.step(mu_env)
```

这意味着训练 rollout 直接暴露在当前生成 latent 的状态分布下，仍然是在线蒸馏。

### 4.8 推理过程

推理时才执行完整 reverse DDPM。

初始化：

```text
x_T = 0             if diffusion_stochastic_infer=False
x_T ~ N(0, I)      if diffusion_stochastic_infer=True
```

反向迭代：

```text
for t = T-1, ..., 0:
    epsilon_hat = g_phi(h_t, x_t, t)
    mean = 1/sqrt(alpha_t) * (x_t - (1-alpha_t)/sqrt(1-alpha_bar_t) * epsilon_hat)
    x_{t-1} = mean                         deterministic
    x_{t-1} = mean + sqrt(beta_t) * noise  stochastic
```

代码：

```python
for t_idx in reversed(range(self.diffusion_steps_infer)):
    eps_pred = self.diffusion_model(proprio_hist, x, t)
    mean = ...
    x = mean or mean + sqrt(beta_t) * randn_like(x)
```

最终：

```text
z_S = x_0
a_S = pi(o_t, z_S)
```

Residual 模式下：

```text
z_S = z_B + x_0 / residual_target_scale
```

### 4.9 Eval 诊断

`test()` 会输出：

```text
EvalSummary
EvalReconSummary
EvalResidualSummary   # residual_base=True 时
```

关键指标：

```text
latent_mse
latent_l1
action_mse_to_teacher
residual_abs_mean
base_action_mse_to_teacher
```

`diffusion_eval_decode_only=True` 时，eval 不走 diffusion，而是直接使用：

```text
z_S = tanh(adapt_tconv(h_t))
```

这用于判断 diffusion 是否真的超过 PADAPT base latent decode。

## 5. 算法二：ConsistencyLatentStudent

### 5.1 代码位置

```text
dexscrew/algo/ppo/consistency_latent_student.py
```

核心类：

```python
class SinusoidalPosEmbed(nn.Module)
class ConsistencyHead(nn.Module)
class ConsistencyLatentStudent(ProprioAdapt)
```

输出目录：

```text
stage2_consistency_nn/
stage2_consistency_tb/
```

### 5.2 建模目标

Consistency model 不预测 diffusion noise，而是直接学习从任意噪声状态 `x_t` 映射回 clean latent：

```text
f_phi(h_t, x_t, t) -> z_T
```

直观理解：

```text
DDPM        : 多步预测噪声，逐步去噪
Consistency: 直接预测 clean x0，并要求不同 t 的预测一致
```

### 5.3 Head 结构

`ConsistencyHead` 和 diffusion latent head 很像：

```text
hist_encoder(flatten(proprio_hist))
SinusoidalPosEmbed(t_cont)
denoiser(concat(hist_feat, x_t, t_embed)) -> latent_dim
```

区别：

```text
DiffusionLatentHead  使用离散整数 timestep embedding
ConsistencyHead      使用连续 t 的 sinusoidal embedding
```

公式：

```text
f_phi(h_t, x_t, t) = z_hat
```

### 5.4 Consistency 路径

代码使用从 target latent 到 noise 的线性路径：

```text
x_1 ~ N(0, I)
x_t = (1 - t) * z_T + t * x_1
```

其中：

```text
t = 0 -> x_t = z_T
t = 1 -> x_t = x_1
```

训练时采样两个相邻时间：

```text
t_hi ~ Uniform(0, 1)
dt = 1 / consistency_num_scales
t_lo = max(t_hi - dt, 0)

x_hi = (1 - t_hi) * z_T + t_hi * x_1
x_lo = (1 - t_lo) * z_T + t_lo * x_1
```

### 5.5 一致性损失

高噪声点的预测应当和低噪声点的预测一致：

```text
z_hi = f_phi(h_t, x_hi, t_hi)
z_lo = f_target(h_t, x_lo, t_lo)

L_cons = mean(||z_hi - stopgrad(z_lo)||_2^2)
```

代码：

```python
pred_hi = self.consistency_model(input_dict["proprio_hist"], x_hi, t_hi)
with torch.no_grad():
    pred_lo_target = target_model(input_dict["proprio_hist"], x_lo, t_lo)
consistency_loss = ((pred_hi - pred_lo_target.detach()) ** 2).mean()
```

`target_model` 可以是：

```text
当前 consistency_model 的 stopgrad 输出
EMA consistency_ema_model
```

由配置控制：

```yaml
consistency_use_ema_target
consistency_ema_decay
consistency_infer_use_ema
```

### 5.6 Boundary loss

因为 `t=0` 时输入就是 clean target latent：

```text
x_0 = z_T
```

因此要求：

```text
f_phi(h_t, z_T, 0) = z_T
```

损失：

```text
L_boundary = mean(||f_phi(h_t, z_T, 0) - z_T||_2^2)
```

代码：

```python
t_zero = torch.zeros((batch_size,), device=self.device)
pred_zero = self.consistency_model(input_dict["proprio_hist"], target_latent, t_zero)
boundary_loss = ((pred_zero - target_latent) ** 2).mean()
```

### 5.7 BC 和辅助损失

默认用 `pred_hi` 作为当前生成 latent：

```text
z_hat = pred_hi
a_S = pi(o_t, z_hat)
a_T = pi(o_t, z_T)
```

动作 BC：

```text
L_BC = mean(sum_i (a_S[i] - a_T[i])^2)
```

如果开启：

```yaml
consistency_train_align_infer: True
```

训练时会改用真实 inference 采样路径：

```python
pred_latent = self.sample_latent_train(input_dict["proprio_hist"])
```

这让训练中的 action BC 更贴近测试时的 latent 生成方式。

辅助项：

```text
L_anchor    : 靠近 PADAPT base action
L_action_l2 : action L2
```

总损失：

```text
L =
  c_cons     * L_cons
+ c_boundary * L_boundary
+ c_bc       * L_BC
+ c_anchor   * L_anchor
+ c_l2       * L_action_l2
```

代码：

```python
loss = (
    consistency_loss_coef * consistency_loss
    + consistency_boundary_coef * boundary_loss
    + bc_loss_coef * bc_loss
    + base_action_anchor_coef * base_action_anchor_loss
    + action_l2_coef * action_l2_loss
)
```

### 5.8 推理过程

推理初始化：

```text
latent_0 = 0          if consistency_stochastic_infer=False
latent_0 ~ N(0, I)   if consistency_stochastic_infer=True
```

然后反复调用 consistency model：

```text
for idx = 0 ... consistency_infer_steps-1:
    t_cur = 1 - idx / consistency_infer_steps
    latent = f_phi(h_t, latent, t_cur)
```

代码：

```python
for idx in range(self.consistency_infer_steps):
    t_cur = 1.0 - idx / max(1, self.consistency_infer_steps)
    latent = model(proprio_hist, latent, t)
```

最后：

```text
a_S = pi(o_t, latent)
```

### 5.9 与 DDPM latent 的区别

```text
DiffusionLatentStudent:
  学 epsilon，按 DDPM reverse chain 多步去噪

ConsistencyLatentStudent:
  学 x0 映射，强调不同噪声等级输出一致
  推理可 1 step，也可多 step
```

Consistency 的优势是推理步数可以很少，甚至 1 step；代价是训练目标对 boundary、EMA、BC 权重比较敏感。

## 6. 算法三：FlowMatchingLatentStudent

### 6.1 代码位置

```text
dexscrew/algo/ppo/flow_matching_latent_student.py
```

核心类：

```python
class SinusoidalPosEmbed(nn.Module)
class FlowMatchingHead(nn.Module)
class FlowMatchingLatentStudent(ProprioAdapt)
```

输出目录：

```text
stage2_flow_nn/
stage2_flow_tb/
```

### 6.2 建模目标

Flow matching 不预测 noise，也不直接做 consistency x0 映射，而是学习一条从 noise 到 teacher latent 的 ODE velocity field。

目标是：

```text
v_phi(h_t, x_t, t) ≈ dx_t / dt
```

代码的线性路径仍然是：

```text
x_1 ~ N(0, I)
x_t = (1 - t) * z_T + t * x_1
```

因此真实速度是：

```text
v_target = d x_t / dt = x_1 - z_T
```

### 6.3 Head 结构

`FlowMatchingHead` 和 consistency head 同型：

```text
hist_encoder(flatten(proprio_hist))
SinusoidalPosEmbed(t)
denoiser(concat(hist_feat, x_t, t_embed)) -> latent velocity
```

公式：

```text
v_hat = v_phi(h_t, x_t, t)
```

### 6.4 Flow matching loss

采样：

```text
x_1 ~ N(0, I)
t ~ Uniform([sigma_min, 1])
x_t = (1 - t) * z_T + t * x_1
v_target = x_1 - z_T
```

训练损失：

```text
L_flow = mean(||v_phi(h_t, x_t, t) - (x_1 - z_T)||_2^2)
```

代码：

```python
t = torch.rand((batch_size,), device=self.device)
t = torch.clamp(t, min=self.flow_sigma_min, max=1.0)
x_t = (1.0 - t.unsqueeze(-1)) * target_latent + t.unsqueeze(-1) * x1
v_target = x1 - target_latent
v_pred = self.flow_model(input_dict["proprio_hist"], x_t, t)
flow_loss = ((v_pred - v_target) ** 2).mean()
```

### 6.5 单步 latent reconstruction

如果 velocity 正确，则：

```text
x_t = z_T + t * (x_1 - z_T)
v_target = x_1 - z_T
z_T = x_t - t * v_target
```

因此代码用：

```text
z_hat = x_t - t * v_pred
```

代码：

```python
pred_latent = x_t - t.unsqueeze(-1) * v_pred
```

然后走 frozen actor head 做 BC：

```text
a_S = pi(o_t, z_hat)
a_T = pi(o_t, z_T)
L_BC = mean(sum_i (a_S[i] - a_T[i])^2)
```

### 6.6 可选 rollout BC

Flow matching 有一个很重要的配置：

```yaml
flow_train_align_infer: True
flow_rollout_bc_coef: <coef>
```

如果打开，训练时会实际执行 ODE-style inference 得到 rollout latent，再对 rollout action 做 BC：

```text
z_rollout = ODE_sample(h_t)
a_rollout = pi(o_t, z_rollout)
L_rollout_BC = mean(sum_i (a_rollout[i] - a_T[i])^2)
```

代码：

```python
rollout_latent = self._sample_latent_rollout(...)
rollout_mu = self.model.mu(self.model.actor_mlp(rollout_obs_input))
rollout_bc_loss = ...
```

这个项用于减小“训练时单步 reconstruction”和“测试时多步 ODE rollout”之间的差异。

### 6.7 总损失

辅助项同样包括：

```text
L_anchor
L_action_l2
```

总损失：

```text
L =
  c_flow       * L_flow
+ c_bc         * L_BC
+ c_rollout_bc * L_rollout_BC
+ c_anchor     * L_anchor
+ c_l2         * L_action_l2
```

代码：

```python
loss = (
    flow_loss_coef * flow_loss
    + bc_loss_coef * bc_loss
    + flow_rollout_bc_coef * rollout_bc_loss
    + base_action_anchor_coef * base_action_anchor_loss
    + action_l2_coef * action_l2_loss
)
```

### 6.8 推理过程

初始化：

```text
z = N(0, I)   if flow_stochastic_infer=True
z = 0         if flow_stochastic_infer=False
```

离散 ODE 反向积分：

```text
dt = 1 / flow_infer_steps
for idx = 0 ... flow_infer_steps-1:
    t_cur = 1 - idx / flow_infer_steps
    v_hat = v_phi(h_t, z, t_cur)
    z = z - dt * v_hat
```

代码：

```python
latent = self._init_latent(batch_size, init_mode)
for idx in range(infer_steps):
    t_cur = 1.0 - idx / infer_steps
    v_pred = self.flow_model(proprio_hist, latent, t)
    latent = latent - dt * v_pred
```

最后：

```text
a_S = pi(o_t, z)
```

### 6.9 与 Consistency 的区别

```text
Consistency:
  f(h, x_t, t) 直接输出 x0 / z_T

Flow Matching:
  v(h, x_t, t) 输出速度场
  通过 ODE 积分从噪声走到 latent
```

Flow matching 在形式上更连续、更像概率流 ODE；但实际效果依赖 `flow_infer_steps`、初始化方式和 rollout BC 对齐。

## 7. 算法四：DiffusionActionChunkStudent

### 7.1 代码位置

```text
dexscrew/algo/ppo/diffusion_action_chunk_student.py
```

核心类/函数：

```python
build_action_chunk_rollout_dataset(...)
class ActionChunkDiffusionHead(nn.Module)
class DiffusionActionChunkStudent(ProprioAdapt)
```

输出目录：

```text
stage2_diffusion_action_chunk_nn/
stage2_diffusion_action_chunk_tb/
```

### 7.2 建模目标

Action-chunk diffusion 不生成 latent，而是直接生成未来 `K` 步动作：

```text
A_t = [a_t, a_{t+1}, ..., a_{t+K-1}]
A_t in R^{K * action_dim}
```

条件输入是：

```text
condition = concat(obs_t, proprio_hist_t)
```

目标是：

```text
p_phi(A_t | obs_t, proprio_hist_t)
```

推理时生成完整 action chunk，但环境只执行第一步：

```text
A_hat_t = sample_action_chunk(obs_t, h_t)
a_S = A_hat_t[0]
```

这就是 receding horizon / MPC-like 的执行方式。

### 7.3 ActionChunkDiffusionHead 结构

```text
cond_encoder:
  concat(obs, flatten(proprio_hist)) -> MLP

t_embed:
  integer diffusion timestep -> Embedding

denoiser:
  concat(cond_feat, flat_action_chunk_x_t, t_embed) -> predicted noise
```

公式：

```text
epsilon_hat = g_phi(o_t, h_t, X_t, t)
```

其中：

```text
X_t in R^{K * action_dim}
```

代码：

```python
cond = torch.cat([obs, proprio_hist.reshape(B, -1)], dim=-1)
cond_feat = self.cond_encoder(cond)
t_feat = self.t_embed(t)
eps_pred = self.denoiser(torch.cat([cond_feat, x_t, t_feat], dim=-1))
```

### 7.4 Teacher action chunk 构造

在线训练时，代码每一步先查询 teacher action：

```text
a_T(t) = pi(o_t, z_T)
```

然后放入 rolling buffer：

```python
self.teacher_action_buffer[:, -1, :] = teacher_mu
self.cond_obs_buffer[:, -1, :] = obs
self.cond_prop_buffer[:, -1, :, :] = proprio_hist
self.valid_window_len += 1
```

当窗口长度达到 `chunk_len` 后，训练样本是：

```text
condition = (obs at window start, proprio_hist at window start)
target_chunk = [teacher_mu_t, ..., teacher_mu_{t+K-1}]
```

也就是说，它学习“在当前条件下，teacher 接下来 K 步会怎么做”。

### 7.5 Offline rollout pretrain

该算法还支持从提前收集的 teacher rollout 里做离线预训练：

```yaml
rollout_pretrain_path
rollout_pretrain_updates
rollout_pretrain_batch_size
```

数据构造函数：

```python
build_action_chunk_rollout_dataset(rollout_payload, chunk_len)
```

输入 payload 需要：

```text
obs           : [T, N, obs_dim]
proprio_hist  : [T, N, hist_len, proprio_dim]
actions       : [T, N, action_dim]
```

输出：

```text
cond_obs      : [W*N, obs_dim]
cond_prop     : [W*N, hist_len, proprio_dim]
action_chunks : [W*N, chunk_len, action_dim]
```

### 7.6 Action chunk DDPM

目标 action chunk flatten：

```text
x_0 = flatten(A_t)
```

前向加噪：

```text
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
```

预测：

```text
epsilon_hat = g_phi(o_t, h_t, x_t, t)
```

noise loss：

```text
L_diff = mean(||epsilon_hat - epsilon||_2^2)
```

代码：

```python
target_flat = target_chunk.reshape(B, -1).detach()
t = torch.randint(0, self.diffusion_steps, (B,), device=self.device)
noise = torch.randn_like(target_flat)
x_t = self._q_sample(target_flat, t, noise)
eps_pred = self.diffusion_model(cond_obs, cond_prop, x_t, t)
diffusion_loss = ((eps_pred - noise) ** 2).mean()
```

### 7.7 Chunk reconstruction loss

从 `x_t` 反推 `x_0_hat`：

```text
x_0_hat = (x_t - sqrt(1-alpha_bar_t) * epsilon_hat) / sqrt(alpha_bar_t)
```

因为 action bounded 在 `[-1, 1]`，代码对重构 chunk 做：

```text
A_hat = tanh(x_0_hat)
```

然后有两个 BC 项：

```text
L_first = mean(||A_hat[:, 0] - A_target[:, 0]||_2^2)
L_chunk = mean(||A_hat - A_target||_2^2)
```

总训练损失：

```text
L =
  c_diff  * L_diff
+ c_first * L_first
+ c_chunk * L_chunk
```

代码：

```python
chunk_recon = torch.tanh(x0_pred).reshape(-1, chunk_len, actions_num)
first_action_bc_loss = ((chunk_recon[:, 0, :] - target_chunk[:, 0, :]) ** 2).mean()
chunk_bc_loss = ((chunk_recon - target_chunk) ** 2).mean()

loss = (
    diffusion_loss_coef * diffusion_loss
    + first_action_bc_loss_coef * first_action_bc_loss
    + chunk_bc_loss_coef * chunk_bc_loss
)
```

### 7.8 推理过程

推理初始化：

```text
X_T = 0             if action_chunk_stochastic_infer=False
X_T ~ N(0, I)      if action_chunk_stochastic_infer=True
```

反向 DDPM：

```text
for t = T-1, ..., 0:
    epsilon_hat = g_phi(o_t, h_t, X_t, t)
    X_{t-1} = DDPM reverse mean/noisy sample
```

最终：

```text
A_hat = tanh(X_0).reshape(B, chunk_len, action_dim)
a_S = clamp(A_hat[:, 0, :], -1, 1)
```

代码：

```python
chunk_actions = self.sample_action_chunk(obs, proprio_hist)
mu = torch.clamp(chunk_actions[:, 0, :], -1.0, 1.0)
```

### 7.9 Teacher mix

Action-chunk diffusion 有一个训练稳定化机制：

```yaml
action_chunk_teacher_mix_steps
```

混合比例：

```text
mix = max(0, 1 - agent_steps / teacher_mix_steps)
```

环境动作：

```text
a_env = mix * a_T + (1 - mix) * a_S
```

代码：

```python
if mix > 0.0:
    mu_env = clamp(mix * teacher_mu + (1.0 - mix) * student_action)
else:
    mu_env = student_action
```

这相当于先让环境分布不要立刻被未训练好的 student 拉崩，再逐步切到 pure student。

### 7.10 Checkpoint selection

Action-chunk diffusion 的 checkpoint 选择比前三个复杂，因为 mixed-policy reward 不等于 pure-student 部署效果。

代码中保存：

```text
model_best                  : online mean episode reward
model_best_student_reward   : teacher mix 结束后的 pure-student reward
model_best_student          : first-step student_action_mse 最低
model_best_deploy_probe     : fixed deploy probe first-action MSE 最低
model_last                  : 周期性 latest
```

相关指标：

```text
teacher_mix_ratio
student_action_mse
student_eval_episode_reward
deploy_probe_first_action_mse
deploy_probe_chunk_mse
```

这个设计是为了避免选到“训练时靠 teacher mix 高 reward，但部署时 student 自己不稳”的 checkpoint。

## 8. 四种算法的核心公式对比

### 8.1 Latent DDPM

```text
x_0 = z_T
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
epsilon_hat = g_phi(h, x_t, t)

L = ||epsilon_hat - epsilon||^2
  + BC(pi(o, x0_hat), pi(o, z_T))
  + optional regularizers
```

### 8.2 Consistency latent

```text
x_t = (1-t) z_T + t x_1
z_hi = f_phi(h, x_hi, t_hi)
z_lo = f_target(h, x_lo, t_lo)

L = ||z_hi - stopgrad(z_lo)||^2
  + ||f_phi(h, z_T, 0) - z_T||^2
  + BC(pi(o, z_hi), pi(o, z_T))
```

### 8.3 Flow matching latent

```text
x_t = (1-t) z_T + t x_1
v_target = x_1 - z_T
v_hat = v_phi(h, x_t, t)
z_hat = x_t - t v_hat

L = ||v_hat - v_target||^2
  + BC(pi(o, z_hat), pi(o, z_T))
```

### 8.4 Action-chunk DDPM

```text
x_0 = flatten([a_T(t), ..., a_T(t+K-1)])
x_t = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon
epsilon_hat = g_phi(o, h, x_t, t)
A_hat = tanh(x0_hat).reshape(K, action_dim)

L = ||epsilon_hat - epsilon||^2
  + ||A_hat[0] - A_target[0]||^2
  + ||A_hat - A_target||^2
```

## 9. 代码执行路径对比

| 阶段 | Latent DDPM | Consistency | Flow Matching | Action Chunk |
|---|---|---|---|---|
| 类 | `DiffusionLatentStudent` | `ConsistencyLatentStudent` | `FlowMatchingLatentStudent` | `DiffusionActionChunkStudent` |
| 生成 head | `LatentDiffusionHead` | `ConsistencyHead` | `FlowMatchingHead` | `ActionChunkDiffusionHead` |
| 训练对象 | latent noise | latent x0 consistency | latent velocity | action chunk noise |
| 条件 | `proprio_hist` | `proprio_hist` | `proprio_hist` | `obs + proprio_hist` |
| teacher 标签 | `z_T`, `a_T` | `z_T`, `a_T` | `z_T`, `a_T` | future `a_T` chunk |
| 测试采样 | reverse DDPM | repeated x0 map | ODE integration | reverse DDPM |
| 环境动作 | `pi(o,z)` | `pi(o,z)` | `pi(o,z)` | first action |
| 输出目录 | `stage2_diffusion_*` | `stage2_consistency_*` | `stage2_flow_*` | `stage2_diffusion_action_chunk_*` |

## 10. 训练循环差异

### 10.1 三种 latent generator

三种 latent generator 的训练循环都是：

```text
obs_dict = env.reset()
while training:
    normalize obs/proprio/priv/point_cloud
    query teacher latent z_T
    compute teacher action a_T
    generate z_hat with current method
    compute supervised losses
    update generator
    a_S = pi(o, z_hat)
    env.step(a_S)
```

关键点：

```text
env 是由 student action 推进的。
teacher 只提供监督标签。
reward 不进入 loss。
```

### 10.2 Action-chunk generator

Action-chunk 的训练循环多了 temporal buffer：

```text
query teacher action a_T
append a_T into rolling window
if window length >= K:
    train on action chunk
sample student chunk
execute first student action or teacher-mixed action
```

它的 temporal credit 不是通过 RNN 或 value learning，而是通过短时 teacher action window 的 supervised chunk target。

## 11. 推理和部署差异

### 11.1 Latent 方法部署

部署输入：

```text
obs
proprio_hist
```

推理：

```text
z_S = generated_latent(proprio_hist)
a_S = frozen_actor(obs, z_S)
```

不需要：

```text
priv_info
point_cloud_info
```

但 checkpoint 里仍保存这些 normalizer，方便 eval/复现实验。

### 11.2 Action-chunk 方法部署

部署输入：

```text
obs
proprio_hist
```

推理：

```text
chunk = sample_action_chunk(obs, proprio_hist)
a_S = chunk[0]
```

下一步重新观察，再重新生成一个新 chunk。这是 receding horizon，而不是一次生成后连续执行完整 chunk。

## 12. Checkpoint 内容

### 12.1 DiffusionLatentStudent

```text
model
diffusion_model
running_mean_std
sa_mean_std
priv_mean_std
point_cloud_mean_std
diffusion_optim
```

### 12.2 ConsistencyLatentStudent

```text
model
consistency_model
consistency_ema_model      # optional
consistency_optim
agent_steps
running_mean_std
sa_mean_std
priv_mean_std
point_cloud_mean_std
```

### 12.3 FlowMatchingLatentStudent

```text
model
flow_model
flow_optim
agent_steps
running_mean_std
sa_mean_std
priv_mean_std
point_cloud_mean_std
```

### 12.4 DiffusionActionChunkStudent

```text
model
diffusion_model
running_mean_std
sa_mean_std
priv_mean_std
point_cloud_mean_std
```

注意：action-chunk 当前 `save()` 没保存 optimizer 和 `agent_steps`，更偏向保存部署/评测 ckpt，而不是完整 resume ckpt。

## 13. 关键配置字段

### 13.1 DiffusionLatentStudent

```yaml
diffusion_steps
diffusion_steps_infer
diffusion_beta_start
diffusion_beta_end
diffusion_hidden_dim
diffusion_t_dim
diffusion_lr
diffusion_loss_coef
bc_loss_coef
diffusion_latent_recon_coef
diffusion_latent_recon_coef_start
diffusion_latent_recon_coef_end
diffusion_residual_base
diffusion_residual_target_scale
diffusion_base_action_anchor_coef
diffusion_action_l2_coef
diffusion_teacher_delta_tail_coef
diffusion_stochastic_infer
```

### 13.2 ConsistencyLatentStudent

```yaml
consistency_hidden_dim
consistency_t_dim
consistency_lr
consistency_loss_coef
consistency_boundary_coef
consistency_num_scales
consistency_ema_decay
consistency_use_ema_target
consistency_infer_use_ema
consistency_infer_steps
consistency_stochastic_infer
consistency_train_align_infer
consistency_action_l2_coef
bc_loss_coef
base_action_anchor_coef
```

### 13.3 FlowMatchingLatentStudent

```yaml
flow_hidden_dim
flow_t_dim
flow_lr
flow_loss_coef
flow_sigma_min
flow_infer_steps
flow_stochastic_infer
flow_train_init_mode
flow_train_align_infer
flow_rollout_bc_coef
flow_action_l2_coef
bc_loss_coef
base_action_anchor_coef
```

### 13.4 DiffusionActionChunkStudent

```yaml
action_chunk_len
action_chunk_diffusion_steps
action_chunk_diffusion_steps_infer
action_chunk_diffusion_beta_start
action_chunk_diffusion_beta_end
action_chunk_diffusion_hidden_dim
action_chunk_diffusion_t_dim
action_chunk_diffusion_lr
action_chunk_diffusion_loss_coef
action_chunk_first_action_bc_loss_coef
action_chunk_bc_loss_coef
action_chunk_stochastic_infer
action_chunk_teacher_mix_steps
action_chunk_model_selection_warmup_steps
action_chunk_ckpt_interval_steps
action_chunk_deploy_probe_size
rollout_pretrain_path
rollout_pretrain_updates
```

## 14. 适用性理解

### 14.1 Latent DDPM

优点：

- 和 PADAPT 接口最接近。
- 不改变 actor/action 语义。
- 可以用 residual mode 在 PADAPT base latent 上学修正。

风险：

- 多步 reverse diffusion 推理成本更高。
- 训练时用随机 timestep 的 `x0_pred` 推环境，测试时用完整 reverse chain，两者有分布差异。
- latent MSE 好不一定 action 稳。

### 14.2 Consistency

优点：

- 推理步数少，适合部署。
- 直接输出 clean latent。
- boundary loss 能强制 `t=0` 处对齐 teacher latent。

风险：

- 对 `boundary_coef`、`num_scales`、`bc_loss_coef` 很敏感。
- EMA target 和 train/infer alignment 会显著影响结果。

### 14.3 Flow Matching

优点：

- 连续时间建模清晰。
- 训练目标是 velocity field，形式上比 DDPM reverse chain 更直接。
- 可以通过 `flow_infer_steps` 调推理精度。

风险：

- 如果训练只用单步 reconstruction，测试多步 ODE rollout 可能不对齐。
- 需要 `flow_train_align_infer` / `flow_rollout_bc_coef` 这种额外项来减少偏差。

### 14.4 Action Chunk

优点：

- 直接建模短时动作时序。
- 适合表达多步接触动作、恢复动作、节律动作。
- 不依赖 latent 解释是否正确。

风险：

- 训练更复杂，需要 rolling window。
- 首步动作质量最关键，否则 chunk 后续动作没机会执行。
- mixed teacher reward 可能误导 checkpoint 选择。
- 部署侧只执行第一步，所以 `first_action_bc_loss` 和 deploy probe 很重要。

## 15. 和 PADAPT 的关系

PADAPT 学：

```text
z_S = tanh(adapt_tconv(h_t))
a_S = pi(o_t, z_S)

L_PADAPT = L_latent + L_BC
```

Diffusion/生成式方法替换的是：

```text
adapt_tconv(h_t) -> generated latent/action
```

但它们保留 PADAPT 的关键蒸馏思想：

```text
student 只看 proprio_hist
teacher 用 priv_info/point_cloud 提供监督
actor head 大多继承 teacher
环境由 student action 推进
```

区别是：

```text
PADAPT             : deterministic one-shot latent regression
Diffusion latent   : stochastic/iterative latent generation
Consistency latent : one/few-step clean latent mapping
Flow latent        : ODE velocity latent generation
Action chunk       : direct future action sequence generation
```

## 16. 最小伪代码

### 16.1 Latent DDPM

```python
obs_dict = env.reset()
while train:
    o, h, p, c = normalize(obs_dict)
    z_T = teacher_latent(o, p, c)
    a_T = actor(o, z_T)

    t = randint(0, T)
    eps = randn_like(z_T)
    x_t = sqrt_ab[t] * z_T + sqrt_1m_ab[t] * eps
    eps_hat = diffusion_head(h, x_t, t)
    z_hat = predict_x0(x_t, t, eps_hat)
    a_S = actor(o, z_hat)

    loss = mse(eps_hat, eps) + bc(a_S, a_T) + optional_terms
    update(diffusion_head)
    obs_dict = env.step(clamp(a_S))
```

### 16.2 Consistency

```python
z_T = teacher_latent(o, p, c)
x1 = randn_like(z_T)
t_hi = rand()
t_lo = max(t_hi - 1 / num_scales, 0)
x_hi = (1 - t_hi) * z_T + t_hi * x1
x_lo = (1 - t_lo) * z_T + t_lo * x1

z_hi = consistency_head(h, x_hi, t_hi)
z_lo = target_head(h, x_lo, t_lo).detach()
z_zero = consistency_head(h, z_T, 0)
a_S = actor(o, z_hi)
a_T = actor(o, z_T)

loss = mse(z_hi, z_lo) + boundary*mse(z_zero, z_T) + bc(a_S, a_T)
```

### 16.3 Flow Matching

```python
z_T = teacher_latent(o, p, c)
x1 = randn_like(z_T)
t = uniform(sigma_min, 1)
x_t = (1 - t) * z_T + t * x1
v_target = x1 - z_T
v_hat = flow_head(h, x_t, t)
z_hat = x_t - t * v_hat
a_S = actor(o, z_hat)
a_T = actor(o, z_T)

loss = mse(v_hat, v_target) + bc(a_S, a_T)
```

### 16.4 Action Chunk

```python
obs_dict = env.reset()
while train:
    o, h, p, c = normalize(obs_dict)
    z_T = teacher_latent(o, p, c)
    a_T = actor(o, z_T)
    append_window(o, h, a_T)

    if window_ready:
        cond_o, cond_h, A_T = get_window()
        x0 = flatten(A_T)
        t = randint(0, T)
        eps = randn_like(x0)
        x_t = q_sample(x0, t, eps)
        eps_hat = chunk_head(cond_o, cond_h, x_t, t)
        A_hat = tanh(predict_x0(x_t, t, eps_hat)).reshape(K, action_dim)
        loss = mse(eps_hat, eps) + first_bc(A_hat, A_T) + chunk_bc(A_hat, A_T)
        update(chunk_head)

    A_sample = sample_action_chunk(o, h)
    a_S = A_sample[0]
    obs_dict = env.step(mix_teacher(a_T, a_S))
```

## 17. 当前代码里最需要注意的实现细节

1. 四种算法都不是 RL loss，reward 只用于 checkpoint selection。
2. 前三种 latent 算法的 `actor_mlp/mu` 是 frozen teacher actor head。
3. `z_T` 是通过 `model._actor_critic(input_dict)` 得到的 teacher latent target。
4. `a_T` 不是 teacher checkpoint 里单独采样出来的 action，而是用 `z_T` 重新走同一个 actor head 得到的 clamped mean action。
5. `sa_mean_std` 对 student 很关键；缺失或不匹配会让 `proprio_hist` 分布错位。
6. Deterministic eval 通常把初始 latent/action chunk 设为 0，而不是随机噪声。
7. `stochastic_infer=True` 才会在推理时用随机初始噪声和/或 stochastic reverse step。
8. Action-chunk diffusion 的 `model_best` 未必是部署最佳，部署更应该看 `model_best_student` / `model_best_student_reward` / `model_best_deploy_probe`。
9. Residual latent diffusion 是 `DiffusionLatentStudent` 的模式，不是单独算法类。
10. Consistency/Flow 的训练-推理一致性问题比 DDPM latent 更显著，因此有 `train_align_infer`、EMA、rollout BC 等补丁。

## 18. 总结

四种 diffusion 的共同目标都是让 student 在只使用本体历史的情况下接近 PPO teacher。差异可以一句话概括：

```text
DiffusionLatentStudent:
  用 DDPM 在 teacher latent 空间生成 z。

ConsistencyLatentStudent:
  学一个从任意噪声等级直接回到 clean latent 的一致性映射。

FlowMatchingLatentStudent:
  学从噪声到 teacher latent 的连续速度场，再用 ODE 积分生成 z。

DiffusionActionChunkStudent:
  不生成 latent，直接用 DDPM 生成未来 K 步 teacher action chunk，并执行第一步。
```

如果论文或汇报要讲“方法演进”，可以按这个逻辑叙述：

```text
PADAPT deterministic latent regression
  -> latent DDPM 提高 latent 分布表达能力
  -> consistency / flow 尝试降低推理步数与改善生成路径
  -> action chunk diffusion 尝试直接建模短时动作时序
```
