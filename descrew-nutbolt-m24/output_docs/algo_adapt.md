# ProprioAdapt 学生蒸馏算法说明

本文总结本仓库里 `ProprioAdapt` / PADAPT 学生蒸馏的技术路线、数学形式和代码入口。这里的 PADAPT 不是重新训练一个完整策略，而是在已经训练好的 PPO teacher 上，额外学习一个从本体历史 `proprio_hist` 预测 teacher latent 的适配模块 `adapt_tconv`。

## 1. 总体路线

训练分两阶段：

1. 训练 PPO teacher。
   - 入口：`train.py`
   - 算法类：`dexscrew/algo/ppo/ppo.py`
   - 配置：`configs/train/Dexh13HoraLightbulb.yaml`
   - 输出：`outputs/.../stage1_nn/*.pth`

2. 用 PPO teacher 蒸馏 ProprioAdapt student。
   - 入口：`train.py`
   - 算法类：`dexscrew/algo/ppo/padapt.py`
   - 网络：`dexscrew/algo/models/models.py`
   - 适配器：`dexscrew/algo/models/block.py`
   - 输出：`outputs/.../stage2_nn/model_best.ckpt`

蒸馏阶段的核心思想是：

```text
teacher: obs + priv_info + point_cloud -> teacher latent -> teacher action
student: obs + proprio_hist            -> student latent -> student action
```

其中 teacher 使用仿真里的特权信息，student 只依赖真机可获得的本体历史。训练时仍在仿真中拿到 `priv_info` 和 `point_cloud_info`，用于给 student 提供监督信号；测试和部署时不需要 `priv_info`。

## 2. 关键张量

环境在 `dexscrew/tasks/xhand_hora.py` 中构造 observation。

对 DexH13，`numActions=16`，单帧本体信息是：

```text
frame_t = concat(q_t, target_t)  # shape = 32
```

其中：

- `q_t`：当前手部关节角，16 维。
- `target_t`：当前目标关节角，16 维。

环境保存历史窗口：

```text
obs_t          = last 3 frames flattened   # shape = 3 * 32 = 96
proprio_hist_t = last 30 frames            # shape = 30 * 32
```

对应代码：

- `obs_buf_lag_history`：`dexscrew/tasks/xhand_hora.py`
- `obs_buf`：最后 3 帧 flatten
- `proprio_hist_buf`：最后 `propHistoryLen=30` 帧

相关代码片段：

```python
t_buf = self.obs_buf_lag_history[:, -3:, :self.obs_buf.shape[1] // 3].reshape(self.num_envs, -1)
self.obs_buf[:, :t_buf.shape[1]] = t_buf

cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)
cur_tar_buf = cur_targets_obs[:, None, :self.num_actions]
cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)

self.proprio_hist_buf[:] = self.obs_buf_lag_history[:, -self.prop_hist_len:, :self.numActions * 2]
```

## 3. 网络结构

网络类是 `ActorCritic`：

```text
dexscrew/algo/models/models.py
```

主要模块：

```text
env_mlp      : priv_info -> privileged latent
point_mlp    : point cloud point-wise MLP -> max pooling -> visual latent
adapt_tconv  : proprio_hist -> student latent
actor_mlp    : concat(obs, latent) -> policy feature
mu           : policy feature -> action mean
value        : policy feature -> value
```

### 3.1 Teacher 模式

PPO teacher 使用：

```yaml
train.ppo.proprio_adapt: False
train.ppo.priv_info: True
train.ppo.use_point_cloud_info: True
```

代码路径：

```python
extrin = self.env_mlp(obs_dict["priv_info"])
pcs = self.point_mlp(obs_dict["point_cloud_info"])
pcs = torch.max(pcs, 1)[0]
extrin = torch.cat([extrin, pcs], dim=-1)
extrin = torch.tanh(extrin)
obs_input = torch.cat([obs, extrin], dim=-1)
mu = self.mu(self.actor_mlp(obs_input))
```

符号化写成：

```text
z_priv = f_env(p_t)
z_pc   = max_i f_pc(c_{t,i})
z_T    = tanh(concat(z_priv, z_pc))
mu_T   = pi_theta(concat(o_t, z_T))
```

其中：

- `o_t` 是归一化后的 `obs`。
- `p_t` 是归一化后的 `priv_info`。
- `c_t` 是归一化后的 point cloud。
- `mu_T` 是 teacher action mean。

### 3.2 Student 模式

PADAPT student 使用：

```yaml
train.algo: ProprioAdapt
train.ppo.proprio_adapt: True
```

此时 `ActorCritic` 会额外创建：

```python
self.adapt_tconv = TemporalConv(temporal_fusing_input_dim, temporal_fusing_output_dim)
```

对 DexH13 + point cloud：

```text
temporal_fusing_input_dim  = proprio_dim = 32
temporal_fusing_output_dim = 8 + 32 = 40
```

`TemporalConv` 定义在 `dexscrew/algo/models/block.py`：

```python
self.channel_transform = Linear(input_dim, 32) -> ReLU -> Linear(32, 32) -> ReLU
self.temporal_aggregation = Conv1d(32, 32, 9, stride=2) -> ReLU
                          -> Conv1d(32, 32, 5) -> ReLU
                          -> Conv1d(32, 32, 5) -> ReLU
self.low_dim_proj = Linear(32 * 3, output_dim)
```

对 `T=30` 的历史，三层卷积后的时间长度变为 3，所以最终展平为 `32 * 3`。

Student forward：

```python
extrin = self.adapt_tconv(obs_dict["proprio_hist"])
extrin = torch.tanh(extrin)
obs_input = torch.cat([obs, extrin], dim=-1)
mu = self.mu(self.actor_mlp(obs_input))
```

符号化写成：

```text
z_S  = tanh(g_phi(h_t))
mu_S = pi_theta(concat(o_t, z_S))
```

其中：

- `h_t` 是归一化后的 `proprio_hist`。
- `g_phi` 是 `adapt_tconv`。
- `theta` 是从 PPO teacher 继承来的 actor 参数。
- PADAPT 训练时只更新 `phi`。

## 4. 蒸馏训练过程

蒸馏入口是 `ProprioAdapt.train()`：

```text
dexscrew/algo/ppo/padapt.py
```

训练开始时：

```python
agent = ProprioAdapt(...)
agent.restore_train(config.train.load_path)
agent.train()
```

`config.train.load_path` 默认指向顶层 `checkpoint`，student 脚本里传入的是 PPO teacher checkpoint：

```bash
bash scripts/dexh13_lightbulb_student_padapt.sh 0 42 <cache> <TEACHER_CKPT>
```

### 4.1 加载 teacher 并冻结大部分参数

`ProprioAdapt.__init__()` 中只允许 `adapt_tconv` 训练：

```python
adapt_params = []
for name, p in self.model.named_parameters():
    if "adapt_tconv" in name:
        adapt_params.append(p)
    else:
        p.requires_grad = False
self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
```

`restore_train()` 用非 strict 方式加载 PPO teacher：

```python
self.model.load_state_dict(checkpoint["model"], strict=False)
self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])
```

因为 PPO teacher checkpoint 没有 `adapt_tconv`，所以：

- `actor_mlp/mu/value/env_mlp/point_mlp` 从 teacher 继承。
- `adapt_tconv` 随机初始化。
- 除 `adapt_tconv` 外，其余参数冻结。

### 4.2 在线 DAgger 风格监督

PADAPT 不是离线读取 expert buffer，而是在 student 当前 rollout 的状态分布上持续向 teacher 查询监督信号：

```python
obs_dict = self.env.reset()
while self.agent_steps <= 1e9:
    input_dict = {
        "obs": self.running_mean_std(obs_dict["obs"]).detach(),
        "priv_info": self.priv_mean_std(obs_dict["priv_info"]),
        "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
        "point_cloud_info": normalized_point_cloud,
    }
    mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
    ...
    obs_dict, r, done, info = self.env.step(clamp(mu, -1, 1))
```

这就是 DAgger 风格的关键：环境由 student action 推进，teacher 在这些 student 访问到的状态上提供 latent/action 标签。

## 5. 损失函数

记：

```text
o_t  = normalized obs
h_t  = normalized proprio_hist
p_t  = normalized priv_info
C_t  = normalized point_cloud_info
z_S  = student latent from adapt_tconv
z_T  = teacher latent target
mu_S = student action
mu_T = reconstructed teacher action
```

### 5.1 Student latent

```text
z_S = tanh(g_phi(h_t))
```

代码：

```python
extrin = self.adapt_tconv(obs_dict["proprio_hist"])
extrin = torch.tanh(extrin)
```

### 5.2 Teacher latent target

当前代码在 `proprio_adapt=True` 时这样构造 teacher target：

```python
extrin_gt = self.env_mlp(obs_dict["priv_info"])
extrin_gt = torch.tanh(extrin_gt)

pcs = self.point_mlp(obs_dict["point_cloud_info"])
pcs = torch.max(pcs, 1)[0]
extrin_gt = torch.cat([extrin_gt, pcs], dim=-1)
```

符号化写成：

```text
z_priv = tanh(f_env(p_t))
z_pc   = max_i f_pc(C_{t,i})
z_T    = concat(z_priv, z_pc)
```

注意：这里严格按当前代码描述。Teacher PPO 模式里是对 `concat(env_mlp(priv), point_mlp(point_cloud))` 整体再 `tanh`，而 PADAPT target 里只对 `env_mlp(priv)` 部分显式 `tanh`，point cloud 部分直接拼接 `pcs`。这是一个实现细节，做实验对齐时要注意。

### 5.3 Latent loss

```python
latent_loss = ((e - e_gt.detach()) ** 2).mean()
```

公式：

```text
L_latent = mean(||z_S - stopgrad(z_T)||_2^2)
```

这里的 `mean` 会对 batch 和 latent 维度一起平均。

### 5.4 Action-level behavior cloning loss

代码中先用 teacher latent 重新走同一个 actor head，构造 teacher action：

```python
teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
with torch.no_grad():
    teacher_x = self.model.actor_mlp(teacher_obs_input)
    teacher_mu = self.model.mu(teacher_x)
```

然后计算 action BC：

```python
bc_loss = torch.sum(
    (torch.clamp(mu, -1, 1) - torch.clamp(teacher_mu, -1, 1)).pow(2),
    dim=-1,
).mean()
```

公式：

```text
mu_T = pi_theta(concat(o_t, stopgrad(z_T)))
L_BC = mean_batch sum_a ||clip(mu_S)_a - clip(mu_T)_a||_2^2
```

### 5.5 总损失

当前实现没有额外权重：

```python
loss = bc_loss + latent_loss
```

公式：

```text
L_PADAPT = L_BC + L_latent
```

然后只更新 `adapt_tconv`：

```python
self.optim.zero_grad()
loss.backward()
self.optim.step()
```

## 6. 与 PPO teacher 的关系

PADAPT student 不是从零学一个 actor。它继承 teacher 的大部分参数：

```text
env_mlp      loaded from teacher, frozen
point_mlp    loaded from teacher, frozen
actor_mlp    loaded from teacher, frozen
mu           loaded from teacher, frozen
value        loaded from teacher, frozen
adapt_tconv  new module, trainable
```

所以 student 的学习目标可以理解为：

```text
让 adapt_tconv(proprio_hist) 产生一个 actor 能理解的 latent，
使 frozen teacher actor 在没有 priv_info 的情况下仍输出接近 teacher 的 action。
```

这也是为什么 student ckpt 里需要同时保存：

```text
model
running_mean_std
sa_mean_std
priv_mean_std
point_cloud_mean_std
```

测试部署时虽然不使用 `priv_info`，但 checkpoint 仍保留 `priv_mean_std`，因为它是训练/复现 teacher latent 所需的统计。

## 7. 归一化

归一化模块是 `RunningMeanStd`：

```text
dexscrew/algo/models/running_mean_std.py
```

默认形式：

```text
norm(x) = clamp((x - mean) / sqrt(var + eps), -5, 5)
```

PADAPT 中有几类统计：

```text
running_mean_std      : obs
sa_mean_std           : proprio_hist
priv_mean_std         : priv_info
point_cloud_mean_std  : point_cloud_info
```

训练时：

- `running_mean_std` 从 teacher checkpoint 加载并保持 eval。
- `priv_mean_std` 从 teacher checkpoint 加载并保持 eval。
- `point_cloud_mean_std` 从 teacher checkpoint 加载并保持 eval。
- `sa_mean_std` 新建并处于 train，会随着 student rollout 的 `proprio_hist` 更新。

测试时：

```python
self.running_mean_std.eval()
self.sa_mean_std.eval()
self.point_cloud_mean_std.eval()
```

部署时必须使用 student ckpt 或 TorchScript 中保存的统计来归一化输入。

## 8. Action 语义

PADAPT 输出的 `mu` 不是绝对关节角，也不是直接 torque。环境会先 clamp 到 `[-1, 1]`，再解释为目标关节角增量：

```python
targets = self.prev_targets + self.action_scale * actions_for_hand
self.cur_targets = tensor_clamp(targets, lower, upper)
self.prev_targets[:] = self.cur_targets
```

对当前 DexH13 lightbulb 配置：

```yaml
controller:
  controlFrequencyInv: 10
  action_scale: 0.05
sim:
  dt: 0.005
```

所以 policy 每 `10 * 0.005 = 0.05s` 更新一次，即 20Hz。单步最大目标角增量：

```text
0.05 rad
```

底层 torque control 里再用 PD 跟踪目标：

```python
torques = p_gain * (noise_action - dof_pos) - d_gain * dof_vel
torques = torch.clip(torques, -torque_limit, torque_limit)
```

因此真机部署时应复刻：

```text
mu = clamp(policy(obs, proprio_hist), -1, 1)
target_q = prev_target + action_scale * mu
target_q = clip(target_q, joint_lower, joint_upper)
```

并且 policy target 更新频率应与训练一致。

## 9. 测试和导出

### 9.1 Repo 内测试

测试走 `ProprioAdapt.test()`：

```python
input_dict = {
    "obs": self.running_mean_std(obs_dict["obs"]),
    "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
    "point_cloud_info": point_cloud_info,
}
mu, extrin, extrin_gt = self.model.act_inference(input_dict)
obs_dict, r, done, info = self.env.step(clamp(mu, -1, 1))
```

可视化脚本示例：

```bash
bash scripts/vis_dexh13_lightbulb_student_padapt.sh 0 42 <STUDENT_CKPT>
```

### 9.2 TorchScript 导出

导出入口：

```text
student_eval.py
```

`PolicyWrapper.forward()` 只调用模型，不在 wrapper 内部做归一化：

```python
mu, sigma, value, extrin, extrin_gt = self.model._actor_critic(input_dict)
return mu, extrin, extrin_gt
```

导出前的 trace 输入已经被归一化：

```python
input_dict = {
    "obs": agent.running_mean_std(student_obs).cpu(),
    "proprio_hist": agent.sa_mean_std(obs_dict["proprio_hist"]).cpu(),
    "point_cloud_info": agent.point_cloud_mean_std(obs_dict["point_cloud_info"]).cpu(),
}
```

但 wrapper 会把统计量作为 buffers 保存：

```python
running_mean, running_var
sa_mean, sa_var
pc_mean, pc_var
```

所以外部部署脚本的正确做法是：

1. 从 TorchScript module 读取这些 buffers。
2. 在脚本中手动归一化 raw `obs/proprio_hist/point_cloud_info`。
3. 把归一化后的 input_dict 传给 `jit_model`。

这就是 `xhand-deploy/dexh13_deploy.py` 里的部署逻辑。

## 10. Checkpoint 类型

常见文件含义：

```text
*.pth  : PPO teacher checkpoint，来自 stage1_nn
*.ckpt : ProprioAdapt student checkpoint，来自 stage2_nn
*.pt   : TorchScript 部署模型，从 student ckpt 导出
```

例如 sim2real 包中：

```text
teacher_ppo_best_reward_3171.29.pth  # teacher
model_best.ckpt                      # ProprioAdapt student
student_policy.pt                    # TorchScript deploy artifact
```

## 11. 常见排查点

### 11.1 Student 行为和 teacher 差距大

优先检查：

- teacher checkpoint 是否是同一 task / 同一 hand / 同一 observation 设置。
- `train.ppo.proprio_dim` 是否等于 `numActions * 2`。
- `train.ppo.proprio_adapt=True` 是否打开。
- `use_point_cloud_info` 和 `normalize_point_cloud` 是否与 teacher 训练一致。
- `running_mean_std/priv_mean_std/point_cloud_mean_std` 是否成功从 teacher 加载。
- `sa_mean_std` 是否随 student 数据正确更新。

### 11.2 部署效果和仿真 student 不一致

优先检查：

- TorchScript 输入是否已经按 `running_mean/sa_mean/pc_mean` 归一化。
- `obs` 是否是最后 3 帧 `[q, target]` flatten。
- `proprio_hist` 是否是最后 30 帧 `[q, target]`。
- `target_q = prev_target + action_scale * mu` 是否按训练语义实现。
- action 更新频率是否与 `controlFrequencyInv * dt` 一致。
- joint order 是否与训练 URDF 顺序一致。
- action mask 是否与训练 task yaml 一致。
- 初始 `prev_target` 和 history 是否与训练 reset 姿态一致。

### 11.3 当前实现的一个细节

在 `proprio_adapt=True` 且 `use_point_cloud_info=True` 时，当前代码的 teacher latent target 是：

```text
concat(tanh(env_mlp(priv_info)), max(point_mlp(point_cloud)))
```

而 teacher PPO actor 在原模式下使用的是：

```text
tanh(concat(env_mlp(priv_info), max(point_mlp(point_cloud))))
```

如果后续要严格重构 PADAPT 或比较不同实现，这个差异需要特别标记。改动它会影响已训练 student 的可复现性，不能随手改。

## 12. 最小伪代码

```python
# Build student-shaped ActorCritic.
model = ActorCritic(proprio_adapt=True)

# Load PPO teacher weights, except new adapt_tconv.
model.load_state_dict(teacher_ckpt["model"], strict=False)

# Freeze all modules except adapt_tconv.
for name, p in model.named_parameters():
    p.requires_grad = "adapt_tconv" in name

obs_dict = env.reset()
while training:
    obs = rms_obs(obs_dict["obs"]).detach()
    hist = rms_hist(obs_dict["proprio_hist"].detach())
    priv = rms_priv(obs_dict["priv_info"])
    pc = rms_pc(obs_dict["point_cloud_info"])

    mu_s, _, _, z_s, z_t = model._actor_critic({
        "obs": obs,
        "proprio_hist": hist,
        "priv_info": priv,
        "point_cloud_info": pc,
    })

    with torch.no_grad():
        mu_t = model.mu(model.actor_mlp(torch.cat([obs, z_t], dim=-1)))

    loss_latent = ((z_s - z_t.detach()) ** 2).mean()
    loss_bc = ((mu_s.clamp(-1, 1) - mu_t.clamp(-1, 1)) ** 2).sum(-1).mean()
    loss = loss_latent + loss_bc

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    obs_dict, reward, done, info = env.step(mu_s.detach().clamp(-1, 1))
```

