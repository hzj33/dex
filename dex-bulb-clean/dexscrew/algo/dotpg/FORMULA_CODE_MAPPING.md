# DOTPG 公式-代码对应关系

本文档详细记录了DOTPG论文中的每条公式与代码实现的对应关系。

## 1. 核心理论公式

### 1.1 Kantorovich对偶（论文公式21）

**公式**:
```
min_θ max_{f∈Lip_1} [E_{(s,a)~ρ_E}[f(s,a)] - E_{(s,a)~ρ_π_θ}[f(s,a)]]
```

**代码对应** (`dotpg.py` - `update_dual_network`):
```python
# E_ρE[fφ(s,a)]
expert_value = self.dual(expert_states, expert_actions)
# E_ρπ[fφ(s,a)]
policy_value = self.dual(policy_states, policy_actions)

# Wasserstein距离估计: W(ρπ, ρE) ≈ E_ρE[f] - E_ρπ[f]
wasserstein_dist = expert_value.mean() - policy_value.mean()
```

---

### 1.2 Stage 1 - Critic更新（论文公式27）

**公式**:
```
max_φ [E_ρE[f_φ] - E_ρπ[f_φ] - λ·E[(||∇f_φ(x̂)||_2 - 1)²]]
```

**代码对应** (`dotpg.py` - `update_dual_network`):
```python
# 对偶网络损失（最小化负的Wasserstein距离 + 梯度惩罚）
# 等价于最大化 Wasserstein距离 - λ·L_GP
# L_dual = -W + λ·L_GP
dual_loss = -wasserstein_dist + self.config.lambda_gp * gradient_penalty
```

---

### 1.3 梯度惩罚（论文公式27的GP项）

**公式**:
```
L_GP = E[(||∇_{(ŝ,â)}f_φ(ŝ,â)||_2 - 1)²]
其中: (ŝ,â) = ε(s_E,a_E) + (1-ε)(s_π,a_π), ε ~ U[0,1]
```

**代码对应** (`dotpg.py` - `compute_gradient_penalty`):
```python
# 随机插值系数 ε ~ U[0,1]
epsilon = torch.rand(batch_size, 1, device=self.device)

# 计算插值点 - 论文公式: (ŝ,â) = ε(s_E,a_E) + (1-ε)(s_π,a_π)
# ŝ = ε*s_E + (1-ε)*s_π
interp_states = epsilon * expert_states[:batch_size] + \
               (1 - epsilon) * policy_states[:batch_size]
# â = ε*a_E + (1-ε)*a_π
interp_actions = epsilon * expert_actions[:batch_size] + \
                (1 - epsilon) * policy_actions[:batch_size]

# 计算梯度 ∇fφ(ŝ,â)
gradients = torch.autograd.grad(
    outputs=dual_output,
    inputs=[interp_states, interp_actions],
    grad_outputs=torch.ones_like(dual_output),
    create_graph=True,
    retain_graph=True
)

# 计算梯度范数 ||∇fφ||_2
gradient_norm = gradients.norm(2, dim=1)

# 梯度惩罚: (||∇fφ|| - 1)² - 强制1-Lipschitz
gradient_penalty = ((gradient_norm - 1) ** 2).mean()
```

---

### 1.4 Stage 2 - Q网络更新（论文公式28）

**公式**:
```
Q_ψ(s,a) ← f_φ(s,a) + γ·E_{s'~P}[Q_{ψ_target}(s', π_θ(s'))]
```

**代码对应** (`dotpg.py` - `update_q_network`):
```python
# 计算下一状态的动作: a' = πθ(s')
next_actions = self.policy(next_states)

# 计算目标Q值: Q_target(s', a')
target_q = self.q_target(next_states, next_actions)

# 使用对偶网络输出作为即时奖励: r = fφ(s,a)
# 这是DOTPG的核心创新 - 从OT对偶变量自动提取奖励
dual_reward = self.dual(states, actions)

# TD目标: y = r + γ·(1-done)·Q_target(s', a')
# 论文公式28
y = dual_reward + self.config.gamma * (1 - dones) * target_q

# Q网络损失: L_Q = MSE(Qψ(s,a), y)
q_loss = F.mse_loss(current_q, y)
```

---

### 1.5 Stage 3 - 策略更新（论文公式29）

**公式**:
```
min_θ E_{s~ρ_π}[Q_ψ(s, π_θ(s))]
```

**代码对应** (`dotpg.py` - `update_policy`):
```python
# 生成新动作: a = πθ(s)
new_actions = self.policy(states)

# 策略损失: L_π = -E[Qψ(s, πθ(s))]
# 负号表示最大化Q值（最小化Wasserstein距离）
policy_loss = -self.q_network(states, new_actions).mean()
```

---

### 1.6 目标网络软更新

**公式**:
```
ψ_target ← τ·ψ + (1-τ)·ψ_target
```

**代码对应** (`dotpg.py` - `update_target_network`):
```python
for param, target_param in zip(self.q_network.parameters(), 
                               self.q_target.parameters()):
    target_param.data.copy_(
        self.config.tau * param.data + 
        (1 - self.config.tau) * target_param.data
    )
```

---

## 2. 网络架构

### 2.1 策略网络 πθ(s)

**论文描述** (Section 4.2.1):
- 输入: 状态 s
- 架构: MLP + Layer Normalization + 残差连接
- 输出: 确定性动作 μθ(s)，通过tanh限制范围

**代码对应** (`networks.py` - `PolicyNetwork`):
```python
class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, state):
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        # tanh输出边界限制: a ∈ [-1, 1]
        return torch.tanh(self.fc3(x))
```

---

### 2.2 对偶网络 fφ(s,a)

**论文描述** (Section 4.2.1):
- 输入: 拼接的状态-动作向量 (s, a)
- 架构: MLP + 谱归一化 (Spectral Normalization)
- 约束: 梯度惩罚强制1-Lipschitz条件
- 输出: 标量值

**代码对应** (`networks.py` - `DualNetwork`):
```python
class DualNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        # 使用谱归一化强制Lipschitz约束
        self.fc1 = nn.utils.spectral_norm(nn.Linear(state_dim + action_dim, hidden_dim))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

---

### 2.3 Q网络 Qψ(s,a)

**论文描述** (Section 4.2.1):
- 输入: 拼接的状态-动作向量 (s, a)
- 架构: MLP
- 输出: 标量Q值

**代码对应** (`networks.py` - `QNetwork`):
```python
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

---

## 3. 训练流程

### 3.1 Algorithm 1: DOT-PG

**论文Algorithm 1伪代码**:
```
1. 初始化网络 πθ, fφ, Qψ, Q_target
2. 收集专家数据 D_E
3. for each iteration:
   a. 使用πθ与环境交互，收集数据到B
   b. for j = 1 to J:  # 对偶网络更新J次
      - 采样专家批次 B_E ~ D_E
      - 采样策略批次 B_π ~ B
      - 更新对偶网络（公式27）
   c. 更新Q网络（公式28）
   d. 更新策略网络（公式29）
   e. 软更新目标网络
```

**代码对应** (`dotpg.py` - `train_step`):
```python
def train_step(self):
    # Phase 1: 数据采样
    policy_states, policy_actions, next_states, dones = \
        self.replay_buffer.sample(self.config.batch_size)
    expert_states, expert_actions = \
        self.expert_buffer.sample(self.config.batch_size)
    
    # Phase 2: 对偶网络更新（J次）
    for _ in range(self.config.dual_updates_per_iter):
        dual_loss, wasserstein_dist, gradient_penalty = self.update_dual_network(
            expert_states, expert_actions, policy_states, policy_actions
        )
    
    # Phase 3: Q网络更新
    q_loss = self.update_q_network(policy_states, policy_actions, next_states, dones)
    
    # Phase 4: 策略更新（延迟更新）
    if self.total_it % self.config.policy_delay == 0:
        policy_loss_value = self.update_policy(policy_states)
        
        # Phase 5: 目标网络软更新
        self.update_target_network()
```

---

## 4. 超参数对应

| 参数 | 论文/核心笔记 | 代码变量 | 默认值 |
|------|--------------|----------|--------|
| γ (折扣因子) | Table 1 | `config.gamma` | 0.99 |
| λ (梯度惩罚系数) | Section 4.2.2 | `config.lambda_gp` | 10.0 |
| τ (软更新率) | Section 4.3 | `config.tau` | 0.005 |
| αφ (对偶网络学习率) | 定理9 | `config.lr_dual` | 3e-4 |
| αψ (Q网络学习率) | 定理9 | `config.lr_q` | 3e-4 |
| αθ (策略学习率) | 定理9 | `config.lr_policy` | 1e-4 |
| J (对偶更新次数) | Algorithm 1 | `config.dual_updates_per_iter` | 5 |
| 批大小 | Table 1 | `config.batch_size` | 256 |
| 隐藏层维度 | Section 9.2 | `config.hidden_dim` | 256 |

---

## 5. dexscrew适配

### 5.1 状态空间适配

**原始dexscrew状态**:
- `obs`: 基础观察 (96,)
- `priv_info`: 特权信息
- `proprio_hist`: 本体感知历史 (30, 24)
- `point_cloud_info`: 点云信息

**DOTPG状态构建**:
```python
# 状态 = normalized_obs + extrin
# extrin = adapt_tconv(proprio_hist) 或 env_mlp(priv_info)
state = torch.cat([obs, extrin], dim=-1)
```

### 5.2 动作空间适配

**dexscrew动作空间**: 12维连续动作（手部关节）

**DOTPG适配**:
```python
# 策略输出通过tanh限制在[-1, 1]
action = torch.tanh(self.fc3(x))
# 环境执行时直接使用
obs_dict, r, done, info = self.env.step(action)
```

### 5.3 专家数据收集

**使用教师模型（PPO）生成专家数据**:
```python
# 使用教师模型的actor获取动作
teacher_obs_input = torch.cat([obs, extrin_gt], dim=-1)
teacher_x = self.teacher_model.actor_mlp(teacher_obs_input)
teacher_action = self.teacher_model.mu(teacher_x)
teacher_action = torch.clamp(teacher_action, -1.0, 1.0)

# 存储专家数据
self.expert_buffer.add_batch(state, teacher_action)
```
