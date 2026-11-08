"""
DOT-PG: Dual Optimal Transport Policy Gradient - 网络架构
基于论文 "Bridging the Reality Gap: Dual Optimal Transport Policy Gradient"

适配dexscrew项目的网络架构实现

核心组件:
- PolicyNetwork (Actor): 确定性策略网络 πθ(s)
- DualNetwork (Critic): OT对偶变量网络 fφ(s,a)，带谱归一化
- QNetwork: 长期价值估计网络 Qψ(s,a)

论文对应:
- Section 4.2.1: Three-Component Architecture
- Section 9: Network Architecture (核心笔记)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PolicyNetwork(nn.Module):
    """
    策略网络 (Actor) πθ(a|s)
    
    论文对应: 
    - Section 4.2.1 - Policy Network (Actor) πθ(a|s)
    - 核心笔记 Section 7.1 - 策略网络
    
    架构特点:
    - 输入: 状态 s ∈ R^d_s (包含obs和extrin)
    - 主干: MLP + Layer Normalization
    - 残差连接: 改善梯度流
    - 输出: 确定性动作 μθ(s)，通过tanh限制动作范围 [-1, 1]
    - 初始化: 正交初始化（orthogonal initialization）
    
    公式:
    πθ(s) = tanh(MLP(s))  # 动作范围限制在[-1, 1]
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化策略网络
        
        Args:
            state_dim: 状态维度（obs_dim + extrin_dim）
            action_dim: 动作维度（numActions = 12）
            hidden_dim: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        
        # 主干网络 - 论文建议使用Layer Normalization
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 正交初始化 - 论文建议
        self._init_weights()
    
    def _init_weights(self):
        """
        正交初始化
        
        论文建议: 使用正交初始化提高训练稳定性
        - 隐藏层: gain=√2
        - 输出层: 小初始化（uniform[-3e-3, 3e-3]）
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        
        # 输出层使用小初始化 - 确保初始动作接近0
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, state):
        """
        前向传播
        
        公式: πθ(s) = tanh(MLP(s))
        
        Args:
            state: 状态 s，shape (batch_size, state_dim)
        
        Returns:
            action: 动作 a = πθ(s)，shape (batch_size, action_dim)
                   范围限制在 [-1, 1]
        """
        x = F.relu(self.ln1(self.fc1(state)))
        x = F.relu(self.ln2(self.fc2(x)))
        
        # tanh输出边界限制: a ∈ [-1, 1]
        # 论文公式: πθ(s) = tanh(MLP(s))
        return torch.tanh(self.fc3(x))


class DualNetwork(nn.Module):
    """
    对偶网络 (Critic) fφ(s, a)
    
    论文对应: 
    - Section 4.2.1 - Dual Network (Critic) fφ(s,a)
    - Section 3.2 - Kantorovich Duality
    - 核心笔记 Section 7.2 - 对偶网络
    
    架构特点:
    - 输入: 拼接的状态-动作向量 (s, a) ∈ R^(d_s + d_a)
    - 主干: MLP + 谱归一化 (Spectral Normalization)
    - 输出: 标量值 fφ(s,a) ∈ R
    - 约束: 梯度惩罚强制1-Lipschitz条件
    
    Lipschitz约束实现:
    1. 谱归一化: 强制每层权重矩阵的最大奇异值≤1
    2. 梯度惩罚: L_GP = E[(||∇fφ||_2 - 1)²]
    
    公式（Kantorovich对偶，论文公式21）:
    W(ρπ, ρE) = max_{f ∈ Lip_1} [E_ρE[f(s,a)] - E_ρπ[f(s,a)]]
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化对偶网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(DualNetwork, self).__init__()
        
        # 使用谱归一化强制Lipschitz约束
        # 谱归一化确保权重矩阵的最大奇异值≤1
        # 论文Section 4.2.1: Spectral normalization for Lipschitz enforcement
        self.fc1 = nn.utils.spectral_norm(nn.Linear(state_dim + action_dim, hidden_dim))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(hidden_dim, hidden_dim))
        self.fc3 = nn.utils.spectral_norm(nn.Linear(hidden_dim, 1))
        
        # 正交初始化（gain=1.0 for Lipschitz networks）
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化（gain=1.0 for Lipschitz networks）"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        """
        前向传播
        
        公式: fφ(s,a) = MLP_spectral_norm([s, a])
        
        Args:
            state: 状态 s，shape (batch_size, state_dim)
            action: 动作 a，shape (batch_size, action_dim)
        
        Returns:
            dual_value: 对偶函数值 fφ(s,a)，shape (batch_size, 1)
        """
        # 拼接状态和动作
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class QNetwork(nn.Module):
    """
    Q网络 Qψ(s, a)
    
    论文对应: 
    - Section 4.2.1 - Q-Network Qψ(s,a)
    - Section 4.2.2 - Stage 2 (Value Learning)
    - 核心笔记 Section 7.3 - Q网络
    
    架构特点:
    - 输入: 拼接的状态-动作向量 (s, a)
    - 主干: MLP
    - 输出: 标量Q值 Qψ(s,a) ∈ R
    
    更新公式（论文公式28）:
    Qψ(s,a) ← fφ(s,a) + γ * Q_target(s', πθ(s'))
    
    其中 fφ(s,a) 作为即时奖励（从对偶函数自动提取）
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        初始化Q网络
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
        """
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # 正交初始化
        self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
    
    def forward(self, state, action):
        """
        前向传播
        
        公式: Qψ(s,a) = MLP([s, a])
        
        Args:
            state: 状态 s，shape (batch_size, state_dim)
            action: 动作 a，shape (batch_size, action_dim)
        
        Returns:
            q_value: Q值 Qψ(s,a)，shape (batch_size, 1)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DOTPGNetworks(nn.Module):
    """
    DOTPG完整网络架构封装
    
    包含:
    - Policy Network (Actor)
    - Dual Network (Critic)
    - Q Network
    - Target Q Network
    
    适配dexscrew项目的特殊需求:
    - 支持从ActorCritic模型加载预训练权重
    - 支持extrin预测（学生latent）
    """
    def __init__(self, obs_dim, action_dim, extrin_dim, hidden_dim=256, device='cuda'):
        """
        初始化DOTPG网络架构
        
        Args:
            obs_dim: 原始观察维度
            action_dim: 动作维度
            extrin_dim: extrin（latent）维度
            hidden_dim: 隐藏层维度
            device: 设备
        """
        super(DOTPGNetworks, self).__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.extrin_dim = extrin_dim
        self.state_dim = obs_dim + extrin_dim  # 完整状态维度
        self.device = device
        
        # 初始化网络
        self.policy = PolicyNetwork(self.state_dim, action_dim, hidden_dim)
        self.dual = DualNetwork(self.state_dim, action_dim, hidden_dim)
        self.q_network = QNetwork(self.state_dim, action_dim, hidden_dim)
        self.q_target = QNetwork(self.state_dim, action_dim, hidden_dim)
        
        # 复制Q网络参数到目标网络
        self.q_target.load_state_dict(self.q_network.state_dict())
        
        # 移动到设备
        self.to(device)
    
    def get_action(self, state, noise=0.0):
        """
        获取动作（带探索噪声）
        
        Args:
            state: 状态 (batch_size, state_dim)
            noise: 探索噪声标准差
        
        Returns:
            action: 动作 (batch_size, action_dim)
        """
        with torch.no_grad():
            action = self.policy(state)
            
            if noise > 0:
                noise_vec = torch.randn_like(action) * noise
                action = (action + noise_vec).clamp(-1.0, 1.0)
        
        return action
    
    def soft_update_target(self, tau):
        """
        软更新目标网络
        
        论文公式（核心笔记 Section 10.5）:
        ψ_target ← τ·ψ + (1-τ)·ψ_target
        
        Args:
            tau: 软更新率（通常τ=0.005）
        """
        for param, target_param in zip(self.q_network.parameters(), 
                                       self.q_target.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
