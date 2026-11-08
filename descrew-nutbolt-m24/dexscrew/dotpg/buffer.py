"""
DOT-PG: Dual Optimal Transport Policy Gradient - 经验回放缓冲区
基于论文 "Bridging the Reality Gap: Dual Optimal Transport Policy Gradient"

适配dexscrew项目的缓冲区实现

核心组件:
- ReplayBuffer: 策略经验回放缓冲区，存储(s, a, s', done)
- ExpertBuffer: 专家数据缓冲区，存储(s_E, a_E)

论文对应:
- Algorithm 1: DOT-PG
- 核心笔记 Section 10.7 - 数据收集与存储
"""

import numpy as np
import torch
from typing import Optional, Tuple


class ReplayBuffer:
    """
    策略经验回放缓冲区
    
    存储策略与环境交互产生的转移 (s, a, s', done)
    
    论文对应:
    - Algorithm 1 Phase 1: 数据收集
    - 核心笔记 Section 10.7: 策略数据存储
    
    注意: 在DOTPG中，奖励r不需要存储，因为奖励由对偶网络fφ(s,a)自动提取
    """
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device='cuda'):
        """
        初始化经验回放缓冲区
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_size: 缓冲区最大容量
            device: 设备
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = device
        
        # 预分配GPU内存 - 提高采样效率
        self.states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, action_dim), dtype=torch.float32, device=device)
        self.next_states = torch.zeros((max_size, state_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
    
    def add(self, state, action, next_state, done):
        """
        添加一条经验
        
        Args:
            state: 当前状态 s
            action: 执行的动作 a
            next_state: 下一状态 s'
            done: 终止标志
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def add_batch(self, states, actions, next_states, dones):
        """
        批量添加经验（适配Isaac Gym的并行环境）
        
        Args:
            states: 当前状态批次 (batch_size, state_dim)
            actions: 动作批次 (batch_size, action_dim)
            next_states: 下一状态批次 (batch_size, state_dim)
            dones: 终止标志批次 (batch_size,) 或 (batch_size, 1)
        """
        batch_size = states.shape[0]
        
        # 处理dones的维度
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)
        
        # 计算存储位置
        if self.ptr + batch_size <= self.max_size:
            # 不需要循环
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
            self.dones[self.ptr:self.ptr + batch_size] = dones.float()
        else:
            # 需要循环存储
            first_part = self.max_size - self.ptr
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.next_states[self.ptr:] = next_states[:first_part]
            self.dones[self.ptr:] = dones[:first_part].float()
            
            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:].float()
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size):
        """
        随机采样一批经验
        
        论文对应: Algorithm 1 - 从B采样策略批次
        
        Args:
            batch_size: 批大小
        
        Returns:
            states, actions, next_states, dones (all as torch tensors on device)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        
        return (
            self.states[indices],
            self.actions[indices],
            self.next_states[indices],
            self.dones[indices]
        )
    
    def __len__(self):
        return self.size


class ExpertBuffer:
    """
    专家数据缓冲区
    
    存储专家演示 (s_E, a_E)
    
    论文对应:
    - Algorithm 1: 从D_E采样专家批次
    - 核心笔记 Section 10.7: 专家数据存储
    
    在dexscrew项目中，专家数据来自预训练的PPO教师模型
    """
    def __init__(
        self,
        state_dim,
        action_dim,
        max_size=int(1e6),
        device='cuda',
        storage_device=None,
        dtype=torch.float32,
        return_dtype=torch.float32,
    ):
        """
        初始化专家数据缓冲区
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            max_size: 缓冲区最大容量
            device: 采样返回的设备（通常是训练 device，比如 cuda:0）
            storage_device: 存储设备（可设为 cpu 以节省显存并支持更大 expert 数据集）
            dtype: 存储 dtype（默认 float32）
            return_dtype: 采样返回 dtype（默认 float32）
        """
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.dtype = dtype
        self.return_dtype = return_dtype
        self.meta = {}
        
        # 预分配GPU内存
        self.states = torch.zeros((max_size, state_dim), dtype=self.dtype, device=self.storage_device)
        self.actions = torch.zeros((max_size, action_dim), dtype=self.dtype, device=self.storage_device)
    
    def add(self, state, action):
        """
        添加一条专家经验
        
        Args:
            state: 专家状态 s_E
            action: 专家动作 a_E
        """
        if state.device != self.storage_device or state.dtype != self.dtype:
            state = state.to(device=self.storage_device, dtype=self.dtype)
        if action.device != self.storage_device or action.dtype != self.dtype:
            action = action.to(device=self.storage_device, dtype=self.dtype)

        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def add_batch(self, states, actions):
        """
        批量添加专家经验
        
        Args:
            states: 专家状态批次 (batch_size, state_dim)
            actions: 专家动作批次 (batch_size, action_dim)
        """
        if states.device != self.storage_device or states.dtype != self.dtype:
            states = states.to(device=self.storage_device, dtype=self.dtype)
        if actions.device != self.storage_device or actions.dtype != self.dtype:
            actions = actions.to(device=self.storage_device, dtype=self.dtype)

        batch_size = states.shape[0]
        
        if self.ptr + batch_size <= self.max_size:
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
        else:
            first_part = self.max_size - self.ptr
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            
            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
        
        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)
    
    def sample(self, batch_size):
        """
        随机采样一批专家经验
        
        论文对应: Algorithm 1 - 从D_E采样专家批次B_E
        
        Args:
            batch_size: 批大小
        
        Returns:
            states, actions (all as torch tensors on device)
        """
        indices = torch.randint(0, self.size, (batch_size,), device=self.storage_device)

        states = self.states[indices]
        actions = self.actions[indices]

        if states.device != self.device or states.dtype != self.return_dtype:
            states = states.to(device=self.device, dtype=self.return_dtype)
        if actions.device != self.device or actions.dtype != self.return_dtype:
            actions = actions.to(device=self.device, dtype=self.return_dtype)

        return states, actions
    
    def __len__(self):
        return self.size

    def clear(self):
        """清空缓冲区（保留已分配的张量以便复用内存）。"""
        self.ptr = 0
        self.size = 0
        self.meta = {}
    
    def save(self, filename, meta=None):
        """保存专家数据到文件"""
        payload = {
            'states': self.states[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'size': self.size,
        }
        if meta is not None:
            payload['meta'] = meta
        torch.save(payload, filename)
        print(f"专家数据已保存到 {filename} （共 {self.size} 条样本）")
    
    def load(self, filename):
        """从文件加载专家数据"""
        data = torch.load(filename, map_location='cpu')
        
        load_size = min(data['size'], self.max_size)
        self.states[:load_size] = data['states'][:load_size].to(self.storage_device, dtype=self.dtype)
        self.actions[:load_size] = data['actions'][:load_size].to(self.storage_device, dtype=self.dtype)
        
        self.size = load_size
        self.ptr = load_size % self.max_size
        self.meta = data.get('meta', {}) if isinstance(data, dict) else {}
        
        print(f"从 {filename} 加载了 {self.size} 条专家数据")


class RawReplayBuffer:
    """
    Raw 经验回放缓冲区（student-state 专用）

    存储策略与环境交互产生的原始观测转移：
    (obs, proprio_hist, a, next_obs, next_proprio_hist, done)

    目的：off-policy 训练中 encoder（adapt_tconv / sa_mean_std）会变化，
    若直接存 state 会导致 buffer 非平稳；改存原始输入并在采样时用“当前 encoder”动态重建 state。
    """

    def __init__(
        self,
        obs_dim: int,
        proprio_hist_shape: Tuple[int, int],
        action_dim: int,
        max_size: int = int(2e5),
        device: str = 'cuda',
        storage_device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        return_dtype: torch.dtype = torch.float32,
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.dtype = dtype
        self.return_dtype = return_dtype

        proprio_hist_shape = tuple(int(x) for x in proprio_hist_shape)

        self.obs = torch.zeros((self.max_size, int(obs_dim)), dtype=self.dtype, device=self.storage_device)
        self.proprio_hist = torch.zeros((self.max_size, *proprio_hist_shape), dtype=self.dtype, device=self.storage_device)
        self.actions = torch.zeros((self.max_size, int(action_dim)), dtype=self.dtype, device=self.storage_device)
        self.next_obs = torch.zeros((self.max_size, int(obs_dim)), dtype=self.dtype, device=self.storage_device)
        self.next_proprio_hist = torch.zeros((self.max_size, *proprio_hist_shape), dtype=self.dtype, device=self.storage_device)
        self.dones = torch.zeros((self.max_size, 1), dtype=torch.float32, device=self.storage_device)

    def add_batch(self, obs, proprio_hist, actions, next_obs, next_proprio_hist, dones):
        batch_size = int(obs.shape[0])
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)

        if obs.device != self.storage_device or obs.dtype != self.dtype:
            obs = obs.to(device=self.storage_device, dtype=self.dtype)
        if proprio_hist.device != self.storage_device or proprio_hist.dtype != self.dtype:
            proprio_hist = proprio_hist.to(device=self.storage_device, dtype=self.dtype)
        if actions.device != self.storage_device or actions.dtype != self.dtype:
            actions = actions.to(device=self.storage_device, dtype=self.dtype)
        if next_obs.device != self.storage_device or next_obs.dtype != self.dtype:
            next_obs = next_obs.to(device=self.storage_device, dtype=self.dtype)
        if next_proprio_hist.device != self.storage_device or next_proprio_hist.dtype != self.dtype:
            next_proprio_hist = next_proprio_hist.to(device=self.storage_device, dtype=self.dtype)
        if dones.device != self.storage_device:
            dones = dones.to(device=self.storage_device)

        if self.ptr + batch_size <= self.max_size:
            self.obs[self.ptr:self.ptr + batch_size] = obs
            self.proprio_hist[self.ptr:self.ptr + batch_size] = proprio_hist
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.next_obs[self.ptr:self.ptr + batch_size] = next_obs
            self.next_proprio_hist[self.ptr:self.ptr + batch_size] = next_proprio_hist
            self.dones[self.ptr:self.ptr + batch_size] = dones.float()
        else:
            first_part = self.max_size - self.ptr
            self.obs[self.ptr:] = obs[:first_part]
            self.proprio_hist[self.ptr:] = proprio_hist[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.next_obs[self.ptr:] = next_obs[:first_part]
            self.next_proprio_hist[self.ptr:] = next_proprio_hist[:first_part]
            self.dones[self.ptr:] = dones[:first_part].float()

            second_part = batch_size - first_part
            self.obs[:second_part] = obs[first_part:]
            self.proprio_hist[:second_part] = proprio_hist[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.next_obs[:second_part] = next_obs[first_part:]
            self.next_proprio_hist[:second_part] = next_proprio_hist[first_part:]
            self.dones[:second_part] = dones[first_part:].float()

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.storage_device)
        obs = self.obs[indices]
        proprio_hist = self.proprio_hist[indices]
        actions = self.actions[indices]
        next_obs = self.next_obs[indices]
        next_proprio_hist = self.next_proprio_hist[indices]
        dones = self.dones[indices]

        if obs.device != self.device or obs.dtype != self.return_dtype:
            obs = obs.to(device=self.device, dtype=self.return_dtype)
        if proprio_hist.device != self.device or proprio_hist.dtype != self.return_dtype:
            proprio_hist = proprio_hist.to(device=self.device, dtype=self.return_dtype)
        if actions.device != self.device or actions.dtype != self.return_dtype:
            actions = actions.to(device=self.device, dtype=self.return_dtype)
        if next_obs.device != self.device or next_obs.dtype != self.return_dtype:
            next_obs = next_obs.to(device=self.device, dtype=self.return_dtype)
        if next_proprio_hist.device != self.device or next_proprio_hist.dtype != self.return_dtype:
            next_proprio_hist = next_proprio_hist.to(device=self.device, dtype=self.return_dtype)
        if dones.device != self.device:
            dones = dones.to(device=self.device)

        return obs, proprio_hist, actions, next_obs, next_proprio_hist, dones

    def __len__(self):
        return self.size


class RawExpertBuffer:
    """
    Raw 专家数据缓冲区（student-state 专用）

    存储 teacher rollout 收集到的 (obs, proprio_hist, teacher_action)。
    采样时再用当前 encoder 动态重建 state，避免 expert buffer 与 encoder 绑定。
    """

    def __init__(
        self,
        obs_dim: int,
        proprio_hist_shape: Tuple[int, int],
        action_dim: int,
        max_size: int = int(2e6),
        device: str = 'cuda',
        storage_device: Optional[str] = None,
        dtype: torch.dtype = torch.float16,
        return_dtype: torch.dtype = torch.float32,
    ):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.dtype = dtype
        self.return_dtype = return_dtype
        self.meta = {}

        proprio_hist_shape = tuple(int(x) for x in proprio_hist_shape)

        self.obs = torch.zeros((self.max_size, int(obs_dim)), dtype=self.dtype, device=self.storage_device)
        self.proprio_hist = torch.zeros((self.max_size, *proprio_hist_shape), dtype=self.dtype, device=self.storage_device)
        self.actions = torch.zeros((self.max_size, int(action_dim)), dtype=self.dtype, device=self.storage_device)

    def add_batch(self, obs, proprio_hist, actions):
        if obs.device != self.storage_device or obs.dtype != self.dtype:
            obs = obs.to(device=self.storage_device, dtype=self.dtype)
        if proprio_hist.device != self.storage_device or proprio_hist.dtype != self.dtype:
            proprio_hist = proprio_hist.to(device=self.storage_device, dtype=self.dtype)
        if actions.device != self.storage_device or actions.dtype != self.dtype:
            actions = actions.to(device=self.storage_device, dtype=self.dtype)

        batch_size = int(obs.shape[0])
        if self.ptr + batch_size <= self.max_size:
            self.obs[self.ptr:self.ptr + batch_size] = obs
            self.proprio_hist[self.ptr:self.ptr + batch_size] = proprio_hist
            self.actions[self.ptr:self.ptr + batch_size] = actions
        else:
            first_part = self.max_size - self.ptr
            self.obs[self.ptr:] = obs[:first_part]
            self.proprio_hist[self.ptr:] = proprio_hist[:first_part]
            self.actions[self.ptr:] = actions[:first_part]

            second_part = batch_size - first_part
            self.obs[:second_part] = obs[first_part:]
            self.proprio_hist[:second_part] = proprio_hist[first_part:]
            self.actions[:second_part] = actions[first_part:]

        self.ptr = (self.ptr + batch_size) % self.max_size
        self.size = min(self.size + batch_size, self.max_size)

    def sample(self, batch_size: int):
        indices = torch.randint(0, self.size, (int(batch_size),), device=self.storage_device)
        obs = self.obs[indices]
        proprio_hist = self.proprio_hist[indices]
        actions = self.actions[indices]

        if obs.device != self.device or obs.dtype != self.return_dtype:
            obs = obs.to(device=self.device, dtype=self.return_dtype)
        if proprio_hist.device != self.device or proprio_hist.dtype != self.return_dtype:
            proprio_hist = proprio_hist.to(device=self.device, dtype=self.return_dtype)
        if actions.device != self.device or actions.dtype != self.return_dtype:
            actions = actions.to(device=self.device, dtype=self.return_dtype)

        return obs, proprio_hist, actions

    def __len__(self):
        return self.size

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.meta = {}

    def save(self, filename, meta=None):
        payload = {
            'obs': self.obs[:self.size].cpu(),
            'proprio_hist': self.proprio_hist[:self.size].cpu(),
            'actions': self.actions[:self.size].cpu(),
            'size': self.size,
        }
        if meta is not None:
            payload['meta'] = meta
        torch.save(payload, filename)
        print(f"专家数据已保存到 {filename} （共 {self.size} 条样本）")

    def load(self, filename):
        data = torch.load(filename, map_location='cpu')
        load_size = min(int(data.get('size', 0) or 0), self.max_size)
        if load_size <= 0:
            self.size = 0
            self.ptr = 0
            self.meta = {}
            return

        self.obs[:load_size] = data['obs'][:load_size].to(self.storage_device, dtype=self.dtype)
        self.proprio_hist[:load_size] = data['proprio_hist'][:load_size].to(self.storage_device, dtype=self.dtype)
        self.actions[:load_size] = data['actions'][:load_size].to(self.storage_device, dtype=self.dtype)

        self.size = load_size
        self.ptr = load_size % self.max_size
        self.meta = data.get('meta', {}) if isinstance(data, dict) else {}
        print(f"从 {filename} 加载了 {self.size} 条专家数据")
