"""
DOT-PG: Dual Optimal Transport Policy Gradient - 学生模型实现
基于论文 "Bridging the Reality Gap: Dual Optimal Transport Policy Gradient"

适配dexscrew项目的完整DOTPG学生模型实现

核心思想:
1. 将模仿学习建模为Wasserstein距离最小化问题
2. 利用Kantorovich对偶性转化为min-max问题
3. 从对偶函数自动提取策略梯度和价值函数

三阶段交替优化（论文Section 4.2.2）:
Stage 1 - Critic更新: max_φ [E_ρE[fφ] - E_ρπ[fφ] - λ * L_GP]  (公式27)
Stage 2 - Q网络更新: Qψ(s,a) ← fφ(s,a) + γ * Q_target(s', π(s'))  (公式28)
Stage 3 - 策略更新: min_θ E[Qψ(s, πθ(s))]  (公式29)

关键技术:
- 谱归一化: 强制Lipschitz约束
- 梯度惩罚: 进一步强化Lipschitz条件
- 延迟策略更新: 提高训练稳定性
- 双时间尺度优化: αφ >> αθ
- 目标网络软更新: 稳定TD学习
"""

import copy
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from termcolor import cprint

from dexscrew.utils.misc import AverageScalarMeter, tprint
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from dexscrew.dotpg.networks import PolicyNetwork, DualNetwork, QNetwork
from dexscrew.dotpg.buffer import ReplayBuffer, ExpertBuffer, RawReplayBuffer, RawExpertBuffer
from tensorboardX import SummaryWriter


class DOTPGConfig:
    """
    DOTPG超参数配置
    
    基于论文Section 9和核心笔记Section 9.1
    
    关键参数说明:
    - gamma: 折扣因子 γ
    - lambda_gp: 梯度惩罚系数 λ（论文中λ=10）
    - tau: 目标网络软更新率 τ（论文中τ=0.005）
    - lr_dual: 对偶网络学习率 αφ
    - lr_q: Q网络学习率 αψ
    - lr_policy: 策略网络学习率 αθ（重要：αφ >> αθ，双时间尺度）
    - dual_updates_per_iter: 对偶网络每次迭代更新次数 J（论文中J=5）
    - policy_delay: 策略更新延迟（类似TD3，提高稳定性）
    """
    def __init__(self, config_dict=None):
        # 状态空间模式
        # - student: 使用 proprio_hist 预测 extrin（适配 stage2 / DAgger 风格）
        # - teacher: 直接使用 PPO teacher 的 policy 输入（obs + tanh(priv_mlp(priv)+pc_mlp(pc))）
        self.state_mode = 'student'

        # ===== 稳定化开关（TD3-style）=====
        # 目标策略平滑噪声（TD3 target policy smoothing）
        self.target_policy_noise = 0.1
        self.target_noise_clip = 0.2
        # Q 网络是否使用 Huber loss（SmoothL1），对 outlier 更稳
        self.use_huber_q_loss = True

        # ===== Policy 架构/初始化 =====
        # dotpg_mlp: 原 DOTPG 两层 MLP policy。
        # teacher_actor: 使用 PPO teacher 的 actor_mlp+mu 架构，并在 restore_train 后从 teacher 初始化。
        # 对 dexterous contact task，这能避免从零拟合高维手部动作映射。
        self.policy_arch = 'dotpg_mlp'
        self.policy_init_from_teacher = True
        # teacher_actor 输出模式：clamp 对齐 teacher inference；raw 保留未裁剪梯度；tanh 为平滑有界输出。
        self.policy_output_mode = 'clamp'

        # ===== Policy loss 目标 =====
        # q: 原实现，最大化 Qψ(s,π(s))。
        # dual: 直接最大化 fφ(s,π(s))，更贴近论文 Theorem 3 的 immediate OT policy gradient。
        # q_dual: Q 长期项 + direct dual 项混合。
        self.policy_loss_mode = 'q'
        self.policy_dual_coef = 1.0

        # ===== OT / critic 输入度量缩放 =====
        # 这些缩放定义隐式 state-action metric。高维 state 可能压过 action 梯度；
        # CoDrive 接触任务可用 action_scale > 1 或 state_scale < 1 让 dual 更关注动作差异。
        self.dual_state_scale = 1.0
        self.dual_action_scale = 1.0
        self.critic_state_scale = 1.0
        self.critic_action_scale = 1.0

        # ===== BC / TD3+BC 稳定化 =====
        # BC 正则强度（0 表示关闭）。参考 TD3+BC：alpha = bc_coef / |Q|_mean
        self.bc_coef = 0.0
        # 训练前先做多少步纯 BC 预热（用专家 (s,a) 监督学习 policy）
        self.bc_pretrain_steps = 0
        # BC 预热时临时学习率（None 表示沿用 lr_policy）
        self.bc_pretrain_lr = None
        # BC 预热/正则时的 batch（默认沿用 dotpg.batch_size）
        self.bc_batch_size = None
        # 对 TD3+BC 的 alpha 做 clamp（None 表示不 clamp）
        self.bc_alpha_min = None
        self.bc_alpha_max = None

        # ===== Dual 更新策略样本的构造 =====
        # True：dual 更新时用当前 policy(states) 作为 policy_actions（更接近“当前 ρπ”）
        # False：使用 replay buffer 中存下来的 actions（可能是旧策略 + noise）
        self.dual_use_current_policy_actions = True
        # dual 更新时对 policy_actions 额外加噪声（可选，模拟探索分布）
        self.dual_policy_noise = 0.0

        # ===== Dual-reward 稳定化（防止 Q 发散）=====
        # 是否对 fφ(s,a) 做 running mean/std 归一化
        self.normalize_dual_reward = True
        # 归一化后的 dual reward 裁剪（None 关闭）
        self.dual_reward_clip = 5.0
        # dual reward 额外缩放系数
        self.dual_reward_scale = 1.0

        # ===== 专家数据模式 =====
        # False: 只用离线专家数据（训练过程中不再追加 teacher 演示）
        # True: 在线追加 teacher(s) 作为 DAgger 风格数据增强
        self.online_expert = False
        # online_expert 时，每个 env step 追加多少个 env 的 teacher demo（None 表示沿用 expert_add_num_envs；<=0 表示追加全部 env）
        self.online_expert_add_num_envs = None
        # 是否复用磁盘上的 expert buffer（teacher-state 一般可复用；student-state 若未同时恢复相同 encoder，通常建议关闭）
        self.reuse_expert_buffer = True
        # 专家缓冲区存储设备：'cpu'（推荐，省显存且可存更大 expert） / 'cuda' / 'cuda:0'
        self.expert_buffer_device = 'cpu'
        # 专家缓冲区存储 dtype（'float32' / 'float16'）
        self.expert_buffer_dtype = 'float32'
        # 每个 env step 写入 expert 的 env 数量（None/0 表示写入全部 env）
        # 作用：在 num_envs 很大时，避免 buffer 很快被同一时间片填满，提升“时间覆盖率”。
        self.expert_add_num_envs = None
        # 自动将 expert_buffer_size 扩到 warmup_steps * expert_add_num_envs（仅当 storage_device=cpu 时推荐开启）
        self.auto_expert_buffer_size = True
        # 防止误配导致一次性分配过大（样本数上限）
        self.max_auto_expert_buffer_size = 10_000_000

        # ===== student-state（无特权信息输入）表征预热 =====
        # 当 state_mode='student' 时，state 由 obs + proprio_hist->adapt_tconv 的 extrin_pred 构造。
        # DOTPG 属于 off-policy，表征变化会导致 replay/expert buffer 非平稳；推荐先预热 adapt_tconv 再冻结。
        # 预热步数：用 teacher rollout 收集 (proprio_hist, extrin_gt) 并监督训练 adapt_tconv。
        self.adapt_warmup_steps = 0
        # 预热阶段可选临时学习率（None 表示沿用默认 adapt_optimizer lr）
        self.adapt_warmup_lr = None
        # 预热后是否冻结 adapt_tconv（推荐 True 以提升 off-policy 稳定性）
        self.freeze_adapt_after_warmup = False
        # 预热后是否冻结 proprio_hist 的 RunningMeanStd（避免归一化漂移；默认不冻结）
        self.freeze_sa_mean_std_after_warmup = False
        # adapt 监督损失系数：latent MSE + action-level BC（参考原项目 ProprioAdapt）
        self.adapt_latent_coef = 1.0
        self.adapt_action_bc_coef = 0.0
        # student-state: 是否在 buffer 中存 raw obs/proprio_hist，并在采样时用当前 encoder 动态重建 state
        # 目的：允许 adapt_tconv 训练过程中持续更新，而不引入 off-policy 的 state 非平稳问题
        self.dynamic_state = False
        # dynamic_state 下 replay buffer 的存储位置/精度（默认存 CPU fp16 节省显存/内存）
        self.replay_buffer_device = 'cpu'
        self.replay_buffer_dtype = 'float16'

        # ===== 测试参数 =====
        # test_num_episodes: 统计多少个 episode（跨所有并行 env 汇总）
        self.test_num_episodes = 20
        # test_max_steps: 上限步数（0 表示不限，直到收集够 test_num_episodes）
        self.test_max_steps = 0
        # 有头可视化时按仿真控制周期同步到正常速度；headless 数值 eval 默认仍可跑满速。
        self.test_realtime = False
        self.test_realtime_factor = 1.0
        # 可选固定每个 policy step 的 wall-clock 秒数；优先级高于 test_realtime 自动周期。
        self.test_sleep_sec = None

        # ===== 训练恢复 =====
        # checkpoint=... 仍然保留给 PPO teacher；resume_path 用于继续 DOTPG student。
        self.resume_path = ''
        self.resume_load_optimizers = True

        # 折扣因子 - 论文Table 1
        self.gamma = 0.99
        
        # 梯度惩罚系数 λ - 论文Section 4.2.2, 核心笔记Section 9.1
        self.lambda_gp = 10.0
        
        # 目标网络软更新率 τ - 论文Section 4.3
        self.tau = 0.005
        
        # 学习率（双时间尺度：αφ >> αθ）- 核心笔记Section 9.1
        # 论文定理9要求: αφ >> αθ 保证收敛
        self.lr_dual = 3e-4      # 对偶网络学习率 αφ
        self.lr_q = 3e-4         # Q网络学习率 αψ
        self.lr_policy = 1e-4    # 策略网络学习率 αθ（更小！）
        
        # 批大小 - 论文Table 1
        self.batch_size = 256
        # 每个环境交互步执行多少次 DOTPG 参数更新。
        # 大显存云端可适当提高，以增加单位时间内的蒸馏更新量。
        self.updates_per_env_step = 1
        
        # 经验回放容量1000000
        self.buffer_size = 1000000
        self.expert_buffer_size = 500000
        
        # 对偶网络更新次数/迭代（J）- 论文Algorithm 1
        # 核心笔记Section 9.1: J=5
        self.dual_updates_per_iter = 5
        
        # 策略更新延迟 - 类似TD3
        self.policy_delay = 2
        
        # 隐藏层维度 - 核心笔记Section 9.2
        self.hidden_dim = 256
        
        # 探索噪声
        self.exploration_noise = 0.1
        
        # 预热步数（在开始训练前收集的数据量）
        self.warmup_steps = 10000
        
        # 从配置字典更新
        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)


class TeacherActorPolicy(nn.Module):
    """Policy wrapper using the same actor_mlp + mu architecture as the PPO teacher."""

    def __init__(self, actor_mlp: nn.Module, mu: nn.Module, output_mode: str = 'clamp'):
        super().__init__()
        self.actor_mlp = copy.deepcopy(actor_mlp)
        self.mu = copy.deepcopy(mu)
        self.output_mode = str(output_mode or 'clamp').lower()
        for p in self.parameters():
            p.requires_grad_(True)

    def forward(self, state):
        raw_action = self.mu(self.actor_mlp(state))
        if self.output_mode == 'raw':
            return raw_action
        if self.output_mode == 'tanh':
            return torch.tanh(raw_action)
        return torch.clamp(raw_action, -1.0, 1.0)

    def load_from_teacher(self, teacher_model):
        self.actor_mlp.load_state_dict(teacher_model.actor_mlp.state_dict())
        self.mu.load_state_dict(teacher_model.mu.state_dict())


class DOTPGStudent:
    """
    DOTPG学生模型 - 适配dexscrew项目
    
    论文: "Bridging the Reality Gap: Dual Optimal Transport Policy Gradient"
    算法: Algorithm 1 - Dual Optimal Transport Policy Gradient
    
    核心思想:
    1. 将模仿学习建模为Wasserstein距离最小化
    2. 利用Kantorovich对偶性转化为min-max问题
    3. 从对偶函数自动提取策略梯度和价值函数
    
    五阶段训练流程:
    Phase 1: 数据收集（学生策略与环境交互）
    Phase 2: 对偶网络更新（J次，梯度惩罚强制Lipschitz约束）
    Phase 3: Q网络更新（TD学习）
    Phase 4: 策略更新（延迟更新，确定性策略梯度）
    Phase 5: 目标网络软更新
    """
    
    def __init__(self, env, output_dir, full_config, student_dim=24):
        """
        初始化DOTPG学生模型
        
        Args:
            env: Isaac Gym环境
            output_dir: 输出目录
            full_config: 完整配置
            student_dim: 学生本体感知维度
        """
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        self.max_agent_steps = int(self.ppo_config.get('max_agent_steps', int(1e9)))
        
        # 获取DOTPG配置（如果存在）
        dotpg_config_dict = full_config.train.get('dotpg', {})
        if hasattr(dotpg_config_dict, '_content'):
            dotpg_config_dict = dict(dotpg_config_dict)
        self.config = DOTPGConfig(dotpg_config_dict if isinstance(dotpg_config_dict, dict) else None)
        self.test_num_steps = int(full_config.get("test_num_steps", 0) or 0)
        self.state_mode = getattr(self.config, 'state_mode', 'student')
        if self.state_mode not in ('student', 'teacher'):
            raise ValueError(
                f"Unsupported DOTPG state_mode={self.state_mode!r}. Expected 'student' or 'teacher'."
            )
        self.dynamic_state = bool(getattr(self.config, 'dynamic_state', False)) and self.state_mode == 'student'
        
        # ---- 环境设置 ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.proprio_dim = self.ppo_config.get('proprio_dim', 24)
        self.student_obs_shape = (student_dim * 3,)
        
        # ---- 特权信息设置 ----
        self.priv_info = self.ppo_config['priv_info']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.priv_info_dim = self.env.priv_info_dim
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        self.proprio_hist_dim = self.env.prop_hist_len
        if self.state_mode == 'student' and not self.proprio_adapt:
            raise ValueError(
                "DOTPG state_mode='student' requires train.ppo.proprio_adapt=True (needs proprio_hist -> extrin)."
            )
        
        # ---- 点云信息设置 ----
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.proprio_len = self.ppo_config['proprio_len']
        self.use_point_cloud_info = self.ppo_config['use_point_cloud_info']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        
        # ---- 计算状态维度 ----
        # 状态 = obs + extrin
        # extrin维度来自priv_mlp的最后一层
        self.extrin_dim = self.network_config.priv_mlp.units[-1]
        if self.use_point_cloud_info:
            self.extrin_dim += self.network_config.point_mlp.units[-1]
        self.state_dim = self.obs_shape[0] + self.extrin_dim
        
        # ---- 初始化教师模型（用于生成专家数据）----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'point_mlp_units': self.network_config.point_mlp.units,
            'use_point_cloud_info': self.use_point_cloud_info,
            'proprio_len': self.proprio_len,
            'proprio_dim': self.proprio_dim,
        }
        
        self.teacher_model = ActorCritic(net_config)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()  # 教师模型始终处于评估模式
        # teacher 仅用于前向生成 expert/action/label，不参与训练；关闭其参数梯度以节省显存/算力
        for p in self.teacher_model.parameters():
            p.requires_grad_(False)
        
        # ---- 初始化DOTPG网络 ----
        # 论文Section 4.2.1: Three-Component Architecture
        policy_arch = str(getattr(self.config, 'policy_arch', 'dotpg_mlp') or 'dotpg_mlp').lower()
        if policy_arch == 'teacher_actor':
            self.policy = TeacherActorPolicy(
                self.teacher_model.actor_mlp,
                self.teacher_model.mu,
                output_mode=getattr(self.config, 'policy_output_mode', 'clamp'),
            ).to(self.device)
            self.policy_target = TeacherActorPolicy(
                self.teacher_model.actor_mlp,
                self.teacher_model.mu,
                output_mode=getattr(self.config, 'policy_output_mode', 'clamp'),
            ).to(self.device)
        elif policy_arch == 'dotpg_mlp':
            self.policy = PolicyNetwork(
                self.state_dim, self.actions_num, self.config.hidden_dim
            ).to(self.device)
            # TD3: 目标策略网络（用于计算 TD target）
            self.policy_target = PolicyNetwork(
                self.state_dim, self.actions_num, self.config.hidden_dim
            ).to(self.device)
        else:
            raise ValueError(
                f"Unsupported train.dotpg.policy_arch={policy_arch!r}. "
                "Expected 'dotpg_mlp' or 'teacher_actor'."
            )
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.policy_target.eval()
        
        self.dual = DualNetwork(
            self.state_dim, self.actions_num, self.config.hidden_dim
        ).to(self.device)
        
        # TD3: Twin Q networks + targets，缓解过估计并提高稳定性
        self.q1 = QNetwork(
            self.state_dim, self.actions_num, self.config.hidden_dim
        ).to(self.device)
        self.q2 = QNetwork(
            self.state_dim, self.actions_num, self.config.hidden_dim
        ).to(self.device)
        self.q1_target = QNetwork(
            self.state_dim, self.actions_num, self.config.hidden_dim
        ).to(self.device)
        self.q2_target = QNetwork(
            self.state_dim, self.actions_num, self.config.hidden_dim
        ).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        self.q1_target.eval()
        self.q2_target.eval()
        
        # ---- 初始化adapt_tconv（从教师模型复制）----
        # 用于从proprio_hist预测extrin
        from dexscrew.algo.models.block import TemporalConv
        from dexscrew.algo.models.models import MLP
        temporal_fusing_input_dim = int(self.proprio_dim)
        temporal_fusing_output_dim = 8
        if self.use_point_cloud_info:
            temporal_fusing_output_dim += 32
        self.adapt_tconv = TemporalConv(temporal_fusing_input_dim, temporal_fusing_output_dim).to(self.device)
        
        # ---- 归一化模块 ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.proprio_dim)).to(self.device)
        self.sa_mean_std.train()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.point_cloud_mean_std.eval()

        # dual reward（fφ）归一化：仅用于 Q target，抑制数值爆炸
        self.dual_reward_mean_std = RunningMeanStd((1,)).to(self.device)
        if getattr(self.config, 'normalize_dual_reward', True):
            self.dual_reward_mean_std.train()
        else:
            self.dual_reward_mean_std.eval()
        
        # ---- 点云MLP（从教师模型复制）----
        if self.use_point_cloud_info:
            from dexscrew.algo.models.models import MLP
            self.point_mlp = MLP(
                units=self.network_config.point_mlp.units, 
                input_size=3
            ).to(self.device)
            for p in self.point_mlp.parameters():
                p.requires_grad_(False)
        
        # ---- 优化器 ----
        # 论文定理9: 双时间尺度优化，αφ >> αθ
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.config.lr_policy
        )
        self.dual_optimizer = optim.Adam(
            self.dual.parameters(), lr=self.config.lr_dual
        )
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=self.config.lr_q
        )
        # adapt_tconv优化器
        self.adapt_optimizer = optim.Adam(
            self.adapt_tconv.parameters(), lr=3e-4
        )
        
        # ---- 经验回放缓冲区 ----
        if self.dynamic_state:
            storage_device = getattr(self.config, 'replay_buffer_device', None)
            storage_device = str(storage_device).lower() if storage_device is not None else None
            if storage_device in ('cuda', 'gpu'):
                storage_device = str(self.device)

            dtype_cfg = str(getattr(self.config, 'replay_buffer_dtype', 'float16')).lower()
            if dtype_cfg in ('fp16', 'float16', 'half'):
                storage_dtype = torch.float16
            else:
                storage_dtype = torch.float32

            self.replay_buffer = RawReplayBuffer(
                obs_dim=self.obs_shape[0],
                proprio_hist_shape=(self.proprio_hist_dim, self.proprio_dim),
                action_dim=self.actions_num,
                max_size=int(self.config.buffer_size),
                device=str(self.device),
                storage_device=storage_device,
                dtype=storage_dtype,
                return_dtype=torch.float32,
            )
        else:
            self.replay_buffer = ReplayBuffer(
                self.state_dim, self.actions_num,
                self.config.buffer_size, self.device
            )
        # ---- 输出目录 ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'student_output', 'dotpg_nn')
        self.tb_dir = os.path.join(self.output_dir, 'student_output', 'dotpg_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # ---- 专家缓冲区 ----
        expert_storage_device = getattr(self.config, 'expert_buffer_device', 'cpu')
        expert_storage_device = str(expert_storage_device).lower() if expert_storage_device is not None else None
        if expert_storage_device in ('cuda', 'gpu'):
            expert_storage_device = str(self.device)

        expert_dtype_cfg = str(getattr(self.config, 'expert_buffer_dtype', 'float32')).lower()
        if expert_dtype_cfg in ('fp16', 'float16', 'half'):
            expert_dtype = torch.float16
        else:
            expert_dtype = torch.float32

        # 自动扩 expert_buffer_size，使其能覆盖 warmup_steps 的专家数据（按写入的 env 数量计算）
        expert_buffer_size = int(getattr(self.config, 'expert_buffer_size', 0) or 0)
        if expert_buffer_size <= 0:
            expert_buffer_size = int(5e5)

        add_num_envs = getattr(self.config, 'expert_add_num_envs', None)
        if add_num_envs is None:
            add_num_envs = 0
        add_num_envs = int(add_num_envs) if isinstance(add_num_envs, (int, float, np.integer)) else 0
        if add_num_envs <= 0 or add_num_envs >= self.num_actors:
            add_num_envs = self.num_actors

        warmup_steps = int(getattr(self.config, 'warmup_steps', 0) or 0)
        desired_samples = warmup_steps * add_num_envs
        if (
            bool(getattr(self.config, 'auto_expert_buffer_size', True))
            and warmup_steps > 0
            and desired_samples > expert_buffer_size
            and str(expert_storage_device).startswith('cpu')
        ):
            max_cap = int(getattr(self.config, 'max_auto_expert_buffer_size', 0) or 0)
            if max_cap <= 0:
                max_cap = 10_000_000
            if desired_samples > max_cap:
                cprint(
                    f"[DOTPG] 警告: 期望 expert 样本数 warmup_steps*expert_add_num_envs={desired_samples} "
                    f"超过 max_auto_expert_buffer_size={max_cap}，将只分配 {max_cap}。",
                    'yellow',
                )
                expert_buffer_size = max_cap
            else:
                expert_buffer_size = desired_samples

        # 确保后续逻辑读到的是最终 buffer size
        self.config.expert_buffer_size = int(expert_buffer_size)

        if self.dynamic_state:
            self.expert_buffer = RawExpertBuffer(
                obs_dim=self.obs_shape[0],
                proprio_hist_shape=(self.proprio_hist_dim, self.proprio_dim),
                action_dim=self.actions_num,
                max_size=int(self.config.expert_buffer_size),
                device=str(self.device),
                storage_device=str(expert_storage_device),
                dtype=expert_dtype,
                return_dtype=torch.float32,
            )
            self.expert_buffer_path = os.path.join(self.output_dir, f'expert_buffer_{self.state_mode}_raw.pt')
        else:
            self.expert_buffer = ExpertBuffer(
                self.state_dim, self.actions_num,
                self.config.expert_buffer_size,
                device=self.device,
                storage_device=expert_storage_device,
                dtype=expert_dtype,
                return_dtype=torch.float32,
            )
            self.expert_buffer_path = os.path.join(self.output_dir, f'expert_buffer_{self.state_mode}.pt')
        if bool(getattr(self.config, 'reuse_expert_buffer', True)) and os.path.isfile(self.expert_buffer_path):
            try:
                self.expert_buffer.load(self.expert_buffer_path)
            except Exception as e:
                cprint(f'加载专家数据失败，忽略并重新采集: {e}', 'yellow')
        self.expert_collected_env_steps = int(self.expert_buffer.meta.get('collected_env_steps', 0) or 0)
        self.adapt_pretrained_env_steps = int(self.expert_buffer.meta.get('adapt_pretrained_env_steps', 0) or 0)
        self._adapt_frozen = False
        self._sa_mean_std_frozen = False
        
        # ---- 训练统计 ----
        self.direct_info = {}
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        self.total_it = 0
        
        # ---- 训练状态 ----
        self.step_reward = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        
        # ---- 训练指标 ----
        self.metrics = {
            'dual_loss': [],
            'q_loss': [],
            'policy_loss': [],
            'wasserstein_dist': [],
            'gradient_penalty': [],
            'adapt_loss': []
        }

    def _dual_inputs(self, states, actions):
        state_scale = float(getattr(self.config, 'dual_state_scale', 1.0) or 1.0)
        action_scale = float(getattr(self.config, 'dual_action_scale', 1.0) or 1.0)
        return states * state_scale, actions * action_scale

    def _critic_inputs(self, states, actions):
        state_scale = float(getattr(self.config, 'critic_state_scale', 1.0) or 1.0)
        action_scale = float(getattr(self.config, 'critic_action_scale', 1.0) or 1.0)
        return states * state_scale, actions * action_scale

    def _dual_value(self, states, actions):
        states_in, actions_in = self._dual_inputs(states, actions)
        return self.dual(states_in, actions_in)

    def _q1_value(self, states, actions):
        states_in, actions_in = self._critic_inputs(states, actions)
        return self.q1(states_in, actions_in)

    def _q2_value(self, states, actions):
        states_in, actions_in = self._critic_inputs(states, actions)
        return self.q2(states_in, actions_in)

    def _q1_target_value(self, states, actions):
        states_in, actions_in = self._critic_inputs(states, actions)
        return self.q1_target(states_in, actions_in)

    def _q2_target_value(self, states, actions):
        states_in, actions_in = self._critic_inputs(states, actions)
        return self.q2_target(states_in, actions_in)

    def _maybe_init_policy_from_teacher(self):
        if str(getattr(self.config, 'policy_arch', 'dotpg_mlp')).lower() != 'teacher_actor':
            return
        if not bool(getattr(self.config, 'policy_init_from_teacher', True)):
            return
        if not hasattr(self.policy, 'load_from_teacher'):
            return
        self.policy.load_from_teacher(self.teacher_model)
        self.policy_target.load_state_dict(self.policy.state_dict())
        self.policy_target.eval()
        cprint('[DOTPG] policy initialized from PPO teacher actor_mlp+mu', 'green', attrs=['bold'])
    
    def compute_gradient_penalty(self, expert_states, expert_actions, 
                                  policy_states, policy_actions):
        """
        计算梯度惩罚 L_GP
        
        论文公式27（核心笔记公式20）:
        L_GP = E[(||∇fφ(ŝ,â)||_2 - 1)²]
        
        其中插值样本: 
        (ŝ,â) = ε(s_E,a_E) + (1-ε)(s_π,a_π), ε ~ U[0,1]
        
        目的: 强制对偶函数fφ满足1-Lipschitz条件
        
        论文对应:
        - Section 4.2.2 Stage 1: Gradient penalty term
        - 核心笔记 Section 10.1: 梯度惩罚计算
        
        Args:
            expert_states: 专家状态 s_E
            expert_actions: 专家动作 a_E
            policy_states: 策略状态 s_π
            policy_actions: 策略动作 a_π
        
        Returns:
            gradient_penalty: 梯度惩罚 L_GP
        """
        batch_size = min(expert_states.size(0), policy_states.size(0))
        
        # 随机插值系数 ε ~ U[0,1]
        epsilon = torch.rand(batch_size, 1, device=self.device)
        
        # 计算插值点 - 论文公式: (ŝ,â) = ε(s_E,a_E) + (1-ε)(s_π,a_π)
        # ŝ = ε*s_E + (1-ε)*s_π
        interp_states = epsilon * expert_states[:batch_size] + \
                       (1 - epsilon) * policy_states[:batch_size]
        # â = ε*a_E + (1-ε)*a_π
        interp_actions = epsilon * expert_actions[:batch_size] + \
                        (1 - epsilon) * policy_actions[:batch_size]
        
        # 启用梯度计算
        interp_states = interp_states.clone().requires_grad_(True)
        interp_actions = interp_actions.clone().requires_grad_(True)
        
        # 计算对偶网络输出 fφ(ŝ,â)
        dual_output = self._dual_value(interp_states, interp_actions)
        
        # 计算梯度 ∇fφ(ŝ,â)
        gradients = torch.autograd.grad(
            outputs=dual_output,
            inputs=[interp_states, interp_actions],
            grad_outputs=torch.ones_like(dual_output),
            create_graph=True,
            retain_graph=True
        )
        
        # 合并状态和动作的梯度
        gradients = torch.cat([gradients[0], gradients[1]], dim=1)
        
        # 计算梯度范数 ||∇fφ||_2
        gradient_norm = gradients.norm(2, dim=1)
        
        # 梯度惩罚: (||∇fφ|| - 1)² - 强制1-Lipschitz
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        
        return gradient_penalty
    
    def update_dual_network(self, expert_states, expert_actions, 
                           policy_states, policy_actions):
        """
        更新对偶网络（Stage 1）
        
        论文公式27（核心笔记公式19）:
        max_φ [E_ρE[fφ(s,a)] - E_ρπ[fφ(s,a)] - λ·L_GP]
        
        等价于最小化:
        L_dual = -[E_ρE[fφ] - E_ρπ[fφ]] + λ·L_GP
        
        Wasserstein距离估计（论文公式21）:
        W(ρπ, ρE) ≈ E_ρE[fφ] - E_ρπ[fφ]
        
        论文对应:
        - Section 4.2.2 Stage 1: Critic Update
        - 核心笔记 Section 10.2: 对偶网络更新
        
        Args:
            expert_states: 专家状态
            expert_actions: 专家动作
            policy_states: 策略状态
            policy_actions: 策略动作
        
        Returns:
            dual_loss: 对偶损失
            wasserstein_dist: Wasserstein距离估计
            gradient_penalty: 梯度惩罚
        """
        # 计算对偶函数值
        # E_ρE[fφ(s,a)]
        expert_value = self._dual_value(expert_states, expert_actions)
        # E_ρπ[fφ(s,a)]
        policy_value = self._dual_value(policy_states, policy_actions)
        
        # Wasserstein距离估计: W(ρπ, ρE) ≈ E_ρE[f] - E_ρπ[f]
        # 论文公式21
        wasserstein_dist = expert_value.mean() - policy_value.mean()
        
        # 计算梯度惩罚 L_GP
        gradient_penalty = self.compute_gradient_penalty(
            expert_states, expert_actions, policy_states, policy_actions
        )
        
        # 对偶网络损失（最小化负的Wasserstein距离 + 梯度惩罚）
        # 等价于最大化 Wasserstein距离 - λ·L_GP
        # L_dual = -W + λ·L_GP
        dual_loss = -wasserstein_dist + self.config.lambda_gp * gradient_penalty
        
        # 更新对偶网络
        self.dual_optimizer.zero_grad()
        dual_loss.backward()
        self.dual_optimizer.step()
        
        return dual_loss.item(), wasserstein_dist.item(), gradient_penalty.item()
    
    def update_q_network(self, states, actions, next_states, dones):
        """
        更新Q网络（Stage 2）
        
        论文公式28（核心笔记公式21）:
        TD目标: y = fφ(s,a) + γ·Q_target(s', πθ(s'))
        Q网络损失: L_Q = E[(Qψ(s,a) - y)²]
        
        关键: 使用对偶网络输出 fφ(s,a) 作为即时奖励
        
        论文对应:
        - Section 4.2.2 Stage 2: Value Learning
        - 核心笔记 Section 10.3: Q网络TD更新
        
        Args:
            states: 当前状态 s
            actions: 当前动作 a
            next_states: 下一状态 s'
            dones: 终止标志
        
        Returns:
            q_loss: Q网络损失
        """
        with torch.no_grad():
            # 计算下一状态的动作: a' = π_target(s') + clipped noise（TD3 smoothing）
            next_actions = self.policy_target(next_states)
            if self.config.target_policy_noise and self.config.target_policy_noise > 0:
                noise = torch.randn_like(next_actions) * float(self.config.target_policy_noise)
                noise = noise.clamp(-float(self.config.target_noise_clip), float(self.config.target_noise_clip))
                next_actions = (next_actions + noise).clamp(-1.0, 1.0)

            # 目标Q：min(Q1_target, Q2_target)
            target_q1 = self._q1_target_value(next_states, next_actions)
            target_q2 = self._q2_target_value(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)

            # 使用对偶网络输出作为即时奖励: r = fφ(s,a)
            dual_reward = self._dual_value(states, actions)
            if getattr(self.config, 'normalize_dual_reward', True):
                dual_reward = self.dual_reward_mean_std(dual_reward)
            dual_reward = dual_reward * float(getattr(self.config, 'dual_reward_scale', 1.0))
            clip = getattr(self.config, 'dual_reward_clip', None)
            if clip is not None:
                dual_reward = dual_reward.clamp(-float(clip), float(clip))

            # TD目标: y = r + γ·(1-done)·Q_target(s', a')
            y = dual_reward + self.config.gamma * (1 - dones) * target_q

        # 当前Q值
        current_q1 = self._q1_value(states, actions)
        current_q2 = self._q2_value(states, actions)

        # Q网络损失
        if self.config.use_huber_q_loss:
            q_loss = F.smooth_l1_loss(current_q1, y) + F.smooth_l1_loss(current_q2, y)
        else:
            q_loss = F.mse_loss(current_q1, y) + F.mse_loss(current_q2, y)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        return float(q_loss.detach().cpu())
    
    def update_policy(self, states, expert_states=None, expert_actions=None):
        """
        更新策略网络（Stage 3）
        
        论文定理3（OT策略梯度）:
        ∇θ W(ρπ, ρE) = -E[∇θπθ(s) ∇a fφ(s,a)|a=πθ(s)]
        
        论文公式29:
        min_θ E_{s~ρ_π}[Qψ(s, πθ(s))]
        
        等价于最大化Q值:
        max_θ E[Qψ(s, πθ(s))]
        
        即最小化:
        L_π = -E[Qψ(s, πθ(s))]
        
        论文对应:
        - Section 4.2.2 Stage 3: Policy Improvement
        - 核心笔记 Section 10.4: 策略更新
        
        Args:
            states: 状态 s
        
        Returns:
            policy_loss: 策略损失
        """
        # 生成新动作: a = πθ(s)
        new_actions = self.policy(states)
        
        # 策略损失:
        # - q: 原 TD3-style 长期 dual reward 回报。
        # - dual: 直接使用 OT dual potential 的动作梯度，更贴近 Theorem 3。
        # - q_dual: 两者混合，用 direct dual 修正 Q 近似误差。
        policy_q = torch.min(self._q1_value(states, new_actions), self._q2_value(states, new_actions))
        dual_policy_value = self._dual_value(states, new_actions)
        policy_loss_mode = str(getattr(self.config, 'policy_loss_mode', 'q') or 'q').lower()
        if policy_loss_mode == 'q':
            policy_loss = -policy_q.mean()
        elif policy_loss_mode == 'dual':
            policy_loss = -dual_policy_value.mean()
        elif policy_loss_mode == 'q_dual':
            dual_coef = float(getattr(self.config, 'policy_dual_coef', 1.0) or 0.0)
            policy_loss = -policy_q.mean() - dual_coef * dual_policy_value.mean()
        else:
            raise ValueError(
                f"Unsupported train.dotpg.policy_loss_mode={policy_loss_mode!r}. "
                "Expected 'q', 'dual', or 'q_dual'."
            )

        bc_loss = None
        if (
            getattr(self.config, 'bc_coef', 0.0)
            and float(self.config.bc_coef) > 0.0
            and expert_states is not None
            and expert_actions is not None
        ):
            bc_pred = self.policy(expert_states)
            bc_loss = F.mse_loss(bc_pred, expert_actions)
            # TD3+BC 常用的自适应缩放：alpha = λ / |Q|_mean
            q_abs_mean = policy_q.detach().abs().mean().clamp(min=1e-6)
            alpha = float(self.config.bc_coef) / float(q_abs_mean)
            alpha_min = getattr(self.config, 'bc_alpha_min', None)
            alpha_max = getattr(self.config, 'bc_alpha_max', None)
            if alpha_min is not None:
                alpha = max(alpha, float(alpha_min))
            if alpha_max is not None:
                alpha = min(alpha, float(alpha_max))
            policy_loss = policy_loss + alpha * bc_loss
        
        # 更新策略网络
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return (
            float(policy_loss.detach().cpu()),
            (float(bc_loss.detach().cpu()) if bc_loss is not None else None),
            float(policy_q.mean().detach().cpu()),
            float(dual_policy_value.mean().detach().cpu()),
        )

    def pretrain_policy_bc(self):
        """用专家 (s,a) 对 policy 做纯 BC 预热，避免初期随机动作导致分布差太大。"""
        steps = int(getattr(self.config, 'bc_pretrain_steps', 0) or 0)
        coef = float(getattr(self.config, 'bc_coef', 0.0) or 0.0)
        if steps <= 0 or coef <= 0.0:
            return
        if len(self.expert_buffer) <= 0:
            return

        batch_size = getattr(self.config, 'bc_batch_size', None)
        batch_size = int(batch_size) if batch_size is not None else int(self.config.batch_size)
        batch_size = max(1, batch_size)

        cprint(f'BC 预热: {steps} steps (batch={batch_size})...', 'green', attrs=['bold'])

        # BC 预热通常需要更大的 LR；支持临时覆盖 lr_policy
        pretrain_lr = getattr(self.config, 'bc_pretrain_lr', None)
        old_lrs = None
        if pretrain_lr is not None:
            old_lrs = [pg.get('lr', None) for pg in self.policy_optimizer.param_groups]
            for pg in self.policy_optimizer.param_groups:
                pg['lr'] = float(pretrain_lr)

        self.policy.train()
        for i in range(steps):
            if self.dynamic_state:
                obs, proprio_hist, actions = self.expert_buffer.sample(batch_size)
                states = self.build_student_state_from_raw(obs, proprio_hist)
            else:
                states, actions = self.expert_buffer.sample(batch_size)
            pred = self.policy(states)
            loss = F.mse_loss(pred, actions)
            self.policy_optimizer.zero_grad()
            loss.backward()
            self.policy_optimizer.step()
            if (i + 1) % 200 == 0:
                tprint(f'BC 预热进度: {i + 1}/{steps} | loss: {float(loss.detach().cpu()):.4f}')

        if old_lrs is not None:
            for pg, lr in zip(self.policy_optimizer.param_groups, old_lrs):
                if lr is not None:
                    pg['lr'] = lr

        # 预热后同步 target policy
        self.policy_target.load_state_dict(self.policy.state_dict())
    
    def update_target_network(self):
        """
        软更新目标网络（Phase 5）
        
        论文: 目标网络软更新（核心笔记Section 10.5）
        ψ_target ← τ·ψ + (1-τ)·ψ_target
        
        其中 τ 是软更新率（通常τ=0.005）
        
        论文对应:
        - Section 4.3: Target network stabilization
        - 定理10: 目标网络稳定化
        """
        tau = float(self.config.tau)

        def _soft_update(source: nn.Module, target: nn.Module):
            for param, target_param in zip(source.parameters(), target.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

        _soft_update(self.q1, self.q1_target)
        _soft_update(self.q2, self.q2_target)
        _soft_update(self.policy, self.policy_target)
    
    def get_state_from_obs(self, obs_dict, use_teacher_extrin=False):
        """
        从观察字典构建完整状态
        
        状态 = normalized_obs + extrin
        
        Args:
            obs_dict: 观察字典
            use_teacher_extrin: 是否使用教师的extrin（用于专家数据收集）
        
        Returns:
            state: 完整状态 (batch_size, state_dim)
            extrin: extrin向量
            extrin_gt: 教师extrin（ground truth）
        """
        # 归一化观察
        obs = self.running_mean_std(obs_dict['obs'])

        # 处理点云
        point_cloud_info = None
        if self.use_point_cloud_info:
            if 'point_cloud_info' not in obs_dict:
                raise KeyError("DOTPG requires 'point_cloud_info' in obs_dict when use_point_cloud_info=True.")
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(
                    obs_dict['point_cloud_info'].reshape(-1, 3)
                ).reshape((obs.shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']

        # teacher-state：直接对齐 PPO teacher 的 policy 输入
        if self.state_mode == 'teacher':
            if 'priv_info' not in obs_dict:
                raise KeyError("DOTPG state_mode='teacher' requires 'priv_info' in obs_dict.")
            priv_info = self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info']
            with torch.no_grad():
                extrin = self.teacher_model.env_mlp(priv_info)
                if self.use_point_cloud_info:
                    pcs = self.point_mlp(point_cloud_info)
                    pcs = torch.max(pcs, 1)[0]
                    extrin = torch.cat([extrin, pcs], dim=-1)
                extrin = torch.tanh(extrin)
                state = torch.cat([obs, extrin], dim=-1)
            return state, extrin, extrin
        
        # 计算学生extrin（从proprio_hist）
        proprio_hist_normalized = self.sa_mean_std(obs_dict['proprio_hist'].detach())
        extrin = self.adapt_tconv(proprio_hist_normalized)
        
        # teacher 的 extrin 作为监督 label：严格对齐 teacher policy 的输入分布
        # extrin_gt = tanh(concat(env_mlp(priv_info), point_mlp(point_cloud)))
        extrin_gt = extrin
        if 'priv_info' in obs_dict:
            priv_info = self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info']
            with torch.no_grad():
                extrin_gt = self.teacher_model.env_mlp(priv_info)
                if self.use_point_cloud_info:
                    pcs = self.point_mlp(point_cloud_info)
                    pcs = torch.max(pcs, 1)[0]
                    extrin_gt = torch.cat([extrin_gt, pcs], dim=-1)

        extrin_gt = torch.tanh(extrin_gt)
        # student 预测的 extrin（输入给 policy/dual/q）
        extrin = torch.tanh(extrin)
        
        # 构建完整状态
        if use_teacher_extrin:
            state = torch.cat([obs, extrin_gt.detach()], dim=-1)
        else:
            state = torch.cat([obs, extrin], dim=-1)
        
        return state, extrin, extrin_gt

    @torch.no_grad()
    def build_student_state_from_raw(self, obs: torch.Tensor, proprio_hist: torch.Tensor) -> torch.Tensor:
        """
        dynamic_state 专用：用当前 encoder 从 raw (obs, proprio_hist) 动态构造 student-state。

        注意：
        - 这里默认不更新 sa_mean_std（避免采样 replay/expert 时污染归一化统计）。
        - 返回的 state 不带梯度，用于 off-policy 的 dual/q/policy 更新。
        """
        # 固定 obs 归一化（来自 teacher ckpt，不应更新）
        obs_norm = self.running_mean_std(obs)

        # 避免从 buffer 采样时更新 sa_mean_std
        was_training = self.sa_mean_std.training
        self.sa_mean_std.eval()
        proprio_norm = self.sa_mean_std(proprio_hist)
        if was_training:
            self.sa_mean_std.train()

        extrin = torch.tanh(self.adapt_tconv(proprio_norm))
        return torch.cat([obs_norm, extrin], dim=-1)

    def _teacher_action_from_extrin(self, obs: torch.Tensor, extrin: torch.Tensor) -> torch.Tensor:
        """用 teacher 的 actor_mlp+mu 将 (obs, extrin) 映射为动作（对 extrin 可求梯度）。"""
        teacher_x = self.teacher_model.actor_mlp(torch.cat([obs, extrin], dim=-1))
        teacher_action = self.teacher_model.mu(teacher_x)
        return torch.clamp(teacher_action, -1.0, 1.0)

    @torch.no_grad()
    def get_teacher_action(self, obs_dict):
        """
        使用 PPO teacher 策略输出动作（用于专家数据/在线监督动作）。

        这里严格按 teacher policy 的输入构建：obs_norm + tanh(priv_mlp(priv)+pc_mlp(pc))。
        """
        obs = self.running_mean_std(obs_dict['obs'])

        if 'priv_info' not in obs_dict:
            raise KeyError("DOTPG teacher action requires 'priv_info' in obs_dict.")
        priv_info = self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info']
        extrin = self.teacher_model.env_mlp(priv_info)

        if self.use_point_cloud_info:
            if 'point_cloud_info' not in obs_dict:
                raise KeyError("DOTPG teacher action requires 'point_cloud_info' when use_point_cloud_info=True.")
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(
                    obs_dict['point_cloud_info'].reshape(-1, 3)
                ).reshape((obs.shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']

            pcs = self.point_mlp(point_cloud_info)
            pcs = torch.max(pcs, 1)[0]
            extrin = torch.cat([extrin, pcs], dim=-1)

        extrin = torch.tanh(extrin)
        teacher_obs_input = torch.cat([obs, extrin], dim=-1)
        teacher_x = self.teacher_model.actor_mlp(teacher_obs_input)
        teacher_action = self.teacher_model.mu(teacher_x)
        return torch.clamp(teacher_action, -1.0, 1.0)

    @torch.no_grad()
    def get_teacher_action_from_state(self, teacher_state):
        """当 state_mode='teacher' 时，可直接复用已构建的 teacher_state 计算教师动作。"""
        teacher_x = self.teacher_model.actor_mlp(teacher_state)
        teacher_action = self.teacher_model.mu(teacher_x)
        return torch.clamp(teacher_action, -1.0, 1.0)

    def collect_expert_data(self, obs_dict, num_steps=10000):
        """
        使用教师模型收集专家数据
        
        Args:
            obs_dict: 初始观察
            num_steps: 收集步数
        """
        cprint(f'开始收集专家数据（{num_steps}步）...', 'green', attrs=['bold'])

        add_num_envs = getattr(self.config, 'expert_add_num_envs', None)
        if add_num_envs is None:
            add_num_envs = 0
        add_num_envs = int(add_num_envs) if isinstance(add_num_envs, (int, float, np.integer)) else 0
        if add_num_envs <= 0 or add_num_envs >= self.num_actors:
            add_num_envs = self.num_actors
        
        for step in range(num_steps):
            # 在当前 state_mode 下构建状态/或存 raw，但用 teacher policy 给出专家动作
            state = None
            if not self.dynamic_state:
                state, _, _ = self.get_state_from_obs(obs_dict)
            teacher_action = (
                self.get_teacher_action_from_state(state)
                if self.state_mode == 'teacher'
                else self.get_teacher_action(obs_dict)
            )
            teacher_action = teacher_action.contiguous()
            if self.dynamic_state:
                obs = obs_dict['obs'].detach()
                proprio_hist = obs_dict['proprio_hist'].detach()
                if add_num_envs == self.num_actors:
                    self.expert_buffer.add_batch(obs, proprio_hist, teacher_action.detach())
                else:
                    idx = torch.randperm(self.num_actors, device=obs.device)[:add_num_envs]
                    self.expert_buffer.add_batch(obs[idx], proprio_hist[idx], teacher_action[idx].detach())
            else:
                if add_num_envs == self.num_actors:
                    self.expert_buffer.add_batch(state.detach(), teacher_action.detach())
                else:
                    # 随机选择一部分 env 写入 expert buffer，提高时间覆盖率并节省内存/磁盘
                    idx = torch.randperm(self.num_actors, device=state.device)[:add_num_envs]
                    self.expert_buffer.add_batch(state[idx].detach(), teacher_action[idx].detach())
            
            # 执行动作
            obs_dict, r, done, info = self.env.step(teacher_action)
            
            if (step + 1) % 1000 == 0:
                tprint(f'专家数据收集进度: {step + 1}/{num_steps}')
        
        cprint(f'专家数据收集完成，共 {len(self.expert_buffer)} 条样本', 'green', attrs=['bold'])
        self.expert_collected_env_steps += int(num_steps)
        self.save_expert_buffer()

    def save_expert_buffer(self):
        """将专家数据持久化到输出目录，便于断点重启。"""
        if not hasattr(self, 'expert_buffer_path') or self.expert_buffer_path is None:
            return
        try:
            add_num_envs = getattr(self.config, 'expert_add_num_envs', None)
            meta = {
                'state_mode': self.state_mode,
                'buffer_format': 'raw_obs' if self.dynamic_state else 'state',
                'num_envs_at_collect': int(self.num_actors),
                'expert_add_num_envs': int(add_num_envs) if add_num_envs is not None else None,
                'warmup_env_steps_target': int(getattr(self.config, 'warmup_steps', 0) or 0),
                'collected_env_steps': int(self.expert_collected_env_steps),
                'adapt_pretrained_env_steps': int(getattr(self, 'adapt_pretrained_env_steps', 0) or 0),
                'adapt_frozen': bool(getattr(self, '_adapt_frozen', False)),
                'sa_mean_std_frozen': bool(getattr(self, '_sa_mean_std_frozen', False)),
                'expert_buffer_size': int(getattr(self.config, 'expert_buffer_size', 0) or 0),
                'saved_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            }
            self.expert_buffer.save(self.expert_buffer_path, meta=meta)
        except Exception as e:
            cprint(f'专家数据保存失败，后续会重新采集: {e}', 'yellow')
    
    def train_step(self):
        """
        执行一次完整的DOTPG训练步骤
        
        论文Algorithm 1: Dual Optimal Transport Policy Gradient
        
        五阶段流程:
        1. 数据采样
        2. 对偶网络更新（J次）- Stage 1
        3. Q网络更新 - Stage 2
        4. 策略更新（延迟更新）- Stage 3
        5. 目标网络软更新
        
        Returns:
            metrics: 训练指标字典
        """
        if len(self.replay_buffer) < self.config.batch_size:
            return None
        
        self.total_it += 1
        
        # ==================== Phase 1: 数据采样 ====================
        # 从策略缓冲区采样（dynamic_state 时采样 raw，再动态构造 state）
        if self.dynamic_state:
            policy_obs, policy_proprio_hist, policy_actions, next_obs, next_proprio_hist, dones = self.replay_buffer.sample(
                self.config.batch_size
            )
            policy_states = self.build_student_state_from_raw(policy_obs, policy_proprio_hist)
            next_states = self.build_student_state_from_raw(next_obs, next_proprio_hist)
        else:
            policy_states, policy_actions, next_states, dones = self.replay_buffer.sample(self.config.batch_size)

        # dual 更新时，尽量用“当前 policy”对应的动作，减少 stale-action 带来的分布误差
        policy_actions_for_dual = policy_actions
        if bool(getattr(self.config, 'dual_use_current_policy_actions', True)):
            with torch.no_grad():
                policy_actions_for_dual = self.policy(policy_states)
                noise_std = float(getattr(self.config, 'dual_policy_noise', 0.0) or 0.0)
                if noise_std > 0.0:
                    noise = torch.randn_like(policy_actions_for_dual) * noise_std
                    policy_actions_for_dual = (policy_actions_for_dual + noise).clamp(-1.0, 1.0)
                else:
                    policy_actions_for_dual = policy_actions_for_dual.clamp(-1.0, 1.0)
        
        # 从专家缓冲区采样（dynamic_state 时采样 raw，再动态构造 state）
        if self.dynamic_state:
            expert_obs, expert_proprio_hist, expert_actions = self.expert_buffer.sample(self.config.batch_size)
            expert_states = self.build_student_state_from_raw(expert_obs, expert_proprio_hist)
        else:
            expert_states, expert_actions = self.expert_buffer.sample(self.config.batch_size)
        
        # ==================== Phase 2: 对偶网络更新（J次）====================
        # 论文Algorithm 1: 对偶网络更新次数更多（J次），保证分布测量准确性
        # 核心笔记Section 9.1: J=5
        for _ in range(self.config.dual_updates_per_iter):
            dual_loss, wasserstein_dist, gradient_penalty = self.update_dual_network(
                expert_states, expert_actions, policy_states, policy_actions_for_dual
            )
        
        # ==================== Phase 3: Q网络更新 ====================
        q_loss = self.update_q_network(policy_states, policy_actions, next_states, dones)
        
        # ==================== Phase 4: 策略更新（延迟更新）====================
        # 类似TD3的延迟策略更新，提高稳定性
        policy_loss_value = 0.0
        bc_loss_value = None
        expert_action_mse = None
        policy_q_value = None
        policy_dual_value = None
        if self.total_it % self.config.policy_delay == 0:
            policy_loss_value, bc_loss_value, policy_q_value, policy_dual_value = self.update_policy(
                policy_states, expert_states=expert_states, expert_actions=expert_actions
            )

            with torch.no_grad():
                expert_action_mse = float(F.mse_loss(self.policy(expert_states), expert_actions).detach().cpu())
            
            # ==================== Phase 5: 目标网络软更新 ====================
            self.update_target_network()
        
        # 记录指标
        metrics = {
            'dual_loss': dual_loss,
            'q_loss': q_loss,
            'policy_loss': policy_loss_value,
            'wasserstein_dist': wasserstein_dist,
            'gradient_penalty': gradient_penalty
        }
        if bc_loss_value is not None:
            metrics['bc_loss'] = bc_loss_value
        if expert_action_mse is not None:
            metrics['expert_action_mse'] = expert_action_mse
        if policy_q_value is not None:
            metrics['policy_q'] = policy_q_value
        if policy_dual_value is not None:
            metrics['policy_dual'] = policy_dual_value
        
        return metrics
    
    def set_eval(self):
        """设置为评估模式"""
        self.policy.eval()
        self.policy_target.eval()
        self.dual.eval()
        self.q1.eval()
        self.q2.eval()
        self.q1_target.eval()
        self.q2_target.eval()
        self.adapt_tconv.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        self.dual_reward_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()
    
    def set_train(self):
        """设置为训练模式"""
        self.policy.train()
        self.policy_target.eval()
        self.dual.train()
        self.q1.train()
        self.q2.train()
        if self._adapt_frozen:
            self.adapt_tconv.eval()
        else:
            self.adapt_tconv.train()
        if self._sa_mean_std_frozen:
            self.sa_mean_std.eval()
        else:
            self.sa_mean_std.train()
        if getattr(self.config, 'normalize_dual_reward', True):
            self.dual_reward_mean_std.train()

    @staticmethod
    def _set_requires_grad(module: nn.Module, requires_grad: bool):
        for p in module.parameters():
            p.requires_grad_(requires_grad)

    def freeze_student_encoder(self, freeze_adapt: bool = True, freeze_sa_mean_std: bool = False):
        """冻结/解冻 student-state 的表征模块（adapt_tconv 与 proprio 的归一化）。"""
        self._adapt_frozen = bool(freeze_adapt)
        self._sa_mean_std_frozen = bool(freeze_sa_mean_std)

        if self._adapt_frozen:
            self._set_requires_grad(self.adapt_tconv, False)
            self.adapt_tconv.eval()
        else:
            self._set_requires_grad(self.adapt_tconv, True)
            self.adapt_tconv.train()

        if self._sa_mean_std_frozen:
            self.sa_mean_std.eval()
        else:
            self.sa_mean_std.train()

    def pretrain_adapt_tconv(self, obs_dict, num_steps: int):
        """用 teacher rollout 对 adapt_tconv 做监督预热（proprio_hist -> extrin_gt）。"""
        num_steps = int(num_steps)
        if num_steps <= 0:
            return
        if self.state_mode != 'student':
            return
        if not self.proprio_adapt:
            raise ValueError("DOTPG state_mode='student' requires train.ppo.proprio_adapt=True.")

        # 预热阶段允许临时覆盖 adapt lr（更快收敛到可用表征）
        warmup_lr = getattr(self.config, 'adapt_warmup_lr', None)
        old_lrs = None
        if warmup_lr is not None:
            old_lrs = [pg.get('lr', None) for pg in self.adapt_optimizer.param_groups]
            for pg in self.adapt_optimizer.param_groups:
                pg['lr'] = float(warmup_lr)

        self.freeze_student_encoder(freeze_adapt=False, freeze_sa_mean_std=False)
        self.set_train()

        cprint(f'Adapt 预热: {num_steps} steps...', 'green', attrs=['bold'])
        latent_coef = float(getattr(self.config, 'adapt_latent_coef', 1.0) or 1.0)
        action_coef = float(getattr(self.config, 'adapt_action_bc_coef', 0.0) or 0.0)
        for step in range(num_steps):
            # 用当前 obs_dict 构造 extrin_pred/extrin_gt，并监督训练 adapt_tconv
            state, extrin, extrin_gt = self.get_state_from_obs(obs_dict)
            latent_loss = ((extrin - extrin_gt.detach()) ** 2).mean()

            action_loss = None
            teacher_action = self.get_teacher_action(obs_dict).contiguous()
            if action_coef > 0.0:
                obs = state[:, : self.obs_shape[0]].detach()
                pred_action = self._teacher_action_from_extrin(obs, extrin)
                action_loss = F.mse_loss(pred_action, teacher_action)

            adapt_loss = latent_coef * latent_loss
            if action_loss is not None:
                adapt_loss = adapt_loss + action_coef * action_loss
            self.adapt_optimizer.zero_grad()
            adapt_loss.backward()
            self.adapt_optimizer.step()

            # 使用 teacher 动作推进环境（保持探索分布接近 expert）
            obs_dict, _, _, _ = self.env.step(teacher_action)

            if (step + 1) % 1000 == 0:
                msg = f'Adapt 预热进度: {step + 1}/{num_steps} | loss: {float(adapt_loss.detach().cpu()):.4f}'
                msg += f' | latent: {float(latent_loss.detach().cpu()):.4f}'
                if action_loss is not None:
                    msg += f' | action: {float(action_loss.detach().cpu()):.4f}'
                tprint(msg)

        self.adapt_pretrained_env_steps += int(num_steps)

        if old_lrs is not None:
            for pg, lr in zip(self.adapt_optimizer.param_groups, old_lrs):
                if lr is not None:
                    pg['lr'] = lr
    
    def test(self):
        """测试模式（deterministic），统计固定数量 episode 后输出平均回报。"""
        self.set_eval()
        obs_dict = self.env.reset()

        target_episodes = int(getattr(self.config, 'test_num_episodes', 20) or 20)
        target_episodes = max(1, target_episodes)
        max_steps = int(getattr(self.config, 'test_max_steps', 0) or 0)
        test_print_every = int(getattr(self.config, 'test_print_every', 1) or 1)
        test_print_every = max(1, test_print_every)
        raw_realtime = getattr(self.config, 'test_realtime', False)
        if isinstance(raw_realtime, str):
            test_realtime = raw_realtime.strip().lower() in ('1', 'true', 'yes', 'on')
        else:
            test_realtime = bool(raw_realtime)
        realtime_factor = float(getattr(self.config, 'test_realtime_factor', 1.0) or 1.0)
        realtime_factor = max(realtime_factor, 1e-6)

        target_step_sec = None
        fixed_sleep_sec = getattr(self.config, 'test_sleep_sec', None)
        if fixed_sleep_sec is not None:
            target_step_sec = max(0.0, float(fixed_sleep_sec))
        elif test_realtime:
            sim_dt = float(getattr(self.env, 'dt', 0.0) or 0.0)
            control_freq_inv = float(getattr(self.env, 'control_freq_inv', 1.0) or 1.0)
            if sim_dt > 0.0 and control_freq_inv > 0.0:
                target_step_sec = (sim_dt * control_freq_inv) / realtime_factor
                cprint(
                    f'[DOTPG][TEST] realtime playback: target_step_sec={target_step_sec:.4f}, '
                    f'factor={realtime_factor:.3f}',
                    'cyan',
                )

        step_reward = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        step_length = torch.zeros(self.num_actors, dtype=torch.float32, device=self.device)
        episode_rewards = []
        episode_lengths = []
        eval_reward_sum = 0.0
        eval_done_sum = 0.0

        steps = 0
        while (
            self.test_num_steps > 0 or len(episode_rewards) < target_episodes
        ) and (max_steps <= 0 or steps < max_steps):
            loop_t0 = time.time()
            state, _, _ = self.get_state_from_obs(obs_dict)
            with torch.no_grad():
                action = self.policy(state).clamp(-1.0, 1.0).contiguous()

            obs_dict, r, done, _ = self.env.step(action)
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
            step_reward += r
            step_length += 1

            done_indices = done.nonzero(as_tuple=False)
            if done_indices.numel() > 0:
                episode_rewards.extend(step_reward[done_indices].detach().cpu().view(-1).tolist())
                episode_lengths.extend(step_length[done_indices].detach().cpu().view(-1).tolist())

            not_dones = 1.0 - done.float()
            step_reward = step_reward * not_dones
            step_length = step_length * not_dones

            steps += 1
            if (steps % test_print_every) == 0:
                cprint(
                    f'[DOTPG][TEST] step={steps} | reward(mean)={float(r.mean().item()):.3f} | done={int(done.sum().item())}',
                    'cyan',
                )
            if target_step_sec is not None and target_step_sec > 0.0:
                elapsed = time.time() - loop_t0
                if elapsed < target_step_sec:
                    time.sleep(target_step_sec - elapsed)
            if self.test_num_steps > 0 and steps >= self.test_num_steps:
                avg_reward = eval_reward_sum / float(steps)
                avg_done_rate = eval_done_sum / float(steps)
                print(
                    "EvalSummary "
                    f"steps={steps} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                )
                return

        if len(episode_rewards) == 0:
            cprint('[DOTPG][TEST] 未收集到 episode（请检查环境 reset/done 逻辑）。', 'yellow', attrs=['bold'])
            return

        rewards = np.asarray(episode_rewards[:target_episodes], dtype=np.float32)
        lengths = np.asarray(episode_lengths[:target_episodes], dtype=np.float32)
        cprint(
            f'[DOTPG][TEST] episodes={len(rewards)} | reward(mean/std)={rewards.mean():.2f}/{rewards.std():.2f} '
            f'| len(mean/std)={lengths.mean():.1f}/{lengths.std():.1f}',
            'green',
            attrs=['bold'],
        )
    
    def train(self):
        """
        主训练循环
        
        论文Algorithm 1: DOT-PG完整训练流程
        """
        _t = time.time()
        _last_t = time.time()
        
        # 重置环境
        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size

        # ==================== student-state 表征预热（无特权信息输入）====================
        if self.state_mode == 'student':
            # legacy（非 dynamic_state）下，expert buffer 与 encoder 绑定，直接复用会导致 state 不一致。
            if (not self.dynamic_state) and len(self.expert_buffer) > 0:
                cprint('[DOTPG] student-state 训练将清空已有 expert buffer 并重新采集（避免表征不一致）。', 'yellow')
                self.expert_buffer.clear()
                self.expert_collected_env_steps = 0
            # 不依赖历史计数：是否预热由配置决定
            self.adapt_pretrained_env_steps = 0
            
            adapt_warmup_steps = int(getattr(self.config, 'adapt_warmup_steps', 0) or 0)
            if adapt_warmup_steps > 0:
                # 不尝试“跳过”预热：除非同时恢复了同一份 student encoder，否则历史计数不可靠
                cprint(
                    f'[DOTPG] adapt_env_steps: 0 -> {adapt_warmup_steps} (pretrain {adapt_warmup_steps} steps)',
                    'green',
                    attrs=['bold'],
                )
                self.pretrain_adapt_tconv(obs_dict, adapt_warmup_steps)
                obs_dict = self.env.reset()

            if bool(getattr(self.config, 'freeze_adapt_after_warmup', False)):
                freeze_sa = bool(getattr(self.config, 'freeze_sa_mean_std_after_warmup', False))
                self.freeze_student_encoder(freeze_adapt=True, freeze_sa_mean_std=freeze_sa)
        
        # 收集专家数据（使用教师模型）
        warmup_steps = int(getattr(self.config, 'warmup_steps', 0) or 0)
        env_steps_have = int(getattr(self, 'expert_collected_env_steps', 0) or 0)
        if warmup_steps > 0 and env_steps_have < warmup_steps:
            steps_to_collect = int(warmup_steps - env_steps_have)
            cprint(
                f'[DOTPG] expert_env_steps: {env_steps_have} -> {warmup_steps} (collect {steps_to_collect} steps)',
                'green',
                attrs=['bold'],
            )
            self.collect_expert_data(obs_dict, steps_to_collect)
            obs_dict = self.env.reset()

        # 可选：先用专家数据做 BC 预热，让 policy 从一开始就接近 expert
        self.pretrain_policy_bc()

        online_expert = bool(getattr(self.config, 'online_expert', False))
        online_add_num_envs = getattr(self.config, 'online_expert_add_num_envs', None)
        if online_add_num_envs is None:
            online_add_num_envs = getattr(self.config, 'expert_add_num_envs', None)
        if online_add_num_envs is None:
            online_add_num_envs = 0
        online_add_num_envs = int(online_add_num_envs) if isinstance(online_add_num_envs, (int, float, np.integer)) else 0
        if online_add_num_envs <= 0 or online_add_num_envs >= self.num_actors:
            online_add_num_envs = self.num_actors
        
        # 主训练循环
        while self.agent_steps <= self.max_agent_steps:
            # 获取当前状态
            state, extrin, extrin_gt = self.get_state_from_obs(obs_dict)
            
            adapt_loss = None
            if self.state_mode == 'student' and not self._adapt_frozen:
                # ==================== adapt_tconv训练（学生latent预测）====================
                latent_coef = float(getattr(self.config, 'adapt_latent_coef', 1.0) or 1.0)
                action_coef = float(getattr(self.config, 'adapt_action_bc_coef', 0.0) or 0.0)
                latent_loss = ((extrin - extrin_gt.detach()) ** 2).mean()

                action_loss = None
                if action_coef > 0.0:
                    with torch.no_grad():
                        teacher_action = self.get_teacher_action(obs_dict).contiguous()
                    obs = state[:, : self.obs_shape[0]].detach()
                    pred_action = self._teacher_action_from_extrin(obs, extrin)
                    action_loss = F.mse_loss(pred_action, teacher_action)

                adapt_loss = latent_coef * latent_loss
                if action_loss is not None:
                    adapt_loss = adapt_loss + action_coef * action_loss

                self.adapt_optimizer.zero_grad()
                adapt_loss.backward()
                self.adapt_optimizer.step()

                # 重新计算state（因为adapt_tconv已更新）
                state, extrin, extrin_gt = self.get_state_from_obs(obs_dict)

            state = state.detach()
            
            # ==================== 选择动作 ====================
            with torch.no_grad():
                # 添加探索噪声
                action = self.policy(state)
                if self.config.exploration_noise > 0:
                    noise = torch.randn_like(action) * self.config.exploration_noise
                    action = (action + noise).clamp(-1.0, 1.0)
                action = action.contiguous()
            
            # ==================== 环境交互 ====================
            next_obs_dict, r, done, info = self.env.step(action)

            # 存储到回放缓冲区
            if self.dynamic_state:
                self.replay_buffer.add_batch(
                    obs_dict['obs'].detach(),
                    obs_dict['proprio_hist'].detach(),
                    action.detach(),
                    next_obs_dict['obs'].detach(),
                    next_obs_dict['proprio_hist'].detach(),
                    done.float().unsqueeze(1),
                )
            else:
                next_state, _, _ = self.get_state_from_obs(next_obs_dict)
                next_state = next_state.detach()
                self.replay_buffer.add_batch(state, action, next_state, done.float().unsqueeze(1))
            
            # 可选：在线追加专家数据（DAgger 风格）。论文若要求纯离线 expert，保持 online_expert=False。
            if online_expert:
                teacher_action = (
                    self.get_teacher_action_from_state(state)
                    if self.state_mode == 'teacher'
                    else self.get_teacher_action(obs_dict)
                )
                if self.dynamic_state:
                    obs = obs_dict['obs'].detach()
                    proprio_hist = obs_dict['proprio_hist'].detach()
                    if online_add_num_envs == self.num_actors:
                        self.expert_buffer.add_batch(obs, proprio_hist, teacher_action.detach())
                    else:
                        idx = torch.randperm(self.num_actors, device=obs.device)[:online_add_num_envs]
                        self.expert_buffer.add_batch(obs[idx], proprio_hist[idx], teacher_action[idx].detach())
                else:
                    if online_add_num_envs == self.num_actors:
                        self.expert_buffer.add_batch(state, teacher_action.detach())
                    else:
                        idx = torch.randperm(self.num_actors, device=state.device)[:online_add_num_envs]
                        self.expert_buffer.add_batch(state[idx], teacher_action[idx].detach())
            
            # ==================== DOTPG训练步骤 ====================
            updates_per_env_step = max(1, int(getattr(self.config, 'updates_per_env_step', 1) or 1))
            metrics_accum = {}
            metrics_count = 0
            for _ in range(updates_per_env_step):
                step_metrics = self.train_step()
                if step_metrics is None:
                    continue
                metrics_count += 1
                for key, value in step_metrics.items():
                    if value is not None:
                        metrics_accum[key] = metrics_accum.get(key, 0.0) + float(value)
            metrics = (
                {key: value / metrics_count for key, value in metrics_accum.items()}
                if metrics_count > 0
                else None
            )
            
            # 更新观察
            obs_dict = next_obs_dict
            self.agent_steps += self.batch_size
            
            # ==================== 统计信息 ====================
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            
            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones
            
            # 记录训练信息
            if adapt_loss is not None:
                self.direct_info['adapt_loss'] = float(adapt_loss.detach().cpu())
            if metrics is not None:
                self.direct_info.update(metrics)
            
            # 日志记录
            self.log_tensorboard()
            
            # 保存模型
            if self.agent_steps % int(1e8) == 0:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // int(1e8)}00m'))
                self.save(os.path.join(self.nn_dir, f'model_last'))
            
            # 保存最佳模型
            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, f'model_best'))
                self.best_rewards = mean_rewards
            
            # 打印进度
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)

        self.save(os.path.join(self.nn_dir, f'model_last'))
        cprint(
            f'[DOTPG] max_agent_steps reached: {self.agent_steps} >= {self.max_agent_steps}',
            'green',
            attrs=['bold'],
        )
    
    def log_tensorboard(self):
        """记录到TensorBoard"""
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            if v is not None:
                self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)
    
    def restore_train(self, fn):
        """
        从检查点恢复训练
        
        Args:
            fn: 检查点文件路径
        """
        if not fn:
            return
        
        checkpoint = torch.load(fn)
        cprint('加载教师模型检查点...', 'yellow', attrs=['bold'])
        
        # 加载教师模型
        self.teacher_model.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        
        if 'priv_mean_std' in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud and 'point_cloud_mean_std' in checkpoint:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
        
        # 复制教师模型的adapt_tconv和point_mlp
        if hasattr(self.teacher_model, 'adapt_tconv') and self.proprio_adapt:
            self.adapt_tconv.load_state_dict(self.teacher_model.adapt_tconv.state_dict())
        if self.use_point_cloud_info and hasattr(self.teacher_model, 'point_mlp'):
            self.point_mlp.load_state_dict(self.teacher_model.point_mlp.state_dict())
        self._maybe_init_policy_from_teacher()
        
        cprint('教师模型加载完成', 'green', attrs=['bold'])

        resume_path = str(getattr(self.config, 'resume_path', '') or '').strip()
        if resume_path:
            self.restore_student_resume(resume_path)

    def restore_student_resume(self, fn):
        """继续 DOTPG 训练时加载 student/critic/optimizer/normalizer 状态。"""
        if not fn:
            return
        if not os.path.isfile(fn):
            raise FileNotFoundError(f"DOTPG resume_path not found: {fn}")

        cprint(f'加载 DOTPG student 续训检查点: {fn}', 'yellow', attrs=['bold'])
        checkpoint = torch.load(fn, map_location=self.device)

        if 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
        if 'policy_target' in checkpoint:
            self.policy_target.load_state_dict(checkpoint['policy_target'])
        else:
            self.policy_target.load_state_dict(self.policy.state_dict())
        if 'dual' in checkpoint:
            self.dual.load_state_dict(checkpoint['dual'])
        if 'q1' in checkpoint:
            self.q1.load_state_dict(checkpoint['q1'])
        if 'q2' in checkpoint:
            self.q2.load_state_dict(checkpoint['q2'])
        if 'q1_target' in checkpoint:
            self.q1_target.load_state_dict(checkpoint['q1_target'])
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if 'q2_target' in checkpoint:
            self.q2_target.load_state_dict(checkpoint['q2_target'])
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())
        if 'adapt_tconv' in checkpoint:
            self.adapt_tconv.load_state_dict(checkpoint['adapt_tconv'])
        if self.use_point_cloud_info and 'point_mlp' in checkpoint:
            self.point_mlp.load_state_dict(checkpoint['point_mlp'])

        if 'running_mean_std' in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'sa_mean_std' in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        if 'priv_mean_std' in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if 'point_cloud_mean_std' in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
        if 'dual_reward_mean_std' in checkpoint:
            self.dual_reward_mean_std.load_state_dict(checkpoint['dual_reward_mean_std'])

        if bool(getattr(self.config, 'resume_load_optimizers', True)):
            if 'policy_optimizer' in checkpoint:
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
            if 'dual_optimizer' in checkpoint:
                self.dual_optimizer.load_state_dict(checkpoint['dual_optimizer'])
            if 'q_optimizer' in checkpoint:
                self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
            if 'adapt_optimizer' in checkpoint:
                self.adapt_optimizer.load_state_dict(checkpoint['adapt_optimizer'])

        if 'total_it' in checkpoint:
            self.total_it = int(checkpoint['total_it'])
        if 'agent_steps' in checkpoint:
            self.agent_steps = int(checkpoint['agent_steps'])
        if 'best_rewards' in checkpoint:
            self.best_rewards = float(checkpoint['best_rewards'])

        cprint(
            f'DOTPG student 续训检查点加载完成: agent_steps={self.agent_steps}, '
            f'total_it={self.total_it}, best_rewards={self.best_rewards:.2f}',
            'green',
            attrs=['bold'],
        )
    
    def restore_test(self, fn):
        """
        加载模型用于测试
        
        Args:
            fn: 模型文件路径
        """
        if not fn:
            return
        
        checkpoint = torch.load(fn)
        
        # 加载DOTPG网络
        if 'policy' in checkpoint:
            self.policy.load_state_dict(checkpoint['policy'])
        if 'policy_target' in checkpoint:
            self.policy_target.load_state_dict(checkpoint['policy_target'])
        else:
            self.policy_target.load_state_dict(self.policy.state_dict())
        if 'dual' in checkpoint:
            self.dual.load_state_dict(checkpoint['dual'])
        if 'q1' in checkpoint:
            self.q1.load_state_dict(checkpoint['q1'])
        if 'q2' in checkpoint:
            self.q2.load_state_dict(checkpoint['q2'])
        if 'q1_target' in checkpoint:
            self.q1_target.load_state_dict(checkpoint['q1_target'])
        else:
            self.q1_target.load_state_dict(self.q1.state_dict())
        if 'q2_target' in checkpoint:
            self.q2_target.load_state_dict(checkpoint['q2_target'])
        else:
            self.q2_target.load_state_dict(self.q2.state_dict())
        if 'adapt_tconv' in checkpoint:
            self.adapt_tconv.load_state_dict(checkpoint['adapt_tconv'])
        
        # 加载归一化模块
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'sa_mean_std' in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        if 'dual_reward_mean_std' in checkpoint:
            self.dual_reward_mean_std.load_state_dict(checkpoint['dual_reward_mean_std'])
        if self.normalize_point_cloud and 'point_cloud_mean_std' in checkpoint:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])
    
    def save(self, name):
        """
        保存模型
        
        Args:
            name: 保存路径（不含扩展名）
        """
        weights = {
            # DOTPG网络
            'policy': self.policy.state_dict(),
            'policy_target': self.policy_target.state_dict(),
            'dual': self.dual.state_dict(),
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'q1_target': self.q1_target.state_dict(),
            'q2_target': self.q2_target.state_dict(),
            'adapt_tconv': self.adapt_tconv.state_dict(),
            # 优化器
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'dual_optimizer': self.dual_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'adapt_optimizer': self.adapt_optimizer.state_dict(),
            # 归一化模块
            'running_mean_std': self.running_mean_std.state_dict(),
            'sa_mean_std': self.sa_mean_std.state_dict(),
            'dual_reward_mean_std': self.dual_reward_mean_std.state_dict(),
            # 训练状态
            'total_it': self.total_it,
            'agent_steps': self.agent_steps,
            'best_rewards': self.best_rewards,
        }
        
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        if self.use_point_cloud_info:
            weights['point_mlp'] = self.point_mlp.state_dict()
        
        torch.save(weights, f'{name}.ckpt')
        cprint(f'模型已保存到 {name}.ckpt', 'green')
