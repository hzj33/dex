"""
DOT-PG: Dual Optimal Transport Policy Gradient

基于论文 "Bridging the Reality Gap: Dual Optimal Transport Policy Gradient"

模块组成:
- dotpg.py: DOTPG学生模型主类
- networks.py: DOTPG网络架构（Policy, Dual, Q）
- buffer.py: 经验回放和专家数据缓冲区
"""

from dexscrew.dotpg.dotpg import DOTPGStudent, DOTPGConfig
from dexscrew.dotpg.networks import PolicyNetwork, DualNetwork, QNetwork, DOTPGNetworks
from dexscrew.dotpg.buffer import ReplayBuffer, ExpertBuffer

DOTPG = DOTPGStudent

__all__ = [
    'DOTPG',
    'DOTPGStudent',
    'DOTPGConfig',
    'PolicyNetwork',
    'DualNetwork',
    'QNetwork',
    'DOTPGNetworks',
    'ReplayBuffer',
    'ExpertBuffer'
]
