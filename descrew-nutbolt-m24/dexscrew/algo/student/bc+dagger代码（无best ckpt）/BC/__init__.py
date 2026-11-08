"""Pure Behavior Cloning student baseline.

纯 BC（非 DAgger / 非 BCO）：离线、固定 teacher demonstrations + 监督学习。
与 DOTPGStudent / DAggerStudent 同构，仅替换监督训练阶段。
"""

from dexscrew.algo.BC.bc import BCStudent, BCConfig
from dexscrew.algo.BC.buffer import BCBuffer

__all__ = ["BCStudent", "BCConfig", "BCBuffer"]
