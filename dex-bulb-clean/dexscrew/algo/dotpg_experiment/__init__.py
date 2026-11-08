"""
DOTPG ablation experiment implementation.

This package keeps experiment-only switches out of the deployable DOTPG code path.
"""

from dexscrew.algo.dotpg_experiment.dotpg_experiment import DOTPGExperiment, DOTPGExperimentConfig
from dexscrew.algo.dotpg_experiment.networks import PolicyNetwork, DualNetwork, QNetwork, DOTPGNetworks

__all__ = [
    'DOTPGExperiment',
    'DOTPGExperimentConfig',
    'PolicyNetwork',
    'DualNetwork',
    'QNetwork',
    'DOTPGNetworks',
]
