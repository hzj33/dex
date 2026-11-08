"""rl_games squared soft-bound cost, locally restricted to active nonreturn actions."""
import torch


def nonreturn_active_bound_loss(mu, nonreturn_mask):
    assert mu.ndim == 2 and mu.shape[1] == 16
    assert nonreturn_mask.shape == mu.shape[:1] and nonreturn_mask.dtype == torch.bool
    active = mu[:, [0, 1, 2, 3, 12, 13, 14, 15]]
    return (active.abs() - 1.1).clamp(min=0).square().sum(-1) * nonreturn_mask
