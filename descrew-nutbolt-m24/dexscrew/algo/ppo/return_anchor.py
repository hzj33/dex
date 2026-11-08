"""Optional phase-specific frozen-policy auxiliary target for E003 PPO.

Kickstarting (Schmitt et al., 2018) motivates auxiliary teacher guidance.
Return-only masking and MSE are local adaptations. Optional Gaussian KL uses
the analytic teacher-to-student KL, averaged over active action dimensions.
"""
from copy import deepcopy
import torch

class ReturnAnchor:
    def __init__(self, agent):
        assert agent.obs_shape == (101,) and agent.actions_num == 16
        self.model = deepcopy(agent.model).eval().requires_grad_(False)
        self.obs_norm = deepcopy(agent.running_mean_std).eval().requires_grad_(False)
        self.priv_norm = deepcopy(agent.priv_mean_std).eval().requires_grad_(False)
        self.point_norm = deepcopy(agent.point_cloud_mean_std).eval().requires_grad_(False)
        self.normalize_priv = agent.normalize_priv
        self.normalize_point_cloud = agent.normalize_point_cloud

    @torch.no_grad()
    def target(self, obs, priv, points, history):
        # Mask must use raw phase one-hot, never normalized observations.
        mask = obs[:, 98] > .5
        if not mask.any():return mask, None
        pc = points[mask]
        if self.normalize_point_cloud:pc = self.point_norm(pc.reshape(-1,3)).reshape_as(pc)
        batch = dict(obs=self.obs_norm(obs[mask]),priv_info=self.priv_norm(priv[mask]) if self.normalize_priv else priv[mask],point_cloud_info=pc,proprio_hist=history[mask])
        return mask, self.model.act_inference(batch)[0]

    def loss(self, mu, mask, target, sigma=None, mode="mse"):
        assert mode in ("mse", "kl")
        if target is None:return mu.sum()*0
        active = [0,1,2,3,12,13,14,15]
        delta = mu[mask][:,active]-target[:,active]
        if mode == "mse":return delta.square().mean()
        assert sigma is not None
        ref_sigma = self.model.sigma.exp()[active].detach()
        new_sigma = sigma.expand_as(mu)[mask][:,active].clamp_min(1e-8)
        return (torch.log(new_sigma/ref_sigma) + (ref_sigma.square()+delta.square())/(2*new_sigma.square()) - .5).mean()
