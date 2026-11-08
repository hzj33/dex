"""Independent, bounded initial-joint perturbations for E002 robustness.

Task-parameter randomization follows DemoStart's target-distribution framing
(https://arxiv.org/html/2409.06613v1, III-A). Joint-only bounded uniform noise
and a nominal mixture are local adaptations, not its automatic curriculum.
"""
import torch


def perturb_initial_joints(pos, lower, upper, half_width_rad, nominal_fraction):
    assert pos.ndim == 2 and pos.shape[1] == 16
    assert half_width_rad >= 0 and 0 <= nominal_fraction <= 1
    if half_width_rad == 0:
        return pos  # Preserve the original path and RNG sequence exactly.
    noise = (torch.rand_like(pos) * 2 - 1) * half_width_rad
    noise[:, 4:12] = 0  # Middle/ring remain at the user's fixed pose.
    nominal = torch.rand(pos.shape[0], device=pos.device) < nominal_fraction
    noise[nominal] = 0
    perturbed = torch.maximum(torch.minimum(pos + noise, upper), lower)
    perturbed[:, 4:12] = pos[:, 4:12]
    perturbed[nominal] = pos[nominal]
    return perturbed
