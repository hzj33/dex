"""E003 student whose entire runtime input is hand proprioception.

The frozen teacher actor is retained. A new history encoder predicts both its
phase command and latent; neither is accepted as a runtime input. This module
has no IsaacGym dependency and is also the exported real-hand control step.
"""
import math
import torch
from torch import nn

TEACHER_SHA = 'd42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'
HISTORY = 80


class DeployStudent(nn.Module):
    def __init__(self, checkpoint, init_pose, action_scale=0.05, return_entry_clock=False):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(141, 512), nn.ELU(),
                                   nn.Linear(512, 256), nn.ELU(),
                                   nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 16))
        state = checkpoint['model']
        for i in (0, 2, 4):
            self.actor[i].load_state_dict({k: state[f'actor_mlp.mlp.{i}.{k}'] for k in ('weight', 'bias')})
        self.actor[6].load_state_dict({k: state['mu.' + k] for k in ('weight', 'bias')})
        self.actor.requires_grad_(False)
        rms = checkpoint['running_mean_std']
        self.register_buffer('obs_mean', rms['running_mean'].float().clone())
        self.register_buffer('obs_var', rms['running_var'].float().clone())
        assert self.obs_mean.shape == (101,)
        self.encoder = nn.Sequential(nn.Linear(80 * 32 + 96, 256), nn.ELU(),
                                     nn.Linear(256, 128), nn.ELU(), nn.Linear(128, 45))
        self.register_buffer('init_pose', torch.as_tensor(init_pose, dtype=torch.float32)[:16].clone())
        self.register_buffer('lower', torch.tensor([-math.pi / 9, 0., 0., 0.] * 4))
        self.register_buffer('upper', torch.tensor([math.pi / 9, 1.57, 1.57, 1.57] * 4))
        self.register_buffer('action_mask', torch.tensor([1.] * 4 + [0.] * 8 + [1.] * 4))
        assert action_scale in (0.02, 0.05)
        self.action_scale = float(action_scale)
        self.policy_hz = 20.0
        # Explicit local control adaptation: keep return's entry command while
        # learned phase and real feedback determine when to leave return.
        # This never resets a timer, history, targets or measured positions.
        self.return_entry_clock = bool(return_entry_clock)

    def predict(self, obs96, history):
        # Frozen teacher joint statistics also normalize the longer history.
        x = ((obs96 - self.obs_mean[:96]) / torch.sqrt(self.obs_var[:96] + 1e-5)).clamp(-5, 5)
        h = ((history - self.obs_mean[64:96]) / torch.sqrt(self.obs_var[64:96] + 1e-5)).clamp(-5, 5)
        features = self.encoder(torch.cat((x, h.flatten(1)), -1))
        logits = features[:, :4]
        clock = features[:, 4:5].sigmoid()
        if self.return_entry_clock:
            clock = torch.where(logits.argmax(-1, keepdim=True) == 2,
                                torch.zeros_like(clock), clock)
        latent = features[:, 5:].tanh()
        command = torch.cat((logits.softmax(-1), clock), -1)
        obs101 = torch.cat((obs96, command), -1)
        normalized = ((obs101 - self.obs_mean) / torch.sqrt(self.obs_var + 1e-5)).clamp(-5, 5)
        raw = self.actor(torch.cat((normalized, latent), -1))
        # The selected controller profile is explicit and is used in collection,
        # simulation evaluation and the exported integrator without post-hoc caps.
        action = (raw.clamp(-1, 1) * (0.05 / self.action_scale)).clamp(-1, 1) * self.action_mask
        return action, logits, clock, latent

    @torch.jit.export
    def reset_history(self, q, previous_target):
        return torch.cat((q, previous_target), -1).unsqueeze(1).repeat(1, 80, 1)

    def forward(self, q, previous_target, history):
        # Match original M24: observation precedes append, adaptation follows it.
        obs96 = history[:, -3:].reshape(-1, 96)
        frame = torch.cat((q, previous_target), -1).unsqueeze(1)
        updated = torch.cat((history[:, 1:], frame), 1)
        action, _, _, _ = self.predict(obs96, updated)
        target = torch.maximum(torch.minimum(previous_target + self.action_scale * action, self.upper), self.lower)
        return target, updated, action
