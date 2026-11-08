"""Bounded manual phase intervention, keeping measured joints and history intact."""
import torch


@torch.no_grad()
def phase_action(model, obs96, history, command):
    _, _, _, latent = model.predict(obs96, history)
    normalized = ((torch.cat((obs96, command), -1) - model.obs_mean) /
                  torch.sqrt(model.obs_var + 1e-5)).clamp(-5, 5)
    raw = model.actor(torch.cat((normalized, latent), -1))
    return (raw.clamp(-1, 1) * (0.05 / model.action_scale)).clamp(-1, 1) * model.action_mask


def manual_loop_class(base):
    class ManualReleaseLoop(base):
        def __init__(self, policy, request, pulse_steps=20):
            super().__init__(policy)
            if not 1 <= pulse_steps <= 20:
                raise ValueError('Manual release must last 1..20 control steps')
            self.request = request
            self.pulse_steps = pulse_steps
            self.remaining = 0

        @torch.no_grad()
        def step(self, measured_q, policy_hz):
            previous = self.previous.clone()
            history = self.history if self.history is not None else self.model.reset_history(previous, previous)
            obs96 = history[:, -3:].reshape(-1, 96).clone()
            requested = bool(self.request())
            accepted = requested and self.remaining == 0
            if accepted:
                self.remaining = self.pulse_steps
            result = super().step(measured_q, policy_hz)
            _, logits, clock, _ = self.model.predict(obs96, self.history)
            command = torch.cat((logits.softmax(-1), clock), -1)
            active = self.remaining > 0
            if active:
                command[:, :4] = 0
                command[:, 1] = 1
                action = phase_action(self.model, obs96, self.history, command)
                target = torch.maximum(torch.minimum(previous + self.model.action_scale * action,
                                                     self.model.upper), self.model.lower)
                if not torch.isfinite(target).all():
                    raise ValueError('Nonfinite manual-release target')
                self.previous = target
                result['action'] = action.numpy()[0]
                result['next_target'] = target.numpy()[0]
                self.remaining -= 1
            result['manual_release'] = dict(requested=requested, accepted=accepted,
                active=active, remaining_steps=self.remaining,
                predicted_phase_probabilities=logits.softmax(-1)[0].tolist(),
                actor_phase_command=command[0].tolist(), measured_joints_modified=False)
            return result
    return ManualReleaseLoop
