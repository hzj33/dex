"""Rejected automatic-cue experiment; do not enable for deployment.

v1/v2 failed broad recovery checks. Retained only to reproduce diagnostics.
Joint-observable, bounded progress correction; no phase/FK oracle.

Two signs justify a single 20-step correction: a persistent backward phase
transition before the next drive, or saturated return progress with measured
joint stagnation and nonzero commands blocked by SDK limits. Keep the phase,
latent, measured joints, action scale and actual target history untouched.
"""
from collections import deque
import numpy as np


class ProgressRecovery:
    def __init__(self, pulse_steps=20):
        assert 1 <= pulse_steps <= 20
        self.pulse_steps = pulse_steps
        self.remaining = 0
        self.steps = 0
        self.candidate = -1
        self.candidate_count = 0
        self.stable_phase = -1
        self.armed = True
        self.triggers = 0
        self.last_trigger = -1000
        self.q_window = deque(maxlen=15)
        self.blocked_window = deque(maxlen=15)
        self.reason = None
        self.backward_pending = False

    def update(self, q, previous, action, probabilities, clock, lower, upper, scale=.05):
        q, previous, action = map(np.asarray, (q, previous, action))
        prob = np.asarray(probabilities)
        if not all(np.isfinite(x).all() for x in (q, previous, action, prob)) or not np.isfinite(clock):
            raise ValueError('Nonfinite progress-recovery input')
        phase = int(prob.argmax())
        if phase == self.candidate:
            self.candidate_count += 1
        else:
            self.candidate = phase
            self.candidate_count = 1
        target = np.clip(previous+scale*action, lower, upper)
        blocked = (abs(target-previous)<1e-7) & (abs(action)>.05)
        self.q_window.append(q.copy())
        self.blocked_window.append(blocked.copy())
        stationary = False
        if len(self.q_window) == 15:
            active = [0, 1, 2, 12, 13, 14, 15]
            stationary = bool(np.max(np.ptp(np.array(self.q_window)[:, active], axis=0)) < np.deg2rad(1.))
        blocked_count = int(np.sum(np.array(self.blocked_window).mean(0)[[0, 12, 13, 15]]>.8))
        # Require five agreeing predictions; isolated estimator flips are ignored.
        backward = (self.stable_phase == 3 and phase in (1, 2)
                    and self.candidate_count >= 5 and prob[phase]>.9)
        if backward:
            self.backward_pending = True
        stalled = (phase == 2 and self.candidate_count >= 20 and prob[2]>.95
                   and clock>.95 and stationary and blocked_count >= 2)
        if self.candidate_count >= 5:
            self.stable_phase = phase
        if phase == 0 and self.candidate_count >= 5 and self.remaining == 0:
            self.armed = True
            self.backward_pending = False
        backward_ready = self.backward_pending and phase in (1, 2) and clock >= .28 and prob[phase]>.9
        fire = bool(self.armed and self.remaining == 0 and self.steps-self.last_trigger >= 60
                    and (backward_ready or stalled))
        if fire:
            self.remaining = self.pulse_steps
            self.armed = False
            self.last_trigger = self.steps
            self.triggers += 1
            self.reason = 'persistent_backward_phase' if backward_ready else 'return_stagnation_at_limits'
            self.backward_pending = False
        active = self.remaining > 0
        if active:
            self.remaining -= 1
        self.steps += 1
        return active, dict(fired=fire, active=active, reason=self.reason if active else None,
            remaining_steps=self.remaining, triggers=self.triggers,
            stationary=stationary, blocked_joints=blocked_count,
            measured_joints_modified=False, history_reset=False,
            oracle_information_used=False, predicted_phase=phase, predicted_clock=float(clock))


def recovery_loop_class(base):
    class RecoveringLoop(base):
        def __init__(self, policy):
            super().__init__(policy)
            self.recovery = ProgressRecovery()

        def step(self, measured_q, policy_hz):
            import torch
            from m24_e003_release_override import phase_action
            previous = self.previous.clone()
            history = self.history if self.history is not None else self.model.reset_history(previous, previous)
            obs = history[:, -3:].reshape(1, 96).clone()
            result = super().step(measured_q, policy_hz)
            _, logits, clock, _ = self.model.predict(obs, self.history)
            prob = logits.softmax(-1)
            active, event = self.recovery.update(result['q'], previous.numpy()[0], result['action'],
                prob[0].numpy(), float(clock[0, 0]), self.model.lower.numpy(), self.model.upper.numpy(),
                float(self.model.action_scale))
            if active:
                command = torch.cat((prob, clock), -1)
                command[:, 4] = 0
                action = phase_action(self.model, obs, self.history, command)
                target = torch.maximum(torch.minimum(previous+self.model.action_scale*action,
                    self.model.upper), self.model.lower)
                self.previous = target
                result.update(action=action.numpy()[0], next_target=target.numpy()[0])
            result['progress_recovery'] = event
            return result
    return RecoveringLoop
