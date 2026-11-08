"""Experimental deployment-only release cue from measured index abduction.

No object/contact/score access. This is a pose743-specific hypothesis, not a
general tightening detector. The original actor, latent and feedback remain.
"""


class ReleaseGuard:
    def __init__(self, threshold=-0.10, rearm=0.10, pulse_steps=20, min_flexion=0.0, reset_at=None):
        assert -0.30 <= threshold < rearm <= 0.30
        assert 1 <= pulse_steps <= 20
        self.threshold, self.rearm = float(threshold), float(rearm)
        self.pulse_steps = pulse_steps
        assert 0 <= min_flexion <= 1.57
        self.min_flexion = float(min_flexion)
        self.armed = True
        self.cooldown = 0
        self.triggers = 0
        self.reset_at = reset_at
        self.awaiting_reset = False

    def update(self, measured_q):
        self.cooldown = max(0, self.cooldown-1)
        if measured_q[0] >= self.rearm and self.cooldown == 0:
            self.armed = True
        fire = (self.armed and self.cooldown == 0 and measured_q[0] <= self.threshold
                and measured_q[1] >= self.min_flexion)
        if fire:
            self.armed = False
            self.cooldown = self.pulse_steps
            self.triggers += 1
            self.awaiting_reset = True
        return bool(fire)

    def consume_reset(self, measured_q):
        reset = (self.reset_at is not None and self.awaiting_reset and self.cooldown == 0
                 and measured_q[0] >= self.reset_at)
        if reset:
            self.awaiting_reset = False
        return bool(reset)


def release_guard_loop_class(base):
    from m24_e003_release_override import manual_loop_class
    manual = manual_loop_class(base)
    class GuardedLoop(manual):
        def __init__(self, policy, threshold=-.10, pulse_steps=20, min_flexion=0.0, reset_at=None):
            self.guard = ReleaseGuard(threshold=threshold, pulse_steps=pulse_steps, min_flexion=min_flexion, reset_at=reset_at)
            self.fire = False
            super().__init__(policy, lambda: self.fire, pulse_steps)

        def step(self, measured_q, policy_hz):
            self.fire = self.guard.update(measured_q)
            reset = self.guard.consume_reset(measured_q)
            if reset:
                import torch
                q = torch.as_tensor(measured_q, dtype=torch.float32).reshape(1, 16)
                self.history = self.model.reset_history(q, self.previous)
            result = super().step(measured_q, policy_hz)
            result['release_guard'] = dict(threshold=self.guard.threshold,
                triggers=self.guard.triggers, armed=self.guard.armed,
                source='measured_index_0', object_information_used=False, history_reset=reset)
            return result
    return GuardedLoop
