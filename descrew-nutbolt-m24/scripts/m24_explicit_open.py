"""Opt-in diagnostic opening trajectory; no state estimation or task success claim."""
import numpy as np

ACTIVE = np.array([0, 1, 2, 12, 13, 14, 15])
OPEN_JOINTS = np.array([1, 14])
OPEN_DELTA = 0.30
MAX_STEP = 0.025  # rad / 50 ms, no faster than identified proximal velocity.


class ExplicitOpenController:
    def __init__(self):
        self.goal = None
        self.steps = 0

    def reset(self):
        self.goal = None
        self.steps = 0

    def update(self, measured, previous, requested, lower, upper):
        q, previous = np.asarray(measured), np.asarray(previous)
        if not np.isfinite(q).all() or not np.isfinite(previous).all():
            raise ValueError('Nonfinite opening-controller input')
        accepted = bool(requested and self.goal is None)
        if accepted:
            self.goal = previous.copy()
            # Stop chasing stale targets on other active joints; preserve
            # passive index_3 and masked fingers exactly as sent previously.
            self.goal[ACTIVE] = q[ACTIVE]
            self.goal[OPEN_JOINTS] -= OPEN_DELTA
            self.goal = np.clip(self.goal, lower, upper)
        event = dict(requested=bool(requested), accepted=accepted,
                     active=self.goal is not None, mode='policy', steps=self.steps,
                     measured_joints_modified=False, automatic_resume=False)
        if self.goal is None:
            return None, event
        target = previous + np.clip(self.goal-previous, -MAX_STEP, MAX_STEP)
        target = np.clip(target, lower, upper)
        self.steps += 1
        at_target = bool(np.max(np.abs(target-self.goal)) < 1e-6)
        event.update(mode='hold' if at_target else 'opening', steps=self.steps,
                     goal=self.goal.tolist(),
                     measured_goal_error_rad=float(np.max(np.abs(q[ACTIVE]-self.goal[ACTIVE]))))
        return target, event


def explicit_open_loop_class(base):
    class ExplicitOpenLoop(base):
        def __init__(self, policy, request):
            super().__init__(policy)
            self.request = request
            self.opening = ExplicitOpenController()

        def step(self, measured_q, policy_hz):
            import torch
            previous = self.previous.numpy()[0].copy()
            result = super().step(measured_q, policy_hz)
            target, event = self.opening.update(result['q'], previous, self.request(),
                self.model.lower.numpy(), self.model.upper.numpy())
            if target is not None:
                action = (target-previous)/float(self.model.action_scale)
                self.previous = torch.as_tensor(target, dtype=torch.float32).reshape(1, 16)
                result.update(next_target=target, action=action)
            result['explicit_open'] = event
            return result
    return ExplicitOpenLoop
