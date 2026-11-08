#!/usr/bin/env python3
"""Evaluate a hardware-observable release cue, with original physical gates."""
import json
import os
from pathlib import Path
import evaluate_m24_e003_real_exec as execution
import m24_e003_deploy_runtime as runtime
from m24_release_guard import ReleaseGuard
from m24_e003_release_override import phase_action

if os.environ.get('E003_COUPLING_FIT'):
    import evaluate_m24_e003_passive_diagnostic
torch = execution.torch
threshold = float(os.environ.get('E003_GUARD_THRESHOLD', '-0.10'))
pulse_steps = int(os.environ.get('E003_GUARD_PULSE', '20'))
min_flexion = float(os.environ.get('E003_GUARD_MIN_FLEXION', '0'))
reset_at = float(os.environ['E003_GUARD_RESET_AT']) if os.environ.get('E003_GUARD_RESET_AT') else None
records = []


class GuardedInference(runtime.DeployInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guards = [ReleaseGuard(threshold, pulse_steps=pulse_steps, min_flexion=min_flexion, reset_at=reset_at) for _ in range(self.env.num_envs)]
        self.remaining = torch.zeros(self.env.num_envs, dtype=torch.long, device=self.inference_device)

    @torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        if self.history is None:
            self.history = self.model.reset_history(frame[:, 16:], frame[:, 16:])
        measured = frame[:, :16].cpu().numpy()
        fired = [g.update(q) for g, q in zip(self.guards, measured)]
        reset = [g.consume_reset(q) for g, q in zip(self.guards, measured)]
        mask = torch.tensor(reset, device=self.inference_device)
        if mask.any():
            self.history[mask] = self.model.reset_history(frame[mask, :16], frame[mask, 16:])
        prior_obs = self.history[:, -3:].reshape(-1, 96).clone()
        action, extra = super()._eval_select_action(obs)
        fire = torch.tensor(fired, device=self.inference_device)
        self.remaining[fire] = pulse_steps
        active = self.remaining > 0
        _, logits, clock, _ = self.model.predict(prior_obs, self.history)
        command = torch.cat((logits.softmax(-1), clock), -1)
        command[active, :4] = 0
        command[active, 1] = 1
        alternate = phase_action(self.model, prior_obs, self.history, command)
        action = torch.where(active[:, None], alternate, action)
        self.remaining[active] -= 1
        records.append(dict(fired=fired, history_reset=reset, active=active.cpu().tolist(),
            q0=frame[:, 0].cpu().tolist(), command=command.cpu().tolist()))
        return action, extra


runtime.DeployInference = GuardedInference
if __name__ == '__main__':
    try:
        execution.hydra.main(config_name='config', config_path=str(
            Path(__file__).resolve().parents[1]/'configs'))(execution.main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    p = Path(os.environ['E003_REAL_EXEC_OUT'])
    d = json.loads(p.read_text())
    d.update(deployment_assistance=dict(source='measured_index_0', threshold=threshold,
        pulse_steps=pulse_steps, min_flexion=min_flexion, reset_at=reset_at, scorer_used_for_control=False, records=records),
        diagnostic_only=True, acceptance_eligible=False)
    p.write_text(json.dumps(d, indent=2)+'\n')
