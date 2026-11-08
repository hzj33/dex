#!/usr/bin/env python3
"""Closed-loop continuous contact evaluation of a joint-only recovery cue."""
import json
import os
from pathlib import Path
import evaluate_m24_e003_continuous as execution
import m24_e003_deploy_runtime as runtime
from m24_e003_progress_recovery import ProgressRecovery
from m24_e003_release_override import phase_action

torch = execution.execution.torch
records = []


class RecoveringInference(runtime.DeployInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guards = [ProgressRecovery() for _ in range(self.env.num_envs)]

    @torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        if self.history is None:
            self.history = self.model.reset_history(frame[:, 16:], frame[:, 16:])
        prior = self.history[:, -3:].reshape(-1, 96).clone()
        action, extra = super()._eval_select_action(obs)
        _, logits, clock, _ = self.model.predict(prior, self.history)
        prob = logits.softmax(-1)
        q, prev = frame[:, :16].cpu().numpy(), frame[:, 16:].cpu().numpy()
        decisions = [g.update(q[i], prev[i], action[i].cpu().numpy(), prob[i].cpu().numpy(),
                     float(clock[i, 0]), self.model.lower.cpu().numpy(), self.model.upper.cpu().numpy(),
                     float(self.model.action_scale)) for i, g in enumerate(self.guards)]
        active = torch.tensor([x[0] for x in decisions], device=self.inference_device)
        if active.any():
            command = torch.cat((prob, clock), -1)
            command[active, 4] = 0
            alternate = phase_action(self.model, prior, self.history, command)
            action = torch.where(active[:, None], alternate, action)
        records.append([x[1] for x in decisions])
        return action, extra


runtime.DeployInference = RecoveringInference

if __name__ == '__main__':
    try:
        execution.execution.hydra.main(config_name='config', config_path=str(
            Path(__file__).resolve().parents[1]/'configs'))(execution.execution.main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    path = Path(os.environ['E003_REAL_EXEC_OUT'])
    report = json.loads(path.read_text())
    report.update(deployment_assistance=dict(kind='bounded_progress_recovery',
        pulse_steps=20, physical_scorer_used_for_control=False, records=records),
        diagnostic_only=True, acceptance_eligible=False)
    path.write_text(json.dumps(report, indent=2)+'\n')
