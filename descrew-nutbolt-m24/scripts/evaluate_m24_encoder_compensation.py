#!/usr/bin/env python3
"""Bounded encoder-calibration hypothesis with real-state motor actor inputs."""
import json
import os
from pathlib import Path
import evaluate_m24_e003_real_exec as execution
import m24_e003_deploy_runtime as runtime
from m24_encoder_compensation import compensated_action
from m24_release_guard import ReleaseGuard
if os.environ.get('E003_COUPLING_FIT'):
    import evaluate_m24_e003_passive_diagnostic
scale = float(os.environ['E003_ENCODER_ERROR_SCALE'])
assert 0.25 <= scale <= 3.0
records = []
guard_enabled = os.environ.get('E003_ENCODER_GUARD') == '1'


class CompensatedInference(runtime.DeployInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.guards = [ReleaseGuard(min_flexion=float(os.environ.get('E003_GUARD_MIN_FLEXION', '.9')))
                       for _ in range(self.env.num_envs)]
        self.remaining = execution.torch.zeros(self.env.num_envs, dtype=execution.torch.long, device=self.inference_device)

    @execution.torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        if self.history is None:
            self.history = self.model.reset_history(frame[:, 16:], frame[:, 16:])
        prior_obs = self.history[:, -3:].reshape(-1, 96).clone()
        _, extra = super()._eval_select_action(obs)
        active = None
        if guard_enabled:
            fire = execution.torch.tensor([g.update(q) for g, q in zip(self.guards, frame[:, :16].cpu().numpy())], device=self.inference_device)
            self.remaining[fire] = 20
            active = self.remaining > 0
            self.remaining[active] -= 1
        action, command = compensated_action(self.model, prior_obs, self.history, scale, active)
        records.append(command.cpu().tolist())
        return action, extra


runtime.DeployInference = CompensatedInference
if __name__ == '__main__':
    try:
        execution.hydra.main(config_name='config', config_path=str(
            Path(__file__).resolve().parents[1]/'configs'))(execution.main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    path = Path(os.environ['E003_REAL_EXEC_OUT'])
    result = json.loads(path.read_text())
    result.update(diagnostic_only=True, acceptance_eligible=False,
        encoder_compensation=dict(scale=scale, release_guard_enabled=guard_enabled, actor_joint_observation='real',
            encoder_feature_transform='target + scale*(q-target), clamped to joint limits, active joints only',
            commands=records))
    path.write_text(json.dumps(result, indent=2)+'\n')
