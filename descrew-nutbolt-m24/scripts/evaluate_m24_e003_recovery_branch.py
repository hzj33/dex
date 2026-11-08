#!/usr/bin/env python3
"""Diagnostic same-prefix recovery branches, never hardware control.

Before the explicit switch step, execute the unchanged student. Then compare
the teacher or phase/clock interventions from that trajectory. A branch only
counts as matched-state evidence after its entire prefix is compared with the
baseline (including physical state), rather than merely using the same seed.
"""
import json
import os
from pathlib import Path
import evaluate_m24_e003_continuous as execution
import m24_e003_deploy_runtime as runtime
from m24_e003_release_override import phase_action
from dexscrew.algo.ppo.ppo import PPO

torch = execution.execution.torch
mode = os.environ.get('E003_RECOVERY_MODE', 'none')
assert mode in ('none', 'teacher', 'teacher_clock_zero', 'phase_clock',
                'phase_clock_zero', 'clock_zero')
switch = int(os.environ['E003_RECOVERY_SWITCH_STEP'])
assert switch >= 0
duration = int(os.environ.get('E003_RECOVERY_DURATION', '0'))
assert duration >= 0
context = {}
records = []


class RecoveryBranch(runtime.DeployInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.env.num_envs == 1
        self.steps = 0
        self.teacher = None
        if mode in ('teacher', 'teacher_clock_zero'):
            self.teacher = PPO(self.env, str(context['out']/'teacher_inference'),
                               full_config=context['cfg'])
            self.teacher.restore_test(str(context['cfg'].cycle_eval_recovery_teacher_checkpoint))
            self.teacher.set_eval()

    @torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        if self.history is None:
            self.history = self.model.reset_history(frame[:, 16:], frame[:, 16:])
        prior_obs = self.history[:, -3:].reshape(-1, 96).clone()
        action, extra = super()._eval_select_action(obs)
        active = (self.steps >= switch and mode != 'none'
                  and (duration == 0 or self.steps < switch+duration))
        if active:
            if self.teacher is not None:
                teacher_obs = obs
                if mode == 'teacher_clock_zero':
                    teacher_obs = dict(obs)
                    teacher_obs['obs'] = obs['obs'].clone()
                    mask = self.env.e002_cycle.phase == 2
                    teacher_obs['obs'][mask, -1] = 0
                action, extra = self.teacher._eval_select_action(teacher_obs)
                action = action.to(self.env.device)
            else:
                _, logits, clock, _ = self.model.predict(prior_obs, self.history)
                command = torch.cat((logits.softmax(-1), clock), -1)
                if mode in ('phase_clock', 'phase_clock_zero'):
                    command = self.env.e002_cycle.command().to(self.inference_device)
                    if mode == 'phase_clock_zero':
                        command[:, 4] = 0
                else:
                    command[:, 4] = 0
                action = phase_action(self.model, prior_obs, self.history, command)
        records.append(dict(step=self.steps, active=active, mode=mode))
        self.steps += 1
        return action, extra


runtime.DeployInference = RecoveryBranch


@execution.execution.hydra.main(config_name='config', config_path=str(
    Path(__file__).resolve().parents[1]/'configs'))
def main(cfg):
    context.update(cfg=cfg, out=Path(str(cfg.cycle_eval_out)).parent)
    try:
        execution.execution.main.__wrapped__(cfg)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    path = Path(str(cfg.cycle_eval_out))
    report = json.loads(path.read_text())
    report.update(diagnostic_only=True, acceptance_eligible=False,
                  recovery_branch=dict(mode=mode, switch_step=switch, duration_steps=duration,
                      teacher_oracle_information=mode in ('teacher', 'teacher_clock_zero',
                                                         'phase_clock', 'phase_clock_zero'),
                      trigger_source='explicit offline diagnostic step; not hardware observable',
                      prefix_state_matching_verified=False, records=records))
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
