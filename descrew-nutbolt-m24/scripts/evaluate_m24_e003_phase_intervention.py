#!/usr/bin/env python3
"""Simulation-only phase intervention; joint feedback/targets stay untouched.

E003_PHASE_INTERVENTION=phase or phase_clock uses the simulator's physical
scoring phase as an oracle. This is not autonomous deployment acceptance and
does not imply that real hardware has this information. The learned latent,
frozen actor, observed joint feedback and sent-command history are retained.
"""
import json
import os
from pathlib import Path

import evaluate_m24_e003_real_exec as execution
import m24_e003_deploy_runtime as runtime
from m24_e003_release_override import phase_action

torch = execution.torch
mode = os.environ.get('E003_PHASE_INTERVENTION', 'phase_clock')
assert mode in ('phase', 'phase_clock', 'release_pulse')
release_pulse_steps = int(os.environ.get('E003_RELEASE_PULSE_STEPS', '20'))
assert release_pulse_steps > 0
trigger_delay_steps = int(os.environ.get('E003_RELEASE_TRIGGER_DELAY_STEPS', '0'))
assert trigger_delay_steps >= 0
intervention_rows = []


class PhaseIntervention(runtime.DeployInference):
    diagnostic_phase_intervention = True
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_drive_count = 0
        self.pulse_remaining = 0
        self.pending_release = -1

    @torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].to(self.inference_device)
        if self.history is None:
            self.history = self.model.reset_history(frame[:, 16:], frame[:, 16:])
        prior_obs = self.history[:, -3:].reshape(-1, 96).clone()
        _, extra = super()._eval_select_action(obs)
        _, logits, clock, latent = self.model.predict(prior_obs, self.history)
        command = self.env.e002_cycle.command().to(self.inference_device).clone()
        if mode == 'phase':
            command[:, 4:5] = clock
        if mode == 'release_pulse':
            assert self.env.num_envs == 1
            # Simulator event stands in for a user's timely key press, not a
            # deployable contact detector. Hold release briefly, then return
            # all stage decisions to the unchanged learned estimator.
            count = int(self.env.e002_cycle.drive_events[0])
            if count > self.last_drive_count:
                self.pending_release = trigger_delay_steps
                self.last_drive_count = count
            if self.pending_release == 0:
                self.pulse_remaining = release_pulse_steps
                self.pending_release = -1
            elif self.pending_release > 0:
                self.pending_release -= 1
            command = torch.cat((logits.softmax(-1), clock), -1)
            if self.pulse_remaining:
                command[:, :4] = 0
                command[:, 1] = 1
                self.pulse_remaining -= 1
        action = phase_action(self.model, prior_obs, self.history, command)
        intervention_rows.append(dict(
            predicted_phase=logits.argmax(-1).cpu().tolist(),
            supplied_phase_command=command.cpu().tolist()))
        return action, extra


runtime.DeployInference = PhaseIntervention

if __name__ == '__main__':
    try:
        execution.hydra.main(config_name='config', config_path=str(
            Path(__file__).resolve().parents[1]/'configs'))(execution.main.__wrapped__)()
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    path = Path(os.environ['E003_REAL_EXEC_OUT'])
    result = json.loads(path.read_text())
    result['diagnostic_only'] = True
    result['acceptance_eligible'] = False
    result['phase_intervention'] = dict(
        mode=mode, source='simulator physical scoring state',
        release_pulse_steps=release_pulse_steps if mode == 'release_pulse' else None,
        trigger_delay_steps=trigger_delay_steps if mode == 'release_pulse' else None,
        joint_feedback_modified=False, latent_source='learned student',
        note='Predicted phase fields remain the unmodified estimator output; actor phase is overridden.')
    result['phase_command_source'] = 'simulator oracle intervention; not autonomous'
    result['phase_intervention_steps'] = intervention_rows
    result['real_execution_model'] = dict(real_params=execution.REAL_PARAMS,
        observation_target_source='unfiltered_sent_command', diagnostic_only=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
