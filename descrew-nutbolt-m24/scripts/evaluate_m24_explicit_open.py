#!/usr/bin/env python3
"""Contact-physics diagnostic for bounded opening and optional policy reentry."""
import json
import os
from pathlib import Path
import numpy as np
import evaluate_m24_e003_real_exec as execution
import m24_e003_deploy_runtime as runtime
from m24_explicit_open import ExplicitOpenController
from m24_hand_kinematics import HandKinematics

if os.environ.get('E003_COUPLING_FIT'):
    # Additional diagnostic sensitivity: remove independent index DIP drive.
    import evaluate_m24_e003_passive_diagnostic  # noqa: F401

torch = execution.torch
trigger = int(os.environ.get('E003_OPEN_TRIGGER_STEP', '16'))  # zero based
resume_after = int(os.environ.get('E003_OPEN_RESUME_AFTER', '0'))
records = []


class OpeningInference(runtime.DeployInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.env.num_envs == 1
        self.opening = ExplicitOpenController()
        self.step = 0
        self.fk = HandKinematics('assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf')

    @torch.no_grad()
    def _eval_select_action(self, obs):
        frame = obs['proprio_hist'][:, -1].cpu().numpy()[0]
        q, previous = frame[:16], frame[16:]
        action, extra = super()._eval_select_action(obs)
        if resume_after and self.step == trigger + resume_after:
            self.opening.reset()
        target, event = self.opening.update(q, previous, self.step == trigger,
                                           self.model.lower.cpu().numpy(), self.model.upper.cpu().numpy())
        if target is not None:
            action = torch.as_tensor((target-previous)/float(self.model.action_scale),
                                    device=self.inference_device).reshape(1, 16)
        tips = self.env.rigid_body_states[0, self.env.fingertip_handles, :3].cpu().numpy()
        separation = float(np.linalg.norm(tips[0]-tips[3]))
        physical_q = self.env.xhand_hand_dof_pos[0].cpu().numpy()
        fk = self.fk.separation(physical_q)
        # Independent check of URDF parsing, axes, joint order and transforms.
        assert abs(fk['tip_link_origins']-separation) < 2e-5, (self.step, fk, separation)
        records.append(dict(step=self.step, event=event, q=q.tolist(),
            separation_m=separation, kinematics=fk,
            fk_error_m=abs(fk['tip_link_origins']-separation)))
        self.step += 1
        return action, extra


runtime.DeployInference = OpeningInference

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
        explicit_open=dict(trigger_step=trigger, resume_after=resume_after,
                           records=records, hardware_operated=False),
        real_execution_model=dict(real_params=execution.REAL_PARAMS,
                                  observation_target_source='unfiltered_sent_command'))
    path.write_text(json.dumps(result, indent=2)+'\n')
