#!/usr/bin/env python3
"""Fixed-horizon student diagnostic with contact preserved but the nut jammed.

User-confirmed real-hand failure mode (2026-09-17): the fingers gripped the nut,
immediately entered a stuck state, and the release process never triggered.
This probe keeps grasping contact intact and raises the nut joint Coulomb
friction far beyond hand torque capability, so the nut can be gripped but not
rotated. No task acceptance: automatic resets stay suppressed and the original
scorer is telemetry only.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from evaluate_m24_e003_passive_diagnostic import isaacgym_task_map, torch, hydra, root
from calibrate_m24_e002_a0 import omegaconf_to_dict
from omegaconf import open_dict
from isaacgym import gymtorch
from m24_e003_deploy_runtime import DeployInference

jam_friction = float(__import__('os').environ.get('E003_JAM_FRICTION', '0'))
jam_root = Path(__import__('os').environ['E003_COUPLING_OUT'])
jam_root.mkdir(parents=True, exist_ok=True)
assert jam_friction >= 0


def instrument_jam(base):
    class JamDiagnostic(base):
        def create_sim(self):
            real_gym = self.gym
            outer = self

            class _GymProxy:
                def __getattr__(self, item):
                    return getattr(real_gym, item)

                def create_actor(self, *args, **kwargs):
                    handle = real_gym.create_actor(*args, **kwargs)
                    if jam_friction > 0 and len(args) >= 4 and args[3] == 'object':
                        props = real_gym.get_actor_dof_properties(args[0], handle)
                        assert len(props) == 1, \
                            'M24 object actor must own exactly the nut DOF'
                        outer._jam_original_friction = float(props['friction'][0])
                        props['friction'][0] = jam_friction
                        real_gym.set_actor_dof_properties(args[0], handle, props)
                    return handle
            self.gym = _GymProxy()
            try:
                super().create_sim()
            finally:
                self.gym = real_gym
    return JamDiagnostic


if jam_friction > 0:
    for name in list(isaacgym_task_map):
        if 'M24' in name:
            isaacgym_task_map[name] = instrument_jam(isaacgym_task_map[name])
(jam_root/'jam_diagnostic.json').write_text(json.dumps(dict(
    diagnostic_only=True, hardware_calibrated=False, nut_friction=jam_friction,
    base_friction=0.2, intervention=('nut actor DOF Coulomb friction raised right '
    'after create_actor, before sim start; contact and grasp geometry unchanged')
    if jam_friction else 'none',
    note='Not a measured real-nut resistance; single-factor probe of the '
         'user-reported grip-then-stuck failure mode; no real hardware execution'), indent=2))


@hydra.main(config_name='config', config_path=str(Path(__file__).resolve().parents[1]/'configs'))
def main(cfg):
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed))
    calibration = json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert calibration['status'] == 'passed' and calibration['pose_index'] == 743
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration['custom_init_dof_pos']
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=0, headless=True)
    assert env.num_envs == 1
    if jam_friction > 0:
        assert getattr(env, '_jam_original_friction', None) is not None, \
            'jam friction was not applied before object actor creation'
    agent = DeployInference(env, cfg.checkpoint, record_predictions=True, inference_device='cpu')
    original_reward = env.compute_reward
    def diagnostic_reward(*args):
        original_reward(*args)
        env.reset_buf.zero_()
    env.compute_reward = diagnostic_reward
    obs = env.reset()
    rows=[]
    with torch.no_grad():
        for step in range(200):
            action, extra = agent._eval_select_action(obs)
            obs, _, _, _ = env.step(action, **extra)
            q=env.xhand_hand_dof_pos[0].cpu().numpy()
            tip=env.rigid_body_states[0, env.fingertip_handles, :3].cpu().numpy()
            force=torch.norm(env.e002_normal_contact_force,dim=-1)[0].cpu().numpy()
            nut_angle=float(env.nut_dof_pos[0,0].cpu())
            assert np.isfinite(q).all() and np.isfinite(force).all() and np.isfinite(nut_angle)
            rows.append(dict(step=step,q=q.tolist(),target=env.prev_targets[0,:16].cpu().tolist(),
                phase_prediction=agent.prediction[0].cpu().tolist(),
                fingertip_separation_m=float(np.linalg.norm(tip[0]-tip[3])),
                force=force.tolist(),contact=env.e002_contact_state[0].cpu().tolist(),
                nut_angle_rad=nut_angle,
                scoring_phase=int(env.e002_cycle.phase[0]),scoring_reason=int(env.e002_cycle.last_reason[0])))
    prediction=np.array([r['phase_prediction'] for r in rows])[:,:4].argmax(1)
    separation=np.array([r['fingertip_separation_m'] for r in rows])
    nut=np.array([r['nut_angle_rad'] for r in rows])
    spans=[dict(step=i,phase=int(p)) for i,p in enumerate(prediction) if i==0 or p!=prediction[i-1]]
    result=dict(diagnostic_only=True,acceptance_eligible=False,jam_friction=jam_friction,
        applied=(jam_friction>0),reset_suppressed=True,
        intervention=('nut joint friction 0.2 -> %.3g Nm via actor DOF props' % jam_friction)
            if jam_friction else 'Object unchanged',
        policy_steps=200,checkpoint=str(cfg.checkpoint),checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        coupling=json.loads((root/'coupling_diagnostic.json').read_text()),
        phase_counts=np.bincount(prediction,minlength=4).tolist(),phase_spans=spans,
        first_release_estimate_step=int(np.argmax(prediction==1)) if bool((prediction==1).any()) else None,
        max_force=np.array([r['force'] for r in rows]).max(0).tolist(),
        nut_angle_rad=dict(first=float(nut[0]),max=float(nut.max()),min=float(nut.min()),
            abs_max=float(np.abs(nut).max()),last=float(nut[-1])),
        separation_mm=dict(first=float(separation[0]*1000),min=float(separation.min()*1000),max=float(separation.max()*1000),last=float(separation[-1]*1000)),rows=rows)
    if jam_friction > 0:
        assert result['nut_angle_rad']['abs_max'] < 0.2, \
            'Jam friction did not effectively prevent nut rotation'
    else:
        assert float(np.max(result['max_force'])) > 1.0, 'Contact baseline lost nut force'
    (jam_root/'jam_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','coupling')},indent=2))


if __name__ == '__main__':
    main()
