#!/usr/bin/env python3
"""Fixed-horizon student diagnostic; optionally move the nut away after reset.

No task acceptance: automatic resets are suppressed so a no-contact trial can
run its full observation history. Original scorer is retained only as telemetry.
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
    agent = DeployInference(env, cfg.checkpoint, record_predictions=True, inference_device='cpu')
    original_reward = env.compute_reward
    def diagnostic_reward(*args):
        original_reward(*args)
        env.reset_buf.zero_()
    env.compute_reward = diagnostic_reward
    obs = env.reset()
    air = bool(cfg.get('air_diagnostic', False))
    if air:
        indices = env.object_indices.to(torch.int32)
        env.root_state_tensor[env.object_indices, 1] += 1.0
        env.gym.set_actor_root_state_tensor_indexed(env.sim,
            gymtorch.unwrap_tensor(env.root_state_tensor), gymtorch.unwrap_tensor(indices), len(indices))
    rows=[]
    with torch.no_grad():
        for step in range(200):
            action, extra = agent._eval_select_action(obs)
            obs, _, _, _ = env.step(action, **extra)
            q=env.xhand_hand_dof_pos[0].cpu().numpy()
            tip=env.rigid_body_states[0, env.fingertip_handles, :3].cpu().numpy()
            force=torch.norm(env.e002_normal_contact_force,dim=-1)[0].cpu().numpy()
            assert np.isfinite(q).all() and np.isfinite(force).all()
            rows.append(dict(step=step,q=q.tolist(),target=env.prev_targets[0,:16].cpu().tolist(),
                phase_prediction=agent.prediction[0].cpu().tolist(),
                fingertip_separation_m=float(np.linalg.norm(tip[0]-tip[3])),
                force=force.tolist(),contact=env.e002_contact_state[0].cpu().tolist(),
                scoring_phase=int(env.e002_cycle.phase[0]),scoring_reason=int(env.e002_cycle.last_reason[0])))
    prediction=np.array([r['phase_prediction'] for r in rows])[:,:4].argmax(1)
    separation=np.array([r['fingertip_separation_m'] for r in rows])
    spans=[dict(step=i,phase=int(p)) for i,p in enumerate(prediction) if i==0 or p!=prediction[i-1]]
    result=dict(diagnostic_only=True,acceptance_eligible=False,air=air,reset_suppressed=True,
        intervention='Object shifted +1m along world Y after identical initial reset' if air else 'Object unchanged',
        policy_steps=200,checkpoint=str(cfg.checkpoint),checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        coupling=json.loads((root/'coupling_diagnostic.json').read_text()),
        phase_counts=np.bincount(prediction,minlength=4).tolist(),phase_spans=spans,
        max_force=np.array([r['force'] for r in rows]).max(0).tolist(),
        separation_mm=dict(first=float(separation[0]*1000),min=float(separation.min()*1000),max=float(separation.max()*1000),last=float(separation[-1]*1000)),rows=rows)
    if air: assert float(np.max(result['max_force'])) < 1e-6, 'Air intervention still has fingertip/nut force'
    (root/'air_evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','coupling')},indent=2))


if __name__ == '__main__':
    main()
