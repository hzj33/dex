#!/usr/bin/env python3
"""Controlled regrasp followed by unchanged PPO drive; diagnostic, never acceptance."""
import hashlib
import json
from pathlib import Path
import numpy as np
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch, gymtorch
from dexscrew.algo.ppo.ppo import PPO
from omegaconf import open_dict, OmegaConf


def bank():
    result=[dict(name='policy',policy=True),dict(name='zero',action=[0.]*16)]
    for scale in (.05,.1,.2):
        for joint in (0,1,2,3,12,13,14,15):
            for sign in (-1,1):
                u=[0.]*16;u[joint]=sign*scale
                result.append(dict(name='joint_%d_%+.2f'%(joint,sign*scale),action=u))
    for scale in (.05,.1,.2,.4):
        for anchor in (0.,1.,3.):
            result.append(dict(name='close_%.2f_anchor%.1f'%(scale,anchor),scale=scale,anchor=anchor))
        result.append(dict(name='nominal_%.2f'%scale,scale=scale,nominal=True))
    return result


@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    out=Path(str(cfg.regrasp_diagnostic_out));assert not out.exists() and cfg.test
    candidates=bank();assert len(candidates)==int(cfg.task.env.numEnvs)
    cal=json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert cal['status']=='passed' and cal['pose_index']==743
    with open_dict(cfg.task.env):cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
    torch.manual_seed(42);np.random.seed(42)
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,graphics_device_id=0,headless=True)
    out.parent.mkdir(parents=True,exist_ok=True)
    agent=PPO(env,str(out.parent/'inference'),full_config=cfg)
    agent.restore_test(str(cfg.checkpoint));agent.set_eval()
    jac=gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim,'hand'))
    obs=env.reset();assert env.e002_demo.active.all() and (env.e002_cycle.phase==3).all()
    nominal=torch.tensor(cal['custom_init_dof_pos'],device=env.device)
    alive=np.ones(env.num_envs,dtype=bool);rows=[];final=[None]*env.num_envs;states=[]
    original=env.compute_reward;step=0
    def record(*args):
        before=env.e002_cycle.phase.clone();original(*args);c=env.e002_cycle
        for i in np.where(alive)[0]:
            radial=env.rigid_body_states[i,[env.fingertip_handles[0],env.fingertip_handles[3]],:3]-env.rigid_body_states[i,env.screw_nut_rb_handle,:3]
            row=dict(candidate=int(i),step=step,phase_before=int(before[i]),phase=int(c.phase[i]),event=int(c.last_event[i]),
                angle=float(env.nut_dof_pos[i,0]),return_error=float(c.return_error[i]),
                contact=env.e002_contact_state[i].tolist(),force=torch.norm(env.e002_normal_contact_force[i],dim=-1).tolist(),
                distance=radial.norm(dim=-1).tolist(),action=env.actions[i,:16].tolist(),reason=int(c.last_reason[i]),
                drive_events=int(c.drive_events[i]),cycles=int(c.cycles[i]),quality=float(env.e002_forward_quality[i]) if hasattr(env,'e002_forward_quality') else None,
                dof_pos=env.dof_state.view(env.num_envs,17,2)[i,:,0].tolist(),targets=env.prev_targets[i].tolist())
            rows.append(row)
            if row['phase_before']==3 or row['event']:
                states.append(dict(row,reset_seed_state=dict(reconstruction_validated=False,boundary='after_reward_before_observation_roll',
                    dof_state=env.dof_state.view(env.num_envs,17,2)[i].tolist(),
                    actor_root_state=env.root_state_tensor[[int(env.hand_indices[i]),int(env.object_indices[i])]].tolist(),
                    joint_targets=env.prev_targets[i].tolist(),observation_history_before_roll=env.obs_buf_lag_history[i].tolist(),
                    cycle={k:v[i].tolist() for k,v in c.__dict__.items() if torch.is_tensor(v) and v is not c.deadlines})))
            if row['reason'] or row['drive_events']>=1 or step==119:final[i]=row;alive[i]=False
    env.compute_reward=record
    with torch.no_grad():
        for step in range(120):
            env.gym.refresh_jacobian_tensors(env.sim)
            action,extra=agent._eval_select_action(obs)
            for i in np.where(alive)[0]:
                spec=candidates[i]
                if int(env.e002_cycle.phase[i])!=3 or spec.get('policy'):continue
                action[i]=0
                if 'action' in spec:action[i]=torch.tensor(spec['action'],device=action.device);continue
                if spec.get('nominal'):
                    error=nominal-env.prev_targets[i,:16]
                    action[i]=error/error.abs().max().clamp(min=1e-8)*spec['scale'];continue
                nut=env.rigid_body_states[i,env.screw_nut_rb_handle,:3];axis=env.rot_axis_buf[i]
                for f,(h,lo,hi) in enumerate([(env.fingertip_handles[0],0,4),(env.fingertip_handles[3],12,16)]):
                    radial=env.rigid_body_states[i,h,:3]-nut
                    radial=radial-(radial*axis).sum()*axis;unit=radial/radial.norm().clamp(min=1e-8)
                    ref=env.e002_cycle.reference[i,f]
                    tangent=ref-(ref*unit).sum()*unit
                    inward=0. if bool(env.e002_contact_state[i,f]) else .2
                    direction=spec['anchor']*tangent-inward*unit
                    j=jac[i,h-1,:3,lo:hi]
                    u=j.T@torch.linalg.solve(j@j.T+1e-6*torch.eye(3,device=env.device),direction)
                    action[i,lo:hi]=spec['scale']*u/u.abs().max().clamp(min=1e-8)
            obs,_,_,_=env.step(action,**extra)
            if not alive.any():break
    recontacts={r['candidate']:r for r in rows if r['event']==4}
    ranked=sorted(final,key=lambda r:(r['drive_events'],-recontacts.get(r['candidate'],dict(return_error=10))['return_error']),reverse=True)
    finite=all(np.isfinite([r['angle'],r['return_error'],*r['force'],*r['action']]).all() for r in rows)
    report=dict(diagnostic_only=True,goal_completed=False,finite=bool(finite),candidates=candidates,steps=rows,
        first_episodes=final,recontact_events=list(recontacts.values()),top_k=ranked[:10],
        counts=dict(candidates=len(candidates),recontacted=len(recontacts),next_drive=sum(r['drive_events']>=1 for r in final)),
        checkpoint=str(cfg.checkpoint),checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        seed_source=str(cfg.task.env.e002_demo_reset.source),seed_sha256=env.e002_demo.source_sha256)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    out.with_suffix('.states.json').write_text(json.dumps(dict(states=states),allow_nan=False)+'\n')
    OmegaConf.save(cfg,out.with_suffix('.config.yaml'),resolve=True)
    print(json.dumps(dict(counts=report['counts'],top_k=ranked[:3]),indent=2))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
