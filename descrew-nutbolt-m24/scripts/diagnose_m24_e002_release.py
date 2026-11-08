#!/usr/bin/env python3
"""Diagnostic-only action bank after an actual pretrained drive; no PPO updates."""
import hashlib
import json
from pathlib import Path
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch, gymtorch
from dexscrew.algo.ppo.ppo import PPO
from omegaconf import open_dict, OmegaConf
import numpy as np


def bank():
    result=[dict(name='zero',action=[0.]*16)]
    for a in (.05,.1,.2):
        for j in (0,1,2,3,12,13,14,15):
            for sign in (-1,1):
                u=[0.]*16;u[j]=a*sign
                result.append(dict(name='joint_%d_%+.2f'%(j,a*sign),action=u))
        for sign in (-1,1):
            result.append(dict(name='coupled_radial_%+.2f'%(sign*a),radial_scale=sign*a))
    return result


@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
    out=Path(str(cfg.release_out));assert not out.exists() and cfg.test
    candidates=bank()
    if cfg.get('release_radial_only', False):
        candidates=[dict(name='radial_index%.2f_thumb%.2f'%(i,t),radial_scale=[i,t])
                    for i,t in ((.2,.25),(.2,.3),(.2,.4),(.25,.25),(.3,.3),(.4,.4),(.5,.5),(.6,.6))]
    assert int(cfg.task.env.numEnvs)==len(candidates)
    cal=json.loads(Path(str(cfg.release_calibration)).read_text())
    assert cal['status']=='passed' and cal['pose_index']==743
    with open_dict(cfg.task.env):cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
    torch.manual_seed(42);np.random.seed(42)
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,
                                      graphics_device_id=0,headless=True)
    jac=gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim,'hand'))
    agent=PPO(env,str(out.parent/'inference'),full_config=cfg)
    agent.restore_test(str(cfg.checkpoint));agent.set_eval();obs=env.reset()
    first=np.ones(env.num_envs,dtype=bool);bound=np.zeros(env.num_envs,dtype=bool)
    diagnostic=torch.zeros(env.num_envs,16,device=env.device)
    rows=[];final=[None]*env.num_envs;original=env.compute_reward;step=0
    def record(*args):
        before=env.e002_cycle.phase.detach().cpu().tolist();original(*args)
        c=env.e002_cycle
        for i in np.where(first)[0]:
            r=dict(candidate=int(i),step=step,phase_before=int(before[i]),phase=int(c.phase[i]),
                angle=float(env.nut_dof_pos[i,0]),contact=env.e002_contact_state[i].tolist(),
                force=torch.norm(env.e002_normal_contact_force[i],dim=-1).tolist(),
                action=env.actions[i,:16].tolist(),return_error=float(c.return_error[i]),
                drive_events=int(c.drive_events[i]),release_events=int(c.release_events[i]),
                reason=int(c.last_reason[i]))
            rows.append(r)
            # Stop scoring each first episode immediately after a real release.
            if r['reason'] or r['release_events'] or step==119:
                first[i]=False;final[i]=r
    env.compute_reward=record
    with torch.no_grad():
        for step in range(120):
            action,extra=agent._eval_select_action(obs)
            action=action.to(env.device)
            entered=(env.e002_cycle.phase==1).cpu().numpy() & first & ~bound
            if entered.any():
                env.gym.refresh_jacobian_tensors(env.sim)
                for i in np.where(entered)[0]:
                    candidate=candidates[i]
                    if 'action' in candidate:
                        diagnostic[i]=torch.tensor(candidate['action'],device=env.device)
                    else:
                        nut=env.rigid_body_states[i,env.screw_nut_rb_handle,:3]
                        basis=[]
                        for finger,(h,lo,hi) in enumerate([(env.fingertip_handles[0],0,4),(env.fingertip_handles[3],12,16)]):
                            radial=env.rigid_body_states[i,h,:3]-nut
                            axis=env.rot_axis_buf[i];radial=radial-(radial*axis).sum()*axis
                            radial=radial/radial.norm().clamp(min=1e-8)
                            j=jac[i,h-1,:3,lo:hi]
                            u=j.T@torch.linalg.solve(j@j.T+1e-6*torch.eye(3,device=env.device),radial)
                            u=u/u.abs().max().clamp(min=1e-8)
                            scale=candidate['radial_scale']
                            if isinstance(scale,list):scale=scale[finger]
                            diagnostic[i,lo:hi]=scale*u
                            basis.append(u.tolist())
                        candidate['radial_basis']=basis;candidate['action']=diagnostic[i].tolist()
                    bound[i]=True
            active=torch.as_tensor(bound & first,device=env.device)
            action[active,:16]=diagnostic[active]
            obs,_,_,_=env.step(action,**extra)
            if not first.any():break
    assert all(r is not None for r in final)
    finite=all(np.isfinite([r['angle'],*r['force'],*r['action']]).all() for r in rows)
    ranked=sorted(final,key=lambda r:(r['release_events'],r['angle']>=np.radians(10),
                                     -sum(r['force']),r['angle']),reverse=True)
    report=dict(diagnostic_only=True,goal_completed=False,finite=bool(finite),
        checkpoint=str(cfg.checkpoint),checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        candidates=candidates,first_episodes=final,steps=rows,top_k=ranked[:10],
        counts=dict(candidates=len(candidates),entered_release=int(bound.sum()),
                    released=sum(r['release_events']>0 for r in final)))
    out.parent.mkdir(parents=True,exist_ok=True);OmegaConf.save(cfg,out.with_suffix('.config.yaml'),resolve=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report['counts']))
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
