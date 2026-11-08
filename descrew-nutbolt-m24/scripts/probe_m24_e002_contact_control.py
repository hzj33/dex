#!/usr/bin/env python3
"""Diagnostic contact-force/PD inverse probe, never a learned policy or PPO.

J_contact.T F gives a joint torque target; the unchanged PD/action interface
receives bounded increments toward q + tau/kp. Applied contact forces remain
simulator outputs. Commanded force is explicitly not a measured full wrench.
"""
import json,math
import numpy as np
from pathlib import Path
from calibrate_m24_e002_a0 import hydra,isaacgym_task_map,omegaconf_to_dict,torch,gymtorch
from diagnose_m24_e002_a0c import snapshot
from m24_e002_a0c_diagnostics import measure_step
from omegaconf import open_dict
from audit_m24_e002_contacts import plain

def rotate(q,v):
    return v+2*torch.cross(q[:,:3],torch.cross(q[:,:3],v,dim=-1)+q[:,3:4]*v,dim=-1)

@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
    out=Path(str(cfg.contact_control_out));out.parent.mkdir(parents=True,exist_ok=True)
    assert not out.exists()
    source=json.loads(Path(str(cfg.contact_control_calibration)).read_text())
    kin=json.loads(Path(str(cfg.contact_control_basis)).read_text())
    assert source['pose_index']==kin['pose_index']==743
    bank=[dict(normal_force=f,tangent_ratio=r) for f in [2.,4.,8.,16.] for r in [.25,.5,.75]]
    if cfg.get('contact_control_ablation', False):
        bank=[dict(normal_force=16.,tangent_ratio=r) for r in
              [[.5,.5],[0.,.5],[.5,0.],[0.,0.],[-.5,.5],[.5,-.5]]]
    assert int(cfg.task.env.numEnvs)==len(bank)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos=source['custom_init_dof_pos']
        cfg.task.env.e002_motion.force_sign=source['force_sign']
    torch.manual_seed(42)
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,graphics_device_id=0,headless=True)
    suppression=dict(contact_acquire_deadline_steps=100000,contact_loss_grace_steps=100000,drive_deadline_steps=100000,target_angle=100.,hard_step_angle=100.,overshoot_angle=100.,workspace_radius=1.)
    for k,v in suppression.items():setattr(env,'e002_'+k,v)
    thresholds=source['thresholds']
    jac=gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim,'hand'))
    zero=torch.zeros((len(bank),16),device=env.device)
    env.reset()
    for _ in range(6):env.step(zero)
    start=env.nut_dof_pos[:,0].clone();previous=start.clone()
    forces=torch.tensor([b['normal_force'] for b in bank],device=env.device)
    ratios=torch.tensor([b['tangent_ratio'] if isinstance(b['tangent_ratio'],list)
                         else [b['tangent_ratio']]*2 for b in bank],device=env.device)
    candidates=[dict(candidate_id=i,**b,steps=[]) for i,b in enumerate(bank)]
    substeps=[];context=dict(policy_step=-1,substep=0,theta_previous=start.cpu().numpy().copy())
    if cfg.get('contact_control_substeps',False):
        from dexscrew.tasks.contact_moments import collect_env_contacts,normal_contact_wrench
        original_render=env.render
        def record_substep():
            original_render()
            if context['policy_step']<0:return
            env.gym.fetch_results(env.sim,True);env._refresh_gym()
            pair=[env.fingertip_handles[0],env.fingertip_handles[3]]
            states=env.rigid_body_states.cpu().numpy()
            force,moment,counts=normal_contact_wrench(collect_env_contacts(env.gym,env.envs),
                states[:,env.screw_nut_rb_handle,3:7],env.screw_nut_rb_handle,pair,env.num_bodies)
            fm=np.linalg.norm(env.contact_forces[:,pair].cpu().numpy(),axis=-1)
            dist=np.linalg.norm(states[:,pair,:3]-states[:,env.screw_nut_rb_handle,None,:3],axis=-1)
            raw=(-moment*env.rot_axis_buf.cpu().numpy()[:,None,:]).sum(-1)
            theta=env.nut_dof_pos[:,0].cpu().numpy().copy();velocity=env.nut_dof_vel[:,0].cpu().numpy()
            for i in range(len(bank)):
                delta=float(theta[i]-context['theta_previous'][i])
                row=dict(candidate_id=i,policy_step=context['policy_step'],substep=context['substep'],
                    nut_angle=float(theta[i]-float(start[i])),nut_velocity=float(velocity[i]),directed_step=delta,
                    normal_force_on_nut=force[i].tolist(),normal_moment_on_nut=moment[i].tolist(),contact_count=counts[i].tolist())
                row.update(measure_step(fm[i],dist[i],raw[i],delta,thresholds));substeps.append(row)
            context['theta_previous']=theta;context['substep']+=1
        env.render=record_substep
    friction_max=0.;friction_nonzero=0
    for t in range(80):
        env.gym.refresh_jacobian_tensors(env.sim)
        action=zero.clone();commanded=[]
        for f,(handle,cols) in enumerate(zip([env.fingertip_handles[0],env.fingertip_handles[3]],[(0,4),(12,16)])):
            lo,hi=cols
            state=env.rigid_body_states[:,handle];q=state[:,3:7]
            point=torch.tensor(kin['fingers'][f]['local_contact_centroid'],device=env.device).expand(len(bank),3)
            com=torch.tensor(kin['fingers'][f]['com_local'],device=env.device).expand(len(bank),3)
            r=state[:,:3]+rotate(q,point)-env.rigid_body_states[:,env.screw_nut_rb_handle,:3]
            tangent=torch.cross(env.rot_axis_buf,r,dim=-1);tangent/=torch.norm(tangent,dim=-1,keepdim=True).clamp(min=1e-6)
            measured=env.e002_normal_contact_force[:,f]
            n=measured/torch.norm(measured,dim=-1,keepdim=True).clamp(min=1e-6)
            fallback=-r/torch.norm(r,dim=-1,keepdim=True).clamp(min=1e-6)
            n=torch.where((torch.norm(measured,dim=-1)>.15)[:,None],n,fallback)
            desired=forces[:,None]*(n+ratios[:,f,None]*tangent)
            j=jac[:,handle-1,:,lo:hi];arm=rotate(q,point-com)
            jc=j[:,:3]+torch.cross(j[:,3:].transpose(1,2),arm[:,None,:].expand(-1,4,-1),dim=-1).transpose(1,2)
            tau=(jc.transpose(1,2)@desired[:,:,None]).squeeze(-1)
            target=env.xhand_hand_dof_pos[:,lo:hi]+tau/env.p_gain[:,lo:hi]
            action[:,lo:hi]=((target-env.prev_targets[:,lo:hi])/env.action_scale).clamp(-1.,1.)
            commanded.append(desired.detach().cpu().tolist())
        context.update(policy_step=t,substep=0)
        env.step(action)
        if cfg.get('contact_control_substeps',False):assert context['substep']==env.control_freq_inv
        contact_rows=[]
        for e in env.envs:
            records=env.gym.get_env_rigid_contacts(e)
            if cfg.get('contact_control_raw',False):
                contact_rows.append([plain(r) for r in records])
            for field in ['x','y']:
                values=np.abs(records['lambdaFriction'][field])
                if values.size:
                    friction_max=max(friction_max,float(values.max()))
                    friction_nonzero+=int((values>0).sum())
        data=snapshot(env);theta=env.nut_dof_pos[:,0].clone()
        gait=None
        if cfg.get('contact_control_gait',False):
            from m24_e002_contact_gait_diagnostic import measure
            gait=measure(env,jac,[f['com_local'] for f in kin['fingers']])
        for i,c in enumerate(candidates):
            row={k:v[i] for k,v in data.items()}
            if gait is not None:row['contact_gait']=gait[i]
            if contact_rows:
                row['contacts']=contact_rows[i]
                row['nut_quaternion']=env.rigid_body_states[i,env.screw_nut_rb_handle,3:7].cpu().tolist()
            delta=float(theta[i]-previous[i]);angle=float(theta[i]-start[i])
            row.update(step=t,action=action[i].cpu().tolist(),commanded_force_not_measured=[v[i] for v in commanded],directed_step=delta,directed_angle=angle)
            row.update(measure_step(row['force_magnitude'],row['distance'],row['raw_axis_torque'],delta,thresholds))
            c['steps'].append(row)
        previous=theta
    summary=[]
    for c in candidates:
        ss=c['steps'];summary.append(dict(candidate_id=c['candidate_id'],normal_force=c['normal_force'],tangent_ratio=c['tangent_ratio'],net_angle=ss[-1]['directed_angle'],max_angle=max(s['directed_angle'] for s in ss),contact_steps=sum(s['pair_contact'] for s in ss),strict_steps=sum(s['force_signs']['-1']['conditions']['all'] for s in ss),max_gate=max(s['force_signs']['-1']['pair_gate'] for s in ss),strict_positive_angle=sum(max(0,s['directed_step']) for s in ss if s['force_signs']['-1']['conditions']['all'])))
    report=dict(status='completed:diagnostic_only',training_launched=False,policy_learned=False,a0c_status='not_evaluated',pose_index=743,thresholds=thresholds,terminal_suppression=suppression,commanded_force_is_not_measured_wrench=True,physx_use_gpu=bool(env.sim_params.physx.use_gpu),friction_api_max_abs=friction_max,friction_api_nonzero_count=friction_nonzero,summary=summary,candidates=candidates)
    if substeps:report['physics_substeps']=substeps
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2));env.gym.destroy_sim(env.sim)
if __name__=='__main__':main()
