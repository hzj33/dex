#!/usr/bin/env python3
"""Frozen-policy diagnostic of relative-target accumulation, no training.

Compare unchanged control to a diagnostic-only target error clamp derived from
existing torque_limit/kp; no gain, reward, gate or termination modifications.
This probe is not a policy acceptance result or a production controller change.
"""
import json,hashlib,math
from pathlib import Path
import numpy as np
from calibrate_m24_e002_a0 import hydra,isaacgym_task_map,omegaconf_to_dict,torch,gymtorch
from dexscrew.algo.ppo.ppo import PPO
from diagnose_m24_e002_a0c import snapshot
from omegaconf import open_dict

@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
    assert cfg.test and int(cfg.task.env.numEnvs)==16
    out=Path(str(cfg.target_audit_out));assert not out.exists()
    out.parent.mkdir(parents=True,exist_ok=True)
    source=Path(str(cfg.e002_calibration));cal=json.loads(source.read_text())
    assert cal['status']=='passed' and cal['pose_index']==743
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
        cfg.task.env.e002_motion.force_sign=cal['force_sign']
    torch.manual_seed(int(cfg.seed));np.random.seed(int(cfg.seed))
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,graphics_device_id=0,headless=True)
    agent=PPO(env,str(out.parent/'inference'),full_config=cfg)
    agent.restore_test(str(cfg.checkpoint));agent.set_eval()
    obs=env.reset();original_pre=env.pre_physics_step;original_reward=env._compute_e002_a_reward
    mode=str(cfg.get('target_audit_mode','torque_limit_over_kp'))
    assert mode in ['torque_limit_over_kp','settle6','gait_only']
    if mode=='gait_only':
        from m24_e002_contact_gait_diagnostic import measure
        kin=json.loads(Path(str(cfg.target_audit_basis)).read_text())
        assert kin['pose_index']==743
        jac=gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim,'hand'))
        coms=[f['com_local'] for f in kin['fingers']]
    clipped_ids=torch.arange(8,16,device=env.device)
    alive=np.ones(16,dtype=bool);records=[[] for _ in range(16)];final=[None]*16
    clipped=np.zeros(16,dtype=int);step=0
    def pre(actions):
        if mode=='gait_only':
            original_pre(actions)
            return
        if mode=='settle6':
            actions=actions.clone()
            ids=clipped_ids[env.progress_buf[clipped_ids]<6]
            actions[ids]=0.
            original_pre(actions)
            return
        original_pre(actions)
        q=env.xhand_hand_dof_pos[clipped_ids]
        limit=env.torque_limit/env.p_gain[clipped_ids]
        old=env.cur_targets[clipped_ids,:16].clone()
        targets=torch.maximum(torch.minimum(old,q+limit),q-limit)
        clipped[8:]+=((targets-old).abs()>1e-6).any(-1).cpu().numpy()
        env.cur_targets[clipped_ids,:16]=targets
        env.prev_targets[clipped_ids,:16]=targets
    def reward(*args,**kwargs):
        result=original_reward(*args,**kwargs)
        data=snapshot(env)
        gait=measure(env,jac,coms) if mode=='gait_only' else None
        torques=env.torques[:,:,:16].detach().cpu().numpy()
        reasons=env.e002_termination_reason.detach().cpu().numpy()
        angles=(env.e002_direction*(env.nut_dof_pos[:,0]-env.e002_theta_start)).detach().cpu().numpy()
        for i in np.where(alive)[0]:
            row={k:v[i] for k,v in data.items()}
            if gait is not None:row['contact_gait']=gait[i]
            gap=np.asarray(row['joint_target'])-np.asarray(row['joint_position'])
            row.update(step=step,angle=float(angles[i]),reason=int(reasons[i]),
                target_error=gap.tolist(),max_target_error=float(abs(gap).max()),
                actuator_saturation_fraction=float((abs(torques[i])>=env.torque_limit-1e-5).mean()),
                action=env.actions[i].detach().cpu().tolist())
            records[i].append(row)
            if reasons[i] or step==99:
                final[i]=dict(env_id=int(i),condition='unchanged' if i<8 else mode,
                    final_angle=float(angles[i]),steps=step+1,reason=int(reasons[i]),
                    max_target_error=max(x['max_target_error'] for x in records[i]),
                    pair_contact_steps=sum(all(x['contact_hysteresis']) for x in records[i]),
                    clipped_steps=int(clipped[i]))
                alive[i]=False
        return result
    env.pre_physics_step=pre;env._compute_e002_a_reward=reward
    with torch.no_grad():
        for step in range(100):
            action,extra=agent._eval_select_action(obs);obs,_,_,_=env.step(action,**extra)
    summary={}
    for condition in ['unchanged',mode]:
        ss=[s for s in final if s['condition']==condition]
        summary[condition]=dict(median_degrees=float(np.median([s['final_angle'] for s in ss])*180/math.pi),
            median_steps=float(np.median([s['steps'] for s in ss])),max_target_error=max(s['max_target_error'] for s in ss),
            termination_counts={str(i):sum(s['reason']==i for s in ss) for i in range(7)})
    report=dict(status='completed:diagnostic_only',training_launched=False,production_control_changed=False,
        checkpoint=str(cfg.checkpoint),checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        control_limit_derived_from=dict(torque_limit=float(env.torque_limit),kp=env.p_gain[0].cpu().tolist()),
        strict_gates_unchanged=True,termination_suppression=False,summary=summary,episodes=final,steps=records)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2));env.gym.destroy_sim(env.sim)
if __name__=='__main__':main()
