#!/usr/bin/env python3
"""Diagnostic replay of recorded release states; never trains or reports goal success."""
import json
import hashlib
from pathlib import Path
import numpy as np
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch, gymtorch
from omegaconf import open_dict


@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
    out=Path(str(cfg.replay_out));assert not out.exists() and int(cfg.task.env.numEnvs)==1
    source=Path(str(cfg.replay_states));data=json.loads(source.read_text())
    trace=json.loads(Path(data['source_trace']).read_text())
    assert hashlib.sha256(Path(data['source_trace']).read_bytes()).hexdigest()==data['source_sha256']
    cal=json.loads(Path(str(cfg.replay_calibration)).read_text())
    with open_dict(cfg.task.env):cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
    torch.manual_seed(42);np.random.seed(42)
    env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,
                                     graphics_device_id=0,headless=True)
    all_results=[];original=env.compute_reward;recorded=[]
    def record(*args):
        original(*args)
        recorded.append(dict(angle=float(env.nut_dof_pos[0,0]),phase=int(env.e002_cycle.phase[0]),
            event=int(env.e002_cycle.last_event[0]),reason=int(env.e002_cycle.last_reason[0]),
            force=torch.norm(env.e002_normal_contact_force[0],dim=-1).tolist(),
            contact=env.e002_contact_state[0].tolist()))
    env.compute_reward=record
    with torch.no_grad():
        for seed in data['states']:
            env.reset();state=seed['reset_seed_state']
            env.root_state_tensor[:]=torch.tensor(state['actor_root_state'],device=env.device)
            env.dof_state.view(1,-1,2)[0]=torch.tensor(state['dof_state'],device=env.device)
            target=torch.tensor(state['joint_targets'],device=env.device)
            env.prev_targets[0]=target;env.cur_targets[0]=target
            env.gym.set_actor_root_state_tensor(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor))
            env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state))
            env.gym.set_dof_position_target_tensor(env.sim,gymtorch.unwrap_tensor(env.cur_targets))
            for k,v in state['cycle'].items():
                getattr(env.e002_cycle,k)[0]=torch.tensor(v,device=env.device,dtype=getattr(env.e002_cycle,k).dtype)
            env.e002_contact_state[0]=torch.tensor(seed['contact'],device=env.device)
            env.e002_contact_acquired[0]=True
            env.obs_buf_lag_history[0]=torch.tensor(state['observation_history_before_roll'],device=env.device)
            env.at_reset_buf[0]=0;env.compute_observations()
            recorded.clear();expected=[]
            for row in trace['rows']:
                if row['step']<=seed['step']:continue
                expected.append(row)
                env.step(torch.tensor([row['action']],device=env.device))
                if recorded[-1]['reason'] or recorded[-1]['event']==2:break
            assert len(recorded)==len(expected)
            angle_errors=np.degrees([a['angle']-b['angle'] for a,b in zip(recorded,expected)])
            force_errors=np.array([a['force'] for a in recorded])-np.array([b['force'] for b in expected])
            all_results.append(dict(seed_step=seed['step'],steps=recorded,
                max_abs_angle_error_degrees=float(np.max(np.abs(angle_errors))),
                max_abs_force_error=float(np.max(np.abs(force_errors))),
                released=bool(recorded[-1]['event']==2),termination=recorded[-1]['reason'],
                final_net_degrees=float(np.degrees(recorded[-1]['angle']))))
    finite=all(np.isfinite([r['max_abs_angle_error_degrees'],r['max_abs_force_error'],r['final_net_degrees']]).all() for r in all_results)
    result=dict(diagnostic_only=True,goal_completed=False,finite=bool(finite),
        source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        note='Tensor-state reconstruction excludes PhysX contact solver caches; differences are reported, not assumed zero.',
        seeds=all_results,release_count=sum(r['released'] for r in all_results))
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='seeds'},indent=2))
    print([(r['seed_step'],r['released'],r['max_abs_angle_error_degrees'],r['max_abs_force_error']) for r in all_results])
    env.gym.destroy_sim(env.sim)


if __name__=='__main__':main()
