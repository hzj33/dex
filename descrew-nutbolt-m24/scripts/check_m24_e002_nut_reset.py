#!/usr/bin/env python3
"""Check that subset episode resets clear the nut DOF as well as hand DOFs."""
import json
from pathlib import Path
from calibrate_m24_e002_a0 import hydra,isaacgym_task_map,omegaconf_to_dict,torch,gymtorch
from omegaconf import open_dict
@hydra.main(config_name='config',config_path='../configs')
def main(cfg):
 out=Path(str(cfg.reset_check_out));assert not out.exists()
 cal=json.loads(Path(str(cfg.reset_calibration)).read_text())
 with open_dict(cfg.task.env):cfg.task.env.customInitDofPos=cal['custom_init_dof_pos']
 env=isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),sim_device=cfg.sim_device,graphics_device_id=0,headless=True)
 assert env.num_envs==4 and env.num_dofs==17
 env.reset()
 state=env.dof_state.view(4,17,2)
 state[:,16,0]=torch.tensor([.4,.35,.3,.25],device=env.device)
 state[:,16,1]=torch.tensor([.5,.4,.3,.2],device=env.device)
 env.gym.set_dof_state_tensor(env.sim,gymtorch.unwrap_tensor(env.dof_state));env._refresh_gym()
 before=state[:,16].clone()
 ids=torch.tensor([0,2],device=env.device,dtype=torch.long)
 env.reset_idx(ids);env._refresh_gym();after=state[:,16].clone()
 report=dict(diagnostic_only=True,before=before.tolist(),after_reset=after.tolist(),
  reset_ids=[0,2],reset_max_abs=float(after[ids].abs().max()),
  untouched_max_change=float((after[[1,3]]-before[[1,3]]).abs().max()))
 report['passed']=report['reset_max_abs']<1e-6 and report['untouched_max_change']<1e-6
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
 env.gym.destroy_sim(env.sim)
if __name__=='__main__':main()
