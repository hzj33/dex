#!/usr/bin/env python3
"""Actual demo reset helper: subset isolation, actor coordinates and no imported credit."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import torch

def load(name):
 p=Path(__file__).resolve().parents[1]/'dexscrew/tasks'/(name+'.py')
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
class Gym:
 def __init__(self):self.calls=[]
 def __getattr__(self,name):
  def invoke(*args):self.calls.append((name,args[-2].clone(),args[-1]))
  return invoke

def main(return_phase=False, regrasp_phase=False):
 n=4;e=SimpleNamespace(num_envs=n,device='cpu',hand_indices=torch.tensor([0,2,4,6]),object_indices=torch.tensor([1,3,5,7]),sim=None)
 e.e002_cycle=load('e002_cycle').E002Cycle(n,'cpu',{})
 cfg=dict(source='outputs/local_pipeline_m24_e002_cycle_20260910/release_reference_video/reset_seed_candidates.json',seed_steps=[43],fraction=1.)
 if return_phase:
  cfg.update(source='outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/return_seed_candidates.json',seed_steps=[34],phase=2,release_only=False,return_only=True)
  e.e002_contact_force_off=.15
 if regrasp_phase:
  cfg.update(source='outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911/actual_return_seed.json',seed_steps=[30],phase=3,release_only=False,return_only=False)
 d=load('e002_demo_resets').E002DemoResets(e,cfg)
 e.root_state_tensor=d.roots[0].repeat(n,1)
 for i in range(n):e.root_state_tensor[2*i:2*i+2,:3]+=torch.tensor([i*2.,i*3.,0.])
 e.dof_state=torch.randn(n*17,2);e.prev_targets=torch.randn(n,17);e.cur_targets=e.prev_targets.clone();e.init_pose_buf=torch.zeros(n,17)
 e.obs_buf_lag_history=torch.zeros((n,)+d.history.shape[1:]);e.e002_contact_state=torch.zeros(n,2,dtype=torch.bool)
 e.e002_contact_acquired=torch.zeros(n,dtype=torch.bool)
 for name in ['e002_theta_prev','e002_high_water','e002_paid_high_water']:setattr(e,name,torch.zeros(n))
 e.priv_info_buf=torch.randn(n,16);priv_before=e.priv_info_buf.clone()
 def update_priv(ids,name,value):e.priv_info_buf[ids]=value
 e._update_priv_buf=update_priv;e.gym=Gym()
 before=e.dof_state.view(n,17,2).clone();root_before=e.root_state_tensor.clone()
 d.apply(e,torch.tensor([0,2]),SimpleNamespace(unwrap_tensor=lambda t:t))
 assert torch.equal(e.priv_info_buf,priv_before), 'Intermediate reset changed the original episode hand-pose context'
 assert torch.equal(e.dof_state.view(n,17,2)[[1,3]],before[[1,3]])
 assert torch.equal(e.root_state_tensor[[2,3,6,7]],root_before[[2,3,6,7]])
 assert torch.allclose(e.root_state_tensor[4,:3]-d.roots[0,0,:3],torch.tensor([4.,6.,0.]))
 assert torch.equal(e.obs_buf_lag_history[2],d.history[0])
 assert e.e002_cycle.phase.tolist()==([3,0,3,0] if regrasp_phase else ([2,0,2,0] if return_phase else [1,0,1,0]))
 assert not e.e002_cycle.last_event.any() and not e.e002_cycle.return_events.any()
 assert not e.e002_cycle.drive_events.any() and not e.e002_cycle.release_events.any()
 assert torch.equal(e.e002_cycle.high[[0,2]],e.e002_cycle.paid[[0,2]])
 assert torch.equal(e.e002_cycle.previous[[0,2]],e.dof_state.view(n,17,2)[[0,2],16,0])
 call=next(c for c in e.gym.calls if c[0]=='set_dof_state_tensor_indexed')
 assert call[1].tolist()==[0,4,1,5] and call[2]==4
 print('Demo subset isolation, world offsets, complete DOF actor list, history canonical episode context and no imported event/progress credit: OK')
if __name__=='__main__':
 main()
 if '--return' in __import__('sys').argv:main(True)

 if '--regrasp' in __import__('sys').argv:main(regrasp_phase=True)
