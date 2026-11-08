#!/usr/bin/env python3
"""Exercise actual E002 perturbations: independent samples, mask, bounds and RNG."""
import importlib.util
from pathlib import Path
import torch
p=Path(__file__).resolve().parents[1]/'dexscrew/tasks/e002_initial_state.py'
spec=importlib.util.spec_from_file_location('e002_initial_state',p)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
q=torch.full((512,16),.5);lo=torch.zeros(16);hi=torch.ones(16)
torch.manual_seed(43);state=torch.random.get_rng_state().clone()
assert m.perturb_initial_joints(q,lo,hi,0.,.5) is q
assert torch.equal(state,torch.random.get_rng_state())
r=m.perturb_initial_joints(q,lo,hi,.02,0.)
assert torch.equal(r[:,4:12],q[:,4:12]) and (r-q).abs().max()<=.020001
assert torch.unique(r,dim=0).shape[0]==512, 'Independent environments share perturbations'
assert torch.equal(q,torch.full_like(q,.5))
r=m.perturb_initial_joints(q,lo,hi,.02,.5)
n=int((r==q).all(-1).sum());assert 150<n<360,n
q[:,0]=.9999;r=m.perturb_initial_joints(q,lo,hi,.02,0.)
assert (r>=lo).all() and (r<=hi).all()
assert torch.equal(m.perturb_initial_joints(q,lo,hi,.02,1.),q)
print('Independent samples, inactive fingers, joint limits, nominal mixture and zero-noise RNG preservation: OK')
