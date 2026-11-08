"""Check physical-action scope and gradients of the optional E003 bounds cost."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dexscrew.algo.ppo.action_bounds import nonreturn_active_bound_loss
mu=torch.tensor([[3.,-2.,1.,-.5]*4]*4,requires_grad=True)
mask=torch.tensor([True,True,False,True]);loss=nonreturn_active_bound_loss(mu,mask)
active=mu[:,[0,1,2,3,12,13,14,15]]
expected=((active-1.1).clamp(min=0).square()+(active+1.1).clamp(max=0).square()).sum(-1)*mask
torch.testing.assert_close(loss,expected)
loss.mean().backward();assert (mu.grad[2]==0).all() and (mu.grad[:,4:12]==0).all()
assert mu.grad[0,0]>0 and mu.grad[0,1]<0 and mu.grad[0,2]==0
assert torch.isfinite(mu.grad).all()
print('PASS rl_games formula equivalence; return/inactive/interior gradients zero; outside gradients restore executable range')
