"""Final user-authorized gate relaxation retains contact, force cost and old clock."""
from test_m24_e002_cycle import module
import torch,math

def drive(relaxed,contact,scale=.25):
 c=module.E002Cycle(1,'cpu',dict(event_dwell_steps=1,require_paired_drive_progress=not relaxed,force_excess_scale=scale,force_limit=5.))
 theta=torch.tensor([c.target]);c.previous[:]=theta-.01
 r,_=c.update(theta,torch.tensor([contact]),torch.tensor([[6.,0.]]),torch.full((1,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]),torch.tensor([[0.,0,1.]]),torch.ones(1),torch.ones(1,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15)
 return c,r
assert drive(False,[True,False])[0].phase.item()==0
assert drive(True,[True,False])[0].phase.item()==1
assert drive(True,[False,False])[0].phase.item()==0
assert abs((drive(True,[True,False],0)[1]-drive(True,[True,False],.25)[1]).item()-.05)<1e-5
old=module.E002Cycle(4,'cpu',dict(phase_deadlines=[160,80,120,80],phase_clock_scale=.5))
new=module.E002Cycle(4,'cpu',dict(phase_deadlines=[320,160,240,160],phase_clock_scale=.25))
for c in [old,new]:c.phase[:]=torch.arange(4);c.phase_step[:]=17
torch.testing.assert_close(old.command(),new.command(),rtol=0,atol=0)
print('PASS relaxed drive still needs real angle/contact; original paired gate default; force excess unchanged; learned phase clock exact')
