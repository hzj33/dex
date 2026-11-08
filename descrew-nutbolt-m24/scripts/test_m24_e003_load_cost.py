"""Compare actual rewards: static squeeze costs, release handling, legacy gates."""
from test_m24_e002_cycle import module
import torch,math

def run(phase,forces,contacts,scale):
 c=module.E002Cycle(1,'cpu',{'load_cost_scale':scale,'load_force_reference':20.,'unpaired_load_multiplier':2.})
 c.phase[:]=phase;c.reference[:]=torch.tensor([[[1.,0,0],[-1.,0,0]]]);c.reference_valid[:]=True
 r,reason=c.update(torch.zeros(1),torch.tensor([contacts]),torch.tensor([forces]),torch.full((1,2),.04),
 torch.tensor([[[.04,0,0],[-.04,0,0]]]),torch.tensor([[0.,0,1.]]),torch.ones(1),torch.ones(1,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15)
 return r.item(),reason.tolist(),c.phase.tolist(),c.last_event.tolist()
for phase,forces,contacts,cost in [(0,[20.,20.],[True,True],.2),(0,[20.,0.],[True,False],.2),(1,[20.,0.],[True,False],.1),(2,[0.,0.],[False,False],0.),(0,[10.,10.],[True,True],.05)]:
 a=run(phase,forces,contacts,0);b=run(phase,forces,contacts,.1)
 assert abs((a[0]-b[0])-cost)<1e-6,(a,b,cost)
 assert a[1:]==b[1:],'Load objective modified phase/gate semantics'
print('PASS: actual static-load costs, quadratic scaling, drive-only asymmetry, zero-load cost, unchanged phase gates')
