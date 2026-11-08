from test_m24_e002_cycle import module
import torch,math

def run(phase,forces,scale,peak=None):
 c=module.E002Cycle(1,'cpu',dict(force_excess_scale=scale,force_limit=5.))
 c.phase[:]=phase;c.reference[:]=torch.tensor([[[1.,0,0],[-1.,0,0]]]);c.reference_valid[:]=True
 f=torch.tensor([forces]);r,reason=c.update(torch.zeros(1),f>.15,f,torch.full((1,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]),torch.tensor([[0.,0,1.]]),torch.ones(1),torch.ones(1,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15,force_cost_signal=None if peak is None else torch.tensor([peak]))
 return r.item(),reason.tolist(),c.phase.tolist(),c.last_event.tolist()
for phase in range(4):
 for force in ([10.,0.],[0.,10.],[10.,10.],[5.,5.],[0.,0.]):
  a,b=run(phase,force,0),run(phase,force,.1)
  expected=.1*sum(max(0,f-5)/5 for f in force)
  assert abs(a[0]-b[0]-expected)<1e-6
  assert a[1:]==b[1:]
print('PASS all-phase 5N excess cost; bilateral squeeze charged; sub-limit zero; unchanged gates')

for phase in range(4):
 a=run(phase,[1.,1.],.1);b=run(phase,[1.,1.],.1,[10.,5.])
 assert abs(a[0]-b[0]-.1)<1e-6 and a[1:]==b[1:]
 a=run(phase,[10.,1.],.1);b=run(phase,[10.,1.],.1,[1.,1.])
 assert a==b
print('PASS interval peak charged independently of endpoint; endpoint floor and gates preserved')
