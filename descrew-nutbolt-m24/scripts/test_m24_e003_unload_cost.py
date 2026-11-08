from test_m24_e002_cycle import module
import torch,math

def run(phase,forces,scale):
 c=module.E002Cycle(1,'cpu',dict(unload_balance_scale=scale,unload_force_reference=5.))
 c.phase[:]=phase;c.reference[:]=torch.tensor([[[1.,0,0],[-1.,0,0]]]);c.reference_valid[:]=True
 f=torch.tensor([forces]);r,reason=c.update(torch.zeros(1),f>.15,f,torch.full((1,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]),torch.tensor([[0.,0,1.]]),torch.ones(1),torch.ones(1,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15)
 return r.item(),reason.tolist(),c.phase.tolist(),c.last_event.tolist()
for phase in range(4):
 for force in ([10.,0.],[0.,10.],[5.,5.],[0.,0.]):
  a,b=run(phase,force,0),run(phase,force,.1)
  expected=.1*abs(force[0]-force[1])/5 if phase==1 else 0
  assert abs(a[0]-b[0]-expected)<1e-6
  assert a[1:]==b[1:]
print('PASS release-only symmetric cost, zero equal-load cost, unchanged transition gates')
