from test_m24_e002_cycle import module
import torch,math

def run(scale,axis,forces,moments):
 c=module.E002Cycle(1,'cpu',dict(normal_balance_scale=scale,normal_moment_reference=.05))
 c.reference[:]=torch.tensor([[[1.,0,0],[-1.,0,0]]]);c.reference_valid[:]=True
 r,reason=c.update(torch.zeros(1),torch.ones(1,2,dtype=torch.bool),torch.ones(1,2),torch.full((1,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]),torch.tensor([axis]),torch.ones(1),torch.ones(1,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15,normal_force_vectors=torch.tensor([forces]),normal_moment_vectors=torch.tensor([moments]))
 return r.item(),reason.tolist(),c.phase.tolist()
for axis,f,m,cost in [([0.,0,1],[[5.,0,0],[-5.,0,0]],[[0.,0,.1],[0.,0,.1]],0),([0.,0,1],[[5.,0,0],[0.,0,0]],[[0.,.05,0],[0.,0,0]],.05),([1.,0,0],[[0.,5.,0],[0.,0,0]],[[0.,0,.05],[0.,0,0]],.05)]:
 a,b=run(0,axis,f,m),run(.025,axis,f,m);assert abs(a[0]-b[0]-cost)<1e-6;assert a[1:]==b[1:]
print('PASS opposing forces cancel, axial drive torque unpenalized, transverse/tilt cost rotation consistency, gates unchanged')
