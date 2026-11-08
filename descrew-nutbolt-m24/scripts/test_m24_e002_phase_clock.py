#!/usr/bin/env python3
"""Regression: extended termination must preserve checkpoint clock inputs."""
from test_m24_e002_cycle import module
import torch,math

def main():
 base=module.E002Cycle(4,'cpu',{})
 extended=module.E002Cycle(4,'cpu',{'phase_deadlines':[160,80,120,80],'phase_clock_scale':.5})
 for step in (0,1,39,40,59,60,79,80,100,160):
  for c in (base,extended):c.phase[:]=torch.arange(4);c.phase_step[:]=step
  assert torch.equal(base.command(),extended.command())
 for c in (base,extended):c.reset(torch.arange(4));c.phase[:]=1
 def update(c,off):
  contact=torch.full((4,2),not off);force=contact.float()
  return c.update(torch.zeros(4),contact,force,torch.full((4,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]*4),torch.tensor([[0.,0.,1.]]*4),contact.all(-1).float(),torch.ones(4,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15)
 for _ in range(40):
  _,old_reason=update(base,False);_,new_reason=update(extended,False)
 assert old_reason.tolist()==[11]*4 and new_reason.tolist()==[0]*4
 for _ in range(9):update(extended,False)
 update(extended,True);assert extended.phase.tolist()==[1]*4
 _,reason=update(extended,True)
 assert extended.phase.tolist()==[2]*4 and reason.tolist()==[0]*4
 assert extended.release_events.tolist()==[1]*4
 extended.reset(torch.arange(4));assert extended.deadlines.tolist()==[160,80,120,80] and extended.phase_clock_scale==.5
 print('PASS: exact phase-clock inputs, delayed real release, unchanged dwell, reset-safe deadlines')
if __name__=='__main__':main()
