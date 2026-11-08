#!/usr/bin/env python3
"""Actual vectorized cycle tests without loading a simulator."""
import importlib.util
import math
from pathlib import Path
import torch

spec = importlib.util.spec_from_file_location('e002_cycle', Path(__file__).resolve().parents[1] / 'dexscrew/tasks/e002_cycle.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    c = module.E002Cycle(2, 'cpu', {})
    def step(theta, sector=0, contacts=(True, True)):
        a = math.radians(sector)
        radial = torch.tensor([[[.04*math.cos(a),.04*math.sin(a),0],[-.04*math.cos(a),-.04*math.sin(a),0]]]*2)
        contact = torch.tensor([contacts]*2)
        return c.update(torch.full((2,), math.radians(theta)), contact, contact.float(),
            torch.full((2,2),.04), radial, torch.tensor([[0.,0.,1.]]*2),
            contact.all(-1).float(), torch.ones(2,dtype=torch.bool), math.radians(3), .075,
            math.radians(15), .15)
    step(0)
    for end in [20, 40]:
        for theta in [end-15,end-10,end-5,end,end]:
            step(theta,20)
        assert c.phase.tolist() == [1,1]
        step(end,20,(False,True))
        assert c.phase.tolist() == [1,1]  # One finger off is not pair release.
        step(end,20,(False,False)); step(end,20,(False,False))
        assert c.phase.tolist() == [2,2]
        step(end,20,(False,False))
        assert c.phase.tolist() == [2,2]  # Holding open away from return sector is not return.
        step(end,0,(False,False)); step(end,0,(False,False))
        assert c.phase.tolist() == [3,3]
        step(end); reward, reason = step(end)
    assert c.cycles.tolist() == [2,2] and reason.tolist() == [5,5]
    assert torch.allclose(c.paid, torch.full((2,),math.radians(40)))
    before = c.cycles[1].clone()
    c.reset(torch.tensor([0]))
    assert c.cycles[0] == 0 and c.cycles[1] == before and bool((c.deadlines > 0).all())
    c.reset(torch.arange(2)); step(0)
    c.phase[:] = 1
    for _ in range(40):
        _, reason = step(0)
    assert reason.tolist() == [11,11] and c.phase.tolist() == [1,1] and not c.cycles.any()
    c.reset(torch.arange(2)); step(0); step(10); step(20)
    paid = c.paid.clone()
    step(10); reward, _ = step(20)
    assert torch.equal(paid,c.paid) and c.drive_events.max() == 0
    c.reset(torch.arange(2)); step(0)
    _, reason = step(30)
    assert reason.tolist() == [4,4] and not c.drive_events.any()
    # Relative unloading supplies early feedback with the same potential range,
    # and neither unloading nor a force oscillation manufactures phase events.
    relative = module.E002Cycle(2, 'cpu', {'release_potential':'initial_relative_force'})
    def force_step(theta, forces):
        f=torch.tensor([forces]*2,dtype=torch.float32)
        contact=f>.15
        return relative.update(torch.full((2,),math.radians(theta)),contact,f,
            torch.full((2,2),.04),torch.tensor([[[.04,0,0],[-.04,0,0]]]*2),
            torch.tensor([[0.,0.,1.]]*2),contact.all(-1).float(),
            torch.ones(2,dtype=torch.bool),math.radians(3),.075,math.radians(15),.15)
    for theta in (0,5,10,15,20,20):force_step(theta,(10,20))
    assert relative.phase.tolist()==[1,1]
    assert torch.equal(relative.release_entry_force,torch.tensor([[10.,20.]]*2))
    r,_=force_step(20,(8,16));assert torch.allclose(r,torch.full((2,),.1))
    r,_=force_step(20,(10,20));assert torch.allclose(r,torch.full((2,),-.1))
    r,_=force_step(20,(0,0));assert torch.allclose(r,torch.full((2,),.5))
    r,_=force_step(20,(0,0));assert relative.phase.tolist()==[2,2]
    relative.reset(torch.tensor([0]))
    assert not relative.release_entry_force[0].any() and relative.release_entry_force[1].sum()==30
    print('Cycle order, two-finger release, actual return, second drive, reset, timeout and no recycled credit: OK')
    print('Relative unloading: early feedback, fixed potential range, no force-oscillation bonus, partial reset: OK')


if __name__ == '__main__':
    main()
