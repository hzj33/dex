"""Exercise the actual task accumulation method without starting IsaacGym."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
import torch
s=ast.parse(Path('dexscrew/tasks/xhand_pasini.py').read_text());cls=next(c for c in s.body if isinstance(c,ast.ClassDef) and any(isinstance(m,ast.FunctionDef) and m.name=='_e003_collect_peak_contact_force' for m in c.body));fn=next(m for m in cls.body if isinstance(m,ast.FunctionDef) and m.name=='_e003_collect_peak_contact_force');ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_task_method','exec'),ns);collect=ns[fn.name]
e=SimpleNamespace(num_envs=1,device='cpu',hand_rb_names=['right_index_link_3','right_index_link_2','right_thumb_link_3'],contact_forces=torch.zeros(1,3,3),gym=SimpleNamespace(refresh_net_contact_force_tensor=lambda _:None),sim=None)
collect(e,True)
e.contact_forces[0,:,0]=torch.tensor([6.,2.,3.]);collect(e,False)
e.contact_forces[0,:,0]=torch.tensor([1.,0.,1.]);collect(e,False)
torch.testing.assert_close(e.e003_peak_contact_force,torch.tensor([[8.,3.]]))
collect(e,True);torch.testing.assert_close(e.e003_peak_contact_force,torch.zeros(1,2))
print('PASS whole-finger intermediate peak retained, final low sample cannot hide it, next interval resets')
