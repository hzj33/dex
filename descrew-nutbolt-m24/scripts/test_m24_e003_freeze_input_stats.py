"""Execute actual PPO mode switch with real normalization modules."""
import ast,sys
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dexscrew.algo.models.running_mean_std import RunningMeanStd
s=ast.parse(Path('dexscrew/algo/ppo/ppo.py').read_text());cls=next(x for x in s.body if isinstance(x,ast.ClassDef) and x.name=='PPO');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='set_train');ns={};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_PPO_set_train','exec'),ns)
for freeze in [False,True]:
 a=SimpleNamespace(model=torch.nn.Linear(3,2),normalize_input=True,normalize_priv=True,normalize_point_cloud=True,normalize_value=True,freeze_input_stats=freeze,running_mean_std=RunningMeanStd(3),priv_mean_std=RunningMeanStd(2),point_cloud_mean_std=RunningMeanStd(3),value_mean_std=RunningMeanStd(1))
 for _ in range(3):
  ns['set_train'](a)
  for n,size in [('running_mean_std',3),('priv_mean_std',2),('point_cloud_mean_std',3),('value_mean_std',1)]:
   m=getattr(a,n);before={k:v.clone() for k,v in m.state_dict().items()};m(torch.randn(8,size))
   if freeze and n!='value_mean_std':assert all(torch.equal(v,m.state_dict()[k]) for k,v in before.items()),n
   else:assert m.count>before['count'],n
  a.model(a.running_mean_std(torch.randn(8,3))).square().mean().backward()
  assert a.model.training and a.model.weight.grad is not None and torch.isfinite(a.model.weight.grad).all()
print('PASS frozen input/priv/point statistics remain exact across epochs; critic statistics and actor learn; default updates retained')
