#!/usr/bin/env python3
"""Exercise actual PPO restore_train without importing the simulator."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from dexscrew.algo.models.running_mean_std import RunningMeanStd

source = Path('dexscrew/algo/ppo/ppo.py').read_text()
cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == 'PPO')
restore = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'restore_train')
namespace = {'torch':torch}
exec(compile(ast.Module(body=[restore], type_ignores=[]), 'actual_PPO_restore_train', 'exec'), namespace)
path = 'outputs/local_pipeline_m24_e002_cycle_20260910/cycle_warm_start.pth'
weights = torch.load(path, map_location='cpu')
class Sink:
    def load_state_dict(self,state):self.state=state
stats = RunningMeanStd((1,));stats.eval()
reference = RunningMeanStd((1,));reference.load_state_dict(weights['value_mean_std']);reference.eval()
x = torch.tensor([[-2.],[0.],[2.]])
expected = reference(x,unnorm=True)
wrong = stats(x,unnorm=True)
agent = SimpleNamespace(model=Sink(), running_mean_std=Sink(),normalize_priv=False,
    normalize_point_cloud=False,normalize_value=True,value_mean_std=stats)
namespace['restore_train'](agent,path)
actual = stats(x,unnorm=True)
assert not torch.allclose(wrong,expected)
assert torch.equal(actual,expected)
assert all(torch.equal(stats.state_dict()[k],v) for k,v in weights['value_mean_std'].items())
print(json.dumps(dict(normalized_values=x.flatten().tolist(),expected=expected.flatten().tolist(),
 before=wrong.flatten().tolist(),after=actual.flatten().tolist(),
 maximum_error_before=float((wrong-expected).abs().max()),maximum_error_after=float((actual-expected).abs().max()),passed=True),indent=2))
