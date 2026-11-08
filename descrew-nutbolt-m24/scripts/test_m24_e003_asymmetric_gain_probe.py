"""Exercise the evaluator's actual optional gain wrapper without simulation."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
source=ast.parse((Path(__file__).parent/'evaluate_m24_e002_cycle.py').read_text())
fn=next(n for n in ast.walk(source) if isinstance(n,ast.FunctionDef) and n.name=='asymmetric_control')
for gain in [1.2,2.4]:
    env=SimpleNamespace(p_gain=torch.full((1,16),2.4),e002_cycle=SimpleNamespace(phase=torch.zeros(1,dtype=torch.long)))
    seen=[];scope=dict(torch=torch,env=env,index_compliance_gain=gain,base_index_gain=env.p_gain[:,:4].clone(),low_level=lambda step:seen.append(env.p_gain.clone()))
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-evaluator-wrapper>','exec'),scope)
    for phase in [0,1,2,3,0]:
        env.e002_cycle.phase[:]=phase;scope['asymmetric_control'](0)
        expected=torch.full((1,16),2.4);expected[:,:4]=gain if phase<2 else 2.4
        torch.testing.assert_close(env.p_gain,expected,rtol=0,atol=0)
        torch.testing.assert_close(seen[-1],expected,rtol=0,atol=0)
print('PASS actual wrapper modifies only index drive/release gains, restores return/regrasp, base-gain override exact no-op')
