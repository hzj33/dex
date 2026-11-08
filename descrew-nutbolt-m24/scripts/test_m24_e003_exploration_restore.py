"""Exercise actual warm-start method: means unchanged, noise/reference explicit."""
import ast,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
s=ast.parse(Path('dexscrew/algo/ppo/ppo.py').read_text());cls=next(x for x in s.body if isinstance(x,ast.ClassDef) and x.name=='PPO');fn=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='restore_train');ns={'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'actual_PPO_restore_train','exec'),ns)
config=dict(input_shape=(101,),actions_num=16,actor_units=[32,16],priv_mlp_units=[16,8],priv_info_dim=11,priv_info=True,proprio_adapt=False,proprio_dim=32,proprio_len=30,point_mlp_units=[32,32],use_point_cloud_info=True)
model=ActorCritic(config).eval();norms=dict(running_mean_std=RunningMeanStd((101,)),priv_mean_std=RunningMeanStd(11),point_cloud_mean_std=RunningMeanStd(3),value_mean_std=RunningMeanStd(1));batch=dict(obs=torch.randn(8,101),priv_info=torch.randn(8,11),point_cloud_info=torch.randn(8,100,3),proprio_hist=torch.randn(8,30,32));expected=model.act_inference(batch)[0].detach();original={k:v.clone() for k,v in model.state_dict().items()}
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'source.pth';torch.save(dict(model=original,**{k:v.state_dict() for k,v in norms.items()}),p);source_bytes=p.read_bytes()
 for scale in [1.,.25]:
  a=SimpleNamespace(model=model,obs_shape=(101,),actions_num=16,normalize_priv=True,normalize_point_cloud=True,normalize_value=True,return_anchor_coef=1.,init_action_std_scale=scale,**norms)
  ns['restore_train'](a,p)
  torch.testing.assert_close(a.model.act_inference(batch)[0],expected,rtol=0,atol=0)
  torch.testing.assert_close(a.model.sigma.exp(),original['sigma'].exp()*scale)
  torch.testing.assert_close(a.return_anchor.model.sigma,a.model.sigma,rtol=0,atol=0)
  assert all(torch.equal(v,a.model.state_dict()[k]) for k,v in original.items() if k!='sigma')
  assert p.read_bytes()==source_bytes
print('PASS actual restore preserves deterministic means/source checkpoint; explicit std scaling; frozen reference matches adjusted exploration; default exact')
