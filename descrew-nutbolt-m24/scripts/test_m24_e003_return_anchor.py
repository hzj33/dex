import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from dexscrew.algo.ppo.return_anchor import ReturnAnchor

torch.manual_seed(42)
config=dict(input_shape=(101,),actions_num=16,actor_units=[32,16],priv_mlp_units=[16,8],priv_info_dim=11,priv_info=True,proprio_adapt=False,proprio_dim=32,proprio_len=30,point_mlp_units=[32,32],use_point_cloud_info=True)
a=SimpleNamespace(obs_shape=(101,),actions_num=16,model=ActorCritic(config).eval(),running_mean_std=RunningMeanStd((101,)).eval(),priv_mean_std=RunningMeanStd(11).eval(),point_cloud_mean_std=RunningMeanStd(3).eval(),normalize_priv=True,normalize_point_cloud=True)
r=ReturnAnchor(a);obs=torch.randn(8,101);obs[:,96:100]=torch.eye(4).repeat(2,1);priv=torch.randn(8,11);points=torch.randn(8,100,3);hist=torch.randn(8,30,32)
mask,target=r.target(obs,priv,points,hist)
expected=a.model.act_inference(dict(obs=a.running_mean_std(obs),priv_info=a.priv_mean_std(priv),point_cloud_info=a.point_cloud_mean_std(points.reshape(-1,3)).reshape_as(points),proprio_hist=hist))[0]
torch.testing.assert_close(expected[mask],target)
mu=expected.detach().clone().requires_grad_();loss=r.loss(mu,mask,target);assert loss.item()<1e-12
with torch.no_grad():mu[mask]+=1
r.loss(mu,mask,target).backward();assert (mu.grad[~mask]==0).all();assert (mu.grad[:,4:12]==0).all();assert mu.grad.abs().sum()>0
with torch.no_grad():a.model.mu.bias.add_(5);a.running_mean_std.train();a.running_mean_std(obs+10)
torch.testing.assert_close(r.target(obs,priv,points,hist)[1],target,rtol=0,atol=0)
assert all(p.grad is None and not p.requires_grad for p in r.model.parameters())
print('PASS exact frozen actor/normalization target; raw return mask; only active return actions receive gradients')

# Verify analytic KL against PyTorch distributions, including variance changes.
mu=expected.detach().clone().requires_grad_()
sigma=(r.model.sigma.exp().detach().expand_as(mu)*1.3).clone().requires_grad_()
active=[0,1,2,3,12,13,14,15]
reference=torch.distributions.Normal(target[:,active],r.model.sigma.exp()[active])
student=torch.distributions.Normal(mu[mask][:,active],sigma[mask][:,active])
kl=torch.distributions.kl_divergence(reference,student).mean()
torch.testing.assert_close(r.loss(mu,mask,target,sigma,"kl"),kl)
r.loss(mu,mask,target,sigma,"kl").backward()
assert (sigma.grad[~mask]==0).all() and (sigma.grad[:,4:12]==0).all()
assert sigma.grad.abs().sum()>0
print('PASS teacher-to-student Gaussian KL matches torch.distributions; phase/action gradient mask')

shared_sigma=(r.model.sigma.exp().detach()*1.3).requires_grad_()
torch.testing.assert_close(r.loss(mu,mask,target,shared_sigma,"kl"),kl)
r.loss(mu,mask,target,shared_sigma,"kl").backward()
assert shared_sigma.grad.abs().sum()>0 and (shared_sigma.grad[4:12]==0).all()
print('PASS actual PPO shared 16-vector sigma broadcasts across minibatch')
