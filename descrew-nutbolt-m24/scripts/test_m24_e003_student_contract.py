"""Teacher-target parity and absence of privileged data in student actions."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from dexscrew.algo.models.models import ActorCritic

torch.manual_seed(42)
config=dict(input_shape=(101,),actions_num=16,actor_units=[32,16],
            priv_mlp_units=[16,8],priv_info_dim=11,priv_info=True,
            proprio_adapt=False,proprio_dim=32,proprio_len=30,
            point_mlp_units=[32,32],use_point_cloud_info=True)
teacher=ActorCritic(config).eval()
student=ActorCritic(dict(config,proprio_adapt=True,teacher_point_tanh=True)).eval()
student.load_state_dict(teacher.state_dict(),strict=False)
inputs=dict(obs=torch.randn(8,101),priv_info=torch.randn(8,11),
            proprio_hist=torch.randn(8,30,32),point_cloud_info=torch.randn(8,100,3))
with torch.no_grad():
    tm,_,_,te,_=teacher._actor_critic(inputs)
    sm,_,_,_,target=student._actor_critic(inputs)
    inferred_teacher=student.mu(student.actor_mlp(torch.cat([inputs['obs'],target],-1)))
    torch.testing.assert_close(target,te,rtol=0,atol=0)
    torch.testing.assert_close(inferred_teacher,tm,rtol=0,atol=0)
    # Inference cannot depend on privileged labels or point-cloud values.
    plain={k:v for k,v in inputs.items() if k!='priv_info'}
    plain['point_cloud_info']=torch.full_like(plain['point_cloud_info'],17.)
    torch.testing.assert_close(student.act_inference(plain)[0],sm,rtol=0,atol=0)
    student.teacher_point_tanh=False
    legacy=student._actor_critic(inputs)[-1]
    assert (legacy-target).abs().max()>.01, 'Fixture must detect the original target mismatch'
print('PASS: exact teacher action/latent targets; student actions independent of privileged/point inputs; legacy mismatch detected')
