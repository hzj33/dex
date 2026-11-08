"""Execute production pre_physics_step against blocked-joint/reversal fixtures."""
import ast
from pathlib import Path
from types import SimpleNamespace
import torch
source=ast.parse((Path(__file__).resolve().parents[1]/'dexscrew/tasks/xhand_pasini.py').read_text())
cls=next(n for n in source.body if isinstance(n,ast.ClassDef) and n.name=='XHandPasini')
method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='pre_physics_step')
ns={'tensor_clamp':lambda x,lo,hi:torch.maximum(torch.minimum(x,hi),lo)}
exec(compile(ast.Module(body=[method],type_ignores=[]),'<actual pre_physics_step>','exec'),ns)
def fixture(limit):
 e=SimpleNamespace(device='cpu',num_xhand_hand_dofs=16,action_scale=.05,target_smoothing_alpha=0.,config={'env':{'controller':{'target_error_limit':limit}}})
 e.prev_targets=torch.zeros(1,17);e.cur_targets=torch.zeros(1,17);e.xhand_hand_dof_pos=torch.zeros(1,16)
 e.xhand_hand_dof_lower_limits=torch.full((16,),-2.);e.xhand_hand_dof_upper_limits=torch.full((16,),2.)
 for k in ['object_rot','object_pos','nut_dof_pos','ft_rot','ft_pos','dof_vel']:setattr(e,k+'_prev',torch.zeros(1))
 for k in ['object_rot','object_pos','nut_dof_pos','fingertip_orientation','fingertip_pos','dof_vel_finite_diff']:setattr(e,k,torch.zeros(1))
 return e
step=ns['pre_physics_step'];act=torch.ones(1,17);act[:,16]=0
for limit in [0.,.2]:
 e=fixture(limit)
 for _ in range(100):step(e,act)
 assert torch.allclose(e.prev_targets[:,:16],torch.full((1,16),limit or 2.))
 step(e,-act)
 assert torch.allclose(e.prev_targets[:,:16],torch.full((1,16),(limit or 2.)-.05))
 assert e.prev_targets[0,16]==0
# Both position-limit intersections and changing measured joint positions.
e=fixture(.2);e.xhand_hand_dof_pos[:]=1.99;e.prev_targets[:,:16]=2.
step(e,act);assert (e.prev_targets[:,:16]<=2.).all()
e.xhand_hand_dof_pos[:]=0.;step(e,act);assert (e.prev_targets[:,:16]<=.200001).all()
print('PASS: production anti-windup, immediate reverse command, disabled legacy, joint limits, nut target preserved')
