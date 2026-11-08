import json,numpy as np
from pathlib import Path
root=Path('outputs/cloud_pipeline_m24_e003_peak_audit_20260914')
a=json.load(open(root/'control/evaluation.json'));b=json.load(open(root/'peak/evaluation.json'))
ra,rb=a['first_episode_steps'],b['first_episode_steps'];assert len(ra)==len(rb)
for x,y in zip(ra,rb):
 for k in x:assert x[k]==y[k],(k,x[k],y[k])
p=np.load(root/'peak/evaluation.physics.npz');m=json.load(open(root/'peak/evaluation.physics.metadata.json'))
print('shapes',p['policy_step'].shape,p['net_contact'].shape,'policy values',p['policy_step'][:12].flatten().tolist(),'row steps',[r['step'] for r in rb[:3]])
groups=m['whole_finger_groups'];net=p['net_contact'];f=np.stack([np.linalg.norm(net[:,:,g,:],axis=-1).sum(-1) for g in groups],axis=-1)
expected=f.reshape(len(rb),10,1,2).max(1)[:,0,:];actual=np.array([r['force_cost_peak'] for r in rb]);err=float(abs(expected-actual).max());assert err<1e-5,err
result=dict(status='passed:physics_peak_accumulator',rows=len(rb),all_control_trace_fields_exact=True,max_force_error_N=err,peak_N=actual.max(0).tolist(),endpoint_peak_N=np.array([r['force'] for r in rb]).max(0).tolist(),both_completed_five_cycles=a['summary']['completed_two_cycles']==b['summary']['completed_two_cycles']==1,signal_note='whole-finger sum of per-body net resultant magnitudes; non-nut contacts included; not full friction wrench')
(root/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
