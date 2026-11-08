"""Per-finger exceedance and transverse normal-wrench diagnostics.

Normal resultants are not full contact wrenches. Endpoint samples do not
bound substep/solver maxima. No averaging can turn an excess into a pass.
"""
import argparse,json
from pathlib import Path
import numpy as np

def analyze(path):
 d=json.loads(Path(path).read_text());rows=d['first_episode_steps'];f=np.array([r['force'] for r in rows]);final={}
 for r in rows:final[r['env_id']]=r
 phase=np.array([r['phase_before'] for r in rows]);out=dict(path=str(path),n=len(final),successes=sum(r['last_reason']==5 for r in final.values()),sample_dt=.05,force_limit_N=5,peak_N=f.max(0).tolist(),p95_N=np.percentile(f,95,axis=0).tolist(),excess_sample_fraction=(f>5).mean(0).tolist(),mean_excess_duration_s=((f>5).sum(0)*.05/len(final)).tolist(),phase_peak_N={str(k):f[phase==k].max(0).tolist() for k in range(4) if (phase==k).any()},force5_sampled_pass=bool((f<=5).all()),note='20Hz screening, not physics-step acceptance')
 p=Path(path).with_suffix('.physics.npz')
 if p.exists():
  a=np.load(p);meta=json.loads(p.with_suffix('.metadata.json').read_text());valid=a['alive'].astype(bool);force=a['nut_normal'];mag=np.linalg.norm(force,axis=-1)[valid];out['physics']=dict(dt=meta['physics_dt'],peak_N=mag.max(0).tolist(),excess_sample_fraction=(mag>5).mean(0).tolist(),force5_sampled_pass=bool((mag<=5).all()),normal_net_force_peak_N=float(np.linalg.norm(force.sum(2),axis=-1)[valid].max()),normal_net_moment_peak_Nm=float(np.linalg.norm(a['nut_normal_moment'].sum(2),axis=-1)[valid].max()))
  if 'whole_finger_normal' in a:
   whole=np.linalg.norm(a['whole_finger_normal'],axis=-1)[valid]
   out['physics']['whole_finger_peak_N']=whole.max(0).tolist()
   out['physics']['whole_finger_force5_sampled_pass']=bool((whole<=5).all())
  if 'rotation_axis_world' in meta:
   axis=np.asarray(meta['rotation_axis_world']);assert np.isclose(np.linalg.norm(axis),1)
   netf=force.sum(2);netm=a['nut_normal_moment'].sum(2)
   out['physics']['normal_transverse_force_peak_N']=float(np.linalg.norm(netf-(netf*axis).sum(-1,keepdims=True)*axis,axis=-1)[valid].max())
   out['physics']['normal_tilt_moment_peak_Nm']=float(np.linalg.norm(netm-(netm*axis).sum(-1,keepdims=True)*axis,axis=-1)[valid].max())
  if 'solver_wrench' in a:
   out['physics']['solver_wrench_abs_peak']=np.abs(a['solver_wrench'][valid]).max(0).tolist()
   out['physics']['solver_wrench_calibration_passed']=False
   out['physics']['solver_wrench_warning']='Known5N contact gives iteration-dependent25/45N raw root readings. Not a physical contact/support acceptance metric.'
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();out=[analyze(f) for f in a.files];Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
