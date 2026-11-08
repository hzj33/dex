#!/usr/bin/env python3
"""Independent paired acceptance for the frozen-policy compliant profile."""
import json,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from analyze_m24_e003_loads import summarize

def main():
 old=Path('outputs/cloud_pipeline_m24_e002_robust_time_20260911');batch=Path('outputs/cloud_pipeline_m24_e003_compliant_20260911');extra=Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911')
 baseline={'nominal':old/'nominal_time/evaluation.json',**{str(s):old/f'seed{s}_time/evaluation.json' for s in (43,44,45)},'46':Path('outputs/cloud_pipeline_m24_e003_candidate_20260911/baseline_s46/evaluation.json')}
 candidate={'nominal':batch/'nominal/evaluation.json','43':batch/'seed43/evaluation.json','44':batch/'seed44/evaluation.json','45':extra/'candidate_s45/evaluation.json','46':extra/'candidate_s46/evaluation.json'}
 results={};all_before=[];all_after=[]
 for key in baseline:
  b=json.loads(baseline[key].read_text());a=json.loads(candidate[key].read_text());assert a['checkpoint_sha256']==b['checkpoint_sha256']=='d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'
  assert a['initial_joint_offsets']==b['initial_joint_offsets'],key
  assert a['summary']['finite'] and b['summary']['finite'] and len(a['first_episodes'])==len(b['first_episodes'])==128
  good={r['env_id'] for r in a['first_episodes'] if r['last_reason']==5 and r['cycles']==5};oldgood={r['env_id'] for r in b['first_episodes'] if r['last_reason']==5 and r['cycles']==5}
  grouped=defaultdict(list)
  for r in a['first_episode_steps']:grouped[r['env_id']].append(r)
  max_return_force=0.;minimum_stroke=math.inf
  for i in good:
   rr=grouped[i];assert [r['last_event'] for r in rr if r['last_event']]==[1,2,3,4]*5
   start=rr[0]['initial_angle'];contact_start=0.
   for r in rr:
    if r['phase_before']==2:
     max_return_force=max(max_return_force,*r['force']);assert not any(r['contact']) and max(r['force'])<.15,(key,i,'return contact')
    if r['last_event']==1:
     stroke=min(r['angle']-start,r['contact_progress']-contact_start);minimum_stroke=min(minimum_stroke,stroke);assert stroke>=math.radians(20)-1e-6
    if r['last_event']==4:start=r['angle'];contact_start=r['contact_progress']
  bm=summarize(baseline[key]);am=summarize(candidate[key]);results[key]=dict(before={k:v for k,v in bm.items() if k!='episodes'},after={k:v for k,v in am.items() if k!='episodes'},retained=len(good&oldgood),recovered=len(good-oldgood),regressed=len(oldgood-good),initial_offsets_sha256=hashlib.sha256(json.dumps(a['initial_joint_offsets'],separators=(',',':')).encode()).hexdigest(),successful_return_max_normal_N=max_return_force,minimum_valid_drive_degrees=math.degrees(minimum_stroke) if good else None)
  if key!='nominal':
   all_before+=b['first_episode_steps'];all_after+=a['first_episode_steps']
 def pooled(rows):
  f=np.array([r['force'] for r in rows]);return dict(normal_force_peak_N=f.max(0).tolist(),normal_force_p95_N=np.percentile(f,95,axis=0).tolist(),loaded_normal_force_p95_N=[float(np.percentile(f[f[:,j]>.15,j],95)) for j in (0,1)])
 checks=dict(nominal_preserved=results['nominal']['after']['successes']==128,perturbed_success_not_lower=sum(results[str(s)]['after']['successes'] for s in (43,44,45,46))>=sum(results[str(s)]['before']['successes'] for s in (43,44,45,46)),both_pooled_peaks_lower=bool(np.all(np.array(pooled(all_after)['normal_force_peak_N'])<pooled(all_before)['normal_force_peak_N'])),mean_thumb_loss_lower=sum(results[str(s)]['after']['mean_thumb_loss_s'] for s in (43,44,45,46))<sum(results[str(s)]['before']['mean_thumb_loss_s'] for s in (43,44,45,46)))
 report=dict(status='passed:numeric_compliant_comparison' if all(checks.values()) else 'not_promoted:numeric_tradeoff',task='XHandPasiniM24NutBoltE003Compliant',checkpoint_sha256=next(iter(results.values()))['after']['checkpoint_sha256'],checks=checks,results=results,pooled_before=pooled(all_before),pooled_after=pooled(all_after),physics_change={'controller.pgain':[3,2.4]},visual_verification_pending=True,hardware_safe=False)
 (extra/'numeric_acceptance.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k!='results'},indent=2))
if __name__=='__main__':main()
