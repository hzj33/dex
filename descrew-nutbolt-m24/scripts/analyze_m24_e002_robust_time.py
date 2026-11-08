#!/usr/bin/env python3
"""Acceptance of unchanged-policy timing correction on full physical cycles."""
import argparse,collections,hashlib,json,math
from pathlib import Path
from analyze_m24_e002_repetition import analyze

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--diagnostic-root',type=Path,required=True);a=parser.parse_args();p=a.root
 assert json.loads((p/'workflow_status.json').read_text())['status']=='completed:frozen_policy_validation'
 result=dict(status='passed:measured_timeout_correction',training=False,reward_weights_changed=False,physical_success_conditions_changed=False,task='XHandPasiniM24NutBoltE002RobustTime',checkpoint_sha256=None,termination_deadlines=[160,80,120,80],policy_clock_deadlines=[80,40,60,40],phase_clock_scale=.5,results={},full_robustness_goal_completed=False)
 failures=collections.Counter()
 for seed in (43,44,45):
  before=(a.diagnostic_root/('timeout1_seed%d'%seed) if seed!=45 else p/'seed45_original')/'evaluation.json';after=p/('seed%d_time'%seed)/'evaluation.json'
  b=json.loads(before.read_text());d=json.loads(after.read_text());assert b['initial_joint_offsets']==d['initial_joint_offsets'];assert b['checkpoint_sha256']==d['checkpoint_sha256']
  result['checkpoint_sha256']=d['checkpoint_sha256'];result['source_checkpoint']=d['checkpoint']
  metrics=[analyze(path,5) for path in (before,after)];ids=[{e['env_id'] for e in m['episodes'] if e['numeric_target_cycles']} for m in metrics]
  assert metrics[1]['finite'] and metrics[1]['numeric_target_cycle_count']==metrics[1]['target_cycles_with_all_returns_off_count']
  result['results'][str(seed)]=dict(before=len(ids[0]),after=len(ids[1]),total=d['summary']['num_envs'],retained=len(ids[0]&ids[1]),recovered=len(ids[1]-ids[0]),regressed=len(ids[0]-ids[1]),before_source=str(before),after_source=str(after),initial_offsets_sha256=hashlib.sha256(json.dumps(d['initial_joint_offsets']).encode()).hexdigest())
  failures.update(e['termination_reason'] for e in metrics[1]['episodes'] if not e['numeric_target_cycles'])
 nominal=analyze(p/'nominal_time/evaluation.json',5);assert nominal['numeric_target_cycle_count']==nominal['target_cycles_with_all_returns_off_count']==128;result['nominal_five']=128;result['remaining_failures_by_reason']=dict(failures)
 v=json.loads((p/'native_seed43/trace.json').read_text());rows=v['rows'];events=[r['event'] for r in rows if r.get('event')];assert events==[1,2,3,4]*5 and rows[-1]['reason']==5
 assert v['diagnostic_action_override'] is None and v['policy_composition'] is None
 intervals=[];released=None
 for r in rows:
  if r.get('event')==2:released=r['step']
  if r.get('event')==3:intervals.append(all(not any(x['contact']) and max(x['force'])<.15 for x in rows if released<x['step']<=r['step']))
 assert len(intervals)==5 and all(intervals)
 previous=Path('outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_perturbed_video_s43/trace.json');old=json.loads(previous.read_text())['rows'];assert len(rows)>len(old)
 result['video']=dict(trace=str(p/'native_seed43/trace.json'),frames=len(rows),duration_seconds=len(rows)/20,net_degrees=math.degrees(rows[-1]['angle']),completed_cycles=5,all_returns_off=True,previous_failed_frames=len(old),additional_steps=len(rows)-len(old),prefix_action_max_error=max(abs(x-y) for r,s in zip(rows,old) for x,y in zip(r['action'],s['action'])),prefix_angle_max_error=max(abs(r['angle']-s['angle']) for r,s in zip(rows,old)))
 result['limitations']=['Only +/-1degree initial-joint robustness evaluated','Seeds43/44 informed budget selection; seed45 held out from that choice','Some return and workspace failures remain','No new PPO or verification beyondfive cycles']
 (p/'acceptance.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
