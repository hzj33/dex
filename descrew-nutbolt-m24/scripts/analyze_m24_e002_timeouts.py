#!/usr/bin/env python3
"""Analyze paired timeout-only inference, preserving original success metrics."""
import argparse,collections,hashlib,json,math,statistics
from pathlib import Path
from analyze_m24_e002_repetition import analyze

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path);args=parser.parse_args();p=args.root
 state=json.loads((p/'workflow_status.json').read_text());assert state['status']=='completed:diagnostic_only'
 result={'training':False,'seeds':{}}
 for seed in (43,44):
  data=[json.loads((p/('timeout%d_seed%d'%(m,seed))/'evaluation.json').read_text()) for m in (1,2)]
  assert data[0]['initial_joint_offsets']==data[1]['initial_joint_offsets']
  assert data[0]['checkpoint_sha256']==data[1]['checkpoint_sha256']
  counts=[analyze(p/('timeout%d_seed%d'%(m,seed))/'evaluation.json',5) for m in (1,2)]
  groups=[]
  for d in data:
   g=collections.defaultdict(list)
   for r in d['first_episode_steps']:g[r['env_id']].append(r)
   groups.append(g)
  details=[]
  for i,rr in groups[0].items():
   if rr[-1]['last_reason'] not in (10,11,12,13):continue
   old=rr[-1];after=groups[1][i];last=after[-1];phase=old['phase_before'];cycle=old['cycles']
   segment=[r for r in rr if r['cycles']==cycle and r['phase_before']==phase];tail=segment[-10:]
   matched=[r for r in after if r['step']==old['step']][0]
   item=dict(env_id=i,original_reason=old['last_reason'],original_cycle=cycle,extended_reason=last['last_reason'],extended_cycles=last['cycles'],extended_full_success=last['last_reason']==5,
     original_tail_net_deg=math.degrees(tail[-1]['angle']-tail[0]['angle']),original_tail_pair_steps=sum(all(r['contact']) for r in tail),
     original_tail_joint_motion_rad=max(abs(tail[-1]['joint_state'][j][0]-tail[0]['joint_state'][j][0]) for j in [0,1,2,3,12,13,14,15]),
     original_tail_target_motion_rad=max(abs(tail[-1]['joint_targets'][j]-tail[0]['joint_targets'][j]) for j in [0,1,2,3,12,13,14,15]),
     same_step_angle_difference=abs(matched['angle']-old['angle']),same_step_action_max_difference=max(abs(a-b) for a,b in zip(matched['action'],old['action'])))
   details.append(item)
  result['seeds'][str(seed)]=dict(original_completed=counts[0]['numeric_target_cycle_count'],extended_completed=counts[1]['numeric_target_cycle_count'],extended_all_returns_off=counts[1]['target_cycles_with_all_returns_off_count'],failures=details)
 (p/'timeout_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({s:{'before':d['original_completed'],'after':d['extended_completed'],'timeout_failure_recoveries':sum(r['extended_full_success'] for r in d['failures']),'drive_timeout_recoveries':sum(r['extended_full_success'] for r in d['failures'] if r['original_reason']==10),'paired_prefix_action_max_error':max(r['same_step_action_max_difference'] for r in d['failures'])} for s,d in result['seeds'].items()},indent=2))
if __name__=='__main__':main()
