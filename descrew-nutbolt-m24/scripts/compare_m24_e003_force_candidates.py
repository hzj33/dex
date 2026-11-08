"""Paired first-episode comparison, with survivorship-separated force metrics."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def episodes(d):
 g=defaultdict(list)
 for r in d['first_episode_steps']:g[r['env_id']].append(r)
 return g

def compare(base,candidate):
 a=json.loads(Path(base).read_text());b=json.loads(Path(candidate).read_text());assert np.array_equal(a['initial_joint_offsets'],b['initial_joint_offsets']),'Initial offsets differ'
 aa,bb=episodes(a),episodes(b);assert aa.keys()==bb.keys()
 sa={i for i,r in aa.items() if r[-1]['last_reason']==5};sb={i for i,r in bb.items() if r[-1]['last_reason']==5};common=sa&sb
 def metrics(rows,ids):
  if not ids:return None
  f=[np.array([r['force'] for r in rows[i]]) for i in sorted(ids)]
  return dict(n=len(ids),peak_N=np.concatenate(f).max(0).tolist(),mean_force_squared_integral_N2s=float(np.mean([(x*x).sum()*.05 for x in f])),mean_excess5_duration_s=np.mean([(x>5).sum(0)*.05 for x in f],axis=0).tolist())
 return dict(baseline=str(base),candidate=str(candidate),n=len(aa),baseline_successes=len(sa),candidate_successes=len(sb),recovered=sorted(sb-sa),regressed=sorted(sa-sb),common_successes=len(common),common_success_baseline=metrics(aa,common),common_success_candidate=metrics(bb,common),all_baseline=metrics(aa,set(aa)),all_candidate=metrics(bb,set(bb)),note='20Hz screening; paired initial offsets exact; compare common successes separately from failed episodes')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('baseline');p.add_argument('candidate');p.add_argument('--output',required=True);a=p.parse_args();d=compare(a.baseline,a.candidate);Path(a.output).write_text(json.dumps(d,indent=2)+'\n');print(json.dumps(d,indent=2))
