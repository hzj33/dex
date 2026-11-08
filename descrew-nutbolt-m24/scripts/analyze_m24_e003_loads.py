#!/usr/bin/env python3
"""Comparable first-episode metrics; never equate normal loads to full wrench."""
import argparse,json
from pathlib import Path
from collections import defaultdict
import numpy as np

def summarize(path):
 d=json.loads(Path(path).read_text());rows=d['first_episode_steps'];groups=defaultdict(list)
 for r in rows:groups[r['env_id']].append(r)
 force=np.array([r['force'] for r in rows]);episodes=[]
 for rr in groups.values():
  f=np.array([r['force'] for r in rr]);theta=np.array([r['angle'] for r in rr]);contact=np.array([r['contact'] for r in rr]);drive=np.array([r['phase_before']==0 for r in rr]);loss=drive&contact[:,0]&~contact[:,1]
  speed=np.r_[0,np.diff(theta)]/.05
  longest=run=0
  for lost in loss:run=run+1 if lost else 0;longest=max(longest,run)
  release=np.array([r['phase_before']==1 for r in rr]);inc=[]
  for start in np.flatnonzero(release&~np.r_[False,release[:-1]]):
   ends=np.flatnonzero(~release[start:]);end=start+ends[0] if len(ends) else len(rr)
   entry=f[max(0,start-1)];inc.append(float(np.maximum(0,f[start:end].max(0)-entry).max()))
  episodes.append(dict(release_force_increase_N=max(inc,default=0.),env_id=rr[0]['env_id'],success=rr[-1]['last_reason']==5,cycles=rr[-1]['cycles'],
   index_peak_N=float(f[:,0].max()),thumb_peak_N=float(f[:,1].max()),
   thumb_loss_s=float(loss.sum()*.05),longest_thumb_loss_s=longest*.05,
   unilateral_static_load_Ns=float((f[:,0]*loss*(abs(speed)<1e-4)).sum()*.05),
   force_squared_integral_N2s=float((f*f).sum()*.05),force_integral_Ns=float(f.sum()*.05),
   duration_s=len(rr)*.05,net_degrees=float(np.degrees(theta[-1]-rr[0]['initial_angle']))))
 success=[r for r in episodes if r['success']]
 return dict(path=str(path),checkpoint_sha256=d['checkpoint_sha256'],n=len(episodes),successes=len(success),
  normal_force_peak_N=force.max(0).tolist(),normal_force_p95_N=np.percentile(force,95,axis=0).tolist(),
  normal_force_p99_N=np.percentile(force,99,axis=0).tolist(),
  loaded_normal_force_p95_N=[float(np.percentile(force[force[:,j]>.15,j],95)) if (force[:,j]>.15).any() else 0. for j in range(2)],
  successful_mean_force_squared_per_rad=float(np.mean([r['force_squared_integral_N2s']/np.radians(r['net_degrees']) for r in success])) if success else None,
  mean_release_force_increase_N=float(np.mean([r['release_force_increase_N'] for r in episodes])),
  mean_thumb_loss_s=float(np.mean([r['thumb_loss_s'] for r in episodes])),
  mean_force_squared_integral_N2s=float(np.mean([r['force_squared_integral_N2s'] for r in episodes])),
  mean_unilateral_static_load_Ns=float(np.mean([r['unilateral_static_load_Ns'] for r in episodes])),
  successful_mean_thumb_loss_s=float(np.mean([r['thumb_loss_s'] for r in success])) if success else None,
  successful_mean_force_squared_integral_N2s=float(np.mean([r['force_squared_integral_N2s'] for r in success])) if success else None,episodes=episodes)

def main():
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result={'note':'20Hz nut normal resultant; force integrals are endpoint sampled; static uses |angle difference/dt|<1e-4rad/s','results':[summarize(p) for p in a.files]}
 a.output.write_text(json.dumps(result,indent=2)+'\n')
 for r in result['results']:print(Path(r['path']).parent.name, r['successes'], '/',r['n'],'peak',r['normal_force_peak_N'],'loss_s',r['mean_thumb_loss_s'],'F2dt',r['mean_force_squared_integral_N2s'])
if __name__=='__main__':main()
