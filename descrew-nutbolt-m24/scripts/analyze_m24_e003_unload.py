"""First-episode unloading diagnostics, endpoint sampled at 20Hz."""
import argparse,json
from collections import defaultdict
from pathlib import Path

def analyze(path):
 d=json.loads(Path(path).read_text());groups=defaultdict(list);events=[]
 for r in d['first_episode_steps']:groups[r['env_id']].append(r)
 for eid,rows in groups.items():
  i=0
  while i<len(rows):
   if rows[i]['phase_before']!=1:i+=1;continue
   start=i
   while i<len(rows) and rows[i]['phase_before']==1:i+=1
   rr=rows[start:i];off=[]
   for j in range(2):off.append(next((k for k,r in enumerate(rr) if not r['contact'][j] and r['force'][j]<.15),None))
   stable_off=[]
   for j in range(2):
    loaded=[k for k,r in enumerate(rr) if r['contact'][j] or r['force'][j]>=.15]
    k=max(loaded,default=-1)+1
    stable_off.append(k if k<len(rr) else None)
   stable_lag=None if None in stable_off else (stable_off[0]-stable_off[1])*.05
   recontacts=sum(sum((not rr[k-1]['contact'][j]) and rr[k]['contact'][j] for k in range(1,len(rr))) for j in range(2))
   single=[r for r in rr if sum(r['contact'])==1]
   events.append(dict(stable_off_lag_s=stable_lag,recontact_events=recontacts,bilateral_above5_s=sum(min(r['force'])>5 for r in rr)*.05,env_id=eid,completed=rr[-1]['last_event']==2,off_lag_s=None if None in off else (off[0]-off[1])*.05,unilateral_duration_s=len(single)*.05,unilateral_impulse_Ns=sum(sum(r['force']) for r in single)*.05,unilateral_peak_N=max((max(r['force']) for r in single),default=0)))
 stable=[e['stable_off_lag_s'] for e in events if e['stable_off_lag_s'] is not None]
 valid=[e['off_lag_s'] for e in events if e['off_lag_s'] is not None]
 return dict(stable_pair_count=len(stable),mean_abs_stable_off_lag_s=sum(abs(x) for x in stable)/max(1,len(stable)),recontact_events=sum(e['recontact_events'] for e in events),mean_bilateral_above5_s=sum(e['bilateral_above5_s'] for e in events)/max(1,len(events)),path=str(path),release_events=len(events),paired_off_events=len(valid),thumb_first=sum(x>0 for x in valid),index_first=sum(x<0 for x in valid),same_sample=sum(x==0 for x in valid),mean_abs_lag_s=sum(abs(x) for x in valid)/max(1,len(valid)),mean_unilateral_impulse_Ns=sum(e['unilateral_impulse_Ns'] for e in events)/max(1,len(events)),unilateral_peak_N=max((e['unilateral_peak_N'] for e in events),default=0),events=events)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('files',nargs='+');p.add_argument('--output',required=True);a=p.parse_args();out=dict(note='Normal resultant only; signed off lag positive means thumb first; 0.05s resolution; not full wrench balance.',results=[analyze(f) for f in a.files]);Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps([{k:v for k,v in r.items() if k!='events'} for r in out['results']],indent=2))
