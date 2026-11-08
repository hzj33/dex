"""Summarize completed, single-PPO E003 evaluations without changing gates."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();results=[]
    for root in sorted(Path('outputs').glob('cloud_pipeline_m24_e003_*_20260914')):
        status=root/'status.json'
        if not status.exists():continue
        state=json.loads(status.read_text())
        if 'candidate_checkpoints' not in state:continue
        for label,checkpoint in state['candidate_checkpoints'].items():
            row=dict(pipeline=str(root),candidate=label,checkpoint=checkpoint,groups={})
            for suffix in ['nominal','s43','s44','s45','s46']:
                path=root/('eval_'+label+'_'+suffix)/'evaluation.json'
                if not path.exists():continue
                d=json.loads(path.read_text())
                if d.get('policy_composition') or d.get('control_diagnostic') or d.get('policy_algorithm') != 'PPO':continue
                if d['summary'].get('required_cycles',5) != 5 or not d['summary'].get('paired_drive_progress_required',True):continue
                trace=d['first_episode_steps'];f=np.asarray([r['force'] for r in trace]);last={r['env_id']:r for r in trace}
                row['groups'][suffix]=dict(n=len(last),five_cycle_success=sum(r['last_reason']==5 for r in last.values()),peak_N=f.max(0).tolist(),over5_duration_mean_s=((f>5).sum(0)*.05/len(last)).tolist(),force5_endpoint_pass=bool((f<=5).all()),termination_counts=d['summary']['termination_counts'])
            if not row['groups']:continue
            row['all_available_endpoint_force5_pass']=all(g['force5_endpoint_pass'] for g in row['groups'].values())
            row['physics_force5_acceptance']='not established by this endpoint summary'
            results.append(row)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(note='20Hz rejection screen only. Short failed trajectories cannot establish lower load. No acceptance or baseline promotion.',candidates=results),indent=2)+'\n')
    for r in results:
        nominal=r['groups'].get('nominal',{}).get('five_cycle_success','-');pert=[g for k,g in r['groups'].items() if k!='nominal'];peak=np.max([g['peak_N'] for g in r['groups'].values()],axis=0)
        print(Path(r['pipeline']).name,r['candidate'],'nominal',nominal,'perturb',sum(g['five_cycle_success'] for g in pert),'/',sum(g['n'] for g in pert),'peak',np.round(peak,3).tolist())

if __name__=='__main__':main()
