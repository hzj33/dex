#!/usr/bin/env python3
"""Summarize actual return trajectories; episode reward is not acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze(path):
    d=json.loads(path.read_text());groups={}
    for r in d['first_episode_steps']:groups.setdefault(r['env_id'],[]).append(r)
    returned=[]
    for env,rows in groups.items():
        events=[r for r in rows if r['last_event']==3]
        if not events:continue
        r=events[0]
        motion=[x for x in rows if x['step']<=r['step'] and x['phase_before']==2]
        actual_release=any(x['last_event']==2 and x['step']<r['step'] for x in rows)
        actual_drive=any(x['last_event']==1 and x['step']<r['step'] for x in rows)
        off=lambda x:not any(x['contact']) and max(x['force'])<.15
        returned.append(dict(env_id=env,seed_step=r['seed_step'],return_step=r['step'],
            net_degrees=math.degrees(r['angle']),return_error_degrees=math.degrees(r['return_error']),
            endpoint_force=r['force'],endpoint_contact=r['contact'],
            retained_progress=r['angle']>=math.radians(10),
            endpoint_off=off(r),all_return_steps_off=all(off(x) for x in motion),
            recontact_steps=sum(not off(x) for x in motion),
            actual_drive_and_release=actual_drive and actual_release))
    originals=[r for r in returned if r['seed_step']==-1 and r['actual_drive_and_release']]
    single_policy=d['summary']['action_mode']=='policy' and not d.get('policy_composition')
    chain=[r for r in originals if r['retained_progress'] and r['endpoint_off']] if single_policy else []
    return dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],
        finite=d['summary']['finite'],initialization=d['summary']['initialization'],
        action_mode=d['summary']['action_mode'],num_envs=d['summary']['num_envs'],
        original_start_count=sum(rows[0]['seed_step']==-1 for rows in groups.values()),
        returned_count=len(returned),original_learned_chain_count=len(chain),
        original_chain_without_return_recontact=sum(r['all_return_steps_off'] for r in chain),
        full_cycle_goal_completed=False,returned_episodes=returned)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=analyze(a.evaluation);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='returned_episodes'},indent=2))
