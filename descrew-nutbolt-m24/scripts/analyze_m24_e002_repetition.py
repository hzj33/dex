#!/usr/bin/env python3
"""Behavioral evidence from uninterrupted original-start first episodes."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def analyze(path, target_cycles=2):
    assert target_cycles >= 2
    d=json.loads(path.read_text());groups={}
    for r in d['first_episode_steps']:groups.setdefault(r['env_id'],[]).append(r)
    single=d['summary']['action_mode']=='policy' and not d.get('policy_composition')
    episodes=[]
    for env,rows in groups.items():
        if rows[0]['seed_step']!=-1:continue
        events=[r for r in rows if r['last_event']]
        expected=[1,2,3,4]*target_cycles
        prefix=0
        for r in events:
            if prefix>=len(expected) or r['last_event']!=expected[prefix]:break
            prefix+=1
        strokes=[];start_angle=rows[0]['initial_angle'];start_contact=0.
        returns=[]
        for r in events:
            if r['last_event']==1:
                strokes.append(dict(step=r['step'],net_stroke_degrees=math.degrees(r['angle']-start_angle),
                    contact_supported_new_degrees=math.degrees(r['contact_progress']-start_contact),
                    contact=r['contact'],force=r['force']))
            if r['last_event']==3:
                prior_release=max([x['step'] for x in events if x['last_event']==2 and x['step']<r['step']],default=-1)
                motion=[x for x in rows if prior_release<x['step']<=r['step'] and x['phase_before']==2]
                returns.append(dict(step=r['step'],net_degrees=math.degrees(r['angle']),
                    sector_error_degrees=math.degrees(r['return_error']),
                    all_steps_off=bool(motion) and all(not any(x['contact']) and max(x['force'])<.15 for x in motion)))
            if r['last_event']==4:
                start_angle=r['angle'];start_contact=r['contact_progress']
        next_drive=(single and prefix>=5 and len(strokes)>=2 and
            all(s['net_stroke_degrees']>=20-1e-5 and s['contact_supported_new_degrees']>=20-1e-5 and all(s['contact']) for s in strokes[:2]))
        episodes.append(dict(env_id=env,ordered_event_prefix=prefix,second_effective_drive=next_drive,
            numeric_two_cycles=bool(next_drive and prefix>=8),
            numeric_target_cycles=bool(single and prefix>=4*target_cycles and rows[-1]['last_reason']==5 and
                all(s['net_stroke_degrees']>=20-1e-5 and s['contact_supported_new_degrees']>=20-1e-5 and all(s['contact']) for s in strokes[:target_cycles])),
            strokes=strokes,returns=returns,recontacts=[dict(step=r['step'],sector_error_degrees=math.degrees(r['return_error']),
                net_degrees=math.degrees(r['angle']),force=r['force']) for r in events if r['last_event']==4],
            termination_reason=rows[-1]['last_reason'],final_net_degrees=math.degrees(rows[-1]['angle'])))
    return dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        checkpoint=d['checkpoint'],checkpoint_sha256=d['checkpoint_sha256'],finite=d['summary']['finite'],
        single_policy=single,original_episodes=len(episodes),target_cycles=target_cycles,
        numeric_target_cycle_count=sum(e['numeric_target_cycles'] for e in episodes),
        target_cycles_with_all_returns_off_count=sum(e['numeric_target_cycles'] and len(e['returns'])>=target_cycles and all(r['all_steps_off'] for r in e['returns'][:target_cycles]) for e in episodes),
        second_effective_drive_count=sum(e['second_effective_drive'] for e in episodes),
        numeric_two_cycle_count=sum(e['numeric_two_cycles'] for e in episodes),
        two_cycles_with_both_returns_off_count=sum(e['numeric_two_cycles'] and len(e['returns'])>=2 and all(r['all_steps_off'] for r in e['returns'][:2]) for e in episodes),
        full_goal_visual_verified=False,episodes=episodes)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--target-cycles',type=int,default=2)
    a=parser.parse_args();result=analyze(a.evaluation,a.target_cycles);a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))
