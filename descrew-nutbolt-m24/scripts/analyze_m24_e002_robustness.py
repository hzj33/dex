#!/usr/bin/env python3
"""Compare paired initial-state robustness evaluations using measured cycle events."""
import argparse,json,hashlib,math
from pathlib import Path
from analyze_m24_e002_repetition import analyze


def summarize(path,cycles):
    data=json.loads(path.read_text());result=analyze(path,cycles);offsets=data['initial_joint_offsets']
    return dict(source=str(path),checkpoint_sha256=data['checkpoint_sha256'],
        seed=data['summary']['evaluation_seed'],noise_degrees=math.degrees(data['summary']['initial_joint_noise_rad']),
        num_envs=data['summary']['num_envs'],target_cycles=cycles,
        completed=result['numeric_target_cycle_count'],all_returns_off=result['target_cycles_with_all_returns_off_count'],
        completed_env_ids=[e['env_id'] for e in result['episodes'] if e['numeric_target_cycles']],
        finite=result['finite'],canonical_reference_error=data['summary']['initial_hand_reference_max_error'],
        unique_initial_offsets=len(set(tuple(row) for row in offsets)),
        inactive_joint_offset_max=max(abs(row[j]) for row in offsets for j in range(4,12)),
        initial_offsets_sha256=hashlib.sha256(json.dumps(offsets).encode()).hexdigest(),
        termination_counts=data['summary']['termination_counts'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    state=json.loads((args.run/'workflow_status.json').read_text())
    assert state['status']=='completed:bounded_robustness_probe'
    baseline={str(seed):summarize(args.root/('baseline_1deg_s%d'%seed)/'evaluation.json',5) for seed in (43,44)}
    results={}
    for label in state['selected_checkpoints']:
        rows={name:summarize(args.run/('eval_'+label+'_'+name)/'evaluation.json',cycles) for name,cycles in [('nominal_two',2),('nominal_five',5),('perturbed_s43',5),('perturbed_s44',5)]}
        for seed in (43,44):
            after=rows['perturbed_s%d'%seed];before=baseline[str(seed)]
            assert before['initial_offsets_sha256']==after['initial_offsets_sha256'], 'Paired comparison starts differ'
            before_ids=set(before['completed_env_ids']);after_ids=set(after['completed_env_ids'])
            after['paired_outcomes']=dict(both_success=len(before_ids & after_ids),
                newly_successful=len(after_ids-before_ids),regressed=len(before_ids-after_ids),
                both_failed=after['num_envs']-len(before_ids | after_ids))
        results[label]=rows
    result=dict(status='analyzed:paired_robustness',training_seed=42,held_out_seeds=[43,44],baseline=baseline,checkpoints=results)
    args.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
