#!/usr/bin/env python3
"""Audit recorded manual phase hints without opening the SDK or operating hardware.

Counterfactuals change one actor input at each recorded state. They are not
closed-loop rollouts and cannot establish physical release or success.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from deploy_m24_e003_student import Policy, PolicyLoop
from m24_e003_release_override import manual_loop_class, phase_action
import numpy as np
import torch


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    meta = json.loads((args.log_dir / 'run_metadata.json').read_text())
    rows = read_rows(args.log_dir / 'policy_steps.jsonl')
    events = read_rows(args.log_dir / 'manual_release.jsonl')
    digest = hashlib.sha256(args.artifact.read_bytes()).hexdigest()
    assert digest == meta['jit_policy']['sha256']
    assert len(rows) == len(events) > 0
    assert [r['step'] for r in rows] == [e['step'] for e in events]
    current = [False]
    policy = Policy(args.artifact)
    loop = manual_loop_class(PolicyLoop)(policy, lambda: current[0],
                                         meta['manual_release']['pulse_steps'])
    errors = {k: 0.0 for k in ('q', 'previous_target', 'action', 'next_target')}
    comparisons = []
    for row, event in zip(rows, events):
        current[0] = event['requested']
        history = loop.history if loop.history is not None else loop.model.reset_history(loop.previous, loop.previous)
        obs96 = history[:, -3:].reshape(-1, 96).clone()
        result = loop.step(row['q'], row['policy_hz'])
        for key in errors:
            errors[key] = max(errors[key], float(np.max(np.abs(result[key] - np.array(row[key])))))
        assert result['manual_release']['active'] == event['active']
        np.testing.assert_allclose(result['manual_release']['actor_phase_command'],
                                   event['actor_phase_command'], atol=1e-6, rtol=0)
        if not event['active']:
            continue
        _, logits, clock, _ = loop.model.predict(obs96, loop.history)
        learned = torch.cat((logits.softmax(-1), clock), -1)
        release = learned.clone()
        release[:, :4] = 0
        release[:, 1] = 1
        commands = {'learned': learned, 'release_predicted_clock': release}
        for value in (0.0, 0.5, 1.0):
            command = release.clone()
            command[:, 4] = value
            commands['release_clock_%g' % value] = command
        previous = torch.tensor(row['previous_target']).reshape(1, 16)
        targets = {}
        for name, command in commands.items():
            action = phase_action(loop.model, obs96, loop.history, command)
            target = (previous + loop.model.action_scale * action).maximum(loop.model.lower).minimum(loop.model.upper)
            targets[name] = target[0].tolist()
        comparisons.append(dict(step=row['step'], targets=targets))
    assert max(errors.values()) < 1e-5, errors
    q = np.array([r['q'] for r in rows])
    target = np.array([r['next_target'] for r in rows])
    names = meta['joint_order']
    pulses = []
    for start, event in enumerate(events):
        if not event['accepted']:
            continue
        stop = start
        while stop < len(events) and events[stop]['active']:
            stop += 1
        before = np.array(rows[start]['previous_target'])
        observed_end = min(stop, len(rows)-1)
        pulses.append(dict(
            start_step=rows[start]['step'], last_active_step=rows[stop-1]['step'],
            active_steps=stop-start, measured_endpoint_step=rows[observed_end]['step'],
            complete_measured_endpoint=stop < len(rows),
            learned_phase_at_trigger=int(np.argmax(event['predicted_phase_probabilities'])),
            learned_clock_at_trigger=event['actor_phase_command'][-1],
            clock_range=[min(e['actor_phase_command'][-1] for e in events[start:stop]),
                         max(e['actor_phase_command'][-1] for e in events[start:stop])],
            target_delta_deg=dict(zip(names, np.degrees(target[stop-1]-before).tolist())),
            measured_delta_deg=dict(zip(names, np.degrees(q[observed_end]-q[start]).tolist()))))
    active_indices = [0, 1, 2, 12, 13, 14, 15]  # index_3 is mechanically passive.
    sensitivity = {}
    if comparisons:
        reference = np.array([c['targets']['release_predicted_clock'] for c in comparisons])
        for name in comparisons[0]['targets']:
            delta = np.degrees(np.array([c['targets'][name] for c in comparisons])-reference)[:, active_indices]
            sensitivity[name] = dict(mean_abs_target_difference_deg=float(np.abs(delta).mean()),
                                     max_abs_target_difference_deg=float(np.abs(delta).max()))
    times = [datetime.fromisoformat(r['timestamp']).timestamp() for r in rows]
    report = dict(log_dir=str(args.log_dir), artifact_sha256=digest, steps=len(rows),
        torch_num_threads=torch.get_num_threads(),
        source_sha256={name: hashlib.sha256((args.log_dir / name).read_bytes()).hexdigest()
                       for name in ('run_metadata.json', 'policy_steps.jsonl', 'manual_release.jsonl')},
        dry_run=meta['dry_run'], replay_max_errors=errors,
        median_step_seconds=float(np.median(np.diff(times))),
        predicted_phase_counts=np.bincount([np.argmax(e['predicted_phase_probabilities']) for e in events], minlength=4).tolist(),
        pulses=pulses, same_state_counterfactual_sensitivity=sensitivity,
        counterfactual_scope='One-step actor sensitivity on recorded states; not a physical rollout.',
        physical_release_verified=False, hardware_operated_by_analyzer=False,
        same_state_counterfactuals=comparisons)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'same_state_counterfactuals'}, indent=2))


if __name__ == '__main__':
    main()
