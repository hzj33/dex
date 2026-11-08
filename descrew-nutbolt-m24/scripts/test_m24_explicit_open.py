#!/usr/bin/env python3
"""Actual-model, measured-start and fake-device verification; no hardware calls."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from deploy_m24_e003_student import Policy, PolicyLoop
from m24_explicit_open import ExplicitOpenController, explicit_open_loop_class, ACTIVE
from m24_hand_kinematics import HandKinematics
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--log-dir', type=Path, required=True)
    p.add_argument('--legacy-script', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--evaluation', type=Path)
    a = p.parse_args()
    policy = Policy(a.artifact)
    lower, upper = policy.model.lower.numpy(), policy.model.upper.numpy()
    rows = [json.loads(x) for x in (a.log_dir/'policy_steps.jsonl').read_text().splitlines()]
    base = PolicyLoop(policy)
    idle = explicit_open_loop_class(PolicyLoop)(policy, lambda: False)
    for row in rows:
        x, y = base.step(row['q'], 20), idle.step(row['q'], 20)
        for key in ('q', 'previous_target', 'action', 'next_target'):
            np.testing.assert_array_equal(x[key], y[key])
    fk = HandKinematics('assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf')
    gain = np.array([.228, .22, .939, 1]+[1]*8+[.215, .222, .52, .812])
    vmax = np.array([.65, .5, 1.2, 1.2]+[10]*8+[10, 10, .9, 1.08])*.05
    results = []
    for row in rows:
        q = np.array(row['q']); previous = np.array(row['previous_target'])
        delayed = previous.copy(); controller = ExplicitOpenController()
        distance = [fk.separation(q)['distal_tactile_origins']]
        for step in range(60):
            original = q.copy()
            target, event = controller.update(q, previous, step in (0, 10), lower, upper)
            np.testing.assert_array_equal(q, original)
            assert event['accepted'] == (step == 0)
            assert np.max(np.abs(target-previous)) <= .025001
            assert np.all(target >= lower) and np.all(target <= upper)
            np.testing.assert_array_equal(target[4:12], previous[4:12])
            assert target[3] == previous[3]
            cmd = delayed.copy(); cmd[14] = target[14]
            q += np.clip(gain*(cmd-q), -vmax, vmax)
            q[3] = np.clip(np.polyval([-.29349962130965684, 1.3889380482483533, -.0013221000325437673], q[2]), 0, 1.57)
            delayed = target.copy(); previous = target
            distance.append(fk.separation(q)['distal_tactile_origins'])
        assert event['mode'] == 'hold' and event['automatic_resume'] is False
        assert (distance[-1]-distance[0])*1000 > 30
        assert event['measured_goal_error_rad'] < .001
        results.append(dict(step=row['step'], endpoint_gain_mm=(distance[-1]-distance[0])*1000,
                            min_transient_delta_mm=(min(distance)-distance[0])*1000,
                            gain_at_2s_mm=(distance[40]-distance[0])*1000))
    counter = [0]
    loop = explicit_open_loop_class(PolicyLoop)(policy, lambda: counter[0] in (45, 50))
    for step, row in enumerate(rows):
        counter[0] = step
        result = loop.step(row['q'], 20)
        np.testing.assert_array_equal(result['q'], np.asarray(row['q'], dtype=np.float32))
        if step >= 45:
            assert result['explicit_open']['active']
            assert np.max(np.abs(result['next_target']-result['previous_target'])) <= .025001
    try:
        ExplicitOpenController().update(np.full(16, np.nan), lower, True, lower, upper)
    except ValueError:
        pass
    else:
        raise AssertionError('Nonfinite input accepted')
    replay = None
    if a.evaluation:
        evaluation = json.loads(a.evaluation.read_text())
        assert evaluation['checkpoint_sha256'] == hashlib.sha256(a.artifact.read_bytes()).hexdigest()
        trigger = evaluation['explicit_open']['trigger_step']
        assert evaluation['explicit_open']['resume_after'] == 0
        replay_step = [0]
        replay_loop = explicit_open_loop_class(PolicyLoop)(policy, lambda: replay_step[0] == trigger)
        target_error = action_error = 0.0
        records = evaluation['explicit_open']['records']
        for i, (state, actual) in enumerate(zip(records, evaluation['first_episode_steps'])):
            replay_step[0] = i
            result = replay_loop.step(state['q'], 20)
            target_error = max(target_error, float(np.max(np.abs(result['next_target']-actual['joint_targets']))))
            action_error = max(action_error, float(np.max(np.abs(result['action']-actual['action']))))
        assert target_error < 1e-5 and action_error < 1e-4, (target_error, action_error)
        replay = dict(steps=len(records), target_max_error_rad=target_error, action_max_error=action_error,
                      path=str(a.evaluation))
    with tempfile.TemporaryDirectory() as temp:
        log = Path(temp)/'log'
        cmd = [sys.executable, 'scripts/deploy_m24_e003_student.py', '--explicit-open',
               '--jit-policy', str(a.artifact), '--legacy-script', str(a.legacy_script),
               '--dry-run', '--max-policy-steps', '60', '--log-dir', str(log)]
        proc = subprocess.run(cmd, input='r\n', text=True, capture_output=True, timeout=30)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        events = [json.loads(x) for x in (log/'explicit_open.jsonl').read_text().splitlines()]
        assert len(events) == 60 and events[0]['accepted'] and events[-1]['mode'] == 'hold'
        assert all(e['active'] and not e['automatic_resume'] for e in events)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(status='passed', artifact_sha256=digest(a.artifact),
        source_sha256={name: digest(Path('scripts')/name) for name in
            ('deploy_m24_e003_student.py', 'm24_explicit_open.py', 'm24_hand_kinematics.py')},
        no_trigger_bitwise_steps=len(rows), measured_start_states=results,
        joint_feedback_unchanged=True, passive_target_unchanged=True, max_step_rad=.025,
        no_automatic_resume=True, native_fake_device_steps=60, hardware_operated=False,
        physical_trace_replay=replay,
        scope='Kinematic + identified free-space response proxy; not contact/hardware success')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'measured_start_states'}, indent=2))


if __name__ == '__main__':
    main()
