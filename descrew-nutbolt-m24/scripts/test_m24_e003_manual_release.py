#!/usr/bin/env python3
"""Actual-artifact checks for manual release; no SDK or hardware access."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from deploy_m24_e003_student import Policy, PolicyLoop
from m24_e003_release_override import manual_loop_class
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--legacy-script', type=Path, required=True)
    p.add_argument('--evaluation', type=Path)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    policy = Policy(a.artifact)
    manual = manual_loop_class(PolicyLoop)
    baseline = PolicyLoop(policy)
    idle = manual(policy, lambda: False)
    rng = np.random.default_rng(743)
    for _ in range(100):
        q = baseline.previous.numpy()[0] + rng.normal(0, .001, 16)
        x, y = baseline.step(q, 20), idle.step(q, 20)
        for key in ('q', 'previous_target', 'action', 'next_target'):
            np.testing.assert_array_equal(x[key], y[key])
    step = [0]
    active = manual(policy, lambda: step[0] in (5, 10, 35))
    enabled, accepted = [], []
    for i in range(70):
        step[0] = i
        q = active.previous.numpy()[0].copy()
        r = active.step(q, 20)
        np.testing.assert_array_equal(r['q'], q)
        assert np.max(np.abs(r['next_target'] - r['previous_target'])) <= .050001
        assert (r['action'][4:12] == 0).all()
        enabled.append(r['manual_release']['active'])
        if r['manual_release']['accepted']:
            accepted.append(i)
    assert accepted == [5, 35] and sum(enabled) == 40
    assert all(enabled[5:25]) and not any(enabled[25:35])
    replay = None
    if a.evaluation:
        d = json.loads(a.evaluation.read_text())
        assert d['checkpoint_sha256'] == hashlib.sha256(a.artifact.read_bytes()).hexdigest()
        rows = d['first_episode_steps']
        requests = {i+1 for i, r in enumerate(rows) if r['last_event'] == 1}
        step[0] = 0
        loop = manual(policy, lambda: step[0] in requests)
        target_error = action_error = 0.
        for i, row in enumerate(rows):
            step[0] = i
            # Policy sees the previous control endpoint; reset history exposes
            # the saved init pose. Physics substep0 is not that observation.
            q = policy.model.init_pose.numpy() if i == 0 else np.array(rows[i-1]['joint_state'])[:, 0]
            r = loop.step(q, 20)
            target_error = max(target_error, float(np.max(np.abs(r['next_target']-row['joint_targets']))))
            action_error = max(action_error, float(np.max(np.abs(r['action']-row['action']))))
        assert target_error < 1e-5 and action_error < 1e-4
        replay = dict(steps=len(rows), target_max_error_rad=target_error, action_max_error=action_error,
                      measured_joints_modified=False, trace=str(a.evaluation))
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp)/'manual'
        result = subprocess.run([sys.executable, 'scripts/deploy_m24_e003_student.py',
            '--jit-policy', str(a.artifact), '--legacy-script', str(a.legacy_script),
            '--manual-release', '--dry-run', '--max-policy-steps', '22', '--log-dir', str(log)],
            input='r\n', capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = [json.loads(x) for x in (log/'manual_release.jsonl').read_text().splitlines()]
        assert len(rows) == 22 and sum(r['active'] for r in rows) == 20
        assert rows[0]['accepted'] and not rows[-1]['active']
        meta = json.loads((log/'run_metadata.json').read_text())
        assert meta['manual_release']['autonomous'] is False
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(status='passed', artifact_sha256=digest(a.artifact),
        deployment_entry_sha256=digest(Path(__file__).with_name('deploy_m24_e003_student.py')),
        override_source_sha256=digest(Path(__file__).with_name('m24_e003_release_override.py')),
        no_trigger_bitwise_parity_steps=100,
        pulse_steps=20, active_retrigger_ignored=True, joint_feedback_unchanged=True,
        max_action_scale=.05, native_entry_fake_device_steps=22, physical_trace_replay=replay,
        hardware_operated=False)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
