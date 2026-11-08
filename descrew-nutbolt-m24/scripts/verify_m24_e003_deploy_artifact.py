#!/usr/bin/env python3
"""Verify an actual trained control artifact on CPU, without using the hand SDK."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from dexscrew.algo.models.e003_deploy_student import DeployStudent, TEACHER_SHA
from deploy_m24_e003_student import Policy, PolicyLoop
from m24_e003_deploy_runtime import DeployInference
from types import SimpleNamespace


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--legacy-script', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher', type=Path, default=Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/selected_teacher/e003_compliant_frozen_teacher.pth'))
    p.add_argument('--teacher-sha256', default=TEACHER_SHA)
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    teacher = a.teacher
    assert sha(teacher) == a.teacher_sha256
    assert sha(a.legacy_script) == '70e875fcdc41de305e59fa347d4fdeb25d6de27b9659ad8bc1fc7142fe422ed4'
    torch.set_num_threads(1)
    torch.manual_seed(743)
    scripted = torch.jit.load(str(a.artifact), map_location='cpu').eval()
    assert scripted.action_scale == 0.05 and scripted.policy_hz == 20
    calibration = json.loads(Path('outputs/cloud_pipeline_m24_e003_student_20260914/a0c/manifests/a0c_diagnostic.json').read_text())
    checkpoint = torch.load(teacher, map_location='cpu', weights_only=False)
    native = DeployStudent(checkpoint, calibration['custom_init_dof_pos'], 0.05).eval()
    trained = scripted.state_dict()
    # The encoder is the only trainable component; all other tensors must match.
    for key, tensor in native.state_dict().items():
        if not key.startswith('encoder.'):
            torch.testing.assert_close(tensor, trained[key], rtol=0, atol=0)
    native.load_state_dict(trained, strict=True)
    env = SimpleNamespace(device='cpu', rl_device='cpu', num_actions=16,
                          target_smoothing_alpha=0, config={'env': {'controller': {}}})
    simulation = DeployInference(env, a.artifact)
    deployment = PolicyLoop(Policy(a.artifact))
    previous = native.init_pose.reshape(1, 16).clone()
    history = native.reset_history(previous, previous)
    durations = []
    with torch.no_grad():
        for step in range(200):
            q = previous + torch.randn_like(previous) * 0.001
            frame = torch.cat((q, previous), -1).reshape(1, 1, 32)
            obs = {'proprio_hist': frame, 'obs': torch.full((1, 101), float('nan')),
                   'priv_info': torch.full((1, 126), float('nan')),
                   'point_cloud_info': torch.full((1, 100, 3), float('nan'))}
            expected, next_history, expected_action = native(q, previous, history)
            actual = scripted(q, previous, history)
            for x, y in zip((expected, next_history, expected_action), actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            action, _ = simulation._eval_select_action(obs)
            start = time.perf_counter()
            record = deployment.step(q.numpy()[0], 20)
            if step >= 20:
                durations.append((time.perf_counter() - start) * 1000)
            torch.testing.assert_close(action, expected_action, rtol=0, atol=0)
            torch.testing.assert_close(torch.from_numpy(record['next_target']).reshape(1, 16), expected, rtol=0, atol=0)
            torch.testing.assert_close(deployment.history, next_history, rtol=0, atol=0)
            assert torch.isfinite(expected).all() and (action[:, 4:12] == 0).all()
            assert (expected >= native.lower).all() and (expected <= native.upper).all()
            assert float((expected - previous).abs().max()) <= 0.050001
            previous, history = expected, next_history
    # Copy a self-contained bundle and invoke its default model path from another cwd.
    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp)
        shutil.copy2(a.artifact, bundle/'student.pt')
        shutil.copy2(a.legacy_script, bundle/'m24_legacy_deployment.py')
        shutil.copy2(Path(__file__).with_name('deploy_m24_e003_student.py'), bundle/'deploy_m24_e003_student.py')
        result = subprocess.run([sys.executable, str(bundle/'deploy_m24_e003_student.py'),
                                 '--dry-run', '--max-policy-steps', '20', '--log-dir', str(bundle/'dryrun')],
                                cwd='/', capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
        rows = [json.loads(x) for x in (bundle/'dryrun/policy_steps.jsonl').read_text().splitlines()]
        assert len(rows) == 20
    report = dict(artifact=str(a.artifact), artifact_sha256=sha(a.artifact), teacher_sha256=a.teacher_sha256,
                  deployment_entry_sha256=sha(Path(__file__).with_name('deploy_m24_e003_student.py')),
                  legacy_sha256=sha(a.legacy_script), python_executable=sys.executable, torch_version=torch.__version__,
                  actual_trained_artifact=True, frozen_teacher_and_controller_tensors_exact=True,
                  native_jit_exact=True, sim_deploy_exact_steps=200, privileged_inputs_poisoned=True,
                  default_bundle_model_path_dry_run_steps=20, sdk_imported=False, hardware_operated=False,
                  action_scale=0.05, policy_hz=20,
                  cpu_control_step_ms=dict(median=float(np.median(durations)), p99=float(np.percentile(durations, 99)),
                                           max=float(max(durations))),
                  note='Software contract only; does not certify force retention or real hardware behavior.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
