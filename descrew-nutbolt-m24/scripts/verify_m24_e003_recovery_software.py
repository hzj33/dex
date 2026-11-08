#!/usr/bin/env python3
"""Actual recovery artifact's native/JIT/sim/deploy consistency, no hardware."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    entry = args.bundle/'deploy_m24_e003_student.py'
    spec = importlib.util.spec_from_file_location('verified_deploy', entry)
    deploy = importlib.util.module_from_spec(spec); spec.loader.exec_module(deploy)
    import numpy as np
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from dexscrew.algo.models.e003_deploy_student import DeployStudent
    from m24_e003_deploy_runtime import DeployInference
    torch.set_num_threads(12); torch.manual_seed(42)
    assert not args.output.exists()
    args.output.mkdir(parents=True)
    teacher = Path('outputs/local_pipeline_m24_e003_policy_adapter_20260920/selected_teacher/matched_teacher.pth')
    cal = json.loads(Path('outputs/cloud_pipeline_m24_e003_student_20260914/a0c/manifests/a0c_diagnostic.json').read_text())
    jit = torch.jit.load(str(args.artifact), map_location='cpu').eval()
    native = DeployStudent(torch.load(teacher, map_location='cpu', weights_only=False),
        cal['custom_init_dof_pos'], float(jit.action_scale),
        return_entry_clock=bool(getattr(jit, 'return_entry_clock', False))).eval()
    native.load_state_dict(jit.state_dict(), strict=True)
    assert float(jit.action_scale) == .05
    env = SimpleNamespace(device='cpu', rl_device='cpu', num_actions=16,
        target_smoothing_alpha=0, config={'env': {'controller': {}}})
    sim = DeployInference(env, args.artifact, record_predictions=True)
    real = deploy.PolicyLoop(deploy.Policy(args.artifact))
    hist = native.reset_history(native.init_pose[None], native.init_pose[None])
    previous = native.init_pose[None].clone()
    latency = []
    with torch.no_grad():
        for step in range(400):
            q = previous+torch.randn_like(previous)*.005
            expected, hist, action = native(q, previous, hist)
            frame = torch.cat((q, previous), -1).reshape(1, 1, 32)
            env.prev_targets = previous.clone()
            obs = dict(proprio_hist=frame, obs=torch.full((1, 101), float('nan')),
                priv_info=torch.full((1, 126), float('nan')),
                point_cloud_info=torch.full((1, 100, 3), float('nan')))
            simulated, _ = sim._eval_select_action(obs)
            t = time.perf_counter(); result = real.step(q.numpy()[0], 20)
            latency.append((time.perf_counter()-t)*1000)
            torch.testing.assert_close(simulated, action, rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(result['next_target'])[None], expected, rtol=0, atol=0)
            torch.testing.assert_close(torch.tensor(result['action'])[None], action, rtol=0, atol=0)
            torch.testing.assert_close(hist, sim.history, rtol=0, atol=0)
            torch.testing.assert_close(hist, real.history, rtol=0, atol=0)
            assert torch.equal(sim.contract_errors, torch.zeros(1, 2))
            assert (action[:, 4:12] == 0).all()
            assert (expected >= jit.lower-1e-7).all() and (expected <= jit.upper+1e-7).all()
            assert (expected-previous).abs().max() <= .05+1e-6
            previous = expected
    log = args.output/'dry_run'
    cmd = [sys.executable, str(entry), '--legacy-script', str(args.bundle/'m24_legacy_deployment.py'),
        '--jit-policy', str(args.artifact), '--dry-run', '--max-policy-steps', '20', '--log-dir', str(log)]
    process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (args.output/'dry_run.log').write_text(process.stdout)
    assert process.returncode == 0, process.stdout
    rows = [json.loads(x) for x in (log/'policy_steps.jsonl').read_text().splitlines()]
    assert len(rows) == 20
    report = dict(artifact=str(args.artifact), artifact_sha256=sha(args.artifact),
        teacher_sha256=sha(teacher), deployment_entry_sha256=sha(entry),
        legacy_sha256=sha(args.bundle/'m24_legacy_deployment.py'),
        return_entry_clock=bool(getattr(jit, 'return_entry_clock', False)),
        native_jit_sim_deploy_exact_steps=400, privileged_inputs_poisoned=True,
        max_target_step_rad=.05, policy_hz=20,
        latency_ms_percentiles=np.percentile(latency, [50, 99]).tolist(),
        fake_device_steps=20, fake_device_exit=0, hardware_operated=False)
    (args.output/'verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
