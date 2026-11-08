#!/usr/bin/env python3
"""Independent recovery trial bundle; no strict-transfer or hardware claim."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--software', type=Path, required=True)
    parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
    parser.add_argument('--force-evaluation', type=Path, required=True)
    parser.add_argument('--weight-verification', type=Path)
    parser.add_argument('--force-summary', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = sha(args.artifact)
    verification = json.loads(args.software.read_text())
    assert verification['artifact_sha256'] == digest and verification['return_entry_clock']
    assert verification['native_jit_sim_deploy_exact_steps'] >= 400
    assert verification['fake_device_steps'] >= 20 and verification['privileged_inputs_poisoned']
    entry = args.source_bundle/'deploy_m24_e003_student.py'
    legacy = args.source_bundle/'m24_legacy_deployment.py'
    assert sha(entry) == verification['deployment_entry_sha256']
    assert sha(legacy) == verification['legacy_sha256']
    results = []
    for path in args.evaluations:
        d = json.loads(path.read_text());s = d['summary']
        assert d['checkpoint_sha256'] == digest and d['policy_algorithm'] == 'DeployStudent'
        assert s['finite'] and s['required_cycles'] == 20 and s['diagnostic_timeout_multiplier'] == 1
        assert d['continuous_nut_receipt'] and not d.get('deployment_assistance') and not d.get('recovery_branch')
        results.append(dict(path=str(path), sha256=sha(path), seed=s['evaluation_seed'],
            num_envs=s['num_envs'], twenty_cycles=sum(r['cycles']==20 and r['last_reason']==5 for r in d['first_episodes']),
            initial_noise_rad=s['initial_joint_noise_rad']))
    assert any(r['num_envs']==1 and r['initial_noise_rad']==0 and r['twenty_cycles']==1 for r in results)
    assert len({r['seed'] for r in results if r['num_envs']>=16}) >= 2
    force = json.loads(args.force_evaluation.read_text())
    assert force['checkpoint_sha256'] == digest
    physics = args.force_evaluation.with_suffix('.physics.npz')
    assert physics.exists()
    args.output.mkdir(parents=True, exist_ok=False)
    for source, name in [(args.artifact, 'student.pt'), (entry, entry.name), (legacy, legacy.name),
                         (args.software, 'software_verification.json'), (args.force_evaluation, 'force_evaluation.json')]:
        shutil.copy2(source, args.output/name)
    for name, steps, message in [('run_recovery_trial.sh',1000,'回位进度修正试运行：默认50秒，采集新日志。'),
                                 ('run_deploy.sh',0,'回位进度修正持续部署：Ctrl+C结束。')]:
        script = args.output/name
        script.write_text('#!/usr/bin/env bash\nset -euo pipefail\n'
            'bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
            f'echo "{message}"\n'
            'exec "${M24_PYTHON:-/home/vision/software/miniconda3/envs/sim2real/bin/python}" '
            '"$bundle_dir/deploy_m24_e003_student.py" '
            '--legacy-script "$bundle_dir/m24_legacy_deployment.py" '
            '--jit-policy "$bundle_dir/student.pt" --hand-port /dev/ttyUSB_right '
            f'--disable-rm-keyboard --max-policy-steps {steps} "$@"\n')
        script.chmod(0o755)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), artifact_sha256=digest,
        policy_hz=20, action_scale=.05, return_entry_clock=True,
        change='return actor command clock fixed to0 only when learned phase argmax is return',
        measured_joints_modified=False, history_reset=False, teacher_weights_changed=False,
        deployment_entry_unchanged=True, legacy_unchanged=True, acceptance_eligible=False,
        hardware_validated=False, evaluations=results,
        force_physics_archive=str(physics), force_physics_sha256=sha(physics),
        files={f.name:sha(f) for f in args.output.iterdir() if f.is_file()})
    if args.weight_verification:
        weights = json.loads(args.weight_verification.read_text())
        assert weights['sha256'] == digest and weights['all_weights_and_buffers_identical']
        shutil.copy2(args.weight_verification, args.output/'weight_preservation.json')
        manifest.update(student_weights_changed=False, source_model_sha256=sha(Path(weights['source'])))
    if args.force_summary:
        loads = json.loads(args.force_summary.read_text())
        assert loads['artifact_sha256'] == digest and loads['hardware_operated'] is False
        shutil.copy2(args.force_summary, args.output/'force_summary.json')
        manifest['peak_whole_finger_normal_N'] = loads['peak_whole_finger_normal_N']
    (args.output/'README.md').write_text('''# 回位进度修正部署测试包

目标是持续旋拧→松开→回位→重抓，不是独立开手。模型仅在自己判断为return时将actor的控制进度置0，使回位动作不被饱和进度困住；何时进入/离开return仍由模型根据真实关节与发送目标历史推断。它不把仿真阶段/接触/物体状态送入真机。

20Hz、0.05rad、SDK限位、被动关节处理及中指/无名指屏蔽均保留；没有R辅助、预设开手轨迹、按次数重启或历史清零。教师权重未训练；模型是否补训学生见manifest和项目执行记录。

优先试运行：`bash run_recovery_trial.sh`，默认1000策略步约50秒。观察此前停滞时段之后，是否仍能真正回位、重抓并带来螺母净进展；日志位于本包logs。持续入口：`bash run_deploy.sh`。Ctrl+C行为沿用源包的用户设置，当前先复位后禁用电机；不另改停止开关。

这是独立候选包，原20260920完整包保留。仿真20轮和软件一致性不代表真机已经通过，长期稳定性、真实拧紧终点/扭矩及5N未验收。接触物理报告和200Hz归档地址见manifest；不要把手指往返或预测阶段当作真实螺母完成数。

回退：运行原`outputs/local_m24_deployment/e003_matched_full_20260920/run_deploy.sh`。
''')
    manifest['files'] = {f.name:sha(f) for f in args.output.iterdir() if f.is_file()}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    zip_path = args.output.with_suffix('.zip')
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in args.output.rglob('*'):
            if f.is_file():z.write(f, f.relative_to(args.output.parent))
    with zipfile.ZipFile(zip_path) as z:assert z.testzip() is None
    print(json.dumps(dict(bundle=str(args.output), artifact_sha256=digest, zip=str(zip_path), zip_sha256=sha(zip_path))))


if __name__ == '__main__':
    main()
