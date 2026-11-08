#!/usr/bin/env python3
"""Package a verified runtime candidate without turning failed metrics into passes."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from datetime import datetime, timezone
import numpy as np

from analyze_m24_e003_loads import summarize


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--evaluations', type=Path, nargs=5, required=True, help='Seeds 42 (nominal),43,44,45,46')
    p.add_argument('--teacher-root', type=Path, required=True)
    p.add_argument('--legacy-script', type=Path, required=True)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--video-trace', type=Path, required=True)
    p.add_argument('--physics-report', type=Path, required=True)
    p.add_argument('--extended-report', type=Path, help='Optional longer-horizon diagnostic, not part of five-cycle acceptance')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    digest = sha(a.artifact)
    contract = json.loads(a.contract.read_text())
    assert contract['artifact_sha256'] == digest
    assert contract['native_jit_exact'] and contract['sim_deploy_exact_steps'] >= 200
    assert contract['default_bundle_model_path_dry_run_steps'] >= 20
    assert contract['action_scale'] == 0.05 and contract['policy_hz'] == 20
    assert contract['deployment_entry_sha256'] == sha(Path(__file__).with_name('deploy_m24_e003_student.py'))
    assert sha(a.legacy_script) == '70e875fcdc41de305e59fa347d4fdeb25d6de27b9659ad8bc1fc7142fe422ed4'
    results, baseline = [], []
    for seed, report in zip((42, 43, 44, 45, 46), a.evaluations):
        data = json.loads(report.read_text())
        assert data['checkpoint_sha256'] == digest and data['policy_algorithm'] == 'DeployStudent'
        assert data['summary']['required_cycles'] == 5 and data['summary']['finite']
        assert data['summary']['evaluation_seed'] == seed and data['summary']['num_envs'] == 128
        row = summarize(report); row.pop('episodes'); results.append(row)
        original = a.teacher_root/('teacher_%d' % seed)/'evaluation.json'
        teacher_data = json.loads(original.read_text())
        assert teacher_data['checkpoint_sha256'] == contract['teacher_sha256']
        assert teacher_data['summary']['required_cycles'] == 5 and teacher_data['summary']['finite']
        for key in ('evaluation_seed', 'num_envs', 'initial_joint_noise_rad', 'nominal_reset_fraction'):
            assert data['summary'][key] == teacher_data['summary'][key]
        row = summarize(original); row.pop('episodes'); baseline.append(row)
    reliability = results[0]['successes'] == 128 and sum(x['successes'] for x in results[1:]) >= sum(x['successes'] for x in baseline[1:])
    assert reliability, 'Do not distribute a student that fails the original five-cycle reliability checks'
    force = all(results[j]['normal_force_peak_N'][i] <= baseline[j]['normal_force_peak_N'][i]*1.1 for j in range(5) for i in (0, 1))
    slip = all(results[j]['mean_thumb_loss_s'] <= baseline[j]['mean_thumb_loss_s']*1.1+0.05 for j in range(5))
    trace = json.loads(a.video_trace.read_text())
    assert sha(Path(trace['checkpoint'])) == digest and trace['policy_algorithm'] == 'DeployStudent'
    assert trace['rows'][-1]['cycles'] == 5 and trace['rows'][-1]['reason'] == 5
    physics = json.loads(a.physics_report.read_text())
    assert physics['checkpoint_sha256'] == digest and physics['summary']['required_cycles'] == 5
    assert physics['first_episodes'][0]['last_reason'] == 5
    archive = a.physics_report.with_suffix('.physics.npz')
    physical_data = np.load(archive)
    physics_metadata = json.loads(archive.with_suffix('.metadata.json').read_text())
    # IsaacGym stores dt as float32 (0.04999999888 s for this 20 Hz profile).
    assert abs(physics_metadata['policy_dt'] - 0.05) < 1e-7
    assert physical_data['nut_normal'].shape[1] == 1
    physics_summary = dict(archive_sha256=sha(archive), sample_hz=1/physics_metadata['physics_dt'],
                           note='Complete single-environment five-cycle trajectory. Normal resultants, not full friction wrench or motor current.')
    for key in ('nut_normal', 'whole_finger_normal'):
        force_values = np.linalg.norm(physical_data[key][:,0], axis=-1)
        assert np.isfinite(force_values).all()
        physics_summary[key+'_peak_N'] = force_values.max(0).tolist()
    extended = None
    extended_note = '本次验收范围为连续五轮，不承诺无限循环。'
    if a.extended_report:
        extra = json.loads(a.extended_report.read_text())
        assert extra['checkpoint_sha256'] == digest and extra['summary']['finite']
        counts = [row['cycles'] for row in extra['first_episodes']]
        extended = dict(report=str(a.extended_report), required_cycles=extra['summary']['required_cycles'],
                        completed_cycles=counts, termination_counts=extra['summary']['termination_counts'])
        extended_note += f'附加{extended["required_cycles"]}轮试验实际完成{min(counts)}–{max(counts)}轮；详见manifest，不混入原五轮通过率。'
    a.output.mkdir(parents=True, exist_ok=False)
    for source, name in ((a.artifact, 'student.pt'), (a.legacy_script, 'm24_legacy_deployment.py'),
                         (Path(__file__).with_name('deploy_m24_e003_student.py'), 'deploy_m24_e003_student.py'),
                         (a.contract, 'software_verification.json'), (a.video, 'student_20hz.mp4')):
        shutil.copy2(source, a.output/name)
    status = 'passed_simulated_transfer' if force and slip else 'deployable_runtime_candidate;strict_transfer_not_passed'
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(), status=status,
                    artifact_sha256=digest, teacher_sha256=contract['teacher_sha256'],
                    action_scale=0.05, policy_hz=20, external_phase_input=False,
                    inputs=['measured_joint_positions', 'previous_sent_targets', 'joint_target_history'],
                    history_shape=[80, 32], joint_order='index0..3,middle0..3,ring0..3,thumb0..3',
                    masked_action_indices=list(range(4, 12)), normalization_in_model=True,
                    acceptance=dict(five_cycle_reliability=reliability, normal_force=force, thumb_slip=slip),
                    strict_force_criterion='Each seed/finger peak <= 110% of reproduced teacher; unchanged',
                    student_results=results, teacher_results=baseline, software_contract=contract,
                    physics_report=str(a.physics_report), video_trace=str(a.video_trace),
                    physics_audit=physics_summary,
                    extended_diagnostic=extended,
                    real_hardware_validated=False, hardware_operated=False,
                    deferred_goals=['5 N contact force', 'torque balance'],
                    files={x.name:sha(x) for x in a.output.iterdir() if x.is_file()})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    nf = results[0]['normal_force_peak_N']; tf = baseline[0]['normal_force_peak_N']
    note = ('逐组受力保持检查通过。' if force else '逐组受力保持检查未通过；此包是可运行候选，不标记为完整低力转移合格。')
    readme = f'''# E003 可直接运行的 student

仅使用手的关节读数、已发送目标和内部历史。模型内部包含归一化、阶段推断和目标积分，不需要仿真提供阶段、触觉或螺母角度。

控制设置：0.05 rad/步、20 Hz，E003 初始位743；中指和无名指动作屏蔽。使用原M24的串口、SDK、16关节顺序、日志和停止流程。模型SHA256：{digest}。

## 实测状态

- 原五轮固定测试：{results[0]['successes']}/128；扰动：{sum(x['successes'] for x in results[1:])}/512，教师{sum(x['successes'] for x in baseline[1:])}/512。
- 固定测试食指/拇指法向力峰值：{nf[0]:.2f}/{nf[1]:.2f} N，教师{tf[0]:.2f}/{tf[1]:.2f} N。
- {note}
- {extended_note}
- 已通过实际权重CPU导出一致性、200步仿真/部署接口一致性和原M24入口假设备空跑。附student_20hz.mp4为该权重的实际五轮仿真录像。
- 尚无实机执行结果。仿真Kp=2.4不会自动设置成实机增益；沿用原M24 SDK位置控制。5N和力矩平衡仍为后续目标。

## 运行

把此文件夹放到原M24部署机器，使用原来的sim2real Python/SDK环境。无需安装IsaacGym或本项目训练代码。在本目录执行：

```bash
python deploy_m24_e003_student.py --dry-run --max-policy-steps 20
```

仅控制手的实机命令（此命令会启用电机）：

```bash
python deploy_m24_e003_student.py --hand-port /dev/ttyUSB_right --disable-rm-keyboard --max-policy-steps 200
```

本机右手固定别名为/dev/ttyUSB_right（当前ttyUSB1），ttyUSB0为左手；其他机器按实际连接修改。上述200步限制是约10秒试运行上限，不是按时间切换阶段或判定完成五轮；阶段仍由模型推断。需要原M24的RM键盘微调时去掉`--disable-rm-keyboard`，其RM连接设置和按键保持原流程。Ctrl-C沿用原M24的退出/手归零流程。日志位于本目录logs。不要用旧M24的96维模型加载器直接载入此权重，也不要额外归一化或改成0.02步幅。

完整指标、来源和未通过项见manifest.json；提供可运行候选包不改变受力验收结论。
'''
    (a.output/'README.md').write_text(readme)
    print(json.dumps(dict(bundle=str(a.output),status=status,acceptance=manifest['acceptance']),indent=2))


if __name__ == '__main__':
    main()
