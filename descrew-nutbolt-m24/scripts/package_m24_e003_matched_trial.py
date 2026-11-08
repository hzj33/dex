#!/usr/bin/env python3
"""Package a matched-execution cycle trial; never claim strict transfer acceptance."""
import argparse
from collections import Counter
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
    parser.add_argument('--legacy', type=Path, required=True)
    parser.add_argument('--software', type=Path, required=True)
    parser.add_argument('--evaluations', type=Path, nargs='+', required=True)
    parser.add_argument('--original-evaluation', type=Path, required=True)
    parser.add_argument('--force', type=Path, required=True)
    parser.add_argument('--video-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    digest = sha(args.artifact)
    software = json.loads(args.software.read_text())
    source = Path(__file__).parent / 'deploy_m24_e003_student.py'
    assert software['artifact_sha256'] == digest
    assert software['native_jit_exact'] and software['sim_deploy_exact_steps'] >= 200
    assert software['privileged_inputs_poisoned'] and software['default_bundle_model_path_dry_run_steps'] >= 20
    assert software['deployment_entry_sha256'] == sha(source)
    assert software['legacy_sha256'] == sha(args.legacy)

    def result(path, matched):
        data = json.loads(path.read_text())
        assert data['checkpoint_sha256'] == digest and data['policy_algorithm'] == 'DeployStudent'
        summary = data['summary']
        assert summary['finite'] and summary['required_cycles'] == 5
        assert summary['diagnostic_timeout_multiplier'] == 1
        assert bool(data.get('real_execution_model')) == matched
        if matched:
            assert data['real_execution_model']['observation_target_source'] == 'unfiltered_sent_command'
            assert data['passive_coupling_diagnostic']
        cycles = [row['cycles'] for row in data['first_episodes']]
        return dict(source=str(path), source_sha256=sha(path), seed=summary['evaluation_seed'],
                    num_envs=summary['num_envs'], initial_joint_noise_rad=summary['initial_joint_noise_rad'],
                    cycles=dict(Counter(cycles)), min_cycles=min(cycles),
                    five_cycles=sum(row['last_reason'] == 5 for row in data['first_episodes']),
                    median_net_degrees=summary['median_net_degrees'])

    results = [result(path, True) for path in args.evaluations]
    assert {row['seed'] for row in results} == {42, 43, 44, 45, 46}
    nominal = next(row for row in results if row['seed'] == 42)
    assert nominal['initial_joint_noise_rad'] == 0 and nominal['five_cycles'] == nominal['num_envs']
    original = result(args.original_evaluation, False)
    force = json.loads(args.force.read_text())
    # Bind the force audit to the evaluated model, rather than trusting a filename.
    for row in force:
        assert json.loads(Path(row['path']).read_text())['checkpoint_sha256'] == digest
    trace = json.loads((args.video_root / 'trace.json').read_text())
    video_check = json.loads((args.video_root / 'video_verification.json').read_text())
    assert trace['checkpoint_sha256'] == digest and trace['policy_algorithm'] == 'DeployStudent'
    assert trace['rows'][-1]['cycles'] == 5 and trace['rows'][-1]['reason'] == 5
    assert video_check['checkpoint_sha256'] == digest and video_check['fps'] == 20
    assert all(value == 0 for value in video_check['trace_exact_fields'].values())
    args.output.mkdir(parents=True, exist_ok=False)
    for src, name in [(args.artifact, 'student.pt'), (args.legacy, 'm24_legacy_deployment.py'),
                      (source, source.name), (args.software, 'software_verification.json'),
                      (args.force, 'force_diagnostic.json'),
                      (args.video_root / 'student_20hz.mp4', 'student_20hz.mp4'),
                      (args.video_root / 'contact_sheet.jpg', 'contact_sheet.jpg'),
                      (args.video_root / 'video_verification.json', 'video_verification.json')]:
        shutil.copy2(src, args.output / name)
    launcher = args.output / 'run_cycle_trial.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo '完整循环策略候选：自动拧转/松开/回位/重抓，无需按R；默认最多200步。尚未真机验收，5N未通过。'
exec "${M24_PYTHON:-/home/vision/software/miniconda3/envs/sim2real/bin/python}" "$bundle_dir/deploy_m24_e003_student.py" \\
  --hand-port /dev/ttyUSB_right --disable-rm-keyboard --max-policy-steps 200 "$@"
''')
    launcher.chmod(0o755)
    peaks = force[0]['physics']['peak_N']
    total = sum(row['five_cycles'] for row in results if row['seed'] != 42)
    count = sum(row['num_envs'] for row in results if row['seed'] != 42)
    (args.output / 'README.md').write_text(f'''# E003 完整循环策略：真机短试候选

这是匹配实测执行响应后重新适配的策略，不是R开手诊断。模型只使用真实关节、上一发送目标和内部历史，自主推断阶段并输出完整循环。模型SHA256：{digest}。

## 运行

使用此前确认的右手端口、M24装夹和原部署环境：

```bash
bash run_cycle_trial.sh
```

不需要按R。默认200个20Hz策略步（约10秒，不含初始化），0.05rad/步，原限位与中指/无名指屏蔽保留。初始姿态743由模型设置。不要通过反复按R或手动张开来替代该策略的闭环。

先做不连接硬件的软件检查：`bash run_cycle_trial.sh --dry-run --max-policy-steps 20`。
正常步数到期沿用原入口关闭/禁用电机；Ctrl+C沿用原停止流程，入口顶部RETURN_TO_ZERO_ON_STOP=True会先发零位再禁用，零位并非装夹姿态。若Python位置不同，设置M24_PYTHON。

## 已验证与未验证

- 辨识执行+被动DIP接触模型：固定姿态五轮{nominal['five_cycles']}/{nominal['num_envs']}；四组±1°扰动五轮{total}/{count}，各组明细在manifest。
- 附视频为同一学生权重的20Hz原生物理录像，阶段只用于评分；没有人工脉冲、阶段oracle或显式张开轨迹。
- 实际权重CPU native/JIT/仿真/部署200步一致，privileged输入置NaN仍可运行，外部工作目录假设备检查通过。
- 原执行模型五轮{original['five_cycles']}/{original['num_envs']}。本包针对实测响应适配，不能标成原仿真全部验收通过。
- 200Hz仿真食指/拇指法向力峰值{peaks[0]:.2f}/{peaks[1]:.2f}N，5N及完整受力转移未通过。空载执行吻合不等于接触力校准，不能把该数字当作真机受力保证。此包不是无人值守或最终低力合格版本。
- 尚未运行真实硬件。仿真循环进展支持短时验证，但不能保证部署即正常拧M24，也不证明拧紧扭矩达标。

本次实机判据：手指实际离开螺母、回位重抓并再次带动螺母产生净拧转；仅看到关节摆动或模型release标签不算通过。先保持默认短时上限，发现持续夹持/异常则结束本次；不要延长运行来掩盖卡住。

保存logs/<时间>/policy_steps.jsonl与run_metadata.json，并记录实际松开、重抓和螺母净转动。下一步依据这些真实闭环结果判断剩余执行/接触差异。旧部署包和原教师均未覆盖。
''')
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
                    status='matched_execution_cycle_trial;strict_transfer_not_passed',
                    artifact_sha256=digest, teacher_sha256=software['teacher_sha256'],
                    external_phase_input=False, manual_intervention=False,
                    real_hardware_validated=False, strict_transfer_acceptance=False,
                    force5_pass=False, action_scale=0.05, policy_hz=20,
                    matched_results=results, original_result=original, force_diagnostic=force,
                    files={p.name: sha(p) for p in args.output.iterdir() if p.is_file()})
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    archive = args.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as out:
        for path in sorted(args.output.iterdir()):
            out.write(path, str(Path(args.output.name) / path.name))
    print(json.dumps(dict(bundle=str(args.output), zip=str(archive), zip_sha256=sha(archive)), indent=2))


if __name__ == '__main__':
    main()
