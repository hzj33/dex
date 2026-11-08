#!/usr/bin/env python3
"""Package the verified air-only explicit-opening diagnostic, never task acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-bundle', type=Path, required=True)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    software = json.loads((a.evidence/'software_verification.json').read_text())
    assert software['status'] == 'passed' and software['no_automatic_resume']
    assert software['physical_trace_replay']['target_max_error_rad'] < 1e-5
    assert software['artifact_sha256'] == digest(a.source_bundle/'student.pt')
    for name, sha in software['source_sha256'].items():
        assert digest(Path('scripts')/name) == sha
    physics = json.loads((a.evidence/'physics_summary.json').read_text())
    assert {'early_hold', 'early_resume', 'late_resume', 'passive_hold'} == {r['name'] for r in physics}
    assert all(any(e[1] == 2 for e in r['events']) for r in physics)
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ('student.pt', 'm24_legacy_deployment.py'):
        shutil.copy2(a.source_bundle/name, a.output/name)
    for name in ('deploy_m24_e003_student.py', 'm24_explicit_open.py', 'm24_e003_release_override.py'):
        shutil.copy2(Path('scripts')/name, a.output/name)
    evidence = a.output/'evidence'
    evidence.mkdir()
    for name in ('software_verification.json', 'old_mode_regression.json', 'physics_summary.json'):
        shutil.copy2(a.evidence/name, evidence/name)
    launcher = a.output/'run_explicit_open.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${M24_PYTHON:-/home/vision/software/miniconda3/envs/sim2real/bin/python}" "$bundle_dir/deploy_m24_e003_student.py" \
  --explicit-open --hand-port /dev/ttyUSB_right --disable-rm-keyboard --max-policy-steps 200 "$@"
''')
    launcher.chmod(0o755)
    (a.output/'README.md').write_text('''# 显式张开：空载诊断版

用途：验证按 R 后真机能否明显张开。不是自主拧 M24，也不是接触/受力验收通过。

运行 `bash run_explicit_open.sh`。保持空载，开始运动后前5秒内按 **R 再回车**。
R 会暂停策略输出，以触发时的真实关节角为基准，把 index_1 和 thumb_2 各减小0.30rad（约17.2°）；其他主动关节缓慢保持触发时角度，机械被动 index_3 与中指/无名指目标保持不变。
发送目标每步最多变化0.025rad，约2秒内观察张开趋势；随后保持该目标，**不会自动恢复策略或重抓**。重复 R 不继续增大张开量。
控制台的 holding 表示发送目标已到终点，不等于实测已到位。实测跟踪误差与目标保存在 logs 下的 explicit_open.jsonl 和 policy_steps.jsonl。

默认总运行200步，约10秒后禁用电机并断开；Ctrl+C 沿用原入口的全关节零位后禁用行为（RETURN_TO_ZERO_ON_STOP=True）。机械臂键盘控制默认关闭。

同一模型仍为 a7c00a27…；不按 R 时与原入口逐位一致。该版本是显式辅助轨迹，未训练新策略，也未伪造关节反馈。
200个实测起始姿态的一阶空载模型最终几何间距均增大，但部分起点可先短暂缩小约4.24mm；URDF距离是参考点间距，不是测得的接触间隙。
接触仿真两种执行模型均达到离触；恢复策略仍失败，5N未过，晚触发可越过工作空间。因此本包只交付空载诊断，不把它用于已夹紧螺母的释放验收。

软件验证：实际权重、200步不触发精确、163步物理轨迹目标精确、限步幅/限位/被动关节/真实反馈保持、60步原入口假设备。证据见 evidence/ 与 manifest.json。
''')
    with tempfile.TemporaryDirectory() as temp:
        log_dir = Path(temp)/'logs'
        command = [sys.executable, str((a.output/'deploy_m24_e003_student.py').resolve()),
                   '--explicit-open', '--dry-run', '--max-policy-steps', '60', '--log-dir', str(log_dir)]
        run = subprocess.run(command, input='r\n', capture_output=True, text=True, cwd='/', timeout=30)
        assert run.returncode == 0, run.stdout+run.stderr
        events = [json.loads(x) for x in (log_dir/'explicit_open.jsonl').read_text().splitlines()]
        assert len(events) == 60 and events[0]['accepted'] and events[-1]['mode'] == 'hold'
        assert all(not x['automatic_resume'] for x in events)
    manifest = dict(status='air_only_explicit_open_diagnostic', artifact_sha256=software['artifact_sha256'],
        hardware_validated=False, contact_deployment_validated=False, autonomous_cycles_validated=False,
        automatic_resume=False, force5_pass=False, external_cwd_fake_device_steps=60,
        files={str(f.relative_to(a.output)): digest(f) for f in sorted(a.output.rglob('*')) if f.is_file()})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = a.output.with_suffix('.zip')
    assert not archive.exists()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(a.output.rglob('*')):
            if path.is_file():
                z.write(path, str(path.relative_to(a.output.parent)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    receipt = dict(bundle=str(a.output), archive=str(archive), archive_sha256=digest(archive),
                   status=manifest['status'], external_cwd_fake_device_steps=60)
    (a.evidence/'bundle_verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
