#!/usr/bin/env python3
"""Repackage a verified cycle candidate for operator-stopped deployment."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old = json.loads((a.source / 'manifest.json').read_text())
    assert sha(a.source / 'student.pt') == old['artifact_sha256']
    a.output.mkdir(parents=True, exist_ok=False)
    for name in ('student.pt', 'm24_legacy_deployment.py', 'deploy_m24_e003_student.py',
                 'software_verification.json', 'force_diagnostic.json',
                 'student_20hz.mp4', 'contact_sheet.jpg', 'video_verification.json'):
        shutil.copy2(a.source / name, a.output / name)
    entry = a.output / 'deploy_m24_e003_student.py'
    code = entry.read_text()
    assert code.count('RETURN_TO_ZERO_ON_STOP = True') == 1
    entry.write_text(code.replace('RETURN_TO_ZERO_ON_STOP = True', 'RETURN_TO_ZERO_ON_STOP = False'))
    launcher = a.output / 'run_deploy.sh'
    launcher.write_text('''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo 'M24连续部署：无需按R；Ctrl+C停止，不发零位，禁用电机并断开。'
exec "${M24_PYTHON:-/home/vision/software/miniconda3/envs/sim2real/bin/python}" "$bundle_dir/deploy_m24_e003_student.py" \\
  --hand-port /dev/ttyUSB_right --disable-rm-keyboard --max-policy-steps 0 "$@"
''')
    launcher.chmod(0o755)
    shutil.copy2(a.source / 'README.md', a.output / 'CANDIDATE_EVIDENCE.md')
    (a.output / 'README.md').write_text('''# M24 完整部署包（持续运行）

使用已获实机周期运动正反馈的a819学生，控制与模型不变。自动旋拧、松开、回位、重抓，无需按R。

```bash
bash run_deploy.sh
```

默认持续运行，取消200步/10秒上限。Ctrl+C停止时不发送复位/零位，直接禁用电机并断开，不是带力保持。异常退出同样不发送零位。若希望复位，可将deploy_m24_e003_student.py顶部RETURN_TO_ZERO_ON_STOP改回True。

沿用原环境及右手/dev/ttyUSB_right，20Hz、0.05rad/步、原743初始姿态、关节限位与中指/无名指屏蔽；机械臂键盘禁用。启动仍会移动到初始姿态；“不复位”仅指停止行为。

可选限时：`bash run_deploy.sh --max-policy-steps 600`（约30秒）。
软件自检：`bash run_deploy.sh --dry-run --max-policy-steps 20`（不连接设备）。
Python默认使用原sim2real环境，可用M24_PYTHON覆盖。

日志保存在本包logs/<时间>/，含policy_steps.jsonl、CSV与run_metadata.json。

完整部署指运行功能与封包齐全，不代表无限循环或全部实验指标合格。最新真机200步有周期运动且用户反馈效果好；较早一次后段停滞。螺母净进展、拧紧扭矩及5N未完成验收，没有自动拧紧终止或实测力保护。此前仿真、软件和受力证据见CANDIDATE_EVIDENCE.md与manifest；其中200步说明属于旧短试入口，当前以本README为准。不要无人值守运行。
''')
    manifest = dict(old)
    manifest.update(status='continuous_deployment;strict_transfer_not_passed',
                    source_bundle=str(a.source), source_manifest_sha256=sha(a.source / 'manifest.json'),
                    max_policy_steps_default=0, stop_return_to_zero=False,
                    stop_behavior='disable motors and disconnect; no zero command',
                    deployment_entry_changed_only='RETURN_TO_ZERO_ON_STOP True -> False',
                    historical_software_report_scope='source entry before stop-default change; see deployment_verification.json for this bundle',
                    files={f.name: sha(f) for f in a.output.iterdir() if f.is_file()})
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(a.output.with_suffix('.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            z.write(f, str(Path(a.output.name) / f.name))
    print(a.output)


if __name__ == '__main__':
    main()
