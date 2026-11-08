#!/usr/bin/env python3
"""Package an assisted diagnostic; never upgrades it to autonomous acceptance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--audit-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    root = a.audit_root
    software = json.loads((root/'manual_software_verification.json').read_text())
    standard = json.loads((root/'default_software_verification.json').read_text())
    evaluation = json.loads((root/'release_pulse/release_20_steps/evaluation.json').read_text())
    video_root = root/'video_retry/old_release20_video'
    video = json.loads((video_root/'trace.json').read_text())
    video_check = json.loads((video_root/'video_verification.json').read_text())
    digest = sha(a.artifact)
    assert all(x['artifact_sha256'] == digest for x in (software, standard))
    assert evaluation['checkpoint_sha256'] == video['checkpoint_sha256'] == digest
    assert software['status'] == 'passed' and software['joint_feedback_unchanged']
    assert software['physical_trace_replay']['target_max_error_rad'] == 0
    assert standard['native_jit_exact'] and standard['sim_deploy_exact_steps'] == 200
    source = Path(__file__).parent
    assert software['deployment_entry_sha256'] == standard['deployment_entry_sha256'] == sha(source/'deploy_m24_e003_student.py')
    assert software['override_source_sha256'] == sha(source/'m24_e003_release_override.py')
    assert standard['legacy_sha256'] == sha(a.legacy)
    assert video_check['fps'] == 20 and all(v == 0 for v in video_check['trace_exact_fields'].values())
    assert evaluation['phase_intervention']['mode'] == 'release_pulse'
    force = json.loads((root/'manual_force.json').read_text())[0]['physics']
    a.output.mkdir(parents=True, exist_ok=False)
    files = [(a.artifact, 'student.pt'), (a.legacy, 'm24_legacy_deployment.py'),
             (source/'deploy_m24_e003_student.py', 'deploy_m24_e003_student.py'),
             (source/'m24_e003_release_override.py', 'm24_e003_release_override.py'),
             (root/'manual_software_verification.json', 'manual_software_verification.json'),
             (root/'default_software_verification.json', 'default_software_verification.json'),
             (root/'manual_force.json', 'manual_force.json'),
             (video_root/'manual_release_20hz.mp4', 'manual_release_20hz.mp4'),
             (video_root/'contact_sheet.jpg', 'contact_sheet.jpg'),
             (video_root/'video_verification.json', 'video_verification.json')]
    for src, dst in files:
        shutil.copy2(src, a.output/dst)
    def result(path):
        d = json.loads(path.read_text())
        r = d['first_episodes'][0]
        return dict(path=str(path), cycles=r['cycles'], drive=r['drive_events'],
                    release=r['release_events'], returned=r['return_events'],
                    end_step=r['step'], termination=r['last_reason'])
    trials = {name: result(root/path) for name, path in {
        'prompt_trigger': 'release_pulse/release_20_steps/evaluation.json',
        'delay_0_5s': 'trigger_delay/delay_10_steps/evaluation.json',
        'delay_1s': 'trigger_delay/delay_20_steps/evaluation.json'}.items()}
    (a.output/'run_manual_release.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
bundle_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${M24_PYTHON:-/home/vision/software/miniconda3/envs/sim2real/bin/python}" "$bundle_dir/deploy_m24_e003_student.py" \\
  --manual-release --release-pulse-steps 20 --hand-port /dev/ttyUSB_right \\
  --disable-rm-keyboard --max-policy-steps 200 "$@"
''')
    (a.output/'run_manual_release.sh').chmod(0o755)
    readme = '''# E003 人工触发松开：短时部署诊断包

状态：manual_release_diagnostic_only。用于检验“人为给出松开阶段提示能否打破持续夹持”；未通过自主五轮、时机鲁棒性或5N验收，尚未在真机执行。不是连续正常拧M24的合格版本。

## 运行与触发

在原sim2real环境、右手及已确认的M24初始装夹下，从交互终端运行：

```bash
bash run_manual_release.sh
```

控制开始后，看到一次夹持旋拧需要松开时，按 `r` 再按回车。只将模型内部阶段指定为release，持续20步（约1秒），随后恢复模型自己推断阶段。关节角、发送目标和历史都保持真实，不发送固定张开姿态。连续脉冲期间再次按键会被忽略。

入口默认200个策略步（约10秒，不含初始化），禁用机械臂键盘控制，右手端口`/dev/ttyUSB_right`，20Hz、0.05rad/步，SDK限位保持。观察不到实际松开或动作异常时结束本次，不靠连续按键无限重试。Ctrl+C沿用原停止流程；文件顶部`RETURN_TO_ZERO_ON_STOP=True`会发全关节零位后禁用电机，零位不是初始夹持姿态。可按原使用习惯设置此开关。

先做纯软件检查（假设备，不连接硬件）可运行：

```bash
bash run_manual_release.sh --dry-run --max-policy-steps 22
```

测试时按r回车，终端应显示接收松开提示与恢复自主推断。若Python位置不同，用`M24_PYTHON=/你的/python`指定。直接调用Python入口时，必须加`--manual-release`才启用此模式；默认自主路径保持原行为。

## 已有证据与局限

- 使用原候选a7c00a27；本次修正契约后的新训练模型没有通过执行模型回位测试，未替换此权重。
- 修正后的真机执行近似仿真中，及时触发1秒release完成3轮，第4段drive超时；前10秒完成2轮。仿真触发来自真实drive事件，代替理想按键时刻，不等同人类或真机测试。
- 触发晚0.5秒或1秒：都完成首次松开，但0完整循环，回位失败。辅助方案对时机敏感，不能保证按键后完成整轮。
- 同权重200Hz法向力峰值食指/拇指约65.99/29.76N，5N目标未通过。这是含近似刚度设定的仿真诊断，不能作为真机受力预测或保证。
- 416步物理轨迹与部署辅助入口动作/目标逐位一致；不触发100步与原入口逐位一致；原模型200步CPU native/JIT/部署一致、假设备运行通过。未导入SDK执行真实控制。
- 附20Hz原生仿真录像，已核对416帧角度/循环/回位误差与数值逐位一致。录像标明辅助触发，不是自主成功演示。

## 测试后需要的数据

`logs/<时间>/policy_steps.jsonl`记录实际关节、目标和动作；同目录`manual_release.jsonl`记录每次按键是否接受、覆盖阶段和剩余步数；`run_metadata.json`标明人工辅助模式。请保留这些文件，并记录手指是否真正离开螺母、是否回位重抓。模型预测的release标签不能代替真实松开。
'''
    (a.output/'README.md').write_text(readme)
    manifest = dict(created_at=datetime.now(timezone.utc).isoformat(),
        status='manual_release_diagnostic_only', artifact_sha256=digest,
        autonomous_acceptance=False, five_cycle_acceptance=False,
        timing_robustness=False, force5_pass=force['force5_sampled_pass'],
        hardware_validated=False, hardware_operated=False, pulse_steps=20,
        max_initial_trial_policy_steps=200, physical_trial_results=trials,
        physics_force_peak_N=force['peak_N'], source_audit=str(root),
        files={p.name: sha(p) for p in a.output.iterdir() if p.is_file()})
    (a.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    archive = a.output.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(a.output.iterdir()):
            z.write(f, str(Path(a.output.name)/f.name))
    print(json.dumps(dict(bundle=str(a.output), zip=str(archive), zip_sha256=sha(archive)), indent=2))


if __name__ == '__main__':
    main()
