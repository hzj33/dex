#!/usr/bin/env python3
"""Use the original M24 SDK/lifecycle with a self-contained E003 control step.

The bundle supplies the byte-identical legacy deployment module from sim2real.
This entry only replaces its model/loop and profile, preserving serial, SDK,
20 Hz scheduling, stop behavior, logging and optional arm keyboard controls.
"""
import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
# Preserve the legacy SDK ABI preload order before importing NumPy or Torch.
import ctypes
ctypes.CDLL('/lib/x86_64-linux-gnu/libtiff.so.5', mode=ctypes.RTLD_GLOBAL)
import numpy as np
import torch


# Ctrl+C / 异常退出：True 发送全关节零位；False 直接禁用电机并断开。
RETURN_TO_ZERO_ON_STOP = True
# 发送零位后等待多久再禁用电机；不是到位检测。
RETURN_TO_ZERO_WAIT_S = 1.0


class Policy:
    def __init__(self, path):
        self.path = Path(path)
        self.model = torch.jit.load(str(path), map_location='cpu').eval()
        assert float(self.model.action_scale) in (0.02, 0.05) and float(self.model.policy_hz) == 20.0


class PolicyLoop:
    def __init__(self, policy):
        self.model = policy.model
        self.previous = self.model.init_pose.reshape(1,16).clone()
        self.history = None

    @torch.no_grad()
    def step(self, measured_q, policy_hz):
        q = torch.as_tensor(measured_q, dtype=torch.float32).reshape(1,16)
        if not torch.isfinite(q).all():
            raise ValueError('Nonfinite measured joint positions')
        if self.history is None:
            self.history = self.model.reset_history(self.previous, self.previous)
        previous = self.previous.clone()
        target, self.history, action = self.model(q, previous, self.history)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite student target')
        self.previous = target
        return dict(q=q.numpy()[0], previous_target=previous.numpy()[0], action=action.numpy()[0],
                    next_target=target.numpy()[0], policy_hz=float(policy_hz))


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--legacy-script', type=Path, default=Path(__file__).with_name('m24_legacy_deployment.py'))
    parser.add_argument('--manual-release', action='store_true', help='Diagnostic: R + Enter requests a bounded release phase; joint feedback is unchanged.')
    parser.add_argument('--explicit-open', action='store_true', help='Air diagnostic: R + Enter sends a bounded opening trajectory and holds; no automatic policy resume.')
    parser.add_argument('--release-pulse-steps', type=int, default=20)
    own, rest = parser.parse_known_args()
    if own.manual_release and own.explicit_open:
        raise ValueError('--manual-release and --explicit-open are mutually exclusive')
    spec = importlib.util.spec_from_file_location('m24_legacy', own.legacy_script)
    legacy = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    if not np.isfinite(RETURN_TO_ZERO_WAIT_S) or RETURN_TO_ZERO_WAIT_S < 0:
        raise ValueError('RETURN_TO_ZERO_WAIT_S must be finite and nonnegative')
    legacy.ZERO_TARGET_WAIT_S = float(RETURN_TO_ZERO_WAIT_S)
    original_stop = legacy.stop_after_interrupt_or_error

    def configured_stop(controller):
        if RETURN_TO_ZERO_ON_STOP:
            original_stop(controller)

    legacy.stop_after_interrupt_or_error = configured_stop
    legacy.DEFAULT_JIT_POLICY = Path(__file__).with_name('student.pt')
    legacy.DEFAULT_LOG_ROOT = Path(__file__).parent/'logs'
    args = legacy.parse_args(rest)
    if own.manual_release and not 1 <= own.release_pulse_steps <= 20:
        raise ValueError('--release-pulse-steps must be in 1..20')
    policy = Policy(args.jit_policy)
    pose = policy.model.init_pose.numpy().copy()
    legacy.M24_INIT_POSE = pose
    legacy.ACTIVE_POLICY_NAME = 'e003_proprio_only'
    legacy.ACTION_SCALE = np.float32(policy.model.action_scale)
    legacy.ACTIVE_POLICY = SimpleNamespace(training_action_scale=float(policy.model.action_scale), init_pose=pose)
    legacy.DEFAULT_JIT_POLICY = Path(args.jit_policy)
    legacy.CONTRACT_VERSION = 'e003_proprio_only_80_v1'
    legacy.ACTION_MASK_INDICES = tuple(range(4,12))
    legacy.JitPolicy = Policy
    legacy.M24PolicyLoop = PolicyLoop
    if own.manual_release:
        import json
        import select
        import sys
        from m24_e003_release_override import manual_loop_class
        if not args.dry_run and not sys.stdin.isatty():
            raise ValueError('Manual release requires an interactive terminal')
        def requested():
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip().lower() == 'r'
            return False
        manual_loop = manual_loop_class(PolicyLoop)
        legacy.M24PolicyLoop = lambda policy: manual_loop(policy, requested, own.release_pulse_steps)
        legacy.ACTIVE_POLICY_NAME = 'e003_manual_release_diagnostic'
        class ManualLogger(legacy.DeploymentLogger):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.metadata['manual_release'] = dict(enabled=True, pulse_steps=own.release_pulse_steps,
                    trigger='R + Enter', measured_joints_modified=False, autonomous=False)
                self._write_metadata()
            def write_step(self, step, record):
                super().write_step(step, record)
                event = record['manual_release']
                if event['accepted']:
                    print('[manual-release] step %d: release phase for %d steps' % (step, own.release_pulse_steps), flush=True)
                elif event['requested']:
                    print('[manual-release] existing pulse is active; request ignored', flush=True)
                if event['active'] and event['remaining_steps'] == 0:
                    print('[manual-release] pulse finished; learned phase resumes', flush=True)
                with (self.log_dir/'manual_release.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=step, **record['manual_release']))+'\n')
        legacy.DeploymentLogger = ManualLogger
        print('人工辅助诊断模式：控制开始后按 R 再回车，触发最多 1 秒的松开阶段；Ctrl+C 退出。', flush=True)
    if own.explicit_open:
        import json
        import select
        import sys
        from m24_explicit_open import explicit_open_loop_class, OPEN_DELTA, MAX_STEP
        if not args.dry_run and not sys.stdin.isatty():
            raise ValueError('Explicit opening requires an interactive terminal')
        def requested_open():
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.readline().strip().lower() == 'r'
            return False
        opening_loop = explicit_open_loop_class(PolicyLoop)
        legacy.M24PolicyLoop = lambda policy: opening_loop(policy, requested_open)
        legacy.ACTIVE_POLICY_NAME = 'e003_explicit_open_air_diagnostic'
        class OpeningLogger(legacy.DeploymentLogger):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.metadata['explicit_open'] = dict(enabled=True, trigger='R + Enter',
                    opening_joints=['index_1', 'thumb_2'], delta_rad=OPEN_DELTA,
                    max_step_rad=MAX_STEP, automatic_resume=False,
                    measured_joints_modified=False, contact_validated=False)
                self.last_mode = 'policy'
                self._write_metadata()
            def write_step(self, step, record):
                super().write_step(step, record)
                event = record['explicit_open']
                if event['accepted']:
                    print('[explicit-open] step %d: opening trajectory accepted; policy paused' % step, flush=True)
                if event['mode'] == 'hold' and self.last_mode != 'hold':
                    print('[explicit-open] command reached opening target; holding (physical arrival not confirmed)', flush=True)
                self.last_mode = event['mode']
                with (self.log_dir/'explicit_open.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(step=step, **event))+'\n')
        legacy.DeploymentLogger = OpeningLogger
        print('显式张开空载诊断：R + 回车后缓慢张开并保持，不自动恢复策略；运行上限到达后禁用电机，Ctrl+C 退出。', flush=True)
    return legacy.run(args)


if __name__ == '__main__':
    raise SystemExit(main())
