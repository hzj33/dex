#!/usr/bin/env python3
"""Bounded E003 PAdapt with atomic checkpoints and frozen-teacher receipts.

The existing PAdapt update/loss and optimizer are unchanged. SIGTERM stops at
a completed update; wall-clock timeout is owned by the external driver.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train as entry  # IsaacGym must precede torch.
import torch

STOP = False
TEACHER_SHA = 'd42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'


def now():
    return datetime.now(timezone.utc).isoformat()


def request_stop(signum, frame):
    global STOP
    STOP = True


class BoundedE003Student(entry.ProprioAdapt):
    def __init__(self, env, output_dir, full_config, **kwargs):
        super().__init__(env, output_dir, full_config, **kwargs)
        assert self.obs_shape == (101,) and self.actions_num == 16
        assert env.e002_cycle.goal_cycles == 5
        assert float(full_config.task.env.controller.pgain) == 2.4
        assert tuple(self.student_trainable_param_patterns) == ('adapt_tconv',)
        assert self.model.teacher_point_tanh
        self.snapshot_time = time.monotonic()
        self.started = now()
        self.updates = 0
        self.frozen = {}

    def restore_train(self, fn):
        assert hashlib.sha256(Path(fn).read_bytes()).hexdigest() == TEACHER_SHA
        super().restore_train(fn)
        checkpoint = torch.load(fn, map_location='cpu')
        actual = self.model.state_dict()
        missing = set(actual) - set(checkpoint['model'])
        assert missing and all(k.startswith('adapt_tconv.') for k in missing), missing
        assert not (set(checkpoint['model']) - set(actual))
        for key, value in checkpoint['model'].items():
            assert torch.equal(actual[key].cpu(), value), key
        self.frozen = {k: v.cpu().clone() for k, v in actual.items()
                       if not k.startswith('adapt_tconv.')}
        info = dict(status='verified:frozen_teacher_loaded',teacher_sha256=TEACHER_SHA,
                    missing_keys=sorted(missing),obs_dim=101,proprio_shape=[30,32],
                    phase_interface='4 phase one-hot values plus elapsed scalar; external command required',
                    hardware_ready=False,trainable_patterns=['adapt_tconv'],
                    teacher_point_latent='tanh, matching actual PPO actor inputs')
        Path(self.output_dir,'teacher_binding.json').write_text(json.dumps(info,indent=2)+'\n')

    def save(self, name):
        # A single rename prevents interrupted readers from seeing a partial ckpt.
        super().save(str(name)+'.pending')
        os.replace(str(name)+'.pending.ckpt', str(name)+'.ckpt')

    def snapshot(self, stopping=False):
        for key, value in self.frozen.items():
            assert torch.equal(self.model.state_dict()[key].detach().cpu(),value), key
        self.save(str(Path(self.nn_dir,'model_last')))
        metrics={k:float(v) for k,v in self.direct_info.items()
                 if k in ('latent_loss','bc_loss','total_loss','done_rate')}
        meta=dict(started_at=self.started,completed_at=now(),updates=self.updates,
                  agent_steps=self.agent_steps,frozen_teacher_unchanged=True,
                  stopped_by_timeout=stopping,metrics=metrics)
        p=Path(self.output_dir,'progress.json');p.with_suffix('.tmp').write_text(json.dumps(meta,indent=2)+'\n')
        os.replace(p.with_suffix('.tmp'),p)
        self.writer.flush()
        self.snapshot_time=time.monotonic()
        print('[E003 student snapshot] '+json.dumps(meta),flush=True)

    def log_tensorboard(self):
        super().log_tensorboard()
        self.updates += 1
        if STOP or self.updates == 1 or time.monotonic()-self.snapshot_time >= 60:
            self.snapshot(STOP)
        if STOP:
            self.writer.close()
            raise SystemExit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM,request_stop)
    entry.ProprioAdapt=BoundedE003Student
    main=entry.hydra.main(config_name='config',config_path=str(Path(__file__).resolve().parents[1]/'configs'))(entry.main.__wrapped__)
    main()
