#!/usr/bin/env python3
"""Frozen-teacher feasibility with deployment SDK position bounds.

Default 0.02 uniformly slows increments and requires explicit 2.5x phase
budgets. Optional teacher_action_scale=0.05 retains original timing to
isolate the effect of SDK position bounds. This is a diagnostic, not PPO.
"""
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import evaluate_m24_e002_cycle as evaluator
from m24_e003_deploy_runtime import install_deployment_limits
from calibrate_m24_e002_a0 import torch

OriginalPPO = evaluator.PPO


class RetimedTeacher(OriginalPPO):
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        scale = float(kwargs['full_config'].get('teacher_action_scale', 0.02))
        assert scale in (0.02, 0.05)
        assert env.e002_cycle.deadlines.tolist() == [int(x*0.05/scale) for x in (160,80,120,80)]
        assert env.e002_cycle.phase_clock_scale == 0.5
        install_deployment_limits(env, SimpleNamespace(action_scale=scale,
            lower=torch.tensor([-math.pi/9,0.,0.,0.]*4),
            upper=torch.tensor([math.pi/9,1.57,1.57,1.57]*4)))


if __name__ == '__main__':
    evaluator.PPO = RetimedTeacher
    evaluator.hydra.main(config_name='config',config_path=str(Path(__file__).resolve().parents[1]/'configs'))(evaluator.main.__wrapped__)()
    out=Path(next(x.split('=',1)[1] for x in sys.argv if x.startswith('+cycle_eval_out=')))
    d=json.loads(out.read_text())
    scale=float(next((x.split('=',1)[1] for x in sys.argv if x.startswith('+teacher_action_scale=')),0.02))
    d['control_diagnostic']=dict(acceptance_eligible=False,frozen_teacher=True,
        action_scale=scale,uniform_time_scale=0.05/scale,action_rescaling=False,
        purpose=('Check complete demonstration coverage at legacy hardware slew limit'
                 if scale == 0.02 else 'Isolate SDK position bounds at original teacher timing'))
    out.write_text(json.dumps(d,indent=2)+'\n')
