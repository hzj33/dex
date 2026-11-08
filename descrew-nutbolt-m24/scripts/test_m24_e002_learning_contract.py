#!/usr/bin/env python3
"""Exercise the actual task reward on CPU tensors and offline false-pass cases."""
import ast
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

from m24_e002_a0c_diagnostics import summarize
from m24_e002_evaluation_contract import assess_learning
from test_m24_e002_a0c_diagnostics import candidate, T


def task_fixture(mode, nut_contact=True):
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'dexscrew/tasks/xhand_pasini.py').read_text())
    names = ('_setup_e002_motion_config', '_compute_e002_a_reward')
    body = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name in names]
    scope = dict(torch=torch, np=np)
    exec(compile(ast.Module(body=body, type_ignores=[]), 'actual_e002_methods', 'exec'), scope)
    env = SimpleNamespace(config={'sim': {'use_gpu_pipeline': False}}, device='cpu')
    cfg = yaml.safe_load((root / 'configs/task/XHandPasiniM24NutBoltE002.yaml').read_text())['env']['e002_motion']
    cfg['acceptance_mode'] = mode
    scope['_setup_e002_motion_config'](env, cfg)
    env.fingertip_handles = [0, 1, 2, 3]
    env.rigid_body_states = torch.zeros(1, 4, 13)
    env.rigid_body_states[:, 0, 0] = .04
    env.rigid_body_states[:, 3, 0] = -.04
    env.contact_forces = torch.ones(1, 4, 3)
    env.e002_normal_contact_force = torch.ones(1, 2, 3) * float(nut_contact)
    env.e002_normal_contact_moment = torch.tensor([[[0., 0., .1], [0., 0., -.1]]])
    env.rot_axis_buf = torch.tensor([[0., 0., -1.]])
    env.e002_contact_state = torch.zeros(1, 2, dtype=torch.bool)
    for name in ('theta_start', 'theta_prev', 'paid_high_water', 'high_water', 'gross_travel',
                 'reverse_travel', 'valid_pair_progress', 'normal_proxy_progress', 'contact_potential_prev'):
        setattr(env, 'e002_' + name, torch.zeros(1))
    for name in ('pair_contact_streak', 'pair_contact_loss_streak', 'drive_step', 'target_hold', 'termination_reason'):
        setattr(env, 'e002_' + name, torch.zeros(1, dtype=torch.long))
    for name in ('contact_acquired', 'target_validated', 'success'):
        setattr(env, 'e002_' + name, torch.zeros(1, dtype=torch.bool))
    env.e002_milestone_paid = torch.zeros(1, 2, dtype=torch.bool)
    env.nut_dof_pos = torch.zeros(1, 1)
    env.progress_buf = torch.zeros(1, dtype=torch.long)
    env.reset_buf = torch.zeros(1, dtype=torch.long)
    env.rew_buf = torch.zeros(1)
    env.num_actions = 16
    env.prev_hand_actions = torch.zeros(1, 16)
    env.extras = {}
    def step(degrees):
        env.nut_dof_pos[:] = math.radians(degrees)
        env.progress_buf += 1
        scope['_compute_e002_a_reward'](env, torch.zeros(1, 3), torch.zeros(1, 1), torch.zeros(1, 16))
    return env, step


def main():
    for mode, expected in [('contact_progress', True), ('strict_normal_proxy', False)]:
        env, step = task_fixture(mode)
        for angle in [0, 0, 0, 5, 10, 15, 20, 25, 30, 30, 30, 30]:
            step(angle)
        assert bool(env.e002_success.item()) == expected
        assert env.e002_normal_proxy_progress.item() == 0
        assert bool(env.e002_milestone_paid.any()) == expected
    env, step = task_fixture('contact_progress', nut_contact=False)
    for angle in [0, 0, 0, 5, 10, 15, 20, 25, 30, 30, 30, 30]:
        step(angle)
    assert not env.e002_success.item() and env.e002_valid_pair_progress.item() == 0
    assert env.extras['e002/reward/forward'] == 0  # Bolt force cannot stand in for nut force.
    env, step = task_fixture('contact_progress')
    for angle in [0, 0, 0, 5, 0, 5]:
        step(angle)
    assert math.isclose(env.e002_valid_pair_progress.item(), math.radians(5), rel_tol=1e-6)
    assert env.extras['e002/reward/forward'] == 0  # Revisiting a high is unpaid.

    good = candidate(0, [-.1, .1])  # Normal moments oppose; contact rotation is valid.
    for row in good['steps']:
        row['normal_contact_force_on_nut'] = [[1., 0., 0.], [-1., 0., 0.]]
    assert summarize([copy.deepcopy(good)], T)['status'] != 'passed'
    assert summarize([copy.deepcopy(good)], T, acceptance_mode='contact_progress')['status'] == 'passed'
    for alteration in ('bolt_only', 'uncontacted_motion', 'reset', 'nonfinite'):
        bad = copy.deepcopy(good)
        for row in bad['steps']:
            if alteration == 'bolt_only' or (alteration == 'uncontacted_motion' and row['phase'] == 'probe'):
                row['normal_contact_force_on_nut'][1] = [0., 0., 0.]
        if alteration == 'reset':
            bad['steps'][0]['unexpected_reset'] = True
        if alteration == 'nonfinite':
            bad['steps'][0]['finite'] = False
        assert summarize([bad], T, acceptance_mode='contact_progress')['status'] != 'passed'

    row = dict(env_id=0, step=0, net_angle=0., gross_angle=0., reverse_angle=0.,
               force=[1.,1.], distance=[.04,.04], action=[0.] * 16, contact=[True,True], reason=2)
    assessed = assess_learning([row], [row], 1, 100)
    assert assessed['continuation_allowed'] and not assessed['goal_completed']
    for change in ({'net_angle': float('nan')}, {'step': 2}, {'reason': 4}):
        bad = dict(row, **change)
        assert not assess_learning([bad], [bad], 1, 100)['continuation_allowed']
    assert not assess_learning([None], [], 1, 100)['continuation_allowed']
    print('Actual reward, legacy contrast, bolt exclusion, oscillation, A0c and evaluation contracts: OK')


if __name__ == '__main__':
    main()
