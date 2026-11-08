#!/usr/bin/env python3
"""CPU contract tests for the default-off E003 actuator randomization wrapper.

Uses a dummy task base with the same buffers and call order as the real M24
task, so no IsaacGym import is required. Verifies: default-off identity,
parameter ranges and determinism, deployment semantics (the model-observed
previous target stays the unfiltered command), exact first-order/delay/
saturation arithmetic, and per-episode resampling with PD-multiplier
reapplication after resets.
"""
import math
import importlib.util
import sys
from pathlib import Path
import torch

_root = Path(__file__).resolve().parents[1]
# Load by path: importing dexscrew.tasks.* pulls in IsaacGym-dependent modules.
_spec = importlib.util.spec_from_file_location(
    'e003_actuator_randomization', _root/'dexscrew/tasks/e003_actuator_randomization.py')
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
DEFAULT_SPEC = _mod.DEFAULT_SPEC
resolve_spec = _mod.resolve_spec
instrument_actuator_randomization = _mod.instrument_actuator_randomization

N = 8
INIT = torch.zeros(N, 16)
INIT[:, 1] = 1.0


class DummyBase:
    num_xhand_hand_dofs = 16

    def __init__(self):
        self.num_envs = N
        self.device = 'cpu'
        self.cur_targets = torch.zeros(N, 17)
        self.cur_targets[:, :16] = INIT
        self.prev_targets = self.cur_targets.clone()
        self.p_gain = torch.full((N, 16), 2.4)
        self.reset_calls = 0
        self.observed_targets = None

    def pre_physics_step(self, actions):
        targets = self.prev_targets[:, :16] + 0.05 * actions
        self.cur_targets[:, :16] = targets
        # Real base copies prev_targets from the unfiltered command here.
        self.prev_targets[:] = self.cur_targets

    def compute_observations(self):
        # Match XHandPasini.compute_observations: the observation source is
        # cur_targets, not prev_targets. The wrapper must expose sent commands.
        self.observed_targets = self.cur_targets[:, :16].clone()

    def reset_idx(self, env_ids):
        self.reset_calls += 1
        self.cur_targets[env_ids, :16] = INIT[env_ids]
        # Real base may overwrite randomized gains at reset.
        self.p_gain[env_ids] = 2.4


def make(spec):
    cls = instrument_actuator_randomization(DummyBase, spec)
    return cls()


def test_spec_validation():
    assert resolve_spec(None) == DEFAULT_SPEC
    merged = resolve_spec({'gain_low': 0.3})
    assert merged['gain_low'] == 0.3 and merged['gain_high'] == DEFAULT_SPEC['gain_high']
    for bad in ({'gain_low': 0}, {'gain_low': 1.5}, {'kp_mult_high': 0.5},
                {'identity_fraction': 1.5}):
        try:
            resolve_spec(bad)
        except AssertionError:
            continue
        raise SystemExit(f'spec {bad} should be rejected')


def test_sampling_ranges_and_determinism():
    spec = resolve_spec({'seed': 7})
    _sample = _mod._sample
    a = _sample(4096, spec, torch.Generator().manual_seed(7), 'cpu')
    b = _sample(4096, spec, torch.Generator().manual_seed(7), 'cpu')
    for key in a:
        assert torch.equal(a[key], b[key]), key
    identity = torch.zeros(4096, dtype=torch.bool)
    # Re-derive identity mask from the sampler determinism: rerun with
    # identity_fraction=1 and everything else must be the identity point.
    ones_spec = resolve_spec({'seed': 7, 'identity_fraction': 1.0})
    ones = _sample(4096, ones_spec, torch.Generator().manual_seed(7), 'cpu')
    assert (ones['gain'] == 1).all() and not ones['delay'].any()
    assert not ones['has_vmax'].any() and (ones['kp_mult'] == 1).all()
    active = a['gain'] < 1
    assert (a['gain'][active] >= spec['gain_low'] - 1e-6).all()
    assert (a['gain'][active] <= spec['gain_high'] + 1e-6).all()
    assert (a['vmax'] >= spec['vmax_low'] - 1e-6).all() and (a['vmax'] <= spec['vmax_high'] + 1e-6).all()
    assert (a['kp_mult'] >= spec['kp_mult_low'] - 1e-6).all() and (a['kp_mult'] <= spec['kp_mult_high'] + 1e-6).all()
    assert active.float().mean() > 0.4 and active.float().mean() < 0.9


def test_identity_spec_is_bitwise_noop():
    env = make({'seed': 1, 'identity_fraction': 1.0})
    plain = DummyBase()
    torch.manual_seed(3)
    for _ in range(10):
        actions = torch.rand(N, 16) * 2 - 1
        env.pre_physics_step(actions.clone())
        plain.pre_physics_step(actions.clone())
    assert torch.equal(env.cur_targets, plain.cur_targets)
    assert torch.equal(env.p_gain, plain.p_gain)


def test_observation_uses_sent_command_without_changing_execution():
    env = make(dict(identity_fraction=0.0, gain_low=0.2, gain_high=0.2,
                    delay_probability=1.0, vmax_probability=0.0))
    for _ in range(5):
        env.pre_physics_step(torch.ones(N, 16))
        filtered = env.cur_targets.clone()
        sent = env.prev_targets.clone()
        assert not torch.equal(filtered[:, :16], sent[:, :16])
        env.compute_observations()
        torch.testing.assert_close(env.observed_targets, sent[:, :16], rtol=0, atol=0)
        torch.testing.assert_close(env.cur_targets, filtered, rtol=0, atol=0)
        torch.testing.assert_close(env.prev_targets, sent, rtol=0, atol=0)


def test_filter_math_and_deployment_semantics():
    spec = dict(seed=11, identity_fraction=0.0, gain_low=0.4, gain_high=0.4,
                delay_probability=0.0, vmax_probability=0.0, kp_mult_low=2.0, kp_mult_high=2.0)
    env = make(spec)
    assert torch.allclose(env.p_gain, torch.full((N, 16), 4.8))
    eff = INIT.clone()
    prev_cmd = INIT.clone()
    torch.manual_seed(5)
    for k in range(6):
        actions = torch.rand(N, 16) * 2 - 1
        env.pre_physics_step(actions)
        cmd = prev_cmd + 0.05 * actions  # unfiltered sent command
        # The model-visible previous target is the sent command, not the
        # filtered actuator input.
        assert torch.allclose(env.prev_targets[:, :16], cmd, atol=1e-6), k
        eff = eff + 0.4 * (cmd - eff)
        assert torch.allclose(env.cur_targets[:, :16], eff, atol=1e-6), k
        prev_cmd = cmd


def test_delay_and_saturation():
    spec = dict(seed=13, identity_fraction=0.0, gain_low=1.0, gain_high=1.0,
                delay_probability=1.0, vmax_probability=1.0,
                vmax_low=0.5, vmax_high=0.5, control_dt=0.05)
    env = make(spec)
    assert env._ar_params['delay'].all() and env._ar_params['has_vmax'].all()
    eff = INIT.clone()
    prev_cmd = INIT.clone()
    torch.manual_seed(9)
    for k in range(5):
        actions = (torch.rand(N, 16) * 2 - 1) * 100  # huge jumps saturate
        env.pre_physics_step(actions)
        cmd = prev_cmd + 0.05 * actions
        # With delay the actuator input is the previous step's command
        # (the initial pose at k=0), then velocity-saturated.
        delta = torch.clamp(prev_cmd - eff, -0.5 * 0.05, 0.5 * 0.05)
        eff = eff + delta
        assert torch.allclose(env.cur_targets[:, :16], eff, atol=1e-6), k
        assert (env.cur_targets[:, :16] - INIT).abs().max() <= 5 * 0.5 * 0.05 + 1e-6
        prev_cmd = cmd


def test_per_joint_sampling_and_real_anchor():
    real = {'i0': {'gain': 0.228, 'delay': True, 'vmax': 0.65, 'kp_mult': 4.0},
            'i1': {'gain': 0.220, 'delay': True, 'vmax': 0.5, 'kp_mult': 4.0},
            'i2': {'gain': 0.939, 'delay': True, 'vmax': 1.2, 'kp_mult': 4.0}}
    env = make(dict(seed=23, identity_fraction=0.0, per_joint=True,
                    real_anchor_fraction=1.0, real_params=real))
    gain = env._ar_params['gain']
    assert gain.shape == (N, 16)
    # All envs anchored: specified joints carry the identified values exactly.
    assert torch.allclose(gain[:, 0], torch.full((N,), 0.228))
    assert torch.allclose(gain[:, 1], torch.full((N,), 0.220))
    assert torch.allclose(gain[:, 2], torch.full((N,), 0.939))
    assert env._ar_params['delay'][:, 0].all() and not env._ar_params['delay'][:, 3].any()
    assert env._ar_params['has_vmax'][:, 0].all() and not env._ar_params['has_vmax'][:, 4].any()
    assert torch.allclose(env.p_gain[:, 0], torch.full((N,), 2.4*4.0))
    assert torch.allclose(env.p_gain[:, 3], torch.full((N,), 2.4))
    # Random per-joint spread without anchor: joints differ within one env.
    env2 = make(dict(seed=29, identity_fraction=0.0, per_joint=True,
                     real_anchor_fraction=0.0))
    g2 = env2._ar_params['gain']
    assert (g2.std(dim=1) > 0).float().mean() > 0.5, 'per-joint gains should vary'


def test_reset_resamples_and_reapplies_multiplier():
    env = make({'seed': 17})
    gains_before = env._ar_params['gain'].clone()
    env.pre_physics_step(torch.zeros(N, 16))
    ids = torch.tensor([0, 3, 5])
    env.p_gain[ids] = 2.4  # base reset overwrites gains
    env.reset_idx(ids)
    assert env.reset_calls == 1
    assert torch.allclose(env.cur_targets[ids, :16], INIT[ids])
    assert torch.allclose(env._ar_eff[ids], INIT[ids])
    expected = 2.4 * env._ar_params['kp_mult']
    assert torch.allclose(env.p_gain, expected)
    changed = (gains_before[ids] - env._ar_params['gain'][ids]).abs() > 0
    assert bool(changed.any()), 'reset should resample parameters'


if __name__ == '__main__':
    test_spec_validation()
    test_sampling_ranges_and_determinism()
    test_identity_spec_is_bitwise_noop()
    test_observation_uses_sent_command_without_changing_execution()
    test_filter_math_and_deployment_semantics()
    test_delay_and_saturation()
    test_per_joint_sampling_and_real_anchor()
    test_reset_resamples_and_reapplies_multiplier()
    print('all actuator randomization contract tests passed')
