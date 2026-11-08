"""Default-off actuator randomization for E003 student-side execution.

The delivered proprio-only student infers task phase from the morphology of
its joint/target history. Real DexH13 position-mode execution is a smooth
trapezoidal response (identified 2026-09-17: per-joint first-order gains,
proximal ~0.22/step with 50 ms delay, distal near-instant), while training
only ever showed the sim PD (Kp2.4)+contact morphology, which derailed phase
inference on hardware.

This wrapper, used only inside student distillation, replaces the commanded
target seen by the low-level PD with a randomized first-order actuator model
(gain, optional one-step delay, optional velocity saturation) and multiplies
the per-env PD stiffness. Sampling is per joint, so asymmetric morphologies
like the identified real hand (fast distal, slow proximal) are covered, and a
configurable fraction of environments is anchored to the identified real
per-joint parameters so the real point is inside the training distribution.
The policy-observed previous target stays the unfiltered sent command,
matching deployment semantics. Rewards, gates, teacher and observations are
unchanged. It is never used for acceptance evaluation, which keeps the
original execution model.
"""
import math
import torch

NUM_JOINTS = 16

DEFAULT_SPEC = dict(
    seed=1234,
    identity_fraction=0.25,
    per_joint=False,
    real_anchor_fraction=0.0,
    real_params=None,
    gain_low=0.2,
    gain_high=1.0,
    delay_probability=0.5,
    vmax_probability=0.5,
    vmax_low=0.5,
    vmax_high=1.2,
    kp_mult_low=1.0,
    kp_mult_high=4.0,
    control_dt=0.05,
)


def resolve_spec(spec):
    merged = dict(DEFAULT_SPEC)
    merged.update(dict(spec or {}))
    assert 0.0 <= merged['identity_fraction'] <= 1.0
    assert 0.0 <= merged['real_anchor_fraction'] <= 1.0
    assert isinstance(merged['per_joint'], bool)
    assert 0.0 < merged['gain_low'] <= merged['gain_high'] <= 1.0
    assert 0.0 <= merged['delay_probability'] <= 1.0
    assert 0.0 <= merged['vmax_probability'] <= 1.0
    assert 0.0 < merged['vmax_low'] <= merged['vmax_high']
    assert 1.0 <= merged['kp_mult_low'] <= merged['kp_mult_high']
    assert merged['control_dt'] > 0
    if merged['real_anchor_fraction'] > 0:
        assert merged['real_params'], 'real_anchor_fraction requires real_params'
        for name, params in merged['real_params'].items():
            assert 0.0 < params['gain'] <= 1.0, name
    return merged


def _real_anchor_tensors(spec, device):
    """Expand spec['real_params'] (joint name -> params) to (16,) tensors.

    Joint names follow the Pasini layout i0..i3, m0..m3, r0..r3, t0..t3.
    """
    names = ([f'i{i}' for i in range(4)] + [f'm{i}' for i in range(4)] +
             [f'r{i}' for i in range(4)] + [f't{i}' for i in range(4)])
    gain = torch.ones(NUM_JOINTS)
    delay = torch.zeros(NUM_JOINTS, dtype=torch.bool)
    has_vmax = torch.zeros(NUM_JOINTS, dtype=torch.bool)
    vmax = torch.zeros(NUM_JOINTS)
    kp_mult = torch.ones(NUM_JOINTS)
    for name, params in spec['real_params'].items():
        idx = names.index(name)
        gain[idx] = float(params['gain'])
        delay[idx] = bool(params.get('delay_steps', 0)) or bool(params.get('delay', False))
        if params.get('vmax') is not None:
            has_vmax[idx] = True
            vmax[idx] = float(params['vmax'])
        if params.get('kp_mult') is not None:
            kp_mult[idx] = float(params['kp_mult'])
    return {key: value.to(device) for key, value in
            dict(gain=gain, delay=delay, has_vmax=has_vmax, vmax=vmax, kp_mult=kp_mult).items()}


def _sample(num, spec, generator, device):
    shape = (num, NUM_JOINTS) if spec['per_joint'] else (num, 1)

    def uniform(low, high):
        return low + (high - low) * torch.rand(shape, generator=generator)

    identity = (torch.rand(num, generator=generator) < spec['identity_fraction'])
    gain = uniform(spec['gain_low'], spec['gain_high']).expand(num, NUM_JOINTS).clone()
    delay = (torch.rand(shape, generator=generator) < spec['delay_probability']).expand(num, NUM_JOINTS).clone()
    has_vmax = (torch.rand(shape, generator=generator) < spec['vmax_probability']).expand(num, NUM_JOINTS).clone()
    vmax = uniform(spec['vmax_low'], spec['vmax_high']).expand(num, NUM_JOINTS).clone()
    log_low, log_high = math.log(spec['kp_mult_low']), math.log(spec['kp_mult_high'])
    kp_mult = torch.exp(log_low + (log_high - log_low) * torch.rand(shape, generator=generator))
    kp_mult = kp_mult.expand(num, NUM_JOINTS).clone()
    ident = identity.view(num, 1).expand(num, NUM_JOINTS)
    gain = torch.where(ident, torch.ones_like(gain), gain)
    delay = delay & ~ident
    has_vmax = has_vmax & ~ident
    kp_mult = torch.where(ident, torch.ones_like(kp_mult), kp_mult)
    if spec['real_anchor_fraction'] > 0:
        anchor = (torch.rand(num, generator=generator) < spec['real_anchor_fraction']) & ~identity
        mask = anchor.view(num, 1).expand(num, NUM_JOINTS)
        real = _real_anchor_tensors(spec, device)
        gain = torch.where(mask, real['gain'], gain)
        delay = torch.where(mask, real['delay'], delay)
        has_vmax = torch.where(mask, real['has_vmax'], has_vmax)
        vmax = torch.where(mask, real['vmax'].expand(num, NUM_JOINTS), vmax)
        kp_mult = torch.where(mask, real['kp_mult'], kp_mult)
    return {key: value.to(device) for key, value in
            dict(gain=gain, delay=delay, has_vmax=has_vmax, vmax=vmax, kp_mult=kp_mult).items()}


def instrument_actuator_randomization(base, spec=None):
    """Wrap an M24 task class with randomized student-side execution."""
    resolved = resolve_spec(spec)

    class ActuatorRandomized(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._ar_spec = resolved
            self._ar_generator = torch.Generator().manual_seed(int(resolved['seed']))
            self._ar_params = _sample(self.num_envs, resolved, self._ar_generator, self.device)
            self._ar_pgain_base = self.p_gain.clone()
            ndof = self.num_xhand_hand_dofs
            self._ar_eff = self.cur_targets[:, :ndof].clone()
            self._ar_prev_cmd = self.cur_targets[:, :ndof].clone()
            self._ar_apply_kp()

        def _ar_apply_kp(self):
            self.p_gain = self._ar_pgain_base * self._ar_params['kp_mult']

        def compute_observations(self):
            # XHandPasini appends cur_targets to its history. That buffer is
            # filtered for physical execution below, whereas hardware exposes
            # the unfiltered sent command. Swap only during observation assembly;
            # retain the exact filtered buffer for the next physics step/audit.
            execution_targets = self.cur_targets
            try:
                self.cur_targets = self.prev_targets
                return super().compute_observations()
            finally:
                self.cur_targets = execution_targets

        def pre_physics_step(self, actions):
            # Base copies prev_targets from the unfiltered command, matching the
            # deployment contract where the model observes the sent target.
            super().pre_physics_step(actions)
            ndof = self.num_xhand_hand_dofs
            cmd = self.cur_targets[:, :ndof].clone()
            delayed = torch.where(self._ar_params['delay'], self._ar_prev_cmd, cmd)
            delta = self._ar_params['gain'] * (delayed - self._ar_eff)
            bound = self._ar_params['vmax'] * self._ar_spec['control_dt']
            delta = torch.where(self._ar_params['has_vmax'],
                                torch.clamp(delta, -bound, bound), delta)
            self._ar_eff = self._ar_eff + delta
            self._ar_prev_cmd = cmd
            self.cur_targets[:, :ndof] = self._ar_eff

        def reset_idx(self, env_ids):
            super().reset_idx(env_ids)
            if env_ids is None:
                ids = torch.arange(self.num_envs, device=self.device)
            else:
                ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            if ids.numel() == 0:
                return
            fresh = _sample(ids.numel(), self._ar_spec, self._ar_generator, self.device)
            for key, value in fresh.items():
                self._ar_params[key][ids] = value
            self._ar_eff[ids] = self.cur_targets[ids, :self.num_xhand_hand_dofs]
            self._ar_prev_cmd[ids] = self.cur_targets[ids, :self.num_xhand_hand_dofs]
            self._ar_apply_kp()

    ActuatorRandomized._ar_spec_resolved = resolved
    return ActuatorRandomized
