"""Opt-in measured drive/release/return/regrasp curriculum.

Contact/release/rotation shaping follows the structure of Tac2Motion II-B.
Ordered phase events, opposing contact and global high-water accounting are
local adaptations. Normal moments are diagnostics, never transition inputs.
"""
import math
import torch


class E002Cycle:
    def __init__(self, n, device, cfg):
        self.target = float(cfg.get('stroke_angle', math.radians(20)))
        self.return_tolerance = float(cfg.get('return_sector_tolerance', math.radians(5)))
        self.dwell = int(cfg.get('event_dwell_steps', 2))
        self.goal_cycles = int(cfg.get('goal_cycles', 2))
        self.require_paired_drive_progress = bool(cfg.get('require_paired_drive_progress', True))
        self.release_potential = cfg.get('release_potential', 'saturating_absolute_force')
        # E003 local load objective, disabled for every E002 configuration.
        # Force reference normalizes measured loads; it is not a safety limit.
        self.normal_balance_scale = float(cfg.get('normal_balance_scale', 0.0))
        self.normal_moment_reference = float(cfg.get('normal_moment_reference', 0.05))
        assert self.normal_balance_scale >= 0 and self.normal_moment_reference > 0
        self.force_excess_scale = float(cfg.get('force_excess_scale', 0.0))
        self.force_limit = float(cfg.get('force_limit', 5.0))
        assert self.force_excess_scale >= 0 and self.force_limit > 0
        self.unload_balance_scale = float(cfg.get('unload_balance_scale', 0.0))
        self.unload_force_reference = float(cfg.get('unload_force_reference', 5.0))
        assert self.unload_balance_scale >= 0 and self.unload_force_reference > 0
        self.load_cost_scale = float(cfg.get('load_cost_scale', 0.0))
        self.load_force_reference = float(cfg.get('load_force_reference', 20.0))
        self.unpaired_load_multiplier = float(cfg.get('unpaired_load_multiplier', 1.0))
        assert self.load_cost_scale >= 0 and self.load_force_reference > 0
        assert self.unpaired_load_multiplier >= 1
        assert self.release_potential in ('saturating_absolute_force', 'initial_relative_force')
        self.deadlines = torch.tensor(cfg.get('phase_deadlines', [80, 40, 60, 40]), device=device)
        # Keep a resumed policy's learned clock while allowing more execution
        # time. This changes termination budgets, not physical phase criteria.
        self.phase_clock_scale = float(cfg.get('phase_clock_scale', 1.0))
        assert math.isfinite(self.phase_clock_scale) and self.phase_clock_scale > 0
        assert self.target > 0 and self.dwell >= 1 and self.goal_cycles >= 1
        assert self.deadlines.shape == (4,) and bool((self.deadlines > 0).all())
        for name in ('phase', 'phase_step', 'event_streak', 'cycles', 'drive_events', 'release_events', 'return_events'):
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=device))
        for name in ('previous', 'high', 'paid', 'reverse', 'gross', 'stroke_start',
                     'contact_progress', 'stroke_contact_start', 'potential_prev', 'return_error'):
            setattr(self, name, torch.zeros(n, device=device))
        self.reference = torch.zeros(n, 2, 3, device=device)
        self.reference_valid = torch.zeros(n, dtype=torch.bool, device=device)
        self.release_entry_force = torch.zeros(n, 2, device=device)
        self.last_event = torch.zeros(n, dtype=torch.long, device=device)
        self.last_reason = torch.zeros_like(self.last_event)

    def reset(self, ids):
        for value in self.__dict__.values():
            if torch.is_tensor(value) and value is not self.deadlines:
                value[ids] = 0

    def command(self):
        return torch.cat((torch.nn.functional.one_hot(self.phase, 4).float(),
            (self.phase_step.float() / (self.deadlines[self.phase] * self.phase_clock_scale)).clamp(0, 1)[:, None]), -1)

    def update(self, theta, contact, force, distance, radial, axis, quality, acquired,
               delta_ref, workspace_radius, hard_step_angle, force_off,
               normal_force_vectors=None, normal_moment_vectors=None, force_cost_signal=None):
        old = self.phase.clone()
        self.phase_step += 1
        pair = contact.all(-1)
        off = (~contact).all(-1) & (force < force_off).all(-1)
        projected = radial - (radial * axis[:, None]).sum(-1, keepdim=True) * axis[:, None]
        unit = projected / projected.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        capture = ~self.reference_valid & acquired & pair
        self.reference[capture] = unit[capture]
        self.reference_valid |= capture
        angle_error = torch.acos((unit * self.reference).sum(-1).clamp(-1, 1))
        self.return_error[:] = angle_error.max(-1).values
        delta = theta - self.previous
        self.previous[:] = theta
        self.high[:] = torch.maximum(self.high, theta)
        new = (self.high - self.paid).clamp(min=0)
        self.paid[:] = self.high  # Global across cycles; returns never reopen credit.
        self.reverse += (-delta).clamp(min=0)
        self.gross += delta.abs()
        valid_drive = (old == 0) & pair
        self.contact_progress += new * valid_drive.float()
        stroke = theta - self.stroke_start
        valid_stroke = self.contact_progress - self.stroke_contact_start
        drive_ready = (stroke >= self.target) & (valid_stroke >= self.target) & pair & acquired
        if not self.require_paired_drive_progress:
            # User-authorized final relaxed-gate diagnostic: real net angle
            # and current physical contact remain necessary.
            drive_ready = (stroke >= self.target) & contact.any(-1)
        returned = off & self.reference_valid & (self.return_error <= self.return_tolerance)
        conditions = torch.where(old == 0, drive_ready,
                     torch.where(old == 1, off, torch.where(old == 2, returned, pair)))
        self.event_streak[:] = torch.where(conditions, self.event_streak + 1, 0)
        event = self.event_streak >= self.dwell
        workspace = (distance > workspace_radius).any(-1)
        jump = delta.abs() > hard_step_angle
        event &= ~workspace & ~jump
        # Net progress must survive release and return before a cycle counts.
        cycle_event = event & (old == 3) & (stroke >= .5 * self.target)
        cycle_lost = event & (old == 3) & ~cycle_event
        self.last_event[:] = torch.where(event & ~cycle_lost, old + 1, 0)
        self.drive_events += (event & (old == 0)).long()
        self.release_events += (event & (old == 1)).long()
        self.return_events += (event & (old == 2)).long()
        self.cycles += cycle_event.long()
        entering_release = event & (old == 0)
        self.release_entry_force[entering_release] = force[entering_release]

        # Reuse the established scales: forward/reverse4, approach shaping0.5,
        # acquired-contact/event bonus1, completed-drive bonus10. No force tuning.
        norm_new = (new / delta_ref).clamp(0, 1)
        reward = 4 * quality * norm_new * (old == 0).float()
        reward -= 4 * ((-delta).clamp(min=0) / delta_ref).clamp(0, 1)
        reward -= 4 * (1 - quality) * norm_new
        pressure = (force / (force + 2 * force_off)).mean(-1)
        if self.release_potential == 'initial_relative_force':
            # Measured successful unloading loses10--30N before either finger
            # opens; absolute F/(F+0.3N) saturates there. Keep the same bounded
            # potential/weight but expose fractional unloading from phase entry.
            pressure = (force / self.release_entry_force.clamp(min=2 * force_off)).clamp(0, 1).mean(-1)
        return_potential = torch.exp(-self.return_error / self.return_tolerance)
        potential = torch.where(old == 1, 1 - pressure,
                    torch.where(old == 2, return_potential * off.float(), quality))
        reward += .5 * (potential - self.potential_prev) * (old != 0).float()
        reward += (event & ~cycle_lost).float() * torch.where(old == 0, 10., 1.)
        reward -= .25 * ((delta.abs() - math.radians(5)) / delta_ref).clamp(0, 1)
        if self.load_cost_scale > 0:
            # Unlike work alone, this charges static squeezing as well.
            # Only accidental one-finger DRIVE contact receives extra cost;
            # intentional release/return are never classified as contact loss.
            unpaired_drive = (old == 0) & (contact.sum(-1) == 1)
            multiplier = 1 + (self.unpaired_load_multiplier - 1) * unpaired_drive.float()
            reward -= self.load_cost_scale * (force / self.load_force_reference).square().sum(-1) * multiplier

        # Local phase-specific normal-load mismatch cost. This is not a full
        # wrench estimate and never cancels useful drive-axis torque.
        reward -= self.unload_balance_scale * (old == 1).float() * (force[:, 0] - force[:, 1]).abs() / self.unload_force_reference

        # Linear excess normal resultant (official legged_gym contact-load
        # cost adapted to a per-finger 5N reference). Never clips physics.
        limit_force = force if force_cost_signal is None else torch.maximum(force, force_cost_signal)
        reward -= self.force_excess_scale * ((limit_force - self.force_limit) / self.force_limit).clamp(min=0).sum(-1)

        if self.normal_balance_scale > 0:
            assert normal_force_vectors is not None and normal_moment_vectors is not None
            # Only measured normal components. Preserve the useful axial
            # moment; never interpret this as complete friction-wrench balance.
            net_f = normal_force_vectors.sum(1)
            net_m = normal_moment_vectors.sum(1)
            transverse_f = net_f - (net_f * axis).sum(-1, keepdim=True) * axis
            tilt_m = net_m - (net_m * axis).sum(-1, keepdim=True) * axis
            reward -= self.normal_balance_scale * (transverse_f.norm(dim=-1) / self.force_limit + tilt_m.norm(dim=-1) / self.normal_moment_reference)

        timeout = (self.phase_step >= self.deadlines[old]) & ~event
        complete = self.cycles >= self.goal_cycles
        reason = torch.zeros_like(old)
        reason = torch.where(timeout, 10 + old, reason)
        reason = torch.where(cycle_lost, 14, reason)
        reason = torch.where(workspace, 3, reason)
        reason = torch.where(jump, 4, reason)
        reason = torch.where(complete & ~workspace & ~jump, 5, reason)
        reward -= 2 * ((reason > 0) & (reason != 5)).float() + 8 * jump.float()
        # An event at an invalid physical endpoint cannot count toward a cycle.
        invalid = workspace | jump | cycle_lost
        self.cycles -= (cycle_event & invalid).long()
        self.last_event[invalid] = 0
        transition = event & ~invalid
        self.phase[transition] = (old[transition] + 1) % 4
        self.phase_step[transition] = 0
        self.event_streak[transition] = 0
        regrasp = transition & (old == 3)
        self.stroke_start[regrasp] = theta[regrasp]
        self.stroke_contact_start[regrasp] = self.contact_progress[regrasp]
        # Initialize potential in the NEW phase so switching labels cannot pay.
        next_potential = torch.where(self.phase == 1, 1 - pressure,
                         torch.where(self.phase == 2, return_potential * off.float(), quality))
        self.potential_prev[:] = next_potential
        self.last_reason[:] = reason
        return reward, reason
