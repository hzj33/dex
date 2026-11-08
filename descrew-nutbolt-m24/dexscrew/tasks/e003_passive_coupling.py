"""Diagnostic index DIP coupling shared by physical eval and bounded adaptation."""
from isaacgym import gymtorch


def instrument_passive_coupling(base, coefficients, stiffness, damping):
    assert stiffness >= 0 and damping >= 0
    class Coupled(base):
        def update_low_level_control(self, step_id):
            super().update_low_level_control(step_id)
            if stiffness == 0:
                return
            assert self.torque_control
            q, v = self.xhand_hand_dof_pos, self.xhand_hand_dof_vel
            x = q[:, 2]
            f = x.new_zeros(x.shape)
            df = x.new_zeros(x.shape)
            for c in coefficients:
                df = df*x + f
                f = f*x + c
            reaction = -stiffness*(q[:, 3]-f)-damping*(v[:, 3]-df*v[:, 2])
            torque = self.torques[:, step_id, :].clone()
            torque[:, 3] = reaction
            torque[:, 2] -= df*reaction
            self.torques[:, step_id, :] = torque
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(torque))
    return Coupled
