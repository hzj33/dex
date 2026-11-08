#!/usr/bin/env python3
"""Independent geometry checks for offline contact wrench reconstruction."""

import numpy as np

from analyze_m24_e002_contacts import normal_wrench


def vec(values):
    return dict(zip(("x", "y", "z"), values))


def test_rotated_contact_and_reversed_body_order():
    # Nut rotated +90 degrees about z. Local +x contact -> world +y.
    # World +x force on nut creates -z torque of magnitude 0.04 Nm.
    states = np.zeros((3, 13))
    states[:, 6] = 1
    states[2, :3] = [10, 20, 30]  # Translation must not affect moment about nut.
    states[2, 3:7] = [0, 0, 2 ** -0.5, 2 ** -0.5]
    for reverse in (False, True):
        contact = dict(body0=0 if reverse else 2, body1=2 if reverse else 0,
                       normal=vec([-1, 0, 0] if reverse else [1, 0, 0]), **{"lambda": 2.0})
        contact["localPos1" if reverse else "localPos0"] = vec([.02, 0, 0])
        sample = dict(rigid_body_state=states.tolist(), contacts=[contact], rotation_axis=[0, 0, -1],
                      all_body_contact_force=[[-2, 0, 0], [0, 0, 0], [2, 0, 0]])
        result = normal_wrench(sample, [0, 1], 2)
        np.testing.assert_allclose(result["normal_axis_torque_on_nut"], [.04, 0], atol=1e-12)
        np.testing.assert_allclose(result["active_contact_points"][0]["point_world"], [10, 20.02, 30])
        assert result["normal_tensor_max_abs_residual"] == 0


if __name__ == "__main__":
    test_rotated_contact_and_reversed_body_order()
    print("contact wrench geometry checks passed")
