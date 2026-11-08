# --------------------------------------------------------
# Dexh13 Hora Task
# --------------------------------------------------------

from collections import OrderedDict

from .xhand_hora import XHandHora


class Dexh13Hora(XHandHora):
    """DexH13 hand family wired into the Hora screw-like task pipeline.

    This keeps the shared reward / observation / object logic from XHandHora,
    while swapping in DexH13-specific hand defaults.
    """

    def __init__(self, config, sim_device, graphics_device_id, headless):
        super().__init__(config, sim_device, graphics_device_id, headless)

    def _default_apply_action_mask(self):
        return False

    def _default_fingertip_body_names(self):
        return [
            "right_index_link_3",
            "right_middle_link_3",
            "right_ring_link_3",
            "right_thumb_link_3",
        ]

    def _default_hand_dof_props(self):
        lower = [
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
        ]
        upper = [
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
        ]
        effort = [1.0 for _ in lower]
        velocity = [10.0 for _ in lower]
        return lower, upper, effort, velocity

    def _default_joint_values(self, dof_names):
        if self.config["env"]["initPose"] == "nutbolt_inclined":
            values = [
                -0.17,
                1.40,
                0.0,
                0.4,
                0.0,
                1.4,
                0.0,
                0.4,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.4,
                0.5,
                0.45,
            ]
        elif self.config["env"]["initPose"] in ("bulb_inclined", "lightbulb_inclined"):
            values = [
                0.31805448815226555,
                0.7768147802352905,
                0.2902536576986313,
                1.0633865308761597,
                0.03999972581863403,
                0.5151662063598633,
                0.7383003902435303,
                0.7970210313796997,
                -0.17020170390605927,
                0.8792879581451416,
                0.1139976978302002,
                1.4113634824752808,
                -0.3527474871277809,
                1.57,
                0.07279872894287,
                0.4473716020584106,
            ]
        elif self.config["env"]["initPose"] == "screwdriver_inclined":
            values = [
                0.34266707360744476,
                1.2325279521942139,
                0.28289711773395538,
                1.1292990016937256,
                0.00888176321983337,
                0.7502785110473633,
                0.8105450248718262,
                0.825642421245575,
                -0.3587684601545334,
                1.3127488565444946,
                0.0030379545688629,
                1.5381801319122314,
                -0.35994077682495117,
                1.5656383037567139,
                0.5520286703109741,
                0.531196631193161,
            ]
        else:
            raise ValueError(
                f"Unsupported initPose: {self.config['env']['initPose']} for Dexh13Hora"
            )
        if len(dof_names) != len(values):
            raise ValueError(
                f"Dexh13Hora init pose expects {len(values)} DOFs, got {len(dof_names)}"
            )
        return OrderedDict(zip(dof_names, values))

    def _create_object_asset(self):
        # Keep the shared object loader, but this override marks DexH13's
        # fingertip-body contract as task-specific.
        return super()._create_object_asset()

    def _parse_hand_dof_props(self):
        # DexH13 uses different joint limits/velocity defaults from XHand.
        return super()._parse_hand_dof_props()
