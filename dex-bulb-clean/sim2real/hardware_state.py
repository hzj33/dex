from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class ArmState:
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    eef_pose: np.ndarray
    control: np.ndarray
    connected: bool

    def as_frame_fields(self) -> Dict[str, object]:
        return {
            "arm_joint_positions": self.joint_positions.astype(np.float32),
            "arm_joint_velocities": self.joint_velocities.astype(np.float32),
            "arm_eef_pose": self.eef_pose.astype(np.float32),
            "arm_control": self.control.astype(np.float32),
            "arm_side": "unknown",
            "arm_connected": bool(self.connected),
        }


class NullArmStateReader:
    """Read-only placeholder for future robot-arm integration."""

    def __init__(self, num_joints: int = 6):
        self.num_joints = int(num_joints)

    def read(self) -> Dict[str, object]:
        state = ArmState(
            joint_positions=np.full(self.num_joints, np.nan, dtype=np.float32),
            joint_velocities=np.full(self.num_joints, np.nan, dtype=np.float32),
            eef_pose=np.full(7, np.nan, dtype=np.float32),
            control=np.full(self.num_joints, np.nan, dtype=np.float32),
            connected=False,
        )
        return state.as_frame_fields()

    def close(self) -> None:
        pass


@dataclass
class FixedWristStateReader:
    """Fixed wrist state matching the current hand-only deployment setup."""

    angle_dim: int = 1

    def read(self) -> Dict[str, object]:
        return {
            "wrist_pose": np.full(7, np.nan, dtype=np.float32),
            "wrist_angle": np.zeros(self.angle_dim, dtype=np.float32),
            "wrist_control": np.zeros(self.angle_dim, dtype=np.float32),
            "wrist_locked": True,
            "wrist_connected": False,
        }

    def close(self) -> None:
        pass
