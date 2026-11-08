from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation


MINK_SRC = "/home/vision123/software/remote/Tracker_teleoperate_rm65b/mink/src"
RM_API_SRC = "/home/vision123/software/remote/Tracker_teleoperate_rm65b/RM_API2/Python"
RM65B_XML = "/home/vision123/software/remote/Tracker_teleoperate_rm65b/rm_65_b_description/urdf/scene.xml"

sys.path.insert(0, MINK_SRC)

quest2ur = np.array(
    [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
    dtype=np.float64,
)
ur2quest = np.linalg.inv(quest2ur)


def _quest_pose_ready(pose_data: Dict, pose_key: str) -> bool:
    if pose_key not in pose_data:
        return False
    pos = pose_data[pose_key][:3, 3]
    return bool(np.linalg.norm(pos - np.array([0.0, 0.0, 0.2])) >= 0.05)


def _apply_transfer(mat: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    if len(xyz) == 3:
        xyz = np.append(xyz, 1.0)
    return np.matmul(mat, xyz)[:3]


class RM65BStateClient:
    def __init__(self, robot_ip: str = "192.168.2.19", port: int = 8080):
        sys.path.insert(0, RM_API_SRC)
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e

        self.arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        self.handle = self.arm.rm_create_robot_arm(robot_ip, port, level=3)
        print(f"[RM65BStateClient] Connected to {robot_ip}:{port}, handle={self.handle}")

    def read_joints_deg(self) -> np.ndarray:
        _, state = self.arm.rm_get_current_arm_state()
        return np.asarray(state["joint"], dtype=np.float64)[:6]

    def read_pose6(self) -> np.ndarray:
        _, state = self.arm.rm_get_current_arm_state()
        return np.asarray(state["pose"], dtype=np.float64)[:6]

    def command_joints_deg(self, q_deg: np.ndarray) -> None:
        self.arm.rm_movej_canfd(np.asarray(q_deg, dtype=np.float64)[:6].tolist(), follow=False)

    def close(self) -> None:
        try:
            self.arm.rm_delete_robot_arm()
        except Exception:
            pass


@dataclass
class QuestRM65BConfig:
    quest_ip: str = "192.168.12.38"
    quest_hand: str = "l"
    arm_side: str = "right"
    robot_ip: str = "192.168.2.19"
    robot_port: int = 8080
    connect_robot: bool = True
    command_robot: bool = False
    init_joints_deg: tuple[float, ...] = (0.0, 0.0, 120.0, 0.0, -30.0, 0.0)
    max_joint_vel_rad: float = 15.0
    ema_alpha: float = 0.4
    micro_step: float = 0.001
    micro_fast_step: float = 0.004


class QuestRM65BArmReader:
    """Quest-to-RM65B target generator with optional real RM65B state read.

    This mirrors the RM65B branch from skill-teleop-main's Quest agent, but keeps
    robot command disabled unless explicitly requested.
    """

    def __init__(self, config: QuestRM65BConfig):
        if config.quest_hand not in ("l", "r"):
            raise ValueError("quest_hand must be 'l' or 'r'")

        import mujoco
        import mink
        from oculus_reader.reader import OculusReader

        self.config = config
        self.reader = OculusReader(ip_address=config.quest_ip)
        self.robot: Optional[RM65BStateClient] = (
            RM65BStateClient(config.robot_ip, config.robot_port)
            if config.connect_robot
            else None
        )

        model = mujoco.MjModel.from_xml_path(RM65B_XML)
        self._mink = mink
        self._cfg = mink.Configuration(model)
        self._fk = mink.Configuration(model)
        self._task = mink.FrameTask(
            "arm_tool_site",
            "site",
            position_cost=1.0,
            orientation_cost=0.15,
        )
        self._limits = [
            mink.ConfigurationLimit(model),
            mink.VelocityLimit(
                model,
                {f"joint{i}": float(config.max_joint_vel_rad) for i in range(1, 7)},
            ),
        ]

        self.control_active = False
        self.micro_active = False
        self._trigger_latch_pressed = False
        self._prev_quest_pose: Optional[np.ndarray] = None
        self._target_pos: Optional[np.ndarray] = None
        self._target_rot: Optional[np.ndarray] = None
        self._ema_pos: Optional[np.ndarray] = None
        self._ema_rotvec: Optional[np.ndarray] = None
        self._last_cmd_deg = np.asarray(config.init_joints_deg, dtype=np.float64)[:6]
        self._pending_cmd_deg: Optional[np.ndarray] = None
        self._last_status = "init"
        print(
            "[QuestRM65BArmReader] Ready "
            f"quest_ip={config.quest_ip}, hand={config.quest_hand}, arm_side={config.arm_side}, "
            f"connect_robot={config.connect_robot}, command_robot={config.command_robot}"
        )

    def _read_measured_joints_deg(self) -> np.ndarray:
        if self.robot is None:
            return self._last_cmd_deg.copy()
        try:
            q_deg = self.robot.read_joints_deg()
            if q_deg.shape[0] >= 6:
                return q_deg[:6]
        except Exception as exc:
            self._last_status = f"robot_read_error:{exc}"
        return self._last_cmd_deg.copy()

    def _fk_pose(self, q_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_rad = np.deg2rad(q_deg)
        self._fk.update(q_rad)
        transform = self._fk.get_transform("arm_tool_site", "site", "world", "body")
        return np.asarray(transform.translation()), transform.rotation().as_matrix()

    def _solve_target(self, q_seed_deg: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
        self._cfg.update(np.deg2rad(q_seed_deg))
        target_T = self._mink.SE3.from_rotation_and_translation(
            self._mink.SO3.from_matrix(target_rot),
            target_pos,
        )
        self._task.set_target(target_T)

        ik_dt = 0.01
        err = np.zeros(6, dtype=np.float64)
        for _ in range(200):
            vel = self._mink.solve_ik(
                self._cfg,
                [self._task],
                dt=ik_dt,
                solver="quadprog",
                limits=self._limits,
            )
            self._cfg.integrate_inplace(vel, ik_dt)
            err = self._task.compute_error(self._cfg)
            if np.linalg.norm(err[:3]) < 1e-3 and np.linalg.norm(err[3:]) < 1e-2:
                break

        pos_residual = float(np.linalg.norm(err[:3]))
        if pos_residual > 0.30:
            self._last_status = f"ik_hold_residual:{pos_residual:.4f}"
            return self._last_cmd_deg.copy()
        if pos_residual > 0.01:
            self._last_status = f"ik_best_effort:{pos_residual:.4f}"
        else:
            self._last_status = "quest_active"
        return np.rad2deg(self._cfg.q.copy())

    def _micro_delta(self, button_data: Dict) -> np.ndarray:
        step = self.config.micro_step
        grip_key = "leftGrip" if self.config.quest_hand == "l" else "rightGrip"
        if float(button_data.get(grip_key, (0.0,))[0]) > 0.5:
            step = self.config.micro_fast_step

        if self.config.quest_hand == "l":
            up_key, down_key, js_key = "Y", "X", "leftJS"
        else:
            up_key, down_key, js_key = "B", "A", "rightJS"

        if button_data.get(up_key, False):
            return np.array([0.0, 0.0, step], dtype=np.float64)
        if button_data.get(down_key, False):
            return np.array([0.0, 0.0, -step], dtype=np.float64)

        x, y = button_data.get(js_key, (0.0, 0.0))
        if np.linalg.norm(np.array([x, y], dtype=np.float64)) > 1e-3:
            return np.array([float(x) * step, float(y) * step, 0.0], dtype=np.float64)
        return np.zeros(3, dtype=np.float64)

    def read(self) -> Dict[str, object]:
        q_deg = self._read_measured_joints_deg()
        ee_pos, ee_rot = self._fk_pose(q_deg)
        q_cmd = self._last_cmd_deg.copy()
        pose_data, button_data = self.reader.get_transformations_and_buttons()

        pose_key = self.config.quest_hand
        trigger_key = "leftTrig" if pose_key == "l" else "rightTrig"
        if not pose_data or not button_data:
            self._last_status = "quest_no_data"
            return self._frame(q_deg, ee_pos, ee_rot, q_cmd)

        micro_delta = self._micro_delta(button_data)
        self.micro_active = bool(np.linalg.norm(micro_delta) > 0.0)

        trigger_value = float(button_data.get(trigger_key, (0.0,))[0])
        trigger_pressed = trigger_value > 0.5
        if trigger_pressed and not self._trigger_latch_pressed:
            self._trigger_latch_pressed = True
            if self.control_active:
                self.control_active = False
                self._prev_quest_pose = None
                self._target_pos = None
                self._target_rot = None
                self._last_status = "quest_latch_off"
                return self._frame(q_deg, ee_pos, ee_rot, q_cmd)
            if not _quest_pose_ready(pose_data, pose_key):
                self._last_status = "quest_pose_not_ready"
                return self._frame(q_deg, ee_pos, ee_rot, q_cmd)
            self.control_active = True
            self._prev_quest_pose = pose_data[pose_key].copy()
            self._target_pos = ee_pos.copy()
            self._target_rot = ee_rot.copy()
            self._ema_pos = None
            self._ema_rotvec = None
            self._last_cmd_deg = q_deg.copy()
            self._last_status = "quest_latch_on"
            return self._frame(q_deg, ee_pos, ee_rot, q_deg)

        if (not trigger_pressed) and trigger_value <= 0.15:
            self._trigger_latch_pressed = False

        q_seed = q_deg
        if self.micro_active:
            target_pos = ee_pos + micro_delta
            q_cmd = self._solve_target(q_seed, target_pos, ee_rot)
        elif self.control_active:
            if not _quest_pose_ready(pose_data, pose_key):
                self._last_status = "quest_pose_invalid"
                q_cmd = self._last_cmd_deg.copy()
            else:
                current_pose = pose_data[pose_key]
                prev_pose = self._prev_quest_pose if self._prev_quest_pose is not None else current_pose
                delta_pos = current_pose[:3, 3] - prev_pose[:3, 3]
                delta_rot = current_pose[:3, :3] @ np.linalg.inv(prev_pose[:3, :3])
                self._prev_quest_pose = current_pose.copy()

                delta_rotvec = Rotation.from_matrix(delta_rot).as_rotvec()
                if self._ema_pos is None:
                    self._ema_pos = delta_pos.copy()
                    self._ema_rotvec = delta_rotvec.copy()
                else:
                    a = self.config.ema_alpha
                    self._ema_pos = a * delta_pos + (1.0 - a) * self._ema_pos
                    self._ema_rotvec = a * delta_rotvec + (1.0 - a) * self._ema_rotvec

                delta_pos_ur = _apply_transfer(quest2ur, self._ema_pos)
                delta_rot_ur = quest2ur[:3, :3] @ Rotation.from_rotvec(self._ema_rotvec).as_matrix() @ ur2quest[:3, :3]
                rotvec = Rotation.from_matrix(delta_rot_ur).as_rotvec()
                delta_rot_ur = Rotation.from_rotvec(rotvec * np.array([1.0, 1.0, -1.0])).as_matrix()

                if self._target_pos is None:
                    self._target_pos = ee_pos.copy()
                if self._target_rot is None:
                    self._target_rot = ee_rot.copy()
                self._target_pos = self._target_pos + delta_pos_ur
                self._target_rot = delta_rot_ur @ self._target_rot
                q_cmd = self._solve_target(q_seed, self._target_pos, self._target_rot)
        else:
            self._last_status = "quest_idle"

        self._last_cmd_deg = np.asarray(q_cmd, dtype=np.float64)[:6]
        if self.control_active or self.micro_active:
            self._pending_cmd_deg = self._last_cmd_deg.copy()
        else:
            self._pending_cmd_deg = None
        return self._frame(q_deg, ee_pos, ee_rot, self._last_cmd_deg)

    def command_pending(self) -> None:
        if not self.config.command_robot or self.robot is None:
            return
        if self._pending_cmd_deg is None:
            return
        self.robot.command_joints_deg(self._pending_cmd_deg)

    def _frame(
        self,
        q_deg: np.ndarray,
        ee_pos: np.ndarray,
        ee_rot: np.ndarray,
        q_cmd: np.ndarray,
    ) -> Dict[str, object]:
        quat_xyzw = Rotation.from_matrix(ee_rot).as_quat()
        return {
            "arm_joint_positions": np.asarray(q_deg, dtype=np.float32)[:6],
            "arm_joint_velocities": np.zeros(6, dtype=np.float32),
            "arm_eef_pose": np.concatenate([ee_pos, quat_xyzw]).astype(np.float32),
            "arm_control": np.asarray(q_cmd, dtype=np.float32)[:6],
            "arm_side": self.config.arm_side,
            "arm_connected": self.robot is not None,
            "arm_control_active": bool(self.control_active),
            "arm_micro_active": bool(self.micro_active),
            "arm_command_enabled": bool(self.config.command_robot),
            "quest_status": self._last_status,
        }

    def close(self) -> None:
        if self.robot is not None:
            self.robot.close()
