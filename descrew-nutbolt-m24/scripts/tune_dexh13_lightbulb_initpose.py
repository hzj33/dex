#!/usr/bin/env python3
"""Interactive hand root pose and joint init-pose tuner.

This script starts a single Isaac Gym viewer env, lets the user adjust the
hand root position/orientation and the 16 hand DOF positions, then dumps a YAML
snippet that can be pasted back into the task config.
"""

import argparse
import datetime as _dt
import os
import sys
from typing import Dict, Iterable, List

import isaacgym  # noqa: F401
import torch
from hydra import compose, initialize_config_dir
from isaacgym import gymapi, gymtorch
from isaacgym.torch_utils import quat_from_euler_xyz
from omegaconf import OmegaConf

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.misc import set_seed
from dexscrew.utils.reformat import omegaconf_to_dict


def _register_resolvers() -> None:
    OmegaConf.clear_resolvers()
    OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
    OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
    OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
    OmegaConf.register_new_resolver(
        "resolve_default", lambda default, arg: default if arg == "" else arg
    )


def _fmt(values: Iterable[float]) -> str:
    return "[" + ", ".join(f"{float(v):.6f}" for v in values) + "]"


def _quat_from_rpy(rpy: List[float], device: str) -> torch.Tensor:
    quat = quat_from_euler_xyz(
        torch.tensor([rpy[0]], dtype=torch.float, device=device),
        torch.tensor([rpy[1]], dtype=torch.float, device=device),
        torch.tensor([rpy[2]], dtype=torch.float, device=device),
    )[0]
    return quat


def _cfg_list(cfg, key: str, default: List[float]) -> List[float]:
    value = cfg.get(key)
    if value is None:
        return list(default)
    return [float(x) for x in list(value)]


class InitPoseTuner:
    def __init__(self, env, out_path: str, pos_step: float, rot_step_deg: float, joint_step: float):
        self.env = env
        self.out_path = out_path
        self.pos_step = float(pos_step)
        self.rot_step = float(rot_step_deg) * 3.141592653589793 / 180.0
        self.joint_step = float(joint_step)
        self.mode = "hand"
        self.selected_joint = 0
        self.hand_actor_index = int(env.hand_indices[0].item())
        self.hand_dof_names = list(env.hand_dof_names[: env.num_xhand_hand_dofs])

        env.reset()
        env._refresh_gym()

        asset_cfg = env.config["env"].get("asset", {})
        self.hand_pos = _cfg_list(asset_cfg, "handRootPos", [0.0, 0.0, 0.21])
        self.hand_rpy = _cfg_list(asset_cfg, "handRootRPY", [3.1415, 0.3, 3.1415])
        self.dof_pos = torch.tensor(
            env.joint_values_lst[: env.num_xhand_hand_dofs],
            dtype=torch.float,
            device=env.device,
        )
        self._apply_state()
        self._subscribe_keys()
        self._print_help()
        self._print_status()

    def _subscribe_keys(self) -> None:
        gym = self.env.gym
        viewer = self.env.viewer
        bindings: Dict[str, object] = {
            "QUIT": gymapi.KEY_ESCAPE,
            "toggle_mode": gymapi.KEY_M,
            "save": gymapi.KEY_O,
            "print": gymapi.KEY_C,
            "step_down": gymapi.KEY_MINUS,
            "step_up": gymapi.KEY_EQUAL,
            "z_pos": gymapi.KEY_E,
            "z_neg": gymapi.KEY_Q,
            "roll_pos": gymapi.KEY_R,
            "roll_neg": gymapi.KEY_F,
            "pitch_pos": gymapi.KEY_T,
            "pitch_neg": gymapi.KEY_G,
            "yaw_pos": gymapi.KEY_Y,
            "yaw_neg": gymapi.KEY_H,
            "joint_prev": gymapi.KEY_LEFT,
            "joint_next": gymapi.KEY_RIGHT,
            "joint_inc": gymapi.KEY_UP,
            "joint_dec": gymapi.KEY_DOWN,
            "joint_prev_alt": gymapi.KEY_LEFT_BRACKET,
            "joint_next_alt": gymapi.KEY_RIGHT_BRACKET,
            "joint_dec_alt": gymapi.KEY_COMMA,
            "joint_inc_alt": gymapi.KEY_PERIOD,
            "joint_zero": gymapi.KEY_SPACE,
        }
        for action, key in bindings.items():
            gym.subscribe_viewer_keyboard_event(viewer, key, action)

        for i in range(10):
            action = f"select_{i}"
            main_key = getattr(gymapi, f"KEY_{i}")
            numpad_key = getattr(gymapi, f"KEY_NUMPAD_{i}")
            gym.subscribe_viewer_keyboard_event(viewer, main_key, action)
            gym.subscribe_viewer_keyboard_event(viewer, numpad_key, action)

    def _apply_state(self) -> None:
        env = self.env
        device = env.device
        actor_indices = env.hand_indices[:1].to(torch.int32)
        hand_idx = self.hand_actor_index

        env.root_state_tensor[hand_idx, 0:3] = torch.tensor(
            self.hand_pos, dtype=torch.float, device=device
        )
        env.root_state_tensor[hand_idx, 3:7] = _quat_from_rpy(self.hand_rpy, device)
        env.root_state_tensor[hand_idx, 7:13] = 0.0
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.root_state_tensor),
            gymtorch.unwrap_tensor(actor_indices),
            1,
        )

        full_dof = env.dof_state.view(env.num_envs, env.num_dofs, 2)
        full_dof[0, : env.num_xhand_hand_dofs, 0] = self.dof_pos
        full_dof[0, : env.num_xhand_hand_dofs, 1] = 0.0
        env.prev_targets[0, : env.num_xhand_hand_dofs] = self.dof_pos
        env.cur_targets[0, : env.num_xhand_hand_dofs] = self.dof_pos
        env.xhand_hand_dof_pos[0, :] = self.dof_pos
        env.xhand_hand_dof_vel[0, :] = 0.0
        env.gym.set_dof_state_tensor_indexed(
            env.sim,
            gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(actor_indices),
            1,
        )
        if not env.torque_control:
            env.gym.set_dof_position_target_tensor_indexed(
                env.sim,
                gymtorch.unwrap_tensor(env.prev_targets),
                gymtorch.unwrap_tensor(actor_indices),
                1,
            )

    def _print_help(self) -> None:
        print(
            "\nInteractive hand init-pose tuner\n"
            "  M: toggle hand/joint mode\n"
            "  O: save YAML snippet, C: print current values, ESC: quit\n"
            "  -/=: decrease/increase active step size\n"
            "\nHand mode:\n"
            "  1/3: x -/+    2/5: y -/+    Q/E: z -/+\n"
            "  (main-row and numpad digits are both supported)\n"
            "  F/R: roll -/+ G/T: pitch -/+ H/Y: yaw -/+\n"
            "\nJoint mode:\n"
            "  Left/Right or [/]: select DOF\n"
            "  Down/Up or ,/.: selected DOF -/+\n"
            "  0-9: select DOF 0-9, Space: set selected DOF to 0\n"
        )

    def _print_status(self) -> None:
        joint_name = self.hand_dof_names[self.selected_joint]
        print(
            f"mode={self.mode} pos={_fmt(self.hand_pos)} rpy={_fmt(self.hand_rpy)} "
            f"joint[{self.selected_joint}]={joint_name}:{float(self.dof_pos[self.selected_joint]):.6f} "
            f"steps(pos={self.pos_step:.4f}, rot_deg={self.rot_step * 180.0 / 3.141592653589793:.2f}, joint={self.joint_step:.4f})"
        )

    def _save(self) -> None:
        out_dir = os.path.dirname(self.out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        lines = [
            "# Paste this under env.asset in the task YAML.",
            "handRootPos: " + _fmt(self.hand_pos),
            "handRootRPY: " + _fmt(self.hand_rpy),
            "handInitPose:",
        ]
        for idx, name in enumerate(self.hand_dof_names):
            lines.append(f"  {name}: {float(self.dof_pos[idx]):.10f}")
        text = "\n".join(lines) + "\n"
        with open(self.out_path, "w", encoding="utf-8") as f:
            f.write(text)
        print("\nSaved init-pose YAML snippet to:", self.out_path)
        print(text)

    def _change_step(self, direction: float) -> None:
        if self.mode == "hand":
            self.pos_step = max(0.0001, self.pos_step * (1.25 if direction > 0 else 0.8))
            self.rot_step = max(0.001, self.rot_step * (1.25 if direction > 0 else 0.8))
        else:
            self.joint_step = max(0.0005, self.joint_step * (1.25 if direction > 0 else 0.8))

    def _handle_event(self, action: str) -> bool:
        changed = False
        if action == "QUIT":
            self._save()
            return True
        if action == "toggle_mode":
            self.mode = "joint" if self.mode == "hand" else "hand"
            self._print_status()
            return False
        if action == "save":
            self._save()
            return False
        if action == "print":
            self._print_status()
            return False
        if action == "step_down":
            self._change_step(-1.0)
            self._print_status()
            return False
        if action == "step_up":
            self._change_step(1.0)
            self._print_status()
            return False

        if self.mode == "hand":
            delta_pos = {
                "select_1": (0, -self.pos_step),
                "select_3": (0, self.pos_step),
                "select_2": (1, -self.pos_step),
                "select_5": (1, self.pos_step),
                "z_neg": (2, -self.pos_step),
                "z_pos": (2, self.pos_step),
            }
            delta_rpy = {
                "roll_neg": (0, -self.rot_step),
                "roll_pos": (0, self.rot_step),
                "pitch_neg": (1, -self.rot_step),
                "pitch_pos": (1, self.rot_step),
                "yaw_neg": (2, -self.rot_step),
                "yaw_pos": (2, self.rot_step),
            }
            if action in delta_pos:
                axis, amount = delta_pos[action]
                self.hand_pos[axis] += amount
                changed = True
            elif action in delta_rpy:
                axis, amount = delta_rpy[action]
                self.hand_rpy[axis] += amount
                changed = True
        else:
            if action in {"joint_prev", "joint_prev_alt"}:
                self.selected_joint = (self.selected_joint - 1) % len(self.hand_dof_names)
                self._print_status()
            elif action in {"joint_next", "joint_next_alt"}:
                self.selected_joint = (self.selected_joint + 1) % len(self.hand_dof_names)
                self._print_status()
            elif action in {"joint_dec", "joint_dec_alt"}:
                self.dof_pos[self.selected_joint] -= self.joint_step
                changed = True
            elif action in {"joint_inc", "joint_inc_alt"}:
                self.dof_pos[self.selected_joint] += self.joint_step
                changed = True
            elif action == "joint_zero":
                self.dof_pos[self.selected_joint] = 0.0
                changed = True
            elif action.startswith("select_"):
                idx = int(action.split("_", 1)[1])
                if idx < len(self.hand_dof_names):
                    self.selected_joint = idx
                    self._print_status()

            self.dof_pos[:] = torch.max(
                torch.min(self.dof_pos, self.env.xhand_hand_dof_upper_limits),
                self.env.xhand_hand_dof_lower_limits,
            )

        if changed:
            self._apply_state()
            self._print_status()
        return False

    def run(self) -> None:
        gym = self.env.gym
        sim = self.env.sim
        viewer = self.env.viewer
        while viewer and not gym.query_viewer_has_closed(viewer):
            for evt in gym.query_viewer_action_events(viewer):
                if evt.value <= 0:
                    continue
                if self._handle_event(evt.action):
                    return

            self._apply_state()
            gym.simulate(sim)
            gym.fetch_results(sim, True)
            gym.step_graphics(sim)
            gym.draw_viewer(viewer, sim, True)
            gym.sync_frame_time(sim)

        self._save()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Dexh13HoraLightbulb")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    parser.add_argument("--pos-step", type=float, default=0.002)
    parser.add_argument("--rot-step-deg", type=float, default=1.0)
    parser.add_argument("--joint-step", type=float, default=0.02)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    _register_resolvers()
    out_path = args.out
    if not out_path:
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"outputs/initpose_tuning/{args.task}_{stamp}.yaml"

    base_overrides = [
        f"task={args.task}",
        "headless=False",
        "num_envs=1",
        "task.env.numEnvs=1",
        f"seed={args.seed}",
        f"sim_device=cuda:{args.gpu}",
        f"rl_device=cuda:{args.gpu}",
        "graphics_device_id=0",
        "wandb_activate=False",
        "task.env.randomization.randomizeMass=False",
        "task.env.randomization.randomizeCOM=False",
        "task.env.randomization.randomizeFriction=False",
        "task.env.randomization.randomizeScale=False",
        "task.env.randomization.randomizePDGains=False",
        "task.env.object.init_pos_noise=[0.0,0.0,0.0]",
        "task.env.asset.handRootPosNoise=[0.0,0.0,0.0]",
        "task.env.forceScale=0.0",
        "task.env.randomForceProbScalar=0.0",
    ]
    base_overrides.extend(args.overrides)

    config_dir = os.path.join(REPO_ROOT, "configs")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(config_name="config", overrides=base_overrides)

    set_seed(cfg.seed)
    env = isaacgym_task_map[cfg.task_name](
        config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )
    tuner = InitPoseTuner(
        env,
        out_path=out_path,
        pos_step=args.pos_step,
        rot_step_deg=args.rot_step_deg,
        joint_step=args.joint_step,
    )
    tuner.run()


if __name__ == "__main__":
    main()
