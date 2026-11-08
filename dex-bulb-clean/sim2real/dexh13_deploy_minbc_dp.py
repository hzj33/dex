from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
MINBC_ROOT = REPO_ROOT / "minbc"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MINBC_ROOT) not in sys.path:
    sys.path.insert(0, str(MINBC_ROOT))

try:
    from sim2real import dexh13_deploy as deploy
    from sim2real.data_collector import Sim2RealDataCollector
    from sim2real.dexh13_deploy_collect import TactileReader
    from sim2real.hardware_state import FixedWristStateReader, NullArmStateReader
    from sim2real.quest_rm65b_control import QuestRM65BArmReader, QuestRM65BConfig
except ModuleNotFoundError:
    import dexh13_deploy as deploy
    from data_collector import Sim2RealDataCollector
    from dexh13_deploy_collect import TactileReader
    from hardware_state import FixedWristStateReader, NullArmStateReader
    from quest_rm65b_control import QuestRM65BArmReader, QuestRM65BConfig

from configs.base import MinBCConfig
from dp.agent import Agent as MinBCAgent


TACTILE_DIM = 33


def _merge_dataclass(obj: Any, updates: dict[str, Any]) -> Any:
    values: dict[str, Any] = {}
    for field in dataclasses.fields(obj):
        value = getattr(obj, field.name)
        if field.name in updates:
            update = updates[field.name]
            if dataclasses.is_dataclass(value) and isinstance(update, dict):
                value = _merge_dataclass(value, update)
            elif isinstance(value, tuple) and isinstance(update, list):
                value = tuple(update)
            else:
                value = update
        values[field.name] = value
    return obj.__class__(**values)


def load_minbc_agent(ckpt_path: Path, device: str, diffusion_iters: int | None) -> MinBCAgent:
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing MinBC config: {config_path}")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    config = _merge_dataclass(MinBCConfig(), config_dict)
    if config.policy_type != "dp":
        raise ValueError(f"Expected a DP checkpoint, got policy_type={config.policy_type!r}")
    num_iters = int(diffusion_iters or config.dp.diffusion_iters)

    agent = MinBCAgent(
        config,
        clip_far=False,
        num_diffusion_iters=num_iters,
        load_img="img" in config.data.data_key,
        num_workers=0,
        dit=False,
        device=device,
    )
    agent.load(str(ckpt_path))
    print(f"[MinBCDeploy] Loaded checkpoint: {ckpt_path}")
    print(f"[MinBCDeploy] data_key={config.data.data_key}, diffusion_iters={num_iters}")
    return agent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Deploy a MinBC DP policy on DexH13.")
    parser.add_argument(
        "--ckpt",
        default=str(REPO_ROOT / "outputs" / "minbc" / "dexh13_dp_notactile_seed0" / "model_best.ckpt"),
        help="Path to model_best.ckpt or model_last.ckpt. config/stats/norm must be in the same directory.",
    )
    parser.add_argument("--device", default="cuda:0", help="cuda:0 or cpu.")
    parser.add_argument("--diffusion-iters", type=int, default=None)
    parser.add_argument(
        "--replan-every",
        type=int,
        default=0,
        help="Run DP inference every N control ticks. 0 means use config.dp.act_horizon.",
    )
    parser.add_argument(
        "--action-mode",
        choices=("delta", "target"),
        default="delta",
        help="delta: action * sim_action_scale + previous target. target: action is joint target.",
    )
    parser.add_argument("--action-scale", type=float, default=deploy.ACTION_SCALE)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--max-policy-steps", type=int, default=0)
    parser.add_argument(
        "--dry-run-load",
        action="store_true",
        help="Only load the MinBC policy and exit without opening hardware.",
    )
    parser.add_argument("--hand-side", choices=("right", "left"), default="right")
    parser.add_argument("--handy-port", default=deploy.HANDY_PORT)
    parser.add_argument("--camera-port", default=deploy.CAMERA_PORT)
    parser.add_argument("--save-data", action="store_true")
    parser.add_argument("--sync-write-data", action="store_true")
    parser.add_argument(
        "--data-dir",
        default=str(REPO_ROOT / "data" / "sim2real_minbc_dp"),
        help="Root directory for optional deployment logs.",
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--arm-state-mode",
        choices=("none", "quest-rm65b"),
        default="none",
        help="Read arm state, and optionally command RM65B with Quest targets.",
    )
    parser.add_argument("--quest-ip", default="192.168.12.38")
    parser.add_argument("--quest-hand", choices=("l", "r"), default="l")
    parser.add_argument("--arm-side", choices=("right", "left"), default="right")
    parser.add_argument("--rm65b-ip", default="192.168.2.19")
    parser.add_argument("--rm65b-port", type=int, default=8080)
    parser.add_argument("--no-rm65b-connect", action="store_true")
    parser.add_argument("--command-rm65b", action="store_true")
    parser.add_argument("--no-send-between-policy", action="store_true")
    parser.add_argument("--no-reset-on-exit", action="store_true")
    parser.add_argument("--reset-hold-sec", type=float, default=1.0)
    parser.add_argument("--reset-target", choices=("zero", "initial"), default="zero")
    return parser.parse_args()


def make_minbc_obs(
    *,
    data_key: tuple[str, ...],
    obs: np.ndarray,
    proprio_hist: np.ndarray,
    q_hand: np.ndarray,
    tactile: np.ndarray,
) -> dict[str, np.ndarray]:
    frame: dict[str, np.ndarray] = {}
    if "dexh13_obs" in data_key:
        frame["dexh13_obs"] = np.asarray(obs, dtype=np.float32).reshape(deploy.OBS_DIM)
    if "dexh13_proprio_hist" in data_key:
        frame["dexh13_proprio_hist"] = np.asarray(proprio_hist, dtype=np.float32).reshape(-1)
    if "paxini_pos" in data_key:
        frame["paxini_pos"] = np.asarray(q_hand, dtype=np.float32).reshape(deploy.NUM_DOF)
    if "paxini_tactile" in data_key:
        frame["paxini_tactile"] = np.asarray(tactile, dtype=np.float32).reshape(TACTILE_DIM)
    missing = [key for key in data_key if key not in frame]
    if missing:
        raise KeyError(f"Cannot build MinBC observation keys: {missing}")
    return frame


def main() -> None:
    args = parse_args()
    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is not available.")

    deploy.HANDY_PORT = args.handy_port
    deploy.CAMERA_PORT = args.camera_port
    control_dt = 1.0 / float(args.control_hz)

    agent = load_minbc_agent(ckpt_path, args.device, args.diffusion_iters)
    data_key = tuple(agent.config.data.data_key)
    replan_every = int(args.replan_every or agent.act_horizon)
    replan_every = max(1, replan_every)
    if args.dry_run_load:
        print("[MinBCDeploy] dry-run load OK.")
        return

    controller = deploy.DexH13DeployControl()
    if not controller.open_device():
        sys.exit(1)

    tactile_reader = TactileReader(controller)
    if args.arm_state_mode == "quest-rm65b":
        arm_reader = QuestRM65BArmReader(
            QuestRM65BConfig(
                quest_ip=args.quest_ip,
                quest_hand=args.quest_hand,
                arm_side=args.arm_side,
                robot_ip=args.rm65b_ip,
                robot_port=args.rm65b_port,
                connect_robot=not args.no_rm65b_connect,
                command_robot=args.command_rm65b,
            )
        )
    else:
        arm_reader = NullArmStateReader(num_joints=6)
    wrist_reader = FixedWristStateReader(angle_dim=1)
    collector = (
        Sim2RealDataCollector(
            args.data_dir,
            run_name=args.run_name,
            async_write=not args.sync_write_data,
        )
        if args.save_data
        else None
    )

    initial_target = np.asarray(
        [
            0.34,
            0.92,
            0.28,
            1.13,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -0.34,
            1.56,
            0.22,
            0.41,
        ],
        dtype=np.float32,
    )

    action_queue: collections.deque[np.ndarray] = collections.deque()
    obs_deque: collections.deque[dict[str, np.ndarray]] = collections.deque(
        maxlen=agent.obs_horizon
    )

    try:
        controller.send_joint_target(initial_target)
        time.sleep(1.0)

        prev_target = initial_target.copy()
        target_q = prev_target.copy()
        last_action = np.zeros(deploy.NUM_DOF, dtype=np.float32)
        q_init = deploy.clamp_locked_joints(controller.read_joint_pos())
        deploy.reset_proprio_hist(q_init, prev_target)
        last_tick_time = time.perf_counter() - control_dt
        policy_step = 0

        print("[MinBCDeploy] Running DexH13 MinBC DP policy.")
        print(
            f"[MinBCDeploy] control_hz={args.control_hz:g}, "
            f"replan_every={replan_every}, action_mode={args.action_mode}"
        )

        while True:
            now = time.perf_counter()
            next_tick = last_tick_time + control_dt
            if now < next_tick:
                if not args.no_send_between_policy:
                    controller.send_joint_target(target_q)
                else:
                    time.sleep(min(0.001, next_tick - now))
                continue

            tick_start = time.perf_counter()
            last_tick_time = tick_start

            q_dexh13 = controller.read_joint_pos()
            q_policy = deploy.clamp_locked_joints(q_dexh13)
            deploy.update_proprio_hist(q_policy, prev_target)
            obs = deploy.make_obs_from_proprio_hist().detach().cpu().numpy().reshape(-1)
            proprio_hist = (
                deploy.proprio_hist_buffer.detach()
                .cpu()
                .numpy()
                .reshape(deploy.PROPRIO_HIST_LEN, deploy.PROPRIO_DIM)
            )
            tactile = tactile_reader.read()
            arm_state = arm_reader.read()
            wrist_state = wrist_reader.read()

            obs_deque.append(
                make_minbc_obs(
                    data_key=data_key,
                    obs=obs,
                    proprio_hist=proprio_hist,
                    q_hand=q_dexh13,
                    tactile=tactile,
                )
            )
            while len(obs_deque) < agent.obs_horizon:
                obs_deque.append(obs_deque[-1])

            should_replan = (policy_step % replan_every == 0) or not action_queue
            if should_replan:
                predicted = agent.predict(obs_deque, num_diffusion_iters=agent.policy.num_diffusion_iters)
                action_queue.clear()
                for action in predicted:
                    action_queue.append(np.asarray(action, dtype=np.float32).reshape(deploy.NUM_DOF))

            if action_queue:
                last_action = action_queue.popleft()

            if args.action_mode == "delta":
                last_action = np.clip(last_action, -1.0, 1.0)
                target_q = last_action * float(args.action_scale) + prev_target
            else:
                target_q = last_action.copy()
            target_q = np.clip(
                target_q,
                deploy.dexh13_dof_lower_limits,
                deploy.dexh13_dof_upper_limits,
            )
            target_q = deploy.clamp_locked_joints(target_q)

            if collector is not None:
                collector.save_frame(
                    timestamp=dt.datetime.now(),
                    policy_step=policy_step,
                    hand_side=args.hand_side,
                    q_hand=q_dexh13,
                    target_q=target_q,
                    action=last_action,
                    tactile=tactile,
                    obs=obs,
                    proprio_hist=proprio_hist,
                    q_policy=q_policy,
                    arm_state=arm_state,
                    wrist_state=wrist_state,
                    policy_running=True,
                    sim_action_scale=float(args.action_scale),
                    sim_control_dt=control_dt,
                )

            controller.send_joint_target(target_q)
            if hasattr(arm_reader, "command_pending"):
                arm_reader.command_pending()
            prev_target = target_q.copy()

            policy_step += 1
            if args.max_policy_steps > 0 and policy_step >= args.max_policy_steps:
                print(f"[MinBCDeploy] Reached max policy steps: {args.max_policy_steps}")
                break

    except KeyboardInterrupt:
        print("\n[MinBCDeploy] Received KeyboardInterrupt, exiting...")
    finally:
        if collector is not None:
            collector.save_statistics()
        if not args.no_reset_on_exit:
            reset_target = (
                np.zeros(deploy.NUM_DOF, dtype=np.float32)
                if args.reset_target == "zero"
                else initial_target
            )
            print(f"[MinBCDeploy] Resetting DexH13 hand to {args.reset_target} target...")
            reset_until = time.perf_counter() + max(0.0, args.reset_hold_sec)
            try:
                while time.perf_counter() < reset_until:
                    controller.send_joint_target(reset_target)
                    time.sleep(0.05)
                controller.send_joint_target(reset_target)
            except Exception as exc:
                print(f"[MinBCDeploy] Reset failed: {exc}")
        arm_reader.close()
        wrist_reader.close()
        controller.close()


if __name__ == "__main__":
    main()
