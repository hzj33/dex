from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np
import torch

try:
    from sim2real import dexh13_deploy as deploy
    from sim2real.data_collector import Sim2RealDataCollector
    from sim2real.hardware_state import FixedWristStateReader, NullArmStateReader
    from sim2real.quest_rm65b_control import QuestRM65BArmReader, QuestRM65BConfig
except ModuleNotFoundError:
    import dexh13_deploy as deploy
    from data_collector import Sim2RealDataCollector
    from hardware_state import FixedWristStateReader, NullArmStateReader
    from quest_rm65b_control import QuestRM65BArmReader, QuestRM65BConfig


TACTILE_DIM = 33
PROFILE_FIELDS = (
    "policy_step",
    "timestamp",
    "read_hand_q_ms",
    "policy_infer_ms",
    "action_build_ms",
    "read_tactile_ms",
    "read_arm_ms",
    "read_wrist_ms",
    "enqueue_save_ms",
    "send_hand_ms",
    "command_arm_ms",
    "loop_total_ms",
)


class TactileReader:
    def __init__(self, controller: deploy.DexH13DeployControl, read_detail: bool = False):
        self._controller = controller
        self._read_detail = bool(read_detail)
        self._last_tactile = np.zeros(TACTILE_DIM, dtype=np.int32)
        self._last_tactile_detail = np.zeros(0, dtype=np.int32)
        self._error_count = 0
        self._detail_error_count = 0

    @staticmethod
    def _force_points_to_xyz(raw) -> np.ndarray:
        tactile = []
        for force_point in raw:
            tactile.extend(
                [
                    int(force_point.x),
                    int(force_point.y),
                    int(force_point.z),
                ]
            )
        return np.asarray(tactile, dtype=np.int32)

    def read(self) -> np.ndarray:
        tactile, _ = self.read_with_detail()
        return tactile

    def read_with_detail(self) -> tuple[np.ndarray, np.ndarray]:
        try:
            raw = self._controller.control.getFingerTactile()
            tactile = self._force_points_to_xyz(raw)
            if tactile.size < TACTILE_DIM:
                tactile = np.pad(tactile, (0, TACTILE_DIM - tactile.size))
            self._last_tactile = tactile[:TACTILE_DIM].astype(np.int32, copy=True)
        except Exception as exc:
            self._error_count += 1
            if self._error_count in (1, 20, 100):
                print(f"[TactileReader] getFingerTactile failed: {exc}")

        if self._read_detail:
            try:
                raw_detail = self._controller.control.getFingerTactileDetail()
                self._last_tactile_detail = self._force_points_to_xyz(raw_detail)
            except Exception as exc:
                self._detail_error_count += 1
                if self._detail_error_count in (1, 20, 100):
                    print(f"[TactileReader] getFingerTactileDetail failed: {exc}")

        return self._last_tactile.copy(), self._last_tactile_detail.copy()


class LoopProfiler:
    def __init__(self, enabled: bool):
        self.enabled = enabled
        self.rows: list[dict[str, float | int | str]] = []

    def add(self, row: dict[str, float | int | str]) -> None:
        if self.enabled:
            self.rows.append(row)

    def save(self, output_dir: Path | None) -> None:
        if not self.enabled or not self.rows:
            return
        if output_dir is None:
            output_dir = Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        csv_path = output_dir / "profile_loop.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=PROFILE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rows)

        summary_path = output_dir / "profile_loop_summary.txt"
        numeric_fields = [field for field in PROFILE_FIELDS if field.endswith("_ms")]
        with open(summary_path, "w") as f:
            for field in numeric_fields:
                values = np.asarray([float(row[field]) for row in self.rows], dtype=np.float64)
                f.write(
                    f"{field}: mean={values.mean():.3f} "
                    f"p50={np.percentile(values, 50):.3f} "
                    f"p90={np.percentile(values, 90):.3f} "
                    f"p95={np.percentile(values, 95):.3f} "
                    f"max={values.max():.3f}\n"
                )
        print(f"[LoopProfiler] Saved: {csv_path}")
        print(f"[LoopProfiler] Saved: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DexH13 policy deployment with synchronized data collection."
    )
    parser.add_argument("--save-data", action="store_true", help="Save deployment frames.")
    parser.add_argument(
        "--sync-write-data",
        action="store_true",
        help="Write pkl files in the control loop. Default uses a background writer thread.",
    )
    parser.add_argument(
        "--data-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "sim2real_policy"),
        help="Root directory for saved pkl frames.",
    )
    parser.add_argument("--run-name", default=None, help="Optional run directory name.")
    parser.add_argument("--hand-side", choices=("right", "left"), default="right")
    parser.add_argument("--handy-port", default=deploy.HANDY_PORT)
    parser.add_argument("--camera-port", default=deploy.CAMERA_PORT)
    parser.add_argument(
        "--tactile-detail",
        action="store_true",
        help="Also collect getFingerTactileDetail() array data into paxini_tactile_detail.",
    )
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=0,
        help="Stop after this many policy steps. 0 means run until Ctrl-C.",
    )
    parser.add_argument(
        "--no-send-between-policy",
        action="store_true",
        help="Only send commands on policy ticks. Default matches the original deploy loop.",
    )
    parser.add_argument(
        "--arm-state-mode",
        choices=("none", "quest-rm65b"),
        default="none",
        help="Read-only arm/control source. quest-rm65b reads Quest and RM65B state; command is opt-in.",
    )
    parser.add_argument(
        "--wrist-state-mode",
        choices=("fixed",),
        default="fixed",
        help="Read-only wrist state source. Currently a fixed/locked placeholder.",
    )
    parser.add_argument("--quest-ip", default="192.168.12.38")
    parser.add_argument("--quest-hand", choices=("l", "r"), default="l")
    parser.add_argument("--arm-side", choices=("right", "left"), default="right")
    parser.add_argument("--rm65b-ip", default="192.168.2.19")
    parser.add_argument("--rm65b-port", type=int, default=8080)
    parser.add_argument(
        "--no-rm65b-connect",
        action="store_true",
        help="Do not connect to the real RM65B; use the last command as simulated measured state.",
    )
    parser.add_argument(
        "--command-rm65b",
        action="store_true",
        help="Actually send Quest-generated arm_control to RM65B. Default only records targets.",
    )
    parser.add_argument(
        "--no-reset-on-exit",
        action="store_true",
        help="Do not move the DexH13 hand back to its initial target before closing.",
    )
    parser.add_argument(
        "--reset-hold-sec",
        type=float,
        default=1.0,
        help="How long to keep sending the reset target before disabling the hand.",
    )
    parser.add_argument(
        "--reset-target",
        choices=("zero", "initial"),
        default="zero",
        help="DexH13 target used on exit reset.",
    )
    parser.add_argument(
        "--profile-loop",
        action="store_true",
        help="Record per-policy-step timing to profile_loop.csv and profile_loop_summary.txt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    deploy.HANDY_PORT = args.handy_port
    deploy.CAMERA_PORT = args.camera_port

    controller = deploy.DexH13DeployControl()
    if not controller.open_device():
        sys.exit(1)

    tactile_reader = TactileReader(controller, read_detail=args.tactile_detail)
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
    profiler = LoopProfiler(args.profile_loop)

    prev_target_policy = np.asarray(
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

    try:
        controller.send_joint_target(prev_target_policy)
        time.sleep(1.0)

        prev_target = prev_target_policy.copy()
        target_q = prev_target.copy()
        last_action = np.zeros(deploy.NUM_DOF, dtype=np.float32)
        q_init = deploy.clamp_locked_joints(controller.read_joint_pos())
        deploy.reset_proprio_hist(q_init, prev_target)
        last_policy_time = time.perf_counter() - deploy.POLICY_DT
        policy_step = 0

        print("[DeployCollect] Running DexH13 policy.")
        print(
            f"[DeployCollect] hand_side={args.hand_side}, handy_port={args.handy_port}, "
            f"save_data={args.save_data}, tactile_detail={args.tactile_detail}"
        )

        while True:
            now = time.perf_counter()
            next_policy_time = last_policy_time + deploy.POLICY_DT
            if now < next_policy_time:
                if not args.no_send_between_policy:
                    controller.send_joint_target(target_q)
                else:
                    time.sleep(min(0.001, next_policy_time - now))
                continue

            tick_start = time.perf_counter()
            q_dexh13 = controller.read_joint_pos()
            t_after_read_hand = time.perf_counter()
            q_policy = deploy.clamp_locked_joints(q_dexh13)

            last_policy_time = tick_start

            policy_start = time.perf_counter()
            deploy.update_proprio_hist(q_policy, prev_target)
            obs = deploy.make_obs_from_proprio_hist()
            with torch.no_grad():
                obs = torch.clamp(obs, -5.0, 5.0)

                norm_obs = (obs - deploy.obs_mean) / torch.sqrt(deploy.obs_var + 1e-5)
                norm_obs = torch.clamp(norm_obs, -5.0, 5.0)

                norm_proprio_hist = (
                    deploy.proprio_hist_buffer - deploy.sa_mean
                ) / torch.sqrt(deploy.sa_var + 1e-5)
                norm_proprio_hist = torch.clamp(norm_proprio_hist, -5.0, 5.0)
                point_cloud_info = torch.zeros((1, 100, 3), device=deploy.DEVICE)

                input_dict = {
                    "obs": norm_obs,
                    "proprio_hist": norm_proprio_hist,
                    "point_cloud_info": point_cloud_info,
                }

                mu, _, _ = deploy.jit_model(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
                last_action = mu.detach().cpu().numpy().flatten().astype(np.float32)
            t_after_policy = time.perf_counter()

            target_q = last_action * deploy.ACTION_SCALE + prev_target
            target_q = np.clip(
                target_q,
                deploy.dexh13_dof_lower_limits,
                deploy.dexh13_dof_upper_limits,
            )
            target_q = deploy.clamp_locked_joints(target_q)
            t_after_action = time.perf_counter()
            tactile, tactile_detail = tactile_reader.read_with_detail()
            t_after_tactile = time.perf_counter()
            arm_state = arm_reader.read()
            t_after_arm = time.perf_counter()
            wrist_state = wrist_reader.read()
            t_after_wrist = time.perf_counter()

            if collector is not None:
                collector.save_frame(
                    timestamp=dt.datetime.now(),
                    policy_step=policy_step,
                    hand_side=args.hand_side,
                    q_hand=q_dexh13,
                    target_q=target_q,
                    action=last_action,
                    tactile=tactile,
                    tactile_detail=tactile_detail,
                    obs=obs.detach().cpu().numpy().reshape(-1),
                    proprio_hist=deploy.proprio_hist_buffer.detach()
                    .cpu()
                    .numpy()
                    .reshape(deploy.PROPRIO_HIST_LEN, deploy.PROPRIO_DIM),
                    q_policy=q_policy,
                    arm_state=arm_state,
                    wrist_state=wrist_state,
                    policy_running=True,
                    sim_action_scale=deploy.ACTION_SCALE,
                    sim_control_dt=deploy.POLICY_DT,
                )
            t_after_save = time.perf_counter()

            controller.send_joint_target(target_q)
            t_after_send_hand = time.perf_counter()
            if hasattr(arm_reader, "command_pending"):
                arm_reader.command_pending()
            t_after_command_arm = time.perf_counter()
            prev_target = target_q.copy()

            profiler.add(
                {
                    "policy_step": policy_step,
                    "timestamp": dt.datetime.now().isoformat(),
                    "read_hand_q_ms": (t_after_read_hand - tick_start) * 1000.0,
                    "policy_infer_ms": (t_after_policy - policy_start) * 1000.0,
                    "action_build_ms": (t_after_action - t_after_policy) * 1000.0,
                    "read_tactile_ms": (t_after_tactile - t_after_action) * 1000.0,
                    "read_arm_ms": (t_after_arm - t_after_tactile) * 1000.0,
                    "read_wrist_ms": (t_after_wrist - t_after_arm) * 1000.0,
                    "enqueue_save_ms": (t_after_save - t_after_wrist) * 1000.0,
                    "send_hand_ms": (t_after_send_hand - t_after_save) * 1000.0,
                    "command_arm_ms": (t_after_command_arm - t_after_send_hand) * 1000.0,
                    "loop_total_ms": (t_after_command_arm - tick_start) * 1000.0,
                }
            )

            policy_step += 1
            if args.max_policy_steps > 0 and policy_step >= args.max_policy_steps:
                print(f"[DeployCollect] Reached max policy steps: {args.max_policy_steps}")
                break

    except KeyboardInterrupt:
        print("\n[DeployCollect] Received KeyboardInterrupt, exiting...")
    finally:
        if collector is not None:
            collector.save_statistics()
        profile_dir = collector.save_dir if collector is not None else None
        profiler.save(profile_dir)
        if not args.no_reset_on_exit:
            reset_target = (
                np.zeros(deploy.NUM_DOF, dtype=np.float32)
                if args.reset_target == "zero"
                else prev_target_policy
            )
            print(f"[DeployCollect] Resetting DexH13 hand to {args.reset_target} target...")
            reset_until = time.perf_counter() + max(0.0, args.reset_hold_sec)
            try:
                while time.perf_counter() < reset_until:
                    controller.send_joint_target(reset_target)
                    time.sleep(0.05)
                controller.send_joint_target(reset_target)
            except Exception as exc:
                print(f"[DeployCollect] Reset failed: {exc}")
        arm_reader.close()
        wrist_reader.close()
        controller.close()


if __name__ == "__main__":
    main()
