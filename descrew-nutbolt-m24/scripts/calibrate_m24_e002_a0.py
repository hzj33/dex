#!/usr/bin/env python3
"""GPU-parallel E002-A0b/A0c reset-pose and drive-sign calibration.

This is a deterministic pre-PPO gate. It searches bounded index/thumb joint
offsets around the saved M24 pose, requires stable simultaneous contact, then
probes bounded index/thumb actions and both resultant-force sign conventions.
It exits non-zero unless a stable reset and a physically observed two-finger
positive-drive candidate both exist.
"""

import json
import math
import os
import sys
from pathlib import Path

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401
from isaacgym import gymtorch

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict


OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)

ACTIVE_DOFS = tuple(range(0, 4)) + tuple(range(12, 16))


def _hand_joint_limits(env, dtype):
    """Return task-owned NumPy limits as device tensors for pose operations."""
    lower = torch.as_tensor(
        env.xhand_dof_lower_limits, device=env.device, dtype=dtype
    )
    upper = torch.as_tensor(
        env.xhand_dof_upper_limits, device=env.device, dtype=dtype
    )
    return lower, upper


def _set_hand_poses(env, poses):
    """Install one absolute 16-DOF hand pose per environment."""
    poses = poses.to(device=env.device, dtype=env.xhand_hand_dof_pos.dtype)
    if poses.shape != (env.num_envs, env.num_xhand_hand_dofs):
        raise ValueError(
            f"pose batch must be {(env.num_envs, env.num_xhand_hand_dofs)}, "
            f"got {tuple(poses.shape)}"
        )
    lower, upper = _hand_joint_limits(env, poses.dtype)
    poses = torch.maximum(torch.minimum(poses, upper), lower)
    env.xhand_hand_dof_pos[:] = poses
    env.xhand_hand_dof_vel.zero_()
    env.prev_targets[:, : env.num_xhand_hand_dofs] = poses
    env.cur_targets[:, : env.num_xhand_hand_dofs] = poses
    env.init_pose_buf[:, : env.num_xhand_hand_dofs] = poses
    hand_indices = env.hand_indices.to(dtype=torch.int32)
    env.gym.set_dof_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.dof_state),
        gymtorch.unwrap_tensor(hand_indices),
        len(hand_indices),
    )
    env.at_reset_buf[:] = 1
    env.e002_contact_state.zero_()
    env.e002_pair_contact_streak.zero_()
    env.e002_pair_contact_loss_streak.zero_()
    env.e002_contact_acquired.zero_()


def _pair_measurements(env):
    handles = [env.fingertip_handles[0], env.fingertip_handles[3]]
    pair_pos = env.rigid_body_states[:, handles, :3]
    pair_force = env.contact_forces[:, handles, :]
    nut_pos = env.rigid_body_states[:, env.screw_nut_rb_handle, :3]
    radial = pair_pos - nut_pos.unsqueeze(1)
    force_mag = torch.norm(pair_force, dim=-1)
    distance = torch.norm(radial, dim=-1)
    raw_torque_vec = torch.cross(radial, pair_force, dim=-1)
    if env.e002_torque_source == "contact_point_normal":
        raw_torque_vec = -env.e002_normal_contact_moment
    raw_axis_torque = (
        raw_torque_vec * env.rot_axis_buf.unsqueeze(1).expand_as(raw_torque_vec)
    ).sum(dim=-1)
    contact = (force_mag >= env.e002_contact_force_on) & (
        distance <= env.e002_contact_dist_on
    )
    return force_mag, distance, raw_axis_torque, contact


def _structured_rows(values, active_dim):
    rows = [torch.zeros(active_dim, device=values.device, dtype=values.dtype)]
    for magnitude in (0.25, 0.5, 1.0):
        for sign in (-1.0, 1.0):
            for i in range(active_dim):
                row = torch.zeros_like(rows[0])
                row[i] = sign * magnitude
                rows.append(row)
            rows.append(torch.full_like(rows[0], sign * magnitude))
            opposed = torch.empty_like(rows[0])
            opposed[:4] = sign * magnitude
            opposed[4:] = -sign * magnitude
            rows.append(opposed)
    return torch.stack(rows[: min(len(rows), values.shape[0])])


def _candidate_values(count, dim, seed, device):
    sobol = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    values = sobol.draw(count).to(device=device) * 2.0 - 1.0
    structured = _structured_rows(values, dim)
    values[: structured.shape[0]] = structured
    return values


def _write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


@hydra.main(config_name="config", config_path="../configs")
def main(cfg: DictConfig):
    if cfg.task_name != "XHandPasiniM24NutBolt":
        raise ValueError("calibration requires task=XHandPasiniM24NutBoltE002")
    if not bool(cfg.task.env.e002_motion.enable):
        raise ValueError("calibration requires env.e002_motion.enable=True")

    count = int(cfg.get("calibration_candidates", cfg.task.env.numEnvs))
    if count != int(cfg.task.env.numEnvs):
        raise ValueError("calibration_candidates must equal task.env.numEnvs")
    if count < 64:
        raise ValueError("calibration requires at least 64 parallel candidates")

    seed = int(cfg.get("calibration_seed", cfg.seed))
    pose_radius = float(cfg.get("calibration_pose_radius", 0.16))
    settle_steps = int(cfg.get("calibration_settle_steps", 6))
    hold_steps = int(cfg.get("calibration_hold_steps", 10))
    probe_steps = int(cfg.get("calibration_probe_steps", 10))
    probe_required_steps = int(cfg.get("calibration_probe_required_steps", 5))
    action_amplitude = float(cfg.get("calibration_action_amplitude", 0.5))
    min_probe_angle = float(cfg.get("calibration_min_probe_angle", math.pi / 180.0))
    max_hold_drift = float(cfg.get("calibration_max_hold_drift", math.pi / 180.0))
    output = Path(
        str(cfg.get("calibration_out", "outputs/e002_a0_calibration.json"))
    )
    if min(settle_steps, hold_steps, probe_steps, probe_required_steps) < 1:
        raise ValueError("calibration step counts must be positive")
    if pose_radius <= 0.0 or not (0.0 < action_amplitude <= 1.0):
        raise ValueError("invalid calibration pose radius or action amplitude")

    torch.manual_seed(seed)
    env = isaacgym_task_map[cfg.task_name](
        config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )
    # Calibration owns reset decisions; ordinary E002 terminal conditions must
    # not silently replace a candidate midway through a measurement window.
    env.e002_contact_acquire_deadline_steps = 1_000_000
    env.e002_contact_loss_grace_steps = 1_000_000
    env.e002_drive_deadline_steps = 1_000_000
    env.e002_workspace_radius = 1.0
    env.e002_hard_step_angle = 100.0
    env.e002_target_angle = 100.0
    env.e002_overshoot_angle = 100.0

    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    zero_actions = torch.zeros((env.num_envs, env.num_acts), device=env.rl_device)
    base_pose = torch.tensor(
        env.joint_values_lst, device=env.device, dtype=env.xhand_hand_dof_pos.dtype
    )
    pose_values = _candidate_values(count, len(ACTIVE_DOFS), seed, env.device)
    pose_batch = base_pose.unsqueeze(0).repeat(count, 1)
    active = torch.tensor(ACTIVE_DOFS, device=env.device, dtype=torch.long)
    pose_batch[:, active] += pose_radius * pose_values
    lower, upper = _hand_joint_limits(env, pose_batch.dtype)
    pose_batch = torch.maximum(torch.minimum(pose_batch, upper), lower)

    env.reset_idx(all_ids)
    _set_hand_poses(env, pose_batch)
    for _ in range(settle_steps):
        env.step(zero_actions)
    theta_hold_start = env.nut_dof_pos[:, 0].clone()
    consecutive = torch.zeros(count, device=env.device, dtype=torch.long)
    max_consecutive = torch.zeros_like(consecutive)
    force_min = torch.full((count, 2), float("inf"), device=env.device)
    force_max = torch.zeros((count, 2), device=env.device)
    distance_max = torch.zeros((count, 2), device=env.device)
    for _ in range(hold_steps):
        env.step(zero_actions)
        force, distance, _, contact = _pair_measurements(env)
        pair_contact = contact.all(dim=-1)
        consecutive = torch.where(pair_contact, consecutive + 1, torch.zeros_like(consecutive))
        max_consecutive = torch.maximum(max_consecutive, consecutive)
        force_min = torch.minimum(force_min, force)
        force_max = torch.maximum(force_max, force)
        distance_max = torch.maximum(distance_max, distance)
    hold_drift = torch.abs(env.nut_dof_pos[:, 0] - theta_hold_start)
    pose_ok = (max_consecutive >= hold_steps) & (hold_drift <= max_hold_drift)
    pose_norm = torch.norm(pose_batch[:, active] - base_pose[active], dim=-1)
    pose_score = pose_norm + 10.0 * hold_drift
    pose_score = torch.where(pose_ok, pose_score, torch.full_like(pose_score, float("inf")))
    pose_index = int(torch.argmin(pose_score).item())
    pose_passed = bool(pose_ok[pose_index].item()) and math.isfinite(
        float(pose_score[pose_index].item())
    )
    if not pose_passed:
        report = {
            "status": "failed:no_stable_pair_reset",
            "candidate_count": count,
            "best_contact_streak": int(max_consecutive.max().item()),
            "required_contact_streak": hold_steps,
            "pose_radius": pose_radius,
        }
        _write_report(output, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(3)

    selected_pose = pose_batch[pose_index].clone()

    # A0c: broadcast the selected reset, then probe bounded independent
    # index/thumb actions and both force-on-nut sign conventions.
    env.reset_idx(all_ids)
    selected_batch = selected_pose.unsqueeze(0).repeat(count, 1)
    _set_hand_poses(env, selected_batch)
    for _ in range(settle_steps):
        env.step(zero_actions)
    theta_probe_start = env.nut_dof_pos[:, 0].clone()
    theta_previous = theta_probe_start.clone()
    active_actions = _candidate_values(count, len(ACTIVE_DOFS), seed + 1, env.device)
    actions = torch.zeros_like(zero_actions)
    actions[:, active] = action_amplitude * active_actions
    signs = torch.tensor([-1.0, 1.0], device=env.device)
    streak = torch.zeros((count, 2), device=env.device, dtype=torch.long)
    max_streak = torch.zeros_like(streak)
    contact_steps = torch.zeros(count, device=env.device, dtype=torch.long)
    torque_sum = torch.zeros((count, 2, 2), device=env.device)
    pair_gate_sum = torch.zeros((count, 2), device=env.device)
    for _ in range(probe_steps):
        env.step(actions)
        force, distance, raw_torque, contact = _pair_measurements(env)
        theta = env.nut_dof_pos[:, 0].clone()
        directed_step = env.e002_direction * (theta - theta_previous)
        theta_previous = theta
        pair_contact = contact.all(dim=-1)
        contact_steps += pair_contact.long()
        signed = (
            env.e002_direction
            * raw_torque.unsqueeze(1)
            * signs.view(1, 2, 1)
        )
        torque_sum += signed
        force_conf = torch.clamp(
            (force - env.e002_contact_force_off)
            / (env.e002_contact_force_on - env.e002_contact_force_off),
            min=0.0,
            max=1.0,
        )
        distance_conf = torch.clamp(
            (env.e002_contact_dist_off - distance)
            / (env.e002_contact_dist_off - env.e002_contact_dist_on),
            min=0.0,
            max=1.0,
        )
        contact_conf = torch.minimum(force_conf, distance_conf).unsqueeze(1)
        positive_torque = torch.clamp(signed, min=0.0)
        torque_conf = torch.clamp(
            positive_torque / env.e002_torque_saturation, min=0.0, max=1.0
        )
        torque_total = positive_torque.sum(dim=-1)
        torque_balance = torch.where(
            torque_total > 0.0,
            2.0
            * torch.minimum(positive_torque[..., 0], positive_torque[..., 1])
            / torch.clamp(torque_total, min=1e-6),
            torch.zeros_like(torque_total),
        )
        pair_gate = (
            torch.minimum(
                (contact_conf * torque_conf)[..., 0],
                (contact_conf * torque_conf)[..., 1],
            )
            * torque_balance
        )
        pair_gate_sum += pair_gate
        torque_ok = (signed >= env.e002_torque_min).all(dim=-1)
        step_ok = (
            pair_contact.unsqueeze(-1)
            & torque_ok
            & (pair_gate >= env.e002_pair_gate_min)
            & (directed_step > 0.0).unsqueeze(-1)
        )
        streak = torch.where(step_ok, streak + 1, torch.zeros_like(streak))
        max_streak = torch.maximum(max_streak, streak)
    directed_delta = env.e002_direction * (env.nut_dof_pos[:, 0] - theta_probe_start)
    action_ok = (max_streak >= probe_required_steps) & (
        directed_delta.unsqueeze(-1) >= min_probe_angle
    )
    action_norm = torch.norm(actions[:, active], dim=-1).unsqueeze(-1)
    action_score = (
        max_streak.float() * 100.0
        + directed_delta.unsqueeze(-1) * 10.0
        - action_norm * 0.01
    )
    action_score = torch.where(
        action_ok, action_score, torch.full_like(action_score, float("-inf"))
    )
    flat_index = int(torch.argmax(action_score.reshape(-1)).item())
    action_index, sign_index = divmod(flat_index, 2)
    action_passed = bool(action_ok[action_index, sign_index].item()) and math.isfinite(
        float(action_score[action_index, sign_index].item())
    )
    if not action_passed:
        report = {
            "status": "failed:no_pair_positive_drive",
            "candidate_count": count,
            "pose_index": pose_index,
            "custom_init_dof_pos": selected_pose.detach().cpu().tolist(),
            "best_probe_streak": int(max_streak.max().item()),
            "required_probe_streak": probe_required_steps,
            "best_directed_delta": float(directed_delta.max().item()),
            "best_mean_pair_gate": float(
                (pair_gate_sum / float(probe_steps)).max().item()
            ),
        }
        _write_report(output, report)
        print(json.dumps(report, indent=2))
        raise SystemExit(4)

    selected_action = actions[action_index].detach().cpu().tolist()
    selected_sign = float(signs[sign_index].item())
    report = {
        "status": "passed",
        "candidate_count": count,
        "seed": seed,
        "active_dofs": list(ACTIVE_DOFS),
        "pose_radius": pose_radius,
        "pose_index": pose_index,
        "custom_init_dof_pos": selected_pose.detach().cpu().tolist(),
        "pose_offset_l2": float(pose_norm[pose_index].item()),
        "pose_hold_contact_streak": int(max_consecutive[pose_index].item()),
        "pose_hold_nut_drift": float(hold_drift[pose_index].item()),
        "pose_force_min": force_min[pose_index].detach().cpu().tolist(),
        "pose_force_max": force_max[pose_index].detach().cpu().tolist(),
        "pose_distance_max": distance_max[pose_index].detach().cpu().tolist(),
        "force_sign": selected_sign,
        "drive_action": selected_action,
        "drive_action_index": action_index,
        "drive_probe_streak": int(max_streak[action_index, sign_index].item()),
        "drive_probe_contact_steps": int(contact_steps[action_index].item()),
        "drive_probe_directed_delta": float(directed_delta[action_index].item()),
        "drive_probe_mean_signed_torque": (
            torque_sum[action_index, sign_index] / float(probe_steps)
        ).detach().cpu().tolist(),
        "drive_probe_mean_pair_gate": float(
            (pair_gate_sum[action_index, sign_index] / float(probe_steps)).item()
        ),
        "hydra_overrides": {
            "task.env.customInitDofPos": selected_pose.detach().cpu().tolist(),
            "task.env.e002_motion.force_sign": selected_sign,
        },
    }
    _write_report(output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
