import argparse
import sys
from pathlib import Path

# IsaacGym requires being imported before PyTorch.
import isaacgym  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

from isaacgym import gymapi
from isaacgym import gymtorch
from isaacgym.torch_utils import quat_apply, quat_from_euler_xyz, quat_mul

from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexscrew.tasks import isaacgym_task_map


def _parse_csv_floats(text: str, expected: int) -> List[float]:
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != expected:
        raise ValueError(f'Expected {expected} comma-separated floats, got {len(parts)}: {text!r}')
    return [float(p) for p in parts]


def _parse_csv_ints(text: str) -> List[int]:
    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(',') if p.strip()]
    return [int(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='IsaacGym RBHand<->lightbulb wrist-twist sanity test')
    parser.add_argument('--task_config', type=str, required=True, help='Path to a task YAML under configs/task/')
    parser.add_argument('--sim_device', type=str, default='cuda:0')
    parser.add_argument('--graphics_device_id', type=int, default=0)
    parser.add_argument('--headless', action='store_true', help='Disable viewer rendering')
    parser.add_argument('--num_envs', type=int, default=1, help='Override env.numEnvs from config')
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=200)

    # Controller stabilization (helps reduce micro jitter)
    parser.add_argument('--pd_pgain', type=float, default=None, help='Override env.controller.pgain (torque PD)')
    parser.add_argument('--pd_dgain', type=float, default=0.05, help='Override env.controller.dgain (torque PD)')

    # Viewer helpers (make rotation easier to see)
    parser.add_argument('--draw_axes', action='store_true', help='Enable a local coordinate frame in the viewer')
    parser.add_argument('--axis_length', type=float, default=0.08, help='Axis length (meters)')
    parser.add_argument('--axis_thickness', type=float, default=0.01, help='Axis thickness (meters)')
    parser.add_argument(
        '--axes_frame',
        type=str,
        default='nut',
        choices=['nut', 'object'],
        help='Which frame to draw axes on (nut shows bulb rotation; object is the actor root).',
    )

    # Contact material override (helps reduce slip)
    parser.add_argument('--shape_friction', type=float, default=4.0, help='Override rigid-shape friction for hand+object')
    parser.add_argument('--shape_restitution', type=float, default=0.0, help='Override rigid-shape restitution for hand+object')

    # Wrist motion (kinematic)
    parser.add_argument(
        '--yaw_deg',
        type=float,
        default=25.0,
        help='Wrist orbit yaw amplitude around +Z (degrees). Positive is CCW; negative is CW.',
    )
    parser.add_argument('--grasp_steps', type=int, default=40, help='Steps to close fingers before twisting')
    parser.add_argument('--yaw_steps', type=int, default=60, help='Steps to twist from 0 -> yaw_deg while grasping')
    parser.add_argument('--release_steps', type=int, default=40, help='Steps to open fingers at max yaw')
    parser.add_argument('--return_steps', type=int, default=60, help='Steps to return from yaw_deg -> 0 while released')
    parser.add_argument(
        '--pivot_mode',
        type=str,
        default='nut_pos',
        choices=['nut_pos', 'object_pos', 'fixed'],
        help='Pivot used as the bulb rotation axis center.',
    )
    parser.add_argument('--pivot', type=str, default='0,0,0.06', help='Used when pivot_mode=fixed, CSV xyz in meters')

    # Finger motion (targeted)
    parser.add_argument(
        '--active_dof_indices',
        type=str,
        default='3,4,5,6,7,8,9',
        help='Comma-separated DOF indices to control (others stay at reset pose).',
    )
    parser.add_argument(
        '--open_fraction',
        type=float,
        default=0.0,
        help='0=open_end (closest to 0), 1=close_end (largest |value|).',
    )
    parser.add_argument(
        '--close_fraction',
        type=float,
        default=0.9,
        help='0=open_end (closest to 0), 1=close_end (largest |value|).',
    )
    parser.add_argument(
        '--keep_random_forces',
        action='store_true',
        help='Keep env.random forces enabled (default: disabled for a clean test).',
    )
    parser.add_argument(
        '--keep_action_noise',
        action='store_true',
        help='Keep action noise enabled (default: disabled for a clean test).',
    )
    parser.add_argument(
        '--keep_pd_gain_rand',
        action='store_true',
        help='Keep per-episode random PD gains enabled (default: disabled for a clean test).',
    )
    parser.add_argument(
        '--keep_friction_rand',
        action='store_true',
        help='Keep random friction enabled (default: disabled; use --shape_friction instead).',
    )
    return parser.parse_args()


def _maybe_disable_rand(
    cfg: dict,
    *,
    keep_random_forces: bool,
    keep_action_noise: bool,
    keep_pd_gain_rand: bool,
    keep_friction_rand: bool,
) -> None:
    env_cfg = cfg.setdefault('env', {})
    if not keep_random_forces:
        env_cfg['forceScale'] = 0.0
        env_cfg['randomForceProbScalar'] = 0.0
    rand_cfg = env_cfg.setdefault('randomization', {})
    if not keep_action_noise:
        rand_cfg['action_noise_e_scale'] = 0.0
        rand_cfg['action_noise_t_scale'] = 0.0
    # Observations-only noise (doesn't affect physics), but keeping it off makes logs easier to read.
    rand_cfg['obs_noise_e_scale'] = 0.0
    rand_cfg['obs_noise_t_scale'] = 0.0
    rand_cfg['pose_noise_scale'] = 0.0
    rand_cfg['noisy_rpy_scale'] = 0.0
    rand_cfg['noisy_pos_scale'] = 0.0

    if not keep_pd_gain_rand:
        rand_cfg['randomizePDGains'] = False
    if not keep_friction_rand:
        rand_cfg['randomizeFriction'] = False


def _apply_shape_material(env, *, friction: float, restitution: float) -> None:
    if friction < 0.0:
        raise ValueError(f'friction must be >= 0, got {friction}')
    if restitution < 0.0:
        raise ValueError(f'restitution must be >= 0, got {restitution}')

    for env_ptr, hand_actor in zip(env.envs, env.hand_actors):
        # Hand
        hand_props = env.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
        for p in hand_props:
            p.friction = float(friction)
            p.restitution = float(restitution)
        env.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

        # Object (named "object" in xhand_hora)
        obj_actor = env.gym.find_actor_handle(env_ptr, 'object')
        obj_props = env.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        for p in obj_props:
            p.friction = float(friction)
            p.restitution = float(restitution)
        env.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_props)


def _get_pivot(env, pivot_mode: str, pivot_fixed_xyz: torch.Tensor) -> torch.Tensor:
    if pivot_mode == 'nut_pos':
        return env.nut_pos.clone()
    if pivot_mode == 'object_pos':
        return env.object_pos.clone()
    if pivot_mode == 'fixed':
        return pivot_fixed_xyz.view(1, 3).repeat(env.num_envs, 1)
    raise ValueError(f'Unknown pivot_mode: {pivot_mode}')


def _set_hand_root_pose(
    env,
    hand_actor_indices: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    *,
    linvel: torch.Tensor = None,
    angvel: torch.Tensor = None,
    extra_actor_indices: torch.Tensor = None,
) -> None:
    env.root_state_tensor[hand_actor_indices, 0:3] = pos
    env.root_state_tensor[hand_actor_indices, 3:7] = quat
    if linvel is None:
        env.root_state_tensor[hand_actor_indices, 7:10] = 0.0
    else:
        env.root_state_tensor[hand_actor_indices, 7:10] = linvel
    if angvel is None:
        env.root_state_tensor[hand_actor_indices, 10:13] = 0.0
    else:
        env.root_state_tensor[hand_actor_indices, 10:13] = angvel
    actor_indices = hand_actor_indices
    if extra_actor_indices is not None:
        actor_indices = torch.cat([hand_actor_indices, extra_actor_indices])
    env.gym.set_actor_root_state_tensor_indexed(
        env.sim,
        gymtorch.unwrap_tensor(env.root_state_tensor),
        gymtorch.unwrap_tensor(actor_indices),
        int(actor_indices.numel()),
    )


def main() -> None:
    args = parse_args()

    task_path = Path(args.task_config)
    if not task_path.exists():
        raise FileNotFoundError(task_path)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = OmegaConf.load(str(task_path))
    cfg = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(cfg, dict) and 'name' in cfg

    cfg['env']['numEnvs'] = int(args.num_envs)
    _maybe_disable_rand(
        cfg,
        keep_random_forces=args.keep_random_forces,
        keep_action_noise=args.keep_action_noise,
        keep_pd_gain_rand=args.keep_pd_gain_rand,
        keep_friction_rand=args.keep_friction_rand,
    )
    if args.pd_pgain is not None:
        cfg.setdefault('env', {}).setdefault('controller', {})['pgain'] = float(args.pd_pgain)
    if args.pd_dgain is not None:
        cfg.setdefault('env', {}).setdefault('controller', {})['dgain'] = float(args.pd_dgain)

    if args.draw_axes:
        axis_len = float(args.axis_length)
        axis_th = float(args.axis_thickness)
        if axis_len <= 0.0:
            raise ValueError(f'--axis_length must be > 0, got {axis_len}')
        if axis_th <= 0.0:
            raise ValueError(f'--axis_thickness must be > 0, got {axis_th}')
        viewer_cfg = cfg.setdefault('env', {}).setdefault('viewer', {})
        viewer_cfg['draw_axes'] = True
        viewer_cfg['axes_frame'] = str(args.axes_frame)
        viewer_cfg['axis_length'] = axis_len
        viewer_cfg['axis_thickness'] = axis_th

    env_cls = isaacgym_task_map[cfg['name']]
    env = env_cls(
        config=cfg,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=args.headless,
    )

    device = torch.device(env.device)
    active_dof_indices = _parse_csv_ints(args.active_dof_indices)
    if not active_dof_indices:
        raise ValueError('--active_dof_indices is empty; please provide indices like "3,4,5,6,7,8,9"')
    if not (0.0 <= args.open_fraction <= 1.0) or not (0.0 <= args.close_fraction <= 1.0):
        raise ValueError('--open_fraction/--close_fraction must be in [0, 1]')
    if args.grasp_steps < 0 or args.yaw_steps < 0 or args.release_steps < 0 or args.return_steps < 0:
        raise ValueError('Steps must be non-negative')

    obs = env.reset()

    # Make contact more stable / less slippery for debugging.
    _apply_shape_material(env, friction=float(args.shape_friction), restitution=float(args.shape_restitution))

    # Resolve controlled DOF limits
    dof_names = env.gym.get_asset_dof_names(env.hand_asset)
    dof_props = env.gym.get_asset_dof_properties(env.hand_asset)
    num_hand_dofs = env.num_xhand_hand_dofs
    lower = torch.tensor(dof_props['lower'][:num_hand_dofs], dtype=torch.float, device=device)
    upper = torch.tensor(dof_props['upper'][:num_hand_dofs], dtype=torch.float, device=device)

    for idx in active_dof_indices:
        if idx < 0 or idx >= num_hand_dofs:
            raise ValueError(f'active_dof_indices contains out-of-range index {idx} (num_hand_dofs={num_hand_dofs})')

    # Define "open_end" as the end closer to 0 (smaller abs), "close_end" as the farther end.
    close_end = torch.where(torch.abs(lower) > torch.abs(upper), lower, upper)
    open_end = torch.where(torch.abs(lower) > torch.abs(upper), upper, lower)

    base_targets = env.prev_targets[:, :num_hand_dofs].clone()
    open_targets = base_targets.clone()
    close_targets = base_targets.clone()

    # Only control selected DOFs; keep the rest at reset pose.
    for idx in active_dof_indices:
        o = open_end[idx] + float(args.open_fraction) * (close_end[idx] - open_end[idx])
        c = open_end[idx] + float(args.close_fraction) * (close_end[idx] - open_end[idx])
        open_targets[:, idx] = o
        close_targets[:, idx] = c

    # Kinematic wrist orbit setup
    hand_actor_indices = torch.tensor(env.hand_indices, dtype=torch.int32, device=device)
    # IsaacGym GPU pipeline quirk: if we only update the hand root state tensor each frame, fixed-base articulated
    # objects may snap back to their creation pose on the next simulate(). Include object actor indices in the same
    # indexed set to keep the object pinned.
    object_actor_indices = torch.tensor(env.object_indices, dtype=torch.int32, device=device)
    pivot_fixed_xyz = torch.tensor(_parse_csv_floats(args.pivot, 3), dtype=torch.float, device=device)
    pivot = _get_pivot(env, args.pivot_mode, pivot_fixed_xyz)

    base_pos0 = env.root_state_tensor[hand_actor_indices, 0:3].clone()
    base_quat0 = env.root_state_tensor[hand_actor_indices, 3:7].clone()
    offset0 = base_pos0 - pivot

    yaw_rad = float(args.yaw_deg) * (np.pi / 180.0)
    cycle_len = args.grasp_steps + args.yaw_steps + args.release_steps + args.return_steps
    if cycle_len <= 0:
        raise ValueError('Cycle length is 0; increase grasp/yaw/release/return steps')

    nut_pos0 = env.nut_dof_pos.clone() if hasattr(env, 'nut_dof_pos') else None

    def _ease01(frac: float) -> float:
        # cosine ease-in-out in [0,1], with zero slope at both ends
        frac = min(max(frac, 0.0), 1.0)
        return 0.5 * (1.0 - float(np.cos(np.pi * frac)))

    def _phase(step0: int) -> Tuple[str, float, torch.Tensor]:
        """Return (phase_name, yaw_now_rad, desired_targets)."""
        t = step0 % cycle_len
        if t < args.grasp_steps:
            if args.grasp_steps <= 0:
                return 'grasp', 0.0, close_targets
            frac = _ease01(float(t + 1) / float(args.grasp_steps))
            return 'grasp', 0.0, open_targets + frac * (close_targets - open_targets)
        t -= args.grasp_steps
        if t < args.yaw_steps:
            if args.yaw_steps <= 0:
                return 'twist', yaw_rad, close_targets
            frac = _ease01(float(t + 1) / float(args.yaw_steps))
            return 'twist', yaw_rad * frac, close_targets
        t -= args.yaw_steps
        if t < args.release_steps:
            if args.release_steps <= 0:
                return 'release', yaw_rad, open_targets
            frac = _ease01(float(t + 1) / float(args.release_steps))
            return 'release', yaw_rad, close_targets + frac * (open_targets - close_targets)
        t -= args.release_steps
        if args.return_steps <= 0:
            return 'return', 0.0, open_targets
        frac = _ease01(float(t + 1) / float(args.return_steps))
        return 'return', yaw_rad * (1.0 - frac), open_targets

    # Patch update_low_level_control to smoothly move the wrist at physics substeps.
    orig_update_low_level_control = env.update_low_level_control

    env._wrist_interp_enabled = True
    env._wrist_hand_actor_indices = hand_actor_indices
    env._wrist_object_actor_indices = object_actor_indices
    env._wrist_base_quat0 = base_quat0
    env._wrist_offset0 = offset0
    env._wrist_pivot = pivot
    env._wrist_yaw_prev = torch.zeros((env.num_envs,), device=device, dtype=torch.float)
    env._wrist_yaw_now = torch.zeros((env.num_envs,), device=device, dtype=torch.float)

    def _patched_update_low_level_control(step_id: int):
        if getattr(env, '_wrist_interp_enabled', False):
            yaw_prev_t = env._wrist_yaw_prev
            yaw_now_t = env._wrist_yaw_now
            # Interpolate across control substeps.
            alpha = float(step_id + 1) / float(env.control_freq_inv)
            yaw_t = yaw_prev_t + (yaw_now_t - yaw_prev_t) * alpha
            zeros = torch.zeros_like(yaw_t)
            yaw_quat = quat_from_euler_xyz(zeros, zeros, yaw_t)

            r = quat_apply(yaw_quat, env._wrist_offset0)
            pos_t = env._wrist_pivot + r
            quat_t = quat_mul(yaw_quat, env._wrist_base_quat0)

            # Set consistent root velocities to reduce teleport "jerk" at contacts.
            dt_action = float(env.control_freq_inv) * float(env.dt)
            omega = (yaw_now_t - yaw_prev_t) / max(dt_action, 1e-6)  # rad/s
            linvel = torch.stack([-omega * r[:, 1], omega * r[:, 0], zeros], dim=1)
            angvel = torch.stack([zeros, zeros, omega], dim=1)

            _set_hand_root_pose(
                env,
                env._wrist_hand_actor_indices,
                pos_t,
                quat_t,
                linvel=linvel,
                angvel=angvel,
                extra_actor_indices=env._wrist_object_actor_indices,
            )

        return orig_update_low_level_control(step_id)

    env.update_low_level_control = _patched_update_low_level_control

    try:
        yaw_prev = torch.zeros((env.num_envs,), device=device, dtype=torch.float)
        for step in range(1, args.steps + 1):
            phase_name, yaw_now, desired = _phase(step0=step - 1)

            # Update pivot (optional): keep it tied to the actual nut position to stay centered.
            pivot = _get_pivot(env, args.pivot_mode, pivot_fixed_xyz)
            env._wrist_pivot = pivot

            yaw_now_t = torch.ones((env.num_envs,), device=device, dtype=torch.float) * float(yaw_now)
            env._wrist_yaw_prev = yaw_prev
            env._wrist_yaw_now = yaw_now_t

            # Finger targets -> delta action
            actions = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float)
            delta = desired - env.prev_targets[:, :num_hand_dofs]
            for idx in active_dof_indices:
                actions[:, idx] = delta[:, idx] / float(env.action_scale)
            actions = torch.clamp(actions, -1.0, 1.0)

            obs, rew, resets, info = env.step(actions)
            yaw_prev = yaw_now_t

            done = resets.to(device).bool()
            if done.any():
                done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                # Re-snapshot the base pose and pivot after env.reset_idx() ran internally.
                pivot[done_env_ids] = _get_pivot(env, args.pivot_mode, pivot_fixed_xyz)[done_env_ids]
                done_hand_actor_indices = hand_actor_indices[done_env_ids]
                base_pos0[done_env_ids] = env.root_state_tensor[done_hand_actor_indices, 0:3]
                base_quat0[done_env_ids] = env.root_state_tensor[done_hand_actor_indices, 3:7]
                offset0[done_env_ids] = base_pos0[done_env_ids] - pivot[done_env_ids]
                yaw_prev[done_env_ids] = 0.0
                if nut_pos0 is not None:
                    nut_pos0[done_env_ids] = env.nut_dof_pos[done_env_ids]

            if (step % args.print_every) == 0:
                msg = f'[wrist-test] step={step} phase={phase_name} yaw_deg={yaw_now * 180.0 / np.pi:.1f}'
                if hasattr(env, 'nut_dof_pos'):
                    nut_pos = env.nut_dof_pos
                    msg += f' | nut_pos(mean)={float(nut_pos.mean().item()):.4f}'
                    if nut_pos0 is not None:
                        msg += f' | nut_pos(delta_mean)={float((nut_pos - nut_pos0).mean().item()):.4f}'
                if hasattr(env, 'nut_dof_vel_cf'):
                    nut_vel = env.nut_dof_vel_cf
                    msg += f' | nut_vel(mean)={float(nut_vel.mean().item()):.4f}'
                msg += f' | resets={int(done.sum().item())}'
                print(msg)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
