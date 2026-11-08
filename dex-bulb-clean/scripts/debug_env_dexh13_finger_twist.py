import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# IsaacGym 要求在 torch 之前 import。
import isaacgym  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexscrew.tasks import isaacgym_task_map


def _parse_csv_floats(text: str, expected: int) -> List[float]:
    parts = [p.strip() for p in text.split(',') if p.strip()]
    if len(parts) != expected:
        raise ValueError(f'期望 {expected} 个逗号分隔浮点数，实际 {len(parts)} 个：{text!r}')
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='DexH13（手腕固定）指驱旋拧灯泡的开环小测试')
    parser.add_argument('--task_config', type=str, default='configs/task/Dexh13HoraLightbulb.yaml')
    parser.add_argument('--sim_device', type=str, default='cuda:0')
    parser.add_argument('--graphics_device_id', type=int, default=0)
    parser.add_argument('--headless', action='store_true', help='不打开 viewer')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=100)

    # 稳定性/可控性：默认关掉随机化，给一个更“干净”的开环测试
    parser.add_argument('--keep_random_forces', action='store_true', help='保留随机外力（默认关闭）')
    parser.add_argument('--keep_action_noise', action='store_true', help='保留动作噪声（默认关闭）')
    parser.add_argument('--keep_pd_gain_rand', action='store_true', help='保留 PD 增益随机化（默认关闭）')
    parser.add_argument('--keep_friction_rand', action='store_true', help='保留摩擦随机化（默认关闭）')

    # 接触材料：提高摩擦更容易“带动”灯泡转动（只是用于 demo，不建议直接用于训练配置）
    parser.add_argument('--shape_friction', type=float, default=4.0)
    parser.add_argument('--shape_restitution', type=float, default=0.0)

    # PD 控制（torque_control 模式下生效）
    parser.add_argument('--pd_pgain', type=float, default=None, help='覆盖 env.controller.pgain')
    parser.add_argument('--pd_dgain', type=float, default=0.05, help='覆盖 env.controller.dgain')

    # 手根姿态覆盖（便于你不改 YAML 就能快速微调）
    parser.add_argument('--hand_root_pos', type=str, default='', help='CSV xyz（米），如：0.12,0.02,0.10')
    parser.add_argument('--hand_root_rpy', type=str, default='', help='CSV rpy（弧度），如：3.1416,0,3.1416')

    # 开环动作周期：抓取 -> 扭转 -> 松开
    parser.add_argument('--grasp_steps', type=int, default=80)
    parser.add_argument('--twist_steps', type=int, default=240)
    parser.add_argument('--release_steps', type=int, default=80)

    # 抓取目标（按关节上下限插值）
    parser.add_argument('--open_fraction', type=float, default=0.0, help='0=最开（靠近 0），1=最闭（远离 0）')
    parser.add_argument('--close_fraction', type=float, default=0.85, help='抓取闭合程度（0~1）')
    parser.add_argument(
        '--thumb_j1_bias',
        type=float,
        default=0.0,
        help='拇指横摆根关节（right_thumb_joint_1）常量偏置（rad），控制靠拢灯泡',
    )
    parser.add_argument(
        '--thumb_open_fraction',
        type=float,
        default=-1.0,
        help='拇指弯曲关节（right_thumb_joint_2/3）张开程度（0~1）；设为 -1 表示沿用 --open_fraction',
    )
    parser.add_argument(
        '--thumb_close_fraction',
        type=float,
        default=-1.0,
        help='拇指弯曲关节（right_thumb_joint_2/3）闭合程度（0~1）；设为 -1 表示沿用 --close_fraction',
    )

    # 扭转幅度（只对 *_joint_0 生效，单位 rad）
    parser.add_argument('--twist_amp', type=float, default=0.25)
    parser.add_argument(
        '--thumb_root_bias',
        type=float,
        default=-0.2,
        help='拇指根（right_thumb_joint_0）常量偏置（rad），用于让拇指根对准灯泡',
    )
    parser.add_argument(
        '--thumb_twist_amp',
        type=float,
        default=float('nan'),
        help='拇指根扭转幅度（rad）。默认 NaN 表示使用 -twist_amp（保持旧行为）；设为 0 可让拇指根保持不动。',
    )
    parser.add_argument('--twist_period', type=int, default=60, help='正弦周期（步数）')
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

    # 观测噪声不影响物理，但会影响 debug 读数；这里默认也关掉
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
        raise ValueError(f'friction 必须 >= 0，当前 {friction}')
    if restitution < 0.0:
        raise ValueError(f'restitution 必须 >= 0，当前 {restitution}')

    for env_ptr, hand_actor in zip(env.envs, env.hand_actors):
        hand_props = env.gym.get_actor_rigid_shape_properties(env_ptr, hand_actor)
        for p in hand_props:
            p.friction = float(friction)
            p.restitution = float(restitution)
        env.gym.set_actor_rigid_shape_properties(env_ptr, hand_actor, hand_props)

        obj_actor = env.gym.find_actor_handle(env_ptr, 'object')
        obj_props = env.gym.get_actor_rigid_shape_properties(env_ptr, obj_actor)
        for p in obj_props:
            p.friction = float(friction)
            p.restitution = float(restitution)
        env.gym.set_actor_rigid_shape_properties(env_ptr, obj_actor, obj_props)


def _ease01(frac: float) -> float:
    frac = min(max(frac, 0.0), 1.0)
    return 0.5 * (1.0 - float(np.cos(np.pi * frac)))


def _resolve_fraction(value: float, *, fallback: float, name: str) -> float:
    """解析 fraction：允许 [0,1] 或 -1（表示沿用 fallback）。"""
    # 允许用户传 -1 / -1.0 / -1.0000
    if abs(value + 1.0) < 1e-9:
        return float(fallback)
    if 0.0 <= value <= 1.0:
        return float(value)
    raise ValueError(f'{name} 必须在 [0,1] 或 -1（表示沿用默认/全局），当前：{value}')


def _resolve_dof_indices(env) -> Tuple[List[int], List[int], List[int], int, int, int, int, List[str]]:
    """返回 (flex_indices, twist_indices, thumb_flex_indices, thumb_twist_index, thumb_j1_index, thumb_j2_index, thumb_j3_index, dof_names)。

    拇指分三组：
      - joint_0 → twist（侧摆，由 thumb_root_bias 控制）
      - joint_1 → 独立偏置（横摆根关节，由 thumb_j1_bias 控制靠拢灯泡）
      - joint_2/3 → flex（弯曲，由 thumb_flex_*_fraction 比例控制）
    """
    dof_names = list(env.gym.get_asset_dof_names(env.hand_asset))
    name_to_idx = {n: i for i, n in enumerate(dof_names)}

    # 四指 joint_0：侧摆（扭转阶段正弦摆动）
    twist_names = [
        'right_index_joint_0',
        'right_middle_joint_0',
        'right_ring_joint_0',
        'right_thumb_joint_0',
    ]
    # 弯曲关节：index/middle/ring 的 joint_1/2/3 + 拇指的 joint_2/3
    flex_names = []
    for finger in ('index', 'middle', 'ring'):
        for j in (1, 2, 3):
            flex_names.append(f'right_{finger}_joint_{j}')
    for j in (2, 3):
        flex_names.append(f'right_thumb_joint_{j}')

    thumb_flex_names = [f'right_thumb_joint_{j}' for j in (2, 3)]
    thumb_j1_name = 'right_thumb_joint_1'

    all_expected = twist_names + flex_names + [thumb_j1_name]
    missing = [n for n in all_expected if n not in name_to_idx]
    if missing:
        preview = dof_names[:50]
        raise ValueError(
            f'DexH13 关节名缺失：{missing}。'
            f'可用 DOF（前 {len(preview)}/{len(dof_names)} 个）：{preview}'
        )

    flex_indices = [int(name_to_idx[n]) for n in flex_names]
    twist_indices = [int(name_to_idx[n]) for n in twist_names]
    thumb_flex_indices = [int(name_to_idx[n]) for n in thumb_flex_names]
    thumb_twist_index = int(name_to_idx['right_thumb_joint_0'])
    thumb_j1_index = int(name_to_idx[thumb_j1_name])
    thumb_j2_index = int(name_to_idx['right_thumb_joint_2'])
    thumb_j3_index = int(name_to_idx['right_thumb_joint_3'])
    return (
        flex_indices,
        twist_indices,
        thumb_flex_indices,
        thumb_twist_index,
        thumb_j1_index,
        thumb_j2_index,
        thumb_j3_index,
        dof_names,
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

    cfg.setdefault('env', {})['numEnvs'] = int(args.num_envs)
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

    if args.hand_root_pos.strip():
        cfg.setdefault('env', {}).setdefault('asset', {})['handRootPos'] = _parse_csv_floats(args.hand_root_pos, 3)
    if args.hand_root_rpy.strip():
        cfg.setdefault('env', {}).setdefault('asset', {})['handRootRPY'] = _parse_csv_floats(args.hand_root_rpy, 3)

    env_cls = isaacgym_task_map[cfg['name']]
    env = env_cls(
        config=cfg,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=args.headless,
    )

    device = torch.device(env.device)
    env.reset()

    # 增大摩擦，减少打滑（demo 用）
    _apply_shape_material(env, friction=float(args.shape_friction), restitution=float(args.shape_restitution))

    (
        flex_indices,
        twist_indices,
        thumb_flex_indices,
        thumb_twist_index,
        thumb_j1_index,
        thumb_j2_index,
        thumb_j3_index,
        dof_names,
    ) = _resolve_dof_indices(env)

    if not (0.0 <= args.open_fraction <= 1.0) or not (0.0 <= args.close_fraction <= 1.0):
        raise ValueError('--open_fraction/--close_fraction 必须在 [0,1]')
    thumb_open_fraction = _resolve_fraction(
        float(args.thumb_open_fraction),
        fallback=float(args.open_fraction),
        name='--thumb_open_fraction',
    )
    thumb_close_fraction = _resolve_fraction(
        float(args.thumb_close_fraction),
        fallback=float(args.close_fraction),
        name='--thumb_close_fraction',
    )
    if args.twist_period <= 0:
        raise ValueError('--twist_period 必须 > 0')
    if args.grasp_steps < 0 or args.twist_steps < 0 or args.release_steps < 0:
        raise ValueError('grasp/twist/release steps 必须 >= 0')

    # 解析关节上下限（仅手部 DOF）
    dof_props = env.gym.get_asset_dof_properties(env.hand_asset)
    num_hand_dofs = int(env.num_xhand_hand_dofs)
    lower = torch.tensor(dof_props['lower'][:num_hand_dofs], dtype=torch.float, device=device)
    upper = torch.tensor(dof_props['upper'][:num_hand_dofs], dtype=torch.float, device=device)
    close_end = torch.where(torch.abs(lower) > torch.abs(upper), lower, upper)
    open_end = torch.where(torch.abs(lower) > torch.abs(upper), upper, lower)

    # 以 reset 时的 target 为基准，构造 open/close 两个目标
    base_targets = env.prev_targets[:, :num_hand_dofs].clone()
    open_targets = base_targets.clone()
    close_targets = base_targets.clone()

    thumb_flex_set = set(int(i) for i in thumb_flex_indices)
    for idx in flex_indices:
        cf = thumb_close_fraction if idx in thumb_flex_set else float(args.close_fraction)
        of = thumb_open_fraction if idx in thumb_flex_set else float(args.open_fraction)
        o = open_end[idx] + of * (close_end[idx] - open_end[idx])
        c = open_end[idx] + cf * (close_end[idx] - open_end[idx])
        open_targets[:, idx] = o
        close_targets[:, idx] = c

    # 侧摆关节（joint_0）默认回到 0，扭转阶段再加正弦项
    for idx in twist_indices:
        open_targets[:, idx] = 0.0
        close_targets[:, idx] = 0.0
    # 拇指根给一个常量偏置（更像"捏住灯泡"的自然姿态）
    open_targets[:, thumb_twist_index] = float(args.thumb_root_bias)
    close_targets[:, thumb_twist_index] = float(args.thumb_root_bias)
    # 拇指 joint_1（横摆根关节）：常量偏置靠拢灯泡
    open_targets[:, thumb_j1_index] = float(args.thumb_j1_bias)
    close_targets[:, thumb_j1_index] = float(args.thumb_j1_bias)

    print(
        '[dexh13-finger-twist] config: '
        f'OPEN_FRACTION={float(args.open_fraction):.3f} CLOSE_FRACTION={float(args.close_fraction):.3f} | '
        f'THUMB_FLEX_OPEN={thumb_open_fraction:.3f} THUMB_FLEX_CLOSE={thumb_close_fraction:.3f} | '
        f'THUMB_J1_BIAS={float(args.thumb_j1_bias):.3f} | '
        f'THUMB_ROOT_BIAS={float(args.thumb_root_bias):.3f} THUMB_TWIST_AMP={float(args.thumb_twist_amp)}'
    )

    cycle_len = args.grasp_steps + args.twist_steps + args.release_steps
    if cycle_len <= 0:
        raise ValueError('周期长度为 0，请增大 grasp/twist/release steps')

    nut_pos0 = env.nut_dof_pos.clone() if hasattr(env, 'nut_dof_pos') else None

    def _phase(step0: int) -> Tuple[str, torch.Tensor]:
        t = step0 % cycle_len
        if t < args.grasp_steps:
            if args.grasp_steps <= 0:
                return 'grasp', close_targets
            frac = _ease01(float(t + 1) / float(args.grasp_steps))
            return 'grasp', open_targets + frac * (close_targets - open_targets)
        t -= args.grasp_steps
        if t < args.twist_steps:
            desired = close_targets.clone()
            # 扭转：四个 joint_0 做正弦摆动；拇指反相，给一个“对拧”的趋势
            s = float(np.sin(2.0 * np.pi * float(t) / float(args.twist_period)))
            amp = float(args.twist_amp)
            # twist_indices 顺序：index/middle/ring/thumb
            desired[:, twist_indices[0]] = amp * s
            desired[:, twist_indices[1]] = amp * s
            desired[:, twist_indices[2]] = amp * s
            thumb_amp = float(args.thumb_twist_amp)
            if np.isnan(thumb_amp):
                thumb_amp = -amp  # 默认保持旧行为：拇指反相
            desired[:, twist_indices[3]] = float(args.thumb_root_bias) + thumb_amp * s
            return 'twist', desired
        t -= args.twist_steps
        if args.release_steps <= 0:
            return 'release', open_targets
        frac = _ease01(float(t + 1) / float(args.release_steps))
        return 'release', close_targets + frac * (open_targets - close_targets)

    try:
        for step in range(1, int(args.steps) + 1):
            phase_name, desired = _phase(step0=step - 1)

            # target -> delta action（只控制手部 DOF；其它维度保持 0）
            actions = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float)
            delta = desired - env.prev_targets[:, :num_hand_dofs]
            controlled = flex_indices + twist_indices + [thumb_j1_index]
            for idx in controlled:
                actions[:, idx] = delta[:, idx] / float(env.action_scale)
            actions = torch.clamp(actions, -1.0, 1.0)

            obs, rew, resets, info = env.step(actions)

            done = resets.to(device).bool()
            if done.any() and nut_pos0 is not None:
                done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)
                nut_pos0[done_env_ids] = env.nut_dof_pos[done_env_ids]

            if (step % int(args.print_every)) == 0:
                msg = f'[dexh13-finger-twist] step={step} phase={phase_name}'
                if hasattr(env, 'nut_dof_pos'):
                    nut_pos = env.nut_dof_pos
                    msg += f' | nut_pos(mean)={float(nut_pos.mean().item()):.4f}'
                    if nut_pos0 is not None:
                        msg += f' | nut_pos(delta_mean)={float((nut_pos - nut_pos0).mean().item()):.4f}'
                if hasattr(env, 'nut_dof_vel_cf'):
                    msg += f' | nut_vel(mean)={float(env.nut_dof_vel_cf.mean().item()):.4f}'
                if hasattr(env, 'nut_contact'):
                    msg += f' | nut_contact(mean)={float(env.nut_contact.mean().item()):.3f}'
                if hasattr(env, 'fingertip_states') and hasattr(env, 'nut_pos'):
                    tip_pos = env.fingertip_states[:, :, 0:3]
                    d = torch.norm(tip_pos - env.nut_pos[:, None, :], dim=-1)
                    msg += f' | tip_dist(min)={float(d.min().item()):.3f}'
                # 方便确认“拇指根 vs 拇指弯曲”是否按预期变化
                if hasattr(env, 'xhand_hand_dof_pos'):
                    dof_pos = env.xhand_hand_dof_pos
                    t0 = float(dof_pos[:, thumb_twist_index].mean().item())
                    t1 = float(dof_pos[:, thumb_j1_index].mean().item())
                    t2 = float(dof_pos[:, thumb_j2_index].mean().item())
                    t3 = float(dof_pos[:, thumb_j3_index].mean().item())
                    msg += f' | thumb(j0/j1/j2/j3)={t0:.2f}/{t1:.2f}/{t2:.2f}/{t3:.2f}'
                msg += f' | resets={int(done.sum().item())}'
                print(msg)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
