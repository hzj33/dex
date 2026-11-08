import argparse
import sys
from pathlib import Path

# IsaacGym requires being imported before PyTorch.
import isaacgym  # noqa: F401

import numpy as np
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexscrew.tasks import isaacgym_task_map


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='IsaacGym env debug runner (random actions)')
    parser.add_argument('--task_config', type=str, required=True, help='Path to a task YAML under configs/task/')
    parser.add_argument('--sim_device', type=str, default='cuda:0')
    parser.add_argument('--graphics_device_id', type=int, default=0)
    parser.add_argument('--headless', action='store_true', help='Disable viewer rendering')
    parser.add_argument('--num_envs', type=int, default=None, help='Override env.numEnvs from config')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_every', type=int, default=200)
    parser.add_argument('--action_mode', type=str, default='random', choices=['random', 'zero'])
    parser.add_argument('--action_scale', type=float, default=None, help='Override env.controller.action_scale')
    parser.add_argument('--print_dof_map', action='store_true', help='Print hand DOF name/index mapping and exit')
    parser.add_argument(
        '--random_action_indices',
        type=str,
        default='',
        help='Comma-separated action indices to randomize when action_mode=random (others are set to 0). Example: "1,2,3,4,5,6,7".',
    )
    return parser.parse_args()


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

    if args.num_envs is not None:
        cfg['env']['numEnvs'] = int(args.num_envs)
    if args.action_scale is not None:
        cfg.setdefault('env', {}).setdefault('controller', {})['action_scale'] = float(args.action_scale)

    env_cls = isaacgym_task_map[cfg['name']]
    env = env_cls(
        config=cfg,
        sim_device=args.sim_device,
        graphics_device_id=args.graphics_device_id,
        headless=args.headless,
    )

    if args.print_dof_map:
        dof_names = env.gym.get_asset_dof_names(env.hand_asset)
        dof_props = env.gym.get_asset_dof_properties(env.hand_asset)
        print('[debug] hand action index -> DOF name [lower, upper]')
        for i, name in enumerate(dof_names):
            lower = float(dof_props['lower'][i])
            upper = float(dof_props['upper'][i])
            print(f'  {i:2d}: {name} [{lower:.4f}, {upper:.4f}]')
        print(f'[debug] total hand actions: {len(dof_names)} (env.num_actions={env.num_actions})')
        print('[debug] NOTE: env.step pads +1 object DOF internally; you control only the hand actions above.')
        return

    obs = env.reset()
    device = torch.device(env.device)

    ep_count = torch.zeros(env.num_envs, device=device, dtype=torch.long)
    rew_sum = torch.zeros(env.num_envs, device=device, dtype=torch.float)

    try:
        for step in range(1, args.steps + 1):
            if args.action_mode == 'zero':
                actions = torch.zeros((env.num_envs, env.num_actions), device=device, dtype=torch.float)
            else:
                actions = torch.rand((env.num_envs, env.num_actions), device=device) * 2.0 - 1.0
                if args.random_action_indices.strip():
                    indices: list[int] = []
                    for tok in args.random_action_indices.split(','):
                        tok = tok.strip()
                        if not tok:
                            continue
                        indices.append(int(tok))
                    mask = torch.zeros((env.num_actions,), device=device, dtype=torch.bool)
                    mask[indices] = True
                    actions = torch.where(mask.unsqueeze(0), actions, torch.zeros_like(actions))
            obs, rew, resets, info = env.step(actions)

            rew_sum += rew.to(device)
            done = resets.to(device).bool()
            if done.any():
                ep_count[done] += 1
                rew_sum[done] = 0.0

            if (step % args.print_every) == 0:
                msg = f'[debug] step={step} | resets={int(done.sum().item())} | rew(mean)={float(rew.mean().item()):.3f}'
                if hasattr(env, 'nut_dof_pos'):
                    nut_pos = getattr(env, 'nut_dof_pos')
                    if isinstance(nut_pos, torch.Tensor) and nut_pos.numel() > 0:
                        msg += f' | nut_pos(mean)={float(nut_pos.mean().item()):.4f}'
                if hasattr(env, 'nut_dof_vel_cf'):
                    nut_vel = getattr(env, 'nut_dof_vel_cf')
                    if isinstance(nut_vel, torch.Tensor) and nut_vel.numel() > 0:
                        msg += f' | nut_vel(mean)={float(nut_vel.mean().item()):.4f}'
                print(msg)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
