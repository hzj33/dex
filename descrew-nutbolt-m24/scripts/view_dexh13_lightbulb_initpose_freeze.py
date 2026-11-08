#!/usr/bin/env python3
"""Open a headed viewer and freeze the DexH13 lightbulb reset init pose."""

import argparse
import os
import sys

import isaacgym  # noqa: F401
from hydra import compose, initialize_config_dir
from isaacgym import gymapi
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


def _as_list(value):
    if value is None:
        return None
    return [float(v) for v in list(value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="Dexh13HoraLightbulb")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    _register_resolvers()
    base_overrides = [
        f"task={args.task}",
        "headless=False",
        f"num_envs={args.num_envs}",
        f"task.env.numEnvs={args.num_envs}",
        f"seed={args.seed}",
        f"sim_device=cuda:{args.gpu}",
        f"rl_device=cuda:{args.gpu}",
        "graphics_device_id=0",
        "wandb_activate=False",
        "train.algo=PPO",
        "task.env.object.init_pos_noise=[0.0,0.0,0.0]",
        "task.env.randomization.randomizeMass=False",
        "task.env.randomization.randomizeCOM=False",
        "task.env.randomization.randomizeFriction=False",
        "task.env.randomization.randomizeScale=False",
        "task.env.randomization.randomizePDGains=False",
        "task.env.randomization.action_noise_e_scale=0.0",
        "task.env.randomization.action_noise_t_scale=0.0",
        "task.env.randomization.obs_noise_e_scale=0.0",
        "task.env.randomization.obs_noise_t_scale=0.0",
        "task.env.randomization.noisy_rpy_scale=0.0",
        "task.env.randomization.noisy_pos_scale=0.0",
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
    env.reset()
    env._refresh_gym()

    asset_cfg = cfg.task.env.asset
    object_cfg = cfg.task.env.object
    print("\nFrozen DexH13 lightbulb init-pose viewer")
    print("Close the viewer window or press ESC in the window to quit.")
    print("task:", args.task)
    print("num_envs:", args.num_envs)
    print("object.init_pos:", _as_list(object_cfg.get("init_pos")))
    print("object.init_pos_noise:", _as_list(object_cfg.get("init_pos_noise")))
    print("handRootPos:", _as_list(asset_cfg.get("handRootPos")))
    print("handRootRPY:", _as_list(asset_cfg.get("handRootRPY")))

    gym = env.gym
    sim = env.sim
    viewer = env.viewer
    gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_ESCAPE, "QUIT")

    while viewer and not gym.query_viewer_has_closed(viewer):
        should_quit = False
        for evt in gym.query_viewer_action_events(viewer):
            if evt.value > 0 and evt.action == "QUIT":
                should_quit = True
        if should_quit:
            break

        env._refresh_gym()
        gym.step_graphics(sim)
        gym.draw_viewer(viewer, sim, True)
        gym.sync_frame_time(sim)


if __name__ == "__main__":
    main()
