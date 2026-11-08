"""Minimal Pasini env debug runner (no PPO training).

Goals:
- Instantiate IsaacGym task via the same Hydra config as train.py.
- Reset/step for a few iterations.
- Detect NaN/Inf (and suspiciously large values) in observations and key sim tensors.

Usage examples:
- Headless quick check:
  python scripts/debug_pasini_env.py task=XHandPasiniScrewDriver headless=True task.env.numEnvs=6 steps=200

- Viewer check (small env count):
  python scripts/debug_pasini_env.py task=XHandPasiniScrewDriver headless=False task.env.numEnvs=1 steps=200

- Bisect: disable point cloud:
  python scripts/debug_pasini_env.py task=XHandPasiniScrewDriver task.env.hora.point_cloud_sampled_dim=0

- Bisect: disable privileged info (all):
  python scripts/debug_pasini_env.py task=XHandPasiniScrewDriver task.env.privInfo.enableObjPos=False task.env.privInfo.enableObjScale=False \
    task.env.privInfo.enableObjMass=False task.env.privInfo.enableObjCOM=False task.env.privInfo.enableObjFriction=False \
    task.env.privInfo.enable_obj_restitution=False task.env.privInfo.enable_obj_orientation=False task.env.privInfo.enable_obj_linvel=False \
    task.env.privInfo.enable_obj_angvel=False task.env.privInfo.enable_ft_pos=False task.env.privInfo.enable_ft_orientation=False \
    task.env.privInfo.enable_ft_linvel=False task.env.privInfo.enable_ft_angvel=False task.env.privInfo.enable_hand_scale=False \
    task.env.privInfo.enable_nut_contact=False task.env.privInfo.enable_nut_pos=False task.env.privInfo.enable_nut_dof_vel=False \
    task.env.privInfo.enable_nut_dof_pos=False task.env.privInfo.enable_screw_joint_friction=False
"""

import os
import sys

# IsaacGym requires being imported before torch.
import isaacgym  # noqa: F401

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

# Ensure repo root is importable even after Hydra changes cwd.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dexscrew.tasks import isaacgym_task_map
from dexscrew.utils.reformat import omegaconf_to_dict, print_dict


# Match train.py's resolvers (required by configs).
OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver(
    "resolve_default", lambda default, arg: default if arg == "" else arg
)


def _stat(t: torch.Tensor):
    t = t.detach()
    finite = torch.isfinite(t)
    nonfinite = int((~finite).sum().item())
    t_f = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    return {
        "shape": tuple(t.shape),
        "dtype": str(t.dtype).replace("torch.", ""),
        "device": str(t.device),
        "nonfinite": nonfinite,
        "min": float(t_f.min().item()) if t.numel() else 0.0,
        "max": float(t_f.max().item()) if t.numel() else 0.0,
        "mean": float(t_f.mean().item()) if t.numel() else 0.0,
        "absmax": float(t_f.abs().max().item()) if t.numel() else 0.0,
    }


def _check(name: str, t: torch.Tensor, *, absmax_warn: float = 1e4):
    s = _stat(t)
    bad = s["nonfinite"] > 0
    big = s["absmax"] > absmax_warn
    if bad or big:
        print(
            f"[DEBUG] {name}: shape={s['shape']} {s['dtype']} {s['device']} "
            f"nonfinite={s['nonfinite']} min={s['min']:.3g} max={s['max']:.3g} "
            f"mean={s['mean']:.3g} absmax={s['absmax']:.3g}"
        )

        if bad and t.dim() == 2:
            # Help localize which rows are corrupt (e.g., which DOFs or rigid bodies).
            finite = torch.isfinite(t)
            row_bad = (~finite).any(dim=1)
            bad_rows = row_bad.nonzero(as_tuple=False).squeeze(-1)
            if bad_rows.numel() > 0:
                head = bad_rows[: min(20, bad_rows.numel())].detach().cpu().tolist()
                print(
                    f"[DEBUG] {name}: bad_rows(first)={head} total_bad_rows={int(bad_rows.numel())}"
                )
    return bad


@hydra.main(config_name="config", config_path="../configs")
def main(cfg: DictConfig):
    # Keep config printing for reproducibility, but this runner does not touch git.
    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    steps = int(cfg.get("steps", 200))
    random_actions = bool(cfg.get("random_actions", True))
    absmax_warn = float(cfg.get("absmax_warn", 1e4))

    env = isaacgym_task_map[cfg.task_name](
        config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )

    obs = env.reset()
    # Initial observation checks.
    for k, v in obs.items():
        if torch.is_tensor(v):
            _check(f"obs_dict[{k}]", v, absmax_warn=absmax_warn)

    # Also check common sim tensors if present.
    for attr in [
        "root_state_tensor",
        "dof_state",
        "rigid_body_states",
        "xhand_hand_dof_pos",
        "xhand_hand_dof_vel",
        "object_pos",
        "object_rot",
        "object_linvel",
        "object_angvel",
    ]:
        if hasattr(env, attr):
            t = getattr(env, attr)
            if torch.is_tensor(t):
                _check(f"env.{attr}", t, absmax_warn=absmax_warn)

    for i in range(steps):
        if random_actions:
            actions = (
                torch.rand((env.num_envs, env.num_acts), device=cfg.rl_device) * 2.0
                - 1.0
            )
        else:
            actions = torch.zeros((env.num_envs, env.num_acts), device=cfg.rl_device)

        obs, rew, done, info = env.step(actions)

        bad_any = False
        for k, v in obs.items():
            if torch.is_tensor(v):
                bad_any = (
                    _check(f"step{i}.obs_dict[{k}]", v, absmax_warn=absmax_warn)
                    or bad_any
                )

        # Check a few internal tensors periodically.
        if i % 10 == 0:
            for attr in [
                "root_state_tensor",
                "dof_state",
                "rigid_body_states",
                "object_pos",
                "object_rot",
                "object_linvel",
                "object_angvel",
            ]:
                if hasattr(env, attr):
                    t = getattr(env, attr)
                    if torch.is_tensor(t):
                        bad_any = (
                            _check(f"step{i}.env.{attr}", t, absmax_warn=absmax_warn)
                            or bad_any
                        )

        if bad_any:
            print(f"[DEBUG] Found anomaly at step {i}; stopping.")
            break

    print("[DEBUG] Done.")


if __name__ == "__main__":
    main()
