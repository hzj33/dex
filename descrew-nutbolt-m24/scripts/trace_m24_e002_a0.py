"""Controlled one-environment E002-A0 contact/torque trace (no PPO).

This runner intentionally applies a small set of named, constant candidate
actions from the saved M24 grasp.  It does not claim that any candidate is a
valid pinch-twist policy; it records the actual Isaac Gym fingertip mapping,
resultant-force convention, nut-axis torque projection, and nut response so
the E002 pair gate can be corrected from evidence rather than reward tuning.

Example (cloud, headless):
  python scripts/trace_m24_e002_a0.py task=XHandPasiniM24NutBoltE002 \
    headless=True task.env.numEnvs=1 trace_steps=16 \
    task.env.e002_motion.contact_grace_steps=10000
"""

import csv
import json
import os
import sys
from pathlib import Path

# Isaac Gym must be imported before torch.
import isaacgym  # noqa: F401

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


def _candidate_actions(env, device, calibrated_action=None):
    """Return named index/thumb candidate increments in the task action order."""
    base = torch.zeros((env.num_envs, env.num_acts), device=device)

    def candidate(name, indices, value):
        action = base.clone()
        action[:, indices] = value
        return name, action

    def pair_candidate(name, index_value, thumb_value):
        action = base.clone()
        action[:, 0:4] = index_value
        action[:, 12:16] = thumb_value
        return name, action

    candidates = [
        ("zero", base.clone()),
        *[
            candidate(name, indices, value)
            for name, indices, value in (
                ("index_plus", slice(0, 4), 1.0),
                ("index_minus", slice(0, 4), -1.0),
                ("thumb_plus", slice(12, 16), 1.0),
                ("thumb_minus", slice(12, 16), -1.0),
                ("pair_plus", list(range(0, 4)) + list(range(12, 16)), 1.0),
                ("pair_minus", list(range(0, 4)) + list(range(12, 16)), -1.0),
            )
        ],
        pair_candidate("index_plus_thumb_minus", 1.0, -1.0),
        pair_candidate("index_minus_thumb_plus", -1.0, 1.0),
    ]
    if calibrated_action is not None:
        action = torch.tensor(calibrated_action, device=device, dtype=base.dtype)
        if action.numel() != env.num_acts:
            raise ValueError(
                f"calibrated drive action must have {env.num_acts} values; "
                f"got {action.numel()}"
            )
        candidates.append(("calibrated_pair_drive", action.reshape(1, -1)))
    return candidates


def _row(env, profile, step, theta_start, reward, done):
    pair_handles = [env.fingertip_handles[0], env.fingertip_handles[3]]
    fingertip_pos = env.rigid_body_states[0, pair_handles, :3]
    fingertip_force = env.contact_forces[0, pair_handles, :]
    nut_pos = env.rigid_body_states[0, env.screw_nut_rb_handle, :3]
    radial = fingertip_pos - nut_pos.unsqueeze(0)
    force_on_nut = fingertip_force * env.e002_force_sign
    torque = torch.cross(radial, force_on_nut, dim=-1)
    if env.e002_torque_source == "contact_point_normal":
        torque = -env.e002_force_sign * env.e002_normal_contact_moment[0]
    axis = env.rot_axis_buf[0]
    signed_torque = env.e002_direction * (torque * axis.unsqueeze(0)).sum(dim=-1)
    force_mag = torch.norm(fingertip_force, dim=-1)
    dist = torch.norm(radial, dim=-1)
    theta = env.nut_dof_pos[0, 0]
    names = ("index", "thumb")
    result = {
        "profile": profile,
        "step": step,
        "theta": float(theta.item()),
        "theta_delta": float((theta - theta_start).item()),
        "reward": float(reward[0].item()),
        "done": int(done[0].item()),
        "axis_x": float(axis[0].item()),
        "axis_y": float(axis[1].item()),
        "axis_z": float(axis[2].item()),
        "pair_gate": float(env.extras.get("e002/pair_gate", 0.0)),
        "term_contact_lost": float(env.extras.get("e002/term/contact_lost", 0.0)),
    }
    for i, name in enumerate(names):
        result[f"{name}_force_x"] = float(fingertip_force[i, 0].item())
        result[f"{name}_force_y"] = float(fingertip_force[i, 1].item())
        result[f"{name}_force_z"] = float(fingertip_force[i, 2].item())
        result[f"{name}_force_mag"] = float(force_mag[i].item())
        result[f"{name}_distance"] = float(dist[i].item())
        result[f"{name}_torque_x"] = float(torque[i, 0].item())
        result[f"{name}_torque_y"] = float(torque[i, 1].item())
        result[f"{name}_torque_z"] = float(torque[i, 2].item())
        result[f"{name}_signed_torque"] = float(signed_torque[i].item())
        result[f"{name}_contact"] = int(env.e002_contact_state[0, i].item())
    return result


@hydra.main(config_name="config", config_path="../configs")
def main(cfg: DictConfig):
    if cfg.task_name != "XHandPasiniM24NutBolt":
        raise ValueError("trace_m24_e002_a0.py requires task=XHandPasiniM24NutBoltE002")
    if not bool(cfg.task.env.e002_motion.enable):
        raise ValueError("trace_m24_e002_a0.py requires env.e002_motion.enable=True")
    if int(cfg.task.env.numEnvs) != 1:
        raise ValueError("controlled A0 trace requires task.env.numEnvs=1")

    trace_steps = int(cfg.get("trace_steps", 16))
    if trace_steps < 1:
        raise ValueError("trace_steps must be positive")
    output = Path(str(cfg.get("trace_out", "outputs/e002_a0_controlled_trace.csv")))
    output.parent.mkdir(parents=True, exist_ok=True)
    calibration_path = str(cfg.get("trace_calibration", "")).strip()
    calibrated_action = None
    calibrated_schedule = None
    if calibration_path:
        calibration = json.loads(Path(calibration_path).read_text(encoding="utf-8"))
        if calibration.get("status") != "passed":
            raise ValueError(f"calibration did not pass: {calibration_path}")
        calibrated_action = calibration["drive_action"]
        calibrated_schedule = calibration.get("drive_action_scale_schedule")

    env = isaacgym_task_map[cfg.task_name](
        config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id,
        headless=cfg.headless,
    )
    rows = []
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    for profile, action in _candidate_actions(env, env.rl_device, calibrated_action):
        env.reset_idx(all_ids)
        _, _, _, _ = env.step(torch.zeros_like(action))
        theta_start = env.nut_dof_pos[0, 0].clone()
        for step in range(trace_steps):
            scale = calibrated_schedule[min(step, len(calibrated_schedule) - 1)] if (
                profile == "calibrated_pair_drive" and calibrated_schedule) else 1.0
            _, reward, done, _ = env.step(action * scale)
            rows.append(_row(env, profile, step, theta_start, reward, done))
            if bool(done[0].item()):
                break

    if not rows:
        raise RuntimeError("A0 trace produced no samples")
    if not all(torch.isfinite(torch.tensor(list(row.values())[2:])) .all() for row in rows):
        raise RuntimeError("A0 trace produced a non-finite numeric value")
    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[E002-A0] wrote {len(rows)} rows to {output}")
    for profile in sorted({row['profile'] for row in rows}):
        last = [row for row in rows if row["profile"] == profile][-1]
        print(
            "[E002-A0] "
            f"{profile}: theta_delta={last['theta_delta']:.6g} "
            f"tau(index,thumb)=({last['index_signed_torque']:.6g},"
            f"{last['thumb_signed_torque']:.6g}) "
            f"contact=({last['index_contact']},{last['thumb_contact']})"
        )


if __name__ == "__main__":
    main()
