#!/usr/bin/env python3
"""Fixed candidate-743 A0c diagnostics only; this entrypoint cannot launch PPO.

Run with the E002 task, 67 environments and +diagnostic_pose_json=<old A0 JSON>.
All samples are 20 Hz policy-step endpoints (net force collection is unchanged).
"""

import hashlib
import json
import math
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

# The existing calibration module establishes the Isaac Gym-before-torch order
# and Hydra resolvers. Importing it does not execute its decorated main.
from calibrate_m24_e002_a0 import (
    _hand_joint_limits, _set_hand_poses, gymtorch, hydra, isaacgym_task_map,
    omegaconf_to_dict, OmegaConf, torch,
)
import numpy as np
from omegaconf import open_dict

from m24_e002_a0c_diagnostics import build_candidates, measure_step, summarize


def as_list(value):
    return value.detach().cpu().tolist()


def snapshot(env):
    handles = [env.fingertip_handles[0], env.fingertip_handles[3]]
    position = env.rigid_body_states[:, handles, :3]
    force = env.contact_forces[:, handles, :]
    nut = env.rigid_body_states[:, env.screw_nut_rb_handle, :3]
    radial = position - nut.unsqueeze(1)
    torque = torch.cross(radial, force, dim=-1)
    legacy_torque = torque
    if env.e002_torque_source == "contact_point_normal":
        torque = -env.e002_normal_contact_moment
    values = dict(
        force=force, force_magnitude=torch.norm(force, dim=-1),
        distance=torch.norm(radial, dim=-1), fingertip_position=position,
        raw_torque_vector=torque, raw_axis_torque=(torque * env.rot_axis_buf[:, None]).sum(-1),
        legacy_link_origin_raw_torque_vector=legacy_torque,
        rotation_axis=env.rot_axis_buf, nut_position=nut,
        nut_angle=env.nut_dof_pos[:, 0], nut_velocity=env.nut_dof_vel[:, 0],
        contact_hysteresis=env.e002_contact_state,
        joint_position=env.xhand_hand_dof_pos,
        joint_target=env.prev_targets[:, :16], progress=env.progress_buf,
        all_body_contact_force=env.contact_forces)
    values["reward"] = env.rew_buf
    values["forward_quality"] = env.e002_forward_quality
    if env.e002_torque_source == "contact_point_normal":
        values.update(normal_contact_force_on_nut=env.e002_normal_contact_force,
                      normal_contact_moment_on_nut=env.e002_normal_contact_moment,
                      normal_contact_count=env.e002_normal_contact_count)
    return {name: as_list(value) for name, value in values.items()}


def tangent_actions(env, jacobian):
    """Damped least-squares Cartesian tangent from the settled pose of env 0.

    Gym fixed-base Jacobians omit the base link, hence handle - 1. The saved
    basis and predicted velocities make this kinematic hypothesis reviewable.
    """
    env.gym.refresh_jacobian_tensors(env.sim)
    handles = [env.fingertip_handles[0], env.fingertip_handles[3]]
    if jacobian.shape != (env.num_envs, env.num_xhand_hand_bodies - 1, 6, 16):
        raise ValueError(f"unexpected fixed-hand Jacobian shape {jacobian.shape}")
    axis = env.rot_axis_buf[0]
    nut = env.rigid_body_states[0, env.screw_nut_rb_handle, :3]
    basis, metadata = [], []
    for handle, columns in zip(handles, (slice(0, 4), slice(12, 16))):
        radial = env.rigid_body_states[0, handle, :3] - nut
        tangent = env.e002_direction * torch.cross(axis, radial, dim=-1)
        tangent /= torch.clamp(torch.norm(tangent), min=1e-8)
        j = jacobian[0, handle - 1, :3, columns]
        damping = 0.001  # metres/radian; affects only action proposal, never gating
        direction = j.T @ torch.linalg.solve(
            j @ j.T + damping ** 2 * torch.eye(3, device=j.device), tangent)
        if not bool(torch.isfinite(direction).all()) or float(direction.abs().max()) < 1e-8:
            raise ValueError("no finite nonzero tangent action basis")
        direction /= direction.abs().max()
        basis.append(as_list(direction))
        metadata.append(dict(handle=handle, jacobian_linear=as_list(j),
                             desired_tangent=as_list(tangent), normalized_action=as_list(direction),
                             predicted_cartesian_motion=as_list(j @ direction), damping=damping))
    return basis, metadata


@hydra.main(config_name="config", config_path="../configs")
def main(cfg):
    if cfg.task_name != "XHandPasiniM24NutBolt" or not cfg.task.env.e002_motion.enable:
        raise ValueError("requires task=XHandPasiniM24NutBoltE002")
    source = Path(str(cfg.diagnostic_pose_json)).resolve()
    original = json.loads(source.read_text())
    pose = original["custom_init_dof_pos"]
    if original.get("pose_index") != 743 or len(pose) != 16 or not all(map(math.isfinite, pose)):
        raise ValueError("requires the recorded finite 16-DOF candidate-743 pose")
    amplitudes = list(map(float, cfg.get("diagnostic_amplitudes", [0.025, 0.05, 0.1])))
    amplitude_cap = 0.3 if cfg.get("diagnostic_contact_basis_json") else 0.1
    if not amplitudes or any(not 0 < a <= amplitude_cap for a in amplitudes):
        raise ValueError(f"diagnostic amplitudes must be in (0, {amplitude_cap}]; larger recovery actions require a validated contact basis")
    expected_count = 1 + 22 * len(amplitudes)
    external_bank, action_bank_source = None, None
    if cfg.get("diagnostic_action_bank_json"):
        bank_path = Path(str(cfg.diagnostic_action_bank_json)).resolve()
        external_bank = json.loads(bank_path.read_text())
        assert external_bank["status"] == "prepared:contact_action_bank"
        assert external_bank["pose_index"] == 743 and external_bank["custom_init_dof_pos"] == pose
        expected_count = len(external_bank["candidates"])
        for i, candidate in enumerate(external_bank["candidates"]):
            action = np.array(candidate["action"])
            assert candidate["candidate_id"] == i and action.shape == (16,)
            assert np.isfinite(action).all() and abs(action).max() <= 1.0 and not action[4:12].any()
            if "action_scale_schedule" in candidate:
                schedule = np.asarray(candidate["action_scale_schedule"])
                assert schedule.shape == (int(cfg.get("diagnostic_probe_steps", 10)),)
                assert np.isfinite(schedule).all() and (schedule >= 0).all() and (schedule <= 1).all()
        action_bank_source = dict(path=str(bank_path), sha256=hashlib.sha256(bank_path.read_bytes()).hexdigest(),
                                  definition=external_bank)
    if int(cfg.task.env.numEnvs) != expected_count:
        raise ValueError(f"task.env.numEnvs must be {expected_count}")
    if cfg.task.env.hand_init_dof_noise_fraction != 0:
        raise ValueError("fixed-pose diagnostics require zero reset noise")
    output = Path(str(cfg.get("diagnostic_out", "outputs/e002_a0c_diagnostic.json"))).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = pose
    env = isaacgym_task_map[cfg.task_name](
        config=omegaconf_to_dict(cfg.task), sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id, headless=cfg.headless)
    thresholds = {name: float(getattr(env, "e002_" + name)) for name in (
        "direction", "contact_force_on", "contact_force_off", "contact_dist_on",
        "contact_dist_off", "torque_min", "torque_saturation", "pair_gate_min")}
    probe_steps = int(cfg.get("diagnostic_probe_steps", 10))
    if probe_steps not in (10, 20, 30):
        raise ValueError("bounded protocol uses 10, 20 or 30 probe steps")
    thresholds.update(settle_steps=6, hold_steps=10, probe_steps=probe_steps,
                      required_probe_streak=5, min_probe_angle=math.pi / 180,
                      max_hold_drift=math.pi / 180)
    if external_bank:
        assert {k: v for k, v in external_bank["thresholds"].items() if k != "probe_steps"} == {k: v for k, v in thresholds.items() if k != "probe_steps"}
    # Same diagnostic-only terminal suppression as the historical A0 runner.
    # Record these separately from the untouched task config and pair thresholds.
    suppression = dict(contact_acquire_deadline_steps=1_000_000,
                       contact_loss_grace_steps=1_000_000, drive_deadline_steps=1_000_000,
                       workspace_radius=1.0, hard_step_angle=100.0,
                       target_angle=100.0, overshoot_angle=100.0)
    for name, value in suppression.items():
        setattr(env, "e002_" + name, value)
    ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    poses = torch.tensor(pose, device=env.device).repeat(env.num_envs, 1)
    lower, upper = _hand_joint_limits(env, poses.dtype)
    if bool(((poses < lower) | (poses > upper)).any()):
        raise ValueError("candidate-743 pose would be clamped")
    dof_names = list(env.gym.get_asset_dof_names(env.hand_asset))
    expected_names = [f"right_{finger}_joint_{j}" for finger in ("index", "middle", "ring", "thumb") for j in range(4)]
    if dof_names != expected_names:
        raise ValueError(f"joint order mismatch: {dof_names}")
    if env.num_dofs != 17 or env.num_bodies - env.num_xhand_hand_bodies != 3:
        raise ValueError("requires 16 hand DOFs, 3 object bodies and 1 nut DOF")
    jacobian = gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim, "hand"))
    zero = torch.zeros((env.num_envs, 16), device=env.device)
    # Count actual calls so an internal reset cannot silently pollute a streak.
    reset_counts = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    original_reset = env.reset_idx

    def counted_reset(env_ids):
        reset_counts[env_ids] += 1
        return original_reset(env_ids)

    env.reset_idx = counted_reset
    candidates = build_candidates(amplitudes, np.zeros((2, 4)))
    if external_bank:
        candidates = [dict(c, steps=[]) for c in external_bank["candidates"]]
    reset_audits = []

    def reset_stage(name):
        before = dict(nut_angle=as_list(env.nut_dof_pos[:, 0]),
                      nut_velocity=as_list(env.nut_dof_vel[:, 0]))
        env.reset_idx(ids)
        _set_hand_poses(env, poses)
        after = dict(nut_angle=as_list(env.nut_dof_pos[:, 0]),
                     nut_velocity=as_list(env.nut_dof_vel[:, 0]))
        reset_audits.append(dict(stage=name, before=before, after=after))
        return reset_counts.clone()

    def step_trace(phase, index, actions, start, previous, baseline_resets):
        env.step(actions)
        data = snapshot(env)
        theta = env.nut_dof_pos[:, 0].clone()
        directed_step = as_list(env.e002_direction * (theta - previous))
        directed_angle = as_list(env.e002_direction * (theta - start))
        changed = as_list(reset_counts != baseline_resets)
        action_rows = as_list(actions)
        for i, c in enumerate(candidates):
            row = {key: values[i] for key, values in data.items()}
            finite = all(np.isfinite(np.asarray(v)).all() for v in row.values())
            row.update(phase=phase, step=index, action=action_rows[i], finite=bool(finite),
                       unexpected_reset=changed[i], directed_step=directed_step[i],
                       directed_angle=directed_angle[i])
            row.update(measure_step(row["force_magnitude"], row["distance"],
                                    row["raw_axis_torque"], row["directed_step"], thresholds))
            c["steps"].append(row)
        return theta

    baseline = reset_stage("fixed_pose_hold")
    start = previous = env.nut_dof_pos[:, 0].clone()
    for step in range(thresholds["settle_steps"]):
        previous = step_trace("hold_settle", step, zero, start, previous, baseline)
    start = previous.clone()
    for step in range(thresholds["hold_steps"]):
        previous = step_trace("hold", step, zero, start, previous, baseline)
    baseline = reset_stage("a0c_probe")
    start = previous = env.nut_dof_pos[:, 0].clone()
    for step in range(thresholds["settle_steps"]):
        previous = step_trace("probe_settle", step, zero, start, previous, baseline)
    basis, tangent_metadata = tangent_actions(env, jacobian)
    contact_basis_source = None
    if cfg.get("diagnostic_contact_basis_json"):
        basis_path = Path(str(cfg.diagnostic_contact_basis_json)).resolve()
        measured = json.loads(basis_path.read_text())
        assert measured["status"] == "passed:contact_kinematic_basis"
        assert measured["custom_init_dof_pos"] == pose and measured["pose_index"] == 743
        assert {k: v for k, v in measured["thresholds"].items() if k != "probe_steps"} == {k: v for k, v in thresholds.items() if k != "probe_steps"}
        basis = np.asarray(measured["basis"], dtype=float)
        assert basis.shape == (2, 4) and np.isfinite(basis).all() and abs(basis).max() <= 1.0
        contact_basis_source = dict(path=str(basis_path), sha256=hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                                    legacy_tangent_basis=tangent_metadata, validation=measured)
        tangent_metadata = measured["fingers"]
    if not external_bank:
        for old, new in zip(candidates, build_candidates(amplitudes, basis)):
            old["action"] = new["action"]
    actions = torch.tensor([c["action"] for c in candidates], dtype=zero.dtype, device=env.device)
    start = previous.clone()
    for step in range(thresholds["probe_steps"]):
        scales = torch.tensor([c.get("action_scale_schedule", [1.] * thresholds["probe_steps"])[step]
                               for c in candidates], device=env.device, dtype=actions.dtype)
        previous = step_trace("probe", step, actions * scales[:, None], start, previous, baseline)
    result = summarize(candidates, thresholds, top_k=int(cfg.get("diagnostic_top_k", 5)),
                       acceptance_mode=env.e002_acceptance_mode)
    report = dict(
        schema_version=1, diagnostic_only=True, training_launched=False,
        completed_at=datetime.now(timezone.utc).isoformat(), seed=seed,
        source_pose_json=str(source), source_pose_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        pose_index=743, custom_init_dof_pos=pose, candidate_count=len(candidates),
        thresholds=thresholds, terminal_suppression=suppression,
        action_mode="scheduled_increment" if any("action_scale_schedule" in c for c in candidates) else "constant_increment",
        action_scale=env.action_scale,
        policy_dt=env.dt * env.control_freq_inv,
        physics_dt=env.dt, contact_collection=int(env.sim_params.physx.contact_collection),
        force_source="net_contact_force_tensor: fingertip resultant, not nut-specific",
        torque_source=env.e002_torque_source,
        invalid_forward_source=env.e002_invalid_forward_source,
        forward_source=env.e002_forward_source,
        acceptance_mode=env.e002_acceptance_mode,
        force_sign_source=("diagnostic_convention_not_friction_calibration"
                           if env.e002_acceptance_mode == "contact_progress"
                           else "normal_proxy_conjunction"),
        torque_component="normal_only; friction unmeasured",
        sim_use_gpu_pipeline=bool(env.sim_params.use_gpu_pipeline),
        physx_use_gpu=bool(env.sim_params.physx.use_gpu),
        finger_order=["index", "thumb"], force_signs=[-1, 1], dof_names=dof_names,
        body_names=env.hand_rb_names + list(env.gym.get_actor_rigid_body_names(env.envs[0], 1)),
        reset_audits=reset_audits, tangent_basis=tangent_metadata,
        contact_basis_source=contact_basis_source,
        action_bank_source=action_bank_source,
        candidates=candidates, **result)
    if result["selected"]:
        selected = result["selected"]
        report.update(force_sign=selected["force_sign"], drive_action=selected["action"],
                      drive_action_index=selected["candidate_id"])
        if "action_scale_schedule" in selected:
            report["drive_action_scale_schedule"] = selected["action_scale_schedule"]
    OmegaConf.save(cfg, output.with_suffix(".config.yaml"), resolve=True)
    output.with_suffix(".command.txt").write_text(shlex.join(sys.argv) + "\n")
    # Strict JSON prevents NaN/Infinity artifacts from being labelled as passed.
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "candidate_count", "condition_counts", "selected")}, indent=2))
    print(f"[A0c diagnostic-only] {output}")
    env.gym.destroy_sim(env.sim)
    raise SystemExit(0 if report["status"] == "passed" else 4)


if __name__ == "__main__":
    main()
