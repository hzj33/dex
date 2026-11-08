#!/usr/bin/env python3
"""One fresh cloud environment per recorded A0c candidate; no training gate.

tensor_reference keeps the GPU tensor path; rigid_contacts uses the CPU tensor
path solely to expose Gym's per-contact records, retaining GPU PhysX/settings.
Read every physics-substep endpoint as well as the original policy endpoints.
"""

import hashlib
import json
import random
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

from diagnose_m24_e002_a0c import (
    _set_hand_poses, as_list, hydra, isaacgym_task_map, measure_step,
    omegaconf_to_dict, OmegaConf, open_dict, snapshot, torch,
)
import numpy as np
from isaacgym import gymapi, gymtorch


def plain(value):
    """Keep named fields in Gym's nested NumPy contact records (including Vec3)."""
    if isinstance(value, np.void) and value.dtype.names:
        return {name: plain(value[name]) for name in value.dtype.names}
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value] if value.ndim else plain(value.item())
    if isinstance(value, np.generic):
        return value.item()
    return value


def vec(value):
    return [float(getattr(value, k)) for k in ("x", "y", "z")]


@hydra.main(config_name="config", config_path="../configs")
def main(cfg):
    if cfg.task_name != "XHandPasiniM24NutBolt" or not cfg.task.env.e002_motion.enable:
        raise ValueError("requires the M24 E002 task")
    if int(cfg.task.env.numEnvs) != 1:
        raise ValueError("contact audit requires exactly one environment")
    mode = str(cfg.get("contact_audit_mode", "rigid_contacts"))
    sensor_audit = bool(cfg.get("contact_force_sensor_audit", False))
    kinematic_audit = bool(cfg.get("contact_kinematic_audit", False))
    no_contact_control = bool(cfg.get("contact_no_contact_control", False))
    if no_contact_control and not sensor_audit:
        raise ValueError("no-contact negative control is only for sensor validation")
    if mode not in ("rigid_contacts", "tensor_reference"):
        raise ValueError(mode)
    if bool(cfg.task.sim.use_gpu_pipeline) != (mode == "tensor_reference"):
        raise ValueError("choose CPU tensor pipeline for contact records; GPU for reference")
    source_path = Path(str(cfg.contact_audit_source)).resolve()
    source = json.loads(source_path.read_text())
    candidate_id = int(cfg.contact_candidate_id)
    candidate = next(c for c in source["candidates"] if c["candidate_id"] == candidate_id)
    pose = source["custom_init_dof_pos"]
    assert source["pose_index"] == 743 and len(pose) == 16
    assert not any(candidate["action"][4:12])
    output = Path(str(cfg.contact_audit_out)).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = pose
    task_class = isaacgym_task_map[cfg.task_name]
    sensor_metadata = []
    if sensor_audit:
        class SensorAuditTask(task_class):
            def _create_object_asset(self):
                super()._create_object_asset()
                for asset, bodies in [(self.hand_asset, ["right_index_link_3", "right_thumb_link_3"]),
                                      (self.object_asset_list[0], ["nut"])]:
                    for body in bodies:
                        handle = self.gym.find_asset_rigid_body_index(asset, body)
                        assert handle >= 0
                        for label, forward, solver in [("forward", True, False),
                                                       ("solver", False, True),
                                                       ("both", True, True)]:
                            properties = gymapi.ForceSensorProperties()
                            properties.enable_forward_dynamics_forces = forward
                            properties.enable_constraint_solver_forces = solver
                            properties.use_world_frame = True
                            index = self.gym.create_asset_force_sensor(
                                asset, handle, gymapi.Transform(), properties)
                            assert index >= 0
                            sensor_metadata.append(dict(body=body, kind=label,
                                                        asset_sensor_index=index,
                                                        forward=forward, solver=solver))
        task_class = SensorAuditTask
    env = task_class(
        config=omegaconf_to_dict(cfg.task), sim_device=cfg.sim_device,
        graphics_device_id=cfg.graphics_device_id, headless=cfg.headless)
    sensor_tensor = None
    if sensor_audit:
        sensor_tensor = gymtorch.wrap_tensor(env.gym.acquire_force_sensor_tensor(env.sim))
        assert tuple(sensor_tensor.shape) == (9, 6)
    jacobian = (gymtorch.wrap_tensor(env.gym.acquire_jacobian_tensor(env.sim, "hand"))
                if kinematic_audit else None)
    kinematic_samples = []
    assert env.num_dofs == 17 and env.num_bodies - env.num_xhand_hand_bodies == 3
    thresholds = source["thresholds"]
    for name, value in thresholds.items():
        if hasattr(env, "e002_" + name):
            assert float(getattr(env, "e002_" + name)) == value, name
    for name, value in source["terminal_suppression"].items():
        setattr(env, "e002_" + name, value)
    expected_names = [f"right_{finger}_joint_{j}" for finger in ("index", "middle", "ring", "thumb") for j in range(4)]
    assert list(env.gym.get_asset_dof_names(env.hand_asset)) == expected_names
    body_names, body_com, shape_properties = [], [], []
    for actor in (0, 1):
        body_names += list(env.gym.get_actor_rigid_body_names(env.envs[0], actor))
        for prop in env.gym.get_actor_rigid_body_properties(env.envs[0], actor):
            body_com.append(dict(position=vec(prop.com),
                                 mass=float(prop.mass)))
        shape_properties.append([dict(friction=float(p.friction),
                                      rolling_friction=float(p.rolling_friction),
                                      torsion_friction=float(p.torsion_friction),
                                      restitution=float(p.restitution))
                                 for p in env.gym.get_actor_rigid_shape_properties(env.envs[0], actor)])
    # Detect any ordinary task reset; audit samples are never silently spliced.
    original_reset = env.reset_idx
    reset_count = [0]

    def counted_reset(ids):
        reset_count[0] += 1
        return original_reset(ids)

    env.reset_idx = counted_reset
    zero = torch.zeros((1, 16), device=env.rl_device)
    action = torch.tensor([candidate["action"]], dtype=zero.dtype, device=env.rl_device)
    poses = torch.tensor([pose], device=env.device)
    ids = torch.tensor([0], device=env.device)
    phases = [("hold_settle", 6), ("hold", 10), ("probe_settle", 6), ("probe", 10)]
    samples = []
    context = {}
    original_render = env.render

    def record_substep():
        original_render()
        # CPU tensor stepping normally fetches at the start of the next step.
        # Explicitly complete this step before associating states and contacts.
        env.gym.fetch_results(env.sim, True)
        env._refresh_gym()
        data = {key: values[0] for key, values in snapshot(env).items()}
        data["rigid_body_state"] = as_list(env.rigid_body_states[0])
        if sensor_audit:
            env.gym.refresh_force_sensor_tensor(env.sim)
            data["force_sensor_wrench"] = as_list(sensor_tensor)
            data["commanded_dof_torque"] = as_list(env.torques[0, context["substep"]])
        data.update(context)
        data["physics_substep"] = context["substep"]
        data["unexpected_reset"] = reset_count[0] != context["reset_count"]
        theta = float(env.nut_dof_pos[0, 0])
        data["directed_step"] = env.e002_direction * (theta - context["theta_previous"])
        data["directed_angle"] = env.e002_direction * (theta - context["theta_start"])
        context["theta_previous"] = theta
        context["substep"] += 1
        data.update(measure_step(data["force_magnitude"], data["distance"],
                                 data["raw_axis_torque"], data["directed_step"], thresholds))
        if mode == "rigid_contacts":
            contacts = env.gym.get_env_rigid_contacts(env.envs[0])
            data["contacts"] = [plain(c) for c in contacts]
        if kinematic_audit and context["policy_step"] == 9 and data["physics_substep"] == 9:
            env.gym.refresh_jacobian_tensors(env.sim)
            kinematic_samples.append(dict(phase=context["phase"], sample_index=len(samples),
                                          pair_jacobian=as_list(jacobian[0, [3, 24]])))
        samples.append(data)

    env.render = record_substep
    for phase, steps in phases:
        if phase in ("hold_settle", "probe_settle"):
            env.reset_idx(ids)
            _set_hand_poses(env, poses)
            if no_contact_control:
                # Diagnostic fixture only: translate the still-fixed assembly
                # away from the hand. Never a training task/reset modification.
                objects = env.object_indices.to(dtype=torch.int32)
                env.root_state_tensor[env.object_indices, 2] += 0.5
                env.gym.set_actor_root_state_tensor_indexed(
                    env.sim, gymtorch.unwrap_tensor(env.root_state_tensor),
                    gymtorch.unwrap_tensor(objects), len(objects))
        theta = float(env.nut_dof_pos[0, 0])
        context.update(phase=phase, theta_start=theta, theta_previous=theta,
                       reset_count=reset_count[0])
        for step in range(steps):
            context.update(policy_step=step, substep=0,
                           action=as_list(action[0] if phase == "probe" else zero[0]))
            env.step(action if phase == "probe" else zero)
            assert context["substep"] == env.control_freq_inv
    assert len(samples) == 320 and not any(s["unexpected_reset"] for s in samples)
    report = dict(
        status="completed:contact_audit", diagnostic_only=True,
        a0c_status="not_evaluated", training_eligible=False, mode=mode,
        source=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        pose_index=743, custom_init_dof_pos=pose, candidate_id=candidate_id,
        candidate_name=candidate["name"], action=candidate["action"], thresholds=thresholds,
        pipeline_device=env.device, physx_use_gpu=bool(env.sim_params.physx.use_gpu),
        physics_dt=env.dt, control_frequency_inv=env.control_freq_inv,
        contact_collection=int(env.sim_params.physx.contact_collection),
        pair_handles=[env.fingertip_handles[0], env.fingertip_handles[3]],
        nut_handle=env.screw_nut_rb_handle, body_names=body_names, body_com=body_com,
        actor_shape_properties=shape_properties,
        force_sensor_audit=sensor_audit, force_sensor_metadata=sensor_metadata,
        kinematic_samples=kinematic_samples,
        no_contact_control=no_contact_control,
        diagnostic_fixture_translation=[0, 0, 0.5] if no_contact_control else [0, 0, 0],
        contact_field_note="lambda/normal and local points are raw API values; friction magnitudes alone do not determine a total wrench",
        samples=samples, completed_at=datetime.now(timezone.utc).isoformat())
    OmegaConf.save(cfg, output.with_suffix(".config.yaml"), resolve=True)
    output.with_suffix(".command.txt").write_text(shlex.join(sys.argv) + "\n")
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=report["status"], mode=mode, candidate=candidate_id,
                          samples=len(samples), contacts=sum(len(s.get("contacts", [])) for s in samples),
                          final_nut_angle=samples[-1]["nut_angle"], output=str(output))))
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
