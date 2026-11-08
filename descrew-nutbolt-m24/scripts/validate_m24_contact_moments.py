#!/usr/bin/env python3
"""Two-environment, real-simulator validation of batched normal contact moments.

No policy is constructed and no training is launched.
"""
import json
from pathlib import Path

from calibrate_m24_e002_a0 import (
    _set_hand_poses, hydra, isaacgym_task_map, omegaconf_to_dict, torch,
)
import numpy as np
from omegaconf import open_dict
from analyze_m24_e002_contacts import normal_wrench
from dexscrew.tasks.contact_moments import collect_env_contacts, normal_contact_wrench


def plain(value):
    if isinstance(value, np.void):
        return {name: plain(value[name]) for name in value.dtype.names}
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value]
    return value.item() if isinstance(value, np.generic) else value


@hydra.main(config_name="config", config_path="../configs")
def main(cfg):
    assert int(cfg.task.env.numEnvs) == 2
    source = json.loads(Path(str(cfg.diagnostic_pose_json)).read_text())
    assert source["pose_index"] == 743
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = source["custom_init_dof_pos"]
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id, headless=True)
    assert env.e002_torque_source == "contact_point_normal"
    env.e002_contact_acquire_deadline_steps = 10000
    env.e002_contact_loss_grace_steps = 10000
    env.e002_drive_deadline_steps = 10000
    env.reset()
    _set_hand_poses(env, torch.tensor(source["custom_init_dof_pos"], device=env.device).repeat(2, 1))
    pair = [env.fingertip_handles[0], env.fingertip_handles[3]]
    nut = env.screw_nut_rb_handle
    samples = []
    for step in range(16):
        obs, _, done, _ = env.step(torch.zeros((2, 16), device=env.device))
        assert not bool(done.any())
        contacts = collect_env_contacts(env.gym, env.envs)
        if step == 0:
            print("simulation_contact_index_layout", {
                name: np.unique(contacts[name]).tolist()
                for name in ("body0", "body1", "env0", "env1")})
            print("simulation_contact_sample", plain(contacts[:4]))
        force, moment, count = normal_contact_wrench(contacts,
            env.rigid_body_states[:, nut, 3:7].numpy(), nut, pair, env.num_bodies)
        residuals = []
        for i in range(2):
            records = env.gym.get_env_rigid_contacts(env.envs[i])
            sample = dict(rigid_body_state=env.rigid_body_states[i].tolist(),
                contacts=plain(records), rotation_axis=env.rot_axis_buf[i].tolist(),
                all_body_contact_force=env.contact_forces[i].tolist())
            reference = normal_wrench(sample, pair, nut)
            # The independent scalar implementation reports axial moments.
            np.testing.assert_allclose(moment[i] @ env.rot_axis_buf[i].numpy(),
                reference["normal_axis_torque_on_nut"], atol=1e-8, rtol=1e-5)
            np.testing.assert_allclose(force[i], -env.contact_forces[i, pair].numpy(), atol=1e-5, rtol=1e-5)
            residuals.append(reference["normal_tensor_max_abs_residual"])
        np.testing.assert_allclose(moment, env.e002_normal_contact_moment.numpy(), atol=1e-8)
        assert all(obs[k].device.type == "cuda" for k in
                   ("obs", "priv_info", "proprio_hist", "point_cloud_info", "rot_axis_buf"))
        samples.append(dict(step=step, counts=count.tolist(), force=force.tolist(),
            moment=moment.tolist(), tensor_residual=residuals))
    assert np.array([s["counts"] for s in samples[-10:]]).min() > 0
    report = dict(status="passed:batched_normal_contact_moments", training_launched=False,
        a0c_status="not_evaluated", num_envs=2, body_count=env.num_bodies,
        tensor_device=str(env.device), policy_device=str(obs["obs"].device),
        physx_use_gpu=bool(env.sim_params.physx.use_gpu), samples=samples)
    Path(str(cfg.validation_out)).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["status"])
    env.gym.destroy_sim(env.sim)


if __name__ == "__main__":
    main()
