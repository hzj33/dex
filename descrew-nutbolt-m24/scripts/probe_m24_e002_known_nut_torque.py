#!/usr/bin/env python3
"""Known external nut-torque diagnostic; no PPO, reward credit or policy pass.

Compare free nut motion to the fixed candidate743 grasp while recording each
physics endpoint. Artificial body torque is diagnostic input only.
"""
import json
from pathlib import Path

from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch, gymtorch
from isaacgym import gymapi
from omegaconf import open_dict
from dexscrew.tasks.contact_moments import collect_env_contacts, normal_contact_wrench


@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    source = json.loads(Path(str(cfg.torque_probe_calibration)).read_text())
    out = Path(str(cfg.torque_probe_out))
    assert not out.exists() and source['pose_index'] == 743
    bank = [dict(hand_contact=contact, torque=tau) for contact in [False, True]
            for tau in [0., .001, .01, .05, .1, -.01]]
    assert int(cfg.task.env.numEnvs) == len(bank)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = source['custom_init_dof_pos']
    torch.manual_seed(42)
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=0, headless=True)
    suppression = dict(contact_acquire_deadline_steps=100000, contact_loss_grace_steps=100000,
        drive_deadline_steps=100000, target_angle=1000., hard_step_angle=1000.,
        overshoot_angle=1000., workspace_radius=2.)
    for name, value in suppression.items():
        setattr(env, 'e002_' + name, value)
    zero = torch.zeros((len(bank), 16), device=env.device)
    env.reset()
    # Move only the diagnostic no-contact hands, leaving each bolt base fixed.
    indices = env.hand_indices[:len(bank)//2].to(dtype=torch.int32)
    env.root_state_tensor[indices.long(), 0] += .5
    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_state_tensor),
        gymtorch.unwrap_tensor(indices), len(indices))
    for _ in range(6):
        env.step(zero)
    start = env.nut_dof_pos[:, 0].clone()
    force = torch.zeros((len(bank), env.num_bodies, 3), device=env.device)
    torque = torch.zeros_like(force)
    magnitude = torch.tensor([b['torque'] for b in bank], device=env.device)
    torque[:, env.screw_nut_rb_handle] = magnitude[:, None] * env.rot_axis_buf
    original_control = env.update_low_level_control
    original_render = env.render
    rows = []
    def control(step):
        original_control(step)
        env.gym.apply_rigid_body_force_tensors(env.sim, gymtorch.unwrap_tensor(force),
            gymtorch.unwrap_tensor(torque), gymapi.ENV_SPACE)
    def record():
        original_render()
        env.gym.fetch_results(env.sim, True)
        env._refresh_gym()
        pair = [env.fingertip_handles[0], env.fingertip_handles[3]]
        f, m, counts = normal_contact_wrench(collect_env_contacts(env.gym, env.envs),
            env.rigid_body_states[:, env.screw_nut_rb_handle, 3:7].numpy(),
            env.screw_nut_rb_handle, pair, env.num_bodies)
        rows.append(dict(substep=len(rows), angle=(env.nut_dof_pos[:, 0]-start).tolist(),
            velocity=env.nut_dof_vel[:, 0].tolist(), nut_normal_force=f.tolist(),
            nut_normal_moment=m.tolist(), nut_contact_count=counts.tolist()))
    env.update_low_level_control = control
    env.render = record
    for _ in range(20):
        env.step(zero)
    assert len(rows) == 20 * env.control_freq_inv
    summary = []
    for i, b in enumerate(bank):
        angles = [r['angle'][i] for r in rows]
        summary.append(dict(b, net_angle=angles[-1],
            sampled_velocity_integral=sum(r['velocity'][i] * env.dt for r in rows),
            max_physics_step=max(abs(a-p) for a,p in zip(angles, [0.] + angles[:-1])),
            contact_substeps=sum(sum(r['nut_contact_count'][i]) > 0 for r in rows)))
    assert all(s['contact_substeps'] == 0 for s in summary if not s['hand_contact'])
    report = dict(status='completed:diagnostic_only', training_launched=False,
        policy_learned=False, external_torque_is_artificial=True, terminal_suppression=suppression,
        physics_dt=env.dt, policy_dt=env.dt * env.control_freq_inv, summary=summary, physics_substeps=rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, indent=2))
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
