#!/usr/bin/env python3
"""Verify cold and subset-reset actor histories in real E002, without training."""
import hashlib
import json
from pathlib import Path
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch
from omegaconf import open_dict

@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    assert int(cfg.task.env.numEnvs) == 2
    source = Path(str(cfg.reset_audit_calibration))
    calibration = json.loads(source.read_text())
    assert calibration['status'] == 'passed' and calibration['pose_index'] == 743
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration['custom_init_dof_pos']
        cfg.task.env.e002_motion.force_sign = calibration['force_sign']
    torch.manual_seed(42)
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=0, headless=True)
    obs = env.reset()
    n = env.numActions * 2 * 3
    expected = env.init_pose_buf[:, :env.num_actions].repeat(1, 6).to(env.rl_device)
    cold_error = float((obs['obs'][:, :n] - expected).abs().max())
    action = torch.zeros((2, env.num_actions), device=env.rl_device)
    action[:, 0] = torch.tensor([.1, -.1], device=env.rl_device)
    env.step(action)
    other_history = env.obs_buf_lag_history[1, -3:, :env.numActions * 2].reshape(-1).clone()
    env.reset_idx(torch.tensor([0], device=env.device))
    env.compute_observations()
    subset_error = float((env.obs_buf[0, :n] - expected[0].to(env.device)).abs().max())
    other_error = float((env.obs_buf[1, :n] - other_history).abs().max())
    finite = bool(torch.isfinite(env.obs_buf).all())
    passed = cold_error < 1e-6 and subset_error < 1e-6 and other_error == 0. and finite
    report = dict(status='passed' if passed else 'failed', training_launched=False,
        num_envs=2, cold_reset_max_error=cold_error, subset_reset_max_error=subset_error,
        nonreset_observation_max_change=other_error, finite=finite,
        task_sha256=hashlib.sha256(Path('dexscrew/tasks/xhand_pasini.py').read_bytes()).hexdigest())
    out = Path(str(cfg.reset_audit_out));out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    env.gym.destroy_sim(env.sim)
    raise SystemExit(0 if passed else 4)

if __name__ == '__main__':
    main()
