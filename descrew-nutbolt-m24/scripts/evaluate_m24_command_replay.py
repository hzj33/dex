#!/usr/bin/env python3
"""Same-command, free-space physical replay of a real log (no policy success claim)."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np

if os.environ.get('E003_REPLAY_EXECUTION', 'identified') == 'identified':
    from evaluate_m24_e003_real_exec import isaacgym_task_map, torch, hydra
else:
    from evaluate_m24_e002_cycle import isaacgym_task_map, torch, hydra
if os.environ.get('E003_COUPLING_FIT'):
    import evaluate_m24_e003_passive_diagnostic
from calibrate_m24_e002_a0 import omegaconf_to_dict
from m24_e003_deploy_runtime import install_deployment_limits
from isaacgym import gymtorch
from omegaconf import open_dict


@hydra.main(config_name='config', config_path=str(Path(__file__).resolve().parents[1]/'configs'))
def main(cfg):
    source = Path(os.environ['E003_REPLAY_LOG'])
    out = Path(os.environ['E003_REPLAY_OUT'])
    assert not out.exists()
    rows = [json.loads(x) for x in (source/'policy_steps.jsonl').read_text().splitlines()]
    meta = json.loads((source/'run_metadata.json').read_text())
    assert not meta['dry_run'] and rows
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed))
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = meta['m24_init_pose']
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=0, headless=True)
    assert env.num_envs == 1 and env.num_dofs == 17
    policy = torch.jit.load(str(cfg.checkpoint), map_location='cpu').eval()
    install_deployment_limits(env, policy)
    reward = env.compute_reward
    def no_reset(*args):
        reward(*args)
        env.reset_buf.zero_()
    env.compute_reward = no_reset
    env.reset()
    # Start at the first logged observation; do not teleport thereafter.
    state = env.dof_state.view(1, -1, 2)
    q0 = torch.tensor(rows[0]['q'], device=env.device)
    state[0, :16, 0] = q0
    state[0, :16, 1] = 0
    env.gym.set_dof_state_tensor(env.sim, gymtorch.unwrap_tensor(env.dof_state))
    previous = torch.tensor(rows[0]['previous_target'], device=env.device)
    env.prev_targets[0, :16] = previous
    env.cur_targets[0, :16] = previous
    if hasattr(env, '_ar_eff'):
        env._ar_eff[0] = q0
        env._ar_prev_cmd[0] = previous
    indices = env.object_indices.to(torch.int32)
    env.root_state_tensor[env.object_indices, 1] += 1.0
    env.gym.set_actor_root_state_tensor_indexed(env.sim, gymtorch.unwrap_tensor(env.root_state_tensor),
                                              gymtorch.unwrap_tensor(indices), len(indices))
    records = []
    with torch.no_grad():
        for row in rows:
            before = env.dof_state.view(1, -1, 2)[0, :16, 0].cpu().tolist()
            target = torch.tensor(row['next_target'], device=env.device).reshape(1, 16)
            action = (target-env.prev_targets[:, :16])/env.action_scale
            env.step(action)
            sent = env.prev_targets[0, :16].cpu().numpy()
            assert np.max(np.abs(sent-np.array(row['next_target']))) < 1e-6
            force = torch.norm(env.e002_normal_contact_force, dim=-1)[0].cpu().numpy()
            assert np.max(force) < 1e-6, 'Replay contains object contact'
            records.append(dict(step=row['step'], q_before=before,
                q_after=env.xhand_hand_dof_pos[0].cpu().tolist(), target=sent.tolist()))
    real = np.array([r['q'] for r in rows]); sim = np.array([r['q_before'] for r in records])
    error = np.degrees(sim-real)
    result = dict(diagnostic_only=True, acceptance_eligible=False, reset_suppressed=True,
        execution=os.environ.get('E003_REPLAY_EXECUTION', 'identified'),
        passive_coupling=bool(os.environ.get('E003_COUPLING_FIT')),
        real_parameters=json.loads(os.environ['E003_REAL_EXEC_PARAMS']) if os.environ.get('E003_REAL_EXEC_PARAMS') else None,
        source=str(source), source_sha256=hashlib.sha256((source/'policy_steps.jsonl').read_bytes()).hexdigest(),
        initial_velocity_assumption='zero at first logged observation; initial filtered target=q0',
        rmse_deg=np.sqrt(np.mean(error**2, axis=0)).tolist(),
        peak_error_deg=np.max(np.abs(error), axis=0).tolist(),
        last_error_deg=error[-1].tolist(), records=records)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
