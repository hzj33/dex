"""Opt-in demonstrated phase-state initialization; PPO still supplies actions.

DemoStart III-A (https://arxiv.org/html/2409.06613v1) motivates intermediate
initial states. This fixed selected-state curriculum is a local adaptation,
not the paper's full automatic curriculum or zero-variance filter.
"""
import hashlib
import json
from pathlib import Path
import torch


class E002DemoResets:
    def __init__(self, env, cfg):
        self.cfg = cfg
        source = Path(str(cfg['source']))
        data = json.loads(source.read_text())
        self.source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if cfg.get('source_sha256'):
            assert self.source_sha256 == cfg['source_sha256']
        selected = list(cfg.get('seed_steps', [40, 43]))
        lookup = {r['step']: r for r in data['states']}
        rows = [lookup[s] for s in selected]
        self.phase = int(cfg.get('phase', 1))
        assert self.phase in (1, 2, 3)
        assert all(r['phase'] == self.phase and
                   r['event'] in ((0,) if self.phase == 1 else (0, self.phase)) for r in rows)
        assert all(r['reset_seed_state']['cycle']['phase'] == self.phase for r in rows)
        if self.phase == 2:
            assert all(not any(r['contact']) and max(r['force']) < env.e002_contact_force_off
                       and r['reset_seed_state']['cycle']['reference_valid'] for r in rows)
        if self.phase == 3:
            assert all(r['reset_seed_state']['cycle']['reference_valid'] for r in rows)
        self.rows = rows
        device = env.device
        self.steps = torch.tensor(selected, device=device, dtype=torch.long)
        def tensor(key):
            return torch.tensor([r['reset_seed_state'][key] for r in rows], device=device)
        self.dof = tensor('dof_state')
        self.roots = tensor('actor_root_state')
        self.targets = tensor('joint_targets')
        self.history = tensor('observation_history_before_roll')
        assert self.dof.shape[1:] == (17, 2) and self.roots.shape[1:] == (2, 13)
        self.contact = torch.tensor([r['contact'] for r in rows], device=device, dtype=torch.bool)
        self.cycle = {k:torch.tensor([r['reset_seed_state']['cycle'][k] for r in rows],
                     device=device, dtype=getattr(env.e002_cycle,k).dtype)
                     for k in rows[0]['reset_seed_state']['cycle']}
        self.active = torch.zeros(env.num_envs, device=device, dtype=torch.bool)
        self.seed_step = torch.full((env.num_envs,), -1, device=device, dtype=torch.long)
        self.initial_angle = torch.zeros(env.num_envs, device=device)
        self.fraction = float(cfg.get('fraction', 1.0))
        self.release_only = bool(cfg.get('release_only', True))
        self.return_only = bool(cfg.get('return_only', False))
        assert not (self.release_only and self.return_only)
        assert 0 <= self.fraction <= 1

    def apply(self, env, env_ids, gymtorch):
        self.active[env_ids] = False
        self.seed_step[env_ids] = -1
        self.initial_angle[env_ids] = 0
        ids = env_ids[torch.rand(len(env_ids),device=env.device) < self.fraction]
        if len(ids) == 0:
            return
        choice = torch.randint(len(self.steps), (len(ids),), device=env.device)
        self.active[ids] = True
        self.seed_step[ids] = self.steps[choice]
        self.initial_angle[ids] = self.dof[choice, 16, 0]
        hands = env.hand_indices[ids].long()
        objects = env.object_indices[ids].long()
        offset = env.root_state_tensor[hands,:3] - self.roots[choice,0,:3]
        for actor_ids, index in ((hands,0),(objects,1)):
            roots = self.roots[choice,index].clone()
            roots[:,:3] += offset
            env.root_state_tensor[actor_ids] = roots
        env.dof_state.view(env.num_envs,17,2)[ids] = self.dof[choice]
        env.prev_targets[ids] = self.targets[choice]
        env.cur_targets[ids] = self.targets[choice]
        env.init_pose_buf[ids] = self.dof[choice,:,0]
        env.obs_buf_lag_history[ids] = self.history[choice]
        env.e002_contact_state[ids] = self.contact[choice]
        env.e002_contact_acquired[ids] = True
        for name in ('e002_theta_prev','e002_high_water','e002_paid_high_water'):
            getattr(env,name)[ids] = self.dof[choice,16,0]
        for key, value in self.cycle.items():
            getattr(env.e002_cycle,key)[ids] = value[choice]
        # Imported history has no earned events in this new episode. Keep the
        # angle high-water and potential so initialization cannot pay old work.
        for key in ('phase_step','event_streak','cycles','drive_events','release_events',
                    'return_events','last_event','last_reason'):
            getattr(env.e002_cycle,key)[ids] = 0
        # hand_joint_pos is written by reset_idx BEFORE reset noise and remains
        # the episode's nominal initial pose. It is not a live joint sensor.
        # Keep the canonical task context when jumping into an intermediate
        # state; proprioception already contains the restored physical pose.
        actors = torch.cat((hands,objects)).to(torch.int32)
        hand_actor_indices = hands.to(torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.root_state_tensor),
            gymtorch.unwrap_tensor(actors),len(actors))
        env.gym.set_dof_state_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.dof_state),
            gymtorch.unwrap_tensor(actors),len(actors))
        env.gym.set_dof_position_target_tensor_indexed(env.sim,gymtorch.unwrap_tensor(env.prev_targets),
            gymtorch.unwrap_tensor(hand_actor_indices),len(hands))
