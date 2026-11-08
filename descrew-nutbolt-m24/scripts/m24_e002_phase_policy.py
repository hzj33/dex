"""Measured-stage composition of two independently learned PPO policies.

Local two-option inference adaptation of Sutton, Precup & Singh (1999).
The stage selector is the existing measured contact/progress state machine;
this is not a learned high-level selector or a single consolidated network.
"""
from pathlib import Path
import hashlib
import json
import torch
from dexscrew.algo.ppo.ppo import PPO


class MeasuredPhasePolicy:
    def __init__(self, drive, env, output, cfg):
        assert cfg.test and env.e002_cycle is not None
        self.drive = drive
        self.env = env
        self.checkpoint = str(cfg.release_policy_checkpoint)
        self.release = PPO(env, str(Path(output)/'release_policy'), full_config=cfg)
        self.release.restore_test(self.checkpoint)
        self.release.set_eval()
        self.metadata = dict(mode='measured_stage_two_ppo_policies',
            release_checkpoint=self.checkpoint,
            release_checkpoint_sha256=hashlib.sha256(Path(self.checkpoint).read_bytes()).hexdigest(),
            selection='drive PPO in phase0; release PPO after measured drive event',
            scripted_joint_actions=False,learned_high_level_selector=False)
        self.reference = None
        if cfg.get('release_hand_reference_source'):
            source = Path(str(cfg.release_hand_reference_source))
            seed = int(cfg.release_hand_reference_seed)
            row = next(r for r in json.loads(source.read_text())['states'] if r['step']==seed)
            self.reference = torch.tensor(row['reset_seed_state']['dof_state'])[:16,0]
            self.metadata['diagnostic_hand_reference_override'] = dict(seed=seed,source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),diagnostic_only=True)

    def _eval_select_action(self, obs):
        drive_action, drive_extra = self.drive._eval_select_action(obs)
        release_obs = obs
        if self.reference is not None:
            release_obs = dict(obs)
            release_obs['priv_info'] = obs['priv_info'].clone()
            start,end = self.env.priv_info_dict['hand_joint_pos']
            release_obs['priv_info'][:,start:end] = self.reference.to(release_obs['priv_info'].device)
        release_action, release_extra = self.release._eval_select_action(release_obs)
        mask = (self.env.e002_cycle.phase != 0).to(drive_action.device)[:, None]
        action = torch.where(mask, release_action, drive_action)
        assert drive_extra.keys() == release_extra.keys()
        extra = {key:torch.where(mask, release_extra[key],value) for key,value in drive_extra.items()}
        return action,extra
