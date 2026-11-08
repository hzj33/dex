# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: In-Hand Object Rotation via Rapid Motor Adaptation
# Copyright (c) 2022 Haozhi Qi
# Licensed under MIT License
# https://github.com/HaozhiQi/hora/
# --------------------------------------------------------

import os
import time
import torch
import numpy as np
from termcolor import cprint

from dexscrew.algo.eval_select import EvalSelectMixin
from dexscrew.utils.misc import AverageScalarMeter, tprint
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from tensorboardX import SummaryWriter


class ProprioAdapt(EvalSelectMixin, object):
    def __init__(self, env, output_dir, full_config, student_dim=24):
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = self.ppo_config['num_actors']
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.proprio_dim = self.ppo_config.get('proprio_dim', 24)
        self.student_obs_shape = (student_dim * 3,)
        # ---- Priv Info ----
        self.priv_info = self.ppo_config['priv_info']
        self.normalize_priv = self.ppo_config['normalize_priv']
        self.priv_info_dim = self.env.priv_info_dim
        self.proprio_adapt = self.ppo_config['proprio_adapt']
        self.proprio_hist_dim = self.env.prop_hist_len
        # ---- Critic Info
        self.asymm_actor_critic = self.ppo_config.get('asymm_actor_critic', False)
        self.critic_info_dim = self.ppo_config.get('critic_info_dim', 0)
        # ---- Point Cloud / Proprio Hist Info
        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.proprio_len = self.ppo_config['proprio_len']
        self.use_point_cloud_info = self.ppo_config['use_point_cloud_info']
        self.normalize_point_cloud = self.ppo_config['normalize_point_cloud']
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'priv_mlp_units': self.network_config.priv_mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'priv_info': self.priv_info,
            'proprio_adapt': self.proprio_adapt,
            'priv_info_dim': self.priv_info_dim,
            'critic_info_dim': self.critic_info_dim,
            'asymm_actor_critic': self.asymm_actor_critic,
            'point_mlp_units': self.network_config.point_mlp.units,
            'use_point_cloud_info': self.use_point_cloud_info,
            'proprio_len': self.proprio_len,
            'proprio_dim': self.proprio_dim,
            'teacher_point_tanh': self.ppo_config.get('teacher_point_tanh', False),
        }

        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd((self.proprio_hist_dim, self.proprio_dim)).to(self.device)
        self.sa_mean_std.train()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.point_cloud_mean_std.eval()
        # ---- Output Dir ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, 'stage2_nn')
        self.tb_dir = os.path.join(self.output_dir, 'stage2_tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        writer = SummaryWriter(self.tb_dir)
        self.writer = writer
        self.direct_info = {}
        # ---- Misc ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.best_rewards = -10000
        self.agent_steps = 0
        # ---- Optim ----
        self.student_trainable_param_patterns = self._resolve_trainable_param_patterns()
        adapt_params = []
        for name, p in self.model.named_parameters():
            if self._is_trainable_param(name):
                adapt_params.append(p)
            else:
                p.requires_grad = False
        if not adapt_params:
            raise ValueError(
                "No trainable params matched train.ppo.student_trainable_param_patterns="
                f"{list(self.student_trainable_param_patterns)}"
            )
        self.optim = torch.optim.Adam(adapt_params, lr=3e-4)
        self.trainable_param_count = int(sum(p.numel() for p in adapt_params))
        tprint(
            "ProprioAdapt trainable patterns: "
            f"{list(self.student_trainable_param_patterns)} | "
            f"trainable params: {self.trainable_param_count}"
        )
        # ---- Training Misc
        self.internal_counter = 0
        self.latent_loss_stat = 0
        self.loss_stat_cnt = 0
        batch_size = self.num_actors
        self.step_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.step_length = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.test_num_steps = int(full_config.get("test_num_steps", 0))
        self._init_eval_select(role="student", artifact_ext="ckpt")

    def _resolve_trainable_param_patterns(self):
        patterns = self.ppo_config.get("student_trainable_param_patterns", ["adapt_tconv"])
        if isinstance(patterns, (str, bytes)):
            patterns = [patterns]
        elif not isinstance(patterns, (list, tuple)):
            # Hydra's ListConfig is iterable; keep parsing logic generic to avoid hard dependency.
            try:
                patterns = list(patterns)
            except TypeError as exc:
                raise TypeError(
                    "train.ppo.student_trainable_param_patterns must be list/tuple/str, "
                    f"got {type(patterns)}"
                ) from exc
        if not isinstance(patterns, (list, tuple)):
            raise TypeError(
                "train.ppo.student_trainable_param_patterns must be list/tuple/str, "
                f"got {type(patterns)}"
            )
        cleaned = []
        for p in patterns:
            token = str(p).strip()
            if token:
                cleaned.append(token)
        if not cleaned:
            raise ValueError(
                "train.ppo.student_trainable_param_patterns cannot be empty"
            )
        return tuple(cleaned)

    def _is_trainable_param(self, param_name):
        return any(pattern in param_name for pattern in self.student_trainable_param_patterns)

    def recon_criterion(self, out, target):
        return (out - target).pow(2)

    def _update_env_info(self, info):
        """Log numeric fields from env info dict for richer acceptance metrics."""
        if not isinstance(info, dict):
            return
        for k, v in info.items():
            val = None
            if torch.is_tensor(v):
                if v.numel() == 0:
                    continue
                val = float(v.float().mean().detach().cpu())
            elif isinstance(v, (int, float, np.number)):
                val = float(v)
            if val is None or not np.isfinite(val):
                continue
            self.direct_info[f"env/{k}"] = val

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    def _eval_select_action(self, obs_dict):
        if self.normalize_point_cloud:
            point_cloud_info = self.point_cloud_mean_std(
                obs_dict['point_cloud_info'].reshape(-1, 3)
            ).reshape((obs_dict['obs'].shape[0], -1, 3))
        else:
            point_cloud_info = obs_dict['point_cloud_info']
        input_dict = {
            'obs': self.running_mean_std(obs_dict['obs']),
            'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
            'point_cloud_info': point_cloud_info,
        }
        mu, _, _ = self.model.act_inference(input_dict)
        return torch.clamp(mu, -1.0, 1.0), {}

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        eval_reward_sum = 0.0
        eval_done_sum = 0.0
        while True:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(obs_dict['point_cloud_info'].reshape(-1, 3)).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']),
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
                'point_cloud_info': point_cloud_info,
            }
            mu, extrin, extrin_gt = self.model.act_inference(input_dict)
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            c += 1
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
            print(f"Step {c}")
            if self.test_num_steps > 0 and c >= self.test_num_steps:
                avg_reward = eval_reward_sum / float(c)
                avg_done_rate = eval_done_sum / float(c)
                print(
                    "EvalSummary "
                    f"steps={c} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                )
                break

    def collect_rollout(self, num_steps=256, save_path=None, save_point_cloud=True):
        self.set_eval()
        obs_dict = self.env.reset()

        obs_buf = []
        proprio_hist_buf = []
        priv_info_buf = []
        point_cloud_buf = [] if save_point_cloud else None
        action_buf = []
        reward_buf = []
        done_buf = []
        done_rate_buf = []
        extra_scalar_buf = {}

        with torch.no_grad():
            for _ in range(int(num_steps)):
                if self.normalize_point_cloud:
                    point_cloud_info = self.point_cloud_mean_std(
                        obs_dict["point_cloud_info"].reshape(-1, 3)
                    ).reshape((obs_dict["obs"].shape[0], -1, 3))
                else:
                    point_cloud_info = obs_dict["point_cloud_info"]
                input_dict = {
                    "obs": self.running_mean_std(obs_dict["obs"]),
                    "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
                    "point_cloud_info": point_cloud_info,
                }
                mu, extrin, extrin_gt = self.model.act_inference(input_dict)
                mu = torch.clamp(mu, -1.0, 1.0)
                next_obs_dict, rewards, done, info = self.env.step(mu)

                obs_buf.append(obs_dict["obs"].detach().cpu())
                proprio_hist_buf.append(obs_dict["proprio_hist"].detach().cpu())
                priv_info_buf.append(obs_dict["priv_info"].detach().cpu())
                if save_point_cloud:
                    point_cloud_buf.append(obs_dict["point_cloud_info"].detach().cpu())
                action_buf.append(mu.detach().cpu())
                reward_buf.append(rewards.detach().cpu())
                done_buf.append(done.detach().cpu())
                done_rate_buf.append(done.float().mean().item())

                for key, value in info.items():
                    scalar_value = None
                    if torch.is_tensor(value):
                        scalar_value = float(value.detach().float().mean().cpu())
                    elif isinstance(value, (float, int, bool, np.number)):
                        scalar_value = float(value)
                    if scalar_value is None:
                        continue
                    extra_scalar_buf.setdefault(str(key), []).append(scalar_value)

                obs_dict = next_obs_dict

        payload = {
            "meta": {
                "num_steps": int(num_steps),
                "num_envs": int(self.env.num_envs),
                "normalize_input": bool(True),
                "normalize_priv": bool(self.normalize_priv),
                "normalize_point_cloud": bool(self.normalize_point_cloud),
                "save_point_cloud": bool(save_point_cloud),
            },
            "obs": torch.stack(obs_buf, dim=0),
            "proprio_hist": torch.stack(proprio_hist_buf, dim=0),
            "priv_info": torch.stack(priv_info_buf, dim=0),
            "actions": torch.stack(action_buf, dim=0),
            "rewards": torch.stack(reward_buf, dim=0),
            "dones": torch.stack(done_buf, dim=0),
            "done_rate_per_step": torch.tensor(done_rate_buf, dtype=torch.float32),
            "extras": {
                key: torch.tensor(values, dtype=torch.float32)
                for key, values in extra_scalar_buf.items()
            },
        }
        if save_point_cloud:
            payload["point_cloud_info"] = torch.stack(point_cloud_buf, dim=0)

        if not save_path:
            ts = time.strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.join(self.output_dir, "student_rollouts")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"rollout_steps{int(num_steps)}_{ts}.pt")
        else:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)

        torch.save(payload, save_path)

        mean_reward = payload["rewards"].mean().item()
        mean_done_rate = payload["done_rate_per_step"].mean().item()
        print(
            f"Saved rollout dataset: {save_path}\n"
            f"collect_steps={int(num_steps)} | num_envs={self.env.num_envs} | "
            f"mean_reward={mean_reward:.4f} | mean_done_rate={mean_done_rate:.4f}"
        )
        return save_path

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(obs_dict['point_cloud_info'].reshape(-1, 3)).reshape((obs_dict['obs'].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict['point_cloud_info']
            input_dict = {
                'obs': self.running_mean_std(obs_dict['obs']).detach(),
                'priv_info': self.priv_mean_std(obs_dict['priv_info']) if self.normalize_priv else obs_dict['priv_info'],
                'proprio_hist': self.sa_mean_std(obs_dict['proprio_hist'].detach()),
                'point_cloud_info': point_cloud_info,
            }
            mu, _, _, e, e_gt = self.model._actor_critic(input_dict)
            latent_loss = ((e - e_gt.detach()) ** 2).mean() # Latent loss (student-teacher)

            teacher_obs_input = torch.cat([input_dict['obs'], e_gt.detach()], dim=-1)
            with torch.no_grad():
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = self.model.mu(teacher_x)

            #  Behavior cloning loss between student action and teacher action
            bc_loss = torch.sum(self.recon_criterion(torch.clamp(mu, -1, 1),torch.clamp(teacher_mu, -1, 1)), dim=-1).mean()

            # Total loss: latent loss + behavior cloning loss
            loss = bc_loss + latent_loss
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = mu.detach()
            mu = torch.clamp(mu, -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            # ---- statistics
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info['latent_loss'] = float(latent_loss.detach().cpu())
            self.direct_info['bc_loss'] = float(bc_loss.detach().cpu())
            self.direct_info['total_loss'] = float(loss.detach().cpu())
            self.direct_info['done_rate'] = float(done.float().mean().detach().cpu())
            self._update_env_info(info)

            self.log_tensorboard()

            if self.agent_steps % 1e8 == 0:
                self.save(os.path.join(self.nn_dir, f'{self.agent_steps // 1e8}00m'))
                self.save(os.path.join(self.nn_dir, f'model_last'))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                if self.eval_select_enabled:
                    self.save(os.path.join(self.nn_dir, f'model_best_train'))
                else:
                    self.save(os.path.join(self.nn_dir, f'model_best'))
                self.best_rewards = mean_rewards

            eval_metrics = self._run_eval_select_if_due(
                "EVAL/student",
                train_reward=mean_rewards,
                eval_best_stem="model_best_eval",
                alias_stems=("model_best", "model_best_deploy"),
            )
            if eval_metrics is not None:
                obs_dict = eval_metrics["final_obs_dict"]

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | ' \
                          f'Last FPS: {last_fps:.1f} | ' \
                          f'Current Best: {self.best_rewards:.2f}'
            tprint(info_string)

    def log_tensorboard(self):
        self.writer.add_scalar('episode_rewards/step', self.mean_eps_reward.get_mean(), self.agent_steps)
        self.writer.add_scalar('episode_lengths/step', self.mean_eps_length.get_mean(), self.agent_steps)
        for k, v in self.direct_info.items():
            self.writer.add_scalar(f'{k}/frame', v, self.agent_steps)

    def restore_train(self, fn):
        checkpoint = torch.load(fn)
        cprint('careful, using non-strict matching', 'red', attrs=['bold'])
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        if 'priv_mean_std' in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint['priv_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
        self.model.load_state_dict(checkpoint['model'])
        self.sa_mean_std.load_state_dict(checkpoint['sa_mean_std'])
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint['point_cloud_mean_std'])

    def save(self, name):
        weights = {
            'model': self.model.state_dict(),
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights['sa_mean_std'] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights['priv_mean_std'] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights['point_cloud_mean_std'] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f'{name}.ckpt')
