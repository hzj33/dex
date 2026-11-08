# --------------------------------------------------------
# Pure BC student baseline (separate from ProprioAdapt)
# --------------------------------------------------------

import os
import shutil
import time
import torch
from tensorboardX import SummaryWriter

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.utils.misc import tprint


class PureBC(ProprioAdapt):
    """BC-only student baseline.

    Keeps the same student architecture and adapter-only training range as ProprioAdapt,
    but removes latent distillation from the objective (loss = action BC only).
    """

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)
        legacy_nn_dir = self.nn_dir
        legacy_tb_dir = self.tb_dir
        self.writer.close()
        # Remove legacy ProprioAdapt artifact dirs to avoid mixed outputs.
        shutil.rmtree(legacy_nn_dir, ignore_errors=True)
        shutil.rmtree(legacy_tb_dir, ignore_errors=True)
        # Keep PureBC artifacts isolated from ProprioAdapt artifacts.
        self.nn_dir = os.path.join(self.output_dir, "stage2_bc_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_bc_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

    def train(self):
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size
        while self.agent_steps <= 1e9:
            if self.normalize_point_cloud:
                point_cloud_info = self.point_cloud_mean_std(
                    obs_dict["point_cloud_info"].reshape(-1, 3)
                ).reshape((obs_dict["obs"].shape[0], -1, 3))
            else:
                point_cloud_info = obs_dict["point_cloud_info"]

            input_dict = {
                "obs": self.running_mean_std(obs_dict["obs"]).detach(),
                "priv_info": self.priv_mean_std(obs_dict["priv_info"])
                if self.normalize_priv
                else obs_dict["priv_info"],
                "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
                "point_cloud_info": point_cloud_info,
            }

            mu, _, _, _, e_gt = self.model._actor_critic(input_dict)
            teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
            with torch.no_grad():
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = self.model.mu(teacher_x)

            # Pure BC objective only (no latent loss).
            bc_loss = torch.sum(
                self.recon_criterion(torch.clamp(mu, -1, 1), torch.clamp(teacher_mu, -1, 1)),
                dim=-1,
            ).mean()
            loss = bc_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu = torch.clamp(mu.detach(), -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu)
            self.agent_steps += self.batch_size

            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self._update_env_info(info)
            self.log_tensorboard()

            if self.agent_steps % 1e8 == 0:
                self.save(os.path.join(self.nn_dir, f"{self.agent_steps // 1e8}00m"))
                self.save(os.path.join(self.nn_dir, "model_last"))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                self.save(os.path.join(self.nn_dir, "model_best"))
                self.best_rewards = mean_rewards

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )
            tprint(info_string)
