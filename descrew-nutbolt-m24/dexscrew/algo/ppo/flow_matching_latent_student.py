# --------------------------------------------------------
# Flow-matching latent student distillation (ODE velocity field)
# --------------------------------------------------------

import math
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from termcolor import cprint
from tensorboardX import SummaryWriter

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.utils.misc import tprint


class SinusoidalPosEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = int(dim)

    def forward(self, t):
        if t.dim() > 1:
            t = t.reshape(t.shape[0])
        t = t.float()
        half_dim = self.dim // 2
        if half_dim <= 0:
            return t.unsqueeze(-1)
        denom = max(half_dim - 1, 1)
        freq = torch.exp(
            torch.arange(half_dim, device=t.device, dtype=t.dtype)
            * (-math.log(10000.0) / float(denom))
        )
        args = t.unsqueeze(-1) * freq.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), mode="constant", value=0.0)
        return emb


class FlowMatchingHead(nn.Module):
    def __init__(self, proprio_hist_dim, proprio_dim, latent_dim, hidden_dim=256, t_dim=64):
        super().__init__()
        self.hist_encoder = nn.Sequential(
            nn.Linear(proprio_hist_dim * proprio_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.t_embed = SinusoidalPosEmbed(t_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + t_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, proprio_hist, x_t, t_cont):
        hist_feat = self.hist_encoder(proprio_hist.reshape(proprio_hist.shape[0], -1))
        t_feat = self.t_embed(t_cont)
        denoise_in = torch.cat([hist_feat, x_t, t_feat], dim=-1)
        return self.denoiser(denoise_in)


class FlowMatchingLatentStudent(ProprioAdapt):
    """Flow-matching latent student with ODE-style inference."""

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)

        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_flow_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_flow_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        for p in self.model.parameters():
            p.requires_grad = False

        self.latent_dim = self.model.adapt_tconv.low_dim_proj.out_features
        self.flow_model = FlowMatchingHead(
            proprio_hist_dim=self.proprio_hist_dim,
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            hidden_dim=int(self.ppo_config.get("flow_hidden_dim", 256)),
            t_dim=int(self.ppo_config.get("flow_t_dim", 64)),
        ).to(self.device)

        self.flow_loss_coef = float(self.ppo_config.get("flow_loss_coef", 1.0))
        self.flow_sigma_min = float(self.ppo_config.get("flow_sigma_min", 1e-4))
        self.flow_infer_steps = max(1, int(self.ppo_config.get("flow_infer_steps", 1)))
        self.stochastic_infer = bool(self.ppo_config.get("flow_stochastic_infer", False))
        self.flow_train_init_mode = str(
            self.ppo_config.get("flow_train_init_mode", "gaussian")
        ).lower()
        self.flow_train_align_infer = bool(
            self.ppo_config.get("flow_train_align_infer", False)
        )
        self.flow_rollout_bc_coef = float(self.ppo_config.get("flow_rollout_bc_coef", 0.0))
        self.bc_loss_coef = float(self.ppo_config.get("bc_loss_coef", 1.0))
        self.base_action_anchor_coef = float(
            self.ppo_config.get(
                "base_action_anchor_coef",
                self.ppo_config.get("diffusion_base_action_anchor_coef", 0.0),
            )
        )
        self.action_l2_coef = float(self.ppo_config.get("flow_action_l2_coef", 0.0))
        self.optim = torch.optim.Adam(
            self.flow_model.parameters(),
            lr=float(self.ppo_config.get("flow_lr", 3e-4)),
        )
        tprint(
            "FlowMatchingLatent config: "
            f"infer_steps={self.flow_infer_steps}, "
            f"sigma_min={self.flow_sigma_min}, "
            f"stochastic_infer={self.stochastic_infer}, "
            f"train_init_mode={self.flow_train_init_mode}, "
            f"train_align_infer={self.flow_train_align_infer}, "
            f"rollout_bc_coef={self.flow_rollout_bc_coef}"
        )

    def _init_latent(self, batch_size, mode):
        mode = str(mode).lower()
        if mode == "match_infer":
            mode = "gaussian" if self.stochastic_infer else "zero"
        if mode == "gaussian":
            return torch.randn(batch_size, self.latent_dim, device=self.device)
        if mode == "zero":
            return torch.zeros(batch_size, self.latent_dim, device=self.device)
        raise ValueError(f"Unsupported flow latent init mode: {mode}")

    def _sample_latent_rollout(self, proprio_hist, infer_steps=None, init_mode=None):
        infer_steps = max(1, int(self.flow_infer_steps if infer_steps is None else infer_steps))
        init_mode = "match_infer" if init_mode is None else init_mode
        batch_size = proprio_hist.shape[0]
        latent = self._init_latent(batch_size, init_mode)

        dt = 1.0 / float(infer_steps)
        for idx in range(infer_steps):
            t_cur = 1.0 - (float(idx) / float(max(1, infer_steps)))
            t = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.float32)
            v_pred = self.flow_model(proprio_hist, latent, t)
            latent = latent - dt * v_pred
        return latent

    @torch.no_grad()
    def sample_latent(self, proprio_hist):
        return self._sample_latent_rollout(proprio_hist)

    def set_eval(self):
        super().set_eval()
        self.flow_model.eval()

    def _eval_select_action(self, obs_dict):
        if self.normalize_point_cloud:
            point_cloud_info = self.point_cloud_mean_std(
                obs_dict["point_cloud_info"].reshape(-1, 3)
            ).reshape((obs_dict["obs"].shape[0], -1, 3))
        else:
            point_cloud_info = obs_dict["point_cloud_info"]

        proprio_hist = self.sa_mean_std(obs_dict["proprio_hist"].detach())
        obs = self.running_mean_std(obs_dict["obs"])
        input_dict = {
            "obs": obs,
            "priv_info": self.priv_mean_std(obs_dict["priv_info"])
            if self.normalize_priv
            else obs_dict["priv_info"],
            "proprio_hist": proprio_hist,
            "point_cloud_info": point_cloud_info,
        }
        latent = self.sample_latent(proprio_hist)
        student_obs_input = torch.cat([obs, latent], dim=-1)
        student_x = self.model.actor_mlp(student_obs_input)
        mu = self.model.mu(student_x)
        return torch.clamp(mu, -1.0, 1.0), {}

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        eval_reward_sum = 0.0
        eval_done_sum = 0.0
        eval_latent_mse_sum = 0.0
        eval_latent_l1_sum = 0.0
        eval_action_mse_sum = 0.0
        while True:
            with torch.no_grad():
                if self.normalize_point_cloud:
                    point_cloud_info = self.point_cloud_mean_std(
                        obs_dict["point_cloud_info"].reshape(-1, 3)
                    ).reshape((obs_dict["obs"].shape[0], -1, 3))
                else:
                    point_cloud_info = obs_dict["point_cloud_info"]

                proprio_hist = self.sa_mean_std(obs_dict["proprio_hist"].detach())
                obs = self.running_mean_std(obs_dict["obs"])
                input_dict = {
                    "obs": obs,
                    "priv_info": self.priv_mean_std(obs_dict["priv_info"])
                    if self.normalize_priv
                    else obs_dict["priv_info"],
                    "proprio_hist": proprio_hist,
                    "point_cloud_info": point_cloud_info,
                }
                _, _, _, _, e_gt = self.model._actor_critic(input_dict)
                latent = self.sample_latent(proprio_hist)

                student_obs_input = torch.cat([obs, latent], dim=-1)
                student_x = self.model.actor_mlp(student_obs_input)
                mu = torch.clamp(self.model.mu(student_x), -1.0, 1.0)

                teacher_obs_input = torch.cat([obs, e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)

                latent_mse = ((latent - e_gt.detach()) ** 2).mean()
                latent_l1 = (latent - e_gt.detach()).abs().mean()
                action_mse_to_teacher = ((mu - teacher_mu) ** 2).mean()

                obs_dict, r, done, _ = self.env.step(mu)
            c += 1
            print(f"Step {c}")
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
                eval_latent_mse_sum += float(latent_mse.detach().cpu())
                eval_latent_l1_sum += float(latent_l1.detach().cpu())
                eval_action_mse_sum += float(action_mse_to_teacher.detach().cpu())
                if c >= self.test_num_steps:
                    avg_reward = eval_reward_sum / float(c)
                    avg_done_rate = eval_done_sum / float(c)
                    print(
                        "EvalSummary "
                        f"steps={c} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                    )
                    print(
                        "EvalReconSummary "
                        f"steps={c} mode=flow_matching "
                        f"latent_mse={eval_latent_mse_sum / float(c):.6f} "
                        f"latent_l1={eval_latent_l1_sum / float(c):.6f} "
                        f"action_mse_to_teacher={eval_action_mse_sum / float(c):.6f}"
                    )
                    break

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

            with torch.no_grad():
                _, _, _, _, e_gt = self.model._actor_critic(input_dict)
                teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)
                base_student_mu = None
                if self.base_action_anchor_coef > 0.0:
                    base_latent_anchor = torch.tanh(
                        self.model.adapt_tconv(input_dict["proprio_hist"])
                    )
                    base_obs_input = torch.cat([input_dict["obs"], base_latent_anchor], dim=-1)
                    base_x = self.model.actor_mlp(base_obs_input)
                    base_student_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

            target_latent = e_gt.detach()
            x1 = torch.randn_like(target_latent)
            batch_size = target_latent.shape[0]
            t = torch.rand((batch_size,), device=self.device)
            t = torch.clamp(t, min=self.flow_sigma_min, max=1.0)
            x_t = (1.0 - t.unsqueeze(-1)) * target_latent + t.unsqueeze(-1) * x1
            v_target = x1 - target_latent
            v_pred = self.flow_model(input_dict["proprio_hist"], x_t, t)
            flow_loss = ((v_pred - v_target) ** 2).mean()

            pred_latent = x_t - t.unsqueeze(-1) * v_pred
            student_obs_input = torch.cat([input_dict["obs"], pred_latent], dim=-1)
            student_x = self.model.actor_mlp(student_obs_input)
            student_mu = self.model.mu(student_x)
            student_mu_clamped = torch.clamp(student_mu, -1, 1)
            teacher_mu_clamped = torch.clamp(teacher_mu, -1, 1)
            bc_loss = torch.sum(
                self.recon_criterion(student_mu_clamped, teacher_mu_clamped),
                dim=-1,
            ).mean()
            if self.flow_train_align_infer and self.flow_rollout_bc_coef > 0.0:
                rollout_latent = self._sample_latent_rollout(
                    input_dict["proprio_hist"],
                    infer_steps=self.flow_infer_steps,
                    init_mode=self.flow_train_init_mode,
                )
                rollout_obs_input = torch.cat([input_dict["obs"], rollout_latent], dim=-1)
                rollout_x = self.model.actor_mlp(rollout_obs_input)
                rollout_mu = self.model.mu(rollout_x)
                rollout_mu_clamped = torch.clamp(rollout_mu, -1, 1)
                rollout_bc_loss = torch.sum(
                    self.recon_criterion(rollout_mu_clamped, teacher_mu_clamped),
                    dim=-1,
                ).mean()
            else:
                rollout_bc_loss = torch.zeros((), device=self.device, dtype=bc_loss.dtype)
            if base_student_mu is not None:
                base_action_anchor_loss = torch.sum(
                    self.recon_criterion(student_mu_clamped, base_student_mu),
                    dim=-1,
                ).mean()
            else:
                base_action_anchor_loss = torch.zeros(
                    (), device=self.device, dtype=bc_loss.dtype
                )
            action_l2_loss = student_mu_clamped.pow(2).mean()

            loss = (
                self.flow_loss_coef * flow_loss
                + self.bc_loss_coef * bc_loss
                + self.flow_rollout_bc_coef * rollout_bc_loss
                + self.base_action_anchor_coef * base_action_anchor_loss
                + self.action_l2_coef * action_l2_loss
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            mu_env = torch.clamp(student_mu.detach(), -1.0, 1.0)
            obs_dict, r, done, info = self.env.step(mu_env)
            self.agent_steps += self.batch_size

            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])

            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            self.direct_info["flow_loss"] = float(flow_loss.detach().cpu())
            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["rollout_bc_loss"] = float(rollout_bc_loss.detach().cpu())
            self.direct_info["base_action_anchor_loss"] = float(
                base_action_anchor_loss.detach().cpu()
            )
            self.direct_info["action_l2_loss"] = float(action_l2_loss.detach().cpu())
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self._update_env_info(info)
            self.log_tensorboard()

            if self.agent_steps % 1e8 == 0:
                self.save(os.path.join(self.nn_dir, f"{self.agent_steps // 1e8}00m"))
                self.save(os.path.join(self.nn_dir, "model_last"))

            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_rewards:
                if self.eval_select_enabled:
                    self.save(os.path.join(self.nn_dir, "model_best_train"))
                else:
                    self.save(os.path.join(self.nn_dir, "model_best"))
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
            tprint(
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        cprint("careful, using non-strict matching", "red", attrs=["bold"])
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        else:
            cprint(
                "Checkpoint missing sa_mean_std during train restore; using current defaults.",
                "yellow",
            )
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])
        if "flow_model" in checkpoint:
            self.flow_model.load_state_dict(checkpoint["flow_model"])
            cprint("Loaded flow_model for train resume", "green")
        if "flow_optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["flow_optim"])
            cprint("Loaded flow optimizer state for train resume", "green")
        if "agent_steps" in checkpoint:
            self.agent_steps = checkpoint["agent_steps"]
        else:
            cprint(
                "Checkpoint missing agent_steps during train restore; resume will start from current counter.",
                "yellow",
            )

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "flow_model" in checkpoint:
            self.flow_model.load_state_dict(checkpoint["flow_model"])
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        else:
            cprint(
                "Checkpoint missing sa_mean_std during test restore; using current defaults.",
                "yellow",
            )
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
            "flow_model": self.flow_model.state_dict(),
            "flow_optim": self.optim.state_dict(),
            "agent_steps": self.agent_steps,
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights["priv_mean_std"] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f"{name}.ckpt")
