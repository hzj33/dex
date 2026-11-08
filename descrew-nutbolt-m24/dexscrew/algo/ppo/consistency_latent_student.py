# --------------------------------------------------------
# Consistency latent student distillation (single-step x0 mapping)
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


class ConsistencyHead(nn.Module):
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


class ConsistencyLatentStudent(ProprioAdapt):
    """Consistency-style latent student with one-step x0 prediction."""

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)

        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_consistency_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_consistency_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        for p in self.model.parameters():
            p.requires_grad = False

        self.latent_dim = self.model.adapt_tconv.low_dim_proj.out_features
        self.consistency_model = ConsistencyHead(
            proprio_hist_dim=self.proprio_hist_dim,
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            hidden_dim=int(self.ppo_config.get("consistency_hidden_dim", 256)),
            t_dim=int(self.ppo_config.get("consistency_t_dim", 64)),
        ).to(self.device)

        self.consistency_loss_coef = float(self.ppo_config.get("consistency_loss_coef", 1.0))
        self.consistency_boundary_coef = float(
            self.ppo_config.get("consistency_boundary_coef", 0.5)
        )
        self.consistency_num_scales = max(
            2, int(self.ppo_config.get("consistency_num_scales", 10))
        )
        self.consistency_ema_decay = float(self.ppo_config.get("consistency_ema_decay", 0.999))
        self.consistency_use_ema_target = bool(
            self.ppo_config.get("consistency_use_ema_target", False)
        )
        self.bc_loss_coef = float(self.ppo_config.get("bc_loss_coef", 1.0))
        self.base_action_anchor_coef = float(
            self.ppo_config.get(
                "base_action_anchor_coef",
                self.ppo_config.get("diffusion_base_action_anchor_coef", 0.0),
            )
        )
        self.action_l2_coef = float(self.ppo_config.get("consistency_action_l2_coef", 0.0))
        self.stochastic_infer = bool(self.ppo_config.get("consistency_stochastic_infer", False))
        self.consistency_infer_steps = max(
            1, int(self.ppo_config.get("consistency_infer_steps", 1))
        )
        self.consistency_infer_use_ema = bool(
            self.ppo_config.get("consistency_infer_use_ema", False)
        )
        self.train_align_infer = bool(
            self.ppo_config.get("consistency_train_align_infer", False)
        )
        self.train_align_use_ema = bool(
            self.ppo_config.get("consistency_train_align_use_ema", False)
        )
        self.student_max_agent_steps = int(self.ppo_config.get("student_max_agent_steps", 0))
        self.student_progress_log_interval_agent_steps = int(
            self.ppo_config.get("student_progress_log_interval_agent_steps", 0)
        )
        self._next_progress_log_agent_steps = self.student_progress_log_interval_agent_steps
        # Optional obs-noise curriculum: linearly/staged ramp env observation noise during training.
        self.obs_noise_curriculum = bool(
            self.ppo_config.get("consistency_obs_noise_curriculum", False)
        )
        self.obs_noise_curriculum_start = int(
            self.ppo_config.get("consistency_obs_noise_curriculum_start", 0)
        )
        self.obs_noise_curriculum_steps = max(
            1, int(self.ppo_config.get("consistency_obs_noise_curriculum_steps", 1))
        )
        self.obs_noise_curriculum_mode = str(
            self.ppo_config.get("consistency_obs_noise_curriculum_mode", "linear")
        )
        self.obs_noise_curriculum_t_phase_ratio = float(
            self.ppo_config.get("consistency_obs_noise_curriculum_t_phase_ratio", 0.6)
        )
        self.obs_noise_curriculum_t_phase_ratio = min(
            max(self.obs_noise_curriculum_t_phase_ratio, 0.0), 1.0
        )
        self.obs_noise_curriculum_hold_ratio = float(
            self.ppo_config.get("consistency_obs_noise_curriculum_hold_ratio", 0.2)
        )
        self.obs_noise_curriculum_hold_ratio = min(
            max(self.obs_noise_curriculum_hold_ratio, 0.0), 1.0
        )
        self.obs_noise_e_base = float(getattr(self.env, "random_obs_noise_e_scale", 0.0))
        self.obs_noise_t_base = float(getattr(self.env, "random_obs_noise_t_scale", 0.0))
        self.obs_noise_e_target = float(
            self.ppo_config.get("consistency_obs_noise_e_target", self.obs_noise_e_base)
        )
        self.obs_noise_t_target = float(
            self.ppo_config.get("consistency_obs_noise_t_target", self.obs_noise_t_base)
        )
        self.consistency_lr_base = float(self.ppo_config.get("consistency_lr", 3e-4))
        self.lr_schedule = str(
            self.ppo_config.get("consistency_lr_schedule", "none")
        ).lower()
        self.lr_decay_start_agent_steps = int(
            self.ppo_config.get("consistency_lr_decay_start_agent_steps", 0)
        )
        self.lr_decay_end_agent_steps = int(
            self.ppo_config.get(
                "consistency_lr_decay_end_agent_steps",
                self.student_max_agent_steps,
            )
        )
        self.lr_final_scale = float(
            self.ppo_config.get("consistency_lr_final_scale", 1.0)
        )
        self.lr_final_scale = max(0.0, self.lr_final_scale)
        if self.lr_decay_end_agent_steps <= self.lr_decay_start_agent_steps:
            self.lr_decay_end_agent_steps = self.lr_decay_start_agent_steps + 1
        self.optim = torch.optim.Adam(
            self.consistency_model.parameters(),
            lr=self.consistency_lr_base,
        )
        self.consistency_ema_model = None
        if self.consistency_use_ema_target:
            self.consistency_ema_model = ConsistencyHead(
                proprio_hist_dim=self.proprio_hist_dim,
                proprio_dim=self.proprio_dim,
                latent_dim=self.latent_dim,
                hidden_dim=int(self.ppo_config.get("consistency_hidden_dim", 256)),
                t_dim=int(self.ppo_config.get("consistency_t_dim", 64)),
            ).to(self.device)
            self.consistency_ema_model.load_state_dict(self.consistency_model.state_dict())
            self.consistency_ema_model.eval()
            for p in self.consistency_ema_model.parameters():
                p.requires_grad = False
        tprint(
            "ConsistencyLatent config: "
            f"infer_steps={self.consistency_infer_steps}, "
            f"consistency_num_scales={self.consistency_num_scales}, "
            f"infer_use_ema={self.consistency_infer_use_ema}, "
            f"stochastic_infer={self.stochastic_infer}, "
            f"train_align_infer={self.train_align_infer}, "
            f"train_align_use_ema={self.train_align_use_ema}, "
            f"max_agent_steps={self.student_max_agent_steps}, "
            f"lr={self.consistency_lr_base}, "
            f"lr_schedule={self.lr_schedule}, "
            f"lr_decay_start={self.lr_decay_start_agent_steps}, "
            f"lr_decay_end={self.lr_decay_end_agent_steps}, "
            f"lr_final_scale={self.lr_final_scale}"
        )

    def _consistency_lr_scale(self):
        if self.lr_schedule in ("", "none", "constant"):
            return 1.0
        if self.lr_schedule == "step":
            if self.agent_steps >= self.lr_decay_start_agent_steps:
                return self.lr_final_scale
            return 1.0
        if self.lr_schedule not in ("linear_decay", "linear", "cosine_decay", "cosine"):
            return 1.0

        denom = float(self.lr_decay_end_agent_steps - self.lr_decay_start_agent_steps)
        progress = (self.agent_steps - self.lr_decay_start_agent_steps) / denom
        progress = min(max(progress, 0.0), 1.0)
        if self.lr_schedule in ("cosine_decay", "cosine"):
            eased = 0.5 * (1.0 - math.cos(math.pi * progress))
        else:
            eased = progress
        return 1.0 + (self.lr_final_scale - 1.0) * eased

    def _update_consistency_lr(self):
        lr = self.consistency_lr_base * self._consistency_lr_scale()
        for group in self.optim.param_groups:
            group["lr"] = lr
        return lr

    @torch.no_grad()
    def _update_consistency_ema(self):
        if self.consistency_ema_model is None:
            return
        decay = min(max(self.consistency_ema_decay, 0.0), 1.0)
        for ema_p, cur_p in zip(
            self.consistency_ema_model.parameters(), self.consistency_model.parameters()
        ):
            ema_p.data.mul_(decay).add_(cur_p.data, alpha=(1.0 - decay))

    def _apply_obs_noise_curriculum(self):
        if not self.obs_noise_curriculum:
            return
        progress = (self.agent_steps - self.obs_noise_curriculum_start) / float(
            self.obs_noise_curriculum_steps
        )
        progress = min(max(progress, 0.0), 1.0)

        if self.obs_noise_curriculum_mode == "staged_te":
            split = self.obs_noise_curriculum_t_phase_ratio
            if split <= 0.0:
                cur_t = self.obs_noise_t_target
                e_progress = progress
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
            elif split >= 1.0:
                t_progress = progress
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            elif progress <= split:
                t_progress = progress / split
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            else:
                cur_t = self.obs_noise_t_target
                e_progress = (progress - split) / (1.0 - split)
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
        elif self.obs_noise_curriculum_mode == "staged_hold_te":
            hold = self.obs_noise_curriculum_hold_ratio
            split = max(self.obs_noise_curriculum_t_phase_ratio, hold)
            if progress <= hold:
                cur_t = self.obs_noise_t_base
                cur_e = self.obs_noise_e_base
            elif split >= 1.0:
                t_progress = (progress - hold) / max(1e-8, 1.0 - hold)
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            elif progress <= split:
                t_progress = (progress - hold) / max(1e-8, split - hold)
                cur_t = self.obs_noise_t_base + (
                    self.obs_noise_t_target - self.obs_noise_t_base
                ) * t_progress
                cur_e = self.obs_noise_e_base
            else:
                cur_t = self.obs_noise_t_target
                e_progress = (progress - split) / max(1e-8, 1.0 - split)
                cur_e = self.obs_noise_e_base + (
                    self.obs_noise_e_target - self.obs_noise_e_base
                ) * e_progress
        else:
            cur_e = self.obs_noise_e_base + (
                self.obs_noise_e_target - self.obs_noise_e_base
            ) * progress
            cur_t = self.obs_noise_t_base + (
                self.obs_noise_t_target - self.obs_noise_t_base
            ) * progress
        self.env.random_obs_noise_e_scale = cur_e
        self.env.random_obs_noise_t_scale = cur_t
        self.direct_info["obs_noise_e_scale_cur"] = cur_e
        self.direct_info["obs_noise_t_scale_cur"] = cur_t
        self.direct_info["obs_noise_curriculum_progress"] = progress
        self.direct_info["obs_noise_curriculum_mode"] = (
            2.0
            if self.obs_noise_curriculum_mode == "staged_hold_te"
            else (1.0 if self.obs_noise_curriculum_mode == "staged_te" else 0.0)
        )

    def _get_infer_consistency_model(self):
        if self.consistency_infer_use_ema and self.consistency_ema_model is not None:
            return self.consistency_ema_model
        return self.consistency_model

    @torch.no_grad()
    def _sample_latent_with_model(self, model, proprio_hist):
        batch_size = proprio_hist.shape[0]
        if self.stochastic_infer:
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            latent = torch.zeros(batch_size, self.latent_dim, device=self.device)
        for idx in range(self.consistency_infer_steps):
            t_cur = 1.0 - (float(idx) / float(max(1, self.consistency_infer_steps)))
            t = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.float32)
            latent = model(proprio_hist, latent, t)
        return latent

    def _sample_latent_with_model_grad(self, model, proprio_hist):
        batch_size = proprio_hist.shape[0]
        if self.stochastic_infer:
            latent = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            latent = torch.zeros(batch_size, self.latent_dim, device=self.device)
        for idx in range(self.consistency_infer_steps):
            t_cur = 1.0 - (float(idx) / float(max(1, self.consistency_infer_steps)))
            t = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.float32)
            latent = model(proprio_hist, latent, t)
        return latent

    @torch.no_grad()
    def sample_latent(self, proprio_hist):
        return self._sample_latent_with_model(self._get_infer_consistency_model(), proprio_hist)

    def sample_latent_train(self, proprio_hist):
        if self.train_align_use_ema and self.consistency_ema_model is not None:
            return self._sample_latent_with_model(
                self.consistency_ema_model, proprio_hist
            )
        return self._sample_latent_with_model_grad(self.consistency_model, proprio_hist)

    def set_eval(self):
        super().set_eval()
        self.consistency_model.eval()
        if self.consistency_ema_model is not None:
            self.consistency_ema_model.eval()

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
                        f"steps={c} mode=consistency "
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
            if (
                self.student_max_agent_steps > 0
                and self.agent_steps >= self.student_max_agent_steps
            ):
                self.save(os.path.join(self.nn_dir, "model_last"))
                tprint(
                    "ConsistencyLatent reached "
                    f"student_max_agent_steps={self.student_max_agent_steps}"
                )
                break

            self._apply_obs_noise_curriculum()
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
            current_lr = self._update_consistency_lr()

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
            batch_size = target_latent.shape[0]
            x1 = torch.randn_like(target_latent)
            t_hi = torch.rand((batch_size,), device=self.device)
            dt = 1.0 / float(self.consistency_num_scales)
            t_lo = torch.clamp(t_hi - dt, min=0.0)

            x_hi = (1.0 - t_hi.unsqueeze(-1)) * target_latent + t_hi.unsqueeze(-1) * x1
            x_lo = (1.0 - t_lo.unsqueeze(-1)) * target_latent + t_lo.unsqueeze(-1) * x1

            pred_hi = self.consistency_model(input_dict["proprio_hist"], x_hi, t_hi)
            with torch.no_grad():
                if self.consistency_ema_model is not None:
                    pred_lo_target = self.consistency_ema_model(
                        input_dict["proprio_hist"], x_lo, t_lo
                    )
                else:
                    pred_lo_target = self.consistency_model(
                        input_dict["proprio_hist"], x_lo, t_lo
                    )
            consistency_loss = ((pred_hi - pred_lo_target.detach()) ** 2).mean()

            t_zero = torch.zeros((batch_size,), device=self.device, dtype=torch.float32)
            pred_zero = self.consistency_model(
                input_dict["proprio_hist"], target_latent, t_zero
            )
            boundary_loss = ((pred_zero - target_latent) ** 2).mean()

            pred_latent = pred_hi
            if self.train_align_infer:
                pred_latent = self.sample_latent_train(input_dict["proprio_hist"])
            student_obs_input = torch.cat([input_dict["obs"], pred_latent], dim=-1)
            student_x = self.model.actor_mlp(student_obs_input)
            student_mu = self.model.mu(student_x)
            student_mu_clamped = torch.clamp(student_mu, -1, 1)
            teacher_mu_clamped = torch.clamp(teacher_mu, -1, 1)
            bc_loss = torch.sum(
                self.recon_criterion(student_mu_clamped, teacher_mu_clamped),
                dim=-1,
            ).mean()
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
                self.consistency_loss_coef * consistency_loss
                + self.consistency_boundary_coef * boundary_loss
                + self.bc_loss_coef * bc_loss
                + self.base_action_anchor_coef * base_action_anchor_loss
                + self.action_l2_coef * action_l2_loss
            )
            self.optim.zero_grad()
            loss.backward()
            grad_norm_sq = None
            for p in self.consistency_model.parameters():
                if p.grad is None:
                    continue
                cur = p.grad.detach().pow(2).sum()
                grad_norm_sq = cur if grad_norm_sq is None else grad_norm_sq + cur
            consistency_grad_norm = (
                float(torch.sqrt(grad_norm_sq).detach().cpu())
                if grad_norm_sq is not None
                else 0.0
            )
            self.optim.step()
            self._update_consistency_ema()

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

            self.direct_info["consistency_loss"] = float(consistency_loss.detach().cpu())
            self.direct_info["consistency_boundary_loss"] = float(boundary_loss.detach().cpu())
            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["base_action_anchor_loss"] = float(
                base_action_anchor_loss.detach().cpu()
            )
            self.direct_info["action_l2_loss"] = float(action_l2_loss.detach().cpu())
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self.direct_info["consistency_grad_norm"] = consistency_grad_norm
            self.direct_info["consistency_lr"] = current_lr
            self.direct_info["train_align_infer"] = 1.0 if self.train_align_infer else 0.0
            self.direct_info["train_align_use_ema"] = 1.0 if self.train_align_use_ema else 0.0
            self.direct_info["train_align_latent_requires_grad"] = (
                1.0 if pred_latent.requires_grad else 0.0
            )
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
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M "
                f"({int(self.agent_steps)}) | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )
            if (
                self.student_progress_log_interval_agent_steps > 0
                and self.agent_steps >= self._next_progress_log_agent_steps
            ):
                tprint(
                    "ConsistencyProgress "
                    f"agent_steps={int(self.agent_steps)} "
                    f"max_agent_steps={self.student_max_agent_steps} "
                    f"best_reward={self.best_rewards:.2f} "
                    f"consistency_lr={current_lr:.8f}"
                )
                while self._next_progress_log_agent_steps <= self.agent_steps:
                    self._next_progress_log_agent_steps += (
                        self.student_progress_log_interval_agent_steps
                    )

        final_train_reward = self.mean_eps_reward.get_mean()
        final_eval_metrics = self._run_final_eval_select(
            "FINAL/student",
            train_reward=final_train_reward,
            eval_best_stem="model_best_eval",
            alias_stems=("model_best", "model_best_deploy"),
        )
        if final_eval_metrics is not None:
            obs_dict = final_eval_metrics["final_obs_dict"]
        self._publish_eval_select_deploy_best(
            "model_best_eval", "model_best_deploy"
        )

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
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
        if "consistency_model" in checkpoint:
            self.consistency_model.load_state_dict(checkpoint["consistency_model"])
            cprint("Loaded consistency_model for train resume", "green")
            if self.consistency_ema_model is not None and "consistency_ema_model" not in checkpoint:
                self.consistency_ema_model.load_state_dict(checkpoint["consistency_model"])
                cprint("Initialized consistency EMA from consistency_model", "yellow")
        if "consistency_ema_model" in checkpoint and self.consistency_ema_model is not None:
            self.consistency_ema_model.load_state_dict(checkpoint["consistency_ema_model"])
            cprint("Loaded consistency_ema_model for train resume", "green")
        if "consistency_optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["consistency_optim"])
            cprint("Loaded consistency optimizer state for train resume", "green")
        if "agent_steps" in checkpoint:
            self.agent_steps = int(checkpoint["agent_steps"])
            cprint(f"Loaded agent_steps={self.agent_steps} for train resume", "green")
        else:
            cprint(
                "Checkpoint missing agent_steps during train restore; resume will start from current counter.",
                "yellow",
            )

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "consistency_model" in checkpoint:
            self.consistency_model.load_state_dict(checkpoint["consistency_model"])
            if self.consistency_ema_model is not None and "consistency_ema_model" not in checkpoint:
                self.consistency_ema_model.load_state_dict(checkpoint["consistency_model"])
        if "consistency_ema_model" in checkpoint and self.consistency_ema_model is not None:
            self.consistency_ema_model.load_state_dict(checkpoint["consistency_ema_model"])
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
            "consistency_model": self.consistency_model.state_dict(),
            "consistency_optim": self.optim.state_dict(),
            "agent_steps": int(self.agent_steps),
        }
        if self.consistency_ema_model is not None:
            weights["consistency_ema_model"] = self.consistency_ema_model.state_dict()
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights["priv_mean_std"] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f"{name}.ckpt")
