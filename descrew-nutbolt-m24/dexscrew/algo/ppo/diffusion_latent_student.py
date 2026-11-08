# --------------------------------------------------------
# Latent Diffusion student distillation (separate from ProprioAdapt)
# --------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
from termcolor import cprint
from tensorboardX import SummaryWriter

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.utils.misc import tprint


class LatentDiffusionHead(nn.Module):
    def __init__(self, proprio_hist_dim, proprio_dim, latent_dim, num_steps, hidden_dim=256, t_dim=64):
        super().__init__()
        self.hist_encoder = nn.Sequential(
            nn.Linear(proprio_hist_dim * proprio_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.t_embed = nn.Embedding(num_steps, t_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + t_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, proprio_hist, x_t, t):
        hist_feat = self.hist_encoder(proprio_hist.reshape(proprio_hist.shape[0], -1))
        t_feat = self.t_embed(t)
        denoise_in = torch.cat([hist_feat, x_t, t_feat], dim=-1)
        return self.denoiser(denoise_in)


class DiffusionLatentStudent(ProprioAdapt):
    """Diffusion student that replaces deterministic latent prediction with latent diffusion."""

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)

        # Keep diffusion artifacts isolated from ProprioAdapt artifacts.
        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_diffusion_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_diffusion_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # Freeze teacher/student backbone; train diffusion head only.
        for p in self.model.parameters():
            p.requires_grad = False
        self.diffusion_student_trainable_param_patterns = (
            self._resolve_diffusion_trainable_param_patterns()
        )
        extra_trainable_params = []
        if self.diffusion_student_trainable_param_patterns:
            for name, p in self.model.named_parameters():
                if self._is_diffusion_trainable_param(name):
                    p.requires_grad = True
                    extra_trainable_params.append(p)

        self.latent_dim = self.model.adapt_tconv.low_dim_proj.out_features
        self.diffusion_steps = int(self.ppo_config.get("diffusion_steps", 10))
        requested_diffusion_steps_infer = int(
            self.ppo_config.get("diffusion_steps_infer", self.diffusion_steps)
        )
        # Keep inference schedule valid under current training schedule.
        self.diffusion_steps_infer = min(requested_diffusion_steps_infer, self.diffusion_steps)
        if self.diffusion_steps_infer != requested_diffusion_steps_infer:
            tprint(
                "DiffusionLatent note: "
                f"diffusion_steps_infer={requested_diffusion_steps_infer} "
                f"clamped to {self.diffusion_steps_infer} "
                f"(diffusion_steps={self.diffusion_steps})."
            )
        self.stochastic_infer = bool(self.ppo_config.get("diffusion_stochastic_infer", False))
        # Optional residual mode: diffuse only the correction around the frozen adapt_tconv latent.
        self.residual_base = bool(self.ppo_config.get("diffusion_residual_base", False))
        self.residual_target_scale = float(
            self.ppo_config.get("diffusion_residual_target_scale", 1.0)
        )
        self.residual_target_scale = max(self.residual_target_scale, 1e-6)
        # Optional obs-noise curriculum: linearly ramp env observation noise during training.
        self.obs_noise_curriculum = bool(
            self.ppo_config.get("diffusion_obs_noise_curriculum", False)
        )
        self.obs_noise_curriculum_start = int(
            self.ppo_config.get("diffusion_obs_noise_curriculum_start", 0)
        )
        self.obs_noise_curriculum_steps = max(
            1, int(self.ppo_config.get("diffusion_obs_noise_curriculum_steps", 1))
        )
        self.obs_noise_curriculum_mode = str(
            self.ppo_config.get("diffusion_obs_noise_curriculum_mode", "linear")
        )
        self.obs_noise_curriculum_t_phase_ratio = float(
            self.ppo_config.get("diffusion_obs_noise_curriculum_t_phase_ratio", 0.6)
        )
        self.obs_noise_curriculum_t_phase_ratio = min(
            max(self.obs_noise_curriculum_t_phase_ratio, 0.0), 1.0
        )
        self.obs_noise_curriculum_hold_ratio = float(
            self.ppo_config.get("diffusion_obs_noise_curriculum_hold_ratio", 0.2)
        )
        self.obs_noise_curriculum_hold_ratio = min(
            max(self.obs_noise_curriculum_hold_ratio, 0.0), 1.0
        )
        self.obs_noise_e_base = float(getattr(self.env, "random_obs_noise_e_scale", 0.0))
        self.obs_noise_t_base = float(getattr(self.env, "random_obs_noise_t_scale", 0.0))
        self.obs_noise_e_target = float(
            self.ppo_config.get("diffusion_obs_noise_e_target", self.obs_noise_e_base)
        )
        self.obs_noise_t_target = float(
            self.ppo_config.get("diffusion_obs_noise_t_target", self.obs_noise_t_base)
        )

        beta_start = float(self.ppo_config.get("diffusion_beta_start", 1e-4))
        beta_end = float(self.ppo_config.get("diffusion_beta_end", 2e-2))
        self.betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.diffusion_model = LatentDiffusionHead(
            proprio_hist_dim=self.proprio_hist_dim,
            proprio_dim=self.proprio_dim,
            latent_dim=self.latent_dim,
            num_steps=self.diffusion_steps,
            hidden_dim=int(self.ppo_config.get("diffusion_hidden_dim", 256)),
            t_dim=int(self.ppo_config.get("diffusion_t_dim", 64)),
        ).to(self.device)

        self.diffusion_loss_coef = float(self.ppo_config.get("diffusion_loss_coef", 1.0))
        self.bc_loss_coef = float(self.ppo_config.get("bc_loss_coef", 1.0))
        self.latent_recon_coef = float(
            self.ppo_config.get("diffusion_latent_recon_coef", 0.0)
        )
        self.latent_recon_coef_start = float(
            self.ppo_config.get(
                "diffusion_latent_recon_coef_start", self.latent_recon_coef
            )
        )
        self.latent_recon_coef_end = float(
            self.ppo_config.get(
                "diffusion_latent_recon_coef_end", self.latent_recon_coef
            )
        )
        self.latent_recon_coef_schedule_steps = max(
            1,
            int(self.ppo_config.get("diffusion_latent_recon_coef_schedule_steps", 1)),
        )
        self.base_action_anchor_coef = float(
            self.ppo_config.get("diffusion_base_action_anchor_coef", 0.0)
        )
        self.action_l2_coef = float(
            self.ppo_config.get("diffusion_action_l2_coef", 0.0)
        )
        self.teacher_delta_tail_coef = float(
            self.ppo_config.get("diffusion_teacher_delta_tail_coef", 0.0)
        )
        self.teacher_delta_tail_threshold = float(
            self.ppo_config.get("diffusion_teacher_delta_tail_threshold", 0.25)
        )
        self.teacher_delta_tail_selective = bool(
            self.ppo_config.get("diffusion_teacher_delta_tail_selective", False)
        )
        self.teacher_delta_tail_mid_only = bool(
            self.ppo_config.get("diffusion_teacher_delta_tail_mid_only", False)
        )
        self.teacher_delta_tail_progress_start = float(
            self.ppo_config.get("diffusion_teacher_delta_tail_progress_start", 0.25)
        )
        self.teacher_delta_tail_progress_end = float(
            self.ppo_config.get("diffusion_teacher_delta_tail_progress_end", 0.75)
        )
        self.optim = torch.optim.Adam(
            list(self.diffusion_model.parameters()) + extra_trainable_params,
            lr=float(self.ppo_config.get("diffusion_lr", 3e-4)),
        )
        self.diffusion_extra_trainable_param_count = int(
            sum(p.numel() for p in extra_trainable_params)
        )
        if self.diffusion_student_trainable_param_patterns:
            tprint(
                "DiffusionLatent extra trainable patterns: "
                f"{list(self.diffusion_student_trainable_param_patterns)} | "
                f"extra params: {self.diffusion_extra_trainable_param_count}"
            )
        else:
            tprint("DiffusionLatent extra trainable patterns: [] | extra params: 0")
        # Eval-only switches for Plan v2 M2 evidence hardening.
        self.eval_decode_only = bool(
            self.ppo_config.get("diffusion_eval_decode_only", False)
        )
        self.eval_report_recon = bool(
            self.ppo_config.get("diffusion_eval_report_recon", True)
        )

    def _resolve_diffusion_trainable_param_patterns(self):
        patterns = self.ppo_config.get("diffusion_student_trainable_param_patterns", [])
        if patterns is None:
            return tuple()
        if isinstance(patterns, (str, bytes)):
            patterns = [patterns]
        elif not isinstance(patterns, (list, tuple)):
            try:
                patterns = list(patterns)
            except TypeError as exc:
                raise TypeError(
                    "train.ppo.diffusion_student_trainable_param_patterns must be list/tuple/str, "
                    f"got {type(patterns)}"
                ) from exc
        cleaned = []
        for p in patterns:
            token = str(p).strip()
            if token:
                cleaned.append(token)
        return tuple(cleaned)

    def _is_diffusion_trainable_param(self, param_name):
        return any(
            pattern in param_name for pattern in self.diffusion_student_trainable_param_patterns
        )

    def _gather_vec(self, vec, t):
        return vec[t].unsqueeze(-1)

    def _q_sample(self, x0, t, noise):
        s1 = self._gather_vec(self.sqrt_alpha_bars, t)
        s2 = self._gather_vec(self.sqrt_one_minus_alpha_bars, t)
        return s1 * x0 + s2 * noise

    def _predict_x0(self, x_t, t, eps_pred):
        s1 = self._gather_vec(self.sqrt_alpha_bars, t)
        s2 = self._gather_vec(self.sqrt_one_minus_alpha_bars, t)
        return (x_t - s2 * eps_pred) / (s1 + 1e-8)

    @torch.no_grad()
    def _get_base_latent(self, proprio_hist):
        base_latent = self.model.adapt_tconv(proprio_hist)
        return torch.tanh(base_latent)

    @torch.no_grad()
    def sample_latent(self, proprio_hist):
        batch_size = proprio_hist.shape[0]
        base_latent = self._get_base_latent(proprio_hist) if self.residual_base else None
        # Keep deterministic eval truly deterministic across seeds.
        if self.stochastic_infer:
            x = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            x = torch.zeros(batch_size, self.latent_dim, device=self.device)
        for t_idx in reversed(range(self.diffusion_steps_infer)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            eps_pred = self.diffusion_model(proprio_hist, x, t)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bars[t_idx]
            beta_t = self.betas[t_idx]

            mean = (1.0 / torch.sqrt(alpha_t)) * (
                x - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t + 1e-8)) * eps_pred
            )
            if t_idx > 0 and self.stochastic_infer:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean
        if base_latent is not None:
            x = x / self.residual_target_scale
            x = x + base_latent
        return x

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

    def set_eval(self):
        super().set_eval()
        self.diffusion_model.eval()

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
        if self.eval_decode_only:
            latent = self._get_base_latent(proprio_hist)
        else:
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
        eval_residual_abs_sum = 0.0
        eval_residual_l2_sum = 0.0
        eval_residual_ratio_sum = 0.0
        eval_action_corr_abs_sum = 0.0
        eval_action_corr_l2_sum = 0.0
        eval_base_action_mse_sum = 0.0
        eval_pred_residual_abs_sum = 0.0
        eval_pred_to_target_ratio_sum = 0.0
        eval_mode = "decode_only" if self.eval_decode_only else "diffusion"
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
                base_latent = self._get_base_latent(proprio_hist)
                if self.eval_decode_only:
                    latent = base_latent
                else:
                    latent = self.sample_latent(proprio_hist)

                student_obs_input = torch.cat([obs, latent], dim=-1)
                student_x = self.model.actor_mlp(student_obs_input)
                mu = self.model.mu(student_x)
                mu = torch.clamp(mu, -1.0, 1.0)

                teacher_obs_input = torch.cat([obs, e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)
                base_obs_input = torch.cat([obs, base_latent], dim=-1)
                base_x = self.model.actor_mlp(base_obs_input)
                base_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

                latent_mse = ((latent - e_gt.detach()) ** 2).mean()
                latent_l1 = (latent - e_gt.detach()).abs().mean()
                action_mse_to_teacher = ((mu - teacher_mu) ** 2).mean()
                residual_latent = e_gt.detach() - base_latent
                residual_abs_mean = residual_latent.abs().mean()
                residual_l2_mean = torch.sqrt(
                    (residual_latent.pow(2)).sum(dim=-1) + 1e-8
                ).mean()
                residual_ratio = residual_abs_mean / (e_gt.detach().abs().mean() + 1e-8)
                pred_residual_latent = latent - base_latent
                pred_residual_abs_mean = pred_residual_latent.abs().mean()
                pred_to_target_ratio = pred_residual_abs_mean / (residual_abs_mean + 1e-8)
                action_correction = mu - base_mu
                action_corr_abs_mean = action_correction.abs().mean()
                action_corr_l2_mean = torch.sqrt(
                    (action_correction.pow(2)).sum(dim=-1) + 1e-8
                ).mean()
                base_action_mse_to_teacher = ((base_mu - teacher_mu) ** 2).mean()

                obs_dict, r, done, _ = self.env.step(mu)
            c += 1
            print(f"Step {c}")
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
                if self.eval_report_recon:
                    eval_latent_mse_sum += float(latent_mse.detach().cpu())
                    eval_latent_l1_sum += float(latent_l1.detach().cpu())
                    eval_action_mse_sum += float(action_mse_to_teacher.detach().cpu())
                    if self.residual_base:
                        eval_residual_abs_sum += float(residual_abs_mean.detach().cpu())
                        eval_residual_l2_sum += float(residual_l2_mean.detach().cpu())
                        eval_residual_ratio_sum += float(residual_ratio.detach().cpu())
                        eval_action_corr_abs_sum += float(action_corr_abs_mean.detach().cpu())
                        eval_action_corr_l2_sum += float(action_corr_l2_mean.detach().cpu())
                        eval_base_action_mse_sum += float(
                            base_action_mse_to_teacher.detach().cpu()
                        )
                        eval_pred_residual_abs_sum += float(
                            pred_residual_abs_mean.detach().cpu()
                        )
                        eval_pred_to_target_ratio_sum += float(
                            pred_to_target_ratio.detach().cpu()
                        )
                if c >= self.test_num_steps:
                    avg_reward = eval_reward_sum / float(c)
                    avg_done_rate = eval_done_sum / float(c)
                    print(
                        "EvalSummary "
                        f"steps={c} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                    )
                    if self.eval_report_recon:
                        avg_latent_mse = eval_latent_mse_sum / float(c)
                        avg_latent_l1 = eval_latent_l1_sum / float(c)
                        avg_action_mse = eval_action_mse_sum / float(c)
                        print(
                            "EvalReconSummary "
                            f"steps={c} mode={eval_mode} "
                            f"latent_mse={avg_latent_mse:.6f} "
                            f"latent_l1={avg_latent_l1:.6f} "
                            f"action_mse_to_teacher={avg_action_mse:.6f}"
                        )
                        if self.residual_base:
                            avg_residual_abs = eval_residual_abs_sum / float(c)
                            avg_residual_l2 = eval_residual_l2_sum / float(c)
                            avg_residual_ratio = eval_residual_ratio_sum / float(c)
                            avg_action_corr_abs = eval_action_corr_abs_sum / float(c)
                            avg_action_corr_l2 = eval_action_corr_l2_sum / float(c)
                            avg_base_action_mse = eval_base_action_mse_sum / float(c)
                            avg_pred_residual_abs = eval_pred_residual_abs_sum / float(c)
                            avg_pred_to_target_ratio = (
                                eval_pred_to_target_ratio_sum / float(c)
                            )
                            print(
                                "EvalResidualSummary "
                                f"steps={c} mode={eval_mode} "
                                f"residual_abs_mean={avg_residual_abs:.6f} "
                                f"residual_l2_mean={avg_residual_l2:.6f} "
                                f"residual_to_target_ratio={avg_residual_ratio:.6f} "
                                f"pred_residual_abs_mean={avg_pred_residual_abs:.6f} "
                                f"pred_to_target_ratio={avg_pred_to_target_ratio:.6f} "
                                f"action_correction_abs_mean={avg_action_corr_abs:.6f} "
                                f"action_correction_l2_mean={avg_action_corr_l2:.6f} "
                                f"base_action_mse_to_teacher={avg_base_action_mse:.6f}"
                            )
                    break

    def collect_rollout(self, num_steps=256, save_path=None, save_point_cloud=True):
        """Collect rollout payload using the same diffusion action path as test()."""
        self.set_eval()
        obs_dict = self.env.reset()
        eval_mode = "decode_only" if self.eval_decode_only else "diffusion"

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
                base_latent = self._get_base_latent(proprio_hist)
                if self.eval_decode_only:
                    latent = base_latent
                else:
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

                # Keep info-based scalar stream aligned with PPO collect_rollout payload format.
                for key, value in info.items():
                    scalar_value = None
                    if torch.is_tensor(value):
                        scalar_value = float(value.detach().float().mean().cpu())
                    elif isinstance(value, (float, int, bool)):
                        scalar_value = float(value)
                    if scalar_value is None:
                        continue
                    extra_scalar_buf.setdefault(str(key), []).append(scalar_value)

                # Diffusion-specific rollout diagnostics for failure-mode comparison.
                extra_scalar_buf.setdefault("diag/latent_mse", []).append(
                    float(latent_mse.detach().cpu())
                )
                extra_scalar_buf.setdefault("diag/latent_l1", []).append(
                    float(latent_l1.detach().cpu())
                )
                extra_scalar_buf.setdefault("diag/action_mse_to_teacher", []).append(
                    float(action_mse_to_teacher.detach().cpu())
                )

                obs_dict = next_obs_dict

        payload = {
            "meta": {
                "num_steps": int(num_steps),
                "num_envs": int(self.env.num_envs),
                "normalize_input": bool(getattr(self, "normalize_input", True)),
                "normalize_priv": bool(getattr(self, "normalize_priv", True)),
                "normalize_point_cloud": bool(getattr(self, "normalize_point_cloud", True)),
                "save_point_cloud": bool(save_point_cloud),
                "eval_mode": eval_mode,
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
            save_dir = os.path.join(self.output_dir, "teacher_rollouts")
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

            with torch.no_grad():
                _, _, _, _, e_gt = self.model._actor_critic(input_dict)
                teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)
                residual_base_latent = (
                    self._get_base_latent(input_dict["proprio_hist"])
                    if self.residual_base
                    else None
                )
                base_student_mu = None
                if self.base_action_anchor_coef > 0.0:
                    base_latent_anchor = self._get_base_latent(input_dict["proprio_hist"])
                    base_obs_input = torch.cat([input_dict["obs"], base_latent_anchor], dim=-1)
                    base_x = self.model.actor_mlp(base_obs_input)
                    base_student_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

            batch_size = e_gt.shape[0]
            t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
            target_latent = e_gt.detach()
            target_x0 = (
                target_latent - residual_base_latent
                if residual_base_latent is not None
                else target_latent
            )
            if residual_base_latent is not None:
                target_x0 = target_x0 * self.residual_target_scale
            noise = torch.randn_like(target_x0)
            x_t = self._q_sample(target_x0, t, noise)
            eps_pred = self.diffusion_model(input_dict["proprio_hist"], x_t, t)
            diffusion_loss = ((eps_pred - noise) ** 2).mean()

            x0_pred = self._predict_x0(x_t, t, eps_pred)
            if residual_base_latent is not None:
                x0_pred = x0_pred / self.residual_target_scale
                x0_pred = x0_pred + residual_base_latent
            pred_latent = x0_pred
            latent_recon_loss = ((pred_latent - target_latent) ** 2).mean()
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
            teacher_delta_tail = torch.relu(
                (student_mu_clamped - teacher_mu_clamped).abs()
                - self.teacher_delta_tail_threshold
            )
            teacher_delta_tail_per_sample = teacher_delta_tail.pow(2).mean(dim=-1)
            teacher_delta_tail_active = (teacher_delta_tail > 0).any(dim=-1)
            teacher_delta_tail_sample_mask = teacher_delta_tail_active
            if self.teacher_delta_tail_mid_only:
                episode_progress = torch.clamp(
                    self.step_length / float(max(1, self.env.max_episode_length)),
                    0.0,
                    1.0,
                )
                teacher_delta_tail_mid_mask = (
                    (episode_progress >= self.teacher_delta_tail_progress_start)
                    & (episode_progress <= self.teacher_delta_tail_progress_end)
                )
                teacher_delta_tail_sample_mask = (
                    teacher_delta_tail_sample_mask & teacher_delta_tail_mid_mask
                )
            teacher_delta_tail_active_ratio = teacher_delta_tail_sample_mask.float().mean()
            if self.teacher_delta_tail_selective and teacher_delta_tail_sample_mask.any():
                teacher_delta_tail_loss = teacher_delta_tail_per_sample[
                    teacher_delta_tail_sample_mask
                ].mean()
            elif self.teacher_delta_tail_mid_only and teacher_delta_tail_sample_mask.any():
                teacher_delta_tail_loss = teacher_delta_tail_per_sample[
                    teacher_delta_tail_sample_mask
                ].mean()
            elif self.teacher_delta_tail_mid_only:
                teacher_delta_tail_loss = torch.zeros(
                    (), device=self.device, dtype=teacher_delta_tail_per_sample.dtype
                )
            else:
                teacher_delta_tail_loss = teacher_delta_tail_per_sample.mean()

            recon_progress = min(
                max(self.agent_steps / float(self.latent_recon_coef_schedule_steps), 0.0),
                1.0,
            )
            latent_recon_coef_cur = (
                self.latent_recon_coef_start
                + (self.latent_recon_coef_end - self.latent_recon_coef_start)
                * recon_progress
            )

            loss = (
                self.diffusion_loss_coef * diffusion_loss
                + self.bc_loss_coef * bc_loss
                + latent_recon_coef_cur * latent_recon_loss
                + self.base_action_anchor_coef * base_action_anchor_loss
                + self.action_l2_coef * action_l2_loss
                + self.teacher_delta_tail_coef * teacher_delta_tail_loss
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

            self.direct_info["diffusion_loss"] = float(diffusion_loss.detach().cpu())
            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["latent_recon_loss"] = float(latent_recon_loss.detach().cpu())
            self.direct_info["base_action_anchor_loss"] = float(
                base_action_anchor_loss.detach().cpu()
            )
            self.direct_info["action_l2_loss"] = float(action_l2_loss.detach().cpu())
            self.direct_info["teacher_delta_tail_loss"] = float(
                teacher_delta_tail_loss.detach().cpu()
            )
            self.direct_info["teacher_delta_tail_active_ratio"] = float(
                teacher_delta_tail_active_ratio.detach().cpu()
            )
            self.direct_info["latent_recon_coef_cur"] = float(latent_recon_coef_cur)
            self.direct_info["latent_recon_coef_progress"] = float(recon_progress)
            self.direct_info["residual_target_scale"] = float(self.residual_target_scale)
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
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )
            tprint(info_string)

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
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

        # Resume diffusion student state when available; fallback to teacher-only init otherwise.
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
            cprint("Loaded diffusion_model for train resume", "green")
        if "diffusion_optim" in checkpoint:
            self.optim.load_state_dict(checkpoint["diffusion_optim"])
            cprint("Loaded diffusion optimizer state for train resume", "green")

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn)
        if "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"], strict=False)
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
        if "running_mean_std" in checkpoint:
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if "point_cloud_mean_std" in checkpoint and self.normalize_point_cloud:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
            "diffusion_model": self.diffusion_model.state_dict(),
        }
        if self.running_mean_std:
            weights["running_mean_std"] = self.running_mean_std.state_dict()
        if self.sa_mean_std:
            weights["sa_mean_std"] = self.sa_mean_std.state_dict()
        if self.priv_mean_std:
            weights["priv_mean_std"] = self.priv_mean_std.state_dict()
        if self.normalize_point_cloud:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        weights["diffusion_optim"] = self.optim.state_dict()
        torch.save(weights, f"{name}.ckpt")
