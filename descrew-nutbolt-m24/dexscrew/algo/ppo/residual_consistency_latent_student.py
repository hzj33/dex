# --------------------------------------------------------
# Residual consistency latent student distillation
# --------------------------------------------------------

import os
import time

import torch
from tensorboardX import SummaryWriter
from termcolor import cprint

from dexscrew.algo.ppo.consistency_latent_student import ConsistencyLatentStudent
from dexscrew.utils.misc import tprint


class ResidualConsistencyLatentStudent(ConsistencyLatentStudent):
    """Consistency student that predicts a residual over a trained PAdapt latent.

    The PPO/teacher checkpoint alone is not a valid base for this class because
    it does not contain trained adapt_tconv weights. Training should start from a
    PAdapt-style student bundle such as sim2real/codrive/model_best_codrive.ckpt.
    """

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)

        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_residual_consistency_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_residual_consistency_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        self.residual_target_scale = float(
            self.ppo_config.get("consistency_residual_target_scale", 1.0)
        )
        if self.residual_target_scale <= 0.0:
            raise ValueError("consistency_residual_target_scale must be > 0.0")
        self.residual_gate = float(self.ppo_config.get("consistency_residual_gate", 0.25))
        self.residual_gate = min(max(self.residual_gate, 0.0), 1.0)
        self.residual_recon_coef = float(
            self.ppo_config.get("consistency_residual_recon_coef", 0.25)
        )
        self.residual_action_delta_coef = float(
            self.ppo_config.get("consistency_residual_action_delta_coef", 0.0)
        )
        self.require_base_adapt_tconv = bool(
            self.ppo_config.get("consistency_residual_require_base_adapt_tconv", True)
        )
        self.require_head_on_test = bool(
            self.ppo_config.get("consistency_residual_require_head_on_test", True)
        )
        self.student_max_agent_steps = int(
            self.ppo_config.get(
                "student_max_agent_steps",
                self.ppo_config.get("max_agent_steps", 0),
            )
        )

        self._freeze_backbone()
        tprint(
            "ResidualConsistencyLatent config: "
            f"target_scale={self.residual_target_scale}, "
            f"gate={self.residual_gate}, "
            f"residual_recon_coef={self.residual_recon_coef}, "
            f"action_delta_coef={self.residual_action_delta_coef}, "
            f"max_agent_steps={self.student_max_agent_steps}"
        )

    def _freeze_backbone(self):
        for p in self.model.parameters():
            p.requires_grad = False

    def _checkpoint_has_adapt_tconv(self, checkpoint):
        model_state = checkpoint.get("model", {})
        if not hasattr(model_state, "keys"):
            return False
        return any("adapt_tconv" in str(key) for key in model_state.keys())

    def _checkpoint_has_residual_head(self, checkpoint):
        return "consistency_model" in checkpoint

    def _validate_base_checkpoint(self, checkpoint, fn):
        if not self.require_base_adapt_tconv:
            return
        if self._checkpoint_has_adapt_tconv(checkpoint):
            return
        raise RuntimeError(
            "ResidualConsistencyLatentStudent requires a checkpoint with trained "
            f"adapt_tconv weights. Got {fn}. Use a PAdapt/residual student bundle "
            "such as sim2real/codrive/model_best_codrive.ckpt, not the PPO-only "
            "teacher checkpoint."
        )

    @torch.no_grad()
    def _get_base_latent(self, proprio_hist):
        return torch.tanh(self.model.adapt_tconv(proprio_hist))

    def _compose_latent(self, base_latent, residual):
        delta = self.residual_gate * residual / self.residual_target_scale
        return base_latent + delta

    @torch.no_grad()
    def _sample_residual_with_model(self, model, proprio_hist):
        batch_size = proprio_hist.shape[0]
        if self.stochastic_infer:
            residual = torch.randn(batch_size, self.latent_dim, device=self.device)
        else:
            residual = torch.zeros(batch_size, self.latent_dim, device=self.device)
        for idx in range(self.consistency_infer_steps):
            t_cur = 1.0 - (float(idx) / float(max(1, self.consistency_infer_steps)))
            t = torch.full((batch_size,), t_cur, device=self.device, dtype=torch.float32)
            residual = model(proprio_hist, residual, t)
        return residual

    @torch.no_grad()
    def sample_residual(self, proprio_hist):
        return self._sample_residual_with_model(
            self._get_infer_consistency_model(), proprio_hist
        )

    @torch.no_grad()
    def sample_latent(self, proprio_hist):
        base_latent = self._get_base_latent(proprio_hist)
        residual = self.sample_residual(proprio_hist)
        return self._compose_latent(base_latent, residual)

    def sample_latent_train(self, proprio_hist):
        infer_model = self.consistency_model
        if self.train_align_infer and self.consistency_ema_model is not None:
            infer_model = self.consistency_ema_model
        residual = self._sample_residual_with_model(infer_model, proprio_hist)
        base_latent = self._get_base_latent(proprio_hist)
        return self._compose_latent(base_latent, residual)

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        eval_reward_sum = 0.0
        eval_done_sum = 0.0
        eval_latent_mse_sum = 0.0
        eval_latent_l1_sum = 0.0
        eval_residual_mse_sum = 0.0
        eval_action_mse_sum = 0.0
        eval_action_mse_to_base_sum = 0.0
        eval_delta_ratio_sum = 0.0
        eval_saturation_sum = 0.0
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
                pred_residual = self.sample_residual(proprio_hist)
                latent = self._compose_latent(base_latent, pred_residual)

                student_obs_input = torch.cat([obs, latent], dim=-1)
                student_x = self.model.actor_mlp(student_obs_input)
                mu = torch.clamp(self.model.mu(student_x), -1.0, 1.0)

                teacher_obs_input = torch.cat([obs, e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)

                base_obs_input = torch.cat([obs, base_latent], dim=-1)
                base_x = self.model.actor_mlp(base_obs_input)
                base_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

                residual_target = (
                    (e_gt.detach() - base_latent) * self.residual_target_scale
                )
                latent_mse = ((latent - e_gt.detach()) ** 2).mean()
                latent_l1 = (latent - e_gt.detach()).abs().mean()
                residual_mse = ((pred_residual - residual_target) ** 2).mean()
                action_mse_to_teacher = ((mu - teacher_mu) ** 2).mean()
                action_mse_to_base = ((mu - base_mu) ** 2).mean()
                delta_norm = (latent - base_latent).norm(dim=-1).mean()
                base_norm = base_latent.norm(dim=-1).mean().clamp_min(1e-6)
                delta_ratio = delta_norm / base_norm
                saturation_ratio = (mu.abs() > 0.98).float().mean()

                obs_dict, r, done, _ = self.env.step(mu)
            c += 1
            print(f"Step {c}")
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
                eval_latent_mse_sum += float(latent_mse.detach().cpu())
                eval_latent_l1_sum += float(latent_l1.detach().cpu())
                eval_residual_mse_sum += float(residual_mse.detach().cpu())
                eval_action_mse_sum += float(action_mse_to_teacher.detach().cpu())
                eval_action_mse_to_base_sum += float(action_mse_to_base.detach().cpu())
                eval_delta_ratio_sum += float(delta_ratio.detach().cpu())
                eval_saturation_sum += float(saturation_ratio.detach().cpu())
                if c >= self.test_num_steps:
                    avg_reward = eval_reward_sum / float(c)
                    avg_done_rate = eval_done_sum / float(c)
                    print(
                        "EvalSummary "
                        f"steps={c} avg_reward={avg_reward:.6f} "
                        f"avg_done_rate={avg_done_rate:.6f}"
                    )
                    print(
                        "EvalReconSummary "
                        f"steps={c} mode=residual_consistency "
                        f"latent_mse={eval_latent_mse_sum / float(c):.6f} "
                        f"latent_l1={eval_latent_l1_sum / float(c):.6f} "
                        f"residual_mse={eval_residual_mse_sum / float(c):.6f} "
                        f"action_mse_to_teacher={eval_action_mse_sum / float(c):.6f}"
                    )
                    print(
                        "EvalResidualSummary "
                        f"steps={c} gate={self.residual_gate:.6f} "
                        f"delta_to_base_ratio={eval_delta_ratio_sum / float(c):.6f} "
                        f"action_mse_to_base={eval_action_mse_to_base_sum / float(c):.6f} "
                        f"saturation_ratio={eval_saturation_sum / float(c):.6f}"
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
                tprint(
                    "ResidualConsistencyLatent reached "
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

            with torch.no_grad():
                _, _, _, _, e_gt = self.model._actor_critic(input_dict)
                teacher_obs_input = torch.cat([input_dict["obs"], e_gt.detach()], dim=-1)
                teacher_x = self.model.actor_mlp(teacher_obs_input)
                teacher_mu = torch.clamp(self.model.mu(teacher_x), -1.0, 1.0)

                base_latent = self._get_base_latent(input_dict["proprio_hist"])
                base_obs_input = torch.cat([input_dict["obs"], base_latent], dim=-1)
                base_x = self.model.actor_mlp(base_obs_input)
                base_student_mu = torch.clamp(self.model.mu(base_x), -1.0, 1.0)

            target_residual = (
                (e_gt.detach() - base_latent.detach()) * self.residual_target_scale
            )
            batch_size = target_residual.shape[0]
            x1 = torch.randn_like(target_residual)
            t_hi = torch.rand((batch_size,), device=self.device)
            dt = 1.0 / float(self.consistency_num_scales)
            t_lo = torch.clamp(t_hi - dt, min=0.0)

            x_hi = (
                (1.0 - t_hi.unsqueeze(-1)) * target_residual
                + t_hi.unsqueeze(-1) * x1
            )
            x_lo = (
                (1.0 - t_lo.unsqueeze(-1)) * target_residual
                + t_lo.unsqueeze(-1) * x1
            )

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
                input_dict["proprio_hist"], target_residual, t_zero
            )
            boundary_loss = ((pred_zero - target_residual) ** 2).mean()
            residual_recon_loss = ((pred_hi - target_residual) ** 2).mean()

            pred_residual = pred_hi
            if self.train_align_infer:
                pred_latent = self.sample_latent_train(input_dict["proprio_hist"])
            else:
                pred_latent = self._compose_latent(base_latent, pred_residual)
            student_obs_input = torch.cat([input_dict["obs"], pred_latent], dim=-1)
            student_x = self.model.actor_mlp(student_obs_input)
            student_mu = self.model.mu(student_x)
            student_mu_clamped = torch.clamp(student_mu, -1, 1)
            teacher_mu_clamped = torch.clamp(teacher_mu, -1, 1)
            bc_loss = torch.sum(
                self.recon_criterion(student_mu_clamped, teacher_mu_clamped),
                dim=-1,
            ).mean()
            base_action_anchor_loss = torch.sum(
                self.recon_criterion(student_mu_clamped, base_student_mu),
                dim=-1,
            ).mean()
            action_delta_loss = ((student_mu_clamped - base_student_mu) ** 2).mean()
            action_l2_loss = student_mu_clamped.pow(2).mean()

            loss = (
                self.consistency_loss_coef * consistency_loss
                + self.consistency_boundary_coef * boundary_loss
                + self.residual_recon_coef * residual_recon_loss
                + self.bc_loss_coef * bc_loss
                + self.base_action_anchor_coef * base_action_anchor_loss
                + self.residual_action_delta_coef * action_delta_loss
                + self.action_l2_coef * action_l2_loss
            )
            self.optim.zero_grad()
            loss.backward()
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

            with torch.no_grad():
                delta_latent = pred_latent.detach() - base_latent.detach()
                base_norm = base_latent.detach().norm(dim=-1).mean().clamp_min(1e-6)
                delta_norm = delta_latent.norm(dim=-1).mean()
                target_residual_norm = target_residual.norm(dim=-1).mean()
                pred_residual_norm = pred_residual.detach().norm(dim=-1).mean()
                latent_mse = ((pred_latent.detach() - e_gt.detach()) ** 2).mean()
                latent_l1 = (pred_latent.detach() - e_gt.detach()).abs().mean()
                action_mse_to_teacher = (
                    (student_mu_clamped.detach() - teacher_mu_clamped.detach()) ** 2
                ).mean()
                action_mse_to_base = (
                    (student_mu_clamped.detach() - base_student_mu.detach()) ** 2
                ).mean()
                saturation_ratio = (student_mu_clamped.detach().abs() > 0.98).float().mean()

            self.direct_info["consistency_loss"] = float(consistency_loss.detach().cpu())
            self.direct_info["consistency_boundary_loss"] = float(boundary_loss.detach().cpu())
            self.direct_info["residual_recon_loss"] = float(residual_recon_loss.detach().cpu())
            self.direct_info["bc_loss"] = float(bc_loss.detach().cpu())
            self.direct_info["base_action_anchor_loss"] = float(
                base_action_anchor_loss.detach().cpu()
            )
            self.direct_info["action_delta_loss"] = float(action_delta_loss.detach().cpu())
            self.direct_info["action_l2_loss"] = float(action_l2_loss.detach().cpu())
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self.direct_info["residual_gate"] = float(self.residual_gate)
            self.direct_info["base_latent_norm"] = float(base_norm.detach().cpu())
            self.direct_info["delta_latent_norm"] = float(delta_norm.detach().cpu())
            self.direct_info["delta_to_base_ratio"] = float(
                (delta_norm / base_norm).detach().cpu()
            )
            self.direct_info["target_residual_norm"] = float(
                target_residual_norm.detach().cpu()
            )
            self.direct_info["pred_residual_norm"] = float(pred_residual_norm.detach().cpu())
            self.direct_info["latent_mse"] = float(latent_mse.detach().cpu())
            self.direct_info["latent_l1"] = float(latent_l1.detach().cpu())
            self.direct_info["action_mse_to_teacher"] = float(
                action_mse_to_teacher.detach().cpu()
            )
            self.direct_info["action_mse_to_base"] = float(action_mse_to_base.detach().cpu())
            self.direct_info["action_saturation_ratio"] = float(
                saturation_ratio.detach().cpu()
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
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f}"
            )

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._validate_base_checkpoint(checkpoint, fn)
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
        if not self._checkpoint_has_residual_head(checkpoint):
            cprint(
                "No consistency_model in checkpoint; starting residual head from scratch.",
                "yellow",
            )
        else:
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
        self._freeze_backbone()

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device)
        self._validate_base_checkpoint(checkpoint, fn)
        if self.require_head_on_test and not self._checkpoint_has_residual_head(checkpoint):
            raise RuntimeError(
                "ResidualConsistencyLatentStudent test restore requires a trained "
                "consistency_model. Use a residual checkpoint, or set "
                "+train.ppo.consistency_residual_require_head_on_test=False for "
                "base-only diagnostics."
            )
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
        self._freeze_backbone()
