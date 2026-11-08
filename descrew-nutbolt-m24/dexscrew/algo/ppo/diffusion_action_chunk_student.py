# --------------------------------------------------------
# Action-chunk Diffusion student distillation (separate from ProprioAdapt)
# --------------------------------------------------------

import os
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from termcolor import cprint

from dexscrew.algo.ppo.padapt import ProprioAdapt
from dexscrew.utils.misc import AverageScalarMeter, tprint


def build_action_chunk_rollout_dataset(rollout_payload, chunk_len):
    """Build flattened (condition, target_chunk) tensors from teacher rollout payload."""
    required_keys = ("obs", "proprio_hist", "actions")
    missing = [k for k in required_keys if k not in rollout_payload]
    if missing:
        raise KeyError(f"Rollout payload missing keys: {missing}")

    obs = rollout_payload["obs"]
    proprio_hist = rollout_payload["proprio_hist"]
    actions = rollout_payload["actions"]
    if not (torch.is_tensor(obs) and torch.is_tensor(proprio_hist) and torch.is_tensor(actions)):
        raise TypeError("obs/proprio_hist/actions in rollout payload must all be torch tensors")
    if obs.ndim != 3:
        raise ValueError(f"Expected obs shape [T, N, obs_dim], got {tuple(obs.shape)}")
    if proprio_hist.ndim != 4:
        raise ValueError(
            f"Expected proprio_hist shape [T, N, prop_hist_len, proprio_dim], got {tuple(proprio_hist.shape)}"
        )
    if actions.ndim != 3:
        raise ValueError(f"Expected actions shape [T, N, act_dim], got {tuple(actions.shape)}")

    t_steps, num_envs, obs_dim = obs.shape
    if proprio_hist.shape[0] != t_steps or proprio_hist.shape[1] != num_envs:
        raise ValueError("obs and proprio_hist first two dims [T, N] are inconsistent")
    if actions.shape[0] != t_steps or actions.shape[1] != num_envs:
        raise ValueError("obs and actions first two dims [T, N] are inconsistent")
    if chunk_len > t_steps:
        raise ValueError(f"chunk_len={chunk_len} exceeds rollout steps T={t_steps}")

    window_count = t_steps - chunk_len + 1
    # Condition is taken from the first step of each training window.
    cond_obs = obs[:window_count]  # [W, N, obs_dim]
    cond_prop = proprio_hist[:window_count]  # [W, N, prop_hist_len, proprio_dim]
    # Target is teacher actions over the next chunk_len steps.
    action_chunks = torch.stack(
        [actions[start : start + chunk_len] for start in range(window_count)],
        dim=0,
    ).permute(0, 2, 1, 3)  # [W, N, chunk_len, act_dim]

    cond_obs = cond_obs.reshape(window_count * num_envs, obs_dim).float().contiguous()
    cond_prop = cond_prop.reshape(window_count * num_envs, *proprio_hist.shape[2:]).float().contiguous()
    action_chunks = action_chunks.reshape(window_count * num_envs, chunk_len, actions.shape[-1]).float().contiguous()
    return cond_obs, cond_prop, action_chunks


class ActionChunkDiffusionHead(nn.Module):
    def __init__(
        self,
        obs_dim,
        proprio_hist_dim,
        proprio_dim,
        chunk_dim,
        num_steps,
        hidden_dim=512,
        t_dim=64,
    ):
        super().__init__()
        cond_dim = obs_dim + proprio_hist_dim * proprio_dim
        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
        )
        self.t_embed = nn.Embedding(num_steps, t_dim)
        self.denoiser = nn.Sequential(
            nn.Linear(hidden_dim + chunk_dim + t_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, chunk_dim),
        )

    def forward(self, obs, proprio_hist, x_t, t):
        cond = torch.cat([obs, proprio_hist.reshape(proprio_hist.shape[0], -1)], dim=-1)
        cond_feat = self.cond_encoder(cond)
        t_feat = self.t_embed(t)
        denoise_in = torch.cat([cond_feat, x_t, t_feat], dim=-1)
        return self.denoiser(denoise_in)


class DiffusionActionChunkStudent(ProprioAdapt):
    """Diffusion student that models short-horizon action chunks and executes receding horizon."""

    def __init__(self, env, output_dir, full_config, student_dim=24):
        super().__init__(env, output_dir, full_config, student_dim=student_dim)
        assert self.actions_num == self.env.action_space.shape[0], (
            f"Action dim mismatch: actions_num={self.actions_num}, "
            f"env.action_space={self.env.action_space.shape[0]}"
        )
        assert self.obs_shape[0] == self.env.observation_space.shape[0], (
            f"Obs dim mismatch: obs_shape={self.obs_shape[0]}, "
            f"env.observation_space={self.env.observation_space.shape[0]}"
        )

        # Keep action-chunk diffusion artifacts isolated from other baselines.
        self.writer.close()
        self.nn_dir = os.path.join(self.output_dir, "stage2_diffusion_action_chunk_nn")
        self.tb_dir = os.path.join(self.output_dir, "stage2_diffusion_action_chunk_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # Freeze teacher/student backbone; only train diffusion head.
        for p in self.model.parameters():
            p.requires_grad = False

        self.chunk_len = int(self.ppo_config.get("action_chunk_len", 8))
        self.chunk_len = max(self.chunk_len, 2)
        self.chunk_dim = self.chunk_len * self.actions_num

        self.diffusion_steps = int(self.ppo_config.get("action_chunk_diffusion_steps", 10))
        self.diffusion_steps = max(self.diffusion_steps, 2)
        self.diffusion_steps_infer = int(
            self.ppo_config.get("action_chunk_diffusion_steps_infer", self.diffusion_steps)
        )
        self.diffusion_steps_infer = max(1, min(self.diffusion_steps_infer, self.diffusion_steps))

        beta_start = float(self.ppo_config.get("action_chunk_diffusion_beta_start", 1e-4))
        beta_end = float(self.ppo_config.get("action_chunk_diffusion_beta_end", 2e-2))
        self.betas = torch.linspace(beta_start, beta_end, self.diffusion_steps, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.diffusion_model = ActionChunkDiffusionHead(
            obs_dim=self.obs_shape[0],
            proprio_hist_dim=self.proprio_hist_dim,
            proprio_dim=self.proprio_dim,
            chunk_dim=self.chunk_dim,
            num_steps=self.diffusion_steps,
            hidden_dim=int(self.ppo_config.get("action_chunk_diffusion_hidden_dim", 512)),
            t_dim=int(self.ppo_config.get("action_chunk_diffusion_t_dim", 64)),
        ).to(self.device)

        self.diffusion_loss_coef = float(
            self.ppo_config.get("action_chunk_diffusion_loss_coef", 1.0)
        )
        self.first_action_bc_loss_coef = float(
            self.ppo_config.get("action_chunk_first_action_bc_loss_coef", 1.0)
        )
        self.chunk_bc_loss_coef = float(self.ppo_config.get("action_chunk_bc_loss_coef", 0.1))
        self.stochastic_infer = bool(self.ppo_config.get("action_chunk_stochastic_infer", False))
        self.teacher_mix_steps = int(self.ppo_config.get("action_chunk_teacher_mix_steps", 0))
        # When teacher-mix is used, mixed-policy reward is not aligned with pure-student eval.
        # Track a student-aligned best checkpoint after a warmup window (default: after mix reaches 0).
        self.model_selection_warmup_steps = int(
            self.ppo_config.get("action_chunk_model_selection_warmup_steps", self.teacher_mix_steps)
        )
        self.model_selection_warmup_steps = max(0, self.model_selection_warmup_steps)
        self.best_student_rewards = -10000.0
        self.best_student_action_mse = float("inf")
        self.best_deploy_probe_first_action_mse = float("inf")
        # Track a deployment-aligned student-only episode reward after teacher mix fully exits.
        # We only start counting episodes that begin after the transition to pure-student control.
        self.student_eval_phase_started = False
        self.student_mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.student_mean_eps_length = AverageScalarMeter(window_size=20000)
        self.student_step_reward = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self.student_step_length = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self.student_tracking_active = torch.zeros(
            self.num_actors, dtype=torch.bool, device=self.device
        )
        self.pure_student_window_len = torch.zeros(
            self.num_actors, dtype=torch.long, device=self.device
        )
        # Save periodic latest checkpoints to keep recoverable artifacts under timeout-driven runs.
        self.ckpt_interval_steps = int(
            self.ppo_config.get("action_chunk_ckpt_interval_steps", 50000)
        )
        self.ckpt_interval_steps = max(0, self.ckpt_interval_steps)
        self.next_ckpt_step = self.ckpt_interval_steps if self.ckpt_interval_steps > 0 else 0
        # Build a fixed pure-student condition set and score deterministic first-action MSE on it.
        # This is cheaper than a real rollout probe but more stable than online per-step MSE.
        self.deploy_probe_size = int(
            self.ppo_config.get("action_chunk_deploy_probe_size", 2048)
        )
        self.deploy_probe_size = max(0, self.deploy_probe_size)
        self.deploy_probe_batch_size = int(
            self.ppo_config.get("action_chunk_deploy_probe_batch_size", 512)
        )
        self.deploy_probe_batch_size = max(1, self.deploy_probe_batch_size)
        self.deploy_probe_interval_steps = int(
            self.ppo_config.get(
                "action_chunk_deploy_probe_interval_steps",
                self.ckpt_interval_steps if self.ckpt_interval_steps > 0 else self.batch_size,
            )
        )
        self.deploy_probe_interval_steps = max(1, self.deploy_probe_interval_steps)
        self.next_deploy_probe_step = max(
            self.model_selection_warmup_steps,
            self.deploy_probe_interval_steps,
        )
        self.deploy_probe_count = 0
        self.deploy_probe_ready = False
        self.last_deploy_probe_first_action_mse = float("nan")
        self.last_deploy_probe_chunk_mse = float("nan")
        if self.deploy_probe_size > 0:
            self.deploy_probe_obs = torch.zeros(
                self.deploy_probe_size, self.obs_shape[0], dtype=torch.float32
            )
            self.deploy_probe_prop = torch.zeros(
                self.deploy_probe_size,
                self.proprio_hist_dim,
                self.proprio_dim,
                dtype=torch.float32,
            )
            self.deploy_probe_target_chunk = torch.zeros(
                self.deploy_probe_size,
                self.chunk_len,
                self.actions_num,
                dtype=torch.float32,
            )
        else:
            self.deploy_probe_obs = None
            self.deploy_probe_prop = None
            self.deploy_probe_target_chunk = None
        self.rollout_pretrain_path = str(
            self.ppo_config.get("rollout_pretrain_path", "")
        ).strip()
        self.rollout_pretrain_updates = int(
            self.ppo_config.get("rollout_pretrain_updates", 0)
        )
        self.rollout_pretrain_batch_size = int(
            self.ppo_config.get("rollout_pretrain_batch_size", self.batch_size)
        )
        self.rollout_pretrain_log_interval = int(
            self.ppo_config.get("rollout_pretrain_log_interval", 100)
        )
        self.optim = torch.optim.Adam(
            self.diffusion_model.parameters(),
            lr=float(self.ppo_config.get("action_chunk_diffusion_lr", 3e-4)),
        )

        self.teacher_action_buffer = torch.zeros(
            self.num_actors, self.chunk_len, self.actions_num, device=self.device
        )
        self.cond_obs_buffer = torch.zeros(
            self.num_actors, self.chunk_len, self.obs_shape[0], device=self.device
        )
        self.cond_prop_buffer = torch.zeros(
            self.num_actors, self.chunk_len, self.proprio_hist_dim, self.proprio_dim, device=self.device
        )
        self.valid_window_len = torch.zeros(self.num_actors, dtype=torch.long, device=self.device)

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

    def _append_teacher_window(self, obs, proprio_hist, teacher_mu):
        self.teacher_action_buffer = torch.roll(self.teacher_action_buffer, shifts=-1, dims=1)
        self.teacher_action_buffer[:, -1, :] = teacher_mu

        self.cond_obs_buffer = torch.roll(self.cond_obs_buffer, shifts=-1, dims=1)
        self.cond_obs_buffer[:, -1, :] = obs

        self.cond_prop_buffer = torch.roll(self.cond_prop_buffer, shifts=-1, dims=1)
        self.cond_prop_buffer[:, -1, :, :] = proprio_hist

        self.valid_window_len = torch.clamp(self.valid_window_len + 1, max=self.chunk_len)

    def _reset_done_windows(self, done_mask):
        if done_mask.any():
            self.valid_window_len[done_mask] = 0
            self.teacher_action_buffer[done_mask] = 0.0
            self.cond_obs_buffer[done_mask] = 0.0
            self.cond_prop_buffer[done_mask] = 0.0

    def _teacher_mix_ratio(self):
        if self.teacher_mix_steps <= 0:
            return 0.0
        return max(0.0, 1.0 - float(self.agent_steps) / float(self.teacher_mix_steps))

    def _begin_student_eval_phase(self):
        if self.student_eval_phase_started:
            return
        self.student_eval_phase_started = True
        self.student_mean_eps_reward.clear()
        self.student_mean_eps_length.clear()
        self.student_step_reward.zero_()
        self.student_step_length.zero_()
        self.student_tracking_active.zero_()

    def _update_student_eval_meter(self, rewards, done_mask):
        if not self.student_eval_phase_started:
            return

        active_mask = self.student_tracking_active.float()
        self.student_step_reward += rewards * active_mask
        self.student_step_length += active_mask

        tracked_done = done_mask & self.student_tracking_active
        done_indices = tracked_done.nonzero(as_tuple=False)
        self.student_mean_eps_reward.update(self.student_step_reward[done_indices])
        self.student_mean_eps_length.update(self.student_step_length[done_indices])

        not_tracked_done = 1.0 - tracked_done.float()
        self.student_step_reward = self.student_step_reward * not_tracked_done
        self.student_step_length = self.student_step_length * not_tracked_done

        # Episodes that reset after the pure-student transition become eligible from the next step.
        self.student_tracking_active = self.student_tracking_active & (~done_mask)
        self.student_tracking_active = self.student_tracking_active | done_mask

    def _update_pure_student_window_len(self, mix, done_mask):
        if mix > 0.0:
            self.pure_student_window_len.zero_()
            return
        self.pure_student_window_len = torch.clamp(
            self.pure_student_window_len + 1, max=self.chunk_len
        )
        if done_mask.any():
            self.pure_student_window_len[done_mask] = 0

    def _maybe_collect_deploy_probe_samples(self):
        if self.deploy_probe_size <= 0 or self.deploy_probe_ready:
            return
        candidate_mask = (self.valid_window_len >= self.chunk_len) & (
            self.pure_student_window_len >= self.chunk_len
        )
        if not candidate_mask.any():
            return

        cond_obs = self.cond_obs_buffer[candidate_mask, 0, :].detach().cpu()
        cond_prop = self.cond_prop_buffer[candidate_mask, 0, :, :].detach().cpu()
        target_chunk = self.teacher_action_buffer[candidate_mask].detach().cpu()
        if cond_obs.shape[0] == 0:
            return

        remaining = self.deploy_probe_size - self.deploy_probe_count
        take = min(remaining, cond_obs.shape[0])
        start = self.deploy_probe_count
        end = start + take
        self.deploy_probe_obs[start:end].copy_(cond_obs[:take])
        self.deploy_probe_prop[start:end].copy_(cond_prop[:take])
        self.deploy_probe_target_chunk[start:end].copy_(target_chunk[:take])
        self.deploy_probe_count = end
        self.deploy_probe_ready = self.deploy_probe_count >= self.deploy_probe_size

    @torch.no_grad()
    def _compute_deploy_probe_metrics(self):
        if not self.deploy_probe_ready or self.deploy_probe_size <= 0:
            return None, None

        prev_stochastic_infer = self.stochastic_infer
        self.stochastic_infer = False
        total = 0
        first_action_mse_sum = 0.0
        chunk_mse_sum = 0.0
        try:
            for start in range(0, self.deploy_probe_size, self.deploy_probe_batch_size):
                end = min(start + self.deploy_probe_batch_size, self.deploy_probe_size)
                obs = self.deploy_probe_obs[start:end].to(self.device)
                proprio_hist = self.deploy_probe_prop[start:end].to(self.device)
                target_chunk = self.deploy_probe_target_chunk[start:end].to(self.device)
                pred_chunk = self.sample_action_chunk(obs, proprio_hist)
                batch_size = end - start
                first_action_mse_sum += float(
                    ((pred_chunk[:, 0, :] - target_chunk[:, 0, :]) ** 2).mean().detach().cpu()
                ) * batch_size
                chunk_mse_sum += float(
                    ((pred_chunk - target_chunk) ** 2).mean().detach().cpu()
                ) * batch_size
                total += batch_size
        finally:
            self.stochastic_infer = prev_stochastic_infer

        if total == 0:
            return None, None
        return first_action_mse_sum / float(total), chunk_mse_sum / float(total)

    @torch.no_grad()
    def sample_action_chunk(self, obs, proprio_hist):
        batch_size = obs.shape[0]
        # Keep deterministic eval truly deterministic across seeds:
        # when stochastic inference is disabled, avoid random x_T initialization.
        if self.stochastic_infer:
            x = torch.randn(batch_size, self.chunk_dim, device=self.device)
        else:
            x = torch.zeros(batch_size, self.chunk_dim, device=self.device)
        for t_idx in reversed(range(self.diffusion_steps_infer)):
            t = torch.full((batch_size,), t_idx, device=self.device, dtype=torch.long)
            eps_pred = self.diffusion_model(obs, proprio_hist, x, t)

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
        return torch.tanh(x).reshape(batch_size, self.chunk_len, self.actions_num)

    def set_eval(self):
        super().set_eval()
        self.diffusion_model.eval()

    def _eval_select_action(self, obs_dict):
        obs = self.running_mean_std(obs_dict["obs"])
        proprio_hist = self.sa_mean_std(obs_dict["proprio_hist"].detach())
        chunk_actions = self.sample_action_chunk(obs, proprio_hist)
        return torch.clamp(chunk_actions[:, 0, :], -1.0, 1.0), {}

    def _compute_chunk_training_loss(self, cond_obs, cond_prop, target_chunk):
        target_flat = target_chunk.reshape(target_chunk.shape[0], -1).detach()
        batch_size = target_flat.shape[0]
        t = torch.randint(0, self.diffusion_steps, (batch_size,), device=self.device)
        noise = torch.randn_like(target_flat)
        x_t = self._q_sample(target_flat, t, noise)
        eps_pred = self.diffusion_model(cond_obs, cond_prop, x_t, t)
        diffusion_loss = ((eps_pred - noise) ** 2).mean()

        x0_pred = self._predict_x0(x_t, t, eps_pred)
        chunk_recon = torch.tanh(x0_pred).reshape(-1, self.chunk_len, self.actions_num)
        first_action_bc_loss = ((chunk_recon[:, 0, :] - target_chunk[:, 0, :]) ** 2).mean()
        chunk_bc_loss = ((chunk_recon - target_chunk) ** 2).mean()

        loss = (
            self.diffusion_loss_coef * diffusion_loss
            + self.first_action_bc_loss_coef * first_action_bc_loss
            + self.chunk_bc_loss_coef * chunk_bc_loss
        )
        return diffusion_loss, first_action_bc_loss, chunk_bc_loss, loss

    def _run_rollout_pretrain_if_enabled(self):
        if not self.rollout_pretrain_path or self.rollout_pretrain_updates <= 0:
            return
        if not os.path.isfile(self.rollout_pretrain_path):
            raise FileNotFoundError(
                f"rollout_pretrain_path not found: {self.rollout_pretrain_path}"
            )

        payload = torch.load(self.rollout_pretrain_path, map_location="cpu")
        cond_obs_all, cond_prop_all, target_chunk_all = build_action_chunk_rollout_dataset(
            payload, self.chunk_len
        )
        dataset_size = cond_obs_all.shape[0]
        if dataset_size == 0:
            raise ValueError("Rollout pretrain dataset is empty after chunk window extraction")

        batch_size = max(1, min(self.rollout_pretrain_batch_size, dataset_size))
        log_interval = max(1, self.rollout_pretrain_log_interval)
        tprint(
            "Rollout pretrain start | "
            f"path={self.rollout_pretrain_path} | samples={dataset_size} | "
            f"updates={self.rollout_pretrain_updates} | batch={batch_size}"
        )

        for update_idx in range(1, self.rollout_pretrain_updates + 1):
            batch_idx = torch.randint(0, dataset_size, (batch_size,))
            cond_obs = cond_obs_all[batch_idx].to(self.device)
            cond_prop = cond_prop_all[batch_idx].to(self.device)
            target_chunk = target_chunk_all[batch_idx].to(self.device)

            diffusion_loss, first_action_bc_loss, chunk_bc_loss, loss = (
                self._compute_chunk_training_loss(cond_obs, cond_prop, target_chunk)
            )
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            step_tag = int(update_idx)
            self.writer.add_scalar(
                "rollout_pretrain/diffusion_loss", float(diffusion_loss.detach().cpu()), step_tag
            )
            self.writer.add_scalar(
                "rollout_pretrain/first_action_bc_loss",
                float(first_action_bc_loss.detach().cpu()),
                step_tag,
            )
            self.writer.add_scalar(
                "rollout_pretrain/chunk_bc_loss", float(chunk_bc_loss.detach().cpu()), step_tag
            )
            self.writer.add_scalar(
                "rollout_pretrain/total_loss", float(loss.detach().cpu()), step_tag
            )

            if update_idx == 1 or update_idx % log_interval == 0 or update_idx == self.rollout_pretrain_updates:
                tprint(
                    "Rollout pretrain | "
                    f"{update_idx}/{self.rollout_pretrain_updates} | "
                    f"total={float(loss.detach().cpu()):.4f} | "
                    f"diff={float(diffusion_loss.detach().cpu()):.4f} | "
                    f"fa_bc={float(first_action_bc_loss.detach().cpu()):.4f} | "
                    f"chunk_bc={float(chunk_bc_loss.detach().cpu()):.4f}"
                )

    def test(self):
        self.set_eval()
        obs_dict = self.env.reset()
        c = 0
        eval_reward_sum = 0.0
        eval_done_sum = 0.0
        while True:
            obs = self.running_mean_std(obs_dict["obs"])
            proprio_hist = self.sa_mean_std(obs_dict["proprio_hist"].detach())
            chunk_actions = self.sample_action_chunk(obs, proprio_hist)
            mu = torch.clamp(chunk_actions[:, 0, :], -1.0, 1.0)
            obs_dict, r, done, _ = self.env.step(mu)
            c += 1
            print(f"Step {c}")
            if self.test_num_steps > 0:
                eval_reward_sum += float(r.float().mean().detach().cpu())
                eval_done_sum += float(done.float().mean().detach().cpu())
                if c >= self.test_num_steps:
                    avg_reward = eval_reward_sum / float(c)
                    avg_done_rate = eval_done_sum / float(c)
                    print(
                        "EvalSummary "
                        f"steps={c} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
                    )
                    break

    def train(self):
        _t = time.time()
        _last_t = time.time()

        self._run_rollout_pretrain_if_enabled()

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

            self._append_teacher_window(
                obs=input_dict["obs"],
                proprio_hist=input_dict["proprio_hist"],
                teacher_mu=teacher_mu.detach(),
            )

            train_mask = self.valid_window_len >= self.chunk_len
            if train_mask.any():
                cond_obs = self.cond_obs_buffer[train_mask, 0, :]
                cond_prop = self.cond_prop_buffer[train_mask, 0, :, :]
                target_chunk = self.teacher_action_buffer[train_mask]
                (
                    diffusion_loss,
                    first_action_bc_loss,
                    chunk_bc_loss,
                    loss,
                ) = self._compute_chunk_training_loss(
                    cond_obs=cond_obs,
                    cond_prop=cond_prop,
                    target_chunk=target_chunk,
                )
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                diffusion_loss = torch.zeros((), device=self.device)
                first_action_bc_loss = torch.zeros((), device=self.device)
                chunk_bc_loss = torch.zeros((), device=self.device)
                loss = torch.zeros((), device=self.device)

            with torch.no_grad():
                action_chunk = self.sample_action_chunk(
                    input_dict["obs"], input_dict["proprio_hist"]
                )
                student_action = torch.clamp(action_chunk[:, 0, :], -1.0, 1.0)
                student_action_mse = ((student_action - teacher_mu.detach()) ** 2).mean()
                mix = self._teacher_mix_ratio()
                if mix > 0.0:
                    mu_env = torch.clamp(
                        mix * teacher_mu.detach() + (1.0 - mix) * student_action,
                        -1.0,
                        1.0,
                    )
                else:
                    mu_env = student_action

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

            done_mask = done > 0
            if done_mask.ndim > 1:
                done_mask = done_mask.squeeze(-1)
            self._reset_done_windows(done_mask)
            if mix <= 0.0:
                self._begin_student_eval_phase()
            self._update_student_eval_meter(r, done_mask)
            self._update_pure_student_window_len(mix, done_mask)
            self._maybe_collect_deploy_probe_samples()

            self.direct_info["diffusion_loss"] = float(diffusion_loss.detach().cpu())
            self.direct_info["first_action_bc_loss"] = float(first_action_bc_loss.detach().cpu())
            self.direct_info["chunk_bc_loss"] = float(chunk_bc_loss.detach().cpu())
            self.direct_info["total_loss"] = float(loss.detach().cpu())
            self.direct_info["done_rate"] = float(done.float().mean().detach().cpu())
            self.direct_info["teacher_mix_ratio"] = float(mix)
            self.direct_info["student_action_mse"] = float(student_action_mse.detach().cpu())
            self.direct_info["student_eval_phase_started"] = float(self.student_eval_phase_started)
            self.direct_info["student_eval_episode_reward"] = float(
                self.student_mean_eps_reward.get_mean()
            )
            self.direct_info["student_eval_episode_length"] = float(
                self.student_mean_eps_length.get_mean()
            )
            self.direct_info["student_eval_tracking_frac"] = float(
                self.student_tracking_active.float().mean().detach().cpu()
            )
            self.direct_info["deploy_probe_samples"] = float(self.deploy_probe_count)
            self.direct_info["deploy_probe_ready"] = float(self.deploy_probe_ready)
            self.direct_info["deploy_probe_first_action_mse"] = float(
                self.last_deploy_probe_first_action_mse
            )
            self.direct_info["deploy_probe_chunk_mse"] = float(self.last_deploy_probe_chunk_mse)
            self._update_env_info(info)
            self.log_tensorboard()

            if self.ckpt_interval_steps > 0 and self.next_ckpt_step > 0:
                while self.agent_steps >= self.next_ckpt_step:
                    self.save(os.path.join(self.nn_dir, "model_last"))
                    self.next_ckpt_step += self.ckpt_interval_steps

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
            student_eval_mean_rewards = self.student_mean_eps_reward.get_mean()
            if (
                self.agent_steps >= self.model_selection_warmup_steps
                and self.student_eval_phase_started
                and len(self.student_mean_eps_reward) > 0
                and student_eval_mean_rewards > self.best_student_rewards
            ):
                self.save(os.path.join(self.nn_dir, "model_best_student_reward"))
                self.best_student_rewards = student_eval_mean_rewards
            if (
                self.agent_steps >= self.model_selection_warmup_steps
                and float(student_action_mse.detach().cpu()) < self.best_student_action_mse
            ):
                self.save(os.path.join(self.nn_dir, "model_best_student"))
                self.best_student_action_mse = float(student_action_mse.detach().cpu())
            if (
                self.deploy_probe_ready
                and self.agent_steps >= self.model_selection_warmup_steps
                and self.agent_steps >= self.next_deploy_probe_step
            ):
                probe_first_action_mse, probe_chunk_mse = self._compute_deploy_probe_metrics()
                if probe_first_action_mse is not None:
                    self.last_deploy_probe_first_action_mse = probe_first_action_mse
                    self.last_deploy_probe_chunk_mse = probe_chunk_mse
                    if probe_first_action_mse < self.best_deploy_probe_first_action_mse:
                        self.save(os.path.join(self.nn_dir, "model_best_deploy_probe"))
                        self.best_deploy_probe_first_action_mse = probe_first_action_mse
                self.next_deploy_probe_step += self.deploy_probe_interval_steps

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()
            best_student_str = (
                f"{self.best_student_rewards:.2f}"
                if self.best_student_rewards > -9999.0
                else "N/A"
            )
            best_student_mse_str = (
                f"{self.best_student_action_mse:.4f}"
                if self.best_student_action_mse < float("inf")
                else "N/A"
            )
            best_deploy_probe_str = (
                f"{self.best_deploy_probe_first_action_mse:.4f}"
                if self.best_deploy_probe_first_action_mse < float("inf")
                else "N/A"
            )
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | FPS: {all_fps:.1f} | "
                f"Last FPS: {last_fps:.1f} | Current Best: {self.best_rewards:.2f} | "
                f"Best Student Reward: {best_student_str} | Best Student MSE: {best_student_mse_str} | "
                f"Best Deploy Probe: {best_deploy_probe_str}"
            )
            tprint(info_string)

    def restore_train(self, fn):
        super().restore_train(fn)
        if not fn:
            return
        checkpoint = torch.load(fn)
        if "diffusion_model" in checkpoint:
            self.diffusion_model.load_state_dict(checkpoint["diffusion_model"])
            cprint("Loaded action-chunk diffusion_model for train resume", "green")

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
        torch.save(weights, f"{name}.ckpt")
