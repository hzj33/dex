"""
DAgger (Dataset Aggregation) student learner - dexscrew 集成版本

定位
----
- 只替换原项目中 "Adaptation Module Training"（类似 ProprioAdapt）的学生训练阶段。
- 不修改 teacher PPO；不修改任务/环境/资产；不引入新的 RL 算法。
- 与 DOTPGStudent / ProprioAdapt 同构：共用 ActorCritic / RunningMeanStd；frozen base policy；仅训练 adapt_tconv。

训练流程 (canonical DAgger)
--------------------------
    while training:
        1. student rollout -> 得到 student-visited state (obs_dict)
        2. 在 student-visited state 上查询 teacher -> teacher_action / teacher_extrin
        3. 将 (obs, proprio_hist, priv_info, point_cloud, teacher_action, teacher_extrin)
           追加到 DAgger 聚合数据集
        4. env.step(mixed_action)  (beta-mixing: teacher vs student)
        5. 从聚合数据集采样若干 batch，用监督损失 (latent + BC) 更新 adapt_tconv

损失
----
    z_t      = tanh(env_mlp(priv) [+ maxpool(point_mlp(pc))])      # teacher extrin
    z_hat_t  = tanh(adapt_tconv(proprio_hist))                      # student extrin
    a_teacher= pi_base(obs, z_t)
    a_student= pi_base(obs, z_hat_t)
    latent_loss = || z_hat_t - z_t ||_2^2
    bc_loss     = || clamp(a_student) - clamp(a_teacher) ||_2^2
    adapt_loss  = latent_coef * latent_loss + action_coef * bc_loss

只对 adapt_tconv 反传梯度；其他所有参数（actor_mlp / mu / priv_mlp / point_mlp）保持冻结。
"""

import os
import time
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from termcolor import cprint
from tensorboardX import SummaryWriter

from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from dexscrew.algo.DAgger.buffer import DAggerBuffer
from dexscrew.utils.misc import AverageScalarMeter, tprint


class DAggerConfig:
    """DAgger 训练超参数（可由 train.dagger 覆盖）。"""

    def __init__(self, config_dict: Optional[dict] = None):
        # ----- dataset -----
        self.buffer_size = int(5e5)
        self.buffer_device = "cpu"           # cpu / cuda / cuda:0
        # 默认 float32：fp16 在 teacher_action 上的量化误差 (~1e-3) 会给 BC loss
        # 加一个不可去除的下限，调试期统一用 fp32 更稳。
        self.buffer_dtype = "float32"        # float16 / float32
        # 每个 env step 向 buffer 追加多少 env 的样本（0 / >=num_envs 表示写入全部）
        self.add_num_envs = 0
        self.store_priv_info = True
        # 学生侧不需要 pc 来重建 extrin（teacher_extrin 已入库），默认关闭省内存。
        self.store_point_cloud = False

        # ----- optimization -----
        self.batch_size = 4096
        # 默认每个 env step 跑 4 次 supervised update，远强于 ProprioAdapt 的 1 次
        # on-policy batch（ProprioAdapt 每 step 用当帧 num_envs 样本跑 1 次）；
        # 用聚合 buffer 时通常需要几个 epoch 才收敛。
        self.updates_per_collect = 4
        self.lr = 3e-4

        # ----- DAgger β-mixing -----
        # env 实际执行的动作 = β * teacher + (1-β) * student
        # 默认仅切换（按概率整体选 teacher / student），而不做动作插值：
        # action_mixing = "switch" (recommended) | "blend"
        self.action_mixing = "switch"
        self.beta_start = 1.0
        self.beta_min = 0.0
        # 每一个 env step β 衰减的增量（线性衰减）
        self.beta_decay = 5e-4
        # 纯 student rollout（beta=0, action_mixing='switch'）
        self.pure_student_rollout = False

        # ----- supervised coefficients -----
        self.latent_coef = 1.0
        self.action_bc_coef = 1.0

        # ----- schedule -----
        self.max_agent_steps = int(1e9)
        # 当 buffer 中样本少于 min_buffer_for_update 时，不做 supervised update
        self.min_buffer_for_update = 1024
        # 前 warmup_collect_steps 步只采集、不更新 adapt_tconv（避免 buffer 过小时过拟合）
        self.warmup_collect_steps = 0

        # ----- logging / save -----
        self.save_interval_agent_steps = int(1e8)

        # ----- teacher/student rollout sanity + periodic pure-student eval -----
        # 0 / <0 关闭；>0 表示每 eval_interval_agent_steps 跑一次 pure-student eval
        self.eval_interval_agent_steps = int(5e5)
        self.eval_num_episodes = 32
        self.eval_max_steps = 0
        # 启动 restore_train 后是否跑 teacher/student 各自的 sanity rollout
        self.sanity_check_on_restore = True
        self.sanity_num_episodes = 16
        self.sanity_max_steps = 400

        # ----- strict key audit -----
        # 缺失的 base-policy 权重（actor_mlp / mu / env_mlp / point_mlp / sigma）
        # 会导致 silent-fail：默认严格校验，缺就 RuntimeError。
        self.strict_base_policy = True

        # ----- test -----
        self.test_num_episodes = 20
        self.test_max_steps = 0

        if config_dict is not None:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    setattr(self, key, value)


def _as_plain_dict(maybe_dict):
    if maybe_dict is None:
        return {}
    if isinstance(maybe_dict, dict):
        return dict(maybe_dict)
    try:
        return dict(maybe_dict)
    except Exception:
        return {}


class DAggerStudent:
    """DAgger-based student adapter.

    复用 ProprioAdapt / DOTPGStudent 的整体工程结构：
    - ActorCritic 作为 base policy（加载 teacher ckpt 后冻结所有子模块）
    - 仅 adapt_tconv 参与训练；sa_mean_std 可选更新
    - DAgger buffer 只存 raw 输入 + teacher 标签
    - SummaryWriter 日志、best/last/periodic checkpoint、tprint 进度条
    """

    def __init__(self, env, output_dir, full_config, student_dim: int = 24):
        self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo

        # ---- DAgger 配置 ----
        dagger_config_dict = _as_plain_dict(full_config.train.get("dagger", {}))
        self.config = DAggerConfig(dagger_config_dict)

        # ---- Env ----
        self.env = env
        self.num_actors = self.ppo_config["num_actors"]
        self.observation_space = self.env.observation_space
        self.obs_shape = self.observation_space.shape
        self.action_space = self.env.action_space
        self.actions_num = self.action_space.shape[0]
        self.proprio_dim = self.ppo_config.get("proprio_dim", 24)
        self.student_obs_shape = (student_dim * 3,)

        # ---- Priv / Proprio / PointCloud ----
        self.priv_info = self.ppo_config["priv_info"]
        self.normalize_priv = self.ppo_config["normalize_priv"]
        self.priv_info_dim = self.env.priv_info_dim
        self.proprio_adapt = self.ppo_config["proprio_adapt"]
        if not self.proprio_adapt:
            raise ValueError(
                "DAggerStudent requires train.ppo.proprio_adapt=True (student uses proprio_hist -> extrin)."
            )
        self.proprio_hist_dim = self.env.prop_hist_len

        self.asymm_actor_critic = self.ppo_config.get("asymm_actor_critic", False)
        self.critic_info_dim = self.ppo_config.get("critic_info_dim", 0)

        self.point_cloud_buffer_dim = self.env.point_cloud_buffer_dim
        self.proprio_len = self.ppo_config["proprio_len"]
        self.use_point_cloud_info = self.ppo_config["use_point_cloud_info"]
        self.normalize_point_cloud = self.ppo_config["normalize_point_cloud"]

        # ---- Teacher / student 共用的 ActorCritic ----
        net_config = {
            "actor_units": self.network_config.mlp.units,
            "priv_mlp_units": self.network_config.priv_mlp.units,
            "actions_num": self.actions_num,
            "input_shape": self.obs_shape,
            "priv_info": self.priv_info,
            "proprio_adapt": self.proprio_adapt,
            "priv_info_dim": self.priv_info_dim,
            "critic_info_dim": self.critic_info_dim,
            "asymm_actor_critic": self.asymm_actor_critic,
            "point_mlp_units": self.network_config.point_mlp.units,
            "use_point_cloud_info": self.use_point_cloud_info,
            "proprio_len": self.proprio_len,
            "proprio_dim": self.proprio_dim,
        }
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.model.eval()

        # ---- Running mean / std (保持与 padapt 一致) ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd(
            (self.proprio_hist_dim, self.proprio_dim)
        ).to(self.device)
        self.sa_mean_std.train()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.point_cloud_mean_std.eval()

        # ---- Output ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "dagger_nn")
        self.tb_dir = os.path.join(self.output_dir, "dagger_tb")
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        self.writer = SummaryWriter(self.tb_dir)

        # ---- Optim (只更新 adapt_tconv) ----
        adapt_params = []
        for name, p in self.model.named_parameters():
            if "adapt_tconv" in name:
                adapt_params.append(p)
            else:
                p.requires_grad = False
        if len(adapt_params) == 0:
            raise RuntimeError(
                "DAggerStudent: no adapt_tconv parameters found on ActorCritic. "
                "Check that train.ppo.proprio_adapt=True."
            )
        self.optim = torch.optim.Adam(adapt_params, lr=float(self.config.lr))

        # ---- Stats ----
        self.batch_size = self.num_actors
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        # 训练期 mixed-policy rollout 的奖励（含 β-teacher 成分）——仅作训练诊断
        # 命名: best_mixed_reward。保留 best_rewards 作为同名别名以兼容外部脚本。
        self.best_mixed_reward = -10000.0
        # 真正反映 student policy 水平：pure-student eval 的奖励——这才用于 model_best
        self.best_eval_rewards = -10000.0
        # 最近一次 pure-student eval 的平均 reward / length（用于进度条与 TB 比较曲线）
        self.last_eval_reward = float("nan")
        self.last_eval_length = float("nan")
        self.last_eval_agent_steps = 0
        self.agent_steps = 0
        self.collect_steps = 0
        self.direct_info = {}
        self.step_reward = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self.step_length = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )

        # ---- DAgger buffer ----
        self._extrin_dim = self.network_config.priv_mlp.units[-1]
        if self.use_point_cloud_info:
            self._extrin_dim += self.network_config.point_mlp.units[-1]

        buffer_storage_device = str(self.config.buffer_device).lower()
        if buffer_storage_device in ("cuda", "gpu"):
            buffer_storage_device = str(self.device)
        buffer_dtype_cfg = str(self.config.buffer_dtype).lower()
        buffer_dtype = (
            torch.float16 if buffer_dtype_cfg in ("fp16", "float16", "half") else torch.float32
        )

        pc_shape = (
            (self.point_cloud_buffer_dim, 3) if self.use_point_cloud_info else None
        )
        # 注：student 侧 adapt_tconv 输出维度已在 ActorCritic 构造时包含
        # point cloud 分量，replay 时不需要重新跑 point_mlp（teacher_extrin 已
        # 作为 label 直接存入 buffer）。因此 store_point_cloud 是可选项，
        # 关闭可显著降低 buffer 占用（尤其是 point_cloud_sampled_dim 较大时）。

        self.dataset = DAggerBuffer(
            obs_dim=self.obs_shape[0],
            priv_info_dim=self.priv_info_dim,
            proprio_hist_shape=(self.proprio_hist_dim, self.proprio_dim),
            point_cloud_shape=pc_shape,
            action_dim=self.actions_num,
            teacher_extrin_dim=self._extrin_dim,
            max_size=int(self.config.buffer_size),
            device=str(self.device),
            storage_device=buffer_storage_device,
            dtype=buffer_dtype,
            return_dtype=torch.float32,
            store_priv_info=bool(self.config.store_priv_info),
            store_point_cloud=bool(self.config.store_point_cloud),
        )

        # ---- β (DAgger mixing coefficient) ----
        if bool(self.config.pure_student_rollout):
            self._beta = 0.0
        else:
            self._beta = float(self.config.beta_start)

    # ------------------------------------------------------------
    # Backward-compat alias: historical attribute `best_rewards` was used as
    # "best mixed-policy training reward". That semantic is unchanged, but the
    # name misleads readers into thinking it's pure-student. Expose both names
    # pointing to the same state so old ckpts / dashboards keep working and new
    # code can use the explicit `best_mixed_reward`.
    # ------------------------------------------------------------
    @property
    def best_rewards(self) -> float:
        return float(self.best_mixed_reward)

    @best_rewards.setter
    def best_rewards(self, value: float) -> None:
        self.best_mixed_reward = float(value)

    # ------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------

    def _normalize_point_cloud(self, pc_raw: torch.Tensor) -> torch.Tensor:
        if not self.use_point_cloud_info:
            return pc_raw
        if self.normalize_point_cloud:
            return self.point_cloud_mean_std(pc_raw.reshape(-1, 3)).reshape(
                (pc_raw.shape[0], -1, 3)
            )
        return pc_raw

    def _build_input_dict(self, obs_dict) -> dict:
        """Build normalized input dict (一次性构建 student+teacher 都需要的输入)。"""
        input_dict = {
            "obs": self.running_mean_std(obs_dict["obs"]),
            "proprio_hist": self.sa_mean_std(obs_dict["proprio_hist"].detach()),
        }
        if self.priv_info and "priv_info" in obs_dict:
            input_dict["priv_info"] = (
                self.priv_mean_std(obs_dict["priv_info"])
                if self.normalize_priv
                else obs_dict["priv_info"]
            )
        if self.use_point_cloud_info:
            input_dict["point_cloud_info"] = self._normalize_point_cloud(
                obs_dict["point_cloud_info"]
            )
        return input_dict

    # ------------------------------------------------------------
    # Teacher / student forward passes
    # ------------------------------------------------------------

    @torch.no_grad()
    def _teacher_extrin_from_input(self, input_dict) -> torch.Tensor:
        """Compute teacher extrin z = tanh(env_mlp(priv)[+maxpool(point_mlp(pc))])."""
        extrin = self.model.env_mlp(input_dict["priv_info"])
        if self.use_point_cloud_info:
            pcs = self.model.point_mlp(input_dict["point_cloud_info"])
            pcs = torch.max(pcs, 1)[0]
            extrin = torch.cat([extrin, pcs], dim=-1)
        return torch.tanh(extrin)

    @torch.no_grad()
    def _teacher_action_from_extrin(
        self, obs_norm: torch.Tensor, extrin: torch.Tensor
    ) -> torch.Tensor:
        x = self.model.actor_mlp(torch.cat([obs_norm, extrin], dim=-1))
        return torch.clamp(self.model.mu(x), -1.0, 1.0)

    def _student_extrin_from_input(self, input_dict) -> torch.Tensor:
        """Student extrin from proprio_hist (可带梯度，用于训练 adapt_tconv)。

        注意：ActorCritic 内部 adapt_tconv 的输出维度在构造时已经包含 point cloud
        分量（见 models.py：temporal_fusing_output_dim = 8 + 32 when use_point_cloud_info）。
        student 这一侧不应再 concat pc 特征，否则会与 teacher extrin 维度不一致、
        且与 actor_mlp 输入维度对不上。
        """
        extrin = self.model.adapt_tconv(input_dict["proprio_hist"])
        return torch.tanh(extrin)

    def _student_action_from_extrin(
        self, obs_norm: torch.Tensor, extrin: torch.Tensor
    ) -> torch.Tensor:
        x = self.model.actor_mlp(torch.cat([obs_norm, extrin], dim=-1))
        return self.model.mu(x)

    @torch.no_grad()
    def _get_teacher_labels(self, input_dict):
        teacher_extrin = self._teacher_extrin_from_input(input_dict)
        teacher_action = self._teacher_action_from_extrin(
            input_dict["obs"], teacher_extrin
        )
        return teacher_action, teacher_extrin

    @torch.no_grad()
    def _get_student_action(self, input_dict):
        student_extrin = self._student_extrin_from_input(input_dict)
        student_action = self._student_action_from_extrin(
            input_dict["obs"], student_extrin
        )
        return torch.clamp(student_action, -1.0, 1.0), student_extrin

    # ------------------------------------------------------------
    # β-mixing action
    # ------------------------------------------------------------

    def _decay_beta(self):
        if bool(self.config.pure_student_rollout):
            self._beta = 0.0
            return
        self._beta = max(
            float(self.config.beta_min),
            float(self._beta) - float(self.config.beta_decay),
        )

    def _mix_action(
        self, student_action: torch.Tensor, teacher_action: torch.Tensor
    ) -> torch.Tensor:
        """β-mixing:
            - action_mixing="switch": 每个 env 以概率 β 整体执行 teacher 动作
            - action_mixing="blend":  env_action = β*teacher + (1-β)*student
        """
        if bool(self.config.pure_student_rollout) or self._beta <= 0.0:
            return student_action
        if self._beta >= 1.0 and self.config.action_mixing == "switch":
            return teacher_action

        if self.config.action_mixing == "blend":
            env_action = self._beta * teacher_action + (1.0 - self._beta) * student_action
            return env_action.clamp(-1.0, 1.0)

        # default: per-env switch
        mask = torch.rand(
            (student_action.shape[0], 1), device=student_action.device
        ) < self._beta
        env_action = torch.where(mask, teacher_action, student_action)
        return env_action.clamp(-1.0, 1.0)

    # ------------------------------------------------------------
    # DAgger supervised update from aggregated dataset
    # ------------------------------------------------------------

    def _supervised_update(self):
        """Sample a batch from DAgger dataset and update adapt_tconv.

        Loss:
          latent_loss = || z_hat - z_teacher ||^2
          bc_loss     = || clamp(a_student) - clamp(a_teacher) ||^2
          adapt_loss  = latent_coef * latent_loss + action_coef * bc_loss
        只对 adapt_tconv 反传梯度（其他模块的 requires_grad=False）。
        """
        if len(self.dataset) < int(self.config.min_buffer_for_update):
            return None

        batch = self.dataset.sample(int(self.config.batch_size))

        # 从 buffer 采样出来的 raw obs/proprio_hist，需要再通过 running_mean_std / sa_mean_std。
        # 这里不应再更新 running stats（会被 buffer 样本污染）→ 临时切 eval。
        was_training_obs = self.running_mean_std.training
        was_training_sa = self.sa_mean_std.training
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

        obs_norm = self.running_mean_std(batch["obs"])
        proprio_norm = self.sa_mean_std(batch["proprio_hist"])

        # --- student extrin (可导) ---
        # adapt_tconv 的输出维度已在 ActorCritic 构造时包含 point cloud 分量
        # (见 models.py)。student 侧不再 concat pc，直接 tanh 即可，与 teacher
        # extrin（tanh(env_mlp(priv) [+ maxpool(point_mlp(pc))])) 对齐。
        student_extrin = torch.tanh(self.model.adapt_tconv(proprio_norm))

        # --- latent loss ---
        teacher_extrin_label = batch["teacher_extrin"].detach()
        latent_loss = (student_extrin - teacher_extrin_label).pow(2).mean()

        # --- BC loss ---
        student_action = self._student_action_from_extrin(obs_norm, student_extrin)
        student_action = torch.clamp(student_action, -1.0, 1.0)
        teacher_action = torch.clamp(batch["teacher_action"].detach(), -1.0, 1.0)
        bc_loss = F.mse_loss(student_action, teacher_action)

        adapt_loss = (
            float(self.config.latent_coef) * latent_loss
            + float(self.config.action_bc_coef) * bc_loss
        )

        self.optim.zero_grad()
        adapt_loss.backward()
        self.optim.step()

        if was_training_obs:
            self.running_mean_std.train()
        if was_training_sa:
            self.sa_mean_std.train()

        return {
            "latent_loss": float(latent_loss.detach().cpu()),
            "bc_loss": float(bc_loss.detach().cpu()),
            "adapt_loss": float(adapt_loss.detach().cpu()),
        }

    # ------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------

    def _aggregate(self, obs_dict, teacher_action, teacher_extrin):
        add_n = int(self.config.add_num_envs)
        if add_n <= 0 or add_n >= self.num_actors:
            sel = None
        else:
            sel = torch.randperm(self.num_actors, device=teacher_action.device)[:add_n]

        def _pick(x):
            return x if sel is None else x[sel]

        obs_raw = _pick(obs_dict["obs"].detach())
        proprio_hist_raw = _pick(obs_dict["proprio_hist"].detach())
        t_action = _pick(teacher_action.detach())
        t_extrin = _pick(teacher_extrin.detach())

        priv_info_raw = None
        if self.config.store_priv_info and "priv_info" in obs_dict:
            priv_info_raw = _pick(obs_dict["priv_info"].detach())

        point_cloud_raw = None
        if self.use_point_cloud_info and self.config.store_point_cloud:
            point_cloud_raw = _pick(obs_dict["point_cloud_info"].detach())

        self.dataset.add_batch(
            obs=obs_raw,
            proprio_hist=proprio_hist_raw,
            teacher_action=t_action,
            teacher_extrin=t_extrin,
            priv_info=priv_info_raw,
            point_cloud_info=point_cloud_raw,
        )

    # ------------------------------------------------------------
    # Logging / mode
    # ------------------------------------------------------------

    def _log_training_info(self, extra_info):
        """Canonical unified logging (follows DOTPG reference).

        Shared comparison metrics (identical keys across DOTPG / DAgger / BC):
          - global_step                  self.agent_steps
          - train/avg_episode_return     mean_eps_reward (DAgger: mixed-policy rollout;
                                         compare student quality via eval/student_reward)
          - train/avg_episode_length     mean_eps_length
          - eval/student_reward          last_eval_reward (pure-student periodic eval)
          - eval/student_length          last_eval_length
          - best/student_eval_reward     self.best_eval_rewards (pure-student best)
          - task/angular_velocity / task/angular_position / task/positive_screw_ratio
          - imitation/total_loss         adapt_loss (adapt_tconv supervised total)

        Algorithm-specific diagnostics kept under the dagger/ namespace.
        """
        step = int(self.agent_steps)
        mean_r_train = float(self.mean_eps_reward.get_mean())
        mean_l_train = float(self.mean_eps_length.get_mean())

        shared = {
            "global_step": float(step),
            "train/avg_episode_return": mean_r_train,
            "train/avg_episode_length": mean_l_train,
        }
        if np.isfinite(self.last_eval_reward):
            shared["eval/student_reward"] = float(self.last_eval_reward)
        if np.isfinite(self.last_eval_length):
            shared["eval/student_length"] = float(self.last_eval_length)
        if self.best_eval_rewards > -1e4:
            shared["best/student_eval_reward"] = float(self.best_eval_rewards)

        for src, dst in (
            ("angular_velocity", "task/angular_velocity"),
            ("angular_position", "task/angular_position"),
            ("positive_vel_ratio", "task/positive_screw_ratio"),
        ):
            v = self.direct_info.get(src, None)
            if v is not None:
                shared[dst] = float(v)

        adapt_loss = self.direct_info.get("adapt_loss", None)
        if adapt_loss is not None:
            shared["imitation/total_loss"] = float(adapt_loss)

        # ---- write canonical shared keys to TB ----
        for k, v in shared.items():
            self.writer.add_scalar(k, v, step)

        # ---- algorithm-specific diagnostics (dagger/ namespace) ----
        algo_payload = {}
        beta_val = float(self._beta)
        self.writer.add_scalar("dagger/beta", beta_val, step)
        algo_payload["dagger/beta"] = beta_val
        buf_val = float(len(self.dataset))
        self.writer.add_scalar("dagger/buffer_size", buf_val, step)
        algo_payload["dagger/buffer_size"] = buf_val
        if np.isfinite(mean_r_train):
            self.writer.add_scalar("dagger/mixed_reward", mean_r_train, step)
            algo_payload["dagger/mixed_reward"] = mean_r_train
        if self.best_mixed_reward > -1e4:
            best_mixed_val = float(self.best_mixed_reward)
            self.writer.add_scalar("dagger/best_mixed_reward", best_mixed_val, step)
            algo_payload["dagger/best_mixed_reward"] = best_mixed_val
        for k in ("latent_loss", "bc_loss"):
            v = self.direct_info.get(k, None)
            if v is not None:
                fv = float(v)
                self.writer.add_scalar(f"dagger/{k}", fv, step)
                algo_payload[f"dagger/{k}"] = fv

        if isinstance(extra_info, dict):
            for k, v in extra_info.items():
                if v is None:
                    continue
                if isinstance(v, torch.Tensor):
                    if v.dim() != 0:
                        continue
                    v = float(v.detach().cpu())
                elif isinstance(v, (int, float, np.integer, np.floating)):
                    v = float(v)
                else:
                    continue
                self.writer.add_scalar(f"env/{k}", v, step)

        # ---- direct wandb.log for shared + algo-specific keys (W&B reliability) ----
        # sync_tensorboard 在长 run 下对 dagger/* 会滞后/停更（shared keys 一路显式 commit，
        # TB-only 通道会逐渐落后），所以把 algo_payload 也并入显式 wandb.log 通道。
        self._wandb_log_shared({**shared, **algo_payload})

        # ---- periodic TB flush ----
        iter_idx = step // max(int(self.batch_size), 1)
        if (iter_idx % 50) == 0:
            try:
                self.writer.flush()
            except Exception:
                pass

    def _wandb_log_shared(self, metrics: dict):
        """Minimal direct wandb.log identical across DOTPG / DAgger / BC."""
        try:
            import wandb  # type: ignore
            if getattr(wandb, "run", None) is None:
                return
            payload = {}
            for k, v in metrics.items():
                if v is None:
                    continue
                try:
                    fv = float(v)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(fv):
                    continue
                payload[k] = fv
            if "global_step" not in payload:
                payload["global_step"] = float(self.agent_steps)
            wandb.log(payload, commit=True)
        except Exception:
            pass

    def set_eval(self):
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    def set_train(self):
        # ActorCritic 保持 eval（推理），但 adapt_tconv 显式 train() 仍无状态差异。
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.train()
        self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    # ------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------

    def _base_policy_prefixes(self):
        """Prefixes of parameter names that constitute teacher / base policy.

        `adapt_tconv` is the only tensor the student trains; anything else must
        come from the teacher ckpt or training is silently broken.
        """
        prefixes = ["actor_mlp.", "mu.", "value.", "sigma"]
        if self.priv_info:
            prefixes.append("env_mlp.")
        if self.use_point_cloud_info:
            prefixes.append("point_mlp.")
        return tuple(prefixes)

    def _audit_ckpt_against_model(self, ckpt_state_dict: dict) -> dict:
        """Return dict with keys: loaded, missing_base, missing_adapt, shape_mismatch,
        unexpected. Dumps colored report. Raises if base-policy keys are missing /
        shape-mismatched and strict_base_policy is on.
        """
        model_state = self.model.state_dict()
        model_keys = set(model_state.keys())
        ckpt_keys = set(ckpt_state_dict.keys())

        base_prefixes = self._base_policy_prefixes()
        required_base = [k for k in model_keys if k.startswith(base_prefixes)]

        loaded, shape_mismatch, missing_base, missing_adapt = [], [], [], []
        for k in model_keys:
            if k in ckpt_keys:
                if model_state[k].shape == ckpt_state_dict[k].shape:
                    loaded.append(k)
                else:
                    shape_mismatch.append(
                        (k, tuple(ckpt_state_dict[k].shape),
                         tuple(model_state[k].shape))
                    )
            else:
                if k.startswith("adapt_tconv."):
                    missing_adapt.append(k)
                else:
                    missing_base.append(k)
        unexpected = sorted(ckpt_keys - model_keys)

        cprint(
            f"[DAgger] ckpt audit | loaded={len(loaded)} | "
            f"missing_base={len(missing_base)} | "
            f"missing_adapt_tconv={len(missing_adapt)} | "
            f"shape_mismatch={len(shape_mismatch)} | "
            f"unexpected={len(unexpected)}",
            "cyan",
            attrs=["bold"],
        )
        if missing_adapt:
            cprint(
                f"  • missing adapt_tconv keys (will be TRAINED from init): "
                f"{missing_adapt}",
                "yellow",
            )
        if missing_base:
            cprint(f"  • MISSING BASE-POLICY keys: {missing_base}", "red")
        if shape_mismatch:
            for k, src, dst in shape_mismatch:
                cprint(
                    f"  • SHAPE MISMATCH {k}: ckpt {src} vs model {dst}",
                    "red",
                )
        if unexpected:
            cprint(f"  • unexpected (ignored): {unexpected}", "grey")

        if bool(self.config.strict_base_policy) and (missing_base or shape_mismatch):
            raise RuntimeError(
                "[DAgger] teacher ckpt is missing or incompatible with the base "
                "policy. Required base-policy keys:\n  "
                + ", ".join(required_base)
                + "\nSet train.dagger.strict_base_policy=False to override "
                "(not recommended)."
            )

        return {
            "loaded": loaded,
            "missing_base": missing_base,
            "missing_adapt": missing_adapt,
            "shape_mismatch": shape_mismatch,
            "unexpected": unexpected,
        }

    def restore_train(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, map_location=self.device, weights_only=False)
        cprint(
            f"[DAgger] loading teacher checkpoint: {fn}",
            "yellow",
            attrs=["bold"],
        )

        ckpt_model = checkpoint.get("model", {})
        # Drop ckpt entries whose shape doesn't match model — load_state_dict with
        # strict=False would otherwise raise on shape mismatches in newer torch.
        model_state = self.model.state_dict()
        compatible = {
            k: v for k, v in ckpt_model.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        missing, unexpected = self.model.load_state_dict(compatible, strict=False)
        del missing, unexpected   # _audit_ckpt_against_model does a richer report
        # run full audit (covers missing_base / shape_mismatch / unexpected)
        self._audit_ckpt_against_model(ckpt_model)

        # Load normalization stats — these MUST come from teacher, otherwise obs
        # normalization silently diverges.
        if "running_mean_std" not in checkpoint:
            raise RuntimeError(
                "[DAgger] teacher ckpt has no running_mean_std — "
                "obs normalization would be wrong."
            )
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_priv:
            if "priv_mean_std" not in checkpoint:
                raise RuntimeError(
                    "[DAgger] normalize_priv=True but ckpt has no priv_mean_std"
                )
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if self.normalize_point_cloud:
            if "point_cloud_mean_std" not in checkpoint:
                raise RuntimeError(
                    "[DAgger] normalize_point_cloud=True but ckpt has no "
                    "point_cloud_mean_std"
                )
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )
        if "sa_mean_std" in checkpoint:
            try:
                self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
                cprint(
                    "[DAgger] loaded teacher sa_mean_std (rare; keep in train "
                    "mode to keep adapting to student rollouts).",
                    "grey",
                )
            except Exception:
                pass

        # Freeze base policy (everything except adapt_tconv) once weights loaded
        for name, p in self.model.named_parameters():
            p.requires_grad = ("adapt_tconv" in name)
        self.model.eval()

        cprint(
            "[DAgger] teacher loaded. Base policy frozen; "
            "only adapt_tconv will be trained.",
            "green",
            attrs=["bold"],
        )

        if bool(self.config.sanity_check_on_restore):
            self._sanity_rollouts()

    def restore_test(self, fn):
        if not fn:
            return
        checkpoint = torch.load(fn, weights_only=False)
        self.model.load_state_dict(checkpoint["model"], strict=False)
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if "sa_mean_std" in checkpoint:
            self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
        if "priv_mean_std" in checkpoint and self.normalize_priv:
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if self.normalize_point_cloud and "point_cloud_mean_std" in checkpoint:
            self.point_cloud_mean_std.load_state_dict(checkpoint["point_cloud_mean_std"])

    def save(self, name):
        weights = {
            "model": self.model.state_dict(),
            "running_mean_std": self.running_mean_std.state_dict(),
            "sa_mean_std": self.sa_mean_std.state_dict(),
            "priv_mean_std": self.priv_mean_std.state_dict(),
            "agent_steps": self.agent_steps,
            "collect_steps": self.collect_steps,
            # 兼容老 ckpt / 外部脚本：best_rewards 即历史上的 "best mixed reward"
            "best_rewards": self.best_mixed_reward,
            "best_mixed_reward": self.best_mixed_reward,
            "best_eval_rewards": self.best_eval_rewards,
            "best_student_eval_reward": self.best_eval_rewards,
            "last_eval_reward": self.last_eval_reward,
            "last_eval_length": self.last_eval_length,
            "beta": float(self._beta),
        }
        if self.normalize_point_cloud:
            weights["point_cloud_mean_std"] = self.point_cloud_mean_std.state_dict()
        torch.save(weights, f"{name}.ckpt")

    # ------------------------------------------------------------
    # Eval / rollout
    # ------------------------------------------------------------

    def _rollout_reward(
        self,
        action_fn,
        label: str,
        num_episodes: int = 16,
        max_steps: int = 0,
    ) -> dict:
        """Run ``action_fn(input_dict) -> action`` against the current env until
        ``num_episodes`` finish (or ``max_steps`` passed, if >0). Does NOT touch
        optimizer / buffer; temporarily sets everything to eval mode so running
        stats are not contaminated. Returns a metrics dict with mean/std reward
        and length.

        Single reset before the rollout, no reset at end — next training step
        will reset if needed. This is a pure sanity/eval probe.
        """
        # snapshot training modes so we can restore
        was_run_training = self.running_mean_std.training
        was_sa_training = self.sa_mean_std.training
        was_priv_training = self.priv_mean_std.training
        was_pc_training = self.point_cloud_mean_std.training
        self.set_eval()

        try:
            obs_dict = self.env.reset()
            step_reward = torch.zeros(
                self.num_actors, dtype=torch.float32, device=self.device
            )
            step_length = torch.zeros(
                self.num_actors, dtype=torch.float32, device=self.device
            )
            episode_rewards, episode_lengths = [], []

            steps = 0
            target = max(1, int(num_episodes))
            hard_cap = int(max_steps) if int(max_steps) > 0 else 10_000
            # Minimum rollout horizon before we're allowed to early-exit on
            # episode count. With 2048 envs + termination.grace_steps=0, a few
            # envs terminate at t=1 from PhysX reset / finger_dist outliers;
            # those dominate the head of episode_rewards and look like length=1
            # reset-garbage. Running a full horizon and aggregating ALL dones
            # gives a representative teacher/student quality estimate.
            min_steps_floor = min(max(200, target * 8), hard_cap)
            while steps < hard_cap:
                input_dict = self._build_input_dict(obs_dict)
                action = action_fn(input_dict)
                # 与训练主循环同样的防御性消毒，避免 eval 时模型随机初始值输出 NaN/Inf
                # 触发 Isaac Gym PhysX articulation 写入告警。
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                action = action.clamp(-1.0, 1.0).contiguous()
                obs_dict, r, done, _ = self.env.step(action)
                step_reward += r
                step_length += 1
                done_indices = done.nonzero(as_tuple=False)
                if done_indices.numel() > 0:
                    episode_rewards.extend(
                        step_reward[done_indices].detach().cpu().view(-1).tolist()
                    )
                    episode_lengths.extend(
                        step_length[done_indices].detach().cpu().view(-1).tolist()
                    )
                not_dones = 1.0 - done.float()
                step_reward = step_reward * not_dones
                step_length = step_length * not_dones
                steps += 1
                if steps >= min_steps_floor and len(episode_rewards) >= target:
                    break
        finally:
            # restore training modes
            if was_run_training:
                self.running_mean_std.train()
            if was_sa_training:
                self.sa_mean_std.train()
            if was_priv_training:
                self.priv_mean_std.train()
            if was_pc_training:
                self.point_cloud_mean_std.train()

        if len(episode_rewards) == 0:
            cprint(
                f"[DAgger][{label}] 0 episodes collected in {steps} env steps — "
                "increase max_steps or check env done logic.",
                "yellow",
            )
            return {"reward_mean": float("nan"), "reward_std": float("nan"),
                    "length_mean": float("nan"), "length_std": float("nan"),
                    "n_episodes": 0, "env_steps": steps,
                    "final_obs_dict": obs_dict}

        rewards = np.asarray(episode_rewards, dtype=np.float32)
        lengths = np.asarray(episode_lengths, dtype=np.float32)
        out = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "length_mean": float(lengths.mean()),
            "length_std": float(lengths.std()),
            "n_episodes": int(len(rewards)),
            "env_steps": int(steps),
            # 让调用方（train 主循环）能直接接着用这份 obs_dict 而不必再次 env.reset()，
            # 后者会再触发 PhysX articulation 的非致命告警。
            "final_obs_dict": obs_dict,
        }
        cprint(
            f"[DAgger][{label}] episodes={out['n_episodes']} "
            f"(env_steps={out['env_steps']}) | "
            f"reward={out['reward_mean']:.2f}±{out['reward_std']:.2f} | "
            f"len={out['length_mean']:.1f}±{out['length_std']:.1f}",
            "cyan",
            attrs=["bold"],
        )
        return out

    def _teacher_only_action(self, input_dict):
        teacher_action, _ = self._get_teacher_labels(input_dict)
        return teacher_action

    def _student_only_action(self, input_dict):
        student_action, _ = self._get_student_action(input_dict)
        return student_action

    def _sanity_rollouts(self):
        """Teacher-only and student-only sanity rollouts, called once after
        restore_train(). Confirms teacher ckpt is wired up correctly; gives a
        baseline number for pre-adaptation student performance."""
        n_ep = int(self.config.sanity_num_episodes)
        max_s = int(self.config.sanity_max_steps)
        cprint(
            "[DAgger] running teacher-only sanity rollout "
            "(expect ≈ teacher PPO reward)",
            "magenta", attrs=["bold"],
        )
        teacher_metrics = self._rollout_reward(
            self._teacher_only_action, "SANITY/teacher", n_ep, max_s
        )
        cprint(
            "[DAgger] running student-only sanity rollout "
            "(adapt_tconv is random → expect low)",
            "magenta", attrs=["bold"],
        )
        student_metrics = self._rollout_reward(
            self._student_only_action, "SANITY/student_pre", n_ep, max_s
        )
        # log to TB at step 0 so teacher sanity is visible even if training is
        # short; also expose via direct_info so it appears in training logs.
        self.writer.add_scalar(
            "sanity/teacher_reward", teacher_metrics["reward_mean"], 0
        )
        self.writer.add_scalar(
            "sanity/student_pre_reward", student_metrics["reward_mean"], 0
        )
        self.writer.add_scalar(
            "sanity/teacher_length", teacher_metrics["length_mean"], 0
        )
        self.writer.add_scalar(
            "sanity/student_pre_length", student_metrics["length_mean"], 0
        )
        self._sanity_teacher_reward = teacher_metrics["reward_mean"]

    def _periodic_student_eval(self):
        eval_metrics = self._rollout_reward(
            self._student_only_action,
            "EVAL/student",
            int(self.config.eval_num_episodes),
            int(self.config.eval_max_steps),
        )
        mean_r = eval_metrics["reward_mean"]
        mean_len = eval_metrics["length_mean"]
        if np.isfinite(mean_r):
            # 记录最近一次 pure-student eval 数据（供进度条 / TB 比较曲线使用）
            self.last_eval_reward = float(mean_r)
            self.last_eval_length = float(mean_len) if np.isfinite(mean_len) else float("nan")

            self.writer.add_scalar(
                "eval/student_reward", mean_r, self.agent_steps
            )
            self.writer.add_scalar(
                "eval/student_length",
                mean_len,
                self.agent_steps,
            )
            if mean_r > self.best_eval_rewards:
                self.best_eval_rewards = mean_r
                self.save(os.path.join(self.nn_dir, "model_best"))
                self.save(os.path.join(self.nn_dir, "model_best_student_eval"))
                cprint(
                    f"[DAgger] new best student eval reward "
                    f"= {mean_r:.2f} (saved model_best)",
                    "green", attrs=["bold"],
                )
            # 立即把最新的 best_student_eval_reward 写入 TB（不用等下一次 train step）
            self.writer.add_scalar(
                "best/student_eval_reward",
                float(self.best_eval_rewards),
                self.agent_steps,
            )
        return eval_metrics

    def test(self):
        self.set_eval()
        # TEST = pure-student rollout (no teacher, no β-mixing). 这与
        # _periodic_student_eval 的口径一致，也是 model_best 的保存口径。
        eval_metrics = self._rollout_reward(
            self._student_only_action,
            "TEST/student",
            int(getattr(self.config, "test_num_episodes", 20) or 20),
            int(getattr(self.config, "test_max_steps", 0) or 0),
        )
        if eval_metrics["n_episodes"] == 0:
            cprint(
                "[DAgger][TEST] 未收集到 episode（请检查环境 reset/done 逻辑）。",
                "yellow",
                attrs=["bold"],
            )
            return eval_metrics
        cprint(
            f"[DAgger][TEST] pure-student reward="
            f"{eval_metrics['reward_mean']:.2f}±{eval_metrics['reward_std']:.2f} "
            f"(comparable to DOTPG TEST / eval/student_reward)",
            "green",
            attrs=["bold"],
        )
        return eval_metrics

    # ------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------

    def train(self):
        self.set_train()
        _t = time.time()
        _last_t = time.time()

        obs_dict = self.env.reset()
        self.agent_steps += self.batch_size

        max_agent_steps = int(getattr(self.config, "max_agent_steps", int(1e9)) or int(1e9))
        save_interval = int(self.config.save_interval_agent_steps)
        eval_interval = int(getattr(self.config, "eval_interval_agent_steps", 0) or 0)

        while self.agent_steps <= max_agent_steps:
            # 1) student-visited state -> inputs
            input_dict = self._build_input_dict(obs_dict)

            # 2) teacher labels on student-visited state
            teacher_action, teacher_extrin = self._get_teacher_labels(input_dict)

            # 3) student action (no-grad，用于 rollout)
            with torch.no_grad():
                student_action, _ = self._get_student_action(input_dict)

            # 4) aggregate (student-visited state + teacher label) -> DAgger dataset
            self._aggregate(obs_dict, teacher_action, teacher_extrin)

            # 5) mix action (DAgger β-mixing)
            env_action = self._mix_action(student_action, teacher_action)
            # 防御性消毒：clamp 到 [-1,1] 范围并替换掉 NaN/Inf 再转成连续内存，
            # 避免 Isaac Gym PhysX 在 non-finite 动作张量上写 articulation 导致
            # 非致命 applyArticulationData 告警。与 DOTPG 的 (clamp + contiguous)
            # 模式保持一致。
            env_action = torch.nan_to_num(env_action, nan=0.0, posinf=1.0, neginf=-1.0)
            env_action = env_action.clamp(-1.0, 1.0).contiguous()

            # 6) supervised updates on aggregated dataset
            do_update = self.collect_steps >= int(self.config.warmup_collect_steps)
            last_metrics = None
            if do_update:
                updates = max(1, int(self.config.updates_per_collect))
                for _ in range(updates):
                    m = self._supervised_update()
                    if m is not None:
                        last_metrics = m
            # 清掉上一轮残留的 loss 值，避免当 _supervised_update 返回 None 时 TB
            # 里依然挂着老数字误导人。
            for k in ("latent_loss", "bc_loss", "adapt_loss"):
                self.direct_info.pop(k, None)
            if last_metrics is not None:
                self.direct_info.update(last_metrics)

            # 7) step env
            next_obs_dict, r, done, info = self.env.step(env_action)

            # screw metrics -> TB (short names, matching DOTPG convention)
            if isinstance(info, dict):
                for src, dst in (
                    ("screw/angular_velocity", "angular_velocity"),
                    ("screw/angular_position", "angular_position"),
                    ("screw/positive_vel_ratio", "positive_vel_ratio"),
                ):
                    v = info.get(src, None)
                    if v is None:
                        continue
                    if isinstance(v, torch.Tensor):
                        if v.dim() != 0:
                            continue
                        v = float(v.detach().cpu())
                    else:
                        try:
                            v = float(v)
                        except (TypeError, ValueError):
                            continue
                    self.direct_info[dst] = v

            # 8) stats (mixed-policy rollout — includes teacher β-share, NOT a
            #    clean measure of student policy quality)
            self.step_reward += r
            self.step_length += 1
            done_indices = done.nonzero(as_tuple=False)
            self.mean_eps_reward.update(self.step_reward[done_indices])
            self.mean_eps_length.update(self.step_length[done_indices])
            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            # 9) advance
            obs_dict = next_obs_dict
            self.agent_steps += self.batch_size
            self.collect_steps += 1
            self._decay_beta()

            # 10) log / save
            self._log_training_info(info if isinstance(info, dict) else {})

            if save_interval > 0 and self.agent_steps % save_interval == 0:
                self.save(
                    os.path.join(
                        self.nn_dir, f"{self.agent_steps // int(1e8)}00m"
                    )
                )
                self.save(os.path.join(self.nn_dir, "model_last"))

            # mixed-policy "best" is just a training-side diagnostic —
            # 它包含 β-teacher 成分，并不等于纯 student 水平（后者以
            # best_eval_rewards 衡量，见 _periodic_student_eval）。
            mean_rewards = self.mean_eps_reward.get_mean()
            if mean_rewards > self.best_mixed_reward:
                self.best_mixed_reward = mean_rewards
                self.save(os.path.join(self.nn_dir, "model_best_mixed"))

            # 真正的学生水平：periodic pure-student eval -> model_best
            if (
                eval_interval > 0
                and self.agent_steps - self.last_eval_agent_steps >= eval_interval
            ):
                eval_metrics = self._periodic_student_eval()
                self.last_eval_agent_steps = self.agent_steps
                # eval 内部已经 env.reset() 过，并且跑完多步后 env 处于有效状态；
                # 直接接着用 _rollout_reward 返回的 final_obs_dict，避免再次 env.reset()
                # 触发 PhysX articulation 非致命告警。
                next_obs = eval_metrics.get("final_obs_dict", None) if isinstance(eval_metrics, dict) else None
                if next_obs is not None:
                    obs_dict = next_obs
                else:
                    obs_dict = self.env.reset()
                self.step_reward.zero_()
                self.step_length.zero_()
                # re-enter training mode (eval path put us in set_eval)
                self.set_train()

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()

            # Terminal: 与 DOTPG / BC 对齐的统一进度行。DAgger 的 Train Return 含
            # teacher β-mixing 成分（另见 W&B dagger/mixed_reward），真正的学生水平
            # 看 Best Student Eval / Last Student Eval（由 _periodic_student_eval 更新）。
            if np.isfinite(self.last_eval_reward):
                last_eval_str = f"{self.last_eval_reward:.2f}"
            else:
                last_eval_str = "n/a"
            if self.best_eval_rewards > -1e4:
                best_eval_str = f"{self.best_eval_rewards:.2f}"
            else:
                best_eval_str = "n/a"
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | "
                f"FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | "
                f"Train Return: {float(self.mean_eps_reward.get_mean()):.2f} | "
                f"Best Student Eval: {best_eval_str} | "
                f"Last Student Eval: {last_eval_str}"
            )
            tprint(info_string)
