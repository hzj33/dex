"""
Pure Behavior Cloning (BC) student learner - dexscrew 集成版本

定位
----
- DOTPG 的公平对比基线：纯监督学习，不做 DAgger 聚合，不做 BCO 逆动力学，不做 OT / Q / RL。
- 只替换 "Adaptation Module Training" 阶段（与 ProprioAdapt / DAgger / DOTPGStudent 等位）。
- 与 DAgger 同样：base policy 冻结，只训练 adapt_tconv（+ 可选 sa_mean_std 更新）。

流程
----
    阶段 1（一次性收集）：
        载入 teacher PPO ckpt → 在 teacher-only rollout 下收集固定数量样本：
            (obs, proprio_hist, priv_info?, point_cloud?, teacher_action, teacher_extrin)
        写入离线 BC buffer，持久化到磁盘（方便断点重启 / 复用同一份数据对比不同 BC 超参）。

    阶段 2（纯监督训练）：
        for step in range(max_agent_steps):
            sample batch
            z_hat   = tanh(adapt_tconv(proprio_hist))
            a_pred  = pi_base(obs, z_hat)
            bc_loss = || clamp(a_pred) - clamp(teacher_action) ||_2^2
            [可选] latent_loss = || z_hat - teacher_extrin ||_2^2
            total_loss = action_bc_coef * bc_loss + latent_coef * latent_loss
            Adam step on adapt_tconv
        周期性做 pure-student eval（与 DAgger 相同口径，决定 model_best）。

关键设计
--------
- 默认基线：action_bc_coef = 1.0, latent_coef = 1.0  → 动作 MSE + latent MSE 双监督。
  （早期版本 latent_coef=0 为纯动作 BC，发现 adapt_tconv 梯度要穿冻结 actor_mlp，信号被稀释 +
  被 mu clamp 吃掉，收敛极慢、reward 上不去；默认改成与 DAgger / DOTPGStudent 同口径。）
- latent_coef = 0 等价于原始 pure-action BC，作为 optional ablation。
- agent_steps 语义：与 DOTPG / DAgger 保持一致（+= num_actors per train iter），
  以便 TB 曲线横轴可以直接对齐比较。
- 终端输出：仿 DAgger 口径，同时展示 BC 监督 loss 和 pure-student eval。
- 日志键名：严格对齐 DOTPG / DAgger（avg_episode_return / task/* / imitation/total_loss /
  eval/student_reward / eval/student_length / best/student_eval_reward）。
"""

import os
import time
import hashlib
from typing import Optional

import numpy as np
import torch
from termcolor import cprint
from tensorboardX import SummaryWriter

from dexscrew.algo.eval_select import EvalSelectMixin
from dexscrew.algo.models.models import ActorCritic
from dexscrew.algo.models.running_mean_std import RunningMeanStd
from dexscrew.algo.student.bc_buffer import BCBuffer
from dexscrew.utils.misc import AverageScalarMeter, tprint


class BCConfig:
    """BC 训练超参数（可由 train.bc 覆盖）。"""

    def __init__(self, config_dict: Optional[dict] = None):
        # ----- offline demo dataset -----
        # demo_collect_steps * add_num_envs ≈ 总样本数；实际落盘样本数 ≤ buffer_size
        self.buffer_size = int(5e5)
        self.buffer_device = "cpu"           # cpu / cuda
        self.buffer_dtype = "float32"        # float16 / float32（默认 fp32 避开 BC label 量化）
        self.demo_collect_steps = 4000       # teacher rollout step 数（env step）
        self.add_num_envs = 0                # 每 env step 向 buffer 追加的 env 数；0 / >=num_envs 表示全部
        self.store_priv_info = False         # 纯 BC 默认不存；teacher_extrin 已入库
        self.store_point_cloud = False
        # 复用磁盘上的固定演示数据：便于同一份数据跑多个 BC 超参对比
        self.reuse_demo_buffer = True

        # ----- supervised optimization -----
        self.batch_size = 4096
        # BC 只做监督学习，没有 rollout 推进；updates_per_collect 在这里定义成
        # "每轮统计/日志间做多少次 SGD"，保持训练节奏与 DAgger / DOTPG 对齐。
        self.updates_per_collect = 4
        self.lr = 3e-4

        # ----- loss coefficients -----
        # 主基线：纯动作 BC（action_bc_coef=1, latent_coef=0）
        self.action_bc_coef = 1.0
        self.latent_coef = 0.0

        # ----- schedule -----
        # 0/None means "inherit train.ppo.max_agent_steps"; this keeps the
        # baseline controllable by the same high-level knobs as other students.
        self.max_agent_steps = 0
        self.min_buffer_for_update = 1024
        # demo 收集完后可选地先做一轮 BC 预训练（step 数），有助于 Dexh13 / RBHand 这类高维策略
        # 更快达到可评估水平。默认 0 表示直接进入主训练循环。
        self.pretrain_steps = 0

        # ----- logging / save -----
        self.save_interval_agent_steps = int(1e5)

        # ----- periodic pure-student eval -----
        self.eval_interval_agent_steps = int(1e5)
        self.eval_num_episodes = 32
        self.eval_max_steps = 0
        # 启动 restore_train 后是否跑 teacher / student 各自的 sanity rollout
        self.sanity_check_on_restore = True
        self.sanity_num_episodes = 16
        self.sanity_max_steps = 1000

        # During demo collection, teacher labels must use the normalization
        # statistics loaded from the PPO checkpoint.  Only the student
        # proprio-history normalizer should adapt to the collected demo
        # distribution.
        self.freeze_teacher_stats_during_demo = True

        # ----- strict ckpt audit -----
        # 缺失的 base-policy 权重（actor_mlp / mu / env_mlp / point_mlp / sigma）
        # 会导致 silent-fail：默认严格校验，缺就 RuntimeError。
        self.strict_base_policy = True

        # ----- test -----
        self.test_num_episodes = 20
        self.test_max_steps = 0
        self.test_num_steps = 0

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


class BCStudent(EvalSelectMixin):
    """Pure offline BC student adapter.

    与 DAggerStudent 同构：
    - ActorCritic 作为 base policy（加载 teacher ckpt 后冻结所有子模块）
    - 仅 adapt_tconv 参与训练；sa_mean_std 维持 eval（用 teacher demo 的归一化统计）
    - 固定演示 dataset 只在训练开始时一次性采集
    - SummaryWriter 日志 / best / last / periodic ckpt / tprint 进度条

    不同点：
    - 没有 β-mixing / 在线聚合：训练阶段完全不 env.step()（除 periodic eval）
    - 终端的 "Best Mixed" 概念不存在；改为 "Avg BC Loss" 直接展示监督 loss
    """

    def __init__(self, env, output_dir, full_config, student_dim: int = 24):
        self.device = full_config["rl_device"]
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo

        # ---- BC 配置 ----
        bc_config_dict = _as_plain_dict(full_config.train.get("bc", {}))
        self.config = BCConfig(bc_config_dict)

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
                "BCStudent requires train.ppo.proprio_adapt=True (student uses proprio_hist -> extrin)."
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

        # ---- Running mean / std (与 padapt / DAgger 保持一致) ----
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)
        self.running_mean_std.eval()
        self.sa_mean_std = RunningMeanStd(
            (self.proprio_hist_dim, self.proprio_dim)
        ).to(self.device)
        # 纯 BC 下 sa_mean_std 统计量要覆盖 teacher demo 分布；在 demo 采集阶段 train()，
        # 之后 eval 即可（训练本身只采样 buffer，不会再触碰新数据）。
        self.sa_mean_std.train()
        self.priv_mean_std = RunningMeanStd(self.priv_info_dim).to(self.device)
        self.priv_mean_std.eval()
        self.point_cloud_mean_std = RunningMeanStd(3,).to(self.device)
        self.point_cloud_mean_std.eval()

        # ---- Output ----
        self.output_dir = output_dir
        self.nn_dir = os.path.join(self.output_dir, "bc_nn")
        self.tb_dir = os.path.join(self.output_dir, "bc_tb")
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
                "BCStudent: no adapt_tconv parameters found on ActorCritic. "
                "Check that train.ppo.proprio_adapt=True."
            )
        self.optim = torch.optim.Adam(adapt_params, lr=float(self.config.lr))

        # ---- Stats ----
        self.batch_size = self.num_actors
        # pure-student eval 的平均 reward / length（决定 model_best）
        self.best_eval_rewards = -10000.0
        self.last_eval_reward = float("nan")
        self.last_eval_length = float("nan")
        self.last_eval_agent_steps = 0
        self.last_save_agent_steps = 0
        self.agent_steps = 0
        self.collect_steps = 0
        self.direct_info = {}
        # 用于终端显示的 BC 监督 loss 平均窗口（与 DAgger AverageScalarMeter 口径一致）
        self.loss_meter = AverageScalarMeter(window_size=2000)
        # demo 中 env rollout 对应的 episode stat（只在 demo 采集阶段更新）
        self.mean_eps_reward = AverageScalarMeter(window_size=20000)
        self.mean_eps_length = AverageScalarMeter(window_size=20000)
        self.step_reward = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self.step_length = torch.zeros(
            self.num_actors, dtype=torch.float32, device=self.device
        )
        self._init_eval_select(role="student", artifact_ext="ckpt")

        # ---- BC buffer ----
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

        self.dataset = BCBuffer(
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
        # demo 数据持久化路径（固定与 output_dir 一起走，便于复用与 ablation）
        self.demo_buffer_path = os.path.join(self.output_dir, "bc_demo_buffer.pt")

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
        """构建 normalized input dict（student + teacher 共用）。"""
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
    # Supervised update on offline BC dataset
    # ------------------------------------------------------------

    def _supervised_update(self):
        """Sample a batch from BC demo dataset and update adapt_tconv.

        Default (pure BC):
            a_pred  = pi_base(obs_norm, tanh(adapt_tconv(proprio_hist_norm)))
            bc_loss = || clamp(a_pred) - clamp(teacher_action) ||^2
            total   = action_bc_coef * bc_loss

        Optional latent auxiliary (when latent_coef > 0):
            latent_loss = || z_hat - teacher_extrin ||^2
            total       = action_bc_coef * bc_loss + latent_coef * latent_loss
        """
        if len(self.dataset) < int(self.config.min_buffer_for_update):
            return None

        batch = self.dataset.sample(int(self.config.batch_size))

        # demo 采样出的 raw obs / proprio_hist 过一遍固定归一化（不再更新 running stats）
        was_training_obs = self.running_mean_std.training
        was_training_sa = self.sa_mean_std.training
        self.running_mean_std.eval()
        self.sa_mean_std.eval()

        obs_norm = self.running_mean_std(batch["obs"])
        proprio_norm = self.sa_mean_std(batch["proprio_hist"])

        # --- student extrin (可导，供 adapt_tconv 接收梯度) ---
        student_extrin = torch.tanh(self.model.adapt_tconv(proprio_norm))

        # --- 主 loss：action BC ---
        # 注意：故意不对 student_action 做 pre-MSE clamp —— clamp 会把 |mu|>1 那一维的反向
        # 梯度置零，灵巧手任务里 teacher 动作常贴 ±1，会导致 adapt_tconv 在饱和维完全学不到
        # 信号。teacher_action 在 `_teacher_action_from_extrin` 入库时已 clamp 过，这里的
        # clamp 只是一道防御：即便 buffer 里的 teacher_action 是极端值，MSE 目标也固定在
        # [-1,1]，不会把 student 推到边界外。
        student_action = self._student_action_from_extrin(obs_norm, student_extrin)
        teacher_action = torch.clamp(batch["teacher_action"].detach(), -1.0, 1.0)
        # Match ProprioAdapt's action-BC scale: sum over action dimensions per
        # sample, then average over the batch.  Plain F.mse_loss(..., mean)
        # underweights action supervision by roughly actions_num and lets the
        # latent auxiliary dominate too easily on Dexh13.
        bc_loss = (student_action - teacher_action).pow(2).sum(dim=-1).mean()

        # --- 可选辅助：latent supervision ---
        latent_coef = float(self.config.latent_coef)
        action_coef = float(self.config.action_bc_coef)
        if latent_coef > 0.0:
            teacher_extrin_label = batch["teacher_extrin"].detach()
            latent_loss = (student_extrin - teacher_extrin_label).pow(2).mean()
        else:
            latent_loss = torch.zeros((), device=bc_loss.device, dtype=bc_loss.dtype)

        total_loss = action_coef * bc_loss + latent_coef * latent_loss

        self.optim.zero_grad()
        total_loss.backward()
        self.optim.step()

        if was_training_obs:
            self.running_mean_std.train()
        if was_training_sa:
            self.sa_mean_std.train()

        return {
            "bc_loss": float(bc_loss.detach().cpu()),
            "latent_loss": float(latent_loss.detach().cpu()),
            # 与 DAgger / DOTPG 的键名完全一致：imitation/total_loss 的 TB 写入由此驱动
            "adapt_loss": float(total_loss.detach().cpu()),
        }

    # ------------------------------------------------------------
    # Offline demo collection (teacher-only rollout)
    # ------------------------------------------------------------

    def _update_task_metrics_from_info(self, info):
        """从 env.step() info 中吸收 screw/* 指标，写入 direct_info 缓存。

        DAgger 在 train rollout 里每 iter 都会调用 env.step() → 每 iter 都刷新 task/*；
        BC 训练主循环不 env.step，所以 task/* 的刷新只能来自：
            (1) demo collection 阶段（teacher rollout）
            (2) periodic student eval 阶段
        这里把两处的入口统一到同一个 helper，保证 direct_info 的 task/* 始终是"最近一次
        观察到的 env 指标"，_log_training_info 再把它写到 TB/W&B。
        """
        if not isinstance(info, dict):
            return
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

    def _collect_demos(self, num_steps: int):
        """用 teacher 策略跑 rollout，把 (obs, proprio_hist, teacher_action, teacher_extrin)
        存到 BC buffer。纯离线数据集，训练期间不再追加。

        语义与 DOTPG.collect_expert_data 类似，但：
        - 不需要 state=[obs, extrin] 向量化（BCStudent 从 buffer 采样时会重新通过 adapt_tconv
          + actor_mlp 做 forward）。
        - 默认冻结 teacher 侧 obs/priv/point normalizer，只更新 student proprio-history normalizer。
          teacher action/extrin 标签必须和 PPO checkpoint 的归一化统计保持一致。
        - 顺手把 screw/* 指标吸收到 direct_info，作为训练前期 task/* 的种子值。
        """
        num_steps = int(num_steps)
        if num_steps <= 0:
            return

        add_n = int(self.config.add_num_envs)
        if add_n <= 0 or add_n >= self.num_actors:
            add_n = self.num_actors

        cprint(
            f"[BC] collecting teacher demonstrations: {num_steps} env steps × "
            f"{add_n}/{self.num_actors} envs/step = up to {num_steps * add_n} samples",
            "green",
            attrs=["bold"],
        )

        # Buffer stores raw tensors.  Teacher labels must be computed with
        # the PPO checkpoint's obs/priv/point normalizers; mutating those
        # stats during demo collection shifts the teacher policy itself and
        # produces stale/misaligned labels.  The only normalizer we fit here
        # is the student proprio-history normalizer.
        if bool(self.config.freeze_teacher_stats_during_demo):
            self.running_mean_std.eval()
            self.priv_mean_std.eval()
            if self.normalize_point_cloud:
                self.point_cloud_mean_std.eval()
        else:
            self.running_mean_std.train()
            if self.normalize_priv:
                self.priv_mean_std.train()
            if self.normalize_point_cloud:
                self.point_cloud_mean_std.train()
        self.sa_mean_std.train()

        obs_dict = self.env.reset()

        # 重置每 env 的 episode stat（demo 采集期间顺便汇报 teacher 能拿多少分）
        self.step_reward.zero_()
        self.step_length.zero_()

        last_info = None

        for step in range(num_steps):
            input_dict = self._build_input_dict(obs_dict)
            teacher_action, teacher_extrin = self._get_teacher_labels(input_dict)

            # --- 取样本 ---
            if add_n == self.num_actors:
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

            # --- 推进环境（teacher-only） ---
            env_action = torch.nan_to_num(
                teacher_action, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0).contiguous()
            obs_dict, r, done, info = self.env.step(env_action)
            last_info = info

            # 与 DAgger/_DOTPG 同口径：把 screw/* 吸收到 direct_info；
            # 主训练循环在 eval 之前也能写出非空的 task/*。
            self._update_task_metrics_from_info(info)

            self.step_reward += r
            self.step_length += 1
            done_idx = done.nonzero(as_tuple=False)
            if done_idx.numel() > 0:
                self.mean_eps_reward.update(self.step_reward[done_idx])
                self.mean_eps_length.update(self.step_length[done_idx])
            not_dones = 1.0 - done.float()
            self.step_reward = self.step_reward * not_dones
            self.step_length = self.step_length * not_dones

            if (step + 1) % 500 == 0:
                tprint(
                    f"[BC] demo collect {step + 1}/{num_steps} | "
                    f"buffer={len(self.dataset)} | "
                    f"teacher_reward(avg)={self.mean_eps_reward.get_mean():.2f}"
                )

        # demo 采集完，后续全程用固定 running stats
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

        # 记录 teacher demo 表现，作为 BC 学生的参考上界
        self._last_demo_teacher_reward = float(self.mean_eps_reward.get_mean())
        self._last_demo_teacher_length = float(self.mean_eps_length.get_mean())

        cprint(
            f"[BC] demo collection done. buffer_size={len(self.dataset)} | "
            f"teacher_reward(avg)={self._last_demo_teacher_reward:.2f}",
            "green",
            attrs=["bold"],
        )

    def _log_demo_summary(self):
        """在主训练循环启动前，把 demo 阶段的 task/* 种子指标以 global_step=0 写到 TB，
        保证 W&B 在等 first-eval 前就能看到起点数据。严格只写 canonical 键名。
        demo 阶段 teacher 表现作为 BC-specific 附属指标记录在 bc/ 命名空间下。
        """
        step0 = 0
        demo_r = getattr(self, "_last_demo_teacher_reward", float("nan"))
        demo_l = getattr(self, "_last_demo_teacher_length", float("nan"))
        if np.isfinite(demo_r):
            self.writer.add_scalar("bc/demo_teacher_reward", float(demo_r), step0)
        if np.isfinite(demo_l):
            self.writer.add_scalar("bc/demo_teacher_length", float(demo_l), step0)
        # 把 demo 阶段吸收到的 task/* 种子值写到 step=0，后续训练主循环会继续覆盖
        for src, dst in (
            ("angular_velocity", "task/angular_velocity"),
            ("angular_position", "task/angular_position"),
            ("positive_vel_ratio", "task/positive_screw_ratio"),
        ):
            v = self.direct_info.get(src, None)
            if v is None:
                continue
            self.writer.add_scalar(dst, float(v), step0)
        self.writer.add_scalar(
            "bc/buffer_size", float(len(self.dataset)), step0
        )
        self.writer.add_scalar("global_step", float(step0), step0)
        self.writer.flush()

    def _save_demo_buffer(self):
        try:
            meta = {
                "bc_demo_format_version": 2,
                "num_envs_at_collect": int(self.num_actors),
                "demo_collect_steps": int(self.config.demo_collect_steps),
                "add_num_envs": int(self.config.add_num_envs),
                "buffer_size": int(self.config.buffer_size),
                "buffer_dtype": str(self.config.buffer_dtype),
                "freeze_teacher_stats_during_demo": bool(
                    self.config.freeze_teacher_stats_during_demo
                ),
                "saved_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                # 绑定 demo 所用的 teacher ckpt 指纹，reuse 时校验 (_try_load_demo_buffer)
                "teacher_ckpt_hash": self._teacher_ckpt_hash(),
                # 采集阶段 in-place fit 的 sa_mean_std；stage1 teacher ckpt 里可能没有这条，
                # 持久化下来可以让下次 reuse 运行跳过 demo 采集也能正确归一化 proprio_hist
                "sa_mean_std_state": {
                    k: v.detach().cpu() for k, v in self.sa_mean_std.state_dict().items()
                },
                # running_mean_std is kept for reproducibility.  In the default
                # v2 path it should remain identical to the teacher checkpoint.
                "running_mean_std_state": {
                    k: v.detach().cpu() for k, v in self.running_mean_std.state_dict().items()
                },
                "priv_mean_std_state": {
                    k: v.detach().cpu() for k, v in self.priv_mean_std.state_dict().items()
                },
            }
            if self.normalize_point_cloud:
                meta["point_cloud_mean_std_state"] = {
                    k: v.detach().cpu()
                    for k, v in self.point_cloud_mean_std.state_dict().items()
                }
            self.dataset.save(self.demo_buffer_path, meta=meta)
            cprint(
                f"[BC] demo buffer saved to {self.demo_buffer_path}",
                "green",
            )
        except Exception as e:
            cprint(f"[BC] demo buffer save failed (will recollect next time): {e}", "yellow")

    def _try_load_demo_buffer(self) -> bool:
        if not bool(self.config.reuse_demo_buffer):
            return False
        if not os.path.isfile(self.demo_buffer_path):
            return False
        try:
            meta = self.dataset.load(self.demo_buffer_path)
        except Exception as e:
            cprint(f"[BC] failed to load demo buffer, will recollect: {e}", "yellow")
            return False

        # --- teacher ckpt 指纹校验：不一致就丢弃旧 buffer 强制重采 ---
        current_hash = self._teacher_ckpt_hash()
        stored_hash = meta.get("teacher_ckpt_hash", None) if isinstance(meta, dict) else None
        if current_hash and stored_hash and current_hash != stored_hash:
            cprint(
                f"[BC] teacher ckpt hash mismatch — stored={stored_hash[:12]}… vs "
                f"current={current_hash[:12]}…；丢弃旧 buffer 并重新采集。",
                "yellow", attrs=["bold"],
            )
            self.dataset.clear()
            return False
        if current_hash and not stored_hash:
            cprint(
                "[BC] demo buffer 缺 teacher_ckpt_hash (旧版本格式)；不强制重采，但请留意"
                " teacher/buffer 是否自洽。",
                "yellow",
            )

        expected_freeze = bool(self.config.freeze_teacher_stats_during_demo)
        stored_freeze = (
            meta.get("freeze_teacher_stats_during_demo", None)
            if isinstance(meta, dict) else None
        )
        stored_version = int(meta.get("bc_demo_format_version", 0)) if isinstance(meta, dict) else 0
        if expected_freeze and (stored_version < 2 or stored_freeze is not True):
            cprint(
                "[BC] demo buffer was collected with legacy/stat-updating mode; "
                "discarding it so teacher-normalization labels are regenerated.",
                "yellow",
                attrs=["bold"],
            )
            self.dataset.clear()
            return False

        # --- 恢复 sa_mean_std / running_mean_std（stage1 teacher ckpt 可能没带 sa_mean_std） ---
        sa_state = meta.get("sa_mean_std_state", None) if isinstance(meta, dict) else None
        if isinstance(sa_state, dict) and sa_state:
            try:
                self.sa_mean_std.load_state_dict(
                    {k: v.to(self.device) for k, v in sa_state.items()}
                )
                cprint("[BC] restored sa_mean_std from demo buffer meta.", "grey")
            except Exception as e:
                cprint(
                    f"[BC] sa_mean_std restore failed ({e}); 将保持 teacher ckpt / init 值",
                    "yellow",
                )
        run_state = meta.get("running_mean_std_state", None) if isinstance(meta, dict) else None
        if isinstance(run_state, dict) and run_state:
            try:
                self.running_mean_std.load_state_dict(
                    {k: v.to(self.device) for k, v in run_state.items()}
                )
                cprint("[BC] restored running_mean_std from demo buffer meta.", "grey")
            except Exception as e:
                cprint(
                    f"[BC] running_mean_std restore failed ({e}); 将保持 teacher ckpt 值",
                    "yellow",
                )
        priv_state = meta.get("priv_mean_std_state", None) if isinstance(meta, dict) else None
        if isinstance(priv_state, dict) and priv_state:
            try:
                self.priv_mean_std.load_state_dict(
                    {k: v.to(self.device) for k, v in priv_state.items()}
                )
                cprint("[BC] restored priv_mean_std from demo buffer meta.", "grey")
            except Exception as e:
                cprint(
                    f"[BC] priv_mean_std restore failed ({e}); 将保持 teacher ckpt 值",
                    "yellow",
                )
        pc_state = meta.get("point_cloud_mean_std_state", None) if isinstance(meta, dict) else None
        if self.normalize_point_cloud and isinstance(pc_state, dict) and pc_state:
            try:
                self.point_cloud_mean_std.load_state_dict(
                    {k: v.to(self.device) for k, v in pc_state.items()}
                )
                cprint("[BC] restored point_cloud_mean_std from demo buffer meta.", "grey")
            except Exception as e:
                cprint(
                    f"[BC] point_cloud_mean_std restore failed ({e}); 将保持 teacher ckpt 值",
                    "yellow",
                )

        cprint(
            f"[BC] loaded {len(self.dataset)} demo samples from "
            f"{self.demo_buffer_path} (meta={ {k: v for k, v in meta.items() if k not in ('sa_mean_std_state', 'running_mean_std_state')} })",
            "green",
        )
        return len(self.dataset) > 0

    def _teacher_ckpt_hash(self) -> str:
        """md5 of the teacher ckpt file bytes — 用来在 reuse_demo_buffer=True 时
        防止"旧 buffer 配新 teacher"这种静默不自洽。无 teacher 路径时返回 ''。
        """
        cached = getattr(self, "_teacher_ckpt_hash_cached", None)
        if cached is not None:
            return cached
        path = getattr(self, "_teacher_ckpt_path", None)
        if not path or not os.path.isfile(path):
            self._teacher_ckpt_hash_cached = ""
            return ""
        try:
            h = hashlib.md5()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            self._teacher_ckpt_hash_cached = h.hexdigest()
        except Exception as e:
            cprint(f"[BC] teacher ckpt hash failed ({e}); 不校验 buffer 指纹。", "yellow")
            self._teacher_ckpt_hash_cached = ""
        return self._teacher_ckpt_hash_cached

    # ------------------------------------------------------------
    # Logging / mode
    # ------------------------------------------------------------

    def _log_training_info(self, extra_info):
        """Canonical unified logging (follows DOTPG reference).

        Shared comparison metrics (identical keys across DOTPG / DAgger / BC):
          - global_step                  self.agent_steps
          - train/avg_episode_return     BC has no training rollout; mirror last_eval_reward
                                         when available (best student proxy available to BC).
          - train/avg_episode_length     mirror last_eval_length when available.
          - eval/student_reward          last_eval_reward (pure-student periodic eval)
          - eval/student_length          last_eval_length
          - best/student_eval_reward     self.best_eval_rewards (pure-student best)
          - task/angular_velocity / task/angular_position / task/positive_screw_ratio
          - imitation/total_loss         adapt_loss (action BC + optional latent auxiliary)

        Algorithm-specific diagnostics kept under the bc/ namespace.
        """
        step = int(self.agent_steps)

        shared = {"global_step": float(step)}
        if np.isfinite(self.last_eval_reward):
            shared["train/avg_episode_return"] = float(self.last_eval_reward)
            shared["eval/student_reward"] = float(self.last_eval_reward)
        if np.isfinite(self.last_eval_length):
            shared["train/avg_episode_length"] = float(self.last_eval_length)
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

        # ---- algorithm-specific diagnostics (bc/ namespace) ----
        algo_payload = {}
        buf_val = float(len(self.dataset))
        self.writer.add_scalar("bc/buffer_size", buf_val, step)
        algo_payload["bc/buffer_size"] = buf_val
        for k in ("bc_loss", "latent_loss"):
            v = self.direct_info.get(k, None)
            if v is not None:
                fv = float(v)
                self.writer.add_scalar(f"bc/{k}", fv, step)
                algo_payload[f"bc/{k}"] = fv
        window_loss = self.loss_meter.get_mean()
        if np.isfinite(window_loss):
            wl = float(window_loss)
            self.writer.add_scalar("bc/loss_window", wl, step)
            algo_payload["bc/loss_window"] = wl

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
        # sync_tensorboard 在长 run 下对 bc/* 会滞后/停更（shared keys 一路显式 commit，
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
        # 与 DAgger 相同：base policy 保持 eval，只 adapt_tconv 的 parameter.grad 参与更新。
        # running_mean_std / sa_mean_std 在主训练循环里不会更新（采样 buffer 时临时 eval）。
        self.model.eval()
        self.running_mean_std.eval()
        self.sa_mean_std.eval()
        self.priv_mean_std.eval()
        if self.normalize_point_cloud:
            self.point_cloud_mean_std.eval()

    # ------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------

    def _base_policy_prefixes(self):
        prefixes = ["actor_mlp.", "mu.", "value.", "sigma"]
        if self.priv_info:
            prefixes.append("env_mlp.")
        if self.use_point_cloud_info:
            prefixes.append("point_mlp.")
        return tuple(prefixes)

    def _audit_ckpt_against_model(self, ckpt_state_dict: dict) -> dict:
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
                        (k, tuple(ckpt_state_dict[k].shape), tuple(model_state[k].shape))
                    )
            else:
                if k.startswith("adapt_tconv."):
                    missing_adapt.append(k)
                else:
                    missing_base.append(k)
        unexpected = sorted(ckpt_keys - model_keys)

        cprint(
            f"[BC] ckpt audit | loaded={len(loaded)} | "
            f"missing_base={len(missing_base)} | "
            f"missing_adapt_tconv={len(missing_adapt)} | "
            f"shape_mismatch={len(shape_mismatch)} | "
            f"unexpected={len(unexpected)}",
            "cyan",
            attrs=["bold"],
        )
        if missing_adapt:
            cprint(
                f"  • missing adapt_tconv keys (will be TRAINED from init): {missing_adapt}",
                "yellow",
            )
        if missing_base:
            cprint(f"  • MISSING BASE-POLICY keys: {missing_base}", "red")
        if shape_mismatch:
            for k, src, dst in shape_mismatch:
                cprint(f"  • SHAPE MISMATCH {k}: ckpt {src} vs model {dst}", "red")
        if unexpected:
            cprint(f"  • unexpected (ignored): {unexpected}", "grey")

        if bool(self.config.strict_base_policy) and (missing_base or shape_mismatch):
            raise RuntimeError(
                "[BC] teacher ckpt is missing or incompatible with the base policy. "
                "Required base-policy keys:\n  " + ", ".join(required_base) +
                "\nSet train.bc.strict_base_policy=False to override (not recommended)."
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
        self._teacher_ckpt_path = fn
        self._teacher_ckpt_hash_cached = None
        checkpoint = torch.load(fn, map_location=self.device, weights_only=False)
        cprint(f"[BC] loading teacher checkpoint: {fn}", "yellow", attrs=["bold"])

        ckpt_model = checkpoint.get("model", {})
        model_state = self.model.state_dict()
        compatible = {
            k: v for k, v in ckpt_model.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        self.model.load_state_dict(compatible, strict=False)
        self._audit_ckpt_against_model(ckpt_model)

        # normalization stats from teacher
        if "running_mean_std" not in checkpoint:
            raise RuntimeError(
                "[BC] teacher ckpt has no running_mean_std — obs normalization would be wrong."
            )
        self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
        if self.normalize_priv:
            if "priv_mean_std" not in checkpoint:
                raise RuntimeError(
                    "[BC] normalize_priv=True but ckpt has no priv_mean_std"
                )
            self.priv_mean_std.load_state_dict(checkpoint["priv_mean_std"])
        if self.normalize_point_cloud:
            if "point_cloud_mean_std" not in checkpoint:
                raise RuntimeError(
                    "[BC] normalize_point_cloud=True but ckpt has no point_cloud_mean_std"
                )
            self.point_cloud_mean_std.load_state_dict(
                checkpoint["point_cloud_mean_std"]
            )
        if "sa_mean_std" in checkpoint:
            try:
                self.sa_mean_std.load_state_dict(checkpoint["sa_mean_std"])
                cprint(
                    "[BC] loaded teacher sa_mean_std (demo collection will keep "
                    "updating it during rollout to capture teacher distribution).",
                    "grey",
                )
            except Exception:
                pass

        # freeze base policy
        for name, p in self.model.named_parameters():
            p.requires_grad = ("adapt_tconv" in name)
        self.model.eval()

        cprint(
            "[BC] teacher loaded. Base policy frozen; only adapt_tconv will be trained.",
            "green", attrs=["bold"],
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
            "best_eval_rewards": self.best_eval_rewards,
            "best_student_eval_reward": self.best_eval_rewards,
            "last_eval_reward": self.last_eval_reward,
            "last_eval_length": self.last_eval_length,
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
        """Pure student/teacher rollout probe — 复用 DAgger._rollout_reward 同口径。"""
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
            min_steps_floor = min(max(200, target * 8), hard_cap)

            # 把最后一帧 env-info 带回调用方，便于把 task/* 指标写入 TB
            last_info = None

            while steps < hard_cap:
                input_dict = self._build_input_dict(obs_dict)
                action = action_fn(input_dict)
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                action = action.clamp(-1.0, 1.0).contiguous()
                obs_dict, r, done, info = self.env.step(action)
                last_info = info
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
                f"[BC][{label}] 0 episodes collected in {steps} env steps — "
                "increase max_steps or check env done logic.",
                "yellow",
            )
            return {
                "reward_mean": float("nan"), "reward_std": float("nan"),
                "length_mean": float("nan"), "length_std": float("nan"),
                "n_episodes": 0, "env_steps": steps,
                "final_obs_dict": obs_dict, "last_info": last_info,
            }

        rewards = np.asarray(episode_rewards, dtype=np.float32)
        lengths = np.asarray(episode_lengths, dtype=np.float32)
        out = {
            "reward_mean": float(rewards.mean()),
            "reward_std": float(rewards.std()),
            "length_mean": float(lengths.mean()),
            "length_std": float(lengths.std()),
            "n_episodes": int(len(rewards)),
            "env_steps": int(steps),
            "final_obs_dict": obs_dict,
            "last_info": last_info,
        }
        cprint(
            f"[BC][{label}] episodes={out['n_episodes']} "
            f"(env_steps={out['env_steps']}) | "
            f"reward={out['reward_mean']:.2f}±{out['reward_std']:.2f} | "
            f"len={out['length_mean']:.1f}±{out['length_std']:.1f}",
            "cyan", attrs=["bold"],
        )
        return out

    def _teacher_only_action(self, input_dict):
        teacher_action, _ = self._get_teacher_labels(input_dict)
        return teacher_action

    def _student_only_action(self, input_dict):
        student_action, _ = self._get_student_action(input_dict)
        return student_action

    def _eval_select_action(self, obs_dict):
        input_dict = self._build_input_dict(obs_dict)
        return self._student_only_action(input_dict), {}

    def _sanity_rollouts(self):
        n_ep = int(self.config.sanity_num_episodes)
        max_s = int(self.config.sanity_max_steps)
        cprint(
            "[BC] running teacher-only sanity rollout (expect ≈ teacher PPO reward)",
            "magenta", attrs=["bold"],
        )
        teacher_metrics = self._rollout_reward(
            self._teacher_only_action, "SANITY/teacher", n_ep, max_s
        )
        cprint(
            "[BC] running student-only sanity rollout "
            "(adapt_tconv is random → expect low)",
            "magenta", attrs=["bold"],
        )
        student_metrics = self._rollout_reward(
            self._student_only_action, "SANITY/student_pre", n_ep, max_s
        )
        self.writer.add_scalar("sanity/teacher_reward", teacher_metrics["reward_mean"], 0)
        self.writer.add_scalar("sanity/student_pre_reward", student_metrics["reward_mean"], 0)
        self.writer.add_scalar("sanity/teacher_length", teacher_metrics["length_mean"], 0)
        self.writer.add_scalar("sanity/student_pre_length", student_metrics["length_mean"], 0)

    def _periodic_student_eval(self):
        eval_metrics = self._rollout_reward(
            self._student_only_action,
            "EVAL/student",
            int(self.config.eval_num_episodes),
            int(self.config.eval_max_steps),
        )
        mean_r = eval_metrics["reward_mean"]
        mean_len = eval_metrics["length_mean"]

        # 把 eval 的最后一帧 env info 里的 screw metrics 写到 direct_info（sticky cache）
        # 走同一个 helper —— 和 demo collection / DAgger 行为一致。
        last_info = eval_metrics.get("last_info", None)
        self._update_task_metrics_from_info(last_info)

        if np.isfinite(mean_r):
            self.last_eval_reward = float(mean_r)
            self.last_eval_length = float(mean_len) if np.isfinite(mean_len) else float("nan")

            self.writer.add_scalar("eval/student_reward", mean_r, self.agent_steps)
            self.writer.add_scalar("eval/student_length", mean_len, self.agent_steps)
            # eval 阶段一并写入 avg_episode_return（与 DAgger 同口径）
            self.writer.add_scalar(
                "avg_episode_return", float(mean_r), self.agent_steps
            )

            if mean_r > self.best_eval_rewards:
                self.best_eval_rewards = mean_r
                if not getattr(self, "eval_select_enabled", False):
                    self.save(os.path.join(self.nn_dir, "model_best"))
                self.save(os.path.join(self.nn_dir, "model_best_student_eval"))
                cprint(
                    f"[BC] new best student eval reward = {mean_r:.2f}"
                    + (
                        " (saved model_best_student_eval; model_best is eval-select)"
                        if getattr(self, "eval_select_enabled", False)
                        else " (saved model_best)"
                    ),
                    "green", attrs=["bold"],
                )
            self.writer.add_scalar(
                "best/student_eval_reward",
                float(self.best_eval_rewards),
                self.agent_steps,
            )
        return eval_metrics

    def test(self):
        self.set_eval()
        fixed_steps = int(getattr(self.config, "test_num_steps", 0) or 0)
        if fixed_steps > 0:
            return self._fixed_step_eval(self._student_only_action, "TEST/student", fixed_steps)
        eval_metrics = self._rollout_reward(
            self._student_only_action,
            "TEST/student",
            int(getattr(self.config, "test_num_episodes", 20) or 20),
            int(getattr(self.config, "test_max_steps", 0) or 0),
        )
        if eval_metrics["n_episodes"] == 0:
            cprint(
                "[BC][TEST] 未收集到 episode（请检查环境 reset/done 逻辑）。",
                "yellow", attrs=["bold"],
            )
            return eval_metrics
        cprint(
            f"[BC][TEST] pure-student reward="
            f"{eval_metrics['reward_mean']:.2f}±{eval_metrics['reward_std']:.2f} "
            f"(comparable to DOTPG TEST / eval/student_reward)",
            "green", attrs=["bold"],
        )
        return eval_metrics

    def _fixed_step_eval(self, action_fn, label: str, num_steps: int):
        was_run_training = self.running_mean_std.training
        was_sa_training = self.sa_mean_std.training
        was_priv_training = self.priv_mean_std.training
        was_pc_training = self.point_cloud_mean_std.training
        self.set_eval()

        obs_dict = self.env.reset()
        reward_sum = 0.0
        done_sum = 0.0
        steps = int(num_steps)
        try:
            for _ in range(steps):
                input_dict = self._build_input_dict(obs_dict)
                action = action_fn(input_dict)
                action = torch.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
                action = action.clamp(-1.0, 1.0).contiguous()
                obs_dict, r, done, info = self.env.step(action)
                del info
                reward_sum += float(r.float().mean().detach().cpu())
                done_sum += float(done.float().mean().detach().cpu())
        finally:
            if was_run_training:
                self.running_mean_std.train()
            if was_sa_training:
                self.sa_mean_std.train()
            if was_priv_training:
                self.priv_mean_std.train()
            if was_pc_training:
                self.point_cloud_mean_std.train()

        avg_reward = reward_sum / float(max(steps, 1))
        avg_done_rate = done_sum / float(max(steps, 1))
        cprint(
            f"[BC][{label}] EvalSummary steps={steps} "
            f"avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}",
            "green",
            attrs=["bold"],
        )
        print(
            "EvalSummary "
            f"steps={steps} avg_reward={avg_reward:.6f} avg_done_rate={avg_done_rate:.6f}"
        )
        return {
            "steps": steps,
            "avg_reward": avg_reward,
            "avg_done_rate": avg_done_rate,
            "final_obs_dict": obs_dict,
        }

    # ------------------------------------------------------------
    # Main training loop (offline supervised on fixed demo buffer)
    # ------------------------------------------------------------

    def train(self):
        _t = time.time()
        _last_t = time.time()

        # ---- stage 1: demo collection (offline, once) ----
        loaded = self._try_load_demo_buffer()
        if not loaded or len(self.dataset) < int(self.config.min_buffer_for_update):
            self._collect_demos(int(self.config.demo_collect_steps))
            self._save_demo_buffer()
        else:
            cprint(
                f"[BC] reusing existing demo buffer "
                f"(size={len(self.dataset)} ≥ min_buffer_for_update="
                f"{int(self.config.min_buffer_for_update)}); skip collection.",
                "green",
            )
            # 复用磁盘 buffer 时没走 env.reset → 没有 screw/* 种子值，也没有 teacher demo reward，
            # 直接跳过 _log_demo_summary 里与 env 相关的那几条，仅写 buffer_size。

        # 把 demo 阶段吸收到的 task/* + teacher demo reward 写到 W&B step=0，
        # 让用户在等第一次 eval 之前就能看到起点数据；sanity/* 已经在 restore_train 里写过。
        self._log_demo_summary()

        # 从此以后 running / sa stats 保持固定
        self.set_train()
        self.save(os.path.join(self.nn_dir, "model_last"))

        # ---- stage 2: pure supervised training ----
        max_agent_steps = int(
            getattr(self.config, "max_agent_steps", 0)
            or self.ppo_config.get("max_agent_steps", int(1e9))
            or int(1e9)
        )
        save_interval = int(self.config.save_interval_agent_steps)
        eval_interval = int(getattr(self.config, "eval_interval_agent_steps", 0) or 0)
        updates = max(1, int(self.config.updates_per_collect))

        # 可选：训练主循环前再跑一轮纯 BC 预训练（快速消除 adapt_tconv 初始噪声）
        pretrain_steps = int(getattr(self.config, "pretrain_steps", 0) or 0)
        if pretrain_steps > 0:
            cprint(
                f"[BC] pretrain adapt_tconv: {pretrain_steps} SGD steps "
                f"(batch={int(self.config.batch_size)})",
                "green", attrs=["bold"],
            )
            for i in range(pretrain_steps):
                m = self._supervised_update()
                if m is not None:
                    self.loss_meter.update(
                        torch.tensor([m["adapt_loss"]], device=self.device)
                    )
                if (i + 1) % 200 == 0:
                    tprint(
                        f"[BC] pretrain {i + 1}/{pretrain_steps} | "
                        f"avg_loss={self.loss_meter.get_mean():.4f}"
                    )
            self.save(os.path.join(self.nn_dir, "model_last"))
            self.last_save_agent_steps = self.agent_steps

            # Run one pure-student eval right after BC pretraining so short
            # timeout-based runs still get a meaningful `model_best.ckpt`.
            if eval_interval > 0:
                eval_metrics = self._periodic_student_eval()
                self.last_eval_agent_steps = self.agent_steps
                del eval_metrics
                self.set_train()

        # 主循环：每 iter = updates 次 SGD + 步进 agent_steps + 可能的 eval
        while self.agent_steps <= max_agent_steps:
            last_metrics = None
            for _ in range(updates):
                m = self._supervised_update()
                if m is not None:
                    last_metrics = m
                    self.loss_meter.update(
                        torch.tensor([m["adapt_loss"]], device=self.device)
                    )

            # 每 iter 刷新本轮 SGD 的 loss 值；task/* 等 sticky 缓存保留不动。
            if last_metrics is not None:
                self.direct_info["bc_loss"] = float(last_metrics["bc_loss"])
                self.direct_info["latent_loss"] = float(last_metrics["latent_loss"])
                self.direct_info["adapt_loss"] = float(last_metrics["adapt_loss"])

            self.agent_steps += self.batch_size
            self.collect_steps += 1

            self._log_training_info({})

            if (
                save_interval > 0
                and self.agent_steps - self.last_save_agent_steps >= save_interval
            ):
                self.save(os.path.join(
                    self.nn_dir, f"{self.agent_steps // int(1e8)}00m"
                ))
                self.save(os.path.join(self.nn_dir, "model_last"))
                self.last_save_agent_steps = self.agent_steps

            # periodic pure-student eval → model_best
            if (
                eval_interval > 0
                and self.agent_steps - self.last_eval_agent_steps >= eval_interval
            ):
                eval_metrics = self._periodic_student_eval()
                self.last_eval_agent_steps = self.agent_steps
                # eval 结束后回到训练模式（set_train）
                self.set_train()

            eval_select_metrics = self._run_eval_select_if_due(
                "EVAL_SELECT/student",
                train_reward=self.last_eval_reward,
                eval_best_stem="model_best_eval",
                alias_stems=("model_best", "model_best_deploy"),
            )
            if eval_select_metrics is not None:
                self.set_train()

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = self.batch_size / (time.time() - _last_t)
            _last_t = time.time()

            # Terminal 进度条：与 DAgger 口径对齐（BC 没有 mixed rollout → 用 Avg Return =
            # 最近一次 pure-student eval 奖励；其余列名/顺序和 DAgger 完全一致，便于同屏对比）：
            #   Agent Steps / FPS / Last FPS / Train Return /
            #   Best Student Eval / Last Student Eval  (| Buffer: BC-specific tail)
            if np.isfinite(self.last_eval_reward):
                last_eval_str = f"{self.last_eval_reward:.2f}"
                avg_ret_str = f"{self.last_eval_reward:.2f}"
            else:
                last_eval_str = "n/a"
                avg_ret_str = "n/a"
            if self.best_eval_rewards > -1e4:
                best_eval_str = f"{self.best_eval_rewards:.2f}"
            else:
                best_eval_str = "n/a"
            info_string = (
                f"Agent Steps: {int(self.agent_steps // 1e6):04}M | "
                f"FPS: {all_fps:.1f} | Last FPS: {last_fps:.1f} | "
                f"Train Return: {avg_ret_str} | "
                f"Best Student Eval: {best_eval_str} | "
                f"Last Student Eval: {last_eval_str} | "
                f"Buffer: {len(self.dataset)}"
            )
            tprint(info_string)
