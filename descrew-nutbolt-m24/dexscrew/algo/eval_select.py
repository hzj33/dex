import csv
import json
import os
import shutil
import time

import numpy as np
import torch
from termcolor import cprint


def cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _as_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _as_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


class EvalSelectMixin:
    """Small eval-best helper shared by teacher PPO and student distillers.

    The mixin only evaluates the current task/env. Multi-condition deploy eval is
    handled by scripts/eval_select_checkpoints.py after training.
    """

    def _init_eval_select(self, role, artifact_ext):
        cfg = cfg_get(self.ppo_config, "eval_select", {})
        self.eval_select_role = str(role)
        self.eval_select_artifact_ext = str(artifact_ext).lstrip(".")
        self.eval_select_enabled = _as_bool(cfg_get(cfg, "enabled", False))
        self.eval_select_interval_agent_steps = _as_int(
            cfg_get(cfg, "interval_agent_steps", 20_000_000), 20_000_000
        )
        self.eval_select_min_agent_steps = _as_int(
            cfg_get(cfg, "min_agent_steps", 0), 0
        )
        self.eval_select_num_steps = max(
            1, _as_int(cfg_get(cfg, "num_steps", 256), 256)
        )
        self.eval_select_done_penalty = _as_float(
            cfg_get(cfg, "done_penalty", 2000.0), 2000.0
        )
        self.eval_select_min_score_improvement = _as_float(
            cfg_get(cfg, "min_score_improvement", 0.0), 0.0
        )
        self.eval_select_final_eval = _as_bool(cfg_get(cfg, "final_eval", True))
        self.eval_select_save_deploy_best = _as_bool(
            cfg_get(cfg, "save_deploy_best", True)
        )
        self.eval_select_best_score = -float("inf")
        self.eval_select_best_reward = float("nan")
        self.eval_select_best_done_rate = float("nan")
        self.eval_select_last_agent_steps = -1
        self.eval_select_count = 0
        self.eval_select_dir = os.path.join(self.output_dir, "eval_select")
        self.eval_select_history_path = os.path.join(
            self.eval_select_dir, "train_eval_history.tsv"
        )
        if self.eval_select_enabled:
            os.makedirs(self.eval_select_dir, exist_ok=True)
            cprint(
                "EvalSelect enabled: "
                f"role={self.eval_select_role} steps={self.eval_select_num_steps} "
                f"interval={self.eval_select_interval_agent_steps} "
                f"done_penalty={self.eval_select_done_penalty}",
                "cyan",
            )

    def _eval_select_artifact_path(self, stem):
        return os.path.join(self.nn_dir, f"{stem}.{self.eval_select_artifact_ext}")

    def _eval_select_should_run(self):
        if not getattr(self, "eval_select_enabled", False):
            return False
        if self.eval_select_interval_agent_steps <= 0:
            return False
        if int(self.agent_steps) < self.eval_select_min_agent_steps:
            return False
        if self.eval_select_last_agent_steps < 0:
            return True
        return (
            int(self.agent_steps) - int(self.eval_select_last_agent_steps)
            >= self.eval_select_interval_agent_steps
        )

    def _eval_select_score(self, avg_reward, avg_done_rate):
        return float(avg_reward) - self.eval_select_done_penalty * float(avg_done_rate)

    def _snapshot_eval_select_modes(self):
        seen = set()
        modes = []
        module_names = (
            "model",
            "running_mean_std",
            "sa_mean_std",
            "priv_mean_std",
            "point_cloud_mean_std",
            "value_mean_std",
            "diffusion_model",
            "consistency_model",
            "consistency_ema_model",
            "flow_model",
        )
        for name in module_names:
            module = getattr(self, name, None)
            if module is None or id(module) in seen or not hasattr(module, "training"):
                continue
            seen.add(id(module))
            modes.append((module, bool(module.training)))
        return modes

    def _restore_eval_select_modes(self, modes):
        for module, was_training in modes:
            module.train(was_training)

    def _eval_select_accumulate_info(self, info, acc):
        if not isinstance(info, dict):
            return
        for key, value in info.items():
            scalar = None
            if torch.is_tensor(value):
                if value.numel() == 0:
                    continue
                scalar = float(value.float().mean().detach().cpu())
            elif isinstance(value, (int, float, np.number)):
                scalar = float(value)
            if scalar is None or not np.isfinite(scalar):
                continue
            acc.setdefault(str(key), []).append(scalar)

    def _eval_select_summarize_info(self, acc):
        out = {}
        for key, values in acc.items():
            if not values:
                continue
            val = float(np.mean(values))
            if np.isfinite(val):
                out[key] = val
        return out

    def _eval_select_action(self, obs_dict):
        raise NotImplementedError

    def _reset_eval_select_training_state(self):
        tensor_names = (
            "current_rewards",
            "current_lengths",
            "step_reward",
            "step_length",
            "student_step_reward",
            "student_step_length",
            "student_tracking_active",
            "teacher_action_buffer",
            "cond_obs_buffer",
            "cond_prop_buffer",
            "valid_window_len",
        )
        for name in tensor_names:
            value = getattr(self, name, None)
            if torch.is_tensor(value):
                value.zero_()

    def _run_eval_select_rollout(self, label):
        modes = self._snapshot_eval_select_modes()
        self.set_eval()
        obs_dict = self.env.reset()
        train_obs_dict = obs_dict
        reward_sum = 0.0
        done_sum = 0.0
        info_acc = {}
        steps = int(self.eval_select_num_steps)
        try:
            with torch.no_grad():
                for _ in range(steps):
                    action_result = self._eval_select_action(obs_dict)
                    if isinstance(action_result, tuple):
                        action, step_kwargs = action_result
                    else:
                        action, step_kwargs = action_result, {}
                    action = torch.nan_to_num(
                        action, nan=0.0, posinf=1.0, neginf=-1.0
                    )
                    action = torch.clamp(action, -1.0, 1.0).contiguous()
                    obs_dict, reward, done, info = self.env.step(action, **step_kwargs)
                    reward_sum += float(reward.float().mean().detach().cpu())
                    done_sum += float(done.float().mean().detach().cpu())
                    self._eval_select_accumulate_info(info, info_acc)
                train_obs_dict = self.env.reset()
                self._reset_eval_select_training_state()
        finally:
            self._restore_eval_select_modes(modes)

        avg_reward = reward_sum / float(steps)
        avg_done_rate = done_sum / float(steps)
        score = self._eval_select_score(avg_reward, avg_done_rate)
        return {
            "label": str(label),
            "steps": steps,
            "avg_reward": avg_reward,
            "avg_done_rate": avg_done_rate,
            "score": score,
            "info_means": self._eval_select_summarize_info(info_acc),
            "final_obs_dict": train_obs_dict,
        }

    def _write_eval_select_history(self, metrics, train_reward, saved_best):
        os.makedirs(self.eval_select_dir, exist_ok=True)
        file_exists = os.path.exists(self.eval_select_history_path)
        row = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "role": self.eval_select_role,
            "algo": self.__class__.__name__,
            "label": metrics["label"],
            "agent_steps": int(self.agent_steps),
            "epoch": int(getattr(self, "epoch_num", 0)),
            "eval_steps": int(metrics["steps"]),
            "train_reward": float(train_reward)
            if train_reward is not None and np.isfinite(train_reward)
            else "",
            "avg_reward": float(metrics["avg_reward"]),
            "avg_done_rate": float(metrics["avg_done_rate"]),
            "score": float(metrics["score"]),
            "best_score": float(self.eval_select_best_score),
            "saved_best": int(bool(saved_best)),
            "info_means_json": json.dumps(
                metrics.get("info_means", {}), sort_keys=True
            ),
        }
        with open(self.eval_select_history_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()), delimiter="\t")
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def _log_eval_select_scalars(self, metrics):
        if not hasattr(self, "writer"):
            return
        self.writer.add_scalar(
            f"eval_select/{self.eval_select_role}_reward",
            float(metrics["avg_reward"]),
            self.agent_steps,
        )
        self.writer.add_scalar(
            f"eval_select/{self.eval_select_role}_done_rate",
            float(metrics["avg_done_rate"]),
            self.agent_steps,
        )
        self.writer.add_scalar(
            f"eval_select/{self.eval_select_role}_score",
            float(metrics["score"]),
            self.agent_steps,
        )
        self.writer.add_scalar(
            f"eval_select/{self.eval_select_role}_best_score",
            float(self.eval_select_best_score),
            self.agent_steps,
        )
        for key, value in metrics.get("info_means", {}).items():
            self.writer.add_scalar(f"eval_select_info/{key}", value, self.agent_steps)

    def _save_eval_select_best(self, eval_best_stem, alias_stems):
        self.save(os.path.join(self.nn_dir, eval_best_stem))
        src = self._eval_select_artifact_path(eval_best_stem)
        for alias in alias_stems:
            if not alias or alias == eval_best_stem:
                continue
            shutil.copy2(src, self._eval_select_artifact_path(alias))

    def _consider_eval_select_metrics(
        self, metrics, train_reward=None, eval_best_stem="best_eval", alias_stems=()
    ):
        score = float(metrics["score"])
        saved_best = (
            np.isfinite(score)
            and score > self.eval_select_best_score + self.eval_select_min_score_improvement
        )
        if saved_best:
            self.eval_select_best_score = score
            self.eval_select_best_reward = float(metrics["avg_reward"])
            self.eval_select_best_done_rate = float(metrics["avg_done_rate"])
            self._save_eval_select_best(eval_best_stem, alias_stems)
            cprint(
                "EvalSelect new best: "
                f"role={self.eval_select_role} score={score:.6f} "
                f"reward={metrics['avg_reward']:.6f} "
                f"done={metrics['avg_done_rate']:.6f}",
                "green",
            )
        self._write_eval_select_history(metrics, train_reward, saved_best)
        self._log_eval_select_scalars(metrics)
        self.eval_select_count += 1
        return saved_best

    def _run_eval_select_if_due(
        self, label, train_reward=None, eval_best_stem="best_eval", alias_stems=()
    ):
        if not self._eval_select_should_run():
            return None
        metrics = self._run_eval_select_rollout(label)
        self.eval_select_last_agent_steps = int(self.agent_steps)
        self._consider_eval_select_metrics(
            metrics,
            train_reward=train_reward,
            eval_best_stem=eval_best_stem,
            alias_stems=alias_stems,
        )
        cprint(
            "EvalSelectSummary "
            f"role={self.eval_select_role} label={label} "
            f"steps={metrics['steps']} avg_reward={metrics['avg_reward']:.6f} "
            f"avg_done_rate={metrics['avg_done_rate']:.6f} "
            f"score={metrics['score']:.6f} "
            f"best_score={self.eval_select_best_score:.6f}",
            "cyan",
        )
        return metrics

    def _run_final_eval_select(
        self, label, train_reward=None, eval_best_stem="best_eval", alias_stems=()
    ):
        if not getattr(self, "eval_select_enabled", False) or not self.eval_select_final_eval:
            return None
        metrics = self._run_eval_select_rollout(label)
        self.eval_select_last_agent_steps = int(self.agent_steps)
        self._consider_eval_select_metrics(
            metrics,
            train_reward=train_reward,
            eval_best_stem=eval_best_stem,
            alias_stems=alias_stems,
        )
        return metrics

    def _publish_eval_select_deploy_best(self, eval_best_stem, deploy_stem):
        if not getattr(self, "eval_select_enabled", False):
            return
        if not self.eval_select_save_deploy_best:
            return
        src = self._eval_select_artifact_path(eval_best_stem)
        dst = self._eval_select_artifact_path(deploy_stem)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            cprint(
                f"EvalSelect deploy best: {src} -> {dst}",
                "green",
            )
