#!/usr/bin/env python3
"""
Export DOTPG student checkpoint to TorchScript .pt file.

Example:
    conda run -n dexscrew python sim2real/export_dotpg_single_pt.py \
        sim2real/codrive/dotpg_deployv1/model_best.ckpt

This writes the corresponding file next to the checkpoint:
    sim2real/codrive/dotpg_deployv1/model_best.pt

You can also choose an explicit output path:
    conda run -n dexscrew python sim2real/export_dotpg_single_pt.py \
        sim2real/codrive/dotpg_deployv1/model_best.ckpt \
        --output sim2real/codrive/dotpg_deployv1/dotpg_policy.pt
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as exc:
    raise SystemExit(
        "PyTorch is required. Please run this script in the dexscrew environment, e.g. "
        "`conda run -n dexscrew python sim2real/export_dotpg_single_pt.py <ckpt>`."
    ) from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dexscrew.algo.models.block import TemporalConv  # noqa: E402
from dexscrew.algo.models.models import MLP  # noqa: E402


class DOTPGDeployWrapper(nn.Module):
    """TorchScript wrapper for DOTPG student-state deployment."""

    def __init__(self, actor_mlp, mu, adapt_tconv, stats):
        super().__init__()
        self.actor_mlp = actor_mlp
        self.mu = mu
        self.adapt_tconv = adapt_tconv

        self.register_buffer("running_mean", stats["running_mean"].float())
        self.register_buffer("running_var", stats["running_var"].float())
        self.register_buffer("running_count", stats["running_count"].float().reshape(()))
        self.register_buffer("sa_mean", stats["sa_mean"].float())
        self.register_buffer("sa_var", stats["sa_var"].float())
        self.register_buffer("sa_count", stats["sa_count"].float().reshape(()))
        self.register_buffer("running_sa", torch.zeros(1))
        self.register_buffer("pc_mean", stats["pc_mean"].float())
        self.register_buffer("pc_var", stats["pc_var"].float())
        self.register_buffer("pc_count", stats["pc_count"].float().reshape(()))

    def forward(self, input_dict):
        obs = input_dict["obs"]
        proprio_hist = input_dict["proprio_hist"]

        extrin = torch.tanh(self.adapt_tconv(proprio_hist))
        state = torch.cat([obs, extrin], dim=-1)
        mu = self.mu(self.actor_mlp(state))
        return mu, extrin, extrin


def strip_prefix(state_dict, prefix):
    return {
        key[len(prefix):]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def load_checkpoint(ckpt_path):
    try:
        return torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(str(ckpt_path), map_location="cpu")
    except pickle.UnpicklingError:
        return torch.load(str(ckpt_path), map_location="cpu")


def build_wrapper(ckpt):
    required = {
        "policy",
        "adapt_tconv",
        "running_mean_std",
        "sa_mean_std",
        "point_cloud_mean_std",
    }
    missing = sorted(required - set(ckpt.keys()))
    if missing:
        raise KeyError("Not a DOTPG student checkpoint. Missing keys: " + ", ".join(missing))

    policy_sd = ckpt["policy"]
    state_dim = int(policy_sd["actor_mlp.mlp.0.weight"].shape[1])
    hidden_units = [
        int(policy_sd["actor_mlp.mlp.0.weight"].shape[0]),
        int(policy_sd["actor_mlp.mlp.2.weight"].shape[0]),
        int(policy_sd["actor_mlp.mlp.4.weight"].shape[0]),
    ]
    action_dim = int(policy_sd["mu.weight"].shape[0])

    obs_dim = int(ckpt["running_mean_std"]["running_mean"].numel())
    proprio_len, proprio_dim = ckpt["sa_mean_std"]["running_mean"].shape
    proprio_len = int(proprio_len)
    proprio_dim = int(proprio_dim)
    extrin_dim = state_dim - obs_dim
    if extrin_dim <= 0:
        raise ValueError(f"Invalid dimensions: state_dim={state_dim}, obs_dim={obs_dim}")

    actor_mlp = MLP(units=hidden_units, input_size=state_dim)
    actor_mlp.load_state_dict(strip_prefix(policy_sd, "actor_mlp."))

    mu = nn.Linear(hidden_units[-1], action_dim)
    mu.load_state_dict(strip_prefix(policy_sd, "mu."))

    adapt_tconv = TemporalConv(proprio_dim, extrin_dim)
    adapt_tconv.load_state_dict(ckpt["adapt_tconv"])

    stats = {
        "running_mean": ckpt["running_mean_std"]["running_mean"],
        "running_var": ckpt["running_mean_std"]["running_var"],
        "running_count": ckpt["running_mean_std"]["count"],
        "sa_mean": ckpt["sa_mean_std"]["running_mean"],
        "sa_var": ckpt["sa_mean_std"]["running_var"],
        "sa_count": ckpt["sa_mean_std"]["count"],
        "pc_mean": ckpt["point_cloud_mean_std"]["running_mean"],
        "pc_var": ckpt["point_cloud_mean_std"]["running_var"],
        "pc_count": ckpt["point_cloud_mean_std"]["count"],
    }

    wrapper = DOTPGDeployWrapper(actor_mlp, mu, adapt_tconv, stats).cpu().eval()
    shapes = {
        "obs_dim": obs_dim,
        "proprio_len": proprio_len,
        "proprio_dim": proprio_dim,
        "extrin_dim": extrin_dim,
        "action_dim": action_dim,
    }
    return wrapper, shapes


def export_single(ckpt_path, output_path=None):
    ckpt_path = Path(ckpt_path).expanduser().resolve()
    if output_path is None:
        output_path = ckpt_path.with_suffix(".pt")
    else:
        output_path = Path(output_path).expanduser().resolve()

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = load_checkpoint(ckpt_path)
    wrapper, shapes = build_wrapper(ckpt)

    example = {
        "obs": torch.zeros(1, shapes["obs_dim"]),
        "proprio_hist": torch.zeros(1, shapes["proprio_len"], shapes["proprio_dim"]),
        "point_cloud_info": torch.zeros(1, 100, 3),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        eager_out = wrapper(example)
        traced = torch.jit.trace(wrapper, example, strict=False)
        traced.save(str(output_path))

        loaded = torch.jit.load(str(output_path), map_location="cpu").eval()
        loaded_out = loaded(example)

    max_diff = max(float((a - b).abs().max()) for a, b in zip(eager_out, loaded_out))
    print(f"Exported: {output_path}")
    print(
        "Interface: "
        f"obs=(1,{shapes['obs_dim']}), "
        f"proprio_hist=(1,{shapes['proprio_len']},{shapes['proprio_dim']}), "
        "point_cloud_info=(1,100,3)"
    )
    print(
        "Outputs: "
        f"mu=(1,{shapes['action_dim']}), "
        f"extrin=(1,{shapes['extrin_dim']}), "
        f"extrin_gt=(1,{shapes['extrin_dim']})"
    )
    print(f"JIT reload check max_diff={max_diff:.3e}")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Export one DOTPG .ckpt to one TorchScript .pt.")
    parser.add_argument("ckpt", type=Path, help="Path to one DOTPG model_best.ckpt.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output .pt path. Defaults to the same path with .pt suffix.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    export_single(args.ckpt, args.output)


if __name__ == "__main__":
    main()
