#!/usr/bin/env python3
import argparse
from typing import Dict, Iterable, Tuple

import torch


DEFAULT_KEYS = [
    "done_rate_per_step",
    "extras/step_all_reward",
    "extras/rotation_reward",
    "extras/pose_diff_penalty",
    "extras/torques",
    "extras/work_done",
    "extras/screw/angular_velocity",
    "extras/screw/angular_position",
    "extras/screw/positive_vel_ratio",
]

DEFAULT_SEGMENTS = (
    ("early", 0.0, 1.0 / 3.0),
    ("mid", 1.0 / 3.0, 2.0 / 3.0),
    ("late", 2.0 / 3.0, 1.0),
)


def load_series(payload: Dict, key: str) -> torch.Tensor:
    if key == "reward_per_step":
        rewards = payload["rewards"].float()
        return rewards.mean(dim=1)
    if key == "done_rate_per_step":
        return payload["done_rate_per_step"].float()
    if key.startswith("extras/"):
        extra_key = key.split("/", 1)[1]
        extras = payload.get("extras", {})
        if extra_key not in extras:
            raise KeyError(extra_key)
        return extras[extra_key].float()
    raise KeyError(key)


def fmt(value: float) -> str:
    return f"{value:.6f}"


def segment_bounds(length: int, start_ratio: float, end_ratio: float) -> Tuple[int, int]:
    start = int(length * start_ratio)
    end = int(length * end_ratio)
    start = max(0, min(start, length - 1))
    end = max(start + 1, min(end, length))
    return start, end


def summarize_series(name: str, ref: torch.Tensor, cand: torch.Tensor, segments: Iterable[Tuple[str, float, float]]) -> str:
    lines = []
    ref_mean = float(ref.mean().item())
    cand_mean = float(cand.mean().item())
    delta = cand_mean - ref_mean
    lines.append(
        f"- {name}: ref={fmt(ref_mean)} | cand={fmt(cand_mean)} | delta={fmt(delta)}"
    )
    for seg_name, start_ratio, end_ratio in segments:
        ref_s, ref_e = segment_bounds(ref.shape[0], start_ratio, end_ratio)
        cand_s, cand_e = segment_bounds(cand.shape[0], start_ratio, end_ratio)
        ref_seg = float(ref[ref_s:ref_e].mean().item())
        cand_seg = float(cand[cand_s:cand_e].mean().item())
        lines.append(
            f"  {seg_name}: ref={fmt(ref_seg)} | cand={fmt(cand_seg)} | delta={fmt(cand_seg - ref_seg)}"
        )
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare rollout diagnostics between two .pt payloads.")
    parser.add_argument("ref", help="Reference rollout .pt")
    parser.add_argument("cand", help="Candidate rollout .pt")
    parser.add_argument(
        "--keys",
        nargs="*",
        default=["reward_per_step", *DEFAULT_KEYS],
        help="Series keys to compare. Supports reward_per_step, done_rate_per_step, extras/<key>.",
    )
    args = parser.parse_args()

    ref = torch.load(args.ref, map_location="cpu")
    cand = torch.load(args.cand, map_location="cpu")

    print(f"ref:  {args.ref}")
    print(f"cand: {args.cand}")
    print()

    for key in args.keys:
        try:
            ref_series = load_series(ref, key)
            cand_series = load_series(cand, key)
        except KeyError:
            print(f"- {key}: missing in one of the rollout payloads")
            continue
        print(summarize_series(key, ref_series, cand_series, DEFAULT_SEGMENTS))


if __name__ == "__main__":
    main()
