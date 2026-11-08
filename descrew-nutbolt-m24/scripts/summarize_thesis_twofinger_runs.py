#!/usr/bin/env python3
"""Summarize thesis two-finger lightbulb training runs from TensorBoard logs."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


DEFAULT_ROOT = Path("outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger")

TAGS = [
    "episode_rewards/step",
    "episode_lengths/step",
    "info/best_reward",
    "info/best_reward_step",
    "screw/angular_velocity",
    "screw/positive_vel_ratio",
    "two_finger/gate",
    "two_finger/thumb_contact_w",
    "two_finger/other_contact_w",
    "pose_diff_penalty",
    "pose_diff_penalty/thumb_raw",
    "pose_diff_penalty/thumb_weighted",
    "thumb_slip/contact_drop_frac",
    "thumb_slip/far_frac",
    "thumb_slip/active_detach_frac",
    "thumb_slip/active_far_frac",
    "thumb_slip/ejection_frac",
    "thumb_slip/tip_speed_mean",
    "thumb_slip/tip_speed_p95",
    "thumb_slip/joint_vel_abs_mean",
    "thumb_slip/joint_vel_abs_p95",
    "thumb_slip/dist_p95",
    "thumb_slip/contact_w_p05",
    "thumb_slip/active_screw_frac",
    "thumb_slip/score",
    "term/any_reset_frac",
    "term/finger_dist_frac",
    "term/no_contact_frac",
]

REWARD_FROM_CKPT = re.compile(r"best_reward_(-?\d+(?:\.\d+)?)\.pth$")


def parse_run_arg(value: str) -> Tuple[str, Path]:
    if "=" in value:
        label, path = value.split("=", 1)
        return label.strip(), Path(path.strip())
    return value, DEFAULT_ROOT / value


def latest_event_file(run_dir: Path) -> Optional[Path]:
    events = sorted(
        run_dir.rglob("events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    return events[-1] if events else None


def best_checkpoint(run_dir: Path) -> Tuple[Optional[Path], Optional[float]]:
    stage = run_dir / "stage1_nn"
    candidates = sorted(stage.glob("best_reward_*.pth"))
    if not candidates:
        candidates = sorted(run_dir.rglob("best_reward_*.pth"))
    if not candidates:
        return None, None
    scored = []
    for path in candidates:
        match = REWARD_FROM_CKPT.search(path.name)
        reward = float(match.group(1)) if match else float("-inf")
        scored.append((reward, path))
    reward, path = max(scored, key=lambda item: item[0])
    return path, reward if math.isfinite(reward) else None


def load_scalars(event_file: Optional[Path], tags: Iterable[str]) -> Dict[str, Optional[float]]:
    values = {tag: None for tag in tags}
    if event_file is None:
        return values
    acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    acc.Reload()
    scalar_tags = set(acc.Tags().get("scalars", []))
    for tag in tags:
        if tag not in scalar_tags:
            continue
        scalars = acc.Scalars(tag)
        if scalars:
            values[tag] = float(scalars[-1].value)
    return values


def fmt(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f}"


def summarize_run(label: str, run_dir: Path) -> Dict[str, object]:
    event_file = latest_event_file(run_dir)
    ckpt, ckpt_reward = best_checkpoint(run_dir)
    values = load_scalars(event_file, TAGS)
    return {
        "label": label,
        "run_dir": run_dir,
        "event_file": event_file,
        "best_checkpoint": ckpt,
        "best_checkpoint_reward": ckpt_reward,
        **values,
    }


def print_table(rows: List[Dict[str, object]]) -> None:
    headers = [
        "run",
        "best_ckpt",
        "reward",
        "vel",
        "gate",
        "slip_score",
        "active_far",
        "active_detach",
        "ejection",
        "tip_p95",
        "dist_p95",
        "cw_p05",
        "reset",
        "no_contact",
    ]
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["label"]),
                    fmt(row.get("best_checkpoint_reward"), 2),
                    fmt(row.get("episode_rewards/step"), 2),
                    fmt(row.get("screw/angular_velocity"), 4),
                    fmt(row.get("two_finger/gate"), 4),
                    fmt(row.get("thumb_slip/score"), 5),
                    fmt(row.get("thumb_slip/active_far_frac"), 5),
                    fmt(row.get("thumb_slip/active_detach_frac"), 5),
                    fmt(row.get("thumb_slip/ejection_frac"), 5),
                    fmt(row.get("thumb_slip/tip_speed_p95"), 4),
                    fmt(row.get("thumb_slip/dist_p95"), 4),
                    fmt(row.get("thumb_slip/contact_w_p05"), 4),
                    fmt(row.get("term/any_reset_frac"), 5),
                    fmt(row.get("term/no_contact_frac"), 5),
                ]
            )
            + " |"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "runs",
        nargs="+",
        help="Cache name under thesis output root, or label=/path/to/run",
    )
    args = parser.parse_args()

    rows = [summarize_run(*parse_run_arg(run)) for run in args.runs]
    print_table(rows)
    print()
    for row in rows:
        print(f"{row['label']}:")
        print(f"  run_dir: {row['run_dir']}")
        print(f"  event: {row['event_file']}")
        print(f"  best_checkpoint: {row['best_checkpoint']}")


if __name__ == "__main__":
    main()
