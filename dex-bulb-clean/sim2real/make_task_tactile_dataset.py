from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


TACTILE_DIM = 33
TACTILE_POINT_DIM = 3
NUM_TACTILE_POINTS = TACTILE_DIM // TACTILE_POINT_DIM


def _load_pickle(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict.")
    return data


def _zero_tactile_points(tactile: Any, points: tuple[int, ...]) -> np.ndarray:
    tactile_arr = np.asarray(tactile, dtype=np.float32).reshape(TACTILE_DIM).copy()
    for point in points:
        if point < 0 or point >= NUM_TACTILE_POINTS:
            raise ValueError(
                f"tactile point index {point} out of range [0, {NUM_TACTILE_POINTS - 1}]"
            )
        start = point * TACTILE_POINT_DIM
        tactile_arr[start : start + TACTILE_POINT_DIM] = 0.0
    return tactile_arr


def _write_pickle(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)


def convert_split(src_split: Path, dst_split: Path, zero_points: tuple[int, ...]) -> tuple[int, int]:
    episodes = sorted(path for path in src_split.iterdir() if path.is_dir())
    num_episodes = 0
    num_frames = 0
    for episode in episodes:
        out_episode = dst_split / episode.name
        frame_paths = sorted(episode.glob("*.pkl"))
        if not frame_paths:
            continue
        num_episodes += 1
        for frame_path in frame_paths:
            data = _load_pickle(frame_path)
            if "paxini_tactile" not in data:
                raise KeyError(f"{frame_path} missing paxini_tactile")
            data["paxini_tactile"] = _zero_tactile_points(
                data["paxini_tactile"],
                zero_points,
            )
            _write_pickle(out_episode / frame_path.name, data)
            num_frames += 1
    return num_episodes, num_frames


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a raw sim2real_policy train/test dataset and zero selected "
            "paxini_tactile points while preserving the original pkl schema."
        )
    )
    parser.add_argument(
        "--input-root",
        default="data/sim2real_policy",
        help="Dataset root containing train/ and test/ episode directories.",
    )
    parser.add_argument(
        "--output-root",
        default="data/sim2real_policy_task_tactile",
        help="Output root containing train/ and test/ episode directories.",
    )
    parser.add_argument(
        "--zero-points",
        type=int,
        nargs="*",
        default=(3, 4, 5, 6),
        help="Tactile point indices to zero. Default zeros non-task points 3 4 5 6.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete output-root before writing.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    zero_points = tuple(args.zero_points)

    if args.clean and output_root.exists():
        shutil.rmtree(output_root)

    total_episodes = 0
    total_frames = 0
    for split in ("train", "test"):
        src_split = input_root / split
        if not src_split.exists():
            raise FileNotFoundError(src_split)
        episodes, frames = convert_split(src_split, output_root / split, zero_points)
        total_episodes += episodes
        total_frames += frames
        print(f"[task-tactile] {split}: episodes={episodes}, frames={frames}")

    print(f"[task-tactile] wrote: {output_root}")
    print(f"[task-tactile] zeroed tactile points: {list(zero_points)}")
    print(f"[task-tactile] total episodes={total_episodes}, frames={total_frames}")


if __name__ == "__main__":
    main()
