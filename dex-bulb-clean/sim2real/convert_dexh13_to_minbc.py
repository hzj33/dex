from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path
from typing import Any

import numpy as np


OBS_DIM = 96
PROPRIO_HIST_SHAPE = (30, 32)
ACTION_DIM = 16
TACTILE_DIM = 33
TACTILE_POINT_DIM = 3
NUM_TACTILE_POINTS = TACTILE_DIM // TACTILE_POINT_DIM


def _load_frame(path: Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not contain a dict.")
    return data


def _as_float32(data: dict[str, Any], key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in data:
        raise KeyError(f"missing key: {key}")
    arr = np.asarray(data[key], dtype=np.float32)
    if arr.shape != shape:
        raise ValueError(f"{key} shape {arr.shape}, expected {shape}")
    return arr


def _read_action(data: dict[str, Any]) -> np.ndarray:
    key = "action" if "action" in data else "control"
    return _as_float32(data, key, (ACTION_DIM,))


def _zero_tactile_points(tactile: np.ndarray, points: tuple[int, ...]) -> np.ndarray:
    tactile = np.asarray(tactile, dtype=np.float32).reshape(TACTILE_DIM).copy()
    for point in points:
        if point < 0 or point >= NUM_TACTILE_POINTS:
            raise ValueError(
                f"tactile point index {point} out of range [0, {NUM_TACTILE_POINTS - 1}]"
            )
        start = point * TACTILE_POINT_DIM
        tactile[start : start + TACTILE_POINT_DIM] = 0.0
    return tactile


def _read_run(run_dir: Path, alignment_tolerance: float) -> list[tuple[Path, dict[str, Any]]]:
    frames: list[tuple[Path, dict[str, Any]]] = []
    last_step: int | None = None
    for path in sorted(run_dir.glob("*.pkl")):
        frame = _load_frame(path)
        obs = _as_float32(frame, "obs", (OBS_DIM,))
        proprio_hist = _as_float32(frame, "proprio_hist", PROPRIO_HIST_SHAPE)
        _read_action(frame)
        _as_float32(frame, "paxini_pos", (ACTION_DIM,))
        _as_float32(frame, "paxini_tactile", (TACTILE_DIM,))

        diff = float(np.max(np.abs(obs.reshape(3, 32) - proprio_hist[-3:])))
        if diff > alignment_tolerance:
            raise SystemExit(
                f"{path}: obs/proprio_hist mismatch max_diff={diff:.6g}. "
                "Recollect the run or pass a larger --alignment-tolerance only for debugging."
            )

        step = int(frame.get("policy_step", len(frames)))
        if last_step is not None and step - last_step != 1:
            print(f"[warn] policy_step gap in {run_dir.name}: {last_step} -> {step}")
        last_step = step
        frames.append((path, frame))

    if len(frames) < 2:
        raise SystemExit(f"{run_dir}: need at least 2 valid frames, got {len(frames)}.")
    return frames


def _make_minbc_frame(frame: dict[str, Any], zero_tactile_points: tuple[int, ...]) -> dict[str, Any]:
    proprio_hist = _as_float32(frame, "proprio_hist", PROPRIO_HIST_SHAPE)
    tactile = _as_float32(frame, "paxini_tactile", (TACTILE_DIM,))
    if zero_tactile_points:
        tactile = _zero_tactile_points(tactile, zero_tactile_points)
    return {
        "dexh13_obs": _as_float32(frame, "obs", (OBS_DIM,)),
        "dexh13_proprio_hist": proprio_hist.reshape(-1).astype(np.float32),
        "paxini_pos": _as_float32(frame, "paxini_pos", (ACTION_DIM,)),
        "paxini_tactile": tactile,
        "action": _read_action(frame),
        "policy_step": int(frame.get("policy_step", -1)),
        "timestamp": frame.get("timestamp", ""),
    }


def _write_episode(
    frames: list[tuple[Path, dict[str, Any]]],
    episode_dir: Path,
    zero_tactile_points: tuple[int, ...],
) -> None:
    episode_dir.mkdir(parents=True, exist_ok=True)
    for i, (_, frame) in enumerate(frames):
        out = episode_dir / f"step_{i:06d}.pkl"
        with open(out, "wb") as f:
            pickle.dump(
                _make_minbc_frame(frame, zero_tactile_points),
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )


def convert(
    run_dirs: list[Path],
    output_dir: Path,
    *,
    test_ratio: float,
    clean: bool,
    alignment_tolerance: float,
    zero_tactile_points: tuple[int, ...],
) -> None:
    runs = [(run_dir, _read_run(run_dir, alignment_tolerance)) for run_dir in run_dirs]
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)

    num_runs = len(runs)
    num_test = max(1, int(round(num_runs * test_ratio))) if num_runs > 1 else 0
    split_tags = ["train"] * num_runs
    for idx in range(num_runs - num_test, num_runs):
        if 0 <= idx < num_runs:
            split_tags[idx] = "test"

    for idx, ((run_dir, frames), split) in enumerate(zip(runs, split_tags)):
        episode_dir = output_dir / split / f"episode_{idx:04d}_{run_dir.name}"
        _write_episode(frames, episode_dir, zero_tactile_points)
        print(f"[convert-minbc] {run_dir} -> {episode_dir} ({len(frames)} frames)")

    if num_test == 0:
        # MinBC expects a test directory. For a single run, make a tiny tail split
        # as a smoke-test/eval set while keeping frame order inside each episode.
        run_dir, frames = runs[0]
        split_at = max(2, int(len(frames) * (1.0 - test_ratio)))
        split_at = min(split_at, len(frames) - 2)
        train_frames = frames[:split_at]
        test_frames = frames[split_at:]
        if clean and output_dir.exists():
            shutil.rmtree(output_dir)
        _write_episode(
            train_frames,
            output_dir / "train" / f"episode_0000_{run_dir.name}",
            zero_tactile_points,
        )
        _write_episode(
            test_frames,
            output_dir / "test" / f"episode_0000_{run_dir.name}",
            zero_tactile_points,
        )
        print(
            f"[convert-minbc] single-run split: train={len(train_frames)} "
            f"test={len(test_frames)}"
        )

    print(f"[convert-minbc] wrote dataset root: {output_dir}")
    if zero_tactile_points:
        print(f"[convert-minbc] zeroed tactile points: {list(zero_tactile_points)}")
    print("[convert-minbc] no-tactile data_key: dexh13_obs paxini_pos")
    print("[convert-minbc] tactile data_key: dexh13_obs paxini_pos paxini_tactile")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DexH13 sim2real runs to MinBC episode-pkl format."
    )
    parser.add_argument(
        "run_dirs",
        nargs="+",
        help="One or more sim2real_policy run directories containing per-frame pkl files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/dexh13_real_minbc",
        help="Output dataset root with train/test subdirectories.",
    )
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--clean", action="store_true", help="Delete output-dir before writing.")
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=1e-5,
        help="Maximum allowed max(abs(obs.reshape(3,32) - proprio_hist[-3:])).",
    )
    parser.add_argument(
        "--zero-tactile-points",
        type=int,
        nargs="*",
        default=(),
        help=(
            "Tactile point indices to zero before writing MinBC pkl files. "
            "Each point occupies xyz dims, so point3 maps to tactile[9:12]."
        ),
    )
    args = parser.parse_args()

    run_dirs = [Path(p).expanduser().resolve() for p in args.run_dirs]
    output_dir = Path(args.output_dir).expanduser().resolve()
    convert(
        run_dirs,
        output_dir,
        test_ratio=float(args.test_ratio),
        clean=bool(args.clean),
        alignment_tolerance=float(args.alignment_tolerance),
        zero_tactile_points=tuple(args.zero_tactile_points),
    )


if __name__ == "__main__":
    main()
