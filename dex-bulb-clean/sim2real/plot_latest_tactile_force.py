from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    message=r"numpy\.core\.numeric is deprecated.*",
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TACTILE_KEY = "paxini_tactile"
NUM_SENSORS = 11
TACTILE_DIM = 33
AXIS_LABELS = ("Fx", "Fy", "Fz")


def _run_latest_mtime(run_dir: Path) -> float:
    return max(path.stat().st_mtime for path in run_dir.glob("*.pkl"))


def find_run_dirs(root_dir: Path) -> list[Path]:
    runs = []
    for candidate in root_dir.rglob("*"):
        if not candidate.is_dir():
            continue
        if any(candidate.glob("*.pkl")):
            runs.append(candidate)

    if not runs:
        raise SystemExit(f"No pkl run directories found under {root_dir}")
    return sorted(
        runs,
        key=lambda path: (path.relative_to(root_dir).as_posix(), _run_latest_mtime(path)),
    )


def load_frame(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def timestamp_seconds(frame: dict[str, Any], fallback_index: int) -> float:
    timestamp = frame.get("timestamp")
    if isinstance(timestamp, str):
        try:
            return dt.datetime.fromisoformat(timestamp).timestamp()
        except ValueError:
            pass
    return float(fallback_index)


def load_tactile_force(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    frame_paths = sorted(run_dir.glob("*.pkl"))
    if not frame_paths:
        raise SystemExit(f"No .pkl files found in {run_dir}")

    times = []
    forces = []
    for i, path in enumerate(frame_paths):
        frame = load_frame(path)
        if TACTILE_KEY not in frame:
            raise KeyError(f"{path} missing {TACTILE_KEY}")
        tactile = np.asarray(frame[TACTILE_KEY], dtype=np.float32).reshape(TACTILE_DIM)
        tactile_xyz = tactile.reshape(NUM_SENSORS, 3)
        times.append(timestamp_seconds(frame, i))
        forces.append(tactile_xyz)

    times_arr = np.asarray(times, dtype=np.float64)
    times_arr -= times_arr[0]
    return times_arr, np.asarray(forces, dtype=np.float32)


def output_path_for_run(
    run_dir: Path,
    root_dir: Path,
    output: Path | None,
    multiple_runs: bool,
) -> Path:
    if output is None:
        return run_dir / "tactile_force_xyz.png"

    if not multiple_runs:
        return output

    relative_run_dir = run_dir.relative_to(root_dir)
    return output / relative_run_dir / "tactile_force_xyz.png"


def plot_force(times: np.ndarray, forces: np.ndarray, run_dir: Path, output: Path) -> None:
    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    axes_flat = axes.reshape(-1)

    for sensor_idx in range(NUM_SENSORS):
        ax = axes_flat[sensor_idx]
        sensor_force = forces[:, sensor_idx, :]
        for axis_idx, label in enumerate(AXIS_LABELS):
            ax.plot(times, sensor_force[:, axis_idx], linewidth=1.2, label=label)
        max_abs = float(np.max(np.abs(sensor_force)))
        y_limit = max(1.0, max_abs * 1.05)
        ax.set_ylim(-y_limit, y_limit)
        ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
        ax.set_title(f"Sensor {sensor_idx}", fontsize=10)
        ax.set_ylabel("force")
        ax.grid(True, alpha=0.3)
        if sensor_idx == 0:
            ax.legend(loc="upper right", fontsize=8, frameon=False)

    axes_flat[-1].axis("off")
    for ax in axes[-1, :2]:
        ax.set_xlabel("time (s)")

    fig.suptitle(f"Tactile xyz force over time\n{run_dir}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot 11 PaXini tactile sensor xyz force curves for every run under a root directory."
    )
    parser.add_argument(
        "--root-dir",
        default=str(Path(__file__).resolve().parent.parent / "data" / "sim2real_policy"),
        help="Root directory to search for run folders when --run-dir is not set.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Specific run directory containing per-frame pkl files.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output png path. Default: <run-dir>/tactile_force_xyz.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_dir = Path(args.root_dir).expanduser().resolve()
    run_dirs = (
        [Path(args.run_dir).expanduser().resolve()]
        if args.run_dir
        else find_run_dirs(root_dir)
    )
    output_root = Path(args.output).expanduser().resolve() if args.output else None

    if output_root is not None and len(run_dirs) > 1 and output_root.suffix:
        raise SystemExit("When plotting multiple runs, --output must be a directory or omitted.")

    for run_dir in run_dirs:
        times, forces = load_tactile_force(run_dir)
        output = output_path_for_run(run_dir, root_dir, output_root, len(run_dirs) > 1)
        plot_force(times, forces, run_dir, output)
        print(f"[tactile-force] run: {run_dir}")
        print(f"[tactile-force] frames: {forces.shape[0]}")
        print(f"[tactile-force] saved: {output}")


if __name__ == "__main__":
    main()
