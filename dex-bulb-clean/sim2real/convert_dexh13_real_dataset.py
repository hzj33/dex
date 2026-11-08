from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch


OBS_DIM = 96
PROPRIO_HIST_SHAPE = (30, 32)
ACTION_DIM = 16


CORE_KEYS = ("obs", "proprio_hist")


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


def _read_valid_frames(data_dir: Path, max_gap: int) -> tuple[list[Path], list[dict[str, Any]]]:
    paths = sorted(data_dir.glob("*.pkl"))
    valid_paths: list[Path] = []
    frames: list[dict[str, Any]] = []
    last_step: int | None = None

    for path in paths:
        frame = _load_frame(path)
        if not all(key in frame for key in CORE_KEYS):
            continue
        _as_float32(frame, "obs", (OBS_DIM,))
        _as_float32(frame, "proprio_hist", PROPRIO_HIST_SHAPE)
        _read_action(frame)

        step = int(frame.get("policy_step", len(frames)))
        if last_step is not None and max_gap >= 0 and step - last_step > max_gap:
            print(
                f"[warn] policy_step gap before {path.name}: "
                f"{last_step} -> {step}"
            )
        last_step = step

        valid_paths.append(path)
        frames.append(frame)

    if len(frames) < 2:
        raise SystemExit(f"Need at least 2 valid pkl frames in {data_dir}, got {len(frames)}.")
    return valid_paths, frames


def _check_obs_proprio_alignment(
    paths: list[Path],
    frames: list[dict[str, Any]],
    tolerance: float,
    allow_mismatch: bool,
) -> None:
    max_diff = 0.0
    worst_path: Path | None = None
    for path, frame in zip(paths, frames):
        obs = _as_float32(frame, "obs", (OBS_DIM,)).reshape(3, 32)
        proprio_hist = _as_float32(frame, "proprio_hist", PROPRIO_HIST_SHAPE)
        diff = float(np.max(np.abs(obs - proprio_hist[-3:])))
        if diff > max_diff:
            max_diff = diff
            worst_path = path

    if max_diff <= tolerance:
        return

    message = (
        "obs/proprio_hist alignment check failed: "
        f"max_diff={max_diff:.6g} at {worst_path}. "
        "This usually means the run was recorded before async-save arrays were copied."
    )
    if allow_mismatch:
        print(f"[warn] {message}")
        return
    raise SystemExit(message)


def _stack(frames: list[dict[str, Any]], key: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.from_numpy(np.stack([_as_float32(frame, key, shape) for frame in frames]))


def _stack_actions(frames: list[dict[str, Any]]) -> torch.Tensor:
    return torch.from_numpy(np.stack([_read_action(frame) for frame in frames]))


def convert_run(
    data_dir: Path,
    output: Path,
    *,
    buffer_type: str,
    mark_last_done: bool,
    max_gap: int,
    alignment_tolerance: float,
    allow_alignment_mismatch: bool,
) -> None:
    paths, frames = _read_valid_frames(data_dir, max_gap=max_gap)
    _check_obs_proprio_alignment(
        paths,
        frames,
        tolerance=float(alignment_tolerance),
        allow_mismatch=bool(allow_alignment_mismatch),
    )

    obs_all = _stack(frames, "obs", (OBS_DIM,))
    proprio_all = _stack(frames, "proprio_hist", PROPRIO_HIST_SHAPE)
    actions_all = _stack_actions(frames)

    meta = {
        "source_dir": str(data_dir),
        "num_source_frames": len(frames),
        "first_file": paths[0].name,
        "last_file": paths[-1].name,
        "task": "Dexh13HoraLightbulb",
        "obs_dim": OBS_DIM,
        "proprio_hist_shape": PROPRIO_HIST_SHAPE,
        "action_dim": ACTION_DIM,
        "action_semantics": "normalized policy action in [-1, 1]",
        "target_semantics": "paxini_act ~= previous target + action * sim_action_scale",
        "sim_action_scale": float(frames[0].get("sim_action_scale", 0.05)),
        "sim_control_dt": float(frames[0].get("sim_control_dt", 0.05)),
    }

    expert_payload = {
        "obs": obs_all,
        "proprio_hist": proprio_all,
        "actions": actions_all,
        "size": int(obs_all.shape[0]),
        "meta": {**meta, "buffer_format": "raw_expert"},
    }

    dones = torch.zeros((len(frames) - 1, 1), dtype=torch.float32)
    if mark_last_done:
        dones[-1, 0] = 1.0
    replay_payload = {
        "obs": obs_all[:-1],
        "proprio_hist": proprio_all[:-1],
        "actions": actions_all[:-1],
        "next_obs": obs_all[1:],
        "next_proprio_hist": proprio_all[1:],
        "dones": dones,
        "size": int(obs_all.shape[0] - 1),
        "meta": {**meta, "buffer_format": "raw_replay"},
    }

    if buffer_type == "expert":
        payload: dict[str, Any] = expert_payload
    elif buffer_type == "replay":
        payload = replay_payload
    else:
        payload = {"meta": meta, "expert": expert_payload, "replay": replay_payload}

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"[convert] source frames: {len(frames)}")
    print(f"[convert] wrote: {output}")
    if buffer_type == "expert":
        print(f"[convert] expert samples: {payload['size']}")
    elif buffer_type == "replay":
        print(f"[convert] replay transitions: {payload['size']}")
    else:
        print(f"[convert] expert samples: {payload['expert']['size']}")
        print(f"[convert] replay transitions: {payload['replay']['size']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert DexH13 sim2real pkl frames into raw tensors for Dexh13HoraLightbulb training."
    )
    parser.add_argument("data_dir", help="Run directory containing per-frame .pkl files.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .pt file. Default: <data_dir>/dexh13_real_dataset.pt",
    )
    parser.add_argument(
        "--buffer-type",
        choices=("expert", "replay", "both"),
        default="expert",
        help="Which tensor layout to write.",
    )
    parser.add_argument(
        "--no-mark-last-done",
        action="store_true",
        help="Keep all replay done flags at 0 instead of marking the last transition done.",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=1,
        help="Warn if adjacent policy_step values differ by more than this. Use -1 to disable.",
    )
    parser.add_argument(
        "--alignment-tolerance",
        type=float,
        default=1e-5,
        help="Maximum allowed max(abs(obs.reshape(3,32) - proprio_hist[-3:])).",
    )
    parser.add_argument(
        "--allow-alignment-mismatch",
        action="store_true",
        help="Warn instead of failing if obs/proprio_hist alignment is inconsistent.",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else data_dir / "dexh13_real_dataset.pt"
    )
    convert_run(
        data_dir,
        output,
        buffer_type=args.buffer_type,
        mark_last_done=not args.no_mark_last_done,
        max_gap=int(args.max_gap),
        alignment_tolerance=float(args.alignment_tolerance),
        allow_alignment_mismatch=bool(args.allow_alignment_mismatch),
    )


if __name__ == "__main__":
    main()
