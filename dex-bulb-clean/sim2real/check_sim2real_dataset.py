from __future__ import annotations

import argparse
import glob
import pickle
from pathlib import Path

import numpy as np


EXPECTED_SHAPES = {
    "paxini_pos": (16,),
    "paxini_act": (16,),
    "paxini_tactile": (33,),
    "control": (16,),
    "action": (16,),
    "actions": (16,),
    "obs": (96,),
    "proprio_hist": (30, 32),
    "point_cloud_info": (100, 3),
    "q_policy": (16,),
    "hand_dof_pos": (16,),
    "hand_dof_target": (16,),
    "hand_joint_pos": (16,),
    "arm_joint_positions": (6,),
    "arm_joint_velocities": (6,),
    "arm_eef_pose": (7,),
    "arm_control": (6,),
    "wrist_pose": (7,),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="A sim2real data run directory.")
    parser.add_argument("--max-files", type=int, default=5)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    pkls = sorted(glob.glob(str(data_dir / "*.pkl")))
    if not pkls:
        raise SystemExit(f"No .pkl files found in {data_dir}")

    print(f"Found {len(pkls)} pkl files. Checking first {min(len(pkls), args.max_files)}...")
    for pkl_path in pkls[: args.max_files]:
        with open(pkl_path, "rb") as f:
            frame = pickle.load(f)
        for key, shape in EXPECTED_SHAPES.items():
            if key not in frame:
                raise SystemExit(f"{pkl_path}: missing key {key}")
            got = np.asarray(frame[key]).shape
            if got != shape:
                raise SystemExit(f"{pkl_path}: {key} shape {got}, expected {shape}")
        tactile_detail = np.asarray(frame["paxini_tactile_detail"])
        if tactile_detail.ndim != 1 or tactile_detail.size % 3 != 0:
            raise SystemExit(
                f"{pkl_path}: paxini_tactile_detail shape {tactile_detail.shape}, "
                "expected a flat xyz array with length divisible by 3"
            )
    print("Dataset schema OK.")


if __name__ == "__main__":
    main()
