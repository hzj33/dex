#!/usr/bin/env python3
"""Import a physically executed, explicitly passed diagnostic into cloud A1b.

Recheck the saved per-step conjunction and task hashes before writing the
training input. This does not execute a simulator or launch training.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import yaml

from m24_e002_a0c_diagnostics import summarize


def validate(source):
    report = json.loads(source.read_text())
    if report.get("status") != "passed" or not report.get("diagnostic_only"):
        raise ValueError("A0c JSON did not explicitly pass; training remains blocked")
    thresholds = report["thresholds"]
    mode = report.get("acceptance_mode", "strict_normal_proxy")
    current = yaml.safe_load(Path("configs/task/XHandPasiniM24NutBoltE002.yaml").read_text())["env"]["e002_motion"]
    assert mode == current.get("acceptance_mode", "strict_normal_proxy"), "A0c contract differs from training"
    for name in ("pair_gate_min", "torque_min", "torque_saturation", "contact_force_on",
                 "contact_force_off", "contact_dist_on", "contact_dist_off", "direction"):
        assert thresholds[name] == current[name], f"A0c threshold differs: {name}"
    assert thresholds["required_probe_streak"] == 5 and thresholds["min_probe_angle"] == math.pi / 180
    checked = summarize(report["candidates"], thresholds, acceptance_mode=mode)
    assert checked["status"] == "passed" and report["selected"] == checked["selected"]
    assert report["force_sign"] == checked["selected"]["force_sign"]
    assert report["drive_action"] == checked["selected"]["action"]
    assert report.get("drive_action_scale_schedule") == checked["selected"].get("action_scale_schedule")
    assert report["pose_index"] == 743 and len(report["custom_init_dof_pos"]) == 16
    hashes = {}
    for line in (source.parent.parent / "source/SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        hashes[name.lstrip("*")] = digest
    assert report["torque_source"] == "contact_point_normal" and not report["sim_use_gpu_pipeline"]
    assert report["physx_use_gpu"]
    for name in ("dexscrew/tasks/xhand_pasini.py", "dexscrew/tasks/contact_moments.py", "configs/task/XHandPasiniM24NutBoltE002.yaml",
                 "configs/task/XHandPasiniM24NutBolt.yaml", "configs/task/XHandPasiniNutBolt.yaml",
                 "scripts/m24_e002_a0c_diagnostics.py", "scripts/diagnose_m24_e002_a0c.py"):
        assert hashes[name] == hashlib.sha256(Path(name).read_bytes()).hexdigest(), f"task changed since A0c: {name}"
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = validate(args.source)
    if args.destination.exists():
        raise FileExistsError(args.destination)
    args.destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status="passed:training_input_import", source=str(args.source),
                         source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
                         candidate=report["selected"]["candidate_id"], force_sign=report["force_sign"])))
