#!/usr/bin/env python3
"""Synthetic sign/correlation/false-pass regressions; no Isaac Gym required."""
import copy
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from m24_e002_a0c_diagnostics import build_candidates, measure_step, summarize

T = dict(direction=1., contact_force_on=.3, contact_force_off=.15,
         contact_dist_on=.045, contact_dist_off=.05, torque_min=.01,
         torque_saturation=.1, pair_gate_min=.5, required_probe_streak=5,
         min_probe_angle=math.pi / 180, max_hold_drift=math.pi / 180,
         probe_steps=10, hold_steps=10)


def candidate(index, torque, delta=.03):
    c = dict(candidate_id=index, name=str(index), family="fixture", action=[0.] * 16, steps=[])
    for phase in ("hold", "probe"):
        for step in range(10):
            ds = delta / 10 if phase == "probe" else 0
            row = dict(phase=phase, step=step, finite=True, unexpected_reset=False,
                       force_magnitude=[1., 1.], distance=[.04, .04],
                       directed_angle=ds * (step + 1), directed_step=ds)
            row.update(measure_step(row["force_magnitude"], row["distance"], torque, ds, T))
            c["steps"].append(row)
    return c


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/task/XHandPasiniM24NutBoltE002.yaml").read_text())
    joint = ET.parse(root / "assets/screw/m24hex/0000_m24x3_160.urdf").find("./joint[@name='nut_joint']")
    assert joint.find("axis").get("xyz") == "0 0 -1"
    assert joint.find("origin").get("rpy") == "0 0 0"
    assert config["env"]["rotation_axis"] == "-z"
    # Separate maxima must not produce a passing candidate.
    rows = [candidate(0, [-.1, -.1], delta=0), candidate(1, [-.1, .1])]
    report = summarize(rows, T)
    assert report["selected"] is None
    assert report["top_k"]["direction"][0]["candidate_id"] == 1
    assert report["top_k_steps"]["pair_gate"][0]["candidate_id"] == 0
    assert all(r["max_streak"] == 0 for r in report["candidate_sign_summaries"])
    good = candidate(2, [-.1, -.1])
    report = summarize([copy.deepcopy(good)], T)
    assert report["status"] == "passed" and report["selected"]["force_sign"] == -1
    assert report["condition_counts"]["-1"]["all"]["step_count"] == 10
    assert report["condition_counts"]["1"]["torque"]["step_count"] == 0
    for field in ("finite", "unexpected_reset"):
        bad = copy.deepcopy(good)
        bad["steps"][0][field] = field == "unexpected_reset"
        assert summarize([bad], T)["status"] != "passed"
    broken = copy.deepcopy(good)
    broken["steps"][2]["pair_contact"] = False
    assert summarize([broken], T)["selected"] is None
    # Contact/torque pass while balance-weighted gate fails.
    r = measure_step([1, 1], [.04, .04], [-.1, -.01], .002, T)["force_signs"]["-1"]
    assert r["conditions"]["contact"] and r["conditions"]["torque"]
    assert not r["conditions"]["balance"] and not r["conditions"]["all"]
    candidates = build_candidates([.025, .05, .1], [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert len(candidates) == 67
    assert sum(c["family"] == "single_joint" for c in candidates) == 48
    assert sum(c["family"] == "coupled_tangent" for c in candidates) == 18
    assert all(not any(c["action"][4:12]) and max(map(abs, c["action"])) <= .1 for c in candidates)
    print("A0c diagnostic sign/correlation/reset/hold/gate/action checks: OK")


if __name__ == "__main__":
    main()
