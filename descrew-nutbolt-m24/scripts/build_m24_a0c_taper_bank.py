#!/usr/bin/env python3
"""Test whether reducing late target increments prevents observed thumb loss.

Only deterministic diagnostic actions change; task/control/gates are fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(source_path, source_ids=(11, 22, 27), cutoffs=(10, 12, 14, 16, 18), tails=(0., .1, .25, .5)):
    source = json.loads(source_path.read_text())
    assert source["pose_index"] == 743 and source["torque_source"] == "contact_point_normal"
    assert source["thresholds"]["probe_steps"] == 30
    candidates = [dict(candidate_id=0, name="zero", family="zero", action=[0.] * 16)]
    for source_id in source_ids:
        parent = source["candidates"][source_id]
        for cutoff, tail in [(30, 1.)] + [(c, t) for c in cutoffs for t in tails]:
            assert 0 <= cutoff <= 30 and 0 <= tail <= 1
            candidates.append(dict(candidate_id=len(candidates),
                name=f"parent{source_id}_cut{cutoff}_tail{tail:g}", family="late_increment_taper",
                action=parent["action"], action_scale_schedule=[1.] * cutoff + [tail] * (30 - cutoff),
                source_candidate_id=source_id, cutoff=cutoff, tail_scale=tail))
    return dict(status="prepared:contact_action_bank", a0c_status="not_evaluated", training_eligible=False,
        pose_index=743, custom_init_dof_pos=source["custom_init_dof_pos"], thresholds=source["thresholds"],
        source=str(source_path), source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
        candidate_count=len(candidates), candidates=candidates,
        scope="Taper action increments before measured thumb loss; no task, controller or threshold changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-ids", type=int, nargs="+", default=[11, 22, 27])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 12, 14, 16, 18])
    parser.add_argument("--tails", type=float, nargs="+", default=[0., .1, .25, .5])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = build(args.source, args.source_ids, args.cutoffs, args.tails)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(result["candidate_count"])
