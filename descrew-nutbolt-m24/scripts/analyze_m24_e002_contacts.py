#!/usr/bin/env python3
"""Offline normal-contact wrench audit, explicitly NOT an A0c acceptance gate."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from m24_e002_a0c_diagnostics import CONDITIONS, measure_step


def xyz(value):
    return np.array([value[k] for k in ("x", "y", "z")], dtype=float)


def rotate(q, vector):
    """Isaac Gym quaternion order is xyzw; normalize float32 recorded states."""
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    vector = np.asarray(vector, dtype=float)
    return vector + 2 * np.cross(q[:3], np.cross(q[:3], vector) + q[3] * vector)


def normal_wrench(sample, pair_handles, nut_handle):
    """Reconstruct API normal forces, validating their sign against the tensor.

    Empirically in these PhysX records +lambda*normal acts on body0. No dt
    division is applied: lambda is in N. Friction is NOT reconstructed.
    Torque uses the nut-side contact feature, transformed to world coordinates.
    """
    states = np.asarray(sample["rigid_body_state"])
    force = np.zeros((len(states), 3))
    nut_forces = np.zeros((2, 3))
    nut_torques = np.zeros((2, 3))
    active_pairs = Counter()
    contact_points = []
    for c in sample["contacts"]:
        b0, b1 = c["body0"], c["body1"]
        f0 = c["lambda"] * xyz(c["normal"])
        if b0 >= 0:
            force[b0] += f0
        if b1 >= 0:
            force[b1] -= f0
        if c["lambda"] > 1e-8:
            active_pairs[str(tuple(sorted((b0, b1))))] += 1
        if nut_handle not in (b0, b1):
            continue
        finger = b1 if b0 == nut_handle else b0
        if finger not in pair_handles:
            continue
        index = pair_handles.index(finger)
        side = 0 if b0 == nut_handle else 1
        fnut = f0 if side == 0 else -f0
        arm = rotate(states[nut_handle, 3:7], xyz(c[f"localPos{side}"]))
        torque = np.cross(arm, fnut)
        nut_forces[index] += fnut
        nut_torques[index] += torque
        if c["lambda"] > 1e-8:
            contact_points.append(dict(finger_index=index,
                                       point_world=(states[nut_handle, :3] + arm).tolist(),
                                       force_on_nut=fnut.tolist(), torque_on_nut=torque.tolist()))
    axis = np.asarray(sample["rotation_axis"])
    residual = force - np.asarray(sample["all_body_contact_force"])
    return dict(normal_force_by_body=force.tolist(),
                normal_force_on_nut_by_finger=nut_forces.tolist(),
                normal_torque_on_nut_by_finger=nut_torques.tolist(),
                normal_axis_torque_on_nut=(nut_torques @ axis).tolist(),
                normal_tensor_max_abs_residual=float(np.abs(residual).max()),
                active_contact_pairs=dict(active_pairs), active_contact_points=contact_points)


def phase_comparison(reference, contact):
    result = {}
    fields = ("force_magnitude", "distance", "nut_angle", "raw_axis_torque", "joint_position")
    for phase in ("hold_settle", "hold", "probe_settle", "probe"):
        pairs = [(a, b) for a, b in zip(reference["samples"], contact["samples"])
                 if a["phase"] == phase]
        result[phase] = {field: float(max(np.max(np.abs(np.asarray(a[field]) - np.asarray(b[field])))
                                         for a, b in pairs)) for field in fields}
    return result


def analyze(pipeline):
    summaries, derived, hashes = [], [], {}
    for path in sorted((pipeline / "manifests").glob("rigid_contacts_candidate*.json")):
        d = json.loads(path.read_text())
        ref_path = path.with_name(path.name.replace("rigid_contacts", "tensor_reference"))
        ref = json.loads(ref_path.read_text())
        for key in ("candidate_id", "action", "custom_init_dof_pos", "thresholds", "source_sha256"):
            assert d[key] == ref[key], key
        assert len(d["samples"]) == len(ref["samples"]) == 320
        for p in (path, ref_path):
            hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
        for a, b in zip(d["samples"], ref["samples"]):
            assert all(a[k] == b[k] for k in ("phase", "policy_step", "physics_substep", "action"))
            assert not a["unexpected_reset"] and not b["unexpected_reset"]
        rows, active, frictions, lambdas = [], Counter(), [], []
        for sample in d["samples"]:
            normal = normal_wrench(sample, d["pair_handles"], d["nut_handle"])
            active.update(normal["active_contact_pairs"])
            for c in sample["contacts"]:
                frictions.append(c["friction"])
                lambdas.extend(c["lambdaFriction"].values())
            tau = np.asarray(normal["normal_axis_torque_on_nut"])
            # Use identical force_sign semantics to the old fingertip-force proxy.
            score = measure_step(sample["force_magnitude"], sample["distance"], -tau,
                                 sample["directed_step"], d["thresholds"])
            proxy = np.asarray(sample["force_signs"]["-1"]["signed_torque"])
            directed_normal = d["thresholds"]["direction"] * tau
            nonzero = (np.abs(proxy) > 1e-6) & (np.abs(directed_normal) > 1e-6)
            row = dict(candidate_id=d["candidate_id"], phase=sample["phase"],
                       policy_step=sample["policy_step"], physics_substep=sample["physics_substep"],
                       nut_angle=sample["nut_angle"], proxy_signed_torque_minus1=proxy.tolist(),
                       normal_signed_torque_minus1=directed_normal.tolist(),
                       proxy_normal_sign_disagreement=(nonzero & (proxy * directed_normal < 0)).tolist(),
                       normal_only_conditions=score, **normal)
            rows.append(row)
        probe = [r for r in rows if r["phase"] == "probe"]
        endpoints = [r for r in probe if r["physics_substep"] == 9]
        counts = {sign: {k: sum(r["normal_only_conditions"]["force_signs"][sign]["conditions"][k]
                               for r in probe) for k in CONDITIONS} for sign in ("-1", "1")}
        summaries.append(dict(candidate_id=d["candidate_id"], name=d["candidate_name"], action=d["action"],
                              total_samples=len(rows), probe_physics_samples=len(probe),
                              normal_tensor_max_abs_residual=max(r["normal_tensor_max_abs_residual"] for r in rows),
                              active_contact_pairs=dict(active),
                              shape_friction_values=[sorted(set(p["friction"] for p in actor))
                                                     for actor in d["actor_shape_properties"]],
                              raw_contact_friction_values=sorted(set(frictions)),
                              raw_lambda_friction_max_abs=max(map(abs, lambdas)),
                              proxy_normal_probe_sign_disagreement=np.sum([r["proxy_normal_sign_disagreement"] for r in probe], axis=0).tolist(),
                              proxy_normal_probe_endpoint_sign_disagreement=np.sum([r["proxy_normal_sign_disagreement"] for r in endpoints], axis=0).tolist(),
                              max_normal_pair_gate={s: max(r["normal_only_conditions"]["force_signs"][s]["pair_gate"] for r in probe) for s in ("-1", "1")},
                              normal_only_probe_physics_condition_counts=counts,
                              paired_tensor_path_max_abs_difference=phase_comparison(ref, d),
                              last_sample=rows[-1]))
        derived.extend(rows)
    if not summaries:
        raise ValueError("no completed audit pairs")
    return dict(status="completed:normal_contact_audit", diagnostic_only=True,
                a0c_status="not_evaluated", training_eligible=False,
                total_contact_samples=len(derived), paired_simulation_samples=2 * len(derived),
                source_hashes=hashes, candidate_summaries=summaries, samples=derived,
                limitations=[
                    "Only the normal wrench is reconstructed. Zero reported friction fields do not establish zero physical friction.",
                    "Both modes retain GPU PhysX but CPU/GPU tensor paths differ; report paired differences instead of treating trajectories as identical.",
                    "Normal-only condition counts use individual physics endpoint angle increments, not the policy-step A0c acceptance test.",
                    "No full per-finger contact wrench or A0c pass is established; reward/gates must not substitute this normal-only quantity for total torque."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pipeline", type=Path)
    args = parser.parse_args()
    report = analyze(args.pipeline)
    output = args.pipeline / "manifests" / "contact_audit_summary.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "a0c_status", "training_eligible", "total_contact_samples")}))
    for row in report["candidate_summaries"]:
        print(json.dumps({k: row[k] for k in ("candidate_id", "name", "normal_tensor_max_abs_residual", "proxy_normal_probe_sign_disagreement", "max_normal_pair_gate")}))
