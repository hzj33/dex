#!/usr/bin/env python3
"""Validate Gym Jacobian reference using URDF FK, then propose contact tangents.

This changes only fixed-pose diagnostic action proposals, never reward/gates.
The measured contact centroid is a kinematic target, not a guessed torque arm.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from analyze_m24_e002_contacts import rotate, xyz


def axis_rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)


def urdf_frames(urdf, states, joint_positions):
    """Independent geometric FK/Jacobian, not computed from Gym's Jacobian."""
    names = [f"right_{f}_joint_{j}" for f in ("index", "middle", "ring", "thumb") for j in range(4)]
    values = dict(zip(names, joint_positions))
    root = np.eye(4)
    root[:3, :3] = np.stack([rotate(states[0, 3:7], v) for v in np.eye(3)], axis=1)
    root[:3, 3] = states[0, :3]
    frames, axes, anchors = {"right_palm_link": root}, {}, {}
    joints = list(ET.parse(urdf).getroot().findall("joint"))
    while joints:
        changed = False
        for joint in joints[:]:
            parent = joint.find("parent").get("link")
            if parent not in frames:
                continue
            origin = joint.find("origin")
            rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ")
            transform = np.eye(4)
            transform[:3, :3] = (axis_rotation([0, 0, 1], rpy[2]) @
                                axis_rotation([0, 1, 0], rpy[1]) @
                                axis_rotation([1, 0, 0], rpy[0]))
            transform[:3, 3] = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ")
            transform = frames[parent] @ transform
            name = joint.get("name")
            if joint.get("type") != "fixed":
                if joint.get("type") not in ("revolute", "continuous"):
                    raise ValueError("expected this 16-DOF rotational hand")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
                axes[name] = transform[:3, :3] @ axis
                anchors[name] = transform[:3, 3].copy()
                motion = np.eye(4)
                motion[:3, :3] = axis_rotation(axis, values[name])
                transform = transform @ motion
            frames[joint.find("child").get("link")] = transform
            joints.remove(joint)
            changed = True
        if not changed:
            raise ValueError("unmatched hand tree")
    return names, frames, axes, anchors


def build(audit_path, urdf_path):
    d = json.loads(audit_path.read_text())
    assert d["pose_index"] == 743 and d["candidate_id"] == 0 and not any(d["action"])
    assert not d.get("no_contact_control", False)
    kin = next(k for k in d["kinematic_samples"] if k["phase"] == "hold")
    sample = d["samples"][kin["sample_index"]]
    assert sample["pair_contact"] and not sample["unexpected_reset"]
    states = np.array(sample["rigid_body_state"])
    names, frames, axes, anchors = urdf_frames(urdf_path, states, sample["joint_position"])
    rows, basis = [], []
    for index, handle in enumerate(d["pair_handles"]):
        columns = slice(0, 4) if index == 0 else slice(12, 16)
        joint_names = names[columns]
        jac = np.array(kin["pair_jacobian"])[index][:, columns]
        frame = frames[d["body_names"][handle]]
        com_local = np.array(d["body_com"][handle]["position"])
        com_world = frame[:3, 3] + frame[:3, :3] @ com_local
        analytic = np.stack([np.cross(axes[n], com_world - anchors[n]) for n in joint_names], axis=1)
        angular = np.stack([axes[n] for n in joint_names], axis=1)
        errors = dict(fk_origin_max_abs=float(abs(frame[:3, 3] - states[handle, :3]).max()),
                      jacobian_com_max_abs=float(abs(analytic - jac[:3]).max()),
                      jacobian_angular_max_abs=float(abs(angular - jac[3:]).max()))
        assert errors["fk_origin_max_abs"] < 1e-6
        assert errors["jacobian_com_max_abs"] < 1e-6
        assert errors["jacobian_angular_max_abs"] < 1e-5
        points, weights = [], []
        for contact in sample["contacts"]:
            if {contact["body0"], contact["body1"]} != {handle, d["nut_handle"]} or contact["lambda"] <= 0:
                continue
            side = 0 if contact["body0"] == handle else 1
            points.append(xyz(contact[f"localPos{side}"]))
            weights.append(contact["lambda"])
        if not points:
            raise ValueError("missing measured finger-nut contact point")
        local_point = np.average(points, weights=weights, axis=0)
        arm = rotate(states[handle, 3:7], local_point - com_local)
        world_point = states[handle, :3] + rotate(states[handle, 3:7], local_point)
        contact_jac = jac[:3] + np.cross(jac[3:].T, arm).T
        tangent = d["thresholds"]["direction"] * np.cross(sample["rotation_axis"], world_point - sample["nut_position"])
        tangent /= np.linalg.norm(tangent)
        action = contact_jac.T @ np.linalg.solve(contact_jac @ contact_jac.T + 1e-6 * np.eye(3), tangent)
        action /= abs(action).max()
        predicted = contact_jac @ action
        cosine = float(predicted @ tangent / np.linalg.norm(predicted))
        assert cosine > 0.99 and np.isfinite(action).all()
        basis.append(action.tolist())
        rows.append(dict(body=d["body_names"][handle], handle=handle, **errors,
                         local_contact_centroid=local_point.tolist(), com_local=com_local.tolist(),
                         world_contact_centroid=world_point.tolist(), active_contact_count=len(points),
                         jacobian_com=jac[:3].tolist(), jacobian_angular=jac[3:].tolist(),
                         jacobian_contact=contact_jac.tolist(), desired_tangent=tangent.tolist(),
                         normalized_action=action.tolist(), predicted_contact_motion=predicted.tolist(),
                         tangent_cosine=cosine, damping=0.001))
    return dict(status="passed:contact_kinematic_basis", a0c_status="not_evaluated", training_eligible=False,
                source=str(audit_path), source_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
                urdf=str(urdf_path), urdf_sha256=hashlib.sha256(urdf_path.read_bytes()).hexdigest(),
                pose_index=743, custom_init_dof_pos=d["custom_init_dof_pos"],
                thresholds=d["thresholds"], basis=basis, fingers=rows,
                scope="Only action proposal changes. No contact-torque source, reward, threshold or physics change.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--urdf", type=Path, default=Path("assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf"))
    args = parser.parse_args()
    report = build(args.audit, args.urdf)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "a0c_status", "basis")}))
