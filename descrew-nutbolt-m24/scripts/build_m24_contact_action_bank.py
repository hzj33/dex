#!/usr/bin/env python3
"""Bounded contact tangent + inward-bias proposals from validated geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def build(basis_path, amplitudes=(.1, .3, .5), normal_ratios=(.25, .5, 1., 2.),
          thumb_ratios=(.5, 1., 2.), tangent_signs=(-1, 1)):
    if not amplitudes or any(not 0 < a <= 1.0 for a in amplitudes):
        raise ValueError("stay within the existing policy action range")
    basis = json.loads(basis_path.read_text())
    assert basis["status"] == "passed:contact_kinematic_basis" and basis["pose_index"] == 743
    audit_path = Path(basis["source"])
    assert hashlib.sha256(audit_path.read_bytes()).hexdigest() == basis["source_sha256"]
    audit = json.loads(audit_path.read_text())
    kin = next(k for k in audit["kinematic_samples"] if k["phase"] == "hold")
    sample = audit["samples"][kin["sample_index"]]
    inward = -np.asarray(sample["force"])
    inward /= np.linalg.norm(inward, axis=1, keepdims=True)
    candidates = [dict(candidate_id=0, name="zero", family="zero", action=[0.0] * 16)]
    for amplitude in amplitudes:
        for normal_ratio in normal_ratios:
            for thumb_ratio in thumb_ratios:
                for sign in tangent_signs:
                    assert normal_ratio > 0 and thumb_ratio > 0 and sign in (-1, 1)
                    action = np.zeros(16)
                    for i, columns in enumerate((slice(0, 4), slice(12, 16))):
                        row = basis["fingers"][i]
                        jac = np.array(row["jacobian_contact"])
                        target = sign * np.array(row["desired_tangent"]) + normal_ratio * inward[i]
                        q = jac.T @ np.linalg.solve(jac @ jac.T + 1e-6 * np.eye(3), target)
                        q /= abs(q).max()
                        action[columns] = q * (1 if i == 0 else thumb_ratio)
                    action *= amplitude / abs(action).max()
                    candidates.append(dict(candidate_id=len(candidates),
                                           name=f"contact_tangent{sign:+d}_inward{normal_ratio:g}_thumb{thumb_ratio:g}_a{amplitude:g}",
                                           family="contact_tangent_inward", amplitude=amplitude,
                                           normal_ratio=normal_ratio, thumb_ratio=thumb_ratio,
                                           tangent_sign=sign, action=action.tolist()))
    return dict(status="prepared:contact_action_bank", a0c_status="not_evaluated", training_eligible=False,
                pose_index=743, custom_init_dof_pos=basis["custom_init_dof_pos"], thresholds=basis["thresholds"],
                basis_source=str(basis_path), basis_sha256=hashlib.sha256(basis_path.read_bytes()).hexdigest(),
                inward_directions=inward.tolist(), candidate_count=len(candidates), candidates=candidates,
                amplitudes=list(amplitudes),
                normal_ratios=list(normal_ratios), thumb_ratios=list(thumb_ratios),
                tangent_signs=list(tangent_signs),
                scope="Only candidate actions change within the existing [-1, 1] policy bounds; all acceptance thresholds remain unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("basis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--amplitudes", type=float, nargs="+", default=[.1, .3, .5])
    parser.add_argument("--normal-ratios", type=float, nargs="+", default=[.25, .5, 1., 2.])
    parser.add_argument("--thumb-ratios", type=float, nargs="+", default=[.5, 1., 2.])
    parser.add_argument("--tangent-signs", type=int, nargs="+", default=[-1, 1])
    args = parser.parse_args()
    result = build(args.basis, args.amplitudes, args.normal_ratios, args.thumb_ratios, args.tangent_signs)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(status=result["status"], candidate_count=result["candidate_count"])))
