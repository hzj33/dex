"""Offline A0c trace scoring. No simulator, reward, or training side effects."""

import copy
import math

import numpy as np

ACTIVE_DOFS = tuple(range(4)) + tuple(range(12, 16))
FORCE_SIGNS = (-1.0, 1.0)
CONDITIONS = ("contact", "torque", "balance", "direction", "all")


def measure_step(force, distance, raw_torque, directed_step, thresholds):
    """Keep the original A0c conjunction, including its strict contact-on test.

    `balance` names the existing balance-weighted pair gate >= pair_gate_min;
    torque_balance itself has no separate acceptance threshold.
    """
    force, distance, raw_torque = map(np.asarray, (force, distance, raw_torque))
    t = thresholds
    contact = (force >= t["contact_force_on"]) & (distance <= t["contact_dist_on"])
    c = np.minimum(
        np.clip((force - t["contact_force_off"]) /
                (t["contact_force_on"] - t["contact_force_off"]), 0, 1),
        np.clip((t["contact_dist_off"] - distance) /
                (t["contact_dist_off"] - t["contact_dist_on"]), 0, 1),
    )
    signs = {}
    for sign in FORCE_SIGNS:
        signed = t["direction"] * sign * raw_torque
        positive = np.maximum(signed, 0)
        balance = float(2 * min(positive) / max(float(positive.sum()), 1e-6))
        contribution = c * np.clip(positive / t["torque_saturation"], 0, 1)
        gate = float(min(contribution) * balance)
        flags = dict(contact=bool(contact.all()),
                     torque=bool((signed >= t["torque_min"]).all()),
                     balance=bool(gate >= t["pair_gate_min"]),
                     direction=bool(directed_step > 0))
        flags["all"] = all(flags.values())
        signs[str(int(sign))] = dict(
            signed_torque=signed.tolist(), torque_balance=balance,
            contribution=contribution.tolist(), pair_gate=gate,
            torque_finger_pass=(signed >= t["torque_min"]).tolist(), conditions=flags)
    return dict(contact=contact.tolist(), pair_contact=bool(contact.all()),
                contact_confidence=c.tolist(), force_signs=signs)


def longest_streak(flags):
    best = current = 0
    for flag in flags:
        current = current + 1 if flag else 0
        best = max(best, current)
    return best


def nut_contact_step(sample, thresholds):
    """Opposing nut-only contact; never substitute net fingertip/bolt force."""
    force = np.asarray(sample["normal_contact_force_on_nut"], dtype=float)
    distance = np.asarray(sample["distance"], dtype=float)
    return bool(force.shape == (2, 3) and distance.shape == (2,)
                and np.isfinite(force).all() and np.isfinite(distance).all()
                and (np.linalg.norm(force, axis=-1) >= thresholds["contact_force_on"]).all()
                and (distance <= thresholds["contact_dist_on"]).all())


def summarize(candidates, thresholds, top_k=5, acceptance_mode="strict_normal_proxy"):
    """Aggregate only correlated candidate/sign rows; never join global maxima."""
    if acceptance_mode not in ("strict_normal_proxy", "contact_progress"):
        raise ValueError("unknown A0c acceptance mode")
    rows = []
    for candidate in candidates:
        probe = [s for s in candidate["steps"] if s["phase"] == "probe"]
        hold = [s for s in candidate["steps"] if s["phase"] == "hold"]
        valid = (len(probe) == thresholds["probe_steps"] and
                 len(hold) == thresholds["hold_steps"] and
                 all(s["finite"] and not s["unexpected_reset"] for s in candidate["steps"]))
        hold_drift = abs(hold[-1]["directed_angle"]) if hold else math.inf
        hold_contacts = [nut_contact_step(s, thresholds) if acceptance_mode == "contact_progress"
                         else s["pair_contact"] for s in hold]
        hold_ok = (valid and all(hold_contacts) and
                   hold_drift <= thresholds["max_hold_drift"])
        candidate["hold_summary"] = dict(
            passed=hold_ok, nut_drift=hold_drift,
            contact_steps=sum(s["pair_contact"] for s in hold),
            force_min=np.min([s["force_magnitude"] for s in hold], axis=0).tolist(),
            force_max=np.max([s["force_magnitude"] for s in hold], axis=0).tolist(),
            distance_max=np.max([s["distance"] for s in hold], axis=0).tolist())
        for sign in FORCE_SIGNS:
            sign_key = str(int(sign))
            counts = {key: sum(s["force_signs"][sign_key]["conditions"][key] for s in probe)
                      for key in CONDITIONS}
            streak = longest_streak([s["force_signs"][sign_key]["conditions"]["all"] for s in probe])
            delta = probe[-1]["directed_angle"]
            gate_values = [s["force_signs"][sign_key]["pair_gate"] for s in probe]
            torque_values = [min(s["force_signs"][sign_key]["signed_torque"]) for s in probe]
            balanced = [s["force_signs"][sign_key]["torque_balance"] for s in probe]
            passed = (hold_ok and streak >= thresholds["required_probe_streak"] and
                      delta >= thresholds["min_probe_angle"])
            contact_streak = None
            contact_progress = None
            if acceptance_mode == "contact_progress":
                flags = [nut_contact_step(s, thresholds) and s["directed_step"] > 0 for s in probe]
                contact_streak = longest_streak(flags)
                # Only new highs established during actual pair contact count.
                # Rotation before contact or after slipping is never relabelled.
                high = 0.0
                contact_progress = 0.0
                for sample, flag in zip(probe, flags):
                    new_high = max(high, sample["directed_angle"])
                    if flag:
                        contact_progress += new_high - high
                    high = new_high
                passed = (hold_ok and contact_streak >= thresholds["required_probe_streak"]
                          and contact_progress >= thresholds["min_probe_angle"]
                          and delta >= thresholds["min_probe_angle"])
            rows.append(dict(
                candidate_id=candidate["candidate_id"], name=candidate["name"],
                family=candidate["family"], action=candidate["action"], force_sign=sign,
                valid=valid, hold_passed=hold_ok, condition_counts=counts,
                max_streak=streak, directed_delta=delta,
                net_angle_pass=bool(delta >= thresholds["min_probe_angle"]),
                mean_pair_gate=float(np.mean(gate_values)), max_pair_gate=max(gate_values),
                max_min_signed_torque=max(torque_values),
                mean_min_signed_torque=float(np.mean(torque_values)),
                max_torque_balance=max(balanced), passed=passed))
            if acceptance_mode == "contact_progress":
                rows[-1].update(contact_direction_streak=contact_streak,
                                nut_contact_new_progress=contact_progress)
            if "action_scale_schedule" in candidate:
                rows[-1]["action_scale_schedule"] = candidate["action_scale_schedule"]
    counts = {}
    for sign in FORCE_SIGNS:
        subset = [r for r in rows if r["force_sign"] == sign]
        counts[str(int(sign))] = {
            k: dict(step_count=sum(r["condition_counts"][k] for r in subset),
                    candidate_any_count=sum(r["condition_counts"][k] > 0 for r in subset),
                    candidate_all_steps_count=sum(r["condition_counts"][k] == thresholds["probe_steps"] for r in subset))
            for k in CONDITIONS}
        counts[str(int(sign))]["passed_candidate_count"] = sum(r["passed"] for r in subset)
    rankings = {
        "contact": lambda r: r["condition_counts"]["contact"],
        "torque": lambda r: r["max_min_signed_torque"],
        "balance": lambda r: r["max_torque_balance"],
        "pair_gate": lambda r: r["max_pair_gate"],
        "mean_pair_gate": lambda r: r["mean_pair_gate"],
        "direction": lambda r: r["directed_delta"],
        "all_conditions": lambda r: (r["max_streak"], r["condition_counts"]["all"]),
    }
    top = {key: sorted(rows, key=score, reverse=True)[:top_k] for key, score in rankings.items()}
    # Each metric also retains the exact associated time sample, not another maximum.
    best_steps = {}
    for metric in ("pair_gate", "torque_balance", "min_signed_torque", "directed_step"):
        samples = []
        for c in candidates:
            for s in c["steps"]:
                if s["phase"] != "probe":
                    continue
                for sign_key, signed in s["force_signs"].items():
                    value = (min(signed["signed_torque"]) if metric == "min_signed_torque" else
                             s["directed_step"] if metric == "directed_step" else signed[metric])
                    samples.append(dict(candidate_id=c["candidate_id"], name=c["name"],
                                        action=s["action"] if "action" in s else c["action"], force_sign=float(sign_key),
                                        metric_value=value, sample=s))
        best_steps[metric] = sorted(samples, key=lambda s: s["metric_value"], reverse=True)[:top_k]
    passed = sorted([r for r in rows if r["passed"]],
                    key=lambda r: ((r["nut_contact_new_progress"], r["contact_direction_streak"])
                                   if acceptance_mode == "contact_progress" else
                                   (r["max_streak"], r["directed_delta"])), reverse=True)
    return dict(status="passed" if passed else "failed:no_pair_positive_drive",
                condition_counts=counts, candidate_sign_summaries=rows,
                top_k=top, top_k_steps=best_steps, selected=copy.deepcopy(passed[0]) if passed else None)


def build_candidates(amplitudes, tangent_basis):
    """Single joints first, then Jacobian-derived index/thumb tangent coupling."""
    tangent_basis = np.asarray(tangent_basis, dtype=float)
    candidates = []

    def add(name, family, action):
        candidates.append(dict(candidate_id=len(candidates), name=name, family=family,
                               action=np.asarray(action).tolist(), steps=[]))

    add("zero", "zero", np.zeros(16))
    for amplitude in amplitudes:
        for joint in ACTIVE_DOFS:
            for sign in (-1, 1):
                action = np.zeros(16)
                action[joint] = sign * amplitude
                add(f"joint{joint}_{sign:+d}_a{amplitude:g}", "single_joint", action)
    for amplitude in amplitudes:
        for ratio in (0.5, 1.0, 2.0):
            for sign in (-1, 1):
                action = np.zeros(16)
                action[:4] = tangent_basis[0]
                action[12:16] = ratio * tangent_basis[1]
                action *= sign * amplitude / max(float(np.abs(action).max()), 1e-12)
                add(f"tangent_thumb_ratio{ratio:g}_{sign:+d}_a{amplitude:g}",
                    "coupled_tangent", action)
    return candidates
