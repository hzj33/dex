#!/usr/bin/env python3
"""Deterministic A0 checks for E002-A high-water reward bookkeeping.

This intentionally has no Isaac Gym dependency so it can run before a GPU
allocation. Physics smoke still validates tensor shapes and torque/contact sign.
"""

import math


TARGET = math.pi / 6.0
DELTA_REF = math.pi / 60.0


def roll(trace):
    start = trace[0]
    previous = start
    high = 0.0
    paid = 0.0
    gross = 0.0
    reverse = 0.0
    forward_reward = 0.0
    motion_return = 0.0
    milestones = set()
    for theta in trace[1:]:
        directed = theta - start
        step = theta - previous
        previous = theta
        high = max(high, directed)
        next_paid = min(max(high, 0.0), TARGET)
        delta_new = max(next_paid - paid, 0.0)
        paid = next_paid
        delta_reverse = max(-step, 0.0)
        gross += abs(step)
        reverse += delta_reverse
        forward_reward += 4.0 * min(delta_new / DELTA_REF, 1.0)
        motion_return += 4.0 * min(delta_new / DELTA_REF, 1.0)
        motion_return -= 4.0 * min(delta_reverse / DELTA_REF, 1.0)
        for threshold in (TARGET / 2.0, TARGET):
            if high >= threshold and threshold not in milestones:
                milestones.add(threshold)
    return {
        "high": high,
        "paid": paid,
        "gross": gross,
        "reverse": reverse,
        "forward_reward": forward_reward,
        "motion_return": motion_return,
        "milestones": len(milestones),
        "net": trace[-1] - start,
    }


def roll_gated(trace, pair_quality):
    """Minimal mirror of E002 gated high-water and invalid-drive terms."""
    assert len(pair_quality) == len(trace) - 1
    start = trace[0]
    previous = start
    paid = 0.0
    high = 0.0
    reward = 0.0
    milestone_paid = False
    valid_pair_progress = 0.0
    for theta, quality in zip(trace[1:], pair_quality):
        directed = theta - start
        step = theta - previous
        previous = theta
        previous_high = high
        high = max(high, directed)
        next_paid = min(max(high, 0.0), TARGET)
        delta_new = max(next_paid - paid, 0.0)
        paid = next_paid
        normalized_new = min(delta_new / DELTA_REF, 1.0)
        if quality >= 0.5:
            valid_pair_progress += delta_new
        reward += 4.0 * quality * normalized_new
        reward -= 4.0 * (1.0 - quality) * normalized_new
        reward -= 4.0 * min(max(-step, 0.0) / DELTA_REF, 1.0)
        crossed_now = previous_high < TARGET / 2.0 <= high and delta_new > 0.0
        if quality >= 0.5 and crossed_now and not milestone_paid:
            reward += 1.0
            milestone_paid = True
    return reward, paid, milestone_paid, valid_pair_progress


def contact_lifecycle(pair_contact, stable_steps=3, loss_steps=3):
    acquired = False
    contact_streak = 0
    loss_streak = 0
    acquired_at = None
    lost_at = None
    for step, contact in enumerate(pair_contact):
        contact_streak = contact_streak + 1 if contact else 0
        if contact_streak >= stable_steps and not acquired:
            acquired = True
            acquired_at = step
        loss_streak = loss_streak + 1 if acquired and not contact else 0
        if loss_streak >= loss_steps:
            lost_at = step
            break
    return acquired_at, lost_at


def main():
    a = DELTA_REF
    forward = roll([0.0, a, 2 * a, 3 * a])
    assert forward["paid"] == 3 * a and forward["reverse"] == 0.0

    # Returning below a prior high-water cannot earn another forward increment.
    below_high = roll([0.0, a, 0.0, a])
    assert below_high["forward_reward"] == 4.0

    # E001's oscillation pattern cannot repeatedly earn progress and has
    # negative motion return because every reverse step is charged.
    oscillation = roll([0.0, a, 0.0, a, 0.0])
    assert oscillation["forward_reward"] == 4.0
    assert oscillation["motion_return"] < 0.0
    assert oscillation["net"] == 0.0

    # Milestones pay only once and reset isolates a new episode.
    milestone = roll([0.0, TARGET / 2.0, TARGET, TARGET])
    assert milestone["milestones"] == 2
    reset = roll([0.0, a])
    assert reset["paid"] == a and reset["milestones"] == 0

    # Single-finger motion is consumed by high-water bookkeeping and penalized;
    # it cannot be relabelled as valid progress or receive a delayed milestone.
    gated_reward, gated_paid, milestone_paid, valid_pair_progress = roll_gated(
        [0.0, TARGET / 2.0, TARGET / 2.0, TARGET / 2.0 + a],
        [False, True, True],
    )
    assert gated_paid == TARGET / 2.0 + a
    assert milestone_paid is False
    assert math.isclose(valid_pair_progress, a)
    assert gated_reward == 0.0
    invalid_only_reward, _, _, invalid_valid_progress = roll_gated(
        [0.0, a], [False]
    )
    assert invalid_only_reward == -4.0
    assert invalid_valid_progress == 0.0
    imbalanced_reward, _, _, _ = roll_gated([0.0, a], [0.25])
    assert imbalanced_reward < 0.0

    # Static contact without positive torque remains a valid grasp. Contact
    # loss starts only after acquisition and requires consecutive off steps.
    acquired_at, lost_at = contact_lifecycle(
        [False, True, True, True, True, False, False, False]
    )
    assert acquired_at == 3
    assert lost_at == 7
    potential_trace = [0.0, 0.5, 0.0, 0.5, 0.0]
    potential_return = sum(
        current - previous
        for previous, current in zip(potential_trace, potential_trace[1:])
    )
    assert potential_return == 0.0
    print("E002-A0 high-water/contact/gating/reset/oscillation checks: OK")


if __name__ == "__main__":
    main()
