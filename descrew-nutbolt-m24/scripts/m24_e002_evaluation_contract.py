"""Separate a usable learning experiment from mature task completion.

Pure report analysis; no simulator or training side effects. Thresholds in the
historical evaluator remain comparison metrics, not prerequisites to learn.
"""
import math
import numpy as np


def assess_learning(final, traces, num_envs, fixed_steps):
    complete = (len(final) == num_envs and all(r is not None for r in final)
                and {r["env_id"] for r in final if r is not None} == set(range(num_envs)))
    finite = bool(traces) and all(np.isfinite(np.asarray(
        [r["net_angle"], r["gross_angle"], r["reverse_angle"], *r["force"],
         *r["distance"], *r["action"]], dtype=float)).all() for r in traces)
    by_env = {i: [] for i in range(num_envs)}
    for row in traces:
        if row["env_id"] not in by_env:
            complete = False
        else:
            by_env[row["env_id"]].append(row)
    complete = complete and all(
        rows and [r["step"] for r in rows] == list(range(len(rows)))
        and len(rows) <= fixed_steps
        and all(r["reason"] == 0 for r in rows[:-1])
        and (rows[-1]["reason"] != 0 or len(rows) == fixed_steps)
        and final[i]["step"] == rows[-1]["step"]
        for i, rows in by_env.items())
    hard_speed_rate = (sum(r["reason"] == 4 for r in final if r) / num_envs
                       if num_envs else 1.0)
    health = dict(complete_first_episodes=bool(complete), finite_physical_traces=bool(finite),
                  no_frequent_angle_discontinuities=hard_speed_rate < .01)
    paired_motion = []
    for rows in by_env.values():
        paired_motion.append(any(r["net_angle"] >= math.pi / 180
                                 and all(r["contact"]) for r in rows))
    healthy = all(health.values())
    return dict(status="evaluated:learning_in_progress" if healthy else "failed:execution_integrity",
                continuation_allowed=healthy,
                continuation_scope="bounded experiments within the recorded time budget; no automatic long run",
                health_checks=health,
                observed_pair_contact_1deg_envs=sum(paired_motion),
                goal_completed=False,
                goal_scope="drive only; release/return/regrasp/repeated cycles not evaluated")
