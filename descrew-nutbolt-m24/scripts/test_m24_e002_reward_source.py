#!/usr/bin/env python3
"""Exercise the actual task progress expressions without loading Isaac Gym."""
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np


def expressions():
    tree = ast.parse((Path(__file__).resolve().parents[1] / "dexscrew/tasks/xhand_pasini.py").read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_compute_e002_a_reward")
    names = {"invalid_quality", "invalid_forward", "forward_reward", "reverse_cost"}
    body = [n for n in method.body if
            (isinstance(n, ast.Assign) and
             any(isinstance(t, ast.Name) and t.id in names for t in n.targets)) or
            (isinstance(n, ast.If) and any(isinstance(x, ast.Attribute) and
             x.attr == 'e002_forward_source' for x in ast.walk(n.test)))]
    assert len(body) == 5
    return compile(ast.Module(body=body, type_ignores=[]), "task_progress_expressions", "exec")


def score(mode, contact, gate, new=1., reverse=0., forward="pair_quality", nut_force=(1., 1.), distance=(1., 1.)):
    scope = dict(self=SimpleNamespace(e002_invalid_forward_source=mode, e002_delta_ref=1.,
        e002_forward_source=forward, e002_contact_force_off=.15, e002_contact_force_on=.3,
        e002_normal_contact_force=np.asarray([[[x, 0., 0.] for x in nut_force]])),
        contact_potential=np.asarray(contact), pair_gate=np.asarray(gate),
        dist_conf=np.asarray([distance]),
        normalized_new=np.asarray(new), delta_reverse=np.asarray(reverse),
        torch=SimpleNamespace(clamp=lambda x, min, max: np.clip(x, min, max),
            minimum=np.minimum, norm=lambda x, dim: np.linalg.norm(x, axis=dim)))
    exec(expressions(), scope)
    return float(scope["forward_reward"] - 4 * scope["invalid_forward"] - scope["reverse_cost"])


if __name__ == "__main__":
    assert score("pair_quality", 1., .25) == -2.
    assert score("contact_quality", 1., .25) == 1.
    assert score("contact_quality", 1., 0.) == 0.  # Contact alone buys no rotation reward.
    assert score("contact_quality", 0., 0.) == -4.  # Missing opposing finger still penalized.
    assert score("contact_quality", 1., 1., new=0.) == 0.  # Static/high-water revisit.
    assert score("contact_quality", 1., 1., new=0., reverse=1.) == -4.
    assert score("contact_quality", .5, .25) == -1.  # Physical contact deficit remains costly.
    # Nut-contact comparison: positive rotation is learnable at zero normal
    # moment, while static/revisited progress and bolt-only contact earn zero.
    assert score("contact_quality", 1., 0., forward="nut_contact_quality") == 4.
    assert score("contact_quality", 1., 0., forward="nut_contact_quality", nut_force=(1., 0.)) == 0.
    assert score("contact_quality", 1., 0., forward="nut_contact_quality", distance=(1., 0.)) == 0.
    assert score("contact_quality", 1., 0., forward="nut_contact_quality", new=0.) == 0.
    assert score("contact_quality", 1., 0., forward="nut_contact_quality", new=0., reverse=1.) == -4.
    assert np.isclose(score("contact_quality", 1., 0., forward="nut_contact_quality", nut_force=(1., .225)), 2.)
    print("Actual E002 progress expressions: legacy, nut-only contact, distance, static and reverse checks passed")
