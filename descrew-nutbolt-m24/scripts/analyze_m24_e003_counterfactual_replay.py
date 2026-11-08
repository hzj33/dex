#!/usr/bin/env python3
"""Offline counterfactual replay of the deployed student on modified joint streams.

The proprio-only student sees only joint feedback and its own previous targets,
so every real-vs-sim difference must act through the shape of those streams.
This probe rolls the frozen TorchScript student open-loop (its own targets feed
back as previous targets) over recorded streams with single-feature edits to
the index DIP (joint 3) channel, to test whether that channel alone explains
the real-hand stuck signature. No simulator, no training, no hardware.
"""
import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
bundle = Path('outputs/local_m24_deployment/e003_proprio_student_20260915')
model = torch.jit.load(str(bundle/'student.pt'), map_location='cpu').eval()
fit = json.loads(Path('outputs/local_pipeline_e003_passive_diagnostic_20260915/fit.json').read_text())
coef = fit['coefficients_descending']
out_root = Path('outputs/local_pipeline_e003_jam_diagnostic_20260917')


def coupling(q_pip):
    q = np.full_like(q_pip, coef[-1])
    for c in coef[:-1]:
        q = q*q_pip + c
    return q


def load_real():
    d = bundle/'logs'/'20260915T091216_063367Z'
    rows = [json.loads(l) for l in (d/'policy_steps.jsonl').read_text().splitlines()]
    meta = json.loads((d/'run_metadata.json').read_text())
    assert not meta.get('dry_run')
    return np.array([r['q'] for r in rows]), meta['status']


def load_sim(name):
    rows = json.loads((out_root/name/'jam_evaluation.json').read_text())['rows']
    return np.array([r['q'] for r in rows]), 'sim_'+name


def rollout(q_stream, edit):
    """edit(joint_index)->replace that feedback channel; None keeps raw."""
    with torch.no_grad():
        h = model.reset_history(model.init_pose[None], model.init_pose[None])
        prev = model.init_pose[None].clone()
        probs = []
        for i in range(len(q_stream)):
            q = torch.tensor(q_stream[i][None].copy(), dtype=prev.dtype)
            if 'i3rigid' in edit:
                q[0, 3] = float(coupling(np.array([q_stream[i][2]]))[0])
            if 'i3track' in edit:
                q[0, 3] = prev[0, 3]
            if 'alltrack' in edit:
                q[0, :] = prev[0, :]
            for tag, alpha in (('active50', 0.5), ('active25', 0.25)):
                if tag in edit:
                    mask = torch.ones(16, dtype=torch.bool)
                    mask[3] = False
                    q[0, mask] = alpha*q[0, mask] + (1-alpha)*prev[0, mask]
            obs = h[:, -3:].reshape(1, 96)
            target, h, _ = model(q, prev, h)
            _, logits, _, _ = model.predict(obs, h)
            probs.append(logits.softmax(-1)[0].tolist())
            prev = target
    a = np.array(probs)
    phase = a.argmax(1)
    spans = [dict(step=i, phase=int(v)) for i, v in enumerate(phase)
             if i == 0 or v != phase[i-1]]
    rel = phase == 1
    return dict(steps=len(phase),
                phase_counts=np.bincount(phase, minlength=4).tolist(),
                phase_spans=spans[:16],
                first_release_step=int(np.argmax(rel)) if rel.any() else None,
                mean_phase_probability=a.mean(0).tolist())


conditions = [
    ('sim_jam_raw', *load_sim('jam100x'), []),
    ('sim_jam_i3rigid', *load_sim('jam100x'), ['i3rigid']),
    ('sim_jam_i3track', *load_sim('jam100x'), ['i3track']),
    ('sim_jam_alltrack', *load_sim('jam100x'), ['alltrack']),
    ('sim_jam_active50', *load_sim('jam100x'), ['active50']),
    ('sim_jam_active25', *load_sim('jam100x'), ['active25']),
    ('real_raw', *load_real(), []),
    ('real_i3track', *load_real(), ['i3track']),
    ('real_alltrack', *load_real(), ['alltrack']),
]
result = dict(diagnostic_only=True, acceptance_eligible=False,
    reference_real_signature=dict(run='20260915T091216_063367Z', steps=216,
        phase_counts=[86, 23, 107, 0], first_release_step=78, regrasp=0),
    note='Open-loop replay: model outputs feed back as previous targets; '
         'single-channel edits replace only the named feedback channels at '
         'every step; not a physical simulation of the edits.', conditions={})
for name, q_stream, status, edit in conditions:
    r = rollout(q_stream, edit)
    r['source_status'] = status
    r['edits'] = edit
    result['conditions'][name] = r
    print(name, r['phase_counts'], 'first_release', r['first_release_step'])
(out_root/'counterfactual_replay.json').write_text(json.dumps(result, indent=2)+'\n')
