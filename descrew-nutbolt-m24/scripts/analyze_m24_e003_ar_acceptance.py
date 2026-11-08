#!/usr/bin/env python3
"""Pre-declared counterfactual acceptance for actuator-randomized students.

A retrained student passes only if its phase inference recovers a cycling
signature (with regrasp) on real-morphology streams where the delivered
student collapsed. Thresholds are fixed here BEFORE any candidate is scored:
  regrasp-estimated steps >= 15/200, first release estimate <= step 40,
  drive fraction < 0.9 on both the raw real log stream and the identified
  real actuator hybrid. Reference points: delivered student real_raw
  [86,23,107,0]/77, hybrid [110,32,58,0]/87; sim closed loop [76,51,30,43]/16.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
BUNDLE = Path('outputs/local_m24_deployment/e003_proprio_student_20260915')
IDENT = Path('outputs/local_pipeline_e003_jam_diagnostic_20260917/actuator_identification.json')
JOINTS = ['i0', 'i1', 'i2', 'i3', 'm0', 'm1', 'm2', 'm3',
          'r0', 'r1', 'r2', 'r3', 't0', 't1', 't2', 't3']
DT = 0.05
THRESH = dict(min_regrasp=15, max_first_release=40, max_drive_fraction=0.9)


def load_real():
    d = BUNDLE/'logs'/'20260915T091216_063367Z'
    rows = [json.loads(l) for l in (d/'policy_steps.jsonl').read_text().splitlines() if l.strip()]
    return np.array([r['q'] for r in rows])


class RealActuator:
    def __init__(self, ident):
        pj = ident['identification']['real']['per_joint']
        self.a = np.ones(16); self.delay = np.zeros(16, int); self.vmax = np.full(16, np.inf)
        for j, name in enumerate(JOINTS):
            f = pj.get(name)
            if f is None:
                continue
            self.a[j] = min(max(f['a_median'], 0.0), 1.0)
            self.delay[j] = f['delay_steps']
            if f['vmax_median'] is not None:
                self.vmax[j] = f['vmax_median']

    def reset(self, q0):
        self.q = q0.copy(); self.buf = [q0.copy() for _ in range(3)]

    def push_cmd(self, cmd):
        self.buf.append(cmd.copy()); self.buf.pop(0)

    def step(self):
        nxt = self.q.copy()
        for j in range(16):
            u = self.buf[len(self.buf)-1-self.delay[j]][j]
            dq = self.a[j]*(u-self.q[j])
            dq = np.clip(dq, -self.vmax[j]*DT, self.vmax[j]*DT)
            nxt[j] = self.q[j]+dq
        self.q = nxt
        return self.q.copy()


def rollout(model, stream, actuator=None, steps=200):
    with torch.no_grad():
        h = model.reset_history(model.init_pose[None], model.init_pose[None])
        prev = model.init_pose[None].clone()
        probs = []
        if actuator is not None:
            q_fb = model.init_pose.numpy().astype(np.float64).copy()
            actuator.reset(q_fb)
        for k in range(min(steps, len(stream))):
            if actuator is None:
                q_in = stream[k]
            else:
                q_in = actuator.q
            q_t = torch.tensor(np.asarray(q_in)[None], dtype=prev.dtype)
            target, h, _ = model(q_t, prev, h)
            cmd = target[0].numpy().astype(np.float64)
            obs = h[:, -3:].reshape(1, 96)
            _, logits, _, _ = model.predict(obs, h)
            probs.append(logits.softmax(-1)[0].tolist())
            prev = torch.tensor(cmd[None], dtype=prev.dtype)
            if actuator is not None:
                actuator.push_cmd(cmd)
                actuator.step()
    a = np.array(probs); phase = a.argmax(1)
    return dict(phase_counts=np.bincount(phase, minlength=4).tolist(),
                first_release_step=int(np.argmax(phase == 1)) if (phase == 1).any() else None)


def score(cond):
    counts = cond['phase_counts']; total = sum(counts)
    fr = cond['first_release_step']
    ok_regrasp = counts[3] >= THRESH['min_regrasp']
    ok_release = fr is not None and fr <= THRESH['max_first_release']
    ok_drive = counts[0]/total < THRESH['max_drive_fraction']
    return dict(regrasp_ok=bool(ok_regrasp), release_ok=bool(ok_release),
                drive_ok=bool(ok_drive), passed=bool(ok_regrasp and ok_release and ok_drive))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    model = torch.jit.load(str(args.artifact), map_location='cpu').eval()
    ident = json.loads(IDENT.read_text())
    stream = load_real()
    conditions = {
        'real_raw': rollout(model, stream),
        'hybrid_real_actuator': rollout(model, stream, actuator=RealActuator(ident)),
    }
    result = dict(artifact=str(args.artifact), thresholds=THRESH,
                  diagnostic_only=True, acceptance_scope='counterfactual_replay_signature',
                  conditions={})
    all_pass = True
    for name, cond in conditions.items():
        s = score(cond)
        result['conditions'][name] = dict(**cond, **s)
        all_pass &= s['passed']
    result['passed'] = bool(all_pass)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
