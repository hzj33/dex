#!/usr/bin/env python3
"""Identify per-joint actuator response from real logs and sim streams, then
run hybrid replay with the identified models.

The DexH13 position mode is an internal trapezoidal point-to-point motion
(SDK manual 3.5.1); its firmware limits are not queryable. Following the
standard FOPDT/black-box route (fit response from logged command/position
pairs), this fits per joint a discrete first-order law with input delay and
velocity saturation, on the six real deployment logs and, for comparison, on
the sim PD (Kp2.4) streams. It then replays the frozen student in a hybrid
loop: the policy's own targets drive the identified actuator model and the
model's output feeds back as measured joints. Command low-pass shaping is
prototyped as the deployment-side fix. No simulator, no training, no hardware.
"""
import json
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)
DT = 0.05
bundle = Path('outputs/local_m24_deployment/e003_proprio_student_20260915')
model = torch.jit.load(str(bundle/'student.pt'), map_location='cpu').eval()
out_root = Path('outputs/local_pipeline_e003_jam_diagnostic_20260917')
JOINTS = ['i0', 'i1', 'i2', 'i3', 'm0', 'm1', 'm2', 'm3',
          'r0', 'r1', 'r2', 'r3', 't0', 't1', 't2', 't3']
ACTIVE = [0, 1, 2, 3, 12, 13, 14, 15]


def fit_joint(q, cmd, vmax_grid=None):
    """q,cmd: (N,) arrays. Returns first-order gain a, delay d, vmax, R2."""
    deltas = q[1:] - q[:-1]
    best = None
    for d in (0, 1, 2):
        e = cmd[:len(deltas)-d] - q[d:d+len(deltas)-d] if d else cmd[:len(deltas)] - q[:len(deltas)]
        y = deltas[d:] if d else deltas
        e = e[:len(y)]
        # linear fit first
        denom = float((e*e).sum())
        if denom < 1e-12:
            return dict(a=1.0, delay=0, vmax=None, r2=1.0, linear=True, static=True)
        a = float((e*y).sum())/denom
        pred = a*e
        ss_res = float(((y-pred)**2).sum()); ss_tot = float((y*y).sum())
        r2 = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0
        cand = dict(a=a, delay=d, vmax=None, r2=r2, linear=True, static=False)
        if best is None or r2 > best['r2']:
            best = cand
        # saturated fit: pred = min(a*|e|, vmax)*sign(e)
        for vmax in (vmax_grid if vmax_grid is not None else ()):
            pred_sat = np.minimum(np.abs(a)*np.abs(e), vmax*DT)*np.sign(e)
            ss_res = float(((y-pred_sat)**2).sum())
            r2s = 1.0 - ss_res/ss_tot if ss_tot > 1e-12 else 1.0
            if r2s > best['r2']:
                best = dict(a=a, delay=d, vmax=float(vmax), r2=r2s,
                            linear=False, static=False)
    return best


def collect_real():
    qs, cmds = [], []
    for d in sorted((bundle/'logs').iterdir()):
        meta = json.loads((d/'run_metadata.json').read_text())
        if meta.get('dry_run'):
            continue
        rows = [json.loads(l) for l in (d/'policy_steps.jsonl').read_text().splitlines() if l.strip()]
        if not rows:
            continue
        q = np.array([r['q'] for r in rows])
        cmd = np.array([r['next_target'] for r in rows])
        qs.append(q); cmds.append(cmd)
    return qs, cmds


def collect_sim(name):
    rows = json.loads((out_root/name/'jam_evaluation.json').read_text())['rows']
    return [np.array([r['q'] for r in rows])], [np.array([r['target'] for r in rows])]


def identify(qs, cmds, label):
    per_joint = {}
    for j in ACTIVE:
        fits = [fit_joint(q[:, j], c[:, j], vmax_grid=np.arange(0.05, 3.01, 0.05))
                for q, c in zip(qs, cmds)]
        # pick median gain/delay across runs; report spread
        a = float(np.median([f['a'] for f in fits]))
        d = int(np.median([f['delay'] for f in fits]))
        vmaxs = [f['vmax'] for f in fits if f['vmax'] is not None]
        per_joint[JOINTS[j]] = dict(
            a_median=a, delay_steps=d,
            r2_median=float(np.median([f['r2'] for f in fits])),
            r2_min=float(np.min([f['r2'] for f in fits])),
            vmax_median=(float(np.median(vmaxs)) if vmaxs else None),
            n_runs=len(fits), a_runs=[round(f['a'], 4) for f in fits])
    return dict(source=label, dt_s=DT, per_joint=per_joint)


class Actuator:
    """Discrete first-order + delay + optional velocity saturation per joint."""

    def __init__(self, ident):
        pj = ident['per_joint']
        self.a = np.ones(16); self.delay = np.zeros(16, int)
        self.vmax = np.full(16, np.inf)
        for j, name in enumerate(JOINTS):
            f = pj.get(name)
            if f is None:
                continue
            self.a[j] = min(max(f['a_median'], 0.0), 1.0)
            self.delay[j] = f['delay_steps']
            if f['vmax_median'] is not None:
                self.vmax[j] = f['vmax_median']
        self.buf = None

    def reset(self, q0):
        self.q = q0.copy(); self.buf = [q0.copy() for _ in range(3)]

    def push_cmd(self, cmd):
        self.buf.append(cmd.copy()); self.buf.pop(0)

    def step(self):
        nxt = self.q.copy()
        for j in range(16):
            u = self.buf[len(self.buf)-1-int(self.delay[j])][j]
            e = u - self.q[j]
            dq = self.a[j]*e
            if abs(dq) > self.vmax[j]*DT:
                dq = np.sign(dq)*self.vmax[j]*DT
            nxt[j] = self.q[j] + dq
        self.q = nxt
        return self.q.copy()


def hybrid_rollout(act_sim, act_real, filter_tau=None, steps=200):
    """Policy in the loop; feedback generated by act_real on (possibly
    low-pass-shaped) policy targets. filter_tau: command shaping time constant."""
    with torch.no_grad():
        h = model.reset_history(model.init_pose[None], model.init_pose[None])
        prev = model.init_pose[None].clone()
        q_fb = model.init_pose.numpy().astype(np.float64).copy()
        act = act_real if act_real is not None else act_sim
        act.reset(q_fb)
        shaped = q_fb.copy()
        probs = []
        for k in range(steps):
            q_t = torch.tensor(q_fb[None], dtype=prev.dtype)
            target, h, _ = model(q_t, prev, h)
            cmd = target[0].numpy().astype(np.float64)
            if filter_tau:
                alpha = DT/(filter_tau+DT)
                shaped = shaped + alpha*(cmd-shaped)
                cmd = shaped
            act.push_cmd(cmd)
            q_fb = act.step()
            prev = torch.tensor(cmd[None], dtype=prev.dtype)
            obs = h[:, -3:].reshape(1, 96)
            _, logits, _, _ = model.predict(obs, h)
            probs.append(logits.softmax(-1)[0].tolist())
    a = np.array(probs); phase = a.argmax(1)
    return dict(phase_counts=np.bincount(phase, minlength=4).tolist(),
                first_release_step=int(np.argmax(phase == 1)) if (phase == 1).any() else None,
                mean_phase_probability=a.mean(0).tolist())


def main():
    real_qs, real_cmds = collect_real()
    sim_qs, sim_cmds = collect_sim('jam100x')
    id_real = identify(real_qs, real_cmds, 'real_dexh13_position_mode_6runs')
    id_sim = identify(sim_qs, sim_cmds, 'sim_kp2.4_pd_jam_stream')
    act_real = Actuator(id_real); act_sim = Actuator(id_sim)

    conditions = {}
    # sanity: sim PD model in hybrid loop should keep cycling
    conditions['hybrid_simPD_feedback'] = hybrid_rollout(act_sim, act_sim)
    # identified real model feedback: expect the stuck signature
    conditions['hybrid_real_feedback'] = hybrid_rollout(act_sim, act_real)
    # deployment-side command shaping preview on real model
    for tau in (0.05, 0.1, 0.2):
        conditions[f'hybrid_real_shaped_tau{tau}'] = hybrid_rollout(
            act_sim, act_real, filter_tau=tau)
    result = dict(diagnostic_only=True, acceptance_eligible=False,
        identification=dict(real=id_real, sim=id_sim),
        conditions=conditions,
        note='First-order+delay+velocity-saturation fit at 20Hz log resolution; '
             'hybrid replay is estimator-level: no contact physics. Command '
             'shaping is a deployment-side preview, not an accepted fix.')
    (out_root/'actuator_identification.json').write_text(json.dumps(result, indent=2)+'\n')
    print('== real per-joint ==')
    for j, f in id_real['per_joint'].items():
        print(f"  {j}: a={f['a_median']:.3f} delay={f['delay_steps']} "
              f"vmax={f['vmax_median']} R2med={f['r2_median']:.3f}")
    print('== sim per-joint ==')
    for j, f in id_sim['per_joint'].items():
        print(f"  {j}: a={f['a_median']:.3f} delay={f['delay_steps']} "
              f"vmax={f['vmax_median']} R2med={f['r2_median']:.3f}")
    print('== hybrid conditions ==')
    for k, v in conditions.items():
        print(f"  {k}: {v['phase_counts']} first_release={v['first_release_step']}")


if __name__ == '__main__':
    main()
