#!/usr/bin/env python3
"""Summarize paired real/sim command replays without hiding transient errors."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    a = p.parse_args()
    active = [0, 1, 2, 12, 13, 14, 15]
    results = []
    for path in sorted(a.root.glob('*/evaluation.json')):
        d = json.loads(path.read_text())
        real = [json.loads(x) for x in (Path(d['source'])/'policy_steps.jsonl').read_text().splitlines()]
        error = np.degrees(np.array([x['q_before'] for x in d['records']])-np.array([x['q'] for x in real]))
        metrics = {}
        for label, e in [('all', error), ('first80', error[:80])]:
            metrics[label] = dict(active_joint_rmse_deg=np.sqrt(np.mean(e[:, active]**2, axis=0)).tolist(),
                aggregate_active_rmse_deg=float(np.sqrt(np.mean(e[:, active]**2))),
                active_peak_error_deg=float(np.abs(e[:, active]).max()))
        results.append(dict(name=path.parent.name, source=d['source'], execution=d['execution'],
            steps=len(error), passive_coupling=d['passive_coupling'], metrics=metrics))
    out = dict(scope='Same recorded commands, free-space physics, passive index coupling; no contact calibration claim',
               joint_order=['index_0', 'index_1', 'index_2', 'thumb_0', 'thumb_1', 'thumb_2', 'thumb_3'],
               results=results)
    (a.root/'summary.json').write_text(json.dumps(out, indent=2)+'\n')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
