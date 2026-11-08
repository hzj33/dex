#!/usr/bin/env python3
"""Fit a local DIP(PIP) curve on real logs, validating on the last run.

The curve is an empirical kinematic approximation, not calibrated linkage dynamics.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--logs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    runs = []
    for path in sorted(a.logs.glob('*/policy_steps.jsonl')):
        metadata = json.loads(path.with_name('run_metadata.json').read_text())
        if metadata.get('dry_run', True):
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if rows:
            runs.append((str(path), np.asarray([r['q'] for r in rows])))
    assert len(runs) >= 2, 'Need separate training and held-out real runs'
    train = np.concatenate([q for _, q in runs[:-1]])
    test = runs[-1][1]
    assert np.isfinite(train).all() and np.isfinite(test).all()
    coefficients = np.polyfit(train[:, 2], train[:, 3], 2)
    error = np.degrees(np.polyval(coefficients, test[:, 2]) - test[:, 3])
    result = dict(coefficients_descending=coefficients.tolist(),
        train_logs=[path for path, _ in runs[:-1]], heldout_log=runs[-1][0],
        train_range_rad=[float(train[:, 2].min()), float(train[:, 2].max())],
        heldout_rmse_deg=float(np.sqrt(np.mean(error**2))),
        max_heldout_error_deg=float(np.max(np.abs(error))))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
