#!/usr/bin/env python3
"""Summarize one-env return traces without treating phase predictions as events."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(path):
    data = json.loads(path.read_text())
    assert data['summary']['num_envs'] == 1
    rows = data['first_episode_steps']
    phase = np.array([r['phase_before'] for r in rows])
    predicted = np.array([r.get('predicted_phase', -1) for r in rows])
    error = np.degrees([r['return_error'] for r in rows])
    force = np.array([r['force'] for r in rows])
    events = [{k: r[k] for k in ('step', 'last_event', 'cycles')}
              for r in rows if r['last_event']]
    report = dict(path=str(path), checkpoint_sha256=data['checkpoint_sha256'],
                  summary=data['summary'], events=events,
                  final={k: rows[-1][k] for k in
                         ('step', 'cycles', 'last_reason', 'return_error', 'force')},
                  prediction_counts=[int((predicted == i).sum()) for i in range(4)],
                  return_phase={})
    mask = phase == 2
    if mask.any():
        report['return_phase'] = dict(
            min_error_degrees=float(error[mask].min()),
            final_error_degrees=float(error[mask][-1]),
            off_samples=int(((force[mask] < .15).all(-1)).sum()),
            prediction_counts=[int((predicted[mask] == i).sum()) for i in range(4)])
    for key in ('observed_previous_target_max_error_rad', 'deployment_target_max_error_rad'):
        if key in rows[0]:
            report[key] = max(r[key] for r in rows)
    archive = path.with_suffix('.physics.npz')
    if archive.exists():
        a = np.load(archive)
        valid = a['alive'].astype(bool)
        f = np.linalg.norm(a['nut_normal'], axis=-1)[valid]
        report['physics_peak_N'] = f.max(0).tolist()
        report['physics_force5_pass'] = bool((f <= 5).all())
    return report, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('reports', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(args.reports), 3, figsize=(14, 3 * len(args.reports)), squeeze=False)
    reports = []
    for i, path in enumerate(args.reports):
        report, rows = summarize(path)
        reports.append(report)
        time = np.array([r['step'] for r in rows]) * .05
        axes[i, 0].step(time, [r['phase_before'] for r in rows], label='Physical scoring phase')
        if 'predicted_phase' in rows[0]:
            axes[i, 0].step(time, [r['predicted_phase'] for r in rows], alpha=.65, label='Predicted phase')
        axes[i, 0].set(yticks=range(4), yticklabels=['drive', 'release', 'return', 'regrasp'])
        axes[i, 1].plot(time, np.degrees([r['return_error'] for r in rows]))
        axes[i, 1].axhline(5, color='gray', ls='--')
        axes[i, 1].set_ylabel('Return error (degrees)')
        force = np.array([r['force'] for r in rows])
        for j, name in enumerate(['index', 'thumb']):
            axes[i, 2].plot(time, force[:, j], label=name)
        axes[i, 2].set_ylabel('20 Hz nut normal force (N)')
        axes[i, 0].set_title(str(path.parent).split('20260920/')[-1])
        for ax in axes[i]:
            ax.set_xlabel('Time (s)')
            ax.grid(alpha=.2)
        axes[i, 0].legend(fontsize=7)
        axes[i, 2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output/'return_comparison.png', dpi=140)
    plt.close(fig)
    (args.output/'summary.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps([{k: r[k] for k in ('path', 'final', 'return_phase')} for r in reports], indent=2))


if __name__ == '__main__':
    main()
