#!/usr/bin/env python3
"""Replay recorded encoder feedback without importing or connecting the hand SDK."""
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--logs', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    # Preserve the deployment entry's ABI preload before Torch/NumPy imports.
    spec = importlib.util.spec_from_file_location('recorded_deploy', args.bundle/'deploy_m24_e003_student.py')
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.signal import find_peaks
    torch.set_num_threads(12)
    args.output.mkdir(parents=True, exist_ok=False)
    active = [0, 1, 2, 12, 13, 14, 15]
    reports = []
    for log in args.logs:
        meta = json.loads((log/'run_metadata.json').read_text())
        rows = [json.loads(line) for line in (log/'policy_steps.jsonl').read_text().splitlines()]
        if not rows:
            continue
        assert not meta['dry_run']
        assert abs(meta['action_scale']-.05) < 1e-7, 'Externally rescaled runs require their historical entry'
        artifact = args.bundle/'student.pt'
        assert meta['jit_policy']['sha256'] == digest(artifact)
        loop = deploy.PolicyLoop(deploy.Policy(artifact))
        errors = {key: 0. for key in ('action', 'previous_target', 'next_target')}
        predictions = []
        with torch.no_grad():
            for row in rows:
                hist = loop.history if loop.history is not None else loop.model.reset_history(loop.previous, loop.previous)
                obs = hist[:, -3:].reshape(1, 96).clone()
                result = loop.step(row['q'], row['policy_hz'])
                for key in errors:
                    errors[key] = max(errors[key], float(np.max(np.abs(result[key]-np.array(row[key])))))
                _, logits, clock, latent = loop.model.predict(obs, loop.history)
                prob = logits.softmax(-1)[0].numpy()
                z = (loop.history-loop.model.obs_mean[64:96])/torch.sqrt(loop.model.obs_var[64:96]+1e-5)
                predictions.append(dict(step=row['step'], phase=int(prob.argmax()),
                    probabilities=prob.tolist(), clock=float(clock[0, 0]),
                    latent=latent[0].tolist(), normalized_history_clipped_fraction=float((z.abs()>5).float().mean())))
        assert max(errors.values()) < 2e-6, errors
        q = np.array([r['q'] for r in rows])
        previous = np.array([r['previous_target'] for r in rows])
        targets = np.array([r['next_target'] for r in rows])
        action = np.array([r['action'] for r in rows])
        phases = np.array([r['phase'] for r in predictions])
        probs = np.array([r['probabilities'] for r in predictions])
        times = np.array([datetime.fromisoformat(r['timestamp']).timestamp() for r in rows])
        times -= times[0]
        phase_runs = []
        for i, phase in enumerate(phases):
            if not phase_runs or phase_runs[-1]['phase'] != int(phase):
                phase_runs.append(dict(phase=int(phase), start_step=rows[i]['step'], end_step=rows[i]['step']))
            else:
                phase_runs[-1]['end_step'] = rows[i]['step']
        peaks, _ = find_peaks(q[:, 0], prominence=.3, distance=25)
        windows = []
        for start in range(0, len(rows), 50):
            stop = min(start+50, len(rows))
            windows.append(dict(start_step=rows[start]['step'], end_step=rows[stop-1]['step'],
                phase_counts=np.bincount(phases[start:stop], minlength=4).tolist(),
                q_range_deg=np.degrees(np.ptp(q[start:stop], axis=0))[active].tolist(),
                target_step_mean_abs_deg=np.degrees(np.abs(targets[start:stop]-previous[start:stop]).mean(0))[active].tolist(),
                mean_phase_probabilities=probs[start:stop].mean(0).tolist(),
                mean_clock=float(np.mean([x['clock'] for x in predictions[start:stop]]))))
        last = slice(max(0, len(rows)-100), len(rows))
        lower = loop.model.lower.numpy()
        upper = loop.model.upper.numpy()
        blocked = (np.abs(targets-previous) < 1e-7) & (np.abs(action) > .02)
        report = dict(log_dir=str(log), steps=len(rows), status=meta['status'],
            artifact_sha256=digest(artifact), action_scale=meta['action_scale'],
            source_sha256={name: digest(log/name) for name in ('run_metadata.json', 'policy_steps.jsonl')},
            replay_max_errors=errors, duration_s=float(times[-1]),
            interval_ms=np.percentile(np.diff(times)*1000, [50, 95, 99, 100]).tolist() if len(rows)>1 else [],
            phase_runs=phase_runs, windows=windows,
            index0_return_peak_steps=[rows[i]['step'] for i in peaks],
            peak_detection_scope='Measured encoder extrema, not object/contact task completion',
            last100_mean_probabilities=probs[last].mean(0).tolist(),
            last100_q_range_deg=np.degrees(np.ptp(q[last], axis=0))[active].tolist(),
            last100_mean_abs_action=np.abs(action[last]).mean(0)[active].tolist(),
            last100_mean_abs_target_increment_deg=np.degrees(np.abs(targets[last]-previous[last]).mean(0))[active].tolist(),
            last100_blocked_nonzero_action_fraction=blocked[last].mean(0)[active].tolist(),
            last100_at_command_limit_fraction=((targets[last]-lower<1e-7)|(upper-targets[last]<1e-7)).mean(0)[active].tolist(),
            active_joint_order=[meta['joint_order'][i] for i in active],
            hardware_operated=False, physical_contact_or_object_motion_measured=False)
        stem = log.name
        (args.output/(stem+'.predictions.json')).write_text(json.dumps(predictions, indent=2)+'\n')
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        for ax, indices in zip(axes[:3], ([0, 12], [1, 14], [2, 13, 15])):
            for j in indices:
                ax.plot(times, np.degrees(q[:, j]), label=meta['joint_order'][j])
                ax.plot(times, np.degrees(targets[:, j]), '--', alpha=.3)
            ax.set_ylabel('Degrees')
            ax.legend(loc='upper right')
            ax.grid(alpha=.3)
        for j, label in enumerate(('drive', 'release', 'return', 'regrasp')):
            axes[3].plot(times, probs[:, j], label=label)
        axes[3].set_ylabel('Predicted probability')
        axes[3].set_xlabel('Seconds from first recorded control tick')
        axes[3].legend(loc='upper right')
        axes[3].grid(alpha=.3)
        fig.suptitle(stem+' | real encoders (solid), sent targets (dashed)\nPhase inferred offline; no measured nut/contact success')
        fig.tight_layout()
        fig.savefig(args.output/(stem+'.png'))
        plt.close(fig)
        reports.append(report)
        print(json.dumps(dict(log=stem, steps=len(rows), replay=errors,
            peaks=report['index0_return_peak_steps'], tail_probs=report['last100_mean_probabilities'])))
    (args.output/'analysis.json').write_text(json.dumps(dict(runs=reports, hardware_operated=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
