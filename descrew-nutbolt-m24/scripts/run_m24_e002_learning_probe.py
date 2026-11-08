#!/usr/bin/env python3
"""Bounded smoke, warm-start PPO and fixed evaluation; no automatic long run.

Run inside the existing IsaacGym environment. The caller supplies an executed
A0c manifest, location and capacity; every phase has its own timeout and log.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from import_m24_a0c_for_training import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--a0c', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--location', choices=['local', 'cloud'], required=True)
    parser.add_argument('--num-envs', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--completed-smoke', type=Path,
                        help='Reuse a verified smoke of identical A0c and starting weights after a launcher-only fix')
    parser.add_argument('--cycle-preflight', type=Path,
                        help='Enable cycle learning after an actual finite first-drive/phase-switch probe')
    parser.add_argument('--release-preflight', type=Path,
                        help='Actual policy evaluation from the selected demonstration reset states')
    parser.add_argument('--return-preflight', type=Path, help='Actual policy evaluation from released return seeds')
    parser.add_argument('--regrasp-preflight', type=Path, help='Actual policy evaluation from measured returned/regrasp states')
    parser.add_argument('--release-seeds', default='40,43')
    parser.add_argument('--demo-fraction', type=float, default=1.0)
    parser.add_argument('--demo-source', type=Path, help='Explicit measured state archive override, SHA256 pinned in commands')
    parser.add_argument('--learning-rate', type=float,
                        help='Explicit initial PPO learning rate for an evidenced bounded comparison')
    parser.add_argument('--release-potential', choices=['saturating_absolute_force','initial_relative_force'],
                        help='Explicit opt-in cycle release-potential comparison')
    args = parser.parse_args()
    assert sum(bool(x) for x in (args.cycle_preflight,args.release_preflight,args.return_preflight,args.regrasp_preflight)) <= 1
    demo_preflight = args.regrasp_preflight or args.return_preflight or args.release_preflight
    demo_task = ('XHandPasiniM24NutBoltE002Regrasp' if args.regrasp_preflight else
                 ('XHandPasiniM24NutBoltE002Return' if args.return_preflight else 'XHandPasiniM24NutBoltE002Release'))
    if demo_preflight:
        args.cycle_preflight = demo_preflight
    seeds = [int(x) for x in args.release_seeds.split(',')]
    assert args.demo_source is None or demo_preflight
    demo_sha = hashlib.sha256(args.demo_source.read_bytes()).hexdigest() if args.demo_source else None
    assert 0 <= args.demo_fraction <= 1
    report = validate(args.a0c)
    assert report['acceptance_mode'] == 'contact_progress'
    assert args.checkpoint.is_file() and args.num_envs > 0
    assert args.learning_rate is None or 0 < args.learning_rate <= .01
    assert args.release_potential is None or args.cycle_preflight
    if args.cycle_preflight:
        preflight = json.loads(args.cycle_preflight.read_text())
        assert preflight['summary']['finite']
        if demo_preflight:
            assert preflight['summary']['initialization'] == 'demonstration_state'
            assert preflight['summary']['action_mode'] == 'policy'
            assert set(map(int,preflight['summary']['per_seed'])) - {-1} == set(seeds)
        else:
            assert preflight['summary']['first_drive'] > 0
        assert preflight['checkpoint_sha256'] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
        hashes = dict((line.split(maxsplit=1)[1], line.split(maxsplit=1)[0])
                      for line in (args.a0c.parent.parent/'source/SHA256SUMS').read_text().splitlines())
        for name in ('dexscrew/tasks/e002_cycle.py','configs/task/XHandPasiniM24NutBoltE002Cycle.yaml'):
            assert hashes[name] == hashlib.sha256(Path(name).read_bytes()).hexdigest()
        if demo_preflight:
            for name in ('dexscrew/tasks/e002_demo_resets.py','configs/task/'+demo_task+'.yaml'):
                assert hashes[name] == hashlib.sha256(Path(name).read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    p = args.output
    start = datetime.now().astimezone().isoformat()
    state = dict(started_at=start, execution_location=args.location, goal_completed=False,
                 learning_stage='regrasp' if args.regrasp_preflight else 'return' if args.return_preflight else ('release' if args.release_preflight else ('cycle' if args.cycle_preflight else 'drive')),
                 status='running', checkpoint=str(args.checkpoint),
                 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 resume_semantics='policy, critic and observation/value normalization warm start; optimizer and step counter restart',
                 num_envs=args.num_envs, minibatch_size=2 * args.num_envs, phases=[])
    state['initial_learning_rate_override'] = args.learning_rate
    state['release_potential_override'] = args.release_potential
    state['demo_reset'] = dict(seed_steps=seeds,fraction=args.demo_fraction,source=str(args.demo_source) if args.demo_source else None,source_sha256=demo_sha) if demo_preflight else None

    def save():
        (p / 'workflow_status.json').write_text(json.dumps(state, indent=2) + '\n')

    def phase(name, seconds, command):
        full = ['timeout', '--signal=TERM', '--kill-after=30s', str(seconds)] + command
        (p / (name + '.command.sh')).write_text(shlex.join(full) + '\n')
        entry = dict(name=name, command=full, started_at=datetime.now().astimezone().isoformat())
        state['phases'].append(entry)
        save()
        with (p / (name + '.log')).open('w') as log:
            result = subprocess.run(full, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, DEXSCREW_SKIP_GIT_DIFF='1'))
        entry.update(exit_status=result.returncode, finished_at=datetime.now().astimezone().isoformat())
        if name in ('smoke', 'ppo'):
            entry['teacher_exit_status'] = result.returncode
        save()
        print(json.dumps(entry), flush=True)
        return result.returncode

    save()
    shutil.copy2(args.a0c, p / 'a0_calibration.json')
    (p / 'source').mkdir()
    for name in ('dexscrew/algo/ppo/ppo.py', 'scripts/train_m24_e002_bounded.py', 'scripts/evaluate_m24_e002_fixed.py',
                 'scripts/m24_e002_evaluation_contract.py', 'scripts/run_m24_e002_learning_probe.py'):
        shutil.copy2(name, p / 'source' / Path(name).name)
    if args.cycle_preflight:
        shutil.copy2(args.cycle_preflight,p/'cycle_preflight.json')
        for name in ('dexscrew/tasks/e002_cycle.py','scripts/evaluate_m24_e002_cycle.py',
                     'configs/task/XHandPasiniM24NutBoltE002Cycle.yaml',
                     'configs/train/XHandPasiniM24NutBoltE002Cycle.yaml'):
            snapshot_name = ('train_' if name.startswith('configs/train/') else '') + Path(name).name
            shutil.copy2(name,p/'source'/snapshot_name)
        if demo_preflight:
            for name in ('dexscrew/tasks/e002_demo_resets.py','configs/task/'+demo_task+'.yaml',
                         'configs/train/'+demo_task+'.yaml'):
                snapshot_name = ('train_' if name.startswith('configs/train/') else '') + Path(name).name
                shutil.copy2(name,p/'source'/snapshot_name)
    pose = '+task.env.customInitDofPos=[' + ','.join(repr(x) for x in report['custom_init_dof_pos']) + ']'
    task = 'XHandPasiniM24NutBoltE002Cycle' if args.cycle_preflight else 'XHandPasiniM24NutBoltE002'
    if demo_preflight:
        task = demo_task
    common = ['task=' + task, 'headless=True', 'seed=42',
              'sim_device=cuda:0', 'rl_device=cuda:0', 'graphics_device_id=0', 'num_threads=8',
              'wandb_activate=False', pose, 'task.env.e002_motion.force_sign=' + str(report['force_sign'])]
    if args.learning_rate is not None:
        common.append('train.ppo.learning_rate=' + str(args.learning_rate))
    if args.release_potential is not None:
        common.append('+task.env.e002_cycle.release_potential=' + args.release_potential)
    if demo_preflight:
        common += ['task.env.e002_demo_reset.seed_steps=['+','.join(map(str,seeds))+']',
                   'task.env.e002_demo_reset.fraction='+str(args.demo_fraction)]
    if args.demo_source:
        common += ['task.env.e002_demo_reset.source='+str(args.demo_source),
                   'task.env.e002_demo_reset.source_sha256='+demo_sha]
    try:
        for name, n, batch, seconds in [('smoke', 1, 12, 60), ('ppo', args.num_envs, 2 * args.num_envs, 300)]:
            if name == 'smoke' and args.completed_smoke:
                prior = json.loads((args.completed_smoke / 'workflow_status.json').read_text())
                assert prior['checkpoint_sha256'] == state['checkpoint_sha256']
                assert prior.get('learning_stage', 'drive') == state['learning_stage']
                assert prior.get('initial_learning_rate_override') == args.learning_rate
                assert prior.get('release_potential_override') == args.release_potential
                assert prior.get('demo_reset') == state['demo_reset']
                assert (args.completed_smoke / 'a0_calibration.json').read_bytes() == args.a0c.read_bytes()
                smoke = next(x for x in prior['phases'] if x['name'] == 'smoke')
                log = (args.completed_smoke / 'smoke.log').read_text(errors='replace')
                assert smoke['exit_status'] in (0, 124) and 'Agent Steps:' in log
                assert not re.search(r'^Traceback \(most recent call last\):|^.*(?:RuntimeError: CUDA out of memory|illegal memory access|Segmentation fault)', log, re.M)
                state['reused_smoke'] = dict(path=str(args.completed_smoke), phase=smoke)
                save()
                continue
            output_name = 'XHandPasiniM24NutBolt_teacher/' + p.name + '_' + name
            command = [sys.executable, '-u', 'scripts/train_m24_e002_bounded.py'] + common + [
                'train.algo=PPO', 'task.env.numEnvs=' + str(n), 'train.ppo.num_actors=' + str(n),
                'train.ppo.minibatch_size=' + str(batch), 'train.ppo.output_name=' + output_name,
                'checkpoint=' + str(args.checkpoint)]
            rc = phase(name, seconds, command)
            log = (p / (name + '.log')).read_text(errors='replace')
            if rc not in (0, 124) or 'Agent Steps:' not in log or re.search(
                    r'^Traceback \(most recent call last\):|^.*(?:RuntimeError: CUDA out of memory|illegal memory access|Segmentation fault)', log, re.M):
                raise RuntimeError(name + ' failed startup/runtime validation')
            if name == 'ppo':
                checkpoints = Path('outputs') / output_name / 'stage1_nn'
        best = sorted(checkpoints.glob('best_reward_*.pth'), key=lambda x: x.stat().st_mtime)
        # The completed-update snapshot always exists even if the inherited
        # best-reward tracker has not been exceeded after the reward change.
        selected = {'last': checkpoints / 'last.pth'}
        if best:
            selected['best'] = best[-1]
        state['selected_checkpoints'] = {k: str(v) for k, v in selected.items()}
        state['checkpoint_sha256s'] = {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in selected.items()}
        save()
        evaluations = {}
        for label, checkpoint in selected.items():
            out = p / ('eval_' + label) / 'evaluation.json'
            evaluator = 'scripts/evaluate_m24_e002_cycle.py' if args.cycle_preflight else 'scripts/evaluate_m24_e002_fixed.py'
            cal_key = '+cycle_calibration=' if args.cycle_preflight else '+e002_calibration='
            out_key = '+cycle_eval_out=' if args.cycle_preflight else '+e002_eval_out='
            command = [sys.executable, '-u', evaluator] + common + [
                'test=True', 'task.env.numEnvs=512', 'train.ppo.num_actors=512',
                'train.ppo.minibatch_size=1024', 'checkpoint=' + str(checkpoint),
                cal_key + str(p / 'a0_calibration.json'), out_key + str(out)]
            rc = phase('eval_' + label, 300, command)
            if not out.exists():
                raise RuntimeError(label + ' evaluation missing, exit=' + str(rc))
            evaluation = json.loads(out.read_text())
            if rc != 0 or (args.cycle_preflight and not evaluation['summary']['finite']):
                raise RuntimeError(label + ' evaluation failed integrity validation, exit=' + str(rc))
            evaluations[label] = (dict(status=evaluation['status'],summary=evaluation['summary'])
                                  if args.cycle_preflight else
                                  dict(status=evaluation['status'], metrics=evaluation['metrics'],
                                       learning_assessment=evaluation['learning_assessment']))
            if demo_preflight:
                original_out = p / ('eval_' + label + '_original') / 'evaluation.json'
                original_common = [x for x in common if not x.startswith('task=') and
                                   not x.startswith('task.env.e002_demo_reset.')]
                original_cmd = [sys.executable,'-u',evaluator,'task=XHandPasiniM24NutBoltE002Cycle'] + original_common + [
                    'test=True','task.env.numEnvs=512','train.ppo.num_actors=512','train.ppo.minibatch_size=1024',
                    'checkpoint='+str(checkpoint),cal_key+str(p/'a0_calibration.json'),out_key+str(original_out)]
                original_rc = phase('eval_'+label+'_original',300,original_cmd)
                original_result = json.loads(original_out.read_text())
                assert original_rc == 0 and original_result['summary']['finite']
                assert original_result['summary']['initialization'] == 'original_candidate743'
                evaluations[label+'_original'] = dict(status=original_result['status'],summary=original_result['summary'])
        state.update(status='completed:bounded_learning_probe', evaluations=evaluations)
    except Exception as error:
        state.update(status='failed:execution', error=str(error))
        raise
    finally:
        state['finished_at'] = datetime.now().astimezone().isoformat()
        save()


if __name__ == '__main__':
    main()
