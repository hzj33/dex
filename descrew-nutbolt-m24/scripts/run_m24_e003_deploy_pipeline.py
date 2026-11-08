#!/usr/bin/env python3
"""Student-only cloud sequence. Every expensive phase has an independent timeout."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from analyze_m24_e003_loads import summarize


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--legacy-source', type=Path, required=True)
    p.add_argument('--student-seconds', type=int, default=3600)
    p.add_argument('--smoke-seconds', type=int, default=120)
    p.add_argument('--num-envs', type=int, default=64)
    p.add_argument('--execution-location', choices=('cloud','local'), default='cloud')
    p.add_argument('--provider-failure', type=Path)
    p.add_argument('--baseline-source', type=Path, help='Reuse verified, same-location teacher evaluation receipts')
    p.add_argument('--action-scale', type=float, choices=(0.02,0.05), default=0.05)
    p.add_argument('--expert-replay-fraction', type=float, default=0.0)
    p.add_argument('--warm-start', type=Path)
    a=p.parse_args(); assert a.num_envs > 0 and a.num_envs <= 512
    assert a.student_seconds > 0 and a.smoke_seconds > 0
    assert 0 <= a.expert_replay_fraction <= 1
    root=a.output; root.mkdir(parents=True,exist_ok=False)
    teacher=Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/selected_teacher/e003_compliant_frozen_teacher.pth')
    cal=Path('outputs/cloud_pipeline_m24_e003_student_20260914/a0c/manifests/a0c_diagnostic.json')
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(teacher)=='d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'
    assert a.legacy_source.is_file() and cal.is_file()
    now=lambda:datetime.now(timezone.utc).isoformat()
    status=dict(status='running:student_only',started_at=now(),execution_location=a.execution_location,
                teacher_training=False,teacher_exit_status=None,teacher_sha256=sha(teacher),
                legacy_sha256=sha(a.legacy_source),calibration_sha256=sha(cal),phases=[],
                student_seconds=a.student_seconds,smoke_seconds=a.smoke_seconds,num_envs=a.num_envs,
                action_scale=a.action_scale,policy_hz=20,external_phase_input=False,hardware_ready=False)
    status.update(expert_replay_fraction=a.expert_replay_fraction,
                  warm_start=str(a.warm_start) if a.warm_start else None,
                  warm_start_sha256=sha(a.warm_start) if a.warm_start else None)
    if a.execution_location == 'local':
        assert a.provider_failure is not None, 'Local fallback requires a recorded provider failure'
        status['provider_failure'] = json.loads(a.provider_failure.read_text())
    def save(): (root/'status.json').write_text(json.dumps(status,indent=2)+'\n')
    def phase(name, seconds, cmd, training=False):
        full=['timeout','--signal=TERM','--kill-after=30s',str(seconds)]+cmd
        row=dict(name=name,start=now(),command=full);status['phases'].append(row);save()
        (root/(name+'.command.sh')).write_text(shlex.join(full)+'\n')
        with (root/(name+'.log')).open('w') as f:
            rc=subprocess.run(full,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',DEXSCREW_SKIP_GIT_DIFF='1')).returncode
        row.update(end=now(),exit_status=rc)
        if training: row['student_exit_status']=rc
        save()
        assert rc in ((0,124) if training else (0,)), (name,rc)
    common=['task=XHandPasiniM24NutBoltE003Compliant','train=XHandPasiniM24NutBoltE003Compliant',
            'headless=True','sim_device=cuda:0','rl_device=cuda:0','graphics_device_id=0','num_threads=8',
            'wandb_activate=False','+cycle_calibration='+str(cal)]
    def evaluate(label, artifact, seed, nominal=False, student=True):
        out=root/label/'evaluation.json'
        cmd=[sys.executable,'-u','scripts/evaluate_m24_e002_cycle.py']+common+[
            'test=True','checkpoint='+str(artifact),'seed='+str(seed),'task.env.numEnvs=128',
            'train.ppo.num_actors=128','train.ppo.minibatch_size=256',
            'task.env.e002_init_joint_noise_rad='+str(0 if nominal else math.radians(1)),
            'task.env.e002_nominal_reset_fraction=0','+cycle_eval_steps=2200','+cycle_eval_out='+str(out)]
        if student:cmd+=['+cycle_eval_deploy_student=True']
        phase(label,240,cmd)
        data=json.loads(out.read_text()); assert data['summary']['finite'] and data['summary']['required_cycles']==5
        result=summarize(out); result.pop('episodes');return result
    try:
        sources=['dexscrew/algo/models/e003_deploy_student.py','scripts/train_m24_e003_deploy_student.py',
                 'scripts/m24_e003_deploy_runtime.py','scripts/deploy_m24_e003_student.py',
                 'scripts/evaluate_m24_e002_cycle.py','scripts/run_m24_e003_deploy_pipeline.py',
                 'dexscrew/algo/e003_replay.py']
        for source in sources:
            dst=root/'source'/source;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
        status['source_hashes']={x:sha(x) for x in sources}
        status['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv'],text=True)
        if a.baseline_source:
            old=json.loads((a.baseline_source/'status.json').read_text())
            assert old['execution_location']==a.execution_location and old['teacher_sha256']==sha(teacher)
            baseline=[]; receipts={}
            for seed in (42,43,44,45,46):
                report=a.baseline_source/('teacher_'+str(seed))/'evaluation.json'
                d=json.loads(report.read_text()); summary=d['summary']
                assert d['checkpoint_sha256']==sha(teacher) and d['policy_algorithm']=='PPO'
                assert summary['finite'] and summary['required_cycles']==5 and summary['num_envs']==128
                assert summary['evaluation_seed']==seed and summary['nominal_reset_fraction']==0
                assert abs(summary['initial_joint_noise_rad']-(0 if seed==42 else math.radians(1)))<1e-9
                result=summarize(report); result.pop('episodes'); baseline.append(result)
                receipts[str(report)]=sha(report)
            status['reused_teacher_receipts']=receipts
        else:
            baseline=[evaluate('teacher_'+str(seed),teacher,seed,seed==42,False) for seed in (42,43,44,45,46)]
        assert baseline[0]['successes']==128, 'Accepted teacher baseline did not reproduce'
        assert sum(r['successes'] for r in baseline[1:]) >= 468, 'Teacher robustness regressed before student training'
        status['baseline']=baseline;save()
        for name,seconds in [('smoke',a.smoke_seconds),('formal',a.student_seconds)]:
            dest=root/name
            phase(name,seconds,[sys.executable,'-u','scripts/train_m24_e003_deploy_student.py']+common+[
                'test=False','checkpoint='+str(teacher),'seed=42','task.env.numEnvs='+str(a.num_envs),
                'train.ppo.num_actors='+str(a.num_envs),'train.ppo.minibatch_size='+str(a.num_envs*2),
                '+deploy_student_out='+str(dest),'+deploy_student_action_scale='+str(a.action_scale),
                '+deploy_student_expert_replay_fraction='+str(a.expert_replay_fraction)]+
                (['+deploy_student_warm_start='+str(a.warm_start)] if a.warm_start else []),True)
            progress=json.loads((dest/'progress.json').read_text())
            assert progress['steps']>=10 and progress['teacher_unchanged']
            status[name+'_progress']=progress;save()
        candidates=[root/'formal/student_last.pt']+sorted((root/'formal').glob('candidate_*.pt'))[-2:]
        validation={}
        for i,ck in enumerate(candidates):
            validation[str(ck)]=[evaluate('candidate%d_s%d'%(i,s),ck,s,s==42) for s in (42,43,44)]
        selected=max(validation,key=lambda k:(validation[k][0]['successes'],sum(r['successes'] for r in validation[k][1:]),
                     -sum(sum(r['normal_force_peak_N']) for r in validation[k])))
        results=validation[selected]+[evaluate('selected_s%d'%s,selected,s) for s in (45,46)]
        # Student transfer acceptance: original five-cycle task, no hidden oracle,
        # preserve measured reliability and allow at most 10% force/slip regression.
        counts=results[0]['successes']==128 and sum(r['successes'] for r in results[1:])>=sum(r['successes'] for r in baseline[1:])
        force=all(results[j]['normal_force_peak_N'][i]<=baseline[j]['normal_force_peak_N'][i]*1.1 for j in range(5) for i in (0,1))
        slip=all(results[j]['mean_thumb_loss_s']<=baseline[j]['mean_thumb_loss_s']*1.1+0.05 for j in range(5))
        status.update(selected_candidate=selected,validation=validation,selected_results=results,
                      acceptance=dict(five_cycle_reliability=counts,normal_force=force,thumb_slip=slip))
        if counts and force and slip:
            bundle=root/'deployment';bundle.mkdir()
            shutil.copy2(selected,bundle/'student.pt')
            shutil.copy2(a.legacy_source,bundle/'m24_legacy_deployment.py')
            shutil.copy2('scripts/deploy_m24_e003_student.py',bundle/'deploy_m24_e003_student.py')
            phase('deployment_dry_run',60,[sys.executable,str(bundle/'deploy_m24_e003_student.py'),
                '--jit-policy',str(bundle/'student.pt'),'--dry-run','--max-policy-steps','10',
                '--log-dir',str(root/'deployment_dry_run')])
            status['status']='passed:numeric_transfer_and_deployment_dry_run;visual_review_pending'
            status['deployment_bundle']=str(bundle)
            (bundle/'manifest.json').write_text(json.dumps(dict(status,artifact_sha256=sha(bundle/'student.pt')),indent=2)+'\n')
        else:
            status['status']='failed:student_transfer_acceptance;candidate_preserved'
    except Exception as exc:
        status.update(status='failed:execution',error=repr(exc));raise
    finally:
        status['finished_at']=now();save()


if __name__=='__main__':main()
