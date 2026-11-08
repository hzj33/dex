#!/usr/bin/env python3
"""Cloud student-only sequence with separate smoke/formal timeout and evaluations."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from import_m24_a0c_for_training import validate
from analyze_m24_e003_loads import summarize


def now():return datetime.now(timezone.utc).isoformat()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output;root.mkdir(parents=True,exist_ok=False)
    teacher=Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/selected_teacher/e003_compliant_frozen_teacher.pth')
    assert sha(teacher)=='d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'
    state=dict(status='running:E003_student',started_at=now(),execution_location='cloud',
               algorithm='ProprioAdapt',selected_teacher=str(teacher),teacher_sha256=sha(teacher),
               teacher_training=False,student_training=True,smoke_seconds=300,formal_seconds=3600,
               train_envs=64,phase_command_required=True,hardware_ready=False,phases=[])
    def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    def phase(name,seconds,command,training=False):
        cmd=['timeout','--signal=TERM','--kill-after=30s',str(seconds)]+command
        row=dict(name=name,start=now(),command=cmd);state['phases'].append(row);save()
        (root/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
        with (root/(name+'.log')).open('w') as f:
            rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,DEXSCREW_SKIP_GIT_DIFF='1')).returncode
        row.update(end=now(),exit_status=rc)
        if training:row['student_exit_status']=rc
        save();assert rc in ((0,124) if training else (0,)),(name,rc)
        log=(root/(name+'.log')).read_text(errors='replace')
        assert not any(x in log for x in ('Traceback (most recent call last):','CUDA out of memory','illegal memory access')),name
        return row
    sources=['dexscrew/algo/models/models.py','dexscrew/algo/ppo/padapt.py','dexscrew/tasks/xhand_pasini.py',
             'dexscrew/tasks/e002_cycle.py','scripts/train_m24_e003_student.py','scripts/evaluate_m24_e002_cycle.py',
             'scripts/record_m24_e002_policy.py','scripts/export_m24_e003_student.py','scripts/run_m24_e003_student_pipeline.py',
             'scripts/test_m24_e003_student_contract.py','configs/train/XHandPasiniM24NutBoltE003Student.yaml',
             'configs/task/XHandPasiniM24NutBoltE003Compliant.yaml',
             'dexscrew/tasks/contact_moments.py','configs/task/XHandPasiniM24NutBoltE002.yaml',
             'configs/task/XHandPasiniM24NutBolt.yaml','configs/task/XHandPasiniNutBolt.yaml',
             'scripts/m24_e002_a0c_diagnostics.py','scripts/diagnose_m24_e002_a0c.py']
    for source in sources:
        dst=root/'source'/source;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,dst)
    (root/'source_manifest.json').write_text(json.dumps({x:sha(x) for x in sources},indent=2)+'\n')
    shutil.copytree(root/'source',root/'a0c/source')
    (root/'a0c/source/SHA256SUMS').write_text(''.join(sha(x)+'  '+x+'\n' for x in sources))
    save()
    try:
        phase('model_contract',60,[sys.executable,'scripts/test_m24_e003_student_contract.py'])
        calpath=root/'a0c/manifests/a0c_diagnostic.json'
        old=Path('outputs/cloud_pipeline_m24_e002_robust_time_20260911/a0c/manifests/a0c_diagnostic.command.txt').read_text()
        cmd=[x if not x.startswith('+diagnostic_out=') else '+diagnostic_out='+str(calpath) for x in shlex.split(old)]
        phase('a0c',180,[sys.executable,'-u']+cmd)
        cal=validate(calpath);state['a0c_status']=cal['status'];save()
        common=['task=XHandPasiniM24NutBoltE003Compliant','headless=True','sim_device=cuda:0','rl_device=cuda:0',
                'graphics_device_id=0','num_threads=8','wandb_activate=False']
        student=['train=XHandPasiniM24NutBoltE003Student']
        def evaluate(name,checkpoint,seed,nominal=False,is_student=True,n=128,audit=False):
            out=root/name/'evaluation.json'
            cmd=[sys.executable,'-u','scripts/evaluate_m24_e002_cycle.py']+common+(student if is_student else [])+[
                'test=True','checkpoint='+str(checkpoint),'seed='+str(seed),'task.env.numEnvs='+str(n),
                'train.ppo.num_actors='+str(n),'train.ppo.minibatch_size='+str(max(n*2,12)),
                'task.env.e002_init_joint_noise_rad='+str(0 if nominal else math.radians(1)),
                'task.env.e002_nominal_reset_fraction=0','+cycle_eval_steps=2200',
                '+cycle_calibration='+str(calpath),'+cycle_eval_out='+str(out)]
            if audit:cmd+=['+cycle_eval_load_audit=True','+cycle_eval_joint_diagnostics=True']
            row=phase(name,180,cmd);d=json.loads(out.read_text());assert d['summary']['finite']
            m=summarize(out);row['metrics']={k:v for k,v in m.items() if k!='episodes'};save();return m
        baseline=evaluate('teacher_nominal',teacher,42,nominal=True,is_student=False)
        assert baseline['successes']==128,'Current teacher baseline not reproduced'
        for name,seconds in [('smoke',300),('formal',3600)]:
            output='XHandPasiniM24NutBolt_student/'+root.name+'_'+name
            state[name+'_output']='outputs/'+output;save()
            row=phase(name,seconds,[sys.executable,'-u','scripts/train_m24_e003_student.py']+common+student+[
                'seed=42','task.env.numEnvs=64','train.ppo.num_actors=64','train.ppo.minibatch_size=64',
                'train.ppo.output_name='+output,'checkpoint='+str(teacher),
                '+task.env.customInitDofPos='+json.dumps(cal['custom_init_dof_pos'],separators=(',',':'))],training=True)
            progress=json.loads(Path('outputs',output,'progress.json').read_text())
            assert progress['updates']>5 and progress['frozen_teacher_unchanged']
            assert all(math.isfinite(v) for v in progress['metrics'].values())
            row['last_progress']=progress;save()
            if name=='smoke':evaluate('smoke_nominal',Path('outputs',output,'stage2_nn/model_last.ckpt'),42,nominal=True)
        nn=Path('outputs',output,'stage2_nn');candidates={'best':nn/'model_best.ckpt','last':nn/'model_last.ckpt'}
        scores={};state['candidate_checkpoints']={k:{'path':str(v),'sha256':sha(v)} for k,v in candidates.items()};save()
        for label,ck in candidates.items():
            mm=[evaluate('eval_'+label+'_'+suffix,ck,seed,nominal=nominal)
                for suffix,seed,nominal in [('nominal',42,True),('s43',43,False),('s44',44,False)]]
            scores[label]=(mm[0]['successes'],sum(m['successes'] for m in mm[1:]),-sum(m['normal_force_peak_N'][0] for m in mm))
        label=max(scores,key=scores.get);selected=root/'selected_student';selected.mkdir()
        ck=selected/'student_candidate.ckpt';shutil.copy2(candidates[label],ck)
        state['selected_student']=dict(path=str(ck),sha256=sha(ck),source_label=label,selection_scores=scores,
                                     status='numeric_candidate_not_full_acceptance');save()
        for seed in (45,46):evaluate('selected_s'+str(seed),ck,seed)
        evaluate('selected_audit_s43',ck,43,n=1,audit=True)
        phase('native_s43',180,[sys.executable,'-u','scripts/record_m24_e002_policy.py']+common+student+[
            'test=True','seed=43','task.env.numEnvs=1','train.ppo.num_actors=1','train.ppo.minibatch_size=12',
            'task.env.e002_nominal_reset_fraction=0','checkpoint='+str(ck),
            '+record_calibration='+str(calpath),'+record_out='+str(root/'native_s43')])
        phase('export',180,[sys.executable,'-u','scripts/export_m24_e003_student.py']+common+student+[
            'test=True','seed=43','task.env.numEnvs=1','train.ppo.num_actors=1','train.ppo.minibatch_size=12',
            'task.env.e002_nominal_reset_fraction=0','checkpoint='+str(ck),
            '+cycle_calibration='+str(calpath),'+export_out='+str(root/'export'),'+export_steps=2200'])
        state.update(status='completed:student_training_evaluation_export',video_encoding='pending:local_encoding_of_native_frames',
                     acceptance='pending:review_completion_load_and_visual_metrics')
    except Exception as exc:
        state.update(status='failed:execution',error=str(exc));raise
    finally:
        state['updated_at']=now();save()


if __name__=='__main__':main()
