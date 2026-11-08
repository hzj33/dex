#!/usr/bin/env python3
"""One bounded PPO robustness continuation, retaining nominal and held-out evals."""
import argparse
from datetime import datetime
import hashlib,json,math,os,shlex,shutil,subprocess,sys
from pathlib import Path
from import_m24_a0c_for_training import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('a0c','preflight','checkpoint','output'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--noise-degrees',type=float,required=True)
    parser.add_argument('--num-envs',type=int,default=1536)
    parser.add_argument('--train-cycles',type=int,default=2)
    args=parser.parse_args()
    assert args.noise_degrees>0 and args.num_envs>0 and args.train_cycles in (2,5)
    cal=validate(args.a0c);pre=json.loads(args.preflight.read_text())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert pre['summary']['finite'] and pre['summary']['action_mode']=='policy'
    assert pre['summary']['initialization']=='original_candidate743'
    assert pre['checkpoint_sha256']==digest(args.checkpoint)
    assert math.isclose(pre['summary']['initial_joint_noise_rad'],math.radians(args.noise_degrees))
    assert pre['summary']['evaluation_seed']==42 and pre['summary']['nominal_reset_fraction']==0
    hashes={line.split(maxsplit=1)[1]:line.split(maxsplit=1)[0] for line in (args.a0c.parent.parent/'source/SHA256SUMS').read_text().splitlines()}
    for name in ('dexscrew/tasks/e002_initial_state.py','dexscrew/tasks/e002_cycle.py','configs/task/XHandPasiniM24NutBoltE002Cycle.yaml','configs/task/XHandPasiniM24NutBoltE002Robust.yaml'):
        assert hashes[name]==digest(Path(name)),name
    p=args.output;p.mkdir(parents=True,exist_ok=False)
    state=dict(status='running',started_at=datetime.now().astimezone().isoformat(),execution_location='cloud',checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),noise_degrees=args.noise_degrees,train_cycles=args.train_cycles,nominal_reset_fraction=.5,num_envs=args.num_envs,minibatch_size=2*args.num_envs,learning_rate=.0001,reward_weights_changed=False,phases=[])
    def save():(p/'workflow_status.json').write_text(json.dumps(state,indent=2)+'\n')
    def phase(name,seconds,cmd):
        cmd=['timeout','--signal=TERM','--kill-after=30s',str(seconds)]+cmd
        row=dict(name=name,command=cmd,started_at=datetime.now().astimezone().isoformat());state['phases'].append(row);save()
        (p/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
        with (p/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,DEXSCREW_SKIP_GIT_DIFF='1')).returncode
        row.update(exit_status=rc,finished_at=datetime.now().astimezone().isoformat())
        if name in ('smoke','ppo'):row['teacher_exit_status']=rc
        save();return rc
    save();shutil.copy2(args.a0c,p/'a0_calibration.json');shutil.copy2(args.preflight,p/'baseline_preflight.json')
    source=p/'source';source.mkdir()
    for name in list(hashes)+['scripts/run_m24_e002_robustness_probe.py']:
        dest=source/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dest)
    (source/'SHA256SUMS').write_text(''.join(digest(source/name)+'  '+name+'\n' for name in list(hashes)+['scripts/run_m24_e002_robustness_probe.py']))
    pose='+task.env.customInitDofPos=['+','.join(repr(x) for x in cal['custom_init_dof_pos'])+']'
    common=['task=XHandPasiniM24NutBoltE002Robust','headless=True','sim_device=cuda:0','rl_device=cuda:0','graphics_device_id=0','num_threads=8','wandb_activate=False',pose,'train.ppo.learning_rate=0.0001','task.env.e002_motion.force_sign='+str(cal['force_sign'])]
    try:
        for name,n,seconds in [('smoke',1,60),('ppo',args.num_envs,300)]:
            output_name='XHandPasiniM24NutBolt_teacher/'+p.name+'_'+name
            cmd=[sys.executable,'-u','scripts/train_m24_e002_bounded.py']+common+['seed=42','train.algo=PPO','task.env.numEnvs='+str(n),'train.ppo.num_actors='+str(n),'train.ppo.minibatch_size='+str(12 if n==1 else 2*n),'train.ppo.output_name='+output_name,'checkpoint='+str(args.checkpoint),'task.env.e002_init_joint_noise_rad='+str(math.radians(args.noise_degrees)),'task.env.e002_nominal_reset_fraction=0.5','task.env.e002_cycle.goal_cycles='+str(args.train_cycles),'task.env.episodeLength='+str(220*args.train_cycles)]
            rc=phase(name,seconds,cmd);log=(p/(name+'.log')).read_text(errors='replace')
            assert rc in (0,124) and 'Agent Steps:' in log and not any(x in log for x in ('Traceback (most recent call last):','CUDA out of memory','illegal memory access','Segmentation fault'))
        checkpoints=Path('outputs')/output_name/'stage1_nn';best=sorted(checkpoints.glob('best_reward_*.pth'),key=lambda x:x.stat().st_mtime)
        selected={'last':checkpoints/'last.pth'}
        if best:selected['best']=best[-1]
        state['selected_checkpoints']={k:str(v) for k,v in selected.items()};state['checkpoint_sha256s']={k:digest(v) for k,v in selected.items()};save()
        for label,checkpoint in selected.items():
            for name,noise,seed,n,cycles in [('nominal_two',0,42,512,2),('nominal_five',0,42,128,5),('perturbed_s43',args.noise_degrees,43,128,5),('perturbed_s44',args.noise_degrees,44,128,5)]:
                phase_name='eval_'+label+'_'+name;out=p/phase_name/'evaluation.json'
                cmd=[sys.executable,'-u','scripts/evaluate_m24_e002_cycle.py']+common+['test=True','seed='+str(seed),'task.env.numEnvs='+str(n),'train.ppo.num_actors='+str(n),'train.ppo.minibatch_size='+str(2*n),'checkpoint='+str(checkpoint),'task.env.e002_init_joint_noise_rad='+str(math.radians(noise)),'task.env.e002_nominal_reset_fraction=0.0','task.env.e002_cycle.goal_cycles='+str(cycles),'task.env.episodeLength=1100','+cycle_eval_steps=1100','+cycle_calibration='+str(p/'a0_calibration.json'),'+cycle_eval_out='+str(out)]
                rc=phase(phase_name,180,cmd);assert rc==0
                data=json.loads(out.read_text());assert data['summary']['finite'];state['phases'][-1]['summary']=data['summary'];save()
        state['status']='completed:bounded_robustness_probe'
    except Exception as error:
        state.update(status='failed:execution',error=str(error));raise
    finally:
        state['finished_at']=datetime.now().astimezone().isoformat();save()

if __name__=='__main__':main()
