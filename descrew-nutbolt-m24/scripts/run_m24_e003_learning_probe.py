#!/usr/bin/env python3
"""E003 fresh physical preflight, separate60s/300s teacher phases and fixed evals."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys
from datetime import datetime
from pathlib import Path
from import_m24_a0c_for_training import validate

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--learning-rate',type=float,default=.0001);p.add_argument('--kl-threshold',type=float,default=.02);a=p.parse_args();root=a.output;root.mkdir(parents=True,exist_ok=False)
 state=dict(status='running:E003_load',execution_location='cloud',training=True,phases=[],
  learning_rate=a.learning_rate,kl_threshold=a.kl_threshold,
  hypothesis='reduce static normal load and unilateral drive loading with unchanged dynamics/controller',
  selected_teacher='outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth')
 def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 state['initial_checkpoint_sha256']=digest(state['selected_teacher'])
 assert state['initial_checkpoint_sha256']=='d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0'
 def save():(root/'status.json').write_text(json.dumps(state,indent=2)+'\n')
 def phase(name,seconds,cmd,train=False):
  cmd=['timeout','--signal=TERM','--kill-after=30s',str(seconds)]+cmd
  row=dict(name=name,command=cmd,start=datetime.now().astimezone().isoformat());state['phases'].append(row);save()
  (root/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
  with (root/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,DEXSCREW_SKIP_GIT_DIFF='1')).returncode
  row.update(exit_status=rc,end=datetime.now().astimezone().isoformat())
  if train:row['teacher_exit_status']=rc
  save();log=(root/(name+'.log')).read_text(errors='replace')
  assert rc in ((0,124) if train else (0,)),(name,rc)
  assert not any(e in log for e in ['Traceback (most recent call last):','CUDA out of memory','illegal memory access','Segmentation fault']),name
  if train:assert 'Agent Steps:' in log,name
  return row
 sources=['dexscrew/tasks/xhand_pasini.py','dexscrew/tasks/contact_moments.py','dexscrew/tasks/e002_cycle.py','dexscrew/tasks/e002_initial_state.py','configs/task/XHandPasiniM24NutBoltE002.yaml','configs/task/XHandPasiniM24NutBolt.yaml','configs/task/XHandPasiniNutBolt.yaml','configs/task/XHandPasiniM24NutBoltE003Load.yaml','configs/train/XHandPasiniM24NutBoltE003Load.yaml','scripts/m24_e002_a0c_diagnostics.py','scripts/diagnose_m24_e002_a0c.py','scripts/evaluate_m24_e002_cycle.py','scripts/train_m24_e002_bounded.py','scripts/run_m24_e003_learning_probe.py','scripts/m24_e003_load_audit.py']
 source=root/'a0c/source'
 for name in sources:
  dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dst)
 (source/'SHA256SUMS').write_text(''.join(digest(name)+'  '+name+'\n' for name in sources))
 calpath=root/'a0c/manifests/a0c_diagnostic.json'
 try:
  old=Path('outputs/cloud_pipeline_m24_e002_robust_time_20260911/a0c/manifests/a0c_diagnostic.command.txt').read_text();cmd=shlex.split(old)
  cmd=[x if not x.startswith('+diagnostic_out=') else '+diagnostic_out='+str(calpath) for x in cmd]
  phase('a0c',180,[sys.executable,'-u']+cmd)
  cal=validate(calpath);state['a0c_status']=cal['status'];save()
  common=['task=XHandPasiniM24NutBoltE003Load','headless=True','sim_device=cuda:0','rl_device=cuda:0','graphics_device_id=0','num_threads=8','wandb_activate=False',
   '+task.env.customInitDofPos='+json.dumps(cal['custom_init_dof_pos'],separators=(',',':')),'train.ppo.learning_rate='+str(a.learning_rate),'train.ppo.kl_threshold='+str(a.kl_threshold)]
  def evaluate(name,checkpoint,seed,noise):
   out=root/name/'evaluation.json'
   row=phase(name,180,[sys.executable,'-u','scripts/evaluate_m24_e002_cycle.py']+common+['test=True','seed='+str(seed),'task.env.numEnvs=128','train.ppo.num_actors=128','train.ppo.minibatch_size=256','checkpoint='+str(checkpoint),'task.env.e002_init_joint_noise_rad='+str(noise),'task.env.e002_nominal_reset_fraction=0','+cycle_eval_steps=2200','+cycle_calibration='+str(calpath),'+cycle_eval_out='+str(out)])
   d=json.loads(out.read_text());assert d['summary']['finite'];row['summary']=d['summary'];save()
  evaluate('preflight_s43',state['selected_teacher'],43,.017453292519943295)
  for name,n,seconds in [('smoke',1,60),('ppo',1536,300)]:
   output='XHandPasiniM24NutBolt_teacher/'+root.name+'_'+name
   phase(name,seconds,[sys.executable,'-u','scripts/train_m24_e002_bounded.py']+common+['seed=42','train.algo=PPO','task.env.numEnvs='+str(n),'train.ppo.num_actors='+str(n),'train.ppo.minibatch_size='+str(12 if n==1 else n*2),'train.ppo.output_name='+output,'checkpoint='+state['selected_teacher']],train=True)
  nn=Path('outputs')/output/'stage1_nn';best=sorted(nn.glob('best_reward_*.pth'),key=lambda p:p.stat().st_mtime)
  selected={'last':nn/'last.pth'}
  if best:selected['best']=best[-1]
  state['candidate_checkpoints']={k:dict(path=str(v),sha256=digest(v)) for k,v in selected.items()};save()
  for label,ck in selected.items():
   for suffix,seed,noise in [('nominal',42,0),('s43',43,.017453292519943295),('s44',44,.017453292519943295)]:evaluate('eval_'+label+'_'+suffix,ck,seed,noise)
  state['status']='completed:E003_load_comparison'
 except Exception as e:state.update(status='failed:execution',error=str(e));raise
 finally:state['end']=datetime.now().astimezone().isoformat();save()
if __name__=='__main__':main()
