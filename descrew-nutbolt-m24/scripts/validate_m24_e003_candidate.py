#!/usr/bin/env python3
"""Additional seeds, physics audit and native video for an explicitly chosen model."""
import argparse,json,shlex,subprocess,sys,hashlib,shutil
from datetime import datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--task',default='XHandPasiniM24NutBoltE003Load');p.add_argument('--skip-selection-seeds',action='store_true');p.add_argument('--baseline46-source',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 old=json.loads((a.run/'status.json').read_text());assert old['status']=='completed:E003_load_comparison';template=next(r['command'][4:] for r in old['phases'] if r['name']=='eval_last_s43')
 state=dict(status='running:candidate_validation',training=False,task=a.task,checkpoint=str(a.checkpoint),checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),phases=[])
 if a.baseline46_source:
  baseline=json.loads(a.baseline46_source.read_text());assert baseline['checkpoint_sha256']=='d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0' and baseline['summary']['evaluation_seed']==46 and baseline['summary']['finite']
  state['baseline46_reference']=dict(path=str(a.baseline46_source),sha256=hashlib.sha256(a.baseline46_source.read_bytes()).hexdigest(),note='Reused frozen baseline physical trajectory; its force reward does not enter policy observations or termination')
 def save():(a.output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
 def run(name,cmd):
  cmd=['timeout','--signal=TERM','--kill-after=15s','180']+cmd;row=dict(name=name,command=cmd,start=datetime.now().astimezone().isoformat());state['phases'].append(row);save();(a.output/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
  with (a.output/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
  row.update(exit_status=rc,end=datetime.now().astimezone().isoformat());save();assert rc==0,(name,rc)
 try:
  for name,ck,seed,n,audit in [('candidate_s43',a.checkpoint,43,128,False),('candidate_s44',a.checkpoint,44,128,False),('candidate_s45',a.checkpoint,45,128,False),('baseline_s46',Path(old['selected_teacher']),46,128,False),('candidate_s46',a.checkpoint,46,128,False),('candidate_audit_s43',a.checkpoint,43,1,True)]:
   if a.skip_selection_seeds and name in ('candidate_s43','candidate_s44'):continue
   if a.baseline46_source and name=='baseline_s46':continue
   change={'task':'XHandPasiniM24NutBoltE002RobustTime' if name=='baseline_s46' else a.task,'seed':str(seed),'checkpoint':str(ck),'task.env.numEnvs':str(n),'train.ppo.num_actors':str(n),'train.ppo.minibatch_size':str(12 if n==1 else 2*n),'+cycle_eval_out':str(a.output/name/'evaluation.json')}
   cmd=[x for x in template if x.split('=',1)[0] not in change]+[k+'='+v for k,v in change.items()]
   if audit:cmd+=['+cycle_eval_load_audit=True','+cycle_eval_joint_diagnostics=True']
   run(name,cmd)
  cal=next(x.split('=',1)[1] for x in template if x.startswith('+cycle_calibration='))
  run('native_s43',[sys.executable,'-u','scripts/record_m24_e002_policy.py','task='+a.task,'test=True','headless=True','seed=43','task.env.numEnvs=1','train.ppo.num_actors=1','train.ppo.minibatch_size=12','task.env.e002_nominal_reset_fraction=0','sim_device=cuda:0','rl_device=cuda:0','graphics_device_id=0','num_threads=8','checkpoint='+str(a.checkpoint),'+record_calibration='+cal,'+record_out='+str(a.output/'native_s43')])
  if shutil.which('ffmpeg'):run('encode_native_s43',['ffmpeg','-y','-loglevel','error','-framerate','20','-i',str(a.output/'native_s43/frame_%04d.png'),'-c:v','libx264','-pix_fmt','yuv420p',str(a.output/'native_s43/candidate_cycles.mp4')])
  else:state['video_encoding']='pending:local_ffmpeg_from_native_frames'
  state['status']='completed:candidate_validation'
 except Exception as e:state.update(status='failed:execution',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
