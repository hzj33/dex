#!/usr/bin/env python3
"""Validate the explicit extended-time task with a frozen PPO checkpoint."""
import argparse,hashlib,json,shlex,shutil,subprocess
from datetime import datetime
from pathlib import Path

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--baseline-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 old=json.loads((a.baseline_run/'workflow_status.json').read_text());template=next(x['command'][4:] for x in old['phases'] if x['name']=='eval_best_perturbed_s43')
 state=dict(status='running:frozen_policy_validation',training=False,started_at=datetime.now().astimezone().isoformat(),phases=[])
 def save():(a.output/'workflow_status.json').write_text(json.dumps(state,indent=2)+'\n')
 def run(name,cmd):
  cmd=['timeout','--signal=TERM','--kill-after=30s','180']+cmd;row=dict(name=name,command=cmd,started_at=datetime.now().astimezone().isoformat());state['phases'].append(row);save();(a.output/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
  with (a.output/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
  row.update(exit_status=rc,finished_at=datetime.now().astimezone().isoformat());save();assert rc==0
  return row
 source=a.output/'source'
 for name in ['dexscrew/tasks/e002_cycle.py','dexscrew/tasks/xhand_pasini.py','dexscrew/tasks/e002_initial_state.py','scripts/evaluate_m24_e002_cycle.py','scripts/record_m24_e002_policy.py','scripts/validate_m24_e002_robust_time.py','configs/task/XHandPasiniM24NutBoltE002RobustTime.yaml','configs/train/XHandPasiniM24NutBoltE002RobustTime.yaml','scripts/test_m24_e002_phase_clock.py']:
  dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(name,dst)
 (source/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(source))+'\n' for f in sorted(source.rglob('*')) if f.is_file()))
 try:
  for name,seed,noise,extended in [('nominal_time',42,0,True),('seed43_time',43,1,True),('seed44_time',44,1,True),('seed45_original',45,1,False),('seed45_time',45,1,True)]:
   replacements={'task':'XHandPasiniM24NutBoltE002RobustTime' if extended else 'XHandPasiniM24NutBoltE002Robust','seed':str(seed),'task.env.e002_init_joint_noise_rad':str(noise*0.017453292519943295),'task.env.episodeLength':'2200','+cycle_eval_steps':'2200','+cycle_eval_out':str(a.output/name/'evaluation.json')}
   cmd=[x.split('=',1)[0]+'='+replacements[x.split('=',1)[0]] if '=' in x and x.split('=',1)[0] in replacements else x for x in template]
   row=run(name,cmd);d=json.loads((a.output/name/'evaluation.json').read_text());assert d['summary']['finite'];row['summary']=d['summary'];save()
  args={x.split('=',1)[0]:x.split('=',1)[1] for x in template if '=' in x}
  cmd=[template[0],'-u','scripts/record_m24_e002_policy.py','task=XHandPasiniM24NutBoltE002RobustTime','test=True','headless=True','seed=43','task.env.numEnvs=1','train.ppo.num_actors=1','train.ppo.minibatch_size=12','task.env.e002_nominal_reset_fraction=0','sim_device=cuda:0','rl_device=cuda:0','graphics_device_id=0','num_threads=8','checkpoint='+args['checkpoint'],'+record_calibration='+args['+cycle_calibration'],'+record_out='+str(a.output/'native_seed43'),'+record_save_state=True']
  run('record_seed43',cmd)
  state['status']='completed:frozen_policy_validation'
 except Exception as error:state.update(status='failed:validation_execution',error=str(error));raise
 finally:state['finished_at']=datetime.now().astimezone().isoformat();save()
if __name__=='__main__':main()
