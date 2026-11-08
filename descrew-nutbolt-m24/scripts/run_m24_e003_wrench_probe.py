"""Read-only full-rate diagnostics with optional solver-wrench sensors."""
import argparse,json,os,shlex,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 old=json.load(open('outputs/cloud_pipeline_m24_e003_unload_early_20260914/status.json'));template=next(r['command'] for r in old['phases'] if r['name']=='eval_last_s43');template=template[4:];assert template[0].endswith('python')
 st=dict(status='running:wrench_audit',phases=[],note='Solver resultants include joint reactions; sensor validity unproven. No training.')
 def save():(root/'status.json').write_text(json.dumps(st,indent=2)+'\n')
 try:
  for name,ck,offset in [('teacher',old['selected_teacher'],False),('early_last_replay15',old['candidate_checkpoints']['last']['path'],True)]:
   change={'task':'XHandPasiniM24NutBoltE003Compliant','checkpoint':ck,'task.env.numEnvs':'1','train.ppo.num_actors':'1','train.ppo.minibatch_size':'12','+cycle_eval_out':str(root/name/'evaluation.json')}
   cmd=[x for x in template if x.split('=',1)[0] not in change]+[k+'='+v for k,v in change.items()]+['+cycle_eval_load_audit=True','+task.env.e003_wrench_audit=True']
   if offset:cmd+=['+cycle_eval_initial_offsets=outputs/cloud_pipeline_m24_e003_wrench_probe_20260914_driver/env15_offsets.json']
   cmd=['timeout','--signal=TERM','--kill-after=15s','180']+cmd
   row=dict(name=name,command=cmd,start=datetime.now(timezone.utc).isoformat());st['phases'].append(row);save()
   (root/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
   with (root/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
   row.update(exit_status=rc,end=datetime.now(timezone.utc).isoformat());save();assert rc==0,(name,rc)
  st['status']='completed:wrench_audit'
 except Exception as e:st.update(status='failed:wrench_audit',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
