#!/usr/bin/env python3
"""Paired inference timeout comparison with unchanged policy phase clocks."""
import argparse,json,shlex,subprocess,sys
from datetime import datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 previous=json.loads((a.baseline_run/'workflow_status.json').read_text())
 state=dict(status='running:diagnostic_only',started_at=datetime.now().astimezone().isoformat(),training=False,policy_command_deadlines_changed=False,phases=[])
 def save():(a.output/'workflow_status.json').write_text(json.dumps(state,indent=2)+'\n')
 save()
 try:
  for multiplier in (1,2):
   for seed in (43,44):
    template=next(x for x in previous['phases'] if x['name']=='eval_best_perturbed_s%d'%seed)
    name='timeout%d_seed%d'%(multiplier,seed);out=a.output/name/'evaluation.json'
    cmd=template['command'][4:]
    cmd=[('task.env.episodeLength=2200' if x.startswith('task.env.episodeLength=') else '+cycle_eval_steps=2200' if x.startswith('+cycle_eval_steps=') else '+cycle_eval_out='+str(out) if x.startswith('+cycle_eval_out=') else x) for x in cmd]
    cmd=['timeout','--signal=TERM','--kill-after=30s','180']+cmd+['+cycle_eval_timeout_multiplier='+str(multiplier),'+cycle_eval_joint_diagnostics=True']
    row=dict(name=name,command=cmd,started_at=datetime.now().astimezone().isoformat());state['phases'].append(row);save()
    (a.output/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
    with (a.output/(name+'.log')).open('w') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
    row.update(exit_status=rc,finished_at=datetime.now().astimezone().isoformat());save();assert rc==0
    d=json.loads(out.read_text());assert d['summary']['finite'];row['summary']=d['summary'];save()
  state['status']='completed:diagnostic_only'
 except Exception as error:state.update(status='failed:diagnostic_execution',error=str(error));raise
 finally:state['finished_at']=datetime.now().astimezone().isoformat();save()
if __name__=='__main__':main()
