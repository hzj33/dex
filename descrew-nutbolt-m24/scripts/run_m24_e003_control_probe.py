#!/usr/bin/env python3
"""Run explicit frozen-policy cases with independent wall-clock limits."""
import argparse,json,subprocess,shlex,sys
from datetime import datetime
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=False)
 old=json.loads(Path('outputs/cloud_pipeline_m24_e002_robust_time_20260911/workflow_status.json').read_text())
 template=next(p['command'][4:] for p in old['phases'] if p['name']=='seed43_time')
 state=dict(status='running:frozen_control_comparison',training=False,phases=[])
 def save():(a.output/'status.json').write_text(json.dumps(state,indent=2)+'\n')
 try:
  for case in json.loads(a.cases.read_text()):
   name=case['name'];overrides=dict(case['overrides']);overrides['+cycle_eval_out']=str(a.output/name/'evaluation.json')
   overrides.setdefault('checkpoint','outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth')
   cmd=[x for x in template if x.split('=',1)[0] not in overrides]+[k+'='+str(v) for k,v in overrides.items()]
   cmd=['timeout','--signal=TERM','--kill-after=15s','180']+cmd
   row=dict(name=name,command=cmd,start=datetime.now().astimezone().isoformat());state['phases'].append(row);save()
   (a.output/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
   with (a.output/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
   row.update(exit_status=rc,end=datetime.now().astimezone().isoformat());save()
   if rc:raise RuntimeError(f'{name}: {rc}')
   d=json.loads((a.output/name/'evaluation.json').read_text());row['summary']=d['summary'];save()
  state['status']='completed:frozen_control_comparison'
 except Exception as e:state.update(status='failed:execution',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
