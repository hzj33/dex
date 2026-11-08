"""Frozen-policy compliance sweep; no force clipping or reward/physics changes."""
import argparse,json,shlex,subprocess
from pathlib import Path
from datetime import datetime,timezone

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=False)
 old=json.load(open('outputs/cloud_pipeline_m24_e003_force5_20260914/status.json'));template=next(r['command'] for r in old['phases'] if r['name']=='preflight_s43')[4:];st=dict(status='running:gain_probe',training=False,phases=[])
 def save():(root/'status.json').write_text(json.dumps(st,indent=2)+'\n')
 try:
  for kp in [2.0,1.6,1.2,.8,.4]:
   name='kp_'+str(kp);change={'task':'XHandPasiniM24NutBoltE003Compliant','task.env.controller.pgain':str(kp),'seed':'43','task.env.numEnvs':'1','train.ppo.num_actors':'1','train.ppo.minibatch_size':'12','+cycle_eval_out':str(root/name/'evaluation.json')}
   cmd=[x for x in template if x.split('=',1)[0] not in change]+[k+'='+v for k,v in change.items()]+['+cycle_eval_load_audit=True']
   cmd=['timeout','--signal=TERM','--kill-after=15s','180']+cmd;row=dict(name=name,pgain=kp,command=cmd,start=datetime.now(timezone.utc).isoformat());st['phases'].append(row);save();(root/(name+'.command.sh')).write_text(shlex.join(cmd)+'\n')
   with (root/(name+'.log')).open('w') as f:rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
   row.update(exit_status=rc,end=datetime.now(timezone.utc).isoformat());save();assert rc==0
   d=json.load(open(root/name/'evaluation.json'));row['summary']=d['summary'];save()
  st['status']='completed:gain_probe'
 except Exception as e:st.update(status='failed:gain_probe',error=str(e));raise
 finally:save()
if __name__=='__main__':main()
