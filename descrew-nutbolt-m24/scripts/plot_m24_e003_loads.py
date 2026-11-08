#!/usr/bin/env python3
"""Plot the measured physics-endpoint loads of a single first episode."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--title',default='M24 contact and control audit');a=p.parse_args();x=np.load(a.archive);assert x['body_state'].shape[1]==1
meta=json.loads(a.archive.with_suffix('.metadata.json').read_text());t=(np.arange(len(x['phase']))+1)*meta['physics_dt'];f=np.linalg.norm(x['nut_normal'][:,0],axis=-1);phase=x['phase'][:,0]
fig,axes=plt.subplots(4,1,figsize=(12,10),sharex=True,layout='constrained')
axes[0].plot(t,np.degrees(x['nut_angle'][:,0,0]),color='#334155');axes[0].set_ylabel('Nut angle (deg)')
for j,label in enumerate(['Index','Thumb']):axes[1].plot(t,f[:,j],label=label,alpha=.9,lw=1)
axes[1].set_ylabel('Nut normal resultant (N)');axes[1].legend()
for sl,label in [(slice(0,4),'Index joints'),(slice(12,16),'Thumb joints')]:
 axes[2].plot(t,np.abs(x['command'][:,0,sl]).max(-1),label=label,lw=1)
 axes[3].plot(t,np.degrees(np.abs(x['target'][:,0,sl]-x['q_before'][:,0,sl]).max(-1)),label=label,lw=1)
axes[2].set_ylabel('Max |submitted torque|\n(N m)');axes[2].legend();axes[3].set_ylabel('Max target error (deg)');axes[3].set_xlabel('Time (s)')
loss=(phase==0)&(f[:,0]>=.3)&(f[:,1]<.15)
for ax in axes:
 ax.grid(alpha=.2)
 for i in np.flatnonzero(loss&~np.r_[False,loss[:-1]]):
  tail=np.flatnonzero(~loss[i:]);end=i+tail[0] if len(tail) else len(t)-1;ax.axvspan(t[i],t[end],color='red',alpha=.08)
fig.suptitle(a.title+'\nRed: drive with index loaded and thumb normal force below 0.15 N (diagnostic only)')
fig.savefig(a.output,dpi=150);plt.close(fig)
