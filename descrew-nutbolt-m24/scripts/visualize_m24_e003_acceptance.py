#!/usr/bin/env python3
"""Native event contact sheet and baseline/candidate physics plots for review."""
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('outputs/cloud_pipeline_m24_e003_compliant_validate_20260911');v=p/'native_s43';d=json.loads((v/'trace.json').read_text());rows=d['rows'];events=[r for r in rows if r['event']]
assert len(events)==20 and [r['event'] for r in events]==[1,2,3,4]*5
sheet=Image.new('RGB',(1480,1550),'white');draw=ImageDraw.Draw(sheet)
for i,r in enumerate(events):
 tile=Image.open(v/f"frame_{r['step']:04d}.png").crop((270,180,640,460))
 x=(i%4)*370;y=(i//4)*310;sheet.paste(tile,(x,y+30));draw.text((x+5,y+3),f"Cycle {i//4+1} | {['Drive','Released','Returned','Regrasped'][r['event']-1]} | {r['step']/20:.2f}s",fill='black');draw.text((x+5,y+16),f"{np.degrees(r['angle']):.1f} deg | {r['force'][0]:.2f}/{r['force'][1]:.2f} N",fill='black')
sheet.save(p/'native_event_sheet.png')
a=np.load('outputs/cloud_pipeline_m24_e003_load_20260911/baseline_audit/evaluation.physics.npz');b=np.load(p/'candidate_audit_s43/evaluation.physics.npz')
fig,ax=plt.subplots(3,1,figsize=(11,8),sharex=True,layout='constrained')
for x,name,color in [(a,'E002, Kp=3','#64748b'),(b,'E003Compliant, Kp=2.4','#0d9488')]:
 t=(np.arange(len(x['nut_angle']))+1)*.005;f=np.linalg.norm(x['nut_normal'][:,0],axis=-1)
 ax[0].plot(t,np.degrees(x['nut_angle'][:,0,0]),label=name,color=color)
 ax[1].plot(t,f[:,0],label=name,color=color,lw=.9);ax[2].plot(t,f[:,1],color=color,lw=.9)
ax[0].set_ylabel('Nut angle (deg)');ax[1].set_ylabel('Index normal force (N)');ax[2].set_ylabel('Thumb normal force (N)');ax[2].set_xlabel('Time (s)');ax[0].legend();fig.suptitle('Same frozen PPO and initial state: 200 Hz comparison\nEach curve ends at completion of its own first five-cycle episode')
for z in ax:z.grid(alpha=.2)
fig.savefig(p/'force_comparison.png',dpi=150);plt.close(fig)
print('Wrote native event sheet and 200Hz comparison')
