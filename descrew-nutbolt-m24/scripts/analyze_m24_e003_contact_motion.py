#!/usr/bin/env python3
"""Finite-difference material-point tangential motion at loaded nut contacts.

Diagnostic kinematic estimate only; not a complete friction wrench or tactile
slip sensor. Uses body poses rather than problematic endpoint nut velocities.
"""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=np.load(a.archive);meta=json.loads(a.archive.with_suffix('.metadata.json').read_text());assert x['body_state'].shape[1]==1
c=x['contacts'];t=x['contact_step'];s=x['body_state'][:,0];vec=lambda a:np.stack([a[k] for k in 'xyz'],-1)
def point(local,b,t):
 q=s[t,b,3:7];p=vec(local);return s[t,b,:3]+p+2*np.cross(q[:,:3],np.cross(q[:,:3],p)+q[:,3:4]*p)
b0,b1=c['body0'],c['body1'];normal=vec(c['normal']);prev=np.maximum(t-1,0)
rel=((point(c['localPos0'],b0,t)-point(c['localPos0'],b0,prev))-(point(c['localPos1'],b1,t)-point(c['localPos1'],b1,prev)))/meta['physics_dt']
tangent=rel-(rel*normal).sum(-1,keepdims=True)*normal;speed=np.linalg.norm(tangent,axis=-1);rows=[]
for name,handle in zip(['index','thumb'],meta['pair_handles']):
 mask=(t>0)&(c['lambda']>0)&(x['phase'][t,0]==0)&(((b0==handle)&(b1==meta['nut_handle']))|((b1==handle)&(b0==meta['nut_handle'])))
 weights=c['lambda'][mask];v=speed[mask];rows.append(dict(finger=name,loaded_drive_contact_samples=int(mask.sum()),force_weighted_mean_tangential_speed_m_s=float(np.average(v,weights=weights)),p95_tangential_speed_m_s=float(np.percentile(v,95))))
a.output.write_text(json.dumps(dict(note=__doc__,results=rows),indent=2)+'\n');print(rows)
