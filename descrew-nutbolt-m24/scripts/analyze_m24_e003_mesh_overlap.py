#!/usr/bin/env python3
"""Offline source-convex intersection check, not cooked PhysX penetration.

Maximize an inscribed sphere in the intersection of the two source mesh convex
hulls. Positive radius proves source-convex volume overlap. Does not measure
engine contact penetration depth, soft tissue deformation, or motor loads.
"""
import argparse,json,struct,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from scipy.optimize import linprog

def stl(path):
 b=path.read_bytes();n=struct.unpack_from('<I',b,80)[0]
 assert len(b)==84+50*n,path
 d=np.frombuffer(b,offset=84,count=n,dtype=np.dtype([('normal','<f4',3),('v','<f4',(3,3)),('attr','<u2')]))
 return np.unique(d['v'].reshape(-1,3),axis=0)
def colliders(path,name):
 link=ET.parse(path).find("./link[@name='%s']"%name);out=[]
 for shape in link.findall('collision'):
  mesh=shape.find('geometry/mesh');assert mesh is not None
  v=stl((path.parent/mesh.attrib['filename']).resolve())*np.array([float(x) for x in mesh.get('scale','1 1 1').split()])
  o=shape.find('origin');xyz=np.zeros(3);rpy=np.zeros(3)
  if o is not None:
   xyz=np.array([float(x) for x in o.get('xyz','0 0 0').split()]);rpy=np.array([float(x) for x in o.get('rpy','0 0 0').split()])
  v=Rotation.from_euler('xyz',rpy).apply(v)+xyz
  h=ConvexHull(v);out.append((mesh.attrib['filename'],v[h.vertices]))
 return out

def intersection_radius_mm(eq):
 result=linprog([0,0,0,-1],A_ub=np.c_[eq[:,:3],np.ones(len(eq))],b_ub=-eq[:,3],bounds=[(None,None)]*4,method='highs-ipm',options={'presolve':True})
 assert result.success,result.message
 assert np.max(np.c_[eq[:,:3],np.ones(len(eq))]@result.x+eq[:,3])<1e-5, 'Invalid overlap LP residual'
 return max(0.,float(result.x[3]))

def main():
 p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();x=np.load(a.archive);meta=json.loads(a.archive.with_suffix('.metadata.json').read_text());assert x['body_state'].shape[1]==1
 hand=Path('assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf');nut=Path('assets/screw/m24hex/0000_m24x3_160.urdf')
 nv=colliders(nut,'nut')[0][1];neq=ConvexHull(nv*1000).equations
 forces=np.linalg.norm(x['nut_normal'][:,0],axis=-1)
 indices=sorted(set(range(9,len(forces),100))|{int(forces[:,j].argmax()) for j in range(2)})
 rows=[]
 for j,b in enumerate(meta['pair_handles']):
  for mesh,vertices in colliders(hand,meta['body_names'][b]):
   for t in indices:
    state=x['body_state'][t,0];rn=Rotation.from_quat(state[meta['nut_handle'],3:7]);rh=Rotation.from_quat(state[b,3:7])
    v=rn.inv().apply(rh.apply(vertices)+state[b,:3]-state[meta['nut_handle'],:3]);eq=np.r_[neq,ConvexHull(v*1000).equations]
    radius=intersection_radius_mm(eq)
    rows.append(dict(physics_step=t,time_s=(t+1)*.005,finger=['index','thumb'][j],mesh=mesh,source_convex_intersection_radius_mm=radius,normal_force_N=float(forces[t,j])))
 report=dict(note=__doc__,sampled_physics_steps=indices,rows=rows)
 report['max_source_convex_intersection_radius_mm']=max(r['source_convex_intersection_radius_mm'] for r in rows)
 a.output.write_text(json.dumps(report,indent=2)+'\n');print('Maximum source-convex intersection sphere radius mm:',report['max_source_convex_intersection_radius_mm']);print(max(rows,key=lambda r:r['source_convex_intersection_radius_mm']))
if __name__=='__main__':main()
