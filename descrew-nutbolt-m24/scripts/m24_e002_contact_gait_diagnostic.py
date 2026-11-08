"""Nut-contact point tangential velocities for diagnostics, not torque estimates.

Contact-centroid points are measured anew each step. The Gym COM Jacobian is
shifted to each measured point using the previously independently validated
COM convention. Tac2Motion Eq.7 motivates a gait direction check; no reward or
acceptance term is modified by this helper.
"""
import numpy as np
from dexscrew.tasks.contact_moments import collect_env_contacts


def rotate(q,v):
    return v+2*np.cross(q[:,:3],np.cross(q[:,:3],v)+q[:,3:4]*v)


def measure(env,jac,coms):
    env.gym.refresh_jacobian_tensors(env.sim)
    records=collect_env_contacts(env.gym,env.envs)
    n=env.num_envs;pair=[env.fingertip_handles[0],env.fingertip_handles[3]]
    b0=records['body0']%env.num_bodies;b1=records['body1']%env.num_bodies
    nut=env.screw_nut_rb_handle
    states=env.rigid_body_states.cpu().numpy();jd=jac.cpu().numpy()
    qd=env.dof_state.view(n,-1,2)[:,:16,1].cpu().numpy();axis=env.rot_axis_buf.cpu().numpy()
    results=[]
    for f,handle in enumerate(pair):
        valid=(records['body0']>=0)&(records['body1']>=0)&(records['lambda']>0)&(((b0==handle)&(b1==nut))|((b1==handle)&(b0==nut)))
        rr=records[valid];ids=rr['env0'].astype(int);finger0=b0[valid]==handle
        p0=np.stack([rr['localPos0'][k] for k in ['x','y','z']],axis=-1)
        p1=np.stack([rr['localPos1'][k] for k in ['x','y','z']],axis=-1)
        lp=np.where(finger0[:,None],p0,p1);weights=rr['lambda']
        points=np.zeros((n,3));total=np.zeros(n)
        np.add.at(points,ids,lp*weights[:,None]);np.add.at(total,ids,weights)
        points/=np.maximum(total[:,None],1e-12)
        q=states[:,handle,3:7];world=states[:,handle,:3]+rotate(q,points)
        arm=rotate(q,points-np.asarray(coms[f]))
        j=jd[:,handle-1,:,:16]
        jc=j[:,:3]+np.cross(j[:,3:].transpose(0,2,1),arm[:,None,:]).transpose(0,2,1)
        velocity=np.einsum('nij,nj->ni',jc,qd)
        radial=world-states[:,nut,:3];tangent=np.cross(axis,radial)
        radius=np.linalg.norm(tangent,axis=-1);tangent/=np.maximum(radius[:,None],1e-12)
        speed=np.sum(velocity*tangent,axis=-1)*env.e002_direction
        results.append(dict(valid=total>0,point=world,speed=speed,
            radius=radius,nut_speed=env.nut_dof_vel[:,0].cpu().numpy()*radius*env.e002_direction))
    return [dict(contact_point_present=[bool(r['valid'][i]) for r in results],
        contact_point_world=[r['point'][i].tolist() if r['valid'][i] else None for r in results],
        finger_tangent_velocity_mps=[float(r['speed'][i]) if r['valid'][i] else None for r in results],
        nut_tangent_velocity_mps=[float(r['nut_speed'][i]) if r['valid'][i] else None for r in results],
        interpretation='point velocity from Jacobian and joint velocity; not force/torque') for i in range(n)]
