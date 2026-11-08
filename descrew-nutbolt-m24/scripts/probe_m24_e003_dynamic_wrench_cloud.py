"""Validate support sensor scaling against a free probe body's momentum change.

Original fixed M24 asset/8-0 solver settings, diagnostic sphere only. Zero
friction is a calibration control, never a task/training physics change.
"""
import argparse,ctypes,json,sys
from pathlib import Path
ctypes.CDLL(str(Path(sys.prefix)/'lib'/f'libpython{sys.version_info.major}.{sys.version_info.minor}.so'),mode=ctypes.RTLD_GLOBAL)
from isaacgym import gymapi
import numpy as np


def vector(v):return np.array([v[k] if isinstance(v,np.void) else getattr(v,k) for k in ('x','y','z')],dtype=float)
def sensor(v):return [v.force.x,v.force.y,v.force.z,v.torque.x,v.torque.y,v.torque.z]

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--friction',type=float,choices=[0.,1.],required=True);p.add_argument('--tangent',choices=['y','z'],required=True);p.add_argument('--material-before-create',action='store_true');p.add_argument('--cpu',action='store_true');a=p.parse_args()
    gym=gymapi.acquire_gym();s=gymapi.SimParams();s.dt=.005;s.substeps=1;s.gravity=gymapi.Vec3(0,0,0);s.up_axis=gymapi.UP_AXIS_Z;s.use_gpu_pipeline=False;s.physx.use_gpu=not a.cpu;s.physx.solver_type=1;s.physx.num_position_iterations=8;s.physx.num_velocity_iterations=0;s.physx.contact_offset=.002;s.physx.rest_offset=0.;s.physx.num_threads=8
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,s);assert sim is not None
    opts=gymapi.AssetOptions();opts.fix_base_link=True
    asset=gym.load_asset(sim,str(Path('assets').resolve()),'screw/m24hex/0000_m24x3_160.urdf',opts)
    assert gym.get_asset_rigid_body_count(asset)==3 and gym.get_asset_dof_count(asset)==1
    sp=gymapi.ForceSensorProperties();sp.enable_forward_dynamics_forces=False;sp.enable_constraint_solver_forces=True;sp.use_world_frame=True
    gym.create_asset_force_sensor(asset,gym.find_asset_rigid_body_index(asset,'base'),gymapi.Transform(),sp)
    sphere_opts=gymapi.AssetOptions();sphere_opts.density=2000000.;sphere_opts.linear_damping=0.;sphere_opts.angular_damping=0.;sphere_opts.max_angular_velocity=1000.
    sphere=gym.create_sphere(sim,.005,sphere_opts)
    if a.material_before_create:
        for source_asset in [asset,sphere]:
            props=gym.get_asset_rigid_shape_properties(source_asset)
            for q in props:q.friction=a.friction;q.restitution=0.
            gym.set_asset_rigid_shape_properties(source_asset,props)
    env=gym.create_env(sim,gymapi.Vec3(-1,-1,-1),gymapi.Vec3(1,1,1),1);obj=gym.create_actor(env,asset,gymapi.Transform(),'object',0,0)
    pose=gymapi.Transform();pose.p=gymapi.Vec3(.035,0,.125);ball=gym.create_actor(env,sphere,pose,'probe',0,0)
    for actor in ([] if a.material_before_create else [obj,ball]):
        props=gym.get_actor_rigid_shape_properties(env,actor)
        for q in props:q.friction=a.friction;q.restitution=0.
        gym.set_actor_rigid_shape_properties(env,actor,props)
    bp=gym.get_actor_rigid_body_properties(env,ball)[0];mass=float(bp.mass);inertia=np.array([[getattr(getattr(bp.inertia,i),j) for j in ('x','y','z')] for i in ('x','y','z')]);assert np.allclose(inertia,np.eye(3)*inertia[0,0])
    base_com=vector(gym.get_actor_rigid_body_properties(env,obj)[0].com)
    body=gym.get_actor_rigid_body_handle(env,ball,0);gym.prepare_sim(sim);rows=[]
    for k in range(800):
        t=max(0,k-200)*s.dt;dynamic=k>=200
        before=gym.get_actor_rigid_body_states(env,ball,gymapi.STATE_ALL)[0];v0=vector(before['vel']['linear']);w0=vector(before['vel']['angular']);p0=vector(before['pose']['p'])
        nut_q=float(gym.get_actor_dof_states(env,obj,gymapi.STATE_POS)['pos'][0])
        normal=np.array([np.cos(nut_q),-np.sin(nut_q),0.]);tangent=np.array([-normal[1],normal[0],0.]);offset=p0-np.array([0.,0.,.125])
        # Known external centering forces keep the free sphere over a face
        # while the original nut remains free to rotate. No hidden constraint.
        f=-(5.+(2.*np.sin(2*np.pi*2*t) if dynamic else 0.))*normal
        f+=(-500.*np.dot(offset,tangent)-10.*np.dot(v0,tangent))*tangent
        f[2]+=-500.*offset[2]-10.*v0[2]
        if dynamic:f+=np.cos(2*np.pi*(5 if k<500 else 7)*t)*(tangent if a.tangent=='y' else np.array([0.,0.,1.]))
        gym.apply_body_forces(env,body,gymapi.Vec3(*map(float,f)),gymapi.Vec3(0,0,0),gymapi.ENV_SPACE);gym.simulate(sim);gym.fetch_results(sim,True)
        after=gym.get_actor_rigid_body_states(env,ball,gymapi.STATE_ALL)[0];v1=vector(after['vel']['linear']);w1=vector(after['vel']['angular']);pos=vector(after['pose']['p'])
        contact_force=mass*(v1-v0)/s.dt-f;contact_torque=inertia@(w1-w0)/s.dt
        expected=np.r_[-contact_force,-contact_torque-np.cross(pos-base_com,contact_force)]
        raw=np.asarray(sensor(gym.get_actor_force_sensor(env,obj,0).get_forces()))
        contacts=gym.get_env_rigid_contacts(env);positive=contacts['lambda']>0
        nforce=np.zeros(3)
        for c in contacts[positive]:
            if int(c['body0'])==body:nforce+=float(c['lambda'])*vector(c['normal'])
            elif int(c['body1'])==body:nforce-=float(c['lambda'])*vector(c['normal'])
        rows.append(dict(step=k,external_force=f.tolist(),position=pos.tolist(),velocity=v1.tolist(),angular_velocity=w1.tolist(),inertial_contact_on_ball=contact_force.tolist(),normal_readback_on_ball=nforce.tolist(),expected_support_at_base_com=expected.tolist(),raw_base_solver=raw.tolist(),nut_q=float(gym.get_actor_dof_states(env,obj,gymapi.STATE_POS)['pos'][0]),normal_lambda_sum=float(contacts['lambda'].sum())))
    assert np.isfinite(np.array([r['expected_support_at_base_com']+r['raw_base_solver'] for r in rows])).all()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(status='completed:dynamic_wrench_probe',physx_use_gpu=not a.cpu,material_before_create=a.material_before_create,actual_shape_friction=[[float(p.friction) for p in gym.get_actor_rigid_shape_properties(env,h)] for h in [obj,ball]],known_external_centering_PD=[500.,10.],calibration_accepted=False,physics_dt=s.dt,friction=a.friction,tangent=a.tangent,mass=mass,inertia=inertia.tolist(),base_com=base_com.tolist(),solver_iterations=[8,0],note='Momentum-change contact estimate requires independent normal/null-friction validation; raw/8 is only a hypothesis. Exclude useful axis-z drive moment from balance assessment.',rows=rows),indent=2)+'\n');gym.destroy_sim(sim)

if __name__=='__main__':main()
