"""Measure original fixed M24's free-joint response to known wrenches.

Diagnostic only: no hand, no policy or reward modification. A torque sweep is
not a proof of two-finger feasibility. Uses actual baseline actor orientation
and original URDF joint dynamics, gravity and task solver settings.
"""
import argparse, ctypes, hashlib, json, sys
from pathlib import Path
# Standalone cloud probes do not inherit train.py bootstrap.
ctypes.CDLL(str(Path(sys.prefix)/"lib"/f"libpython{sys.version_info.major}.{sys.version_info.minor}.so"), mode=ctypes.RTLD_GLOBAL)
from isaacgym import gymapi
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--baseline-audit',type=Path,required=True);a=p.parse_args()
    data=np.load(a.baseline_audit);meta=json.loads(a.baseline_audit.with_suffix('.metadata.json').read_text())
    base_index=meta['body_names'].index('base');base=data['body_state'][0,0,base_index]
    axis=np.asarray(meta['rotation_axis_world'],dtype=float);axis/=np.linalg.norm(axis)
    lateral=np.cross(axis,np.array([1.,0,0]));lateral/=np.linalg.norm(lateral)
    gym=gymapi.acquire_gym();params=gymapi.SimParams();params.dt=.005;params.substeps=1;params.gravity=gymapi.Vec3(0,0,-9.81);params.up_axis=gymapi.UP_AXIS_Z
    params.physx.use_gpu=True;params.use_gpu_pipeline=False;params.physx.solver_type=1;params.physx.num_position_iterations=8;params.physx.num_velocity_iterations=0;params.physx.num_threads=8;params.physx.contact_offset=.002;params.physx.rest_offset=0.;params.physx.max_depenetration_velocity=1000.
    sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
    asset_path=Path('assets/screw/m24hex/0000_m24x3_160.urdf');opt=gymapi.AssetOptions();opt.fix_base_link=True
    asset=gym.load_asset(sim,str(Path('assets').resolve()),str(asset_path.relative_to('assets')),opt)
    assert gym.get_asset_rigid_body_count(asset)==3 and gym.get_asset_dof_count(asset)==1
    trials=[]
    for f in [0.,1.,2.,5.,10.]:
        for t in [0.,.002,.005,.01,.02,.05,.1,.2,.3]:
            env=gym.create_env(sim,gymapi.Vec3(-1,-1,-1),gymapi.Vec3(1,1,1),7)
            pose=gymapi.Transform();pose.p=gymapi.Vec3(*map(float,base[:3]));pose.r=gymapi.Quat(*map(float,base[3:7]))
            actor=gym.create_actor(env,asset,pose,'object',len(trials),0,2)
            dof=gym.get_actor_dof_properties(env,actor)
            assert np.allclose(dof['friction'],.2) and np.allclose(dof['damping'],.01)
            state=gym.get_actor_dof_states(env,actor,gymapi.STATE_ALL);state['pos']=1.;state['vel']=0.;gym.set_actor_dof_states(env,actor,state,gymapi.STATE_ALL)
            trials.append(dict(env=env,actor=actor,nut=gym.find_actor_rigid_body_handle(env,actor,'nut'),force_N=f,torque_Nm=t,rows=[]))
    gym.prepare_sim(sim)
    for step in range(200):
        for trial in trials:
            gym.apply_body_forces(trial['env'],trial['nut'],gymapi.Vec3(*map(float,lateral*trial['force_N'])),gymapi.Vec3(*map(float,axis*trial['torque_Nm'])),gymapi.ENV_SPACE)
        gym.simulate(sim);gym.fetch_results(sim,True)
        for trial in trials:
            state=gym.get_actor_dof_states(trial['env'],trial['actor'],gymapi.STATE_ALL)
            trial['rows'].append([float(state['pos'][0]),float(state['vel'][0]),float(gym.get_env_rigid_contacts(trial['env'])['lambda'].sum())])
    result=[]
    for trial in trials:
        rows=np.asarray(trial['rows']);assert np.isfinite(rows).all()
        result.append(dict(force_N=trial['force_N'],torque_Nm=trial['torque_Nm'],net_angle_rad=float(rows[-1,0]-1),mean_last_100_reported_velocity_rad_s=float(rows[-100:,1].mean()),mean_last_100_position_difference_velocity_rad_s=float(np.diff(rows[-101:,0]).mean()/params.dt),contact_lambda_max_N=float(rows[:,2].max()),rows=rows.tolist()))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(dict(status='completed:joint_load_diagnostic',training=False,asset_sha256=hashlib.sha256(asset_path.read_bytes()).hexdigest(),axis_world=axis.tolist(),lateral_direction=lateral.tolist(),physics_dt=.005,steps=200,gravity=[0,0,-9.81],position_iterations=8,velocity_iterations=0,initial_joint_angle_rad=1.,note='Direct known loads at COM on original no-hand asset; not learned two-finger feasibility or full contact wrench calibration.',trials=result),indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='rows'} for r in result]));gym.destroy_sim(sim)

if __name__=='__main__':main()
