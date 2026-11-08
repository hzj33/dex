"""Isolated fixed M24 asset: known external load calibration, cloud PhysX."""
from pathlib import Path
import sys,json
from isaacgym import gymapi
import numpy as np

def main():
 gym=gymapi.acquire_gym();params=gymapi.SimParams();params.dt=.005;params.substeps=1;params.gravity=gymapi.Vec3(0,0,0);params.up_axis=gymapi.UP_AXIS_Z;params.physx.use_gpu=True;params.use_gpu_pipeline=False
 sim=gym.create_sim(0,-1,gymapi.SIM_PHYSX,params);assert sim is not None
 opt=gymapi.AssetOptions();opt.fix_base_link=True
 asset=gym.load_asset(sim,str(Path('assets').resolve()),'screw/m24hex/0000_m24x3_160.urdf',opt)
 assert gym.get_asset_rigid_body_count(asset)==3 and gym.get_asset_dof_count(asset)==1
 for body in ['base','bolt','nut']:
  prop=gymapi.ForceSensorProperties();prop.enable_forward_dynamics_forces=False;prop.enable_constraint_solver_forces=True;prop.use_world_frame=True
  gym.create_asset_force_sensor(asset,gym.find_asset_rigid_body_index(asset,body),gymapi.Transform(),prop)
 env=gym.create_env(sim,gymapi.Vec3(-1,-1,-1),gymapi.Vec3(1,1,1),1);actor=gym.create_actor(env,asset,gymapi.Transform(),'object',0,0);gym.prepare_sim(sim)
 nut=gym.find_actor_rigid_body_handle(env,actor,'nut');rows=[]
 for k in range(120):
  force=5. if 40<=k<80 else 0.
  gym.apply_body_forces(env,nut,gymapi.Vec3(force,0,0),gymapi.Vec3(0,0,0),gymapi.ENV_SPACE)
  gym.simulate(sim);gym.fetch_results(sim,True)
  reading=[]
  for i in range(3):
   v=gym.get_actor_force_sensor(env,actor,i).get_forces();reading.append([v.force.x,v.force.y,v.force.z,v.torque.x,v.torque.y,v.torque.z])
  rows.append(dict(step=k,applied_force_N=force,solver_wrench=reading))
 out=Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(dict(note='5N at nut COM, fixed base, zero gravity, no hand/contact; expected base reaction force magnitude5N and moment magnitude0.625Nm about base origin.',rows=rows),indent=2)+'\n');print(np.mean([r['solver_wrench'] for r in rows[50:75]],axis=0));gym.destroy_sim(sim)
if __name__=='__main__':main()
