"""Opt-in physics-endpoint load audit. No policy/control/reward mutation.

Commanded actuation and DOF sensor totals are distinct; neither is a motor
current measurement. Raw contact records include geometry for later validation.
"""
from pathlib import Path
import json
import numpy as np
from isaacgym import gymtorch
from dexscrew.tasks.contact_moments import collect_env_contacts, normal_contact_wrench


def instrumented_task(base):
    class LoadAuditTask(base):
        def _create_envs(self, *args):
            super()._create_envs(*args)
            for e, hand in zip(self.envs, self.hand_actors):
                self.gym.enable_actor_dof_force_sensors(e, hand)
                self.gym.enable_actor_dof_force_sensors(e, self.gym.find_actor_handle(e, 'object'))
    return LoadAuditTask


class LoadAudit:
    def __init__(self, env, alive):
        assert not env.config['sim']['use_gpu_pipeline']
        assert float(env.random_action_noise_t_scale) == 0 and not np.any(self.arr(env.random_action_noise_e)), 'PD request audit requires zero action noise'
        self.env, self.alive = env, alive
        self.rows, self.contacts, self.contact_steps = [], [], []
        self.wrench = None
        if env.config['env'].get('e003_wrench_audit', False):
            self.wrench = gymtorch.wrap_tensor(env.gym.acquire_force_sensor_tensor(env.sim))
            assert self.wrench.shape == (env.num_envs * 3, 6)
        self.sensor = gymtorch.wrap_tensor(env.gym.acquire_dof_force_tensor(env.sim))
        self.metadata = dict(rotation_axis_world=self.arr(env.rot_axis_buf[0]).tolist(),physics_dt=env.dt, policy_dt=env.dt*env.control_freq_inv,
            solver_wrench_note='Optional base/bolt/nut world-frame constraint-solver resultant, includes joint reactions; not isolated contact wrench',
            sensor_note='DOF total generalized force, not isolated actuator torque or motor current',
            contact_note='raw intra-environment body contact records exclude ground; all-body net tensor retained; full friction wrench not available',
            body_names=list(env.hand_rb_names)+['base','bolt','nut'],
            pair_handles=[int(env.fingertip_handles[0]),int(env.fingertip_handles[3])],
            nut_handle=int(env.screw_nut_rb_handle), dof_properties=[], shape_properties=[])
        for actor in [env.hand_actors[0],env.gym.find_actor_handle(env.envs[0],'object')]:
            p=env.gym.get_actor_dof_properties(env.envs[0],actor)
            self.metadata['dof_properties'].append({k:p[k].tolist() for k in p.dtype.names})
            self.metadata['shape_properties'].append([{k:float(getattr(p,k)) for k in
                ['friction','rest_offset','contact_offset']} for p in env.gym.get_actor_rigid_shape_properties(env.envs[0],actor)])
        self.finger_groups = [[i for i,n in enumerate(env.hand_rb_names) if n.startswith('right_'+finger+'_')] for finger in ('index','thumb')]
        self.metadata['whole_finger_groups'] = self.finger_groups
        control, render = env.update_low_level_control, env.render
        def before(step):
            control(step)
            q=env.xhand_hand_dof_pos;v=env.xhand_hand_dof_vel
            self.pre=dict(q_before=self.arr(q), qdot_before=self.arr(v),
                target=self.arr(env.cur_targets[:,:16]),
                pd_request=self.arr(env.p_gain*(env.cur_targets[:,:16]-q)-env.d_gain*v),
                command=self.arr(env.torques[:,step,:]), phase=self.arr(env.e002_cycle.phase),
                alive=np.asarray(alive()).copy(), policy_step=self.arr(env.progress_buf),substep=np.array(step))
        def after():
            render()
            env.gym.fetch_results(env.sim,True);env._refresh_gym()
            env.gym.refresh_dof_force_tensor(env.sim)
            contacts=collect_env_contacts(env.gym,env.envs)
            force,moment,count=normal_contact_wrench(contacts,
                self.arr(env.rigid_body_states[:,env.screw_nut_rb_handle,3:7]),
                env.screw_nut_rb_handle,self.metadata['pair_handles'],env.num_bodies)
            whole_force,whole_moment,whole_count=normal_contact_wrench(contacts,self.arr(env.rigid_body_states[:,env.screw_nut_rb_handle,3:7]),env.screw_nut_rb_handle,self.finger_groups,env.num_bodies)
            self.rows.append(dict(self.pre,whole_finger_normal=whole_force,whole_finger_normal_moment=whole_moment,whole_finger_contact_count=whole_count,q_after=self.arr(env.xhand_hand_dof_pos),
                qdot_after=self.arr(env.xhand_hand_dof_vel),dof_sensor=self.arr(self.sensor).reshape(env.num_envs,-1),
                body_state=self.arr(env.rigid_body_states),net_contact=self.arr(env.contact_forces),
                nut_angle=self.arr(env.nut_dof_pos),nut_normal=force,nut_normal_moment=moment,nut_contact_count=count))
            if self.wrench is not None:
                env.gym.refresh_force_sensor_tensor(env.sim)
                self.rows[-1]['solver_wrench'] = self.arr(self.wrench).reshape(env.num_envs, 3, 6)
            self.contacts.append(contacts);self.contact_steps.append(np.full(len(contacts),len(self.rows)-1))
        env.update_low_level_control,env.render=before,after

    @staticmethod
    def arr(x):return x.detach().cpu().numpy().copy()

    def save(self,path):
        path=Path(path);data={k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]}
        data['contacts']=np.concatenate(self.contacts);data['contact_step']=np.concatenate(self.contact_steps)
        numeric=[v for k,v in data.items() if k!='contacts']
        assert all(np.isfinite(v).all() for v in numeric)
        np.savez_compressed(path,**data)
        self.metadata.update(samples=len(self.rows),finite=True)
        path.with_suffix('.metadata.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
