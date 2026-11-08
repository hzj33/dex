#!/usr/bin/env python3
"""Record one actual first episode before reset; exported playback is 20 Hz."""
import json
from pathlib import Path

from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch
from isaacgym import gymapi
from omegaconf import open_dict
from dexscrew.algo.ppo.ppo import PPO
from PIL import Image, ImageDraw
import numpy as np


@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    out = Path(str(cfg.record_out))
    out.mkdir(parents=True, exist_ok=False)
    calibration = json.loads(Path(str(cfg.record_calibration)).read_text())
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration['custom_init_dof_pos']
        cfg.task.env.enableCameraSensors = True
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    task_type = isaacgym_task_map[cfg.task_name]
    def create_camera(self, env_ptr):
        props = gymapi.CameraProperties()
        props.width, props.height = 960, 720
        props.enable_tensors = False
        self.record_camera = self.gym.create_camera_sensor(env_ptr, props)
        return None
    task_type._create_camera = create_camera  # Diagnostic process only.
    env = task_type(config=omegaconf_to_dict(cfg.task), sim_device=cfg.sim_device,
                    graphics_device_id=0, headless=True)
    assert env.num_envs == 1
    algo = str(cfg.train.algo)
    assert algo in ('PPO', 'ProprioAdapt'), algo
    agent_type = PPO
    if algo == 'ProprioAdapt':
        from dexscrew.algo.ppo.padapt import ProprioAdapt
        agent_type = ProprioAdapt
    deploy_student = bool(cfg.get('record_deploy_student', False))
    if deploy_student:
        from m24_e003_deploy_runtime import DeployInference
        agent = DeployInference(env, cfg.checkpoint, inference_device=cfg.get('record_student_device'))
    else:
        agent = agent_type(env, str(out / 'inference'), full_config=cfg)
        agent.restore_test(str(cfg.checkpoint))
        agent.set_eval()
    composition = None
    if cfg.get('release_policy_checkpoint'):
        from m24_e002_phase_policy import MeasuredPhasePolicy
        agent = MeasuredPhasePolicy(agent,env,out,cfg)
        composition = agent.metadata
        assert cfg.get('record_stop_after_release',False)
    override = None
    if cfg.get('record_release_candidate_file'):
        evidence = json.loads(Path(str(cfg.record_release_candidate_file)).read_text())
        override = evidence['candidates'][int(cfg.record_release_candidate)]['action']
        assert len(override) == 16 and np.isfinite(override).all()
    obs = env.reset()
    initial_joint_positions = env.init_pose_buf[0,:16].detach().cpu().tolist()
    demo_start = getattr(env, 'e002_demo', None) is not None
    stop_after_release = bool(cfg.get('record_stop_after_release', False))
    stop_after_return = bool(cfg.get('record_stop_after_return', False))
    stop_after_regrasp = bool(cfg.get('record_stop_after_regrasp', False))
    nut = env.rigid_body_states[0, env.screw_nut_rb_handle, :3].tolist()
    target = gymapi.Vec3(*nut)
    position = gymapi.Vec3(nut[0]+.18, nut[1]+.19, nut[2]+.12)
    env.gym.set_camera_location(env.record_camera, env.envs[0], position, target)
    original = env.compute_reward
    rows = []
    override_active = False
    def record(*args, **kwargs):
        result = original(*args, **kwargs)
        row = dict(step=len(rows), angle=float(env.nut_dof_pos[0,0]),
            contact=env.e002_contact_state[0].tolist(),
            force=torch.norm(env.e002_normal_contact_force[0], dim=-1).tolist(),
            action=env.actions[0,:16].tolist(), reward=float(env.rew_buf[0]),
            forward_quality=float(env.e002_forward_quality[0]),
            reason=int(env.e002_termination_reason[0]))
        row['scripted_release_action'] = override_active
        cycle = getattr(env, 'e002_cycle', None)
        if cycle is not None:
            row.update(phase=int(cycle.phase[0]), cycles=int(cycle.cycles[0]),
                       event=int(cycle.last_event[0]), return_error=float(cycle.return_error[0]))
        if cfg.get('record_save_state', False):
            row['reset_seed_state'] = dict(
                reconstruction_validated=False, boundary='after_reward_before_observation_roll',
                dof_state=env.dof_state.view(1,-1,2)[0].tolist(),
                actor_root_state=env.root_state_tensor.tolist(),
                joint_targets=env.prev_targets[0].tolist(),
                observation_history_before_roll=env.obs_buf_lag_history[0].tolist(),
                cycle={k:v[0].tolist() for k,v in cycle.__dict__.items()
                       if torch.is_tensor(v) and v is not cycle.deadlines} if cycle is not None else {})
        env.gym.step_graphics(env.sim)
        env.gym.render_all_camera_sensors(env.sim)
        rgba = env.gym.get_camera_image(env.sim, env.envs[0], env.record_camera, gymapi.IMAGE_COLOR)
        frame = Image.fromarray(np.asarray(rgba).reshape(720,960,4)[:,:,:3])
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0,0,960,52), fill=(20,20,20))
        label = ('CYCLE PROBE phase=%d cycles=%d' % (row['phase'], row['cycles'])
                 if cycle is not None else 'LEARNED DRIVE ONLY')
        if override is not None:
            label = 'DIAGNOSTIC: learned drive + scripted release'
        elif demo_start:
            label = 'LEARNED POLICY: demonstrated initial state'
        elif stop_after_release:
            label = 'LEARNED POLICY: original start, drive + release'
        elif stop_after_return:
            label = 'LEARNED POLICY: original start, drive + release + return'
        elif stop_after_regrasp:
            label = 'LEARNED POLICY: original start, drive + release + return + regrasp'
        if composition:
            label = 'TWO LEARNED PPO SKILLS: drive -> release'
        if deploy_student:
            label = 'E003 STUDENT: joint/target history only | cycles=%d' % row['cycles']
        if getattr(agent, 'diagnostic_phase_intervention', False):
            label = 'ASSISTED PHASE DIAGNOSTIC: simulated trigger | cycles=%d' % row['cycles']
        draw.text((12,8), '%s | step %d | nut %.2f deg | contact %s' %
                  (label, row['step'], np.degrees(row['angle']), row['contact']), fill='white')
        draw.text((12,28), 'Nut contact force: %.2f / %.2f N | termination %d | 20 Hz playback' %
                  (*row['force'],row['reason']), fill='white')
        frame.save(out / ('frame_%04d.png' % row['step']))
        rows.append(row)
        return result
    env.compute_reward = record
    with torch.no_grad():
        for _ in range(int(cfg.task.env.episodeLength)):
            action, extra = agent._eval_select_action(obs)
            override_active = override is not None and int(env.e002_cycle.phase[0]) == 1
            if override_active:
                action[0,:16] = torch.tensor(override,device=action.device)
            obs, _, _, _ = env.step(action, **extra)
            if (rows[-1]['reason'] or (override is not None and rows[-1]['phase'] >= 2)
                    or (stop_after_release and rows[-1].get('event') == 2)
                    or (stop_after_return and rows[-1].get('event') == 3)
                    or (stop_after_regrasp and rows[-1].get('event') == 4)):
                break
    (out / 'trace.json').write_text(json.dumps(dict(checkpoint=str(cfg.checkpoint),
        policy_algorithm='DeployStudent' if deploy_student else algo,
        phase_command_source=('learned from joint/target history; simulator phase used only for scoring'
                              if deploy_student else 'environment cycle state machine (not inferred by student)'),
        deployment_action_scale=float(env.action_scale),
        deployment_inference_device=agent.inference_device if deploy_student else str(env.rl_device),
        playback_fps=20, goal_completed=False, diagnostic_action_override=override,
        evaluation_seed=int(cfg.seed),initial_joint_positions=initial_joint_positions,
        initial_joint_noise_rad=float(cfg.task.env.get("e002_init_joint_noise_rad",0.0)),
        nominal_reset_fraction=float(cfg.task.env.get("e002_nominal_reset_fraction",1.0)),
        initialization='demonstration_state' if demo_start else 'original_candidate743',
        policy_composition=composition,
        stop_after_release=stop_after_release,
        stop_after_return=stop_after_return,
        stop_after_regrasp=stop_after_regrasp,
        learned_release_event=override is None and any(r.get('event') == 2 for r in rows),
        learned_return_event=override is None and any(r.get('event') == 3 for r in rows),
        learned_regrasp_event=override is None and any(r.get('event') == 4 for r in rows),
        first_episode_only=True, rows=rows), indent=2)+'\n')
    env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    main()
