#!/usr/bin/env python3
"""Deterministic first-episode cycle traces before any simulator reset."""
import hashlib
import json
from pathlib import Path
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch
from dexscrew.algo.ppo.ppo import PPO
from omegaconf import open_dict, OmegaConf
import numpy as np


@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    out = Path(str(cfg.cycle_eval_out))
    assert not out.exists() and cfg.test
    calibration = json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert calibration['pose_index'] == 743 and calibration['status'] == 'passed'
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration['custom_init_dof_pos']
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed))
    replay_offsets = None
    if cfg.get('cycle_eval_initial_offsets'):
        assert int(cfg.task.env.numEnvs) == 1 and cfg.test
        import dexscrew.tasks.e002_initial_state as initial_state
        replay_offsets = json.loads(Path(str(cfg.cycle_eval_initial_offsets)).read_text())['offsets']
        assert len(replay_offsets) == 16 and np.isfinite(replay_offsets).all()
        assert np.max(np.abs(replay_offsets[4:12])) == 0
        assert float(cfg.task.env.e002_init_joint_noise_rad) > 0
        def fixed_offsets(pos, lower, upper, half_width_rad, nominal_fraction):
            offset = torch.tensor(replay_offsets, device=pos.device, dtype=pos.dtype)
            assert float(offset.abs().max()) <= half_width_rad + 1e-6
            return torch.maximum(torch.minimum(pos + offset, upper), lower)
        initial_state.perturb_initial_joints = fixed_offsets
    task_type = isaacgym_task_map[cfg.task_name]
    load_audit = bool(cfg.get('cycle_eval_load_audit', False))
    if load_audit:
        from m24_e003_load_audit import instrumented_task, LoadAudit
        task_type = instrumented_task(task_type)
    env = task_type(config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=0, headless=True)
    assert env.e002_cycle is not None and env.num_dofs == 17
    assert env.num_bodies - env.num_xhand_hand_bodies == 3
    out.parent.mkdir(parents=True, exist_ok=True)
    algo = str(cfg.train.algo)
    assert algo in ('PPO', 'ProprioAdapt'), algo
    agent_type = PPO
    if algo == 'ProprioAdapt':
        from dexscrew.algo.ppo.padapt import ProprioAdapt
        agent_type = ProprioAdapt
    deploy_student = bool(cfg.get('cycle_eval_deploy_student', False))
    if deploy_student:
        from m24_e003_deploy_runtime import DeployInference
        agent = DeployInference(env, cfg.checkpoint,
                                record_predictions=bool(cfg.get('cycle_eval_student_predictions', False)),
                                inference_device=cfg.get('cycle_eval_student_device'))
    else:
        agent = agent_type(env, str(out.parent/'inference'), full_config=cfg)
        agent.restore_test(str(cfg.checkpoint));agent.set_eval()
    composition = None
    if cfg.get('release_policy_checkpoint'):
        from m24_e002_phase_policy import MeasuredPhasePolicy
        agent = MeasuredPhasePolicy(agent,env,out.parent,cfg)
        composition = agent.metadata
        assert cfg.get('cycle_eval_stop_after_release',False)
    if cfg.get('cycle_eval_return_reference'):
        assert composition is None and algo == 'PPO'
        from m24_e003_return_reference import ReturnReferenceDiagnostic
        agent = ReturnReferenceDiagnostic(agent,env,out.parent,cfg)
        composition = agent.metadata
    obs = env.reset()
    # Diagnostic inference only: extend termination while preserving the
    # policy's original phase-clock observation, including its saturation.
    timeout_multiplier = int(cfg.get('cycle_eval_timeout_multiplier', 1))
    assert timeout_multiplier >= 1
    original_deadlines = env.e002_cycle.deadlines.clone()
    command_deadlines = original_deadlines * env.e002_cycle.phase_clock_scale
    if timeout_multiplier != 1:
        c = env.e002_cycle
        c.deadlines = original_deadlines * timeout_multiplier
        c.command = lambda: torch.cat((torch.nn.functional.one_hot(c.phase, 4).float(),
            (c.phase_step.float() / command_deadlines[c.phase]).clamp(0, 1)[:, None]), -1)
    demo = getattr(env, 'e002_demo', None)
    initial_angle = env.nut_dof_pos[:,0].detach().cpu().tolist()
    initial_seed = demo.seed_step.detach().cpu().tolist() if demo is not None else [-1]*env.num_envs
    hs,he = env.priv_info_dict['hand_joint_pos']
    expected_reference = torch.tensor(calibration['custom_init_dof_pos'],device=env.device)[:16]
    initial_joint_offsets = (env.init_pose_buf[:,:16]-expected_reference).detach().cpu().tolist()
    if replay_offsets is not None:
        assert np.allclose(initial_joint_offsets[0], replay_offsets, atol=1e-6)
    initial_reference_error = float((env.priv_info_buf[:,hs:he]-expected_reference).abs().max())
    action_mode = str(cfg.get('cycle_eval_action_mode','policy'))
    if cfg.get('cycle_eval_return_reference'):action_mode='return_reference_ablation'
    assert action_mode in ('policy','zero','return_reference_ablation')
    original = env.compute_reward
    alive = np.ones(env.num_envs, dtype=bool)
    audit = LoadAudit(env, lambda: alive) if load_audit else None
    index_compliance_gain = float(cfg.get('cycle_eval_index_compliance_gain', 0.0))
    control_diagnostic = None
    if index_compliance_gain > 0:
        assert env.num_envs == 1 and algo == 'PPO' and not env.randomize_pd_gains
        base_index_gain = env.p_gain[:, :4].clone()
        assert index_compliance_gain <= float(base_index_gain.min())
        low_level = env.update_low_level_control
        def asymmetric_control(step):
            env.p_gain[:, :4] = torch.where((env.e002_cycle.phase < 2)[:, None], index_compliance_gain, base_index_gain)
            low_level(step)
        env.update_low_level_control = asymmetric_control
        control_diagnostic = dict(acceptance_eligible=False, index_drive_release_pgain=index_compliance_gain,
            index_return_regrasp_pgain=base_index_gain.tolist(), thumb_pgain=env.p_gain[:,12:16].tolist(),
            note='Frozen-policy controller sensitivity only; unused fixed-gain privileged slots retain baseline convention. No force clipping.')
    final = [None] * env.num_envs
    traces = []
    step_index = 0
    limit = int(cfg.get('cycle_eval_steps', 440))
    stop_after_release = bool(cfg.get('cycle_eval_stop_after_release',False))
    save_state = bool(cfg.get('cycle_eval_save_state', False))
    joint_diagnostics = bool(cfg.get('cycle_eval_joint_diagnostics', False))
    raw_action_audit = bool(cfg.get('cycle_eval_raw_actions', False))
    current_policy_action = None
    policy_inputs = []
    assert not save_state or env.num_envs == 1, 'State archive currently uses one environment'
    def arr(x): return x.detach().cpu().numpy().copy()
    if raw_action_audit:
        assert algo == 'PPO' and composition is None and action_mode == 'policy' and env.num_envs == 1
        original_inference = agent.model.act_inference
        def capture_raw_action(*args, **kwargs):
            nonlocal current_policy_action
            result = original_inference(*args, **kwargs)
            current_policy_action = arr(result[0])
            policy_inputs.append(dict({k:arr(v) for k,v in args[0].items()}, raw_mean=current_policy_action, phase=arr(env.e002_cycle.phase)))
            return result
        agent.model.act_inference = capture_raw_action
    def record(*args):
        phase_before = arr(env.e002_cycle.phase)
        original(*args)
        c=env.e002_cycle
        values={k:arr(getattr(c,k)) for k in ['phase','phase_step','cycles','drive_events','release_events',
            'return_events','return_error','high','gross','reverse','contact_progress','last_event','last_reason']}
        angle=arr(env.nut_dof_pos[:,0]);contact=arr(env.e002_contact_state)
        force=arr(torch.norm(env.e002_normal_contact_force,dim=-1));actions=arr(env.actions[:,:16])
        if joint_diagnostics:
            joints=arr(env.dof_state.view(env.num_envs,-1,2)[:,:16])
            targets=arr(env.prev_targets[:,:16])
        for i in np.where(alive)[0]:
            row={k:float(v[i]) if np.issubdtype(v.dtype,np.floating) else int(v[i]) for k,v in values.items()}
            row.update(env_id=int(i),step=step_index,phase_before=int(phase_before[i]),
                angle=float(angle[i]),contact=contact[i].tolist(),force=force[i].tolist(),action=actions[i].tolist(),
                initial_angle=initial_angle[i],seed_step=initial_seed[i],reward=float(env.rew_buf[i]))
            if env.config["env"].get("e003_peak_force_reward", False):
                row["force_cost_peak"] = env.e003_peak_contact_force[i].tolist()
            if raw_action_audit:
                row['policy_action_raw'] = current_policy_action[i].tolist()
            traces.append(row)
            if deploy_student and agent.prediction is not None:
                row['predicted_phase_command'] = agent.prediction[i].cpu().tolist()
                row['predicted_phase'] = int(agent.prediction[i,:4].argmax())
                row['observed_previous_target_max_error_rad'] = float(agent.contract_errors[i, 0])
                row['deployment_target_max_error_rad'] = float(agent.contract_errors[i, 1])
            if joint_diagnostics:
                row.update(joint_state=joints[i].tolist(), joint_targets=targets[i].tolist())
            if hasattr(env, 'nut_raw_angle'):
                row['raw_nut_angle'] = float(env.nut_raw_angle[i, 0])
            if save_state:
                row['reset_seed_state'] = dict(
                    reconstruction_validated=False, boundary='after_reward_before_observation_roll',
                    dof_state=env.dof_state.view(1,-1,2)[0].tolist(),
                    actor_root_state=env.root_state_tensor.tolist(),
                    joint_targets=env.prev_targets[0].tolist(),
                    observation_history_before_roll=env.obs_buf_lag_history[0].tolist(),
                    cycle={k:v[0].tolist() for k,v in c.__dict__.items()
                           if torch.is_tensor(v) and v is not c.deadlines})
            if composition:
                row['action_source'] = ('reference_return_ppo' if phase_before[i] == 2 else 'candidate_ppo') if composition['mode']=='diagnostic_return_reference' else ('drive_ppo' if phase_before[i] == 0 else 'release_ppo')
            if row['last_reason'] or step_index == limit-1 or (stop_after_release and row['last_event']==2):
                final[i]=row;alive[i]=False
    env.compute_reward=record
    with torch.no_grad():
        for step_index in range(limit):
            action,extra=agent._eval_select_action(obs)
            if action_mode == 'zero':action = torch.zeros_like(action)
            obs,_,_,_=env.step(action,**extra)
            if not alive.any():break
    assert all(r is not None for r in final)
    finite=all(np.isfinite([r['angle'],r['return_error'],*r['force'],*r['action']]).all() for r in traces)
    summary=dict(num_envs=env.num_envs,finite=bool(finite),
        required_cycles=env.e002_cycle.goal_cycles,
        stroke_target_degrees=float(np.degrees(env.e002_cycle.target)),
        return_tolerance_degrees=float(np.degrees(env.e002_cycle.return_tolerance)),
        paired_drive_progress_required=env.e002_cycle.require_paired_drive_progress,
        event_dwell_steps=env.e002_cycle.dwell,
        evaluation_seed=int(cfg.seed),
        diagnostic_timeout_multiplier=timeout_multiplier,
        policy_command_deadlines=command_deadlines.detach().cpu().tolist(),
        termination_deadlines=env.e002_cycle.deadlines.detach().cpu().tolist(),
        initial_joint_noise_rad=float(cfg.task.env.get('e002_init_joint_noise_rad',0.0)),
        nominal_reset_fraction=float(cfg.task.env.get('e002_nominal_reset_fraction',1.0)),
        first_drive=sum(r['drive_events']>=1 for r in final),
        released=sum(r['release_events']>=1 for r in final),
        returned=sum(r['return_events']>=1 for r in final),
        regrasped=sum(r['cycles']>=1 for r in final),
        second_drive=sum(r['drive_events']>=2 for r in final),
        completed_two_cycles=sum(r['cycles']>=2 and r['last_reason']==5 for r in final),
        median_net_degrees=float(np.degrees(np.median([r['angle'] for r in final]))),
        termination_counts={str(k):sum(r['last_reason']==k for r in final) for k in (0,3,4,5,6,7,10,11,12,13,14)},
        initialization='demonstration_state' if demo is not None else 'original_candidate743',
        action_mode=action_mode,release_subtask_success=sum(r['last_reason']==6 for r in final),
        return_subtask_success=sum(r['last_reason']==7 for r in final),
        initial_hand_reference_max_error=initial_reference_error,
        per_seed={str(seed):dict(n=sum(r['seed_step']==seed for r in final),
            released=sum(r['seed_step']==seed and r['release_events']>=1 for r in final),
            returned=sum(r['seed_step']==seed and r['return_events']>=1 for r in final)) for seed in sorted(set(initial_seed))})
    report=dict(status='evaluated:cycle_learning' if finite else 'failed:nonfinite',
        policy_algorithm='DeployStudent' if deploy_student else algo,
        phase_command_source=('learned from joint/target history; simulator phase used only for scoring'
                              if deploy_student else 'environment cycle state machine (not inferred by student)'),
        deployment_action_scale=float(env.action_scale),
        deployment_inference_device=agent.inference_device if deploy_student else str(env.rl_device),
        goal_completed=False,goal_note='Numeric phase events require visual verification of repeated motion.',
        checkpoint=str(cfg.checkpoint), checkpoint_sha256=hashlib.sha256(Path(str(cfg.checkpoint)).read_bytes()).hexdigest(),
        summary=summary,first_episodes=final,first_episode_steps=traces,initial_joint_offsets=initial_joint_offsets,replay_initial_offsets=replay_offsets)
    if control_diagnostic:
        report['control_diagnostic'] = control_diagnostic
    if composition:
        report['policy_composition'] = composition
    if hasattr(env, 'continuous_nut_receipt'):
        report['continuous_nut_receipt'] = env.continuous_nut_receipt
    if hasattr(env, 'return_progress_receipt'):
        report['return_progress_receipt'] = env.return_progress_receipt
    report['stopped_after_release_event'] = stop_after_release
    OmegaConf.save(cfg,out.with_suffix('.config.yaml'),resolve=True)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(summary,indent=2))
    if raw_action_audit:
        np.savez_compressed(out.with_suffix('.policy_inputs.npz'), **{k:np.concatenate([r[k] for r in policy_inputs]) for k in policy_inputs[0]})
    if audit:
        audit.save(out.with_suffix('.physics.npz'))
    env.gym.destroy_sim(env.sim)
    raise SystemExit(0 if finite else 4)


if __name__=='__main__':main()
