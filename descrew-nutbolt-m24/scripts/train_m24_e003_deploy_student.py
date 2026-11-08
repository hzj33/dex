#!/usr/bin/env python3
"""Bounded, student-only DAgger with learned phase/latent and real input contract.

DAgger: https://proceedings.mlr.press/v15/ross11a.html . The finite replay buffer,
frozen actor, phase cross entropy, clock MSE and latent MSE are local adaptations.
No reward loss is optimized. The teacher is never updated.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import time
from datetime import datetime, timezone
from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, torch
from dexscrew.algo.ppo.ppo import PPO
from dexscrew.algo.models.e003_deploy_student import DeployStudent, TEACHER_SHA
from dexscrew.algo.e003_replay import sample_with_expert_anchor
from omegaconf import open_dict
import numpy as np
from m24_e003_deploy_runtime import install_deployment_limits

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


@hydra.main(config_name='config', config_path='../configs')
def main(cfg):
    out = Path(str(cfg.deploy_student_out)); out.mkdir(parents=True, exist_ok=False)
    source_paths = ['scripts/train_m24_e003_deploy_student.py', 'scripts/m24_e003_deploy_runtime.py',
                    'dexscrew/algo/models/e003_deploy_student.py', 'dexscrew/algo/e003_replay.py']
    source_receipts = {}
    for path in source_paths:
        dest = out/'runtime_source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_receipts[path] = hashlib.sha256(dest.read_bytes()).hexdigest()
    teacher_path = Path(str(cfg.checkpoint))
    teacher_sha = str(cfg.get('deploy_student_teacher_sha256', TEACHER_SHA))
    assert len(teacher_sha) == 64
    assert hashlib.sha256(teacher_path.read_bytes()).hexdigest() == teacher_sha
    cal = json.loads(Path(str(cfg.cycle_calibration)).read_text())
    assert cal['status'] == 'passed' and cal['pose_index'] == 743
    torch.manual_seed(int(cfg.seed)); np.random.seed(int(cfg.seed)); torch.set_num_threads(1)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = cal['custom_init_dof_pos']
    task_cls = isaacgym_task_map[cfg.task_name]
    long_cycles = int(cfg.get('deploy_student_goal_cycles', 5))
    assert long_cycles in (5, 20)
    assert int(cfg.task.env.e002_cycle.goal_cycles) == long_cycles
    continuous_nut = bool(cfg.get('deploy_student_continuous_nut', False))
    if continuous_nut:
        from dexscrew.tasks.e003_continuous_nut import instrument_continuous_nut
        task_cls = instrument_continuous_nut(task_cls)
        path = 'dexscrew/tasks/e003_continuous_nut.py'
        dest = out/'runtime_source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); source_receipts[path] = hashlib.sha256(dest.read_bytes()).hexdigest()
    return_entry_clock = bool(cfg.get('deploy_student_return_entry_clock', False))
    if return_entry_clock:
        from dexscrew.tasks.e003_return_progress import instrument_return_entry_clock
        task_cls = instrument_return_entry_clock(task_cls)
        path = 'dexscrew/tasks/e003_return_progress.py'
        dest = out/'runtime_source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); source_receipts[path] = hashlib.sha256(dest.read_bytes()).hexdigest()
    actuator_spec = None
    if cfg.get('deploy_student_actuator_randomization', False):
        from dexscrew.tasks.e003_actuator_randomization import instrument_actuator_randomization
        actuator_spec = json.loads(cfg.get('deploy_student_actuator_spec', '{}'))
        task_cls = instrument_actuator_randomization(task_cls, actuator_spec)
        path = 'dexscrew/tasks/e003_actuator_randomization.py'
        dest = out/'runtime_source'/path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        source_receipts[path] = hashlib.sha256(dest.read_bytes()).hexdigest()
    coupling_spec = None
    if cfg.get('deploy_student_passive_coupling_spec'):
        from dexscrew.tasks.e003_passive_coupling import instrument_passive_coupling
        coupling_spec = json.loads(Path(str(cfg.deploy_student_passive_coupling_spec)).read_text())
        fit_path = Path(coupling_spec['coupling_fit'])
        fit = json.loads(fit_path.read_text())
        coupling_spec['fit_sha256'] = hashlib.sha256(fit_path.read_bytes()).hexdigest()
        task_cls = instrument_passive_coupling(task_cls, fit['coefficients_descending'],
            coupling_spec['coupling_k'], coupling_spec['coupling_d'])
        path = 'dexscrew/tasks/e003_passive_coupling.py'
        dest = out/'runtime_source'/path; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest); source_receipts[path] = hashlib.sha256(dest.read_bytes()).hexdigest()
    env = task_cls(config=omegaconf_to_dict(cfg.task), sim_device=cfg.sim_device,
                   graphics_device_id=0, headless=True)
    assert env.e002_cycle.goal_cycles == long_cycles and float(cfg.task.env.controller.pgain) == 2.4
    teacher = PPO(env, str(out/'teacher'), full_config=cfg)
    teacher.restore_test(str(teacher_path)); teacher.set_eval(); teacher.model.requires_grad_(False)
    learner_device = str(cfg.rl_device)
    model = DeployStudent(torch.load(teacher_path, map_location='cpu'), cal['custom_init_dof_pos'],
                          float(cfg.get('deploy_student_action_scale', 0.05)),
                          return_entry_clock=return_entry_clock).to(learner_device)
    warm_start = cfg.get('deploy_student_warm_start')
    encoder_warm_start = cfg.get('deploy_student_encoder_warm_start')
    assert not (warm_start and encoder_warm_start), 'Choose full-model or encoder-only warm start'
    warm_start_sha = None
    if warm_start:
        initial = torch.jit.load(str(warm_start), map_location=learner_device)
        assert initial.action_scale == model.action_scale and initial.policy_hz == model.policy_hz
        state = initial.state_dict()
        for key, value in model.state_dict().items():
            if not key.startswith('encoder.'):
                assert torch.equal(value, state[key]), key
        model.load_state_dict(state, strict=True)
        warm_start_sha = hashlib.sha256(Path(str(warm_start)).read_bytes()).hexdigest()
    if encoder_warm_start:
        initial = torch.jit.load(str(encoder_warm_start), map_location=learner_device)
        assert initial.action_scale == model.action_scale and initial.policy_hz == model.policy_hz
        assert torch.equal(initial.init_pose, model.init_pose)
        model.encoder.load_state_dict(initial.encoder.state_dict(), strict=True)
        warm_start_sha = hashlib.sha256(Path(str(encoder_warm_start)).read_bytes()).hexdigest()
    install_deployment_limits(env, model)
    optimizer = torch.optim.Adam(model.encoder.parameters(), lr=3e-4)
    # Bounded replay is deliberately explicit (~180 MiB float32 at 16384 samples).
    capacity, batch = 16384, 512
    replay = {k: torch.empty((capacity, *shape), device=learner_device) for k, shape in
              {'obs': (96,), 'hist': (80,32), 'action': (16,), 'command': (5,), 'latent': (40,)}.items()}
    size = cursor = steps = 0
    anchor_fraction = float(cfg.get('deploy_student_expert_replay_fraction', 0.0))
    assert 0 <= anchor_fraction <= 1
    expert_burnin = int(cfg.get('deploy_student_expert_burnin_steps', 500))
    assert 500 <= expert_burnin <= 2200
    anchor = {key: torch.empty_like(value) for key, value in replay.items()} if anchor_fraction else {}
    anchor_size = 0
    phase_counts = torch.zeros(4, dtype=torch.long, device=learner_device)
    started = datetime.now(timezone.utc).isoformat(); start_clock = last_save = time.monotonic()
    obs = env.reset(); frame = obs['proprio_hist'][:, -1]
    history = model.reset_history(frame[:, 16:], frame[:, 16:])
    metrics = {}
    rollout_evidence = dict(max_cycles_seen=0, completed_twenty=0, terminal_counts={})
    if continuous_nut:
        reward = env.compute_reward
        def record_reward(*args):
            reward(*args)
            c = env.e002_cycle
            rollout_evidence['max_cycles_seen'] = max(rollout_evidence['max_cycles_seen'], int(c.cycles.max()))
            terminal = c.last_reason[c.last_reason > 0].cpu().tolist()
            for reason in terminal:
                key = str(reason)
                rollout_evidence['terminal_counts'][key] = rollout_evidence['terminal_counts'].get(key, 0)+1
                if reason == 5 and long_cycles == 20:
                    rollout_evidence['completed_twenty'] += 1
        env.compute_reward = record_reward

    def save(label):
        nonlocal last_save
        model.eval()
        artifact = out / (label + '.pt')
        torch.jit.script(model).save(str(artifact) + '.pending'); os.replace(str(artifact)+'.pending', artifact)
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'steps': steps,
                    'teacher_sha256': teacher_sha}, str(out/'student_last.ckpt')+'.pending')
        os.replace(str(out/'student_last.ckpt')+'.pending', out/'student_last.ckpt')
        frozen = torch.load(teacher_path, map_location=learner_device)['model']
        assert all(torch.equal(v, frozen[k]) for k,v in teacher.model.state_dict().items())
        progress = dict(started_at=started, updated_at=datetime.now(timezone.utc).isoformat(),
                        steps=steps, samples_collected=steps*env.num_envs, metrics=metrics,
                        teacher_sha256=teacher_sha, teacher_unchanged=True, teacher_training=False,
                        inputs=['measured_joint_positions', 'previous_sent_targets', 'joint_target_history'],
                        external_phase_input=False, normalization_included=True, action_scale=model.action_scale, policy_hz=20,
                        artifact=str(artifact), artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                        hardware_ready=False, status='candidate:closed_loop_acceptance_pending',
                        stopped_by_signal=STOP)
        progress['supervision_phase_counts'] = phase_counts.tolist()
        progress['runtime_source_sha256'] = source_receipts
        progress['samples_per_second'] = steps*env.num_envs/max(time.monotonic()-start_clock, 1e-6)
        progress['torch_peak_allocated_bytes'] = torch.cuda.max_memory_allocated(learner_device)
        progress['learner_device'] = learner_device
        progress['simulation_tensor_device'] = env.device
        progress['warm_start_sha256'] = warm_start_sha
        progress['warm_start_scope'] = 'encoder_only' if encoder_warm_start else 'full_model' if warm_start else None
        progress['goal_cycles'] = long_cycles
        progress['continuous_nut'] = continuous_nut
        if return_entry_clock:
            progress['return_progress_receipt'] = env.return_progress_receipt
        if continuous_nut:
            progress['continuous_nut_receipt'] = env.continuous_nut_receipt
            progress['rollout_cycle_evidence'] = rollout_evidence
        if coupling_spec is not None:
            progress['passive_coupling'] = coupling_spec
        progress['expert_anchor_fraction'] = anchor_fraction
        progress['expert_burnin_steps'] = expert_burnin
        progress['expert_anchor_size'] = anchor_size
        if actuator_spec is not None:
            progress['actuator_randomization'] = actuator_spec
        if anchor_size:
            progress['expert_anchor_phase_counts'] = torch.bincount(anchor['command'][:anchor_size,:4].argmax(-1), minlength=4).tolist()
        temp = out/'progress.pending'; temp.write_text(json.dumps(progress, indent=2)+'\n'); temp.replace(out/'progress.json')
        print(json.dumps(progress), flush=True); last_save = time.monotonic()

    try:
        while not STOP and steps < int(cfg.get('deploy_student_steps', 1000000)):
            with torch.no_grad():
                frame = obs['proprio_hist'][:, -1]
                raw = history[:, -3:].reshape(-1, 96).clone()
                target, updated, action = model(frame[:, :16], frame[:, 16:], history)
                expert, info = teacher._eval_select_action(obs)
                expert = (expert * (0.05/model.action_scale)).clamp(-1,1) * model.action_mask
                labels = {'obs':raw, 'hist':updated, 'action':expert,
                          'command':obs['obs'][:,96:101], 'latent':info['extrin_record']}
                phase_counts += torch.bincount(labels['command'][:,:4].argmax(-1), minlength=4)
                indices = (torch.arange(env.num_envs, device=learner_device) + cursor) % capacity
                for key in replay: replay[key][indices] = labels[key]
                cursor = (cursor + env.num_envs) % capacity; size = min(size + env.num_envs, capacity)
                # Preserve the final buffer of the initial all-expert rollout.
                # This bounded 25% replay option is local; DAgger's paper
                # aggregates datasets, rather than retaining only recent states.
                if anchor_fraction and steps == expert_burnin:
                    anchor_size = size
                    for key in replay:
                        anchor[key][:anchor_size].copy_(replay[key][:anchor_size])
                    assert (torch.bincount(anchor['command'][:anchor_size,:4].argmax(-1), minlength=4) > 0).all()
                    torch.save({key:value[:anchor_size].cpu() for key,value in anchor.items()}, out/'expert_anchor.pt')
                # First collect expert trajectories, then decay the expert mixture
                # to zero. Phase/latent labels never enter the student forward call.
                beta = max(0., min(1., 1. - (steps-expert_burnin)/2000.))
                execute = torch.where(torch.rand((env.num_envs,1), device=learner_device) < beta, expert, action)
                next_obs, _, done, _ = env.step(execute)
                history = updated
                reset = done.flatten().bool()
                next_frame = next_obs['proprio_hist'][:, -1]
                history[reset] = model.reset_history(next_frame[reset,16:], next_frame[reset,16:])
                obs = next_obs
            if size >= batch:
                for _ in range(2):
                    sample, anchor_count = sample_with_expert_anchor(replay, size, anchor, anchor_size, batch, anchor_fraction)
                    predicted, logits, clock, latent = model.predict(sample['obs'], sample['hist'])
                    action_loss = ((predicted - sample['action'])[:, [0,1,2,3,12,13,14,15]]**2).mean()
                    phase_loss = torch.nn.functional.cross_entropy(logits, sample['command'][:,:4].argmax(-1))
                    clock_loss = ((clock-sample['command'][:,4:5])**2).mean()
                    latent_loss = ((latent-sample['latent'])**2).mean()
                    loss = action_loss + phase_loss + clock_loss + latent_loss
                    assert torch.isfinite(loss)
                    optimizer.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.)
                    optimizer.step()
                metrics = dict(loss=float(loss), action_mse=float(action_loss), phase_ce=float(phase_loss),
                               clock_mse=float(clock_loss), latent_mse=float(latent_loss), expert_probability=beta)
                metrics['phase_accuracy'] = float((logits.argmax(-1)==sample['command'][:,:4].argmax(-1)).float().mean())
                metrics['expert_anchor_samples_per_batch'] = anchor_count
            steps += 1
            if steps == 10 or time.monotonic()-last_save >= 120:
                save('candidate_%07d' % steps)
        save('student_last')
    finally:
        env.gym.destroy_sim(env.sim)


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, stop)
    main()
