#!/usr/bin/env python3
"""Deterministic E002 PPO evaluation with first-episode, pre-reset metrics.

Only restore/inference are called. No training and no terminal suppression.
"""
import hashlib
import json
import math
from pathlib import Path

from calibrate_m24_e002_a0 import hydra, isaacgym_task_map, omegaconf_to_dict, OmegaConf, torch
import numpy as np
from omegaconf import open_dict
from dexscrew.algo.ppo.ppo import PPO
from m24_e002_a0c_diagnostics import measure_step
from m24_e002_evaluation_contract import assess_learning


@hydra.main(config_name="config", config_path="../configs")
def main(cfg):
    assert cfg.test and int(cfg.task.env.numEnvs) >= 512
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    calibration_path = Path(str(cfg.e002_calibration))
    calibration = json.loads(calibration_path.read_text())
    assert calibration["status"] == "passed" and calibration["pose_index"] == 743
    checkpoint = Path(str(cfg.checkpoint)).resolve()
    output = Path(str(cfg.e002_eval_out)).resolve()
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open_dict(cfg.task.env):
        cfg.task.env.customInitDofPos = calibration["custom_init_dof_pos"]
        cfg.task.env.e002_motion.force_sign = calibration["force_sign"]
    env = isaacgym_task_map[cfg.task_name](config=omegaconf_to_dict(cfg.task),
        sim_device=cfg.sim_device, graphics_device_id=cfg.graphics_device_id, headless=True)
    assert env.e002_torque_source == calibration["torque_source"]
    assert env.e002_forward_source == calibration.get("forward_source", "pair_quality")
    assert env.e002_acceptance_mode == calibration.get("acceptance_mode", "strict_normal_proxy")
    agent = PPO(env, str(output.parent / "inference"), full_config=cfg)
    agent.restore_test(str(checkpoint))
    agent.set_eval()
    obs = env.reset()
    n = env.num_envs
    alive = np.ones(n, dtype=bool)
    final = [None] * n
    credited = np.zeros(n, dtype=int)
    valid_credited = np.zeros(n, dtype=int)
    single_credited = np.zeros(n, dtype=int)
    rewarded = np.zeros(n, dtype=int)
    contact_rewarded = np.zeros(n, dtype=int)
    traces = []
    pair = [env.fingertip_handles[0], env.fingertip_handles[3]]
    original = env._compute_e002_a_reward
    thresholds = calibration["thresholds"]
    step_index = 0

    def array(value):
        return value.detach().cpu().numpy().copy()

    def observed_reward(*args, **kwargs):
        paid_before = array(env.e002_paid_high_water)
        result = original(*args, **kwargs)
        angle = array(env.e002_direction * (env.nut_dof_pos[:, 0] - env.e002_theta_start))
        gross = array(env.e002_gross_travel)
        reverse = array(env.e002_reverse_travel)
        high = array(env.e002_high_water)
        valid_progress = array(env.e002_valid_pair_progress)
        proxy_progress = array(env.e002_normal_proxy_progress)
        reason = array(env.e002_termination_reason)
        force = array(torch.norm(env.contact_forces[:, pair], dim=-1))
        nut_force = array(torch.norm(env.e002_normal_contact_force, dim=-1))
        distance = array(torch.norm(env.rigid_body_states[:, pair, :3] -
            env.rigid_body_states[:, env.screw_nut_rb_handle, :3].unsqueeze(1), dim=-1))
        raw_torque = array((-env.e002_normal_contact_moment * env.rot_axis_buf[:, None]).sum(-1))
        contact = array(env.e002_contact_state)
        delta_new = array(env.e002_paid_high_water) - paid_before
        forward_quality = array(env.e002_forward_quality)
        active_ids = np.where(alive)[0]
        step_rows = []
        for i in active_ids:
            score = measure_step(force[i], distance[i], raw_torque[i], 0., thresholds)
            signed = score["force_signs"][str(int(env.e002_force_sign))]
            contribution = contact[i] & (np.asarray(signed["signed_torque"]) >= env.e002_torque_min)
            paid = bool(delta_new[i] > 0 and signed["pair_gate"] > 0)
            credited[i] += paid
            valid_credited[i] += paid and contribution.all() and signed["pair_gate"] >= env.e002_pair_gate_min
            single_credited[i] += paid and contribution.sum() == 1
            actual_paid = bool(delta_new[i] > 0 and forward_quality[i] > 0)
            rewarded[i] += actual_paid
            contact_rewarded[i] += actual_paid and contact[i].all()
            row = dict(env_id=int(i), step=step_index, net_angle=float(angle[i]), high_water=float(high[i]),
                gross_angle=float(gross[i]), reverse_angle=float(reverse[i]),
                valid_pair_progress=float(valid_progress[i]), reason=int(reason[i]),
                force=force[i].tolist(), distance=distance[i].tolist(),
                nut_force=nut_force[i].tolist(),
                signed_torque=signed["signed_torque"], pair_gate=signed["pair_gate"],
                contact=contact[i].tolist(), credited=paid,
                forward_quality=float(forward_quality[i]),
                rewarded_progress=actual_paid,
                normal_proxy_progress=float(proxy_progress[i]),
                action=array(env.actions[i]).tolist())
            step_rows.append(row)
            if reason[i] or step_index == 99:
                final[i] = dict(row, steps=step_index + 1,
                    credited_steps=int(credited[i]), valid_credited_steps=int(valid_credited[i]),
                    single_credited_steps=int(single_credited[i]))
                alive[i] = False
        traces.extend(step_rows)
        return result

    env._compute_e002_a_reward = observed_reward
    with torch.no_grad():
        for step_index in range(100):
            action, extra = agent._eval_select_action(obs)
            obs, _, _, _ = env.step(action, **extra)
    assert all(row is not None for row in final)
    net = np.asarray([r["net_angle"] for r in final])
    gross = np.asarray([r["gross_angle"] for r in final])
    reverse = np.asarray([r["reverse_angle"] for r in final])
    efficiency = net / np.maximum(gross, 1e-6)
    reverse_ratio = reverse / np.maximum(gross, 1e-6)
    reason = np.asarray([r["reason"] for r in final])
    credited_count = int(credited.sum())
    metrics = dict(success_rate=float((reason == 5).mean()), reached_30deg_rate=float((net >= math.pi / 6).mean()),
        median_net_angle=float(np.median(net)), max_net_angle=float(net.max()),
        efficiency_median=float(np.median(efficiency)), efficiency_p10=float(np.quantile(efficiency, .1)),
        reverse_ratio_median=float(np.median(reverse_ratio)), reverse_ratio_p90=float(np.quantile(reverse_ratio, .9)),
        credited_steps=credited_count,
        valid_credited_ratio=float(valid_credited.sum() / credited_count) if credited_count else None,
        single_credited_ratio=float(single_credited.sum() / credited_count) if credited_count else None,
        hard_speed_termination_rate=float((reason == 4).mean()),
        rewarded_steps=int(rewarded.sum()),
        pair_contact_rewarded_ratio=float(contact_rewarded.sum() / rewarded.sum()) if rewarded.sum() else None,
        termination_counts={str(i): int((reason == i).sum()) for i in range(7)})
    gates = dict(success=metrics["success_rate"] >= .9 and metrics["reached_30deg_rate"] >= .9,
        efficiency=metrics["efficiency_median"] >= .9 and metrics["efficiency_p10"] >= .8,
        reverse=metrics["reverse_ratio_median"] <= .05 and metrics["reverse_ratio_p90"] <= .1,
        contribution=credited_count > 0 and metrics["valid_credited_ratio"] >= .95 and metrics["single_credited_ratio"] <= .02,
        hard_speed=metrics["hard_speed_termination_rate"] < .01)
    assessment = assess_learning(final, traces, n, 100)
    legacy_status = "passed:numeric_gate" if all(gates.values()) else "failed:numeric_gate"
    active_status = (assessment["status"] if env.e002_acceptance_mode == "contact_progress" else legacy_status)
    report = dict(status=active_status,
        training_launched=False, policy_mode="deterministic_mean", num_envs=n, fixed_steps=100,
        episode_scope="first episode per environment, recorded before internal reset",
        checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        calibration=str(calibration_path), calibration_sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
        torque_source=env.e002_torque_source, forward_source=env.e002_forward_source,
        acceptance_mode=env.e002_acceptance_mode,
        learning_assessment=assessment, goal_completed=False,
        legacy_numeric_status=legacy_status,
        legacy_credit_definition="new high-water with positive normal proxy gate; distinct from actual rewarded_steps",
        metrics=metrics, legacy_gates=gates,
        gates=assessment["health_checks"] if env.e002_acceptance_mode == "contact_progress" else gates,
        first_episodes=final, first_episode_steps=traces)
    OmegaConf.save(cfg, output.with_suffix(".config.yaml"), resolve=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "metrics", "gates")}, indent=2))
    env.gym.destroy_sim(env.sim)
    passed = assessment["continuation_allowed"] if env.e002_acceptance_mode == "contact_progress" else all(gates.values())
    raise SystemExit(0 if passed else 4)


if __name__ == "__main__":
    main()
