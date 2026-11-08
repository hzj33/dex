# E002 nut-contact forward reward comparison — 2026-09-08

## Authorized hypothesis and change

The user explicitly authorized the pending exception: "允许，你拥有最高权限，我要你把目标完成". The task remains cloud PPO teacher learning one 30-degree opposing-finger M24 stroke. Full cyclic release/return work is not started.

Following [Tac2Motion Eq.3](https://arxiv.org/html/2509.17812v1), adapt contact-weighted rotation to opposing nut contacts: `4 * min(c_nut_index, c_nut_thumb) * normalized_new_high_water`. This is a local adaptation, not an exact reproduction. The previously audited normal-only moment gate suppresses a cooperative 31.54-degree diagnostic reference. Its measured friction limitation is not hidden.

Only the training forward multiplier changes. Nut-only records exclude bolt contact. All weights, high-water/reverse logic, invalid cost, contact/torque thresholds, strict pair gate0.50, milestones, success, reset pose743, physics and policy interface are unchanged. Legacy `forward_source=pair_quality` is retained. Source edits: `dexscrew/tasks/xhand_pasini.py` and E002 YAML. Diagnostic/evaluator edits only record the new source and forward quality; strict evaluator gates are unchanged.

## Validation and controlled physical comparison

- Actual reward-expression checks: legacy mode, nut-contact progress at zero normal gate, bolt-only rejection, distance rejection, static/revisited high water, reverse cost and fractional confidence pass. A0 and A0c diagnostic unit checks pass on local and cloud Python.
- Fresh A0c JSON explicitly `passed`, 4/27 candidates; selected candidate19. SHA256 `6e344b6ccf47fa9331a8b3a7d7af2887ee43a99b9512dad293ae8bbe21b5e6b7`.
- Same bank and all strict thresholds. Per-step force, distance, normal axial moment, nut angle, joint pose and target match the preceding A0c exactly (maximum absolute difference0).
- Candidate19 probe cumulative reward changes −0.1502897273 -> +9.2365642586 on this identical trajectory. This verifies restored signal, not learned success.
- Complete per-candidate comparison: pipeline `a0c_reward_comparison.json`; all forces, angles, both signs, condition counts and top-k remain in `a0c/manifests/a0c_diagnostic.json`.

## Cloud execution and verdict

Pipeline: `outputs/cloud_pipeline_m24_e002_nutcontact_20260908_175230`.
Exact scripts, start/end times, commands, resolved configuration, status codes and source hashes are retained there. Smoke is separately timeout60; PPO separately timeout300 with completed-update preservation. Resource profile12288 environments /24576 minibatch, RTX4090/49140MiB. No local training.

Smoke ran17:54:05–17:55:06+08, expected124. PPO ran17:55:06–18:00:13+08, timeout300 plus7s completed-update grace, expected124. Best is update10, reward−1.01957; last is completed update11, counter1,769,472 (includes the initial batch offset). Both are inference snapshots, not optimizer-resume state.

Each frozen checkpoint was evaluated at512 environments,100 steps, first episode before internal reset, deterministic mean actions. No terminal suppression:

| Result | Best | Last |
|---|---:|---:|
| Strict success | 0/512 | 0/512 |
| Reached30deg | 0/512 | 0/512 |
| Median net angle | 26.0711deg | 18.3390deg |
| Max net angle | 28.6369deg | 22.3048deg |
| Median net/gross efficiency | approximately1 | approximately1 |
| Strict valid / legacy credited steps | 41.20% | 50.45% |
| Workspace termination | 469/512 | 512/512 |
| Contact-loss / acquire-timeout termination | 1 /42 | 0 /0 |

`phase.txt` and `acceptance.json` are `rejected:fixed_step_numeric_gate`. Neither is an accepted teacher. The goal remains incomplete. No headed acceptance,30-minute extension,B/C or student run is claimed.

Interpretation: rotation improves over the near-zero preceding policies, but typical trajectories plateau near3.7deg while thumb force rises toward30N, then rotate sharply just before thumb contact is lost. This is not stable opposing-finger30deg driving. Total angle alone would misclassify the result.

![Frozen-policy physical traces](../outputs/cloud_pipeline_m24_e002_nutcontact_20260908_175230/policy_comparison.png)

Checkpoint directory: `outputs/XHandPasiniM24NutBolt_teacher/m24_e002_nutcontact_s42_20260908_175230_probe_12288_24576/stage1_nn`.

- `best_reward_-1.02.pth`: SHA256 `b1e8123447fc31340373a93af8ac93cc7d7e988a6b67e8a4594b91bfbad8ef31`.
- `last.pth`: SHA256 `9859db2503758d11b0c92e0c1f56c27ce65270d00e89d177d528a67356fadd65`.

Full training/TensorBoard/configs/checkpoints/evaluations are copied locally and cloud checkpoint hashes match.

## Frozen-policy follow-up diagnostics, no production fixes

1. Target accumulation: compare unchanged control to target error clamping derived from existing torque_limit/kp,8 environments per condition per model. Best median26.27 ->23.83deg; last19.41 ->18.77deg, all still fail. Target error exceeds1rad, but this clamp does not solve the failure. Do not apply it as a demonstrated fix.
2. Startup settling: compare unchanged control to6 zero-action startup steps, matching A0c settling. Best26.27 ->23.13deg; last19.41 ->18.80deg, all workspace failures. Do not apply this change or retrain from this evidence.
3. Contact-point gait: `m24_e002_contact_gait_diagnostic.py` shifts the previously independently validated COM Jacobian to live nut-contact centroids and evaluates joint-velocity-induced tangential velocity. It is a kinematic direction diagnostic, not an inferred force/torque. In the30–60 step stall window, both mean finger velocities are positive for best; direction-only reward is therefore not established as a fix. Last also contains many opposed-motion samples. Do not turn these kinematic samples into strict torque passes.

The first gait run failed before a completed report because the diagnostic indexed the flat DOF-state tensor as3D; `gait_audit_retry` fixes only that diagnostic view. Preserve the failed directory as invalid, not a result.


## Physics-substep follow-up: do not substitute endpoint kinematics for torque

The successful reference's last-substep nut velocities are nearly zero despite cumulative joint angle31.54deg. A fresh diagnostic-only wrapper now records all ten200Hz physics endpoints per20Hz policy step, with same-step nut-specific normal force/moment reconstruction. It preserves all production physics and reward settings.

| Reference,800 physics samples | Current velocity iterations0 | Diagnostic override1 |
|---|---:|---:|
| Actual joint angle change | 31.54448deg | 30.02368deg |
| Sum of sampled qdot ×0.005s | 0.79903deg | 0.00296deg |
| Maximum strict pair gate | 0 | 0 |
| Strict passing samples | 0 | 0 |

Current reference thumb normal axial moment is negative in all800 samples (range−0.15385 to−0.01669Nm). Thus sampling earlier within the policy period does not recover the missing strict pair contribution. The original geometric motion remains reproducible, but it is not a validated full-wrench or velocity-consistent torque-driven success. Endpoint tangential-velocity/slip summaries must not be used as a new reward basis on this evidence.

[NVIDIA's PhysX simulation documentation](https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html) distinguishes position/geometric-error correction, internal TGS steps and unbiased velocity iterations. It gives a plausible mechanism for differing displacement and endpoint velocities, not proof of a bug in this older Isaac Gym build. Changing velocity iterations to1 did not fix the discrepancy or gate; it is diagnostic only and is NOT adopted. The available Gym samples do not expose every internal solver iteration. No claim that all angle change is spurious is warranted either.

Raw data and exact scripts: `reference_substeps/`, `reference_velocity1/`, `substep_summary.json`, `velocity1_summary.json`. Reference-control gait results are in `reference_gait/` and `contact_gait_comparison.json`. The gait helper's contact velocities are diagnostics, not a validated full friction source.

## Current state and single next action

The authorized reward comparison is complete; goal is NOT complete. There is no pending approval for this completed experiment. The unchanged strict gate rejected both learned models, and no evidence-backed second production fix emerged from target-clamp, settling, gait or velocity-iteration diagnostics. Production source retains only the authorized nut-contact forward multiplier from this session.

Cloud is idle; all artifacts and best/last models are preserved locally and remotely. Do not repeat rejected training, start30minutes, or advance toB/C. The single next action is a diagnostic-only known-torque control on the M24 nut, with and without finger contact, to separate actual angular motion, sampled angular velocity and constraint/contact response before defining any further contribution or gait reward. Keep weights and strict pair gate fixed and rerun A0c after any demonstrated task fix. This next diagnostic has NOT been executed in this session.
