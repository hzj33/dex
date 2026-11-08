# E002 candidate-743 cloud recovery, 2026-09-08

The user asked the agent to own cloud execution through A0c, smoke and a
five-minute PPO probe. All training is cloud-only. An explicit passed A0c JSON
is mandatory; no 30-minute run or E002-B/C is authorized. Candidate-743,
reward weights, contact thresholds, torque thresholds and pair gate stay fixed.

## Evidence and bounded correction

The prior contact audit demonstrated incorrect link-origin application points.
Its recommendation to require a full friction wrench before any correction
was an unnecessarily broad prerequisite: the existing net-contact tensor was
itself reconstructed by normal contact forces within 3.48e-6 N. This session
corrects the application-point geometry of that same measured force component.
It does **not** claim to measure friction or full contact wrench.

- `..._m24_e002_sensorcheck_20260908_143436`: candidate 0/64, actual contacts
  and an object translated +0.5 m as negative control, 9 sensors, 1280 physics
  samples. Solver sensors fail to reproduce the net-contact tensor and retain
  a nonzero joint-reaction signal without contact. They were not substituted
  into the reward. Actual static fingertip net force reaches 1.48119 N while
  solver-only sensor force reaches only 0.08734 N; maximum discrepancy is
  1.395 N. Sensor forward-dynamics + solver additivity holds within 4e-9.
- `..._m24_e002_contactjac_20260908_143937`: independent URDF forward
  kinematics verifies Gym linear Jacobians at body COM, not link origin.
  COM Jacobian errors are <7.2e-8; treating them as origin Jacobians yields
  errors 0.01145/0.01839. Measured contact-point Jacobians align both proposed
  tangents with cosine >0.99999. The measured basis and deterministic action
  banks are saved under this pipeline's `manifests/`.
- `..._m24_e002_driveaudit_20260908_150524`: actual moving candidate 64 shows
  last-step contact-point normal moments +0.02352/+0.00889 Nm, whereas the old
  proxy reports +0.01575/-0.02607 Nm under physical -Z / force_sign=-1.

These abbreviated paths all start with
`outputs/cloud_pipeline_m24_humantwist_e002`.

## Implemented source correction

- `dexscrew/tasks/contact_moments.py` reconstructs nut-side normal moments at
  actual nut contact points. It transforms contact local coordinates with the
  nut quaternion, uses validated `lambda * normal` force convention, and
  separates index-nut from thumb-nut records.
- E002 config selects `torque_source=contact_point_normal`. CPU tensor access
  enables the public contact API; simulation remains GPU PhysX at dt=0.005,
  10 physics steps per policy action. CUDA PPO uses the existing algorithm.
- `xhand_pasini.py` fetches the final CPU tensor step before matching contact
  records to poses and refreshes CUDA policy extras each step. Legacy tasks
  retain their default torque source. E002 weights and gate are unchanged.
- Diagnostic, calibration and controlled trace use the same torque source.
  Diagnostic JSON retains old proxy torque, new nut-specific force/moment,
  contact counts, both force signs and all per-step condition results.
- `import_m24_a0c_for_training.py` rejects failed JSONs, recomputes acceptance
  from per-step data, verifies task/config hashes and imports only an explicit
  pass. The existing cloud A1 launcher can use that JSON without another pose
  search. No new PPO algorithm or checkpoint has been produced at this stage.

## Validation and paired A0c

`outputs/cloud_pipeline_m24_e002_contactpoint_envapi_20260908_152940`
passed a 2-environment, 16-step simulator validation. Batched moments match an
independent scalar reconstruction; forces match same-step fingertip tensors;
all policy observations/extras reside on CUDA while simulation tensors are CPU.

The installed simulation-wide `get_rigid_contacts` interface returned invalid
body/env indices. Its failed validation is preserved in
`..._contactpoint_validation_20260908_152751`, including the debug rerun.
Production uses copied `get_env_rigid_contacts` results instead. The first
paired launcher at `..._contactpoint_pair_20260908_153013` also failed before
simulation due to an input-bank path missing `manifests/`; excluded from results.

Authoritative paired pipelines:

- `outputs/cloud_pipeline_m24_e002_link_origin_proxy_20260908_153140`
- `outputs/cloud_pipeline_m24_e002_contact_point_normal_20260908_153140`

Both use identical 73 actions, fixed pose, CPU tensors/GPU PhysX and a 20-step
probe. Force, distance, nut-angle and joint-position traces match **exactly**
across the pair. Only torque measurement/acceptance changes. Both JSONs fail.

| Torque source / sign | Contact steps | Torque steps | Gate steps | All steps | Passed candidates |
|---|---:|---:|---:|---:|---:|
| Origin proxy / -1 | 1184 | 4 | 0 | 0 | 0 |
| Contact point / -1 | 1184 | 28 | 0 | 0 | 0 |
| Origin proxy / +1 | 1184 | 65 | 2 | 2 | 0 |
| Contact point / +1 | 1184 | 622 | 5 | 3 | 0 |

There are 1460 probe samples per sign. Physical reaction convention is -1.
The +1 high-gate candidates push against the lower nut limit and have near-zero
net angle; they do not qualify. The best physical candidate 62 reaches gate
0.469620 at step 19, torque [0.052022, 0.048855] Nm and net 0.077226 rad.
Its earlier steps do not meet the gate, so no accepted streak exists.

## Action diagnostics before and after the correction

All runs are diagnostic-only, with their own timeout and recorded command,
source hashes, per-step JSON, status and timestamps. Small actions were tested
first. Larger actions were then confined to the existing policy range [-1,1].
No controller scale, stiffness, physics rate or threshold was increased.

| Pipeline suffix | Change under test | Probe steps | Result |
|---|---|---:|---|
| `contactbasis_20260908_144533` | Correct contact-point tangent basis, 0.025/0.05/0.1 | 10 | Fail; 645/670 contact, zero torque/gate passes |
| `contactamp_20260908_144919` | Amplitudes 0.15/0.2/0.3 | 10 | Fail; zero torque/gate passes |
| `inward_20260908_145642` | Inward-biased tangents, 0.1/0.3/0.5 | 10 | Fail; best net 0.000414 rad |
| `fullrange_20260908_150220` | Same family, 0.65/0.8/1.0 | 10 | Fail; candidate 64 net 0.098274 rad, streak 0 |
| `window20_20260908_151047` | Same 73 actions, longer observation | 20 | Fail; best streak 1 under +1, zero under -1 |

These suffixes start with
`outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_`.

After the contact-source correction,
`outputs/cloud_pipeline_m24_e002_action_refine_20260908_153426/a0c`
tests 33 candidates centered on physical candidate 62: amplitudes 0.9/1.0,
inward ratio 0.9/1.0/1.1/1.2, thumb ratio 0.4/0.5/0.6/0.7, positive tangent.
It still fails at 20 steps. Candidate 11 reaches physical pair gate 0.674151
and net 0.124820 rad, but only one accepted step; candidate 27 also passes only
one step. These are correlated candidate results, not joined global maxima.

The next diagnostic extends this same bank to 30 steps (1.5 seconds) because
the original physical candidate 62 was still approaching the threshold at
the 20-step boundary. Required streak remains 5, pair gate 0.50, both torques
>=0.01 Nm, and net angle >=pi/180. This explicitly changes observation duration,
not acceptance thresholds or the fixed 20 Hz / 200 Hz timing.


## A0c passed; authorized cloud training started

`outputs/cloud_pipeline_m24_e002_refine_window30_20260908_153552` failed:
longer observation alone did not prevent thumb loss. A deterministic action
schedule then tapered the same constant direction before the observed loss.
The initial 64-candidate taper bank at
`outputs/cloud_pipeline_m24_e002_taper_20260908_154025/a0c` improved the best
streak to 3, retaining pair contact throughout 30 steps, but still failed.

The focused 27-candidate refinement at
`outputs/cloud_pipeline_m24_e002_taper_refine_20260908_154240/a0c` explicitly
**passed** (simulator exit 0). Four candidates passed under force_sign=-1;
none under +1. The selected candidate 19 uses the parent 27 direction for
16 steps, then 0.25 times that increment for the remaining 14 steps. It has
7 consecutive accepted steps, net angle 0.1256776452 rad (~7.20 degrees),
maximum pair gate 0.6420834, valid fixed-pose hold and no unexpected reset.
The alternative candidate 12 has 6 consecutive accepted steps and retains
pair contact for all 30 probe steps. Both meet the unchanged written A0c gate.

Passed JSON SHA256:
`9a2ee6a0e46281a0a12631ead90138b38ea311f90ff45096cff48891f7ed8d5b`.

The JSON records every actual scheduled action. Selection and controlled trace
carry the full action-scale schedule; it is not represented as an unchanged
constant action. `scripts/build_m24_a0c_taper_bank.py` saves source hashes and
all schedule parameters. No task, reward weight, gate or control rate changed
during action refinement.

Cloud training pipeline:
`outputs/cloud_pipeline_m24_e002_passed_a1_20260908_154343`, tmux
`m24-e002-a1`. Import guard revalidated the exact JSON and task/config hashes.
A0 unit checks, guarded import and controlled trace exited 0. The one-env smoke
completed with expected timeout exit 124 at `2026-09-08T15:44:59+08:00`, healthy
PPO updates at ~19 FPS and a saved checkpoint. The five-minute phase then
started at 12288 environments / minibatch 24576, with its own timeout 300.
Initial stable throughput is ~5.8k actor steps/s, VRAM ~10066 MiB / 49140 MiB,
GPU utilization ~35%. CPU contact access is a measured throughput cost.

A deterministic 512-env, 100-step evaluation is queued after training; it uses
`evaluate_m24_e002_fixed.py` and captures first-episode metrics before internal
resets. Healthy training is not an E002-A behavior pass. No 30-minute extension,
E002-B/C or student run is authorized or queued.

A0c pass limitation: selected accepted steps are 17–23. Their summed positive
angle is only 0.0007598922 rad (~0.0435 degrees), including steps 18–21 with
4.47e-7 to 7.45e-7 rad increments. Most of the candidate’s 0.12568 rad net
rotation occurred earlier. This satisfies the existing positive-step plus
whole-probe net-angle criteria; it does not establish a sustained, useful
30-degree pair-contributed stroke. No threshold was relaxed to obtain the pass.
The fixed-step learned-policy evaluation must decide the subsequent A gate.


## Completed result and checkpoint disposition

The five-minute PPO ran from `2026-09-08T15:44:59+08:00` to
`2026-09-08T15:50:00+08:00`, exit 124 as expected. The first resource pair
12288/24576 was healthy; no other capacity/training run was launched. Persisted
TensorBoard reports agent-step counter 1,622,016. Best reward remains -1.06909
from epoch 1 / counter 294,912. Only that best checkpoint was saved; no final
`last.pth` exists, so the evaluation is of the selected best-reward policy,
not an unsaved final optimizer state.

Checkpoint (full local/cloud copies):
`outputs/XHandPasiniM24NutBolt_teacher/m24_e002_contactpoint_s42_20260908_154343_probe_12288_24576/stage1_nn/best_reward_-1.07.pth`

SHA256: `699cee72ec4b1c6d33037577f13368ed0dacbf30089301e16eb50e97589fa21b`.

Fixed evaluation ran `15:50:02`–`15:50:24` (+08:00), exit 4, at 512 environments
and 100 deterministic policy steps. First-episode metrics are captured before
reset and retained individually plus per-step traces. All first episodes end
at 13–14 steps with contact-loss reason 2. Index remains in contact while the
thumb loses force; no environment reaches 30 degrees.

| Metric | Result | Required |
|---|---:|---:|
| Valid 30-degree success | 0/512 | >=90% |
| Reached 30 degrees | 0/512 | >=90% |
| Median net angle | 1.63298e-7 rad | Useful 30-degree progress |
| Maximum net angle | 1.74903e-7 rad | — |
| Median / P10 net-gross efficiency | 0.1633 / 0.1539 | >=0.90 / >=0.80 |
| Median / P90 reverse-gross | 0.00916 / 0.01282 | <=0.05 / <=0.10 |
| Credited progress steps | 0 | Valid contribution must be demonstrated |
| Hard-speed termination | 0/512 | <1% |
| Contact-loss termination | 512/512 | — |

Ratios use the existing denominator floor 1e-6; gross travel is itself near
zero, so these ratios are not evidence of useful motion. Contribution ratios
are null because there are zero credited steps, and that gate fails rather
than passing vacuously. Numeric failure triggers the plan's immediate reject
rule; headed acceptance was not performed or claimed. No longer training,
E002-B/C or student phase was launched. Cloud is idle with no tmux jobs and
1 MiB GPU usage. Pipeline `phase.txt` and `acceptance.json` record
`rejected:fixed_step_numeric_gate`.

Local/cloud Python, shell, deterministic A0 and diagnostic regression checks
passed; actual two-env contact reconstruction and the full simulator A0c passed.
All diagnostics, failed pilots, accepted JSON, training logs, evaluation,
TensorBoard and best checkpoint are preserved. The user's overall motion goal
is **not achieved**. The single next action is a cloud diagnostic comparison
of the selected PPO policy's first 14 steps against zero hold and the accepted
A0c schedule, isolating early thumb contact loss before proposing another
bounded change. Do not repeat this failed training configuration, change
weights, lower the gate, or launch another training phase without new evidence.
