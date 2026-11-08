# E002 reset correction and physical reachability evidence

The learned-policy goal is not complete. The observation defect is corrected,
but the new PPO best/last models still fail. A separate diagnostic reference
controller now demonstrates a 31.54-degree, two-finger-dependent motion that
receives zero progress reward under the current normal-moment gate. This is
stronger evidence for examining that gate than another unexplained weight change.

## Scope and implementation

- Read local and live cloud handoffs and the active plan before execution.
- One production change: `dexscrew/tasks/xhand_pasini.py` initializes E002 reset
  history before constructing the actor observation. Cold and subset resets
  receive the current pose; nonreset observations and unrelated tasks retain
  their prior behavior. Reward formulas, weights, gates and physics are unchanged.
- Added `scripts/audit_m24_e002_reset_history.py` for a real two-env check;
  added `scripts/probe_m24_e002_contact_control.py` for diagnostic reference
  control and per-finger command ablations. The probe cannot start PPO.
- All training is cloud-only. Each training phase has its own timeout. No
  30-minute run, B/C, student, or unapproved alternative reward was launched.
- Literature/official-code comparison and the unapplied proposal are in
  `docs/m24_e002_literature_alignment_2026-09-08.md`. The user explicitly requires
  evidence-grounded reward work; this is now a stable rule in `AGENTS.md`.

## Fixed-observation validation and PPO

Authoritative cloud/local pipeline:
`outputs/cloud_pipeline_m24_e002_reset_history_20260908_171000`.
Exact commands, snapshots, timestamps, statuses and diagnostics are retained.

- `audit_m24_e002_reset_history.py`: passed; cold error0, subset reset error0,
  nonreset observation change0, all observations finite.
- Existing A0/reward-expression/diagnostic checks passed inside the A0c launcher.
- Fixed-pose A0c: passed, same 27-action bank and original thresholds; SHA256
  `8e40add7eae7b95349439e216eb5018aa454bd1142d4eb850f54fe176498ca68`.
- Both previous models were reevaluated first: still 0/512 successes.
- New smoke passed expected timeout124. PPO 12288 envs / minibatch24576,
  17:14:08–17:19:26 +08, timeout300 plus 18-second in-flight update grace,
  expected124. Stable throughput approximately 5.6–6.0k actor steps/s.
- Best reward advanced to update11 (-0.90356); last completed update12,
  counter1,916,928. Counters include an initial batch offset. Final-update
  inference weights are saved; these are not optimizer-resume snapshots.

| Model | 512-env success | Median net angle (rad) | Contact-loss termination | Credited steps |
|---|---:|---:|---:|---:|
| best reward | 0/512 | 1.18154e-12 | step10, all envs | 0 |
| last completed | 0/512 | 1.35711e-11 | steps10–18, all envs | 0 |

Both fixed 100-step evaluations exit4, with first episodes captured before
reset; `acceptance.json` / `phase.txt` remain `rejected:fixed_step_numeric_gate`.
No headed policy acceptance is claimed. Higher return did not mean rotation.

Checkpoint directory:
`outputs/XHandPasiniM24NutBolt_teacher/m24_e002_resethistory_s42_20260908_171300_probe_12288_24576/stage1_nn`.

- `best_reward_-0.90.pth`: SHA256
  `73dedf490603b90280c1fb6e34bac9287f898062a35ee4207cdf5db7c738279a`.
- `last.pth`: SHA256
  `a50570c0b61cf6ea6693e467408d11339680e2f0affdb0f5bfde24fab3c69891`.

Both hashes match local/cloud copies. Full model evaluation and training
scalars are in `results_summary.json` and the raw evaluation JSONs.

## Diagnostic reachability and paired controls

The reference controller computes a joint torque target `J_contact.T * F_des`,
then a PD target `q + tau/kp`, delivered through the existing bounded action
interface. It does not apply virtual torque to the nut, change the asset,
controller gains, action scale, physics rate or reward. `F_des` is a commanded
force, explicitly not a measured total contact wrench. The task is nominal
candidate743, only index/thumb active; probe termination suppression is saved
and makes this a diagnostic, not a training/evaluation acceptance run.

Initial twelve proposals use normal targets 2/4/8/16 N and tangential ratios
0.25/0.50/0.75. At 16 N and ratio0.5, a 12-env probe reaches31.19 degrees with
80/80 pair-contact endpoints and gate0. Six-env paired controls reproduce
31.54 degrees. Small backend/batch trajectory differences are not hidden.

| Tangential ratios: index / thumb | Final nut angle (degrees) | Pair contact /80 | Maximum normal gate |
|---|---:|---:|---:|
| 0.5 / 0.5 | 31.54448 | 80 | 0 |
| 0 / 0.5 | 0.001103 | 80 | 0 |
| 0.5 / 0 | 0.001433 | 80 | 0 |
| 0 / 0 | approximately0 | 80 | 0 |
| -0.5 / 0.5 | 0.000624 | 80 | 0 |
| 0.5 / -0.5 | 0.001423 | 1 | 0.31866 |

Source: `contact_gpu_force_audit/diagnostic.json`, with all raw contacts,
actions, forces, moments, distances, angles and condition flags. Both positive
tangential commands are necessary in this particular controlled comparison.
This supports two-finger cooperation; it does not measure each finger's full
instantaneous wrench, prove universal necessity, or establish an RL policy.

![Diagnostic control and normal-moment gate](../outputs/cloud_pipeline_m24_e002_reset_history_20260908_171000/contact_gate_evidence.png)

## Friction-source cross-check

A real CPU-PhysX reference was explicitly checked with `physx_use_gpu=False`.
Its both-finger control reaches28.63 degrees; physical trajectories differ, so
it is not pooled with the GPU results. Raw `lambdaFriction` fields are zero
in both backends. CPU fingertip-resultant minus nut-normal force can differ by
8.34 N, but raw records identify thumb-bolt contacts as the cause. It is not a
friction-force measurement. Clean contact samples in both backends reconstruct
the resultant within9.54e-7 N of normal forces alone.

`contact_source_audit.json`: GPU samples have zero non-nut contacts and zero
false nut-contact cases; CPU has9 non-nut-contact finger samples and5 cases
where resultant contact appears on while nut-specific force is zero. The
active GPU configuration is unchanged; no speculative force-source fix is made.
The `contact_ablation_cpu` directory misleadingly names only the requested
sim device: its manifest still says GPU PhysX, so it is explicitly classified
as a GPU replicate. `contact_ablation_cpu_physx` is the actual CPU reference.

## Single next action and current boundary

Proceed with the concrete literature-style contact-weighted progress comparison
only after the user's explicit training-gate exception. All coefficients,
A0c and strict physical acceptance gate0.50 remain unchanged. The proposal is
prepared, not applied; silence is not authorization. Any contact-quality
implementation must distinguish nut contact from bolt contact using the
already available nut-specific normal records. The strict gate still needs a
valid treatment of unmeasured friction before final behavior can be accepted.

Do not repeat the failed PPO configuration, infer success from return, or
present this reference controller as a learned policy. Current cloud is idle
(no tmux/train/eval/probe job, GPU1/49140 MiB and0% utilization). Full artifacts,
checkpoints, primary-code snapshot, figure and commands are synced locally/cloud.
