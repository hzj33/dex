# E002 cycle learning, 2026-09-10

## Target and evidence

The user wants learned index/thumb drive, release, return, regrasp and repetition.
The preceding revised-contact policy demonstrated a stable approximately20degree
stroke (512/512 held20degrees for4contact steps), but no cycle was learned. This
experiment starts the missing phases rather than waiting for a mature30degree
single-stroke success rate. Cloud power-on previously failed with no inventory;
the user explicitly authorized local fallback. Cloud remains shutdown.

Primary sources:

- [Tac2Motion, II-B](https://arxiv.org/html/2509.17812v1) includes contact,
  release and contact-weighted rotational reward components. Its Eq3 uses
  contact-related weighting of lid rotation. It motivates explicit rewards for
  making and breaking contact; it does not establish our phase thresholds or
  supply the formula below. Its hand and virtual-lid setup differ from this task.
- [DexScrew](https://arxiv.org/html/2512.02011v1) shows alternating thumb/index
  contacts in learned rotation gaits. It supports measuring useful repeated
  contact changes rather than requiring identical50ms release timestamps.
  Simulated revolute-object rotation and real fastening are separate settings;
  this experiment does not claim complete physical thread fidelity.

Testable local hypothesis: the stable-drive policy can learn to unload and
release if intentional release no longer triggers the single-drive contact-loss
termination, the current phase is observable, and rewards correspond to the
requested measured events. A five-minute failure is learning evidence, not a
reason to restore unvalidated normal-moment gates.

## Implementation and local adaptations

Opt-in `XHandPasiniM24NutBoltE002Cycle`, implemented in
`dexscrew/tasks/e002_cycle.py` and integrated into `xhand_pasini.py`. The existing
E002 drive-only configuration remains available. No physics, actuator, bolt
fixation, finger mask, optimizer algorithm or actor dimension change. Normal-only
torque remains diagnostic. The cycle wrapper replaces single-drive termination
and reward when enabled, so contact loss is permitted during release and return.

Ordered events, measured every50ms:

1. Drive: at least20degrees new nut motion under actual pair nut contact, with
   stable acquired contact. This is the already demonstrated stroke scale.
2. Release: both nut-contact states off and both nut-normal force magnitudes
   below the existing0.15N off threshold for two steps. No same-step onset rule.
3. Return: fingers remain released and both fingertip-link radial sectors return
   within5degrees of their initially acquired directions. Radial clearance is
   unconstrained. This is a local return proxy, requiring visual inspection.
4. Regrasp: pair contact for two steps, retaining at least half of the20degree
   stroke. Count a cycle; resume drive. Global high-water never resets between
   cycles, so rotating backward and revisiting old angles earns no new progress.

Phase deadlines `[80,40,60,40]` steps terminate with separately recorded failure
codes; they cannot create a successful phase event. Workspace escape and a
>15degree/control-step angle discontinuity likewise cannot count a drive/cycle.
Two cycles is a numeric evaluation target; numeric completion alone does not
prove the full visible behavior requested by the user.

Reward is a task-specific adaptation, not a paper formula. It reuses established
scales: contact-weighted new high-water progress4; reverse motion penalty4;
unsupported new progress penalty4; phase-potential increment0.5; event bonus10
for completing a drive and1for release/return/regrasp; failure2 and jump8.
The existing5degree soft-speed penalty0.25 remains. Release potential is
`1-mean(F/(F+0.3N))`; return potential is `exp(-sector_error/5deg)` while released;
regrasp potential is existing opposing nut-contact quality. Potentials initialize
at phase switches, so switching a label cannot manufacture shaping reward.
No blind weight sweep is used. These scales/thresholds are hypotheses to inspect
against trajectories, not claims of published optimal settings.

Actor inputs remain101D: joint history96D plus four phase bits and elapsed
phase fraction. A concrete transfer issue was found before training: previously
constant phase bits had running variances around1e-7. Newly active phases would
normalize to thousands. `prepare_m24_e002_cycle_checkpoint.py` changes only
statistics for indices96:100 to unit variance and mean `[1,0,0,0]`; learned model
weights remain identical. Metadata records original statistics and both hashes.

## Checks and preflight

- `python -m py_compile` on task/helper/scripts.
- `scripts/test_m24_e002_cycle.py`: actual tensor class, ordered two-cycle events,
  one-finger release rejection, release away from return sector, partial reset,
  timeout failure, no revisited high-water credit and no large-jump drive event.
- Existing learning contract CPU tests also passed.
- Actual16-env first-episode baseline with normalized warm start:16first drives,
  0release/return/regrasp/second drive/two cycles;16release timeouts, median net
  24.252degrees. All values finite. This proves the old policy can reach the new
  release phase and identifies unloading as the first behavioral bottleneck.
- Fresh candidate743 diagnostic A0c explicitly `status=passed` against the new
  source snapshot. Old A0c hashes are not reused after task-source changes.

Evidence root: `outputs/local_pipeline_m24_e002_cycle_20260910`.
Starting checkpoint `cycle_warm_start.pth`, SHA256
`54251df13c21a6086fb643d58afdb7961985213f194f635f2a624f88310081e0`.
Source was prior stable drive `last.pth`, SHA256
`ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`.

## Bounded execution protocol

Pipeline `outputs/local_pipeline_m24_e002_cycle_20260910_probe3072` contains
per-phase exact commands, timestamps, logs, source copies, selected checkpoints,
and fixed512-env first-episode traces. Local RTX3060/12GiB, existing IsaacGym
Docker,3072envs/6144minibatch. Smoke60s then PPO300s, each with its own timeout;
124is expected at the training limit. Policy/normalizers are warm-started;
optimizer and step counter restart. Smoke learning does not overwrite the PPO
starting policy. Best and last are evaluated separately. No30minute or student
phase is launched.

The complete user goal is not yet achieved. Next action is to finish this run,
compare actual phase-event counts and trajectory geometry, and identify a
specific bottleneck before another bounded learning change.

## First-run update-size diagnostic

TensorBoard reveals a specific transfer instability: cycle run first recorded
update KL4.3352 at inherited initial learning rate0.005, versus configured
KL threshold0.02. The scheduler reduces the rate only after minibatch updates;
first reported post-scheduler rate is0.000658. The preceding stable-drive run's
first KL was0.3353, so the phase/reward transfer is substantially more disruptive.
Later cycle training becomes dominated by large-angle terminations and has no
observed release events. Fixed deterministic evaluation is still required.

[PPO's primary paper](https://arxiv.org/abs/1707.06347) motivates controlled policy
updates. [OpenAI's official PPO implementation](https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/ppo.py)
uses policy learning rate3e-4 and stops policy updates on excessive approximate
KL. Those settings are evidence for controlling update size, not directly
transferable optimal values for this shared actor-critic implementation.
If fixed evaluation confirms degradation, the next isolated hypothesis is to
start the same cycle run at1e-4 (near the existing scheduler's observed adapted
range), retaining all reward/phase/physics/exploration parameters. Compare first
KL and learned drive/release counts. This is a proposed local learning-rate
adaptation, not an implemented algorithm redesign or a claim of guaranteed gain.

## First comparison completed; directional release is physically feasible

First PPO ran16:19:12–16:24:18 +08,300s timeout plus6s completion grace,
42completed updates,1,585,152step counter including initial-batch offset. Smoke
and PPO exit124; both evaluations exit0. Last checkpoint SHA
`98faac287cb3ef999cbdbc347db9059713260356ccf0cacb16244504f0a0ad80`;
reward-best SHA `4c0f3daaa67c59c17cbe41a0d4f64fad6240bba22dc10f38a5984d024d1845eb`.

| Policy, fixed512 first episodes | First drive | Release | Median net angle | Terminations |
|---|---:|---:|---:|---|
| Last |505|0|56.99deg|505release timeout,7workspace|
| Reward-best |97|0|1.33deg|512workspace|

Neither learns return/regrasp/repetition. Last continues clamping/rotating during
release; env0forces reach roughly43/39N. The normal-pressure release potential is
near saturation under these loads. Greater rotation is not a cycle result.
Default selected drive checkpoint remains the preceding stable-drive policy.

`scripts/diagnose_m24_e002_release.py` follows the original policy to actual drive
completion, then applies a recorded action bank through the unchanged action
interface. It stops each first-episode measurement on release or failure.
55candidates (zero; each active single joint at±0.05/0.1/0.2; coupled Jacobian
radial directions) all enter release,0finish. Positive radial0.2 leaves forces
0/6.77N and23deg; this direction unloads without erasing the twist.

A directed8candidate follow-up finds6true releases. Index/thumb scales0.2/0.3
release both fingers with23.46deg remaining;0.4/0.4 reaches release with23.92deg.
All six have measured force0/0N and two released endpoints. This establishes
local action feasibility, not an RL success. The first diagnostic attempt had a
CPU-action/CUDA-mask mismatch before bank execution; corrected diagnostic moves
the action to the environment device, and the failedlog remains preserved.
No reward, physical criterion or production action was changed by these probes.

The isolated LR comparison is now running in
`outputs/local_pipeline_m24_e002_cycle_lr1e4_20260910_probe3072`.
Its `hypothesis.json` is in the sibling analysis root without `_probe3072`.

## Release-shaping counterfactual (not a weight sweep)

On the physically successful index0.2/thumb0.3 radial trajectory, the first
0.5seconds lowers force from9.86/23.69N to7.22/18.04N. Existing absolute-force
potential gives only0.00356shaping reward in that interval, despite unloading
both fingers. Normalizing force by its value at entry to release gives0.12659
for the same segment, with total release-shaping0.5versus0.48949. This supports a
specific delayed-feedback hypothesis: the original force/(force+0.3N) potential
is saturated at observed10–30N contacts.

Proposed local adaptation: measure fractional unloading from the recorded
release-entry force, keeping the0.5shaping scale and physical release criterion
unchanged. It makes early unloading observable to the learning objective without
increasing the total potential range. This is our diagnostic-supported adaptation
of explicit release shaping, not Tac2Motion's exact formula and not a claim of
policy-invariant potential shaping. Counterfactual JSON:
`outputs/local_pipeline_m24_e002_cycle_20260910/release_shaping_counterfactual.json`.

## LR comparison and release-potential implementation

LR1e-4 PPO ran16:29:16–16:34:23 +08, expected124. Best and last fixed evaluations
both512/512drive,0release,512release timeout, no deterministic workspace/jump
failures. Best median23.25deg; last58.52deg. FirstKL0.01071 is below0.02, but
later stochastic jump failures still dominate. Initial-LR reduction controls
initial update size; it does not establish a solution to the behavioral problem.
Checkpoints: last SHA `dc41fcb4392852fa8c376de689d99cda6996ed63dfd1d2c385146b6625a5dceb`,
best SHA `1bb758eab9b5ee387e355a7fc7f8202f2147fc4d5722014fbe741efdfe76e69c`.

Opt-in `release_potential=initial_relative_force` is now implemented. It records
both forces at the valid drive->release event and uses
`1-mean(clamp(F/F_entry,0,1))`, with denominator floor0.3N and unchanged0.5scale.
Physical event conditions, phase deadlines, other rewards and action distribution
are unchanged. Default mode remains `saturating_absolute_force` for comparisons.
Actual CPU tests verify0.1reward for20%unloading,−0.1for reloading,0.5total range,
unchanged two-step release events and isolated partial reset of entry forces.
Fresh A0c passed; relative-mode16-env physical preflight gives the same16drive,
0release baseline. Active pipeline `outputs/local_pipeline_m24_e002_relativeforce_20260910_probe3072`
starts from the same original normalized checkpoint, initialLR1e-4,3072/6144,
smoke60s/PPO300s then best/last fixed512-env evaluation. Exact host sequence:
`outputs/local_pipeline_m24_e002_relativeforce_20260910/run_after_a0c.sh`.

Source snapshot filename collision was found and repaired in the LR pipeline:
task and train configs share a basename. Training config snapshots now use
`train_`prefix; both original configuration hashes are recorded. This affects
archival names, not executed configs or training behavior.

## Literature-backed follow-up if full-start release remains absent

The user-supplied review recommends demonstration-state resets for difficult
long-horizon exploration. Verified primary source:
[DemoStart, III-A](https://arxiv.org/html/2409.06613v1). It records demonstration
states, trains from states with shorter remaining paths, and selects states
that sometimes succeed and sometimes fail. Its evaluation uses the original
target initial-state distribution. Its nut-and-bolt result is simulation; do
not infer real nut-threading transfer from the other tasks' transfer results.

If the current fixed-pose evaluation again shows0release, the next single action
is to capture and physically validate reset states along the now successful
scripted release, then implement a small release-state curriculum within PPO.
This is an initialization-distribution change, not copying demonstration actions
into PPO's on-policy rollout buffer. The complete goal must still be evaluated
from original candidate743 with no scripted actions or demonstration resets.
`record_m24_e002_policy.py` can archive DOF positions/velocities, actor roots,
joint targets, pre-roll observation history and cycle state. Such archives are
explicitly marked `reconstruction_validated=False`: PhysX contact/cache state is
not serialized, so replay/reset tests are required before training from them.
No student/distillation is authorized or implied by referencing this paper.

## Final result of this execution session

All training/evaluation/recording/replay jobs finished; no GPU compute process
remains. Cloud is still off, and local handoffs await a future cloud sync.
No30minute extension or student training occurred. Full user goal is incomplete.

Relative-force PPO ran16:38:21–16:43:26 +08,42updates,1,585,152counter including
the initial-batch offset; expected124. Best/last512-env evals are finite, exit0:

| Relative-force policy | First drive | Release / return / regrasp / two cycles | Median net angle | Termination |
|---|---:|---:|---:|---|
| Best `best_reward_19.31.pth` |512|0 /0 /0 /0|24.895deg|512release deadline|
| Last |0|0 /0 /0 /0|2.987deg|512drive deadline|

Best SHA `789c32fafdc72d8405e3f5abb5580f65e134e02fb099a824aafc26f809acaed9`;
last SHA `e227379b780a23d04a25447fc591dc23df1fc4ab52806106780c50d45689fdb4`.
The relative potential corrects a measured feedback-saturation issue but did not
produce autonomous release in this bounded comparison. It remains opt-in; no
new cycle checkpoint is promoted over the original stable-drive default.

The recorder produced47frames at20Hz, one episode, learned drive plus explicitly
SCRIPTED release. Last two frames have forces0/0N; final net angle23.796degrees.
Loaded and released frames were inspected. Single-play GIF and full tensor-state
archive: `outputs/local_pipeline_m24_e002_cycle_20260910/release_reference_video`.
This clip is not evidence of learned release or a repeated cycle.

`replay_m24_e002_release_states.py` restores the recorded DOF states, actor roots,
PDtargets, observation history and cycle state, then executes only the remaining
recorded actions. All five seeds reach actual release, with finite measurements:

| Seed step | Max absolute angle error vs original | Max absolute force error |
|---:|---:|---:|
|18|0.3945deg|4.1151N|
|28|0.4128deg|3.4449N|
|36|0.1529deg|0.3166N|
|40|0.00649deg|0.00990N|
|43|0deg|0.02688N|

Early replay differences show why solver-cache reconstruction cannot be assumed.
Near-release states40/43are suitable initial candidates for the next curriculum
integration check. Replay validates recorded-action continuation; it does not
prove that PPO can recover from those states or that a curriculum is implemented.
The archive keeps its original unvalidated marker for provenance; the separate
`release_state_replay/evaluation.json` records this subsequent physical validation.

Single next action: implement the opt-in demonstrated-state release curriculum
using40/43first, then progressively earlier states, within PPO. Retain final
candidate743full-start evaluation, global no-revisit credit and no scripted action
in learned evaluation. This follows DemoStart's initialization-distribution idea;
we have not implemented its full auto-curriculum or zero-variance filtering.

Final exact sources/checksums and consolidated pipeline/checkpoint results:
`outputs/local_pipeline_m24_e002_cycle_20260910/final_source/SHA256.json` and
`session_results.json`. Regression checks passed: actual cycle tensor tests,
relative unloading/force-oscillation/partial-reset tests, existing learning contract,
A0c sign/correlation/false-pass checks and script syntax checks.
