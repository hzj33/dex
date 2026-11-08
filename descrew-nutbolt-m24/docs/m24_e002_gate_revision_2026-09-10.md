# E002 gate revision and authorized local fallback — 2026-09-10

## Authorization and scope

The user explicitly approved removing unvalidated normal-moment vetoes and
mature-success prerequisites for early training, then authorized local training
when cloud resources are unavailable. This supersedes conflicting historical
stage rules. Final goal: index/thumb cooperation, useful net rotation, release,
return, regrasp and repetition. No drive-only result is complete-task success.

AutoDL `power-on pro-786095bb6e03` failed: `InternalError`, current compute
specification has no inventory; request ID `5dc32034b9c3ed3712ae56d689677ce8`.
Status remained `shutdown` (request ID `67267686fdcfdc0e17a1120859ccc84d`).
The remote handoff is unavailable and has not been overwritten. The local
cloud-handoff copy records the new state pending future synchronization.

Local GPU: RTX 3060, 12288 MiB, approximately 11 GiB free before execution.
Runtime: existing `dexscrew:ig20-py38` Docker image, torch2.3.1+cu121, existing
IsaacGym Preview4 binding. The host conda attempt was abandoned at import
because Ninja was unavailable; no training took place in that runtime.

## Evidence and testable hypothesis

- [Tac2Motion, II-B, equations2–3](https://arxiv.org/html/2509.17812v1): combines
  contact/release shaping with contact-quality-weighted rotation. Its virtual
  lid torque and five-finger hardware are not reproduced here. Our opposing
  minimum nut-contact quality and high-water accounting are local adaptations.
- [DexScrew, III-A and Fig4](https://arxiv.org/html/2512.02011v1): learns rotational
  gaits from a simplified revolute object; observed thumb/index contact can
  alternate. The paper's real fastening also uses real demonstrations and arm
  motion. It does not validate our exact synchronous50ms requirement or our
  normal-only wrench estimate.
- Local raw audits show normal contact records omit friction; cooperative
  reference commands can rotate the nut while the normal-only pair gate is0.
  This motivates removing a veto, not claiming a measured complete wrench.
- The previous positive reward already used nut-contact quality, but milestones,
  target validation and success still depended on normal-only torque.

Hypothesis: aligning progress bookkeeping/milestones with nut contact and actual
new net rotation supplies consistent learning signals, without rewarding bolt
contact or oscillation. Warm-start the previous best policy and compare fixed
evaluations. Removing a veto alone does not establish physical success.

## Implemented contract

- `acceptance_mode=contact_progress` is active in the E002 config. Missing this
  field retains `strict_normal_proxy` for historical task/config compatibility.
- Physical contact state, approach and contact-deficit terms use nut-only force
  in active mode. Forward scale, penalties and target/physics/control values
  remain unchanged. Normal torque/balance and `normal_proxy_progress` remain
  explicit diagnostic logs.
- New high-water motion under nut pair contact counts toward drive milestones
  and target validation. Missing-contact motion is consumed by high-water
  accounting and cannot be credited later by touching a stationary nut.
- A0c active mode uses stable nut contact, a contact/direction streak and at
  least1degree of new high-water progress occurring under pair contact. Original
  force-sign torque flags, all-condition streak and associated top-k remain
  visible. The retained sign is not a full-friction sign calibration.
- Import checks active mode, configured thresholds, raw-trace rescoring and
  source hashes; an old report is not silently upgraded to the new contract.
- Evaluation distinguishes execution integrity from learning maturity. Historical
  90%/95% criteria remain reference metrics. `continuation_allowed` does not mean
  a successful drive or complete cycle, and does not launch long training.
- Actual rewarded-step counts are separate from historical normal-gated credit
  counts. Full-task `goal_completed` is explicitly false in drive-only reports.
- Exact single-control-step release/regrasp synchrony is removed from the active
  plan. Release, return and regrasp are still to be trained and evaluated; this
  patch does not pretend the drive-only environment already implements them.

## Validation and current execution

- Actual reward method exercised with CPU torch tensors: equal physical motion
  succeeds with contact-progress mode and fails under opposing normal moments
  in legacy mode; bolt-only contact is excluded; high-water revisits are unpaid.
- Synthetic A0c: opposing normal moments no longer veto contact-driven progress;
  bolt-only forces, uncontacted motion, resets and nonfinite samples do not pass.
- Evaluation: a complete finite failed learning episode permits continued
  learning; missing/truncated traces, NaN or frequent angle discontinuities do not.
- Existing A0c sign/correlation/hold tests and actual progress-expression tests pass.
- Fresh local27-candidate A0c passed, selected22,17 contact/direction steps,
  0.1284928173rad (7.362deg) nut-contact new progress. Old normal conjunction
  has only1 consecutive accepted step for that same candidate. JSON SHA256:
  `e67e3626ae141bf083241edfa67bc959016970b2ae6ee5b11a005fede8a58d49`.
- A0c artifacts: `outputs/local_pipeline_m24_e002_contactprogress_20260910`.
- Bounded learning pipeline:
  `outputs/local_pipeline_m24_e002_contactprogress_20260910_retry3072`.
  The initial launcher falsely matched a historical document diff containing
  the word Traceback; smoke actually finished with expected124. Anchored runtime
  error matching and separate diff logging fix this. The successful smoke is
  reused only after checking identical A0c bytes and initial checkpoint hash;
  no second smoke budget was spent. Original logs remain in the probe3072 path.
  One60second smoke, then300second PPO at3072envs/6144minibatch, each with30second
  kill-after grace. Starts from September8 `best_reward_-1.02.pth` (policy and
  normalization warm start; optimizer and step counter restart). Best/last
  fixed512env evaluations follow. Status and exact commands are in the pipeline.

## Completed results and behavioral selection

Smoke: 15:44:27–15:45:28 +08, expected124. PPO: 15:46:39–15:51:46 +08,
300second timeout plus7second completed-update grace, expected124. Last and
reward-best evaluations finished at15:52:35 +08. PPO collection stabilized
around5.2–5.5k environment steps/second. All jobs have ended.

The starting checkpoint was reevaluated on the **same local hardware, revised
contract, fixed pose and512env protocol**; improvement is not merely relabeling
old trajectories under weaker acceptance.

| Fixed512env evaluation | Starting policy | Reward-best snapshot | Last snapshot |
|---|---:|---:|---:|
| Median net angle |24.30deg|9.30deg|24.64deg|
| At least20deg with pair contact held4steps |0/512|12/512|512/512|
|25–35deg drive target held4steps |0/512|0/512|159/512|
| Contact-loss / workspace terminations |2 /510|0 /0|0 /0|
| Reached30deg |0/512|0/512|0/512|

Select `last.pth` by actual behavior, not the larger scalar training return.
The other353 latest-policy episodes reach the drive deadline while retaining
contact, rather than falling out of contact or escaping the workspace.
No fixed evaluation has an angle-jump termination. Stochastic training did
show many such terminations; this remains a robustness limitation, not a reason
to erase the deterministic improvement or claim mature robustness.

- Directory: `outputs/XHandPasiniM24NutBolt_teacher/local_pipeline_m24_e002_contactprogress_20260910_retry3072_ppo/stage1_nn`.
- Selected `last.pth` SHA256: `ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`.
- `best_reward_5.44.pth` SHA256: `d91fcd9a39c0e7ce3bb388f1f1a6cc0f03e7101bfd8c4831671a597f948e50dc`.
- Summary: A0c pipeline `results_summary.json`; first-episode/raw evaluation
  traces remain in each evaluation directory. No physical success claim is
  based only on `continuation_allowed`.
- Camera verification: `policy_video/learned_drive.gif` in the A0c pipeline,
  a single4.05second first episode at20fps, with actual pre-reset angle/contact
  overlay. This one-env episode reaches23.67deg and holds; it contains no
  release/regrasp cycle and no reset concatenated as a fake cycle. Initial and
  held frames were inspected. The export uses single playback.
- Latest-policy nut normal forces: median index11.68N/thumb28.90N, maxima
  15.81N/33.90N. Load optimization is not established by this motion result.

## Known external nut-torque diagnostic

`probe_m24_e002_known_nut_torque.py` completed12 cases (with/without fingers,
six torque inputs each),200physics endpoints over1second. Artificial external
nut torque is strictly diagnostic and is not used in PPO or claimed as policy
motion. No-contact cases were verified from raw nut-contact counts.

With the hand moved clear,0Nm produces effectively0deg;0.05Nm produces20.03deg
and0.1Nm66.58deg, while endpoint velocity integrals remain near0. With the
candidate743 grasp,0.1Nm produces2.02deg. Thus applied torque and grasp state
affect angle as expected qualitatively, and the angle/endpoint-velocity mismatch
also exists without fingertip contact. Endpoint qdot alone cannot invalidate
the observed rotation. This does not establish complete physical fidelity or a
measured full-friction wrench. Raw evidence: `known_nut_torque.json` in the A0c
pipeline. No production physics/actuation change was made.

## Single next action

Introduce release/return/regrasp learning from the demonstrated stable roughly
20degree stroke, preserving real net progress across cycles. Do not require a
30degree stroke or90% mature success before phase learning. Use measurable
release, actual return and subsequent driven rotation; never count a timeout
as successful phase completion. Full repeated-twist goal remains unmet.
