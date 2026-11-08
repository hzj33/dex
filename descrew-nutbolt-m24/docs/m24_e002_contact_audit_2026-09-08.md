# E002 candidate-743 contact-point audit — 2026-09-08

The link-origin torque proxy is not a valid measurement of the recorded
contact-point normal torque. In all 600 probe physics steps across six
candidates, its index-finger sign differs from the reconstructed normal torque
on the nut. The full contact wrench, including friction, is still unverified.
This result does **not** select a force sign or pass A0c.

## Scope and execution

This continues the single next action in the preceding handoff. The previous
session implemented 67-candidate diagnostic A0c, corrected only the E002 nut
axis to `-z`, and reran A0c with a failed result. This session adds measurement
and analysis only; task code, reward weights, thresholds, fixed pose, action
vectors and physical friction settings are unchanged.

- Cloud: AutoDL `pro-786095bb6e03`, `/root/code/dexscrew-repro`, RTX 4090,
  49140 MiB; `dexscrew-ig` conda environment. No local simulator/training ran.
- Main pipeline, fully preserved locally and on cloud:
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_contactbank_20260908_140118`.
- Start/end: `2026-09-08T14:01:18+08:00` / `2026-09-08T14:02:54+08:00`.
- Phase: `completed:contact_audit_only`; twelve simulator subprocesses exited
  0. Each subprocess had its own `timeout 180`, under a named tmux launcher.
- Source: the fixed pose and exact action vectors in
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_axisfix_s42_20260908_113745/manifests/a0c_diagnostic.json`
  (SHA-256 `88918b8ac3547d50f287b8476e20d77177f1c501fba0b5ec3ff0d4e8c38a5cb1`).
- Six actions: zero; positive joint 0, 1 and 14 at amplitude 0.1; both signs
  of the equal-ratio index/thumb tangent proposal at amplitude 0.1. Middle and
  ring remain masked. No action search or A0b pose search was repeated.

Each candidate uses a fresh process with exactly one environment. Two modes
retain GPU PhysX, the same seed, physics and action sequence:

1. `tensor_reference`: the original GPU tensor pipeline.
2. `rigid_contacts`: CPU tensor pipeline to expose `get_env_rigid_contacts`.

Each mode records 6 settle + 10 hold + reset + 6 settle + 10 probe policy
steps, with all ten 200 Hz physics endpoints inside each 20 Hz policy step.
This gives 320 samples per process, 3840 total, including 1920 samples with
23088 raw contact records. Every physics step is fetched/refreshed before
recording. No unexpected task reset or nonfinite output occurred.

The saved body map is index tip 4, thumb tip 25, nut 30; the M24 object has
three bodies and one nut DOF. Actual actor shape friction values are **1.0**
for both hand and object. Raw records nevertheless report zero `friction`
and `lambdaFriction` in every sample; these fields cannot establish zero
physical friction.

## Reconstruction and evidence

`scripts/analyze_m24_e002_contacts.py` transforms each nut-side `localPos`
through the recorded nut quaternion and sums `r_contact × F_normal` by finger.
The nut COM differs from its origin by less than 3e-10 m, immaterial here.
The force convention is checked empirically against the same-step tensor:
`+lambda * normal` on body0 and its negative on body1, without division by dt.
The maximum absolute component residual is **3.4752e-6 N** across all 1920
contact samples. Only index–nut and thumb–nut pairs have active normal force;
the other geometric contact records have zero lambda.

This checks the API's reported normal force, **not** unobserved friction. A
resultant force at a link origin also omits the moment associated with its
actual application points. Changing the global force sign cannot repair
different per-finger lever-arm errors.

For example, the final zero-action sample has the following signed moments
about the positive nut DOF axis `-z`, using the existing `force_sign=-1`:

| Measurement | Index (Nm) | Thumb (Nm) |
| --- | ---: | ---: |
| Link-origin proxy | +0.00269912 | -0.00913039 |
| Actual contact-point normal moment | -0.00439155 | -0.00529093 |

The comparison uses the same recorded physics endpoint. It does not combine
maxima from different steps or actions. All raw contact points, normals,
lambda values, all-body forces/states, commands, configs and source hashes
remain available for independent checks.

| Candidate | Probe contact steps / 100 | Index sign mismatch / 100 | Thumb sign mismatch / 100 | Max normal-only gate, sign -1 / +1 |
| --- | ---: | ---: | ---: | ---: |
| 0, zero | 100 | 100 | 0 | 0 / 0.040339 |
| 34, joint0 +0.1 | 100 | 100 | 0 | 0 / 0.053007 |
| 36, joint1 +0.1 | 100 | 100 | 0 | 0 / 0.052820 |
| 46, joint14 +0.1 | 100 | 100 | 0 | 0 / 0.037303 |
| 63, equal-ratio tangent −0.1 | 100 | 100 | 0 | 0 / 0.050023 |
| 64, equal-ratio tangent +0.1 | 60 | 100 | 0 | 0 / 0.039073 |

Under **both** force signs the normal-only simultaneous torque, balance and
full-condition counts are all 0/600. No gate reaches 0.50. These are diagnostic
counts at physics endpoints, not a replacement policy-step A0c evaluation.
Candidate 64 has the largest contact-audit net angle, `8.09177e-7 rad`, versus
the unchanged `pi/180 = 0.0174533 rad` A0c requirement. Its contact count is
60/100, and its gate remains far below threshold.

## Limits and decision

The two tensor modes are **not identical trajectories**. In settled hold their
maximum force-magnitude difference is 0.00494 N, position-distance difference
1.45e-7 m and nut-angle difference 8.55e-12 rad. The reset transient reaches
0.660 N force and 0.02088 Nm proxy difference. During probes the maxima over
the six candidates are 0.518 N force, 0.00571 Nm proxy, 5.69e-5 m distance and
1.47e-8 rad angle. Full per-phase differences are saved. The contact-point
comparison above is within the contact-reading mode, so it does not depend
on assuming cross-mode equality.

In this trace, normal records reconstruct the reported net force tensor, while
the friction fields provide no validated tangential wrench. Consequently:

- do not treat this tensor as a validated full per-finger contact wrench;
- do not replace the reward's torque with normal-only torque or a guessed
  contact offset, and do not flip signs to make normal-only gate values larger;
- do not tune the actual friction parameter from a zero API metadata field;
- do not rerun the unchanged failed A0c bank without a demonstrated fix.

The previous session's axis correction and failed A0c rerun remain the only
task-behavior change. This session did not make a second speculative reward
fix or repeat A0c. Latest actual A0c status remains
`failed:no_pair_positive_drive`. Audit JSON status is separately
`completed:normal_contact_audit`, `a0c_status=not_evaluated`,
`training_eligible=false`. No smoke, PPO, checkpoint, 30-minute training or
E002-B/C ran. All future authorized training remains cloud-only.

**Single next action:** validate an actual per-finger total contact-wrench
measurement in a bounded one-environment cloud instrumentation test, including
tangential forces/application points and separation from joint-constraint
reaction. A raw rigid-body force sensor is not automatically contact-only.
Use a validated source before making one E002 torque-source correction and
rerunning A0c with unchanged thresholds. If the installed GPU API cannot
provide this without a major refactor, record that evidence and the smallest
alternative in `codeagent_issue.md` rather than silently changing the gate.

## Files and reproducibility

- Simulator audit: `scripts/audit_m24_e002_contacts.py`.
- Cloud wrapper: `scripts/cloud_run_m24_e002_contact_audit.sh`.
- Offline analysis: `scripts/analyze_m24_e002_contacts.py`.
- Independent geometry check: `scripts/test_m24_e002_contact_wrench.py` tests
  a translated/rotated nut, known torque and reversed contact body order.
- Main result: pipeline `manifests/contact_audit_summary.json`; all twelve raw
  JSONs and resolved configs are in the same directory. Commands, timestamps,
  exit statuses and source snapshots are in the adjacent directories.

```bash
# Cloud, from /root/code/dexscrew-repro; creates a fresh output directory.
PIPE_DIR=outputs/cloud_pipeline_m24_humantwist_e002_contacts_$(date +%Y%m%d_%H%M%S)
cp scripts/cloud_run_m24_e002_contact_audit.sh "${PIPE_DIR}.launcher.sh"
tmux new-session -d -s m24_e002_contacts \
  "PIPE_DIR=${PIPE_DIR} bash ${PIPE_DIR}.launcher.sh"

# Offline, no simulator and no training.
python3 scripts/test_m24_e002_contact_wrench.py
python3 scripts/analyze_m24_e002_contacts.py \
  outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_contactbank_20260908_140118
```

The first pilot, `..._m24_e002_contactpilot_20260908_135419`, failed while
reading metadata (`RigidBodyProperties.com` is a Vec3, not a Transform), before
measurement. It is preserved but excluded from physical conclusions. After
that bounded instrumentation fix, pilot `..._m24_e002_contactpilot_20260908_135509`
completed both zero-action modes. The six-candidate bank adds actual shape
friction metadata and is the authoritative audit result.

API references checked alongside the installed Gym documentation:
[NVIDIA's explanation of the CPU contact API versus GPU tensors](https://forums.developer.nvidia.com/t/gym-get-env-rigid-contacts-returns-empty-list-in-rl/166628/2),
and an [independent Isaac Gym reproduction of zero friction fields despite nonzero shape friction](https://forums.developer.nvidia.com/t/recovering-frictional-forces-from-gym-get-env-rigid-contacts/264682).
The latter is corroborating evidence; the conclusions above come from this
repository's saved measurements, not an assumption about another API version.
