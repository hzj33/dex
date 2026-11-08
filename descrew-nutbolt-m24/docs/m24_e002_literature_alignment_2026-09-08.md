# E002 learning/reward alignment with primary literature

## Evidence, not a coefficient search

User instruction: continue toward actual two-finger twisting, base reward work
on literature, keep weights and pair gate fixed unless explicitly authorized.
The attached review is secondary reference; its engineering suggestions are
not treated as original-paper formulas or as permission to override constraints.

| Primary source | Verified learning/reward construction | Transfer limits |
|---|---|---|
| [DexScrew, appendix V-B/V-C](https://arxiv.org/html/2512.02011v1) | PPO; clipped signed rotation plus proximity, effort and stability terms. The official code directly clips nut joint increments divided by policy dt. | It does not require both instantaneous positive normal moments or a minimum/balance product before rewarding rotation. Teacher observations and training scale also differ. |
| [Tac2Motion, II-A/B/C](https://arxiv.org/html/2509.17812v1) | PPO with incremental joint targets and smoothing; rotation is weighted by finger contact quality. Direction guidance is based on fingertip motion, distinct from contact pressure. | Contact guides, pressure surrogate and virtual friction torque differ from this M24 model. It is not evidence for our normal-only moment gate, and its coefficients are not copied. |
| [Twisting Lids, 3.3](https://arxiv.org/html/2403.02338v2) | Incremental rotation, reference-contact proximity, axis alignment and action/work regularization; smoothed relative joint targets. | Two hands, brake-link object model and contact references differ. Useful evidence for guiding contact geometry, not proof that our actor/physics should converge in five minutes. |

Official DexScrew code:
[compute_reward](https://github.com/x-robotics-lab/dexscrew/blob/main/dexscrew/tasks/xhand_hora.py).
A local read-only snapshot and URL/SHA256 manifest are in
`outputs/m24_e002_literature_primary_20260908/`. Source SHA256:
`9c4144fa3bc5f5164205b37662122a35441b4a8866009609f1aa28f32a1f256c`.
The legacy source computes a reverse variable that is not added separately;
we retain E002's explicit high-water and reverse requirements, not that omission.

## Our current gate is an additional hypothesis

The current E002 forward reward is `4 * G_pair * normalized_new_high_water`.
`G_pair = min(c_index * t_index, c_thumb * t_thumb) * torque_balance`,
where `t_i = clip(positive normal-contact axial moment / 0.1, 0, 1)`.
Only normal-contact moments are measured; friction moments are unmeasured.
This minimum and balance construction is not reproduced from the papers above.
It must be judged by task evidence, not justified by attaching a paper name.

Saved real A0c bank, 27 candidates × 30 steps, fixed candidate-743:

| Sum over candidate new-high-water increments | Radians | Fraction of all progress |
|---|---:|---:|
| all forward high-water progress | 3.2641155 | 100% |
| during physical pair contact | 3.2449665 | 99.41% |
| while G_pair is nonzero | 0.07192953 | 2.20% |
| while all strict conditions pass | 0.00953995 | 0.292% |

Source: `outputs/m24_e002_literature_primary_20260908/gate_coverage.json`,
recomputed from the prior real A0c JSON. These sums across distinct candidates
are signal coverage, not one successful long stroke. Candidate 19 alone has
7.20 degrees total but only 0.04354 degrees during accepted steps.
This establishes reward-signal suppression. It does not establish that the
suppressed progress is a valid two-finger drive, because friction is unmeasured.

## Concrete single-variable comparison (authorized 2026-09-08)

Prepared patch: `outputs/m24_e002_literature_primary_20260908/contact_progress_proposal.patch`.
It would replace only the positive progress multiplier:

```
current:  forward_reward = 4.0 * pair_gate * normalized_new
proposed: forward_reward = 4.0 * contact_potential * normalized_new
```

`contact_potential = min(c_index,c_thumb)`. This adapts the literature's
contact-weighted rotation to opposing fingers; it is not an exact Tac2Motion
reproduction. All numerical weights, invalid-contact cost, reverse/high-water
logic, static no-reward property, A0c, strict valid progress, milestones and
success acceptance (`pair_gate >= 0.50`) would remain unchanged.

The user explicitly authorized this TRAINING multiplier exception. The
implemented nut-specific version preserves strict physical acceptance; the
original one-line proposal remains a historical artifact. Run one fresh A0c,
one cloud smoke and one five-minute PPO; evaluate actual strict behavior of
best and last. Reject a policy that only increases contact-gated reward.

The independent reset-history ordering fix is already validated and does not
change any reward. Its controlled five-minute cloud run finished with 0/512 strict successes for both best and last under
`outputs/cloud_pipeline_m24_e002_reset_history_20260908_171000`.


A later diagnostic reference control provides a stronger counterexample:
31.54 degrees with continuous pair contact, versus about0.001 degrees when
either tangential command is removed; normal gate is zero throughout the
successful reference. Raw GPU records confirm only nut contacts. This supports
the contact-gated comparison while leaving full wrench contribution unmeasured.
Details: `docs/m24_e002_reset_and_reachability_2026-09-08.md`. A CPU-backend
probe did not recover friction forces; apparent extra force was thumb-bolt
contact. Future contact quality must use nut-specific records, not total
fingertip resultant contact. The nut-only training multiplier is now applied for the authorized comparison.

## Authorization and implementation update

The user explicitly approved the proposed comparison. `forward_source: nut_contact_quality` now uses minimum opposing contact confidence from nut-only normal records and existing distance confidence; the legacy pair-quality default remains available. No weights or strict acceptance rules changed. The older prepared patch above is superseded by this nut-specific implementation. Pipeline: `outputs/cloud_pipeline_m24_e002_nutcontact_20260908_175230`. Execution completed: both best/last fail strict512-env acceptance despite median26.07/18.34deg motion. See `docs/m24_e002_nutcontact_comparison_2026-09-08.md` for results and the substep velocity/position discrepancy that limits further kinematic reward interpretation.

The [official DexScrew PPO config](https://github.com/x-robotics-lab/dexscrew/blob/main/configs/train/XHandHoraNutBolt.yaml) also confirms our inherited learning rate 0.005, horizon12, five mini-epochs and KL threshold0.02. These values were not invented in this experiment and are unchanged; source URL/SHA and YAML are archived beside the task snapshot. Matching hyperparameters alone does not establish convergence in five minutes.
