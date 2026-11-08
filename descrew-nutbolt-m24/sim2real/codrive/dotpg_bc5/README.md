# CoDrive DOTPG BC5 Baseline

This folder contains the current best DOTPG student baseline for the CoDrive
Dexh13 lightbulb task.

## Source

- Teacher PPO checkpoint: `best_reward_4159.37.pth`
- DOTPG student checkpoint: `model_best.ckpt`
- Deploy-selected copy: `model_best_deploy.ckpt`
- Task config: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- Train config: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

## DOTPG Settings

The selected run is:

- run id: `theory_iter2_20260504_092416`
- candidate: `dual_bc5`
- policy architecture: `teacher_actor`
- policy loss: `dual`
- `bc_coef=5.0`
- `bc_alpha_max=20.0`
- `dual_state_scale=1.0`
- `dual_action_scale=1.0`
- `lr_policy=0.0001`

## Eval-256 Result

From `eval_256_summary.tsv`:

| condition | avg_reward | avg_done_rate |
|---|---:|---:|
| train_like | 4.420655 | 0.000092 |
| clean | 4.496476 | 0.000122 |
| light | 3.824169 | 0.000198 |
| hard | 2.960864 | 0.000687 |

This is the strongest deploy-evaluated DOTPG checkpoint in the current run set.

## Optimization Summary

Detailed theory and implementation notes are recorded in:

- `docs/dotpg_codrive_optimization.md`

Short version:

- Earlier DOTPG was weak mainly because the actor path was under-conditioned for
  high-DOF contact manipulation.
- The generic Q-only DOTPG actor objective was too sensitive to Q approximation
  error in the lightbulb contact task.
- The current version reuses the PPO teacher actor architecture, initializes the
  DOTPG actor from the teacher, explicitly keeps copied actor parameters
  trainable, and uses the direct OT dual actor objective:
  `policy_loss = -mean(f_phi(s, pi_theta(s)))`.
- This direct dual objective follows the deterministic OT policy-gradient
  theorem in `thesis_reference/DOTPG-draft.md`: descending the Wasserstein
  objective corresponds to moving the actor along the action-gradient of the
  optimal Kantorovich dual potential.
- `bc_coef=5.0` acts as a proximal anchor around the expert action manifold. It
  prevents the dual objective from pushing the policy into poorly covered
  off-manifold actions while still allowing DOTPG improvement pressure.
- This is a medium-size algorithm implementation correction, not merely a small
  hyperparameter tweak and not a full pipeline rewrite.

The key empirical result is:

- Q-only teacher actor: max train best `869.14`.
- Direct dual teacher actor: max train best `2622.18`.
- Direct dual plus BC5: max train best `2904.14`, with fixed-step deploy eval
  clean `4.496476`, light `3.824169`, hard `2.960864`.

For future runs, select DOTPG checkpoints with fixed-step deploy eval rather
than training `Current Best` alone. The BC scan showed that `bc4` and `bc8`
could look competitive by train reward but degrade in deploy eval.
