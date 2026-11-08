# DOTPG CoDrive Optimization Note

Date: 2026-05-05

Scope: CoDrive Dexh13 lightbulb DOTPG student distillation from the PPO teacher
`sim2real/codrive/best_reward_4159.37.pth`.

This note records why the early DOTPG baseline was weak, what was changed, and
why the current `dual_bc5` version has a defensible algorithmic basis under the
DOTPG optimal-transport derivation.

## Current Baseline

The selected DOTPG checkpoint is:

- package: `sim2real/codrive/dotpg_bc5/`
- student deploy checkpoint: `model_best_deploy.ckpt`
- teacher PPO checkpoint: `best_reward_4159.37.pth`
- task YAML: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.task.yaml`
- train YAML: `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.train.yaml`

The selected run is `theory_iter2_20260504_092416 / dual_bc5`.

Core DOTPG settings:

- `policy_arch=teacher_actor`
- `policy_init_from_teacher=True`
- `policy_output_mode=clamp`
- `policy_loss_mode=dual`
- `bc_coef=5.0`
- `bc_alpha_max=20.0`
- `dual_state_scale=1.0`
- `dual_action_scale=1.0`
- `lr_policy=0.0001`

## Theoretical Basis

DOTPG formulates imitation as matching the student state-action occupancy
distribution `rho_pi` to the expert distribution `rho_E` by minimizing a
Wasserstein distance.

For the 1-Wasserstein case, the Kantorovich-Rubinstein dual form is:

```text
W(rho_pi, rho_E) = sup_{f in Lip_1} E_{rho_E}[f(s,a)] - E_{rho_pi}[f(s,a)]
```

Here `f` is the optimal transport dual potential. The draft DOTPG derivation
uses this dual variable as the source of policy gradients. For a deterministic
policy `a = pi_theta(s)`, Theorem 3 in `thesis_reference/DOTPG-draft.md` gives:

```text
grad_theta W(rho_pi, rho_E)
  = - E_{s ~ d_pi}[ grad_theta pi_theta(s) * grad_a f*(s,a)|a=pi_theta(s) ]
```

Therefore, descending the Wasserstein objective can be implemented by moving
the actor in the direction that increases the dual potential at the current
policy action. In code this corresponds to:

```text
policy_loss = - mean( f_phi(s, pi_theta(s)) )
```

This is the implemented `policy_loss_mode=dual` path in
`dexscrew/dotpg/dotpg.py`.

The original DOTPG design also contains a Q network:

```text
Q(s,a) <- f_phi(s,a) + gamma Q_target(s', pi_target(s'))
policy_loss = - mean( Q(s, pi(s)) )
```

That Q route is theoretically motivated as a long-horizon extension of the dual
signal. However, in this dexterous contact task the Q approximation is a major
source of error:

- contacts are discontinuous;
- reset/termination makes off-policy targets sharp;
- small action changes can switch between stable grip and slip;
- replay samples can become stale as the student encoder changes;
- the Q network can extrapolate badly outside the teacher action manifold.

For this task, the direct dual actor update is the lower-bias implementation of
the OT policy-gradient theorem. The Q path remains useful as a diagnostic or
auxiliary signal, but it should not be the primary actor objective unless later
evidence shows reliable deploy improvement.

## Why The Earlier DOTPG Was Weak

Early DOTPG runs underperformed for three separate reasons.

### 1. Actor Parameterization Mismatch

The first DOTPG actor used a generic DOTPG MLP policy. That is a poor match for
this hand-control task:

- the action dimension is high;
- valid behavior is a narrow two-finger contact manifold;
- PPO teacher behavior depends on the existing `actor_mlp + mu` architecture;
- random or weakly initialized policies quickly leave the expert support.

The result was low sample efficiency and unstable contact behavior. This is an
engineering manifestation of a theoretical issue: the dual potential is only a
reliable local guide near distributions covered by expert and policy samples.
If the policy class and initialization place the actor far from the expert
support, the learned dual has to extrapolate over a large state-action gap.

### 2. A Real Migration Bug: Copied Actor Parameters Were Not Trainable

The PPO teacher model is intentionally frozen during student distillation.
When reusing the teacher actor modules, copied parameters inherited frozen
gradient flags unless they were explicitly re-enabled.

The fixed wrapper `TeacherActorPolicy` now deep-copies `actor_mlp + mu` and sets
all copied parameters to `requires_grad=True`.

Without this fix, the apparent DOTPG policy update path can be partially or
fully disabled, depending on how the copied modules are constructed. This is a
code migration issue, not a DOTPG theory issue.

### 3. Q-Only Actor Objective Was Too Noisy For Contact Manipulation

The Q-only actor objective asks the actor to maximize a learned long-horizon
value of dual rewards. That is attractive in theory, but in this task it was
not reliable enough:

- `teacher_actor_q` reached only `869.14` max train best;
- `teacher_actor_dual` reached `2622.18`;
- `teacher_actor_qdual_metric` reached `1327.30`.

The empirical gap supports the theoretical diagnosis: Q approximation error was
overpowering the useful OT dual gradient. Direct dual actor updates are closer
to the theorem and worked better.

## What Changed

The current version is a medium-size algorithm implementation correction. It
does not rewrite the full teacher-student pipeline or change the environment,
but it does change the core DOTPG actor path.

### 1. Teacher Actor Policy Architecture

Added `policy_arch=teacher_actor`.

This reuses the PPO teacher's `actor_mlp + mu` policy architecture for the
DOTPG student actor. The actor can still be trained by DOTPG, but it starts
from a policy class and parameterization known to express the teacher behavior.

Algorithmically, this is a policy-class prior. It does not change the OT
objective. It reduces approximation error and keeps policy updates in a region
where the learned dual potential is meaningful.

### 2. Teacher Initialization With Trainable Copied Parameters

Added `policy_init_from_teacher=True` and fixed trainability in
`TeacherActorPolicy`.

The policy starts close to the expert distribution instead of from a random MLP.
This is important because the dual gradient is most useful where the dual
network has seen both expert and student samples.

### 3. Direct Dual Actor Objective

Added `policy_loss_mode=dual`.

Instead of relying on the Q approximation:

```text
loss_q = - mean(Q(s, pi(s)))
```

the selected DOTPG actor directly optimizes:

```text
loss_dual = - mean(f_phi(s, pi(s)))
```

This is the closest implementation to the deterministic OT policy-gradient
formula in the thesis derivation.

### 4. BC Anchoring

Added BC regularization during policy updates. The selected coefficient is
`bc_coef=5.0` with `bc_alpha_max=20.0`.

This acts as a proximal constraint around the teacher action manifold:

```text
loss = loss_dual + alpha * MSE(pi(s_E), a_E)
```

The role is not to reduce DOTPG to pure BC. The role is to keep the actor inside
the support where the dual potential is well trained, while the dual objective
still supplies improvement pressure.

This is similar in spirit to TD3+BC and trust-region logic: off-policy actor
updates are safer when constrained near demonstrated behavior.

### 5. Fixed-Step DOTPG Eval

DOTPG eval now supports top-level `+test_num_steps` and prints:

```text
EvalSummary steps=... avg_reward=... avg_done_rate=...
```

This is important because train reward alone was not a reliable selector for
DOTPG. The eval path gives a deploy-oriented selection criterion.

### 6. Cloud Log Control

`train.py` supports `DEXSCREW_SKIP_GIT_DIFF=1`.

This does not change the algorithm. It prevents cloud runs from wasting startup
time and log readability on huge dirty git diffs.

## Experiment Evidence

All candidates below were run from the CoDrive PPO teacher.

### Iter1: Objective and Architecture Test

| candidate | timeout status | max train best | conclusion |
|---|---:|---:|---|
| `teacher_actor_q` | 124 | 869.14 | Q-only actor objective weak |
| `teacher_actor_dual` | 124 | 2622.18 | direct dual is the correct main direction |
| `teacher_actor_qdual_metric` | 124 | 1327.30 | adding Q/metric scaling did not help |

### Iter2: Dual Objective With BC Anchoring

| candidate | timeout status | max train best | conclusion |
|---|---:|---:|---|
| `dual_metric_action2` | 124 | 1386.54 | action metric scaling hurt |
| `dual_bc5` | 124 | 2904.14 | selected deploy baseline |

### Iter3: BC Strength Scan

| candidate | timeout status | max train best | deploy result |
|---|---:|---:|---|
| `dual_bc4` | 124 | 2916.54 | train looked good, deploy failed |
| `dual_bc6` | 124 | 2809.14 | weaker than bc5 in deploy eval |
| `dual_bc8` | 124 | 3004.40 | highest train reward, worse deploy eval |

This is the strongest evidence that train `Current Best` is not enough for
DOTPG model selection.

## Selected Deploy Eval

For `dual_bc5`, fixed-step 256 eval produced:

| condition | avg_reward | avg_done_rate |
|---|---:|---:|
| train_like | 4.420655 | 0.000092 |
| clean | 4.496476 | 0.000122 |
| light | 3.824169 | 0.000198 |
| hard | 2.960864 | 0.000687 |

This places DOTPG in the same broad deploy-eval range as the recorded CoDrive
diffusion/flow/consistency baselines, instead of the earlier failed regime.

## Distillation Time Assessment

The 1.5h runs were enough for directional decisions:

- Q-only vs direct dual separated clearly.
- BC anchoring was clearly necessary.
- `bc5` was better than the nearby BC scan in deploy eval.

They are not enough for final paper-grade claims:

- DOTPG uses expert collection, adapt warmup, and BC pretrain before effective
  actor-critic updates dominate.
- Longer runs may help, but only if selection uses deploy eval rather than train
  reward.
- `bc8` showed that more train reward can mean worse deploy behavior.

Recommended next DOTPG training protocol:

1. Keep `teacher_actor + dual + bc5` as the base.
2. Add periodic fixed-step eval selection during DOTPG training.
3. Run a 3h or 4h version and select by eval score, not train best.
4. Run seeds `42,43,44` before using DOTPG as a thesis-level result.

## Current Interpretation

The current improvement should be described as an algorithmically motivated
implementation correction rather than simple hyperparameter tuning.

The important scientific claim is:

- DOTPG was weak when implemented as a generic Q-only off-policy actor-critic.
- DOTPG becomes competitive when the actor update follows the direct OT dual
  policy-gradient theorem and is constrained near the expert manifold.

This is a meaningful result for the baseline discussion because it separates
two questions:

1. Is optimal-transport imitation theoretically valid here?
2. Was the previous code path a faithful and well-conditioned implementation?

The current evidence suggests the answer is:

- the OT direction is valid and useful;
- the previous implementation was under-conditioned for high-DOF contact
  manipulation;
- teacher-actor initialization, direct dual actor loss, and BC anchoring are
  necessary to make DOTPG a fair baseline on this task.
