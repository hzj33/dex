# E002 execution-time correction — 2026-09-11

## Result

An explicit timing configuration lets the existing PPO finish more complete physical cycles. Model weights, rewards and physical success conditions are unchanged. Across384paired +/-1degree initial joint perturbations, five-cycle success rises284→324 (73.96→84.38%). All284previous successes remain successful. Nominal candidate743 remains128/128five cycles.

| Evaluation seed | Original deadlines | Extended deadlines | Newly successful | Regressions |
|---|---:|---:|---:|---:|
|43|94/128|108/128|14|0|
|44|99/128|109/128|10|0|
|45, held out from budget selection|91/128|107/128|16|0|

Seeds43/44 informed the time-budget decision; seed45 was first evaluated after the configuration was fixed. Success still requires five ordered drive/release/off-contact-return/regrasp cycles, each with at least20degrees actual net stroke and20degrees counted pair-contact progress. Higher net angle alone is not promoted.

## Diagnosis and evidence

The prior selected model had31third/fifth-drive timeouts on seeds43/44. At termination,27had already moved the nut at least20degrees from the last regrasp, but all31had less than20degrees of counted pair-contact progress. Fifteen gained less than0.1degree in the last10steps. Joint motion and target motion continued in many cases, so a static angle trace alone did not prove that further progress was impossible.

The controlled diagnostic doubled termination budgets while retaining the original policy-clock denominator and saturation. It recovers24of31drive timeouts into complete five-cycle successes. Exact sampled reset arrays match. All diagnosed seed43timeout endpoints retain identical actions/angles up to the old deadline. One seed44trajectory differs at that endpoint (env64,maximum action difference0.05362,nut angle0.511degree); the source of this numerical divergence was not isolated. Formal configuration reproduces both aggregate results. The single-environment native replay provides an exactly matched prefix independently of that batch discrepancy.

[Tac2Motion II-B, Eq3](https://arxiv.org/html/2509.17812v1) uses contact-weighted incremental rotation, alongside contact/release rewards; SectionII-D terminates on fingertips leaving the object workspace. These motivate retaining contact/rotation evidence and workspace protection. They do not prescribe our20degree conjunction or4/2/3/2second phase deadlines. The doubled budgets are a local, tested adaptation. No paper reward formula or weight was changed in this session. Testable hypothesis: some learned motions can meet the same physical goal if premature phase termination is delayed without altering their learned clock inputs. The paired and held-out tests support this hypothesis for the evaluated distribution.

## Implementation

- `dexscrew/tasks/e002_cycle.py`: optional positive finite `phase_clock_scale`, default1. Legacy behavior remains unchanged. No new per-environment state tensor or checkpoint dimension is introduced.
- `configs/task/XHandPasiniM24NutBoltE002RobustTime.yaml`: inherits the previous robust task, sets execution deadlines160/80/120/80steps (8/4/6/4seconds at20Hz), clock scale0.5, five-cycle goal and2200episode steps. The policy still sees80/40/60/40normalized clock units. Separate matching train config inherits the existing PPO configuration; no training was launched.
- Cycle evaluator: optional joint-state/target logs and diagnostic timeout multiplier, with termination budgets and policy-clock denominators logged separately.
- Diagnostic/validation runners and analyzers preserve exact commands, source hashes, initialization arrays, per-step physical data and before/after outcomes. Each simulator subprocess is independently bounded180seconds.
- `scripts/test_m24_e002_phase_clock.py` tests exact clock equality in all phases/beyond saturation, late physical release, unchanged two-step dwell and reset integrity. Existing vectorized full-cycle, timeout, jump, release, return and no-recycled-progress tests pass. Syntax and whitespace checks pass.

## Native evidence

The previously failed seed43recording stopped after268frames during fifth release. The new configuration produces290frames at20Hz (14.5seconds), five complete cycles and151.77975degrees net rotation. The first268actions and angles match the failed recording exactly. The remaining22steps (1.1seconds) complete release, off-contact return and regrasp. All five return intervals stay physically off-contact. The20event keyframes and angle/force/phase plot were inspected. This is one uninterrupted native-camera episode with one frozen PPO and no scripted action, demonstration reset or episode concatenation.

The MP4 is `outputs/cloud_pipeline_m24_e002_robust_time_20260911/native_seed43/learned_perturbed_cycles.mp4`, SHA256 `be1c58c53805cddc5427d250b77bca7fa0a86c048291f962fcadbd4c02840db4`. Raw trace, PNG frames, event keyframes and plot are preserved alongside it. `visual_verification.json` records the review separately from the raw recorder's default goal flag.

## Selected combination and execution

Use `task=XHandPasiniM24NutBoltE002RobustTime` with checkpoint `outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`, SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. `selected_policy.json` binds this unchanged model to the new task configuration. Previous models and task variants remain available.

Cloud instance pro-786095bb6e03, `/root/code/dexscrew-repro`, conda `dexscrew-ig`, RTX4090/49140MiB, existing task-local NVIDIA EGL Vulkan ICD. Four diagnostic evaluations,five formal evaluations,one native recording and refreshedA0c all exit0. NewA0c JSON explicitly passes with current source snapshots. No PPO, local training, driver change, student or30minute run occurred.

Diagnostic root: `outputs/cloud_pipeline_m24_e002_timeout_diagnostic_20260911`, archive SHA256 `e3a8c745ed1fd5ed51297d73adc01b0c22357ac53fc32cbe8c56be8d6ac559b5`.
Formal root: `outputs/cloud_pipeline_m24_e002_robust_time_20260911`, validation archive SHA256 `bb4825cfa89f978476e4bf079388a241d995b63420117a754ea899f9b73181d9`.
The separate final receipt archive preserves refreshedA0c, commands/logs, source and completion records. Raw archives and final artifacts are SHA-verified locally and on cloud.

## Remaining work

Sixty of384perturbed starts still fail:47return timeouts,10workspace violations,3drive timeouts. In seeds43/44,all29remaining return failures are physically off-contact but21.75–67.88degrees from the target sector (median24.16degrees); none are within6degrees. These are substantial return-geometry errors that the longer budget did not resolve. Wider randomization, more thanfive cycles and real-hand transfer remain unverified.

The single next action is to diagnose and collect the measured late return-failure states, then teach recovery through a bounded cloud continuation from this model/task combination. Preserve original/perturbed evaluations and nominal starts. The older robustness runner still hardcodes the old task; its task/horizon/preflight wiring must be adapted before that next curriculum. Do not keep extending deadlines or repeat the rejected two-cycle training comparison.

Cloud remains ON with no active task jobs after final verification. Git remains94e8007 plus uncommitted robustness and timing work; no commit or push was requested this session. Local and cloud handoffs and `PLANS_m24_ppo.md` are updated.
