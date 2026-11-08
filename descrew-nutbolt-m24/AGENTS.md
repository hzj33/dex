# AGENTS.md

## Scope
This file tells codeagent how to work in this repo.
Keep it short, stable, and execution-oriented.
Do not restate long project summaries or stage plans here.

## Canonical path
Default experiment path:

`XHandPasiniM24NutBolt -> PPO teacher -> checkpoint/evaluation -> optional student follow-up`

The current objective is cloud IsaacGym PPO teacher training for the
user-specified M24x3, 160 mm nut-and-bolt task with the 16-DOF Paxini/Pasini
hand. Do not resume MuJoCo sim2sim work or start student diffusion/distillation
unless the user explicitly changes the objective again.

## What codeagent should do
Codeagent should focus on local development and cloud execution:

- read only the files needed for the current task
- preserve existing train/eval/export entrypoints while adding the M24 task
- prefer small, reversible changes over broad refactors
- keep existing train/eval/export entrypoints usable
- treat `XHandPasiniM24NutBolt` PPO as the default active path
- keep the M24 bolt base fixed; evaluate destabilizing behavior with contact
  and reaction-wrench metrics rather than a movable-base task variant
- use `PLANS_m24_ppo.md` for current priorities and milestones, not this file

## Cloud training execution
Prefer cloud training, including PPO teacher training and training smokes/capacity
probes. User-authorized exception (2026-09-10): when cloud resources are unavailable,
local training is allowed. Record the provider failure, execution location, local
GPU/capacity, wall-clock limits, and artifacts. Do not silently run both locations.
Development, static checks, analysis and visualization may run locally. Student
training/distillation still requires an explicit change of objective.

When working on a remote/cloud development machine, or preparing commands to run there, codeagent should treat cloud execution as a first-class training target.

Cloud machine detection can be based on explicit user wording, SSH aliases, remote paths such as `/root/code/dexscrew-repro`, or GPU checks such as `nvidia-smi`.

Before cloud execution or preparing cloud commands, codeagent must read cloud-side documentation as needed, not only local notes. At minimum, when the cloud repo is reachable, check `/root/code/dexscrew-repro/docs/cloud_session_handoff.md` before starting or modifying cloud training/eval/sync work. Also read any cloud-side docs, manifests, status files, or pipeline logs referenced there when they are relevant to the requested task.

For user requests like "train PPO for N hours, then train/distill student for M hours", codeagent must:

- generate exact, copy-runnable commands or a remote script that implements the requested sequence
- wrap **each** time-limited phase in its own `timeout`, including both PPO teacher and student/distillation phases
- treat `timeout` exit status `124` as expected when the requested wall-clock limit is reached
- record start/end timestamps, exact command lines, selected checkpoint paths, and `teacher_exit_status` / `student_exit_status` in a persistent log
- select the teacher checkpoint explicitly before launching the student, usually the newest or best `stage1_nn/best_reward_*.pth`
- preserve and sync the best student checkpoint after completion or manual stop
- avoid relying on `max_agent_steps: 10000000000` as a substitute for a requested wall-clock limit

Remote long-running jobs should normally be launched from a script under `outputs/cloud_pipeline_*` and run in `tmux`, with logs under the same output directory.

When cloud execution is used, codeagent must also update a cloud-visible handoff:

- `docs/cloud_session_handoff.md`

This file should live on the cloud repo as the source of truth for active/recent cloud work, so Ubuntu-side and Windows-side agents can quickly see:

- active tmux sessions or confirmation that no cloud job is running
- latest cloud pipeline path and phase/status files
- exact teacher/student checkpoints being evaluated or trained
- key result tables and known invalid/polluted summaries
- the single recommended next cloud action

For cloud work, update both `docs/session_handoff_v2.md` locally and `docs/cloud_session_handoff.md` on/synced to the cloud when the session produces meaningful results or changes the cloud state.

Because the cloud machine usually has stronger resources than the local workstation, codeagent may choose more aggressive training settings there when the user has not pinned them exactly:

- prefer higher `task.env.numEnvs` and compatible `train.ppo.minibatch_size` values to better use a 24GB GPU, large RAM, and fast CPU
- for IsaacGym Hora PPO teacher runs on the current 24GB cloud GPU, prefer probing resource pairs in this order when the user wants aggressive cloud utilization and has not pinned exact values:
  - `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`
  - `task.env.numEnvs=8192`, `train.ppo.minibatch_size=16384`
  - `task.env.numEnvs=6144`, `train.ppo.minibatch_size=12288`
- use the first probe that allocates, reaches stable startup, and is expected to leave meaningful training time inside the requested wall-clock budget
- verify GPU model, memory headroom, CUDA visibility, and initial FPS/VRAM usage before committing to a long run
- back off if startup memory use is close to the GPU limit, if IsaacGym fails to allocate buffers, or if FPS collapses
- keep the chosen cloud settings explicit in the launch command or script so the run is reproducible

## What codeagent can handle directly
Codeagent may directly do:

- local bug fixes
- config wiring
- M24/Pasini PPO launch, capacity probing, monitoring, and checkpoint handling
- training/eval/export script fixes
- student-side implementation changes inside the current pipeline
- baseline reproduction
- small ablations and logging improvements
- documentation updates tied to real code changes

No GPT escalation is needed for these if the work stays inside the current plan.

## Visualization execution
For headed IsaacGym visualization intended for human inspection, codeagent should default to normal real-time playback.
Use algorithm-specific realtime or sleep flags when available, and avoid changing physics `dt` or controller `controlFrequencyInv` just to slow visualization.
Fast-as-possible playback is only the default for headless numeric eval or when the user explicitly asks for speed.

## When codeagent must escalate
Codeagent must stop and write `codeagent_issue.md` if any of the following happens:

- the M24/Pasini joint order or saved init pose cannot be matched to the task
- the M24 asset cannot load with three object bodies and one nut DOF
- PPO startup requires an algorithm-level redesign rather than a bounded fix
- the change is no longer local and would require major refactor
- teacher-student compatibility breaks in a nontrivial way
- external proprietary assets/repos become necessary and are unavailable

## Required issue format
If escalation is needed, `codeagent_issue.md` must contain:

- task background
- current blocker
- evidence
- what has been tried
- local conclusion
- recommended next action

## Reward evidence
Reward/learning changes must cite relevant primary papers or official code and
state a testable hypothesis supported by this task's diagnostics. Distinguish
paper formulas from local adaptations. Do not tune weights without evidence
or override the user's pinned gates, weights, or training limits.

## Governance boundary
`AGENTS.md` is for stable workflow rules only.
`PLANS_m24_ppo.md` is for current-stage goals, milestones, priorities, and acceptance criteria.
If something is stage-specific, put it in `PLANS_m24_ppo.md`, not here.

## Session Handoff (Required)
After each meaningful execution session, codeagent must update:

- `docs/session_handoff_v2.md` (the running handoff for the active M24 PPO work)

`docs/session_handoff.md` remains historical archive only.

The handoff update must include:

- what milestone/subgoal was targeted
- what changed (files + behavior impact)
- what was verified (commands + key outcomes)
- what remains blocked/risky
- the single recommended next step

Purpose:

- allow the user to track progress quickly
- let a new session agent onboard fast without replaying full history

## Session Bootstrap (Required)
At the beginning of a new execution session, codeagent must read:

- `docs/session_handoff_v2.md`
- `PLANS_m24_ppo.md`

Optional historical context only (when needed):
- `docs/stage_acceptance_summary.md`
- `docs/session_handoff.md`

Before any code change or new experiment, codeagent must also do a quick bootstrap check:

- confirm the latest "single recommended next step" from `docs/session_handoff_v2.md`
- avoid rerunning already-failed settings unless testing a clear fix hypothesis
- record the current target milestone/subgoal in the first execution update

Purpose:

- recover the latest experiment state before new changes
- avoid repeating already-failed settings or redundant runs

## Continuous Execution Preference
Unless the user explicitly pauses or redirects, codeagent should run in continuous execution mode:

- continue plan-aligned development and cloud validation/probe work across multiple ideas in one stretch
- do not stop for per-probe confirmation when the work is routine and inside current governance
- stop and report when one of the following is true:
  - a significant experimental breakthrough appears
  - a clearly acceptable engineering/code optimization progress is achieved
  - an escalation boundary in this file is triggered
  - user asks to stop or re-prioritize

## Git milestones
After completing each meaningful task milestone, automatically commit its related code, configuration and handoff updates after appropriate verification. Record measured limitations honestly; a commit does not imply experimental acceptance. Do not include secrets or large generated training artifacts, and do not push unless requested.
