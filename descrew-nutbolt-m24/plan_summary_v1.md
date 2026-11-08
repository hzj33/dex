# plan_summary_v1

## 1. Original milestone target
Plan v1 (current `PLANS.md`) targeted a controlled completion of the canonical path:
`XHandHoraScrewDriver -> PPO teacher -> current student -> diffusion student -> unified evaluation`.
The stage objective was to lock reproducible baselines, run at least one comparable diffusion student, and decide the next single contribution axis (efficiency or robustness) without broad refactors.

## 2. What was completed
1. Canonical Hora teacher-student pipeline entrypoints were established and kept usable (`train.py`, `scripts/screwdriver_teacher.sh`, `scripts/screwdriver_student_*.sh`, eval scripts).
2. Baseline algorithm set was implemented and documented as active: `ProprioAdapt`, `PureBC`, `DiffusionLatentStudent`, `DiffusionActionChunkStudent`.
3. P2-style student mechanism split/ablation path was implemented (trainable-range controls in `padapt.py` and train-range scripts).
4. P3 teacher rollout data interface was implemented (`collect_rollout` in PPO path + rollout collection scripts).
5. P3 diffusion consumption hook was implemented for action-chunk rollout pretraining (`rollout_pretrain_*` config path and pretrain script).
6. Robustness/evaluation workflow was expanded with fixed-step and multiseed scripts and summary docs (`eval_screwdriver_student_robustness*.sh`, `docs/robustness_eval_summary.md`, `docs/stage_acceptance_summary.md`).
7. A next-stage draft (`plan2.md`) was produced to focus post-v1 work on latent diffusion gap-closing.

## 3. Validation / evidence
1. Canonical pipeline/entrypoints: code-level evidence in `train.py` dispatch and canonical scripts; this is code-validated (smoke not rerun in this session).
2. Baseline algorithm availability: code-level evidence in `dexscrew/algo/ppo/{padapt,pure_bc,diffusion_latent_student,diffusion_action_chunk_student}.py`; this is code-validated.
3. P2 train-range ablation: code + documented command outcomes in `docs/session_handoff.md` and `docs/stage_acceptance_summary.md`; validation strength is partial (documented runs), not locally re-verified.
4. P3 rollout collection: `collect_rollout` exists in PPO and ProprioAdapt code paths, with collection scripts; documented smoke runs exist; validation strength is partial.
5. Action-chunk rollout pretrain integration: explicit code path (`_run_rollout_pretrain_if_enabled`) and launch script exist; documented smoke exists; validation strength is partial.
6. Robustness evidence: extensive reported fixed-step/multiseed metrics in `docs/robustness_eval_summary.md` and `docs/session_handoff.md`; validation strength is stronger at documentation level, but not directly reproducible from this checkout.
7. Important evidence gap: this checkout has no `outputs/` directory/checkpoints/log artifacts, so run-success claims are currently document-backed only (no local artifact-level verification possible in this session).

## 4. Current working state
Working now:
- Training/eval orchestration for teacher/current student/diffusion students is wired and callable via canonical scripts.
- Unified evaluation scripts for robustness and multiseed summaries are present.
- Teacher rollout collection and action-chunk rollout pretrain hooks are present.

Fragile now:
- Governance intent is split: `PLANS.md` is still action-chunk-first fallback-to-latent, while latest handoff/`plan2.md` is latent-first.
- Diffusion export is not supported in `student_eval.py` (explicitly blocked for diffusion students).

Only scaffolded/partially validated now:
- Most experimental conclusions depend on documented results rather than in-repo artifacts.
- Action-chunk path has substantial documented tuning/scaffolding, but reported deployment-side stability remains weaker than latent/current-student baselines.

## 5. Known gaps and risks
1. Artifact gap: no local run outputs/checkpoints/log files in this repo snapshot; evidence traceability depends on markdown summaries.
2. Documentation consistency gap: some summary docs still carry older values (for example latent best values differ across docs), which can mislead stage closure decisions.
3. Governance drift risk: `PLANS.md` and `plan2.md` imply different main routes; this is unresolved at governance level.
4. Diffusion diagnostics risk (code-path inference): `train.py` `collect_rollout=True` calls `agent.collect_rollout`; diffusion students do not override `collect_rollout`, so diagnostics may use inherited ProprioAdapt-style `act_inference` rather than diffusion sampling semantics.
5. Export-chain risk: diffusion students are excluded from current JIT export path in `student_eval.py`, so “evaluation/export” parity with current student is incomplete.

## 6. Recommended next-stage directions
1. Direction: lock latent diffusion as the active diffusion mainline and run targeted gap-closing against ProprioAdapt.
Execution-level next steps: freeze a single latent representative checkpoint ID, rerun a minimal reproducibility pack (`nominal`, `light_v2`, `hard`, multiseed) with current scripts, and publish one normalized comparison table.
Governance-level decisions for GPT: confirm formal switch from action-chunk-first to latent-first in updated `PLANS.md`.

2. Direction: harden evidence quality before new algorithmic branching.
Execution-level next steps: re-materialize a small on-disk artifact bundle (`outputs/` subset) tied to the accepted claims, and align all summary docs to one canonical metric table.
Governance-level decisions for GPT: define what counts as stage-acceptance evidence (required artifacts vs markdown-only claims).

3. Direction: resolve rollout-diagnostics semantic ambiguity before using it for major decisions.
Execution-level next steps: explicitly validate whether diffusion rollout diagnostics are generated from diffusion policy sampling or inherited base policy inference, then lock a single diagnostic protocol.
Governance-level decisions for GPT: decide whether prior rollout-diagnostics conclusions remain valid if this path is semantically misaligned.

4. Direction: keep action-chunk as a parked exploratory branch, not the primary stage driver.
Execution-level next steps: keep only low-cost maintenance checks and avoid broad action-chunk hyperparameter sweeps until baseline alignment criteria are met.
Governance-level decisions for GPT: decide if action-chunk remains in baseline set for the next stage or is moved to appendix/secondary evidence.

## 7. Recommendation
Recommendation: prioritize the latent-mainline route with evidence hardening first.
Reason: the repo already has the strongest documented diffusion signal on latent, while governance/documentation consistency and artifact traceability are the current bottlenecks; clearing those gives GPT a reliable base for the next AGENTS/PLANS revision.
Assumptions: current markdown-reported results are materially correct, and no hidden external artifact store is required for immediate governance decisions.

## 8. Questions for GPT
1. Should the official next-stage mainline be updated from action-chunk-first (`PLANS.md`) to latent-first (`plan2.md` + latest handoff), and should `PLANS.md` be revised now?
2. What is the mandatory evidence standard for future stage acceptance: markdown summaries only, or markdown plus required artifact bundle (checkpoints/logs/eval outputs)?
3. Should prior rollout-diagnostics conclusions be treated as provisional until diffusion-vs-base rollout semantics are explicitly validated in code/execution?
4. Should diffusion export support be promoted to a required milestone (to match canonical “evaluation/export” intent), or kept out-of-scope for the next stage?
5. Should action-chunk remain a required comparative baseline next stage, or be downgraded to secondary/appendix status until deployment-side metrics become consistently positive?
