# PLANS_v2 Strict Execution Replan (User Override)

## Trigger

- trigger_date: `2026-03-24`
- trigger_reason: user requested strict alignment to `PLANS_v2` stage goals/mainline tasks, and explicitly stated this is not yet the paper-acceptance phase.

## Replan Principles

- keep `PLANS_v2.md` as the only milestone authority (no ad-hoc writing goal as critical path).
- keep mainline order strict:
  - Mainline A: evidence -> latent gap -> latent compare
  - Mainline B: residual fallback when A does not produce acceptable gain
- treat writing/polish artifacts as side outputs, not stage-completion criteria.

## Replanned Future Goals

### G0. Strict Stage/Gate Audit (Immediate)
- build a pass/partial matrix for:
  - milestones `M0-M5`
  - gates `G1-G4`
- each row must include:
  - evidence pointer(s),
  - pass rationale,
  - unresolved risk.
- objective: confirm whether current “M5 closure” is truly final or only provisional.

### G1. Canonical Metrics Consistency Audit
- verify cross-doc consistency of canonical values across:
  - `docs/stage_acceptance_summary.md`
  - `docs/plansv2_m*.md`
  - `docs/plansv2_paper_data_pack.md`
- objective: prevent decision drift from table mismatch.

### G2. If Audit Marks Key Items Partial -> Targeted Execution Pack
- M3 extension pack (latent vs padapt vs purebc):
  - robustness-focused repeated eval expansion (same protocol, extra seeds).
- M4 extension pack (residual branch):
  - incremental robust compare under identical protocol,
  - explicit “edge/no-edge” decision update.
- objective: close only the missing plan-required evidence, no broad method expansion.

### G3. Decision Refresh Under G4
- if extended evidence still shows no robust diffusion edge:
  - keep baseline-first closure as final stage decision.
- if new robust edge appears:
  - reopen route ranking with explicit governance note.

## Immediate Next Milestone

- `S0`: generate strict `M0-M5 + G1-G4` audit checklist document with pass/partial and evidence pointers, then use it to decide whether additional M3/M4 experiments are still required.
