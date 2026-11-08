# Student Acceptance Summary

## Unified Metrics Table

| Algorithm | Max Current Best | Last EpReward | Last EpLen | Last DoneRate | Last TotalLoss | Median LastFPS | Error Hits | Best CKPT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| teacher_ppo | 1550.49 | N/A | N/A | N/A | N/A | N/A | 0 | `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth` |
| padapt | 1496.08 | N/A | N/A | N/A | N/A | 848.10 | 0 | `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt` |
| purebc | 1495.38 | N/A | N/A | N/A | N/A | 873.90 | 0 | `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt` |
| diffusion_latent | 1786.12 | N/A | N/A | N/A | N/A | 794.00 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt` |
| diffusion_action_chunk | 1203.64 | N/A | N/A | N/A | N/A | 828.70 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_tune_v1 | 1280.19 | N/A | N/A | N/A | N/A | 834.50 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_tune_v2 | 1502.21 | N/A | N/A | N/A | N/A | 862.90 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_detfix_retrain | 1692.80 | N/A | N/A | N/A | N/A | 825.10 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt` |
| diffusion_action_chunk_alignsel_v1 | 1166.49 | N/A | N/A | N/A | N/A | 681.90 | 0 | `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt` |
| padapt_trainrange_full | 32.31 | N/A | N/A | N/A | N/A | 800.90 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_nn/model_best.ckpt` |
| padapt_trainrange_adapt_mu | 69.11 | N/A | N/A | N/A | N/A | 793.50 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_nn/model_best.ckpt` |
| padapt_trainrange_adapt_actor | 7.06 | N/A | N/A | N/A | N/A | 745.70 | 0 | `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_nn/model_best.ckpt` |

## Run Artifacts

- `teacher_ppo`
  - run_dir: `outputs/XHandHoraScrewDriver_teacher/run_a`
  - log: `N/A` (historical teacher training log not persisted under run dir)
  - best_ckpt: `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
  - eval_nominal_log: `outputs/robustness_eval/teacher_nominal.log`
  - eval_light_v2_log: `outputs/robustness_eval/teacher_light_v2.log`
  - eval_summary: nominal `avg_reward=2.918504`, `avg_done_rate=0.000407`; light_v2 `avg_reward=2.729195`, `avg_done_rate=0.000732` (`steps=256`, `seed=42`, `num_envs=48`)
- `padapt`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt/run_a`
  - log: `outputs/XHandHoraScrewDriver_student_padapt/run_a/train_15min.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_tb/events.out.tfevents.1774003538.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
- `purebc`
  - run_dir: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_tb/events.out.tfevents.1774005528.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_purebc/run_a_seed42_15min/stage2_bc_nn/model_best.ckpt`
- `diffusion_latent`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_tb/events.out.tfevents.1774281454.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - deploy_ckpt_note: deterministic multiseed512 on `seed=42,43,44` gives `nominal=2.272142±0.078250`, `light=1.907460±0.124687`, `hard=1.650344±0.149604`. Relative to the previous representative `run_a_latent_robust_light_seed42_15min`, the nominal and hard gains are clear (`2.083162±0.225799 -> 2.272142±0.078250`, `1.582382±0.210612 -> 1.650344±0.149604`). The earlier `light` comparison was too pessimistic because it mixed two different perturbation settings. After aligning `light` to the same stronger `v2` setting (`obs_noise=0.03/0.015`, `force=1.0`, `prob=0.2`), the old representative scores `1.935738±0.079995`, so the gap to `latent_recon05` is only `0.028278` rather than a large regression. Follow-up 15min sweeps `latent_recon03` and `latent_recon04` were both rejected, so `latent_recon05` remains the current best diffusion-latent representative; the next step should focus on seed-level `light_v2` diagnostics rather than more coefficient down-sweeps.
- `diffusion_action_chunk`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774009950.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_fix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_tune_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774026282.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix300k_fa2_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_tune_v2`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774027205.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_mix500k_fa3_cb02_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- `diffusion_action_chunk_detfix_retrain`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774171476.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_tune_fa5_cb02_mix500k_w0_detfix_seed42_15min/stage2_diffusion_action_chunk_nn/model_best.ckpt`
  - deploy_ckpt_note: `model_last.ckpt` is the least-bad deploy candidate in this run, but multiseed512 nominal remains negative
- `diffusion_action_chunk_alignsel_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min`
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774185437.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt`
  - deploy_ckpt_note: selector ranking under nominal seed42/512 is `model_best_student (-0.253256) > model_last (-0.388507) > model_best_student_reward (-0.899636)`; reward-aligned selector works mechanically but is not yet a better deploy proxy
- `diffusion_action_chunk_alignsel_len4_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774187181.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_len4_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student_reward.ckpt`
  - deploy_ckpt_note: under nominal seed42/512 with `++train.ppo.action_chunk_len=4`, selector ranking becomes `model_best_student_reward (-0.328458) > model_last (-0.471724) > model_best_student (-0.893365)`; shorter chunk improves selector ordering but still does not beat `alignsel_v1` len8 best `model_best_student (-0.253256)`
- `diffusion_action_chunk_alignsel_probe_v1`
  - run_dir: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min`
  - log: `N/A` (direct command run without a persisted `train*.log`)
  - event: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_tb/events.out.tfevents.1774252709.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_action_chunk/run_a_action_chunk_alignsel_probe_fa5_cb02_mix120k_seed42_15min/stage2_diffusion_action_chunk_nn/model_best_student.ckpt`
  - deploy_ckpt_note: adding `model_best_deploy_probe` did not beat the previous best selector; nominal seed42/512 ranking is `model_best_student (-0.253256) > model_last (-0.388507) > model_best_deploy_probe (-0.590261) > model_best_student_reward (-0.899636)`
- `padapt_trainrange_full`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_tb/events.out.tfevents.1774022154.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_seed42_15min/stage2_nn/model_best.ckpt`
- `padapt_trainrange_adapt_mu`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_tb/events.out.tfevents.1774023374.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_mu_seed42_15min/stage2_nn/model_best.ckpt`
- `padapt_trainrange_adapt_actor`
  - run_dir: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min`
  - log: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/train_900s.log`
  - event: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_tb/events.out.tfevents.1774024292.wbz-ubuntu22-pc`
  - best_ckpt: `outputs/XHandHoraScrewDriver_student_padapt_trainrange/run_a_trainrange_adapt_actor_seed42_15min/stage2_nn/model_best.ckpt`

## Notes

- `Current Best` comes from training stdout parsing, aligned with existing acceptance scripts.
- TensorBoard metrics are read from the latest event file under each run directory.
- If `done_rate/frame` or env-derived metrics are `N/A`, that run likely predates metric instrumentation.
- Teacher robustness entries above are from fixed-step PPO eval (`+test_num_steps=256`), aligned to the student `EvalSummary` format.

## PLANS_v2 M1 Multiseed Baseline Pack (2026-03-24)

- evidence_doc: `docs/plansv2_m1_baseline_pack.md`
- artifacts_root: `outputs/robustness_eval/plansv2_m1/`
- protocol: `nominal + light_v2 + hard` with seeds `42,43,44`, `steps=256`

| Algorithm | Condition | Reward Mean | Reward Std | Done Mean | Done Std |
|---|---|---:|---:|---:|---:|
| teacher_ppo | nominal | 3.055567 | 0.133547 | 0.000733 | 0.000240 |
| teacher_ppo | light_v2 | 2.915410 | 0.157431 | 0.000787 | 0.000139 |
| teacher_ppo | hard | 2.762886 | 0.172094 | 0.000814 | 0.000239 |
| padapt | nominal | 2.167820 | 0.182929 | 0.001302 | 0.000115 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 |
| purebc | nominal | 1.882908 | 0.437617 | 0.001302 | 0.000199 |
| purebc | light_v2 | 2.190508 | 0.123747 | 0.001329 | 0.000214 |
| purebc | hard | 1.849765 | 0.033950 | 0.001411 | 0.000307 |

## PLANS_v2 Acceptance Readiness Snapshot (2026-03-25)

- readiness_status: `ready_for_acceptance_review_local`
- scope: `PLANS_v2` section `8` acceptance criteria + section `13` stage exit conditions
- note: this snapshot is execution-side/local; governance can still reopen if contradictory new evidence appears.

### Section 8 Acceptance Criteria Check

| Criterion | Local Status | Evidence Pointer(s) |
|---|---|---|
| 8.1 工程验收 | PASS | `docs/plansv2_m1_baseline_pack.md`, `docs/plansv2_m2_gap_gate.md`, `docs/plansv2_m4_residual_compare_pack.md`, `outputs/robustness_eval/plansv2_*` |
| 8.2 baseline 验收 | PASS | `docs/plansv2_m1_baseline_pack.md`, `docs/plansv2_m3_min_compare.md` |
| 8.3 latent 主线验收（条件适用） | PASS (local evidence complete; route not selected) | `docs/plansv2_m2_gap_gate.md`, `docs/plansv2_m3_min_compare.md` |
| 8.4 residual 主线验收（条件适用） | PASS (minimal evidence complete; route not selected) | `docs/plansv2_m4_residual_gate_nominal_scale05.md`, `docs/plansv2_m4_residual_compare_pack.md` |
| 8.5 action diffusion 分支验收 | PASS | `PLANS_v2.md` (positioning), `docs/plansv2_m5_baseline_closure.md` |
| 8.6 论文验收（本阶段可写清） | PASS (local) | `docs/plansv2_m5_baseline_closure.md`, `docs/plansv2_thesis_results_subsection_draft.md` |

### Section 13 Stage Exit Conditions Check

| Exit Condition | Local Status | Evidence Pointer(s) |
|---|---|---|
| evidence hardening 已完成 | PASS | `docs/plansv2_m1_baseline_pack.md`, `docs/plansv2_stage_gate_strict_audit.md` |
| canonical comparison table 已建立 | PASS | `docs/plansv2_m3_min_compare.md`, `docs/stage_acceptance_summary.md` |
| G2 或 G3 至少一条完整走通 | PASS | `docs/plansv2_m2_gap_gate.md` (`local_G2_decision=PASS`) |
| 已明确最适合继续推进方向 | PASS | `docs/plansv2_m5_baseline_closure.md` (`selected_path=non_diffusion_baseline_closure`) |
| action diffusion 地位稳定 | PASS | `PLANS_v2.md`, `docs/plansv2_m5_baseline_closure.md` |
| 下一阶段仅保留一个 supporting axis | PASS | `docs/plansv2_m5_baseline_closure.md` (`supporting_axis_only=baseline robustness/reporting consolidation`) |

### Lightweight Validation Log (2026-03-25)

- reproducibility regeneration check:
  - `bash scripts/eval_plansv2_m3_min_compare.sh 42,43,44`
  - outcome: regenerated `docs/plansv2_m3_min_compare.md` keeps canonical evidence-block fields and preserves `local_m3_conclusion=not_support_latent_mainline`.
- low-cost from-logs regeneration mode validated:
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m1_baseline_pack.sh 0 256 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m2_gap_gate.sh 0 256 42,43,44`
  - `PLANSV2_FROM_LOGS_ONLY=1 bash scripts/eval_plansv2_m4_residual_compare_pack.sh 0 256 42,43,44`
  - outcome: scripts regenerate summaries from existing logs without rerunning IsaacGym eval; canonical evidence fields and local conclusions remain stable.
- breakthrough sweep from all existing robustness logs:
  - command: full `outputs/robustness_eval/**/*.log` parse of `EvalSummary`, grouped by family (`padapt`, `diffusion_latent_like`, `diffusion_residual`) and condition.
  - best robust values found:
    - `padapt/light_v2=2.215234`, `padapt/hard=1.972757`
    - `best_diffusion/light_v2=2.005074`, `best_diffusion/hard=1.764717`
  - robust delta vs padapt:
    - `light_v2: -0.210160`
    - `hard: -0.208040`
  - conclusion: no robustness breakthrough beyond current baseline in existing log pool; current M5 closure remains evidence-consistent.
- bounded breakthrough probe (new eval-only runs, no training):
  - seed-42 shortlist scan (8 latent ckpts, `light_v2 + hard`, 16 runs) under `outputs/robustness_eval/plansv2_breakthrough_probe/`.
  - multiseed follow-up on top-3 candidates (`seed=42,43,44`, 18 runs) under `outputs/robustness_eval/plansv2_breakthrough_probe_multiseed/`.
  - aggregated robust means vs padapt reference (`padapt light_v2=2.079074`, `hard=1.838225`):
    - `run_a_latent_robust_light_seed42_1h`: `light_v2=1.672203` (delta `-0.406871`), `hard=1.393417` (delta `-0.444808`)
    - `run_a_latent_robust_fs06_p012_seed42_15min`: `light_v2=1.662662` (delta `-0.416412`), `hard=1.342301` (delta `-0.495924`)
    - `run_a_latent_robust_mid_seed42_15min`: `light_v2=1.657517` (delta `-0.421557`), `hard=1.371422` (delta `-0.466803`)
  - conclusion: bounded breakthrough probe still does not show diffusion robust edge over padapt.
- script syntax checks passed:
  - `bash -n scripts/eval_plansv2_m1_baseline_pack.sh`
  - `bash -n scripts/eval_plansv2_m2_gap_gate.sh`
  - `bash -n scripts/eval_plansv2_m3_min_compare.sh`
  - `bash -n scripts/eval_plansv2_m4_residual_compare_pack.sh`
- core code compile checks passed:
  - `dexscrew/algo/ppo/ppo.py`
  - `dexscrew/algo/ppo/diffusion_latent_student.py`
- artifact existence spot checks passed:
  - `outputs/XHandHoraScrewDriver_teacher/run_a/stage1_nn/best_reward_1550.49.pth`
  - `outputs/XHandHoraScrewDriver_student_padapt/run_a/stage2_nn/model_best.ckpt`
  - `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
  - `outputs/robustness_eval/plansv2_m1/teacher_ppo_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_m2_gap_gate/diffusion_nominal_s42.log`
  - `outputs/robustness_eval/plansv2_m4_residual_compare_pack/padapt_light_v2_s42.log`

## PLANS_v2 M4 Residual Fallback Gate (Nominal Minimal Pack, 2026-03-24)

- evidence_doc: `docs/plansv2_m4_residual_gate_nominal.md`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_base_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- protocol: `nominal`, seeds `42,43,44`, `steps=256`
- residual_target_definition: `target_x0 = e_gt - base_latent`, `base_latent = tanh(adapt_tconv(proprio_hist))`
- local_g3_readiness_decision: `FAIL` (heuristic defined in evidence_doc)

| Metric | Mean | Std |
|---|---:|---:|
| avg_reward | 2.046098 | 0.226010 |
| avg_done_rate | 0.001194 | 0.000307 |
| latent_mse | 0.072796 | 0.003256 |
| latent_l1 | 0.170207 | 0.004061 |
| action_mse_to_teacher | 0.134181 | 0.005050 |
| residual_abs_mean | 0.247690 | 0.001460 |
| residual_l2_mean | 2.297879 | 0.015312 |
| residual_to_target_ratio | 1.050885 | 0.002266 |
| action_correction_abs_mean | 0.235764 | 0.007813 |
| action_correction_l2_mean | 1.144949 | 0.031846 |
| base_action_mse_to_teacher | 0.259222 | 0.006016 |

## PLANS_v2 M4 Residual Scale05 Stabilization Check (2026-03-24)

- evidence_doc: `docs/plansv2_m4_residual_gate_nominal_scale05.md`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_residual_scale05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- protocol: `nominal`, seeds `42,43,44`, `steps=256`
- residual_target_scaling: `diffusion_residual_target_scale=0.5`
- local_g3_readiness_decision: `PASS` (heuristic defined in evidence_doc)
- note: nominal reward dropped vs unscaled residual baseline (`2.046098 -> 1.589266`)

| Metric | Mean | Std |
|---|---:|---:|
| avg_reward | 1.589266 | 0.063520 |
| avg_done_rate | 0.001410 | 0.000277 |
| latent_mse | 0.094090 | 0.002788 |
| latent_l1 | 0.199492 | 0.003170 |
| action_mse_to_teacher | 0.169102 | 0.008643 |
| residual_abs_mean | 0.250647 | 0.001743 |
| residual_l2_mean | 2.335321 | 0.010726 |
| residual_to_target_ratio | 1.049400 | 0.001607 |
| pred_residual_abs_mean | 0.093719 | 0.000807 |
| pred_to_target_ratio | 0.374187 | 0.003459 |
| action_correction_abs_mean | 0.171668 | 0.005947 |
| action_correction_l2_mean | 0.832797 | 0.023775 |
| base_action_mse_to_teacher | 0.273279 | 0.003731 |

## PLANS_v2 M4 Residual Robustness Compare Pack (2026-03-24)

- evidence_doc: `docs/plansv2_m4_residual_compare_pack.md`
- protocol: `light_v2 + hard`, seeds `42,43,44`, `steps=256`
- variants: `residual_unscaled`, `residual_scale05`, `padapt`
- local_m4_conclusion: `not_support_residual_robust_edge`

| Variant | Condition | Reward Mean | Reward Std | Done Mean | Done Std | CorrAbs Mean | Pred/Target Ratio Mean |
|---|---|---:|---:|---:|---:|---:|---:|
| residual_unscaled | light_v2 | 1.767419 | 0.193093 | 0.001356 | 0.000138 | 0.237466 | 0.667432 |
| residual_unscaled | hard | 1.427755 | 0.125767 | 0.001845 | 0.000139 | 0.239002 | 0.665035 |
| residual_scale05 | light_v2 | 1.711370 | 0.046472 | 0.001546 | 0.000199 | 0.296494 | 0.778173 |
| residual_scale05 | hard | 1.548885 | 0.045004 | 0.001628 | 0.000305 | 0.301035 | 0.769964 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 | N/A | N/A |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 | N/A | N/A |

## PLANS_v2 M5 Stage Convergence (Baseline-First Closure, 2026-03-24)

- decision_doc: `docs/plansv2_m5_baseline_closure.md`
- selected_path: `non_diffusion_baseline_closure`
- key_basis:
  - `docs/plansv2_m3_min_compare.md` -> `local_m3_conclusion=not_support_latent_mainline`
  - `docs/plansv2_m4_residual_compare_pack.md` -> `local_m4_conclusion=not_support_residual_robust_edge`
- stage_outcome:
  - keep `padapt` as current executable mainline baseline
  - keep diffusion lines as appendix/comparative evidence in this stage
- next_stage_supporting_axis: `baseline robustness/reporting consolidation`

## PLANS_v2 M5 Thesis Data Pack (2026-03-24)

- evidence_doc: `docs/plansv2_paper_data_pack.md`
- seed_table_csv: `docs/data/plansv2_paper_seed_table.csv`
- agg_table_csv: `docs/data/plansv2_paper_agg_table.csv`
- delta_table_csv: `docs/data/plansv2_paper_delta_table.csv`
- coverage:
  - seed_rows: `69`
  - aggregated_groups: `23`
  - delta_rows: `12`
- note: all values are parsed from existing M1/M2/M4 logs; no new training runs were added in this pack.

## PLANS_v2 M5 Thesis Result Bundle (2026-03-24)

- evidence_doc: `docs/plansv2_thesis_result_bundle.md`
- main_table_csv: `docs/data/plansv2_thesis_main_table.csv`
- negative_delta_csv: `docs/data/plansv2_thesis_negative_delta_table.csv`
- latex_tables: `docs/data/plansv2_thesis_tables.tex`
- bundle_scope:
  - final main table (`teacher/padapt/purebc/latent_diffusion/residual variants`)
  - negative-result delta table (`m3_latent_vs_baselines`, `m4_residual_compare`)
  - concise baseline-first closure narrative draft

## PLANS_v2 M5 Thesis Results Subsection Draft (2026-03-24)

- draft_doc: `docs/plansv2_thesis_results_subsection_draft.md`
- source_bundle: `docs/plansv2_thesis_result_bundle.md`
- status: `ready_for_manuscript_polish`
- note: no new experiment added; this is a writing-layer consolidation based on frozen M1-M4 artifacts.

## PLANS_v2 M2 Latent Gap-Closing Gate Pack (2026-03-24)

- evidence_doc: `docs/plansv2_m2_gap_gate.md`
- artifacts_root: `outputs/robustness_eval/plansv2_m2_gap_gate/`
- representative_ckpt: `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_recon05_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- protocol: `nominal + light_v2 + hard`, seeds `42,43,44`, `steps=256`
- local_G2_decision: `PASS` (heuristic defined in evidence_doc)

| Mode | Condition | Reward Mean | Reward Std | Done Mean | Done Std | Latent MSE Mean | Latent L1 Mean | Action MSE Mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| diffusion | nominal | 2.062867 | 0.115423 | 0.001221 | 0.000133 | 0.080392 | 0.176390 | 0.146311 |
| diffusion | light_v2 | 1.788645 | 0.271742 | 0.001600 | 0.000595 | 0.080991 | 0.175863 | 0.151461 |
| diffusion | hard | 1.572475 | 0.192113 | 0.001845 | 0.000366 | 0.086572 | 0.179890 | 0.168924 |
| decode_only | nominal | 0.909722 | 0.138378 | 0.002089 | 0.000307 | 0.139231 | 0.250919 | 0.262430 |
| decode_only | light_v2 | 0.915083 | 0.074780 | 0.002143 | 0.000233 | 0.140678 | 0.252794 | 0.264398 |
| decode_only | hard | 0.700938 | 0.112248 | 0.002659 | 0.000378 | 0.144016 | 0.257412 | 0.267246 |

## PLANS_v2 M3 Minimum Credible Comparison (2026-03-24)

- evidence_doc: `docs/plansv2_m3_min_compare.md`
- protocol: `nominal + light_v2 + hard`, seeds `42,43,44`, `steps=256`
- upstream_evidence_m1: `docs/plansv2_m1_baseline_pack.md`
- upstream_evidence_m2: `docs/plansv2_m2_gap_gate.md`
- local_m3_conclusion: `not_support_latent_mainline`
- local_next_step_recommendation: `prepare_m4_residual_fallback_gate`

| Algorithm | Condition | Reward Mean | Reward Std | Done Mean | Done Std |
|---|---|---:|---:|---:|---:|
| latent_diffusion | nominal | 2.062867 | 0.115423 | 0.001221 | 0.000133 |
| latent_diffusion | light_v2 | 1.788645 | 0.271742 | 0.001600 | 0.000595 |
| latent_diffusion | hard | 1.572475 | 0.192113 | 0.001845 | 0.000366 |
| padapt | nominal | 2.167820 | 0.182929 | 0.001302 | 0.000115 |
| padapt | light_v2 | 2.079074 | 0.102648 | 0.001221 | 0.000176 |
| padapt | hard | 1.838225 | 0.105458 | 0.001411 | 0.000138 |
| purebc | nominal | 1.882908 | 0.437617 | 0.001302 | 0.000199 |
| purebc | light_v2 | 2.190508 | 0.123747 | 0.001329 | 0.000214 |
| purebc | hard | 1.849765 | 0.033950 | 0.001411 | 0.000307 |

## PLANS_v4 Closure Snapshot (2026-04-15)

- final_status: `completed_suspend` (governance-confirmed)
- plan_doc: `PLANS_v4.md`
- final_verdict_doc: `docs/plansv4_m3_final_verdict.md`
- thesis_bundle_doc: `docs/plansv4_thesis_result_bundle.md`
- thesis_subsection_draft: `docs/plansv4_thesis_results_subsection_draft.md`

### Key Gate Facts (Seed42, steps=256)

- V3 candidates: `6/6 FAIL` (all under buggy code)
- V4-M0 bug-fixed baseline: `FAIL`
  - `nominal=1.738804`, `light_v2=1.588133`, `hard=1.367269`, `hard_done=0.002279`
  - `delta_hard_vs_v3m0=-0.137635` (hard-stop triggered: `< -0.10`)
- Overall V3+V4 candidates under single-seed gate: `7/7 FAIL`

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- diffusion_positioning: `controlled_negative_result`
- execution_policy: `freeze_diffusion_extension_under_current_scope`

## PLANS_v5 Closure Snapshot (2026-04-15)

- final_status: `completed_conclude`
- plan_doc: `PLANS_v5.md`
- final_verdict_doc: `docs/plansv5_m4_final_verdict.md`
- protocol: `seed=42`, `steps=256`, `nominal + light_v2 + hard`

### Key Gate Facts

- consistency direction:
  - `consistency_baseline`: reward deltas all positive, but `hard_done_delta=+0.000814` (FAIL)
  - `consistency_no_bc`: hard/light reward deltas both negative (FAIL)
  - explicit `infer_steps=2` eval-only check: `hard_done_delta=+0.001465` (FAIL)
- flow direction:
  - `flow_baseline`: `hard_delta=-0.137491` and `hard_done_delta=+0.000814` (FAIL)
- overall:
  - single-seed initial gate pass count: `0`
  - `V5-M3` multiseed phase未触发（无初筛通过候选）

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- diffusion_positioning: `family_partial_signal_but_gate_fail`
- execution_policy: `freeze_v5_under_current_gate_definition`

## PLANS_v5_5 Closure Snapshot (2026-04-15)

- final_status: `completed_accept`
- plan_doc: `PLANS_v5_5.md`
- final_verdict_doc: `docs/plansv5_5_final_verdict.md`
- accepted_candidate: `consistency_boundary_bc_tuned`
- accepted_ckpt: `outputs/XHandHoraScrewDriver_student_consistency/v5_5_m1_boundary_bc_tuned_seed42_15min/stage2_consistency_nn/model_best.ckpt`

### Accepted Config Overrides

- `+train.ppo.consistency_boundary_coef=0.8`
- `+train.ppo.consistency_num_scales=16`
- `+train.ppo.bc_loss_coef=1.2`

### Key Gate Facts

- M1 candidates:
  - A `action_l2_stable`: primary PASS, anti-regression FAIL (`hard reward`回撤超阈值)
  - B `anchor_l2_combo`: primary FAIL (`light_v2`), anti-regression FAIL
  - C `boundary_bc_tuned`: primary PASS + anti-regression PASS
- M2 multiseed (candidate C, seeds 42/43/44):
  - nominal mean `2.336284` (delta vs V3-M0 `+0.661172`)
  - light_v2 mean `2.044725` (delta `+0.406459`)
  - hard mean `1.714471` (delta `+0.209567`)
  - hard done mean `0.001601` (`<=0.002372` pass)

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- promoted_diffusion_candidate: `consistency_boundary_bc_tuned` (accepted under V5.5 gate)
- execution_policy: `use_v5_5_accept_candidate_for_next_consistency_reference`

## PLANS_v6 Closure Snapshot (2026-04-16)

- final_status: `completed_conclude`
- plan_doc: `PLANS_v6.md`
- final_verdict_doc: `docs/plansv6_final_verdict.md`
- protocol: `seed=42`, `steps=256`, `nominal + light_v2 + hard`

### Key Gate Facts

- M1 probes:
  - `capacity_boost`: hard `1.255473` (FAIL, `<1.780`)
  - `longer_train`: hard `1.415326` (FAIL, `<1.780`)
  - `lr_schedule`: hard `1.335743` (FAIL, `<1.780`)
  - triggered `M1 -> M3` stop rule (`hard_delta_vs_v55_seed42 < +0.02` for all probes)
- M3 code-level candidates:
  - `obs_noise_curriculum`: hard `1.387970` (FAIL, `<1.780`)
  - `infer2_align`: hard `1.622737` (FAIL, `<1.780`)
  - `ema_target`: hard `1.723353` (best in M3 but still FAIL, `<1.780`)
- overall:
  - no candidate met `hard >= 1.780` precondition
  - `M4` multiseed phase not triggered
  - `PLANS_v6` final decision = `Conclude`

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- consistency_positioning: `competitive_secondary_but_not_hard_surpass`
- execution_policy: `freeze_v6_and_prepare_next_plan_if_scope_changes`

## PLANS_v7 Closure Snapshot (2026-04-16)

- final_status: `completed_conclude`
- plan_doc: `PLANS_v7.md`
- final_verdict_doc: `docs/plansv7_final_verdict.md`
- protocol: `seed=42`, `steps=256`, `nominal + light_v2 + hard`
- corrected_single_seed_hard_gate: `1.733` (`1.838225 - 0.105458 = 1.732767`)

### Key Gate Facts

- V7-M1 EMA recheck on existing best V6 EMA candidate:
  - `hard = 1.608308`, `hard_done = 0.001628`
  - FAIL (`<1.733`)
- V7-M2 candidate A `ema_obs_combo_seed42_15min`:
  - nominal `0.211057`, light_v2 `0.296649`, hard `0.355474`
  - FAIL by large margin
- V7-M2 candidate B `ema_target_seed42_30min`:
  - `model_best.ckpt` sha1 identical to candidate A
  - nominal `0.211057`, light_v2 `0.296649`, hard `0.355474`
  - FAIL; 30min EMA-only 未产生新 best artifact
- V7-M2 candidate C `ema_alignfix_seed42_15min`:
  - nominal `1.124038`, light_v2 `1.177704`, hard `1.135807`
  - FAIL; 尽管训练信号改善，deploy reward 仍明显低于 gate
- overall:
  - single-seed entry pass count: `0`
  - candidate D: `not_triggered`
  - multiseed M3: `not_triggered`

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- strongest_consistency_reference: `V5.5 consistency_boundary_bc_tuned`
- consistency_positioning: `bounded_risk_closure_complete_but_no_hard_gate_breakthrough`
- execution_policy: `accept_v7_conclude_and_freeze_consistency_extension_under_current_scope`

## PLANS_v8 Closure Snapshot (2026-04-17)

- final_status: `completed_conclude`
- plan_doc: `PLANS_v8.md`
- final_verdict_doc: `docs/plansv8_final_verdict.md`
- protocol: `seed=42`, `steps=256`, `nominal + light_v2 + hard`

### Key Gate Facts

- M0 compatibility:
  - old `v5_m2_flow_baseline` remains numerically stable under the new code path
- M1 eval-only probes:
  - `infer2`: nominal `1.805502`, light_v2 `1.681982`, hard `1.428555`
  - `infer4`: nominal `1.885794`, light_v2 `1.640372`, hard `1.189338`
  - interpretation: `infer2 > infer4` on robust conditions, but both remain below continue-worthy level
- M2 fresh candidates:
  - `zeroinit_bc12`: nominal `0.992769`, light_v2 `0.921589`, hard `1.020256`
  - `align2_rollout`: nominal `1.082436`, light_v2 `1.108353`, hard `0.750190`
  - `align2_anchor`: nominal `1.000670`, light_v2 `0.923009`, hard `0.953265`
  - `align2_bcheavy`: nominal `1.124580`, light_v2 `1.055061`, hard `1.053348`
  - `align4_rollout`: nominal `1.114895`, light_v2 `1.152107`, hard `1.079496`
- overall:
  - best fresh candidate = `align4_rollout`
  - no candidate met the single-seed continue gate:
    - `nominal >= 1.95`
    - `light_v2 >= 1.70`
    - `hard >= 1.55`
    - `hard_done <= 0.002372`
  - `M2.5` 30min extension: `not_triggered`
  - `M3` multiseed: `not_triggered`

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- strongest_accepted_diffusion_reference: `V5.5 consistency_boundary_bc_tuned`
- flow_positioning: `recovery_sprint_completed_but_not_continue_worthy_under_current_scope`
- execution_policy: `accept_v8_conclude_and_freeze_flow_extension_under_current_scope`

## PLANS_v9 Closure Snapshot (2026-04-17)

- final_status: `completed_smoke_accept`
- plan_doc: `PLANS_v9.md`
- final_verdict_doc: `docs/plansv9_final_verdict.md`
- comparable_protocol: `teacher smoke + student 64-step no-noise eval`

### Key Gate Facts

- engineering / wiring:
  - `Dexh13HoraLightbulb` and `XHandHoraLightbulb` both instantiate and reset in Docker Isaac Gym
  - 5 core students all construct on both tasks
  - old path regression stayed alive
- `XHandHoraLightbulb`:
  - teacher smoke PASS (`~ -655 -> -46.5`)
  - student smoke evals:
    - `ProprioAdapt`: `-5.448445`
    - `PureBC`: `-5.379851`
    - `DiffusionLatentStudent`: `-5.226565`
    - `ConsistencyLatentStudent`: `-5.652328`
    - `FlowMatchingLatentStudent`: `-5.256454`
- `Dexh13HoraLightbulb`:
  - default teacher smoke FAIL (`~ -5.6k -> -8.5k`)
  - root-position-only probes FAIL
  - reward-relaxed teacher smoke PASS (`~ -705 ~ -769`)
  - student smoke evals:
    - `ProprioAdapt`: `-9.051320`
    - `PureBC`: `-9.049654`
    - `DiffusionLatentStudent`: `-9.163227`
    - `ConsistencyLatentStudent`: `-9.187637`
    - `FlowMatchingLatentStudent`: `-9.202663`
- common bug closure:
  - stage2 `TemporalConv` input dim no longer hardcodes `24`
  - now follows `train.ppo.proprio_dim`, which unblocks `Dexh13` student smoke

### Current Recommended Baseline Positioning

- keep_mainline_baseline: `padapt`
- current_smoke_matrix_status: `accepted_for_lightbulb_transfer_validation`
- new_task_positioning:
  - `XHandHoraLightbulb`: smoke-stable teacher/student chain
  - `Dexh13HoraLightbulb`: smoke-stable after light reward relaxation
- execution_policy: `close_v9_and_only_open_new_plan_for_longer-run ranking work`

## Mesh Lightbulb Teacher Retune Snapshot (2026-04-21)

- scope: `mesh_stl_lightbulb_runtime` only
- note:
  - this snapshot is not directly comparable to the earlier primitive-runtime `PLANS_v9` smoke numbers
  - runtime asset boundary must stay explicit in later reports

### Key Facts

- runtime cutover:
  - `assets/screw/lightbulb/0000_lightbulb.urdf` now uses STL mesh + contact meshes from `assets/lightbulb/`
  - `assets/lightbulb/0000_lightbulb.urdf` pathing also fixed for local viewer use
- new task closure:
  - `XHandPasiniLightbulb` added and trainable
  - `Dexh13HoraLightbulb` mesh regression path still instantiates and trains, but was not advanced into the 5-run mainline

### Best Teacher Tuning Outcomes

| Task | Best Run | Best 5min Tail `mean_rewards` | Confirm Tail `mean_rewards` | Tuned Default |
|---|---|---:|---:|---|
| `XHandHoraLightbulb` | `mesh_r2_root_translation_5min` | `-12.503010` | `-12.503010` | `handRootPos=[0.0,0.004,0.206]` |
| `XHandPasiniLightbulb` | `mesh_r1_object_align_5min` | `-69.932446` | `-73.043506` | `init_pos=[0.009,0.058,0.0]`, `init_pos_noise=[0.002,0.002,0.0]` |

### Supporting Artifacts

- `XHandHoraLightbulb`:
  - best run dir: `outputs/XHandHoraLightbulb_teacher/mesh_r2_root_translation_5min/`
  - confirm run dir: `outputs/XHandHoraLightbulb_teacher/mesh_hora_final_confirm_5min/`
  - smoke ckpt: `outputs/XHandHoraLightbulb_teacher/mesh_hora_teacher_smoke_ckpt_60s/`
- `XHandPasiniLightbulb`:
  - best run dir: `outputs/XHandPasiniLightbulb_teacher/mesh_r1_object_align_5min/`
  - confirm run dir: `outputs/XHandPasiniLightbulb_teacher/mesh_pasini_final_confirm_5min/`
  - pose dump: `outputs/pose_dumps/pasini_pose_env0_1776763821720.json`
  - smoke ckpt: `outputs/XHandPasiniLightbulb_teacher/mesh_pasini_teacher_smoke_ckpt_60s/`

### Current Recommended Baseline Positioning

- `XHandHoraLightbulb(mesh)`:
  - teacher-side initPose retune complete under the current bounded 5-run protocol
- `XHandPasiniLightbulb(mesh)`:
  - teacher-side initPose retune complete, but stability is weaker than `Hora`
- execution_policy:
  - freeze these tuned teacher defaults
  - only continue with student smoke / ranking under the mesh-runtime boundary
