# PLANS_v3 M0 Freeze Record

Date: 2026-04-14  
Scope: Freeze reference checkpoint + unified eval protocol before v3 candidate runs.

## 1. Frozen Reference

- algorithm: `DiffusionLatentStudent`
- reference_run:
  `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_seed42_15min/`
- reference_ckpt:
  `outputs/XHandHoraScrewDriver_student_diffusion_latent/run_a_latent_tailcoef02_thr015_sel_anchor003_seed42_15min/stage2_diffusion_nn/model_best.ckpt`
- reference_ckpt_sha1:
  `31150d26646911325cac2aa408d3d42fdd4640a7`

## 2. Frozen Eval Protocol

- entrypoint:
  `scripts/eval_screwdriver_student_robustness.sh`
- algo:
  `DiffusionLatentStudent`
- seeds:
  - single-seed gate: `42`
  - multiseed gate: `42,43,44`
- steps:
  `256`
- conditions:
  - `nominal`
  - `light_v2` (`obs_noise_e=0.03`, `obs_noise_t=0.015`, `force=1.0`, `prob=0.2`)
  - `hard` (`obs_noise_e=0.05`, `obs_noise_t=0.025`, `force=1.5`, `prob=0.3`)

## 3. Reference Seed42 Metrics (for fast delta check)

From:
`outputs/robustness_eval/plansv2_live_tailcoef02_thr015_sel_anchor003/`

- nominal:
  - reward `1.675112`
  - done `0.001302`
- light_v2:
  - reward `1.638266`
  - done `0.001383`
- hard:
  - reward `1.504904`
  - done `0.001872`

## 4. V3 Single-Seed Gate Thresholds

- `delta_hard >= -0.05`
- `delta_light_v2 >= -0.08`
- `delta_done_hard <= +0.0005`

If any fails => reject at single-seed stage.
