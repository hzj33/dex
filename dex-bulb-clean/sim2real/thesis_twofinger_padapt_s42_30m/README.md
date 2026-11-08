# DexH13 Lightbulb Thesis Two-Finger ProprioAdapt Sim2Real Package

## What This Is

- Task: `Dexh13HoraLightbulbThesisTwoFinger`
- Algorithm: `ProprioAdapt`
- Student run: `thesis_twofinger_padapt_s42_30m`
- Source teacher: stable two-finger PPO `thumbpose02`
- Visual status: usable, but still has occasional thumb slip/ejection.

## Files

- `teacher_ppo_best_reward_3171.29.pth`
  - Stable two-finger PPO teacher selected for thesis and student distillation.
  - Use this to reproduce the teacher visualization/behavior.
  - SHA1: `9a3b8d587dc2aecec6d0cdb8dcf98f0199c1cbdd`
- `student_policy.pt`
  - TorchScript export from the selected ProprioAdapt student.
  - Intended as the primary handoff artifact for a lightweight deployment runtime.
  - SHA1: `e0dd6a2f6b6d91256ba4ea5bed4947d9b4345b06`
- `model_best.ckpt`
  - Original project checkpoint from `stage2_nn/model_best.ckpt`.
  - Keep this file for exact repo-based loading, re-export, or continued distillation.
  - SHA1: `a92dd01eaf90302b68a1008129d36023cde1373d`
- `train_config.yaml`
  - Full Hydra config captured with the student run.
  - SHA1: `9b28596ac4dd540cbec0f87e5d32c84e33dc4f25`
- `task_config_Dexh13HoraLightbulbThesisTwoFinger.yaml`
  - Task YAML snapshot for action mask, object pose, hand init pose, and controller settings.
  - SHA1: `50616aa953ec7841760ee44e2c29f93975554b53`
- `vis_student.sh`
  - Convenience viewer script copied from the repo.
- `assets/screw/contactviz/0000_lightbulb.urdf`
  - Lightbulb object asset used by `task.env.object.type=screw_contactviz`.
  - SHA1: `561d18a14a20c8abfe5a7fa73fb8350784f56316`
- `assets/lightbulb/contact0.stl`
  - Referenced by the contactviz URDF.
  - SHA1: `4824fe2579b08d10dce74a9cc5d05d33255078b4`
- `assets/lightbulb/contact1.stl`
  - Referenced by the contactviz URDF.
  - SHA1: `792d3e06874eb3379e5929818efe216dbe48302a`

## Policy Interface Notes

- Action dimension: `16`.
- Active fingers:
  - index joints/actions `0:4`
  - thumb joints/actions `12:16`
- Masked/locked fingers:
  - middle joints/actions `4:8`
  - ring joints/actions `8:12`
- Proprio history length: `30`.
- The TorchScript policy returns `(mu, extrin, extrin_gt)`.
  - Use `mu` as the normalized action output, clamped to `[-1, 1]`.

## Important Deployment Caveat

The exported TorchScript model follows the current repo's `student_eval.py` export path. It stores normalization buffers, but the traced forward path expects the same normalized input convention used during export. For an exact, safest reproduction inside this repo, load `model_best.ckpt` with `ProprioAdapt`. For an external real-robot runtime, confirm the observation schema and normalization path before sending actions to hardware.

## Source Paths

- Teacher checkpoint source:
  - `outputs/Dexh13HoraLightbulb_teacher_thesis_twofinger/thesis_twofinger_thumbpose02_diag_s42_1h/stage1_nn/best_reward_3171.29.pth`
- Student checkpoint source:
  - `outputs/Dexh13HoraLightbulb_student_padapt_thesis_twofinger/thesis_twofinger_padapt_s42_30m/stage2_nn/model_best.ckpt`
- Contactviz asset source:
  - `assets/screw/contactviz/0000_lightbulb.urdf`
  - `assets/lightbulb/contact0.stl`
  - `assets/lightbulb/contact1.stl`

## Asset Restore Notes

To restore the contactviz lightbulb into a clean branch, copy the packaged `assets/` directory back to the repo root while preserving relative paths:

```bash
cp -r sim2real/thesis_twofinger_padapt_s42_30m/assets/* assets/
```

There is no `assets/screw/contactviz/0000_lightbulb.npy` in this run. The original training path used the code fallback point cloud for this object, so the missing `.npy` is not required for reproducing the saved behavior.

## Re-Visualize In Sim

From repo root:

Teacher PPO:

```bash
./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_teacher_thesis_twofinger.sh 0 42 thesis_twofinger_thumbpose02_diag_s42_1h
```

Student ProprioAdapt:

```bash
./docker-run-isaacgym.sh bash scripts/vis_dexh13_lightbulb_student_padapt_thesis_twofinger.sh 0 42 thesis_twofinger_padapt_s42_30m
```
