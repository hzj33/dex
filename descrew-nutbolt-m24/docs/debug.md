# Pasini Screwdriver Debug Plan

## Why the data was missing
- The extra debug charts (`debug/*`, `term/*`) are only logged when `task.env.log_debug_metrics=True`.
- Your current runs likely had this disabled, so W&B only showed the default reward charts.

## What I changed
- Added a dedicated debug script: `scripts/pasini_screwdriver_teacher_debug.sh`.
- Added debug flags to the task config (default `False`) so you can see the available toggles.

## How to run the 10‑minute debug job
Use the new script (recommended):
```bash
./scripts/pasini_screwdriver_teacher_debug.sh 0 42 debug_run True
```

If you prefer to keep using the original script, add overrides:
```bash
./scripts/pasini_screwdriver_teacher.sh 0 42 debug_run True \
  task.env.log_debug_metrics=True \
  task.env.debug_print_top_hand_contacts=True
```

## Charts to capture from W&B
Reset/termination (this tells us if frequent resets are the issue):
- `term/any_reset_frac`
- `term/finger_dist_frac`
- `term/no_contact_frac`
- `term/nut_stagnant_frac`
- `term/screw_limit_frac`

Contact quality (this tells us if collisions/contacts are too weak):
- `debug/hand_contact_force_mean`
- `debug/hand_contact_force_p95`
- `debug/hand_contact_force/index_mean`
- `debug/hand_contact_force/thumb_mean`

Reach/proximity (this tells us if the hand is even getting close):
- `debug/thumb_dist_mean`
- `debug/index_dist_mean`
- `debug/thumb_dist_margin_mean`
- `debug/index_dist_margin_mean`

Progress/learning:
- `episode_lengths/step`
- `screw/angular_velocity`
- `screw/positive_vel_ratio`

## What I need back from you
- 1–2 screenshots covering the above charts after ~10 minutes.
- Note whether the console shows frequent contact printouts (expected with debug on).

## Notes when resuming from a “best_reward_XXX.pth”
- `Current Best: ...` is a **tracker for this run’s logging/saving**, not a guarantee that the current policy will instantly get that return.
- `mean_rewards` can print as `0` right after starting because it is computed from **completed episodes**; if almost no episodes have terminated yet, the moving average stays near 0.
- If you changed env/reward config (e.g. `initPose`, `reset_dist_threshold`, any `*_scale`), the old `best_reward_XXX` number is **not comparable** anymore. Treat it as “pretrained weights”, not “best score”.

## Fine-tuning the “twitchy tapping” behavior
Goal: reduce the “快速抽搐/点戳” contact style and encourage longer, smoother contact.

Recommended knobs (change one at a time):
- **Lower work penalty** (already set in `configs/task/XHandPasiniScrewDriver.yaml`): `task.env.reward.work_penalty_scale=-0.005`
- **Add smoothness penalty (optional)**: `task.env.reward.action_rate_penalty_scale=-0.002` (try `-0.001 ~ -0.01`)
- **Low-pass target smoothing (optional)**: `task.env.controller.target_smoothing_alpha=0.2` (try `0.1 ~ 0.3`)
- **Make PPO updates smaller** when fine-tuning from a good checkpoint:
  - `train.ppo.learning_rate=5e-4` (or lower)
  - `train.ppo.mini_epochs=2`
  - `train.ppo.kl_threshold=0.01`

Example (resume + fine-tune for 20 min):
```bash
./scripts/pasini_screwdriver_teacher_debug.sh 0 42 ft_run True \
  checkpoint=/home/wbz-ubuntu20/Codefield/dexscrew-repro/outputs/XHandPasiniScrewDriver_teacher/debug_run5_debug/stage1_nn/best_reward_806.36.pth \
  train.ppo.learning_rate=5e-4 train.ppo.mini_epochs=2 train.ppo.kl_threshold=0.01 \
  task.env.reward.work_penalty_scale=-0.005 \
  task.env.reward.action_rate_penalty_scale=-0.002 \
  task.env.controller.target_smoothing_alpha=0.2
```

## Deterministic visualization
To avoid “看起来抖”其实是随机化/噪声导致的错觉，`scripts/vis_pasini_screwdriver_teacher.sh` 已默认关闭 PD/动作/观测噪声与 PD 随机化。
