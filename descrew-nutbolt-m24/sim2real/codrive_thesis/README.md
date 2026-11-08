# DexH13 Lightbulb CoDriveThesis Sim2Real Teacher Baseline

This folder freezes the PPO teacher baseline used for final thesis student-distillation comparisons.

Teacher checkpoint:

```text
best_reward_3655.17.pth
```

Source checkpoint:

```text
outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_reward_3655.17.pth
```

Config files:

```text
Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.task.yaml
Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.train.yaml
```

Notes:
- This is the recommended PPO teacher from the 8h `codrive_thesis_env10000_eval_s42_8h_final` run.
- The run's `best_eval.pth` / `best_deploy.pth` were not used because the current eval selector over-penalized normal timeout/max-episode done events.
- Use this checkpoint as the fixed teacher for PAdapt, PureBC, and the four diffusion-family student baselines.
