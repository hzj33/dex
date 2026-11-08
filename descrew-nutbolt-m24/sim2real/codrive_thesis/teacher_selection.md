# CoDriveThesis PPO 8h Final Selection

Run:
- cache: `codrive_thesis_env10000_eval_s42_8h_final`
- command log: `outputs/local_pipeline_codrive_thesis_ppo8h/codrive_thesis_env10000_eval_s42_8h_final/command.txt`
- status: timeout-complete, exit status `124`
- start: `2026-05-04T04:18:07+08:00`
- end: `2026-05-04T12:18:11+08:00`

Recommended PPO baseline:

```text
outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_reward_3655.17.pth
```

Reason:
- It is the highest training-reward checkpoint from the 8h run.
- The eval row at `501000000` steps reports `avg_reward=5.3087`, `no_contact_frac=0`, `screw_limit_frac=0`, `thumb_slip/score=0.00677`, and `screw/angular_velocity=1.14231`.
- The built-in `eval_select` score did not select it because the current score applies `done_penalty=2000` to all done events, including `term/max_eps_frac` / `time_outs`.

Do not use as the final baseline without a corrected/post-hoc selector:

```text
outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_eval.pth
outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_thesis/codrive_thesis_env10000_eval_s42_8h_final/stage1_nn/best_deploy.pth
```

These files were saved at the first eval point (`20040000` steps, train reward `1703.26`) and were never overwritten. They are early checkpoints selected mainly because their total done rate was very low, not because their task reward or rotation quality was best.
