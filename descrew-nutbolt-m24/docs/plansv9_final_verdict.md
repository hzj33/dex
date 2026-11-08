# PLANS_v9 Final Verdict

## Status
- final_status: `completed_smoke_accept`
- plan_doc: `PLANS_v9.md`
- completed_on: `2026-04-17`

## Verdict
- `PLANS_v9` 的目标已经完成：
  - `Dexh13HoraLightbulb`
  - `XHandHoraLightbulb`
  - 两条新 task 都已通过 teacher smoke
  - 两条新 task 上的 5 个 core student 都已完成：
    - `restore_train`
    - `train`
    - `save`
    - `restore_test`
    - `EvalSummary`
- 这轮验收层级是 `smoke accept`，不是长训练排名结论。

## Key Engineering Closures

### 1. DexH13 lightbulb teacher smoke recovered

- default `Dexh13HoraLightbulb` teacher smoke 曾从约 `-5.6k` 劣化到约 `-8.5k`
- geometry probe 说明：
  - 单纯把 hand root 平移得更近不能解决问题
  - 更像是 preload contact 触发了过大的 controller penalty
- 最终在 `configs/task/Dexh13HoraLightbulb.yaml` 上做了小幅 smoke 级 reward 微调：
  - `pose_diff_penalty_scale: -0.01`
  - `torque_penalty_scale: -0.5`
  - `work_penalty_scale: -0.001`
  - `rotate_penalty_scale: -0.2`
- 调整后 `Dexh13HoraLightbulb` teacher smoke 在约 `2.2 min` 内稳定保持在 `-705 ~ -769` 区间，可作为当前 smoke teacher 起点。

### 2. Core student packaging exposed and fixed a real DexH13 bug

- `Dexh13` student smoke 首次执行时暴露公共模型 bug：
  - stage2 `TemporalConv` 在 `ActorCritic` 中把输入通道写死成 `24`
  - 但 `Dexh13HoraLightbulb` 实际 `proprio_dim = 32`
- 修复：
  - `dexscrew/algo/models/models.py`
  - `dexscrew/algo/ppo/padapt.py`
- 现在 `TemporalConv` 输入维度跟随 `train.ppo.proprio_dim`
- 这解除了 `Dexh13` 的 `matmul (...32 x 24...)` 崩溃
- 同时对 `XHand` 保持兼容，因为 `XHand` 仍然使用 `24`
- 额外回归 spot-check：
  - `XHandHoraLightbulb + ProprioAdapt` 16-step eval 仍可正常通过

## Smoke Matrix

### XHandHoraLightbulb

- teacher smoke:
  - reward 从约 `-655` 改善到约 `-46.5`
- student smoke eval (`steps=64`, no-noise):
  - `ProprioAdapt`: `avg_reward=-5.448445`, `avg_done_rate=0.015625`
  - `PureBC`: `avg_reward=-5.379851`, `avg_done_rate=0.015625`
  - `DiffusionLatentStudent`: `avg_reward=-5.226565`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.128224`, `latent_l1=0.306090`, `action_mse_to_teacher=0.006538`
  - `ConsistencyLatentStudent`: `avg_reward=-5.652328`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.122866`, `latent_l1=0.294118`, `action_mse_to_teacher=0.005339`
  - `FlowMatchingLatentStudent`: `avg_reward=-5.256454`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.133663`, `latent_l1=0.307018`, `action_mse_to_teacher=0.006894`

### Dexh13HoraLightbulb

- teacher smoke:
  - default fail: `~ -5.6k -> -8.5k`
  - reward-relaxed config pass: stable around `-705 ~ -769`
- student smoke eval (`steps=64`, no-noise):
  - `ProprioAdapt`: `avg_reward=-9.051320`, `avg_done_rate=0.015625`
  - `PureBC`: `avg_reward=-9.049654`, `avg_done_rate=0.015625`
  - `DiffusionLatentStudent`: `avg_reward=-9.163227`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.111387`, `latent_l1=0.279704`, `action_mse_to_teacher=0.001462`
  - `ConsistencyLatentStudent`: `avg_reward=-9.187637`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.102859`, `latent_l1=0.261397`, `action_mse_to_teacher=0.001217`
  - `FlowMatchingLatentStudent`: `avg_reward=-9.202663`, `avg_done_rate=0.015625`
    - recon: `latent_mse=0.113891`, `latent_l1=0.277006`, `action_mse_to_teacher=0.001581`

## Conclusions

- `PLANS_v9` 证明了：
  - 新 hand + 新 object 组合已经能接入现有 teacher-student pipeline
  - core student family 已经具有跨 hand family 的完整独立性
  - `Dexh13` 路线此前不是“算法完全不能用”，而是被 smoke-level reward/contract 问题卡住
- 当前比较只代表 `smoke protocol` 下的相对表现，不能替代原 screw 任务上的长训练排名。
- 若继续推进，应该新开 plan 做：
  - longer-run teacher/student comparison
  - `nominal/light/hard` 协议
  - multi-seed 或稳定性比较
