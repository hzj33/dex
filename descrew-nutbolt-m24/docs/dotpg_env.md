# DOTPG 环境差异说明：Dexh13 Lightbulb vs XHand ScrewDriver

本文总结当前 `Dexh13HoraLightbulb` 相对原 HORA `XHandHoraScrewDriver` 的环境设置、奖励机制、惩罚机制、reset/termination 和 domain randomization 差异。重点面向后续 teacher-student / DOTPG 训练排查：当 DOTPG student 表现异常时，可以先判断问题来自算法，还是来自 Dexh13 + lightbulb 环境迁移。

对比对象：

- 新任务：`configs/task/Dexh13HoraLightbulb.yaml`
- 原任务：`configs/task/XHandHoraScrewDriver.yaml`
- 共享环境逻辑：`dexscrew/tasks/xhand_hora.py`

## 1. 总体结论

`Dexh13HoraLightbulb` 不是完全重写 HORA 环境，而是在原 `XHandHoraScrewDriver` 的旋拧任务骨架上做迁移适配。

保留的核心逻辑包括：

- 仍然复用 `XHandHora` 环境逻辑
- 仍然是 screw-like object 的 1-DOF 旋拧任务
- 仍然使用旋转奖励 + pose/torque/work/point-cloud/proximity penalty/reward 组合
- 仍然使用 HORA 的 privileged info、proprio history、point cloud observation 结构
- 仍然可接 PPO teacher、PADAPT student、DOTPG student

主要变化集中在：

- hand 从 XHand 12 DOF 换成 Dexh13 16 DOF
- object 从 `screw_driver` 换成 `screw_lightbulb`
- reset 距离阈值放宽
- 增加 `two_finger_gate`，要求拇指 + 食指/中指形成有效抓持后再鼓励正向旋拧
- 对 Dexh13 的 pose/torque/work penalty 做动作维度归一化
- object scale randomization 收窄
- object tilt 关闭
- yaml 默认 termination 更偏 debug，训练脚本再显式打开严格 termination
- 禁用 XHand 专用 action mask

整体判断：

- reward 主体差异：中等
- penalty 计算差异：中等偏大
- reset/termination 差异：比较大
- domain randomization 差异：主体相似，但 scale/tilt 明显收敛
- task/object/hand 适配差异：比较大

## 2. 基础环境设置差异

| 项目 | XHandHoraScrewDriver | Dexh13HoraLightbulb | 说明 |
|---|---:|---:|---|
| `initPose` | `screwdriver_inclined` | `screwdriver_inclined` | Dexh13 lightbulb 复用原 screwdriver reset/object 放置分支 |
| `numActions` | `12` | `16` | Dexh13 hand 是 16 DOF |
| `episodeLength` | `800` | `800` | 保持一致 |
| `controlFrequencyInv` | `10` | `10` | 保持 20Hz 控制 |
| `pgain` | `3` | `3` | 保持一致 |
| `dgain` | `0.01` | `0.01` | 保持一致 |
| `action_scale` | `0.05` | `0.05` | 保持一致 |
| `torque_limit` | `300.0` | `300.0` | 保持一致 |
| `rotation_axis` | `+z` | `+z` | 保持同方向旋拧 |
| `clipObservations` | `5.0` | `5.0` | 保持一致 |
| `clipActions` | `1.0` | `1.0` | 保持一致 |

基础控制参数基本没有大改。Dexh13 的迁移不是通过改变 controller 主参数完成的，而是通过 hand asset、root pose、init pose、reward gate、termination 和随机化范围适配。

## 3. Reset 与 Termination 差异

### 3.1 reset 距离阈值

`XHandHoraScrewDriver`：

```yaml
reset_dist_threshold: 0.05
reset_z_threshold: 0.0
```

`Dexh13HoraLightbulb`：

```yaml
reset_dist_threshold: 0.20
reset_z_threshold: 0.0
```

变化：

- `reset_dist_threshold` 从 `0.05m` 放宽到 `0.20m`
- 距离阈值放宽 4 倍

影响：

- Dexh13 + lightbulb 更不容易因为初始指尖距离过远而立即 reset
- 也会让 proximity reward 的距离尺度变宽，因为代码中直接使用 `mean_dist / reset_dist_threshold`

代码中 proximity reward 的计算逻辑：

```python
ratio = mean_dist / self.reset_dist_threshold
proximity_reward = torch.clamp(1.0 - ratio, min=0.0, max=1.0)
```

因此 `reset_dist_threshold` 同时影响：

- finger distance termination
- proximity reward 的有效距离范围

### 3.2 yaml 默认 termination 更偏 debug

`Dexh13HoraLightbulb.yaml` 默认写法：

```yaml
termination:
  grace_steps: 50
  enable_finger_dist: False
  enable_nut_stagnation: False
  enable_no_contact: False
  enable_screw_limit: False
  log: False
```

含义：

- 直接运行 yaml 或可视化脚本时，不会因为这些严格 termination 条件频繁 reset
- 这方便调 init pose、root pose、接触关系

但正式训练脚本会覆盖这些选项。

teacher 脚本通常覆盖为：

```bash
task.env.termination.grace_steps=150
task.env.termination.enable_finger_dist=True
task.env.termination.enable_nut_stagnation=True
task.env.termination.enable_no_contact=True
task.env.termination.enable_screw_limit=True
```

student 脚本通常覆盖为：

```bash
task.env.termination.grace_steps=0
task.env.termination.enable_finger_dist=True
task.env.termination.enable_nut_stagnation=True
task.env.termination.enable_no_contact=True
task.env.termination.enable_screw_limit=True
```

### 3.3 termination 检查项

共享代码里主要有四类 termination：

- max episode length
- finger distance reset
- nut stagnation reset
- no contact reset
- screw joint limit reset

关键逻辑：

```python
finger_dist_reset = torch.logical_or(
    thumb_dist > self.reset_dist_threshold,
    index_dist > self.reset_dist_threshold,
)
```

以及：

```python
nut_pos_stagnant = (nut_pos_variance < self.nut_stagnation_eps) & nut_pos_history_filled
no_contact_reset = no_contact & contact_history_filled
screw_at_limit = current_screw_pos > (screw_upper_limit - 5.0)
```

对 DOTPG 的影响：

- teacher 阶段有 `150` grace steps，能给 PPO 一点探索缓冲
- student 阶段通常 `grace_steps=0`，更严格
- 如果 student 一开始策略偏离 teacher，可能更快触发 termination
- 排查 DOTPG 时要注意 train config/脚本是否和 teacher 的 termination 设置一致

## 4. Reward 主体：继承原 HORA Screw 结构

两者 reward 主公式仍然一致：

```python
reward =
    rotate_reward_scale * rotate_reward
  + pose_diff_penalty * pose_diff_penalty_scale
  + torque_penalty * torque_penalty_scale
  + work_penalty * work_penalty_scale
  + z_dist_penalty * pc_z_dist_penalty_scale
  + rotate_penalty * rotate_penalty_scale
  + proximity_reward * proximity_reward_scale
```

保留项：

- `rotate_reward`
- `pose_diff_penalty`
- `torque_penalty`
- `work_penalty`
- `z_dist_penalty`
- `rotate_penalty`
- `proximity_reward`

所以 Dexh13 lightbulb 的 reward 不是换了一套新 reward，而是在原 screw reward 上增加了 Dexh13/lightbulb 适配项。

## 5. Reward Scale 对比

| reward / penalty scale | XHandHoraScrewDriver | Dexh13HoraLightbulb | 说明 |
|---|---:|---:|---|
| `angvelClipMin` | `-4.0` | `-4.0` | 保持一致 |
| `angvelClipMax` | `4.0` | `4.0` | 保持一致 |
| `angvelPenaltyThres` | `[7.5, 15.0, 30000000, 60000000]` | `[7.5, 15.0, 30000000, 60000000]` | 保持一致 |
| `rotate_reward_scale` | `2.5` | `2.5` | 保持一致 |
| `pose_diff_penalty_scale` | `-0.1` | `-1.5` | Dexh13 启用动作维度归一化后重调 |
| `torque_penalty_scale` | `-3.0` | `-30.0` | Dexh13 启用动作维度归一化后重调 |
| `work_penalty_scale` | `-0.01` | `-0.15` | Dexh13 启用动作维度归一化后重调 |
| `rotate_penalty_scale` | `-0.3` | `-0.3` | 保持一致 |
| `pc_z_dist_penalty_scale` | `-1.0` | `-1.0` | 保持一致 |
| `proximity_reward_scale` | `2.0` | `2.0` | 保持一致 |

表面上 Dexh13 的 `pose/torque/work` penalty scale 大很多，但不能直接理解成惩罚大很多，因为 Dexh13 同时开启了：

```yaml
normalize_penalties_by_num_actions: True
```

## 6. Penalty Normalization 差异

`XHandHoraScrewDriver` 默认不启用动作维度归一化。

`Dexh13HoraLightbulb` 显式启用：

```yaml
normalize_penalties_by_num_actions: True
```

代码逻辑：

```python
if normalize_penalties_by_num_actions:
    denom = float(max(int(self.num_actions), 1))
    pose_diff_penalty = pose_diff_penalty / denom
    torque_penalty = torque_penalty / denom
    work_penalty = work_penalty / (denom * denom)
```

含义：

- Dexh13 有 16 DOF，XHand 是 12 DOF
- pose/torque/work 都是对动作维度求和的项
- 如果不归一化，DOF 更多的手天然会产生更大的 sum penalty
- Dexh13 先按动作维度归一化，再重新设置 scale

对 DOTPG 的影响：

- teacher 的 reward landscape 已经针对 Dexh13 重新平衡
- student 如果使用 teacher action / latent 蒸馏，不需要自己理解 reward
- 但 DOTPG 若有 online RL 或 critic 学习，critic 看到的 reward 分布和原 XHand screwdriver 不完全一样
- 对比实验时不能只看 scale 数值，要看归一化后的实际 penalty magnitude

## 7. 新增 Two-Finger Gate

这是 Dexh13 lightbulb 相对原 XHand screwdriver 最大的 reward 结构变化之一。

`Dexh13HoraLightbulb` 开启：

```yaml
two_finger_gate:
  enable: True
  thumb_fingertip_index: 3
  other_fingertip_indices: [0, 1]
  target: nut_pos
  target_offset: [0.0, 0.0, 0.04]
  near: 0.08
  far: 0.13
  min_mult: 0.20
  power: 1.0
  scale_with_object: True
  apply_positive_vel_only: True
  use_contact_force: True
  contact_force_min: 0.5
  contact_force_max: 2.0
  no_grasp_penalty_scale: -0.3
```

原 `XHandHoraScrewDriver` 没有该配置。

### 7.1 gate 的含义

当前 Dexh13 fingertip 顺序是：

```yaml
fingertipBodies:
  - right_index_tip
  - right_middle_tip
  - right_ring_tip
  - right_thumb_tip
```

所以：

- `thumb_fingertip_index: 3` 表示 thumb
- `other_fingertip_indices: [0, 1]` 表示 index 或 middle

two-finger gate 鼓励：

- 拇指靠近 `nut_pos + target_offset`
- 食指或中指也靠近该位置
- 同时接触力达到一定范围

### 7.2 gate 如何改 rotate reward

代码中：

```python
gate = (w_thumb * w_other) ** power
gate_mult = min_mult + (1.0 - min_mult) * gate
rotate_reward = rotate_reward * gate_mult
```

并且由于：

```yaml
apply_positive_vel_only: True
```

所以主要限制正向旋转奖励。

当没形成两指抓持时：

- 仍保留 `min_mult=0.20` 的一小部分正向旋转奖励
- 但拿不到完整正向旋转奖励

### 7.3 no-grasp penalty

代码中：

```python
two_finger_extra = no_grasp_penalty_scale * (positive_velocity * (1.0 - gate))
```

当前：

```yaml
no_grasp_penalty_scale: -0.3
```

含义：

- 如果策略让灯泡正向转动，但没有形成有效两指抓持，会额外扣分
- 这是为了避免利用碰撞/非预期接触刷旋转奖励

对 DOTPG 的影响：

- expert teacher 的动作更可能包含“先抓住，再旋”的结构
- student 如果只模仿 action，但输入里缺少关键接触/历史信息，可能难以复现 gate 所要求的接触时序
- DOTPG 的 student-state 设计更需要关注 `proprio_hist`、`point_cloud_info`、contact/priv 信息是否与 teacher supervision 对齐

## 8. Proximity Reward 的距离尺度变化

原 XHand screwdriver：

```yaml
reset_dist_threshold: 0.05
proximity_reward_scale: 2.0
```

Dexh13 lightbulb：

```yaml
reset_dist_threshold: 0.20
proximity_reward_scale: 2.0
```

虽然 `proximity_reward_scale` 没变，但 proximity reward 的归一化距离变了。

代码：

```python
proximity_reward = clamp(1.0 - mean_dist / reset_dist_threshold, 0.0, 1.0)
```

影响：

- Dexh13 lightbulb 的 proximity reward 更宽松
- 手指离目标较远时仍可能获得靠近奖励
- 更适合迁移初期学习接近灯泡
- 但也意味着 reward shaping 与 XHand screwdriver 不可直接数值对比

## 9. Domain Randomization 差异

### 9.1 保持一致的随机化

以下项基本沿用 XHand screwdriver：

| 项目 | 数值 |
|---|---:|
| `randomizeMass` | `True` |
| `randomizeMassLower` | `0.04` |
| `randomizeMassUpper` | `0.06` |
| `randomizeCOM` | `True` |
| `randomizeCOMLower` | `-0.001` |
| `randomizeCOMUpper` | `0.001` |
| `randomizeFriction` | `True` |
| `randomizeFrictionLower` | `0.5` |
| `randomizeFrictionUpper` | `8.0` |
| `randomizePDGains` | `True` |
| `randomizePGainLower` | `2.7` |
| `randomizePGainUpper` | `3.3` |
| `randomizeDGainLower` | `0.009` |
| `randomizeDGainUpper` | `0.011` |
| `obs_noise_e_scale` | `0.01` |
| `obs_noise_t_scale` | `0.005` |
| `pose_noise_scale` | `0` |
| `action_noise_e_scale` | `0.01` |
| `action_noise_t_scale` | `0.005` |
| `noisy_rpy_scale` | `0.1` |
| `noisy_pos_scale` | `0.02` |

这些说明 Dexh13 lightbulb 没有完全放弃 HORA 的 domain randomization 框架。

### 9.2 Scale randomization 收窄

原 XHand screwdriver：

```yaml
randomizeScaleList: [0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15, 1.20, 1.25]
randomizeScaleLower: 0.80
randomizeScaleUpper: 1.20
```

Dexh13 lightbulb：

```yaml
randomizeScaleList: [1.0, 1.05, 1.10, 1.15]
randomizeScaleMin: 1.0
randomizeScaleMax: 1.15
randomizeScaleLower: 1.0
randomizeScaleUpper: 1.15
```

变化：

- 不再采样小于 `1.0` 的灯泡
- 最大 scale 控制在 `1.15`
- 相比原 screwdriver 的 scale 范围明显更保守

原因：

- lightbulb 的几何和接触区域不同
- Dexh13 初始 root pose 更敏感
- 过小/过大的 object scale 容易导致抓取困难或初始穿模

对 DOTPG 的影响：

- teacher/student 的 object scale 分布比 XHand screwdriver 更窄
- 如果后续希望做更强泛化，需要逐步扩大 scale range，而不是直接复用 screwdriver 的宽范围

### 9.3 Object tilt 关闭

原 XHand screwdriver：

```yaml
object_tilt: True
```

Dexh13 lightbulb：

```yaml
object_tilt: False
```

影响：

- Dexh13 lightbulb 目前没有启用 ±5 度物体倾斜随机化
- 比原 screwdriver 更容易训练
- 也意味着对倾斜灯泡的鲁棒性更弱

## 10. Object 与 Hand 差异

### 10.1 object 类型

原 XHand screwdriver：

```yaml
object:
  type: 'screw_driver'
  thumb_range_limit: True
  object_tilt: True
```

Dexh13 lightbulb：

```yaml
object:
  type: 'screw_lightbulb'
  thumb_range_limit: False
  object_tilt: False
  init_pos: [0.0, 0.0, 0.0]
  init_pos_noise: [0.005, 0.005, 0.0]
```

主要变化：

- 加载路径换成 `assets/screw/lightbulb/*.urdf`
- 显式给灯泡初始位置
- 关闭 `thumb_range_limit`
- 关闭 `object_tilt`

### 10.2 hand 类型

原 XHand screwdriver：

```yaml
asset:
  handAsset: "assets/xhand_left/urdf/xhand_left.urdf"
```

Dexh13 lightbulb：

```yaml
asset:
  handAsset: "assets/dexh13_hand/urdf/dexh13_hand_right_sim.urdf"
  fingertipBodies: ["right_index_tip", "right_middle_tip", "right_ring_tip", "right_thumb_tip"]
  handRootRPY: [3.1415, 0.3, 3.1415]
  handRootPos: [0.11, 0.020, 0.195]
  handRootPosZScaleComp: 0.06
  handInitPose:
    ...
```

主要变化：

- Dexh13 显式指定 fingertip bodies
- Dexh13 显式指定 root pose
- Dexh13 显式指定 16 DOF `handInitPose`
- Dexh13 关闭 XHand action mask

## 11. Action Mask 差异

Dexh13 lightbulb：

```yaml
apply_action_mask: False
```

原因：

- 原 XHand action mask 带有关节顺序假设
- Dexh13 的 DOF 顺序和 XHand 不同
- 直接套用 XHand mask 可能 mask 到错误关节

对 DOTPG 的影响：

- action 空间是完整 16 DOF Dexh13 action
- student 输出维度必须和 `numActions=16` 对齐
- teacher action 的维度和语义不同于 XHand screwdriver

## 12. Privileged Info 与 Observation 结构

`Dexh13HoraLightbulb` 的 `privInfo` 大体沿用 XHand screwdriver：

- object pose / scale / mass / COM / friction
- object orientation / linear velocity / angular velocity
- fingertip position / orientation / linear velocity / angular velocity
- hand scale
- nut contact / nut position / nut DOF velocity / nut DOF position
- screw joint friction

没有启用：

- tactile
- hand position
- hand orientation
- hand joint pos
- pgain / dgain

这说明：

- teacher-state / privileged-state 的整体结构仍接近原 HORA
- 但具体维度会受 hand DOF、fingertip 数量、object 类型影响
- DOTPG 的 student-state 配置必须重新确认输入维度，不能直接假设和 XHand screwdriver 完全一致

## 13. Sim / PhysX 设置

两者大部分 sim 参数保持一致：

- `dt: 0.005`
- `substeps: 1`
- `up_axis: z`
- `gravity: [0.0, 0.0, -9.81]`
- `num_position_iterations: 8`
- `num_velocity_iterations: 0`
- `max_gpu_contact_pairs: 8388608`
- `contact_offset: 0.002`
- `rest_offset: 0.0`
- `bounce_threshold_velocity: 0.2`
- `max_depenetration_velocity: 1000.0`
- `default_buffer_size_multiplier: 5.0`
- `contact_collection: 1`

主要区别：

- Dexh13 yaml 直接写死 `use_gpu_pipeline: True`、`use_gpu: True`
- XHand yaml 里这些通常通过 Hydra resolver 从顶层配置解析

这对训练语义影响不大，更多是配置组织方式差异。

## 14. 对 DOTPG 训练的具体影响

### 14.1 Teacher 质量更关键

Dexh13 lightbulb 的 reward 有 `two_finger_gate`，teacher 学到的动作更依赖接触时序。

如果 PPO teacher 没有学会稳定抓住灯泡，DOTPG student 很难通过蒸馏补回来。

优先检查：

- teacher 是否能稳定形成拇指 + 食指/中指抓持
- `two_finger/gate` 日志是否上升
- `screw/positive_vel_ratio` 是否合理
- `screw/angular_position` 是否持续增长

### 14.2 Student 输入要覆盖接触相关信息

two-finger gate 依赖：

- fingertip 位置
- contact force
- object/nut 位置
- 历史 proprio

如果 DOTPG student-state 输入过弱，可能出现：

- teacher 能旋，student 学不会抓
- student 能靠近，但不形成有效接触
- student 在无 gate 状态下乱转，reward 被压低

### 14.3 Termination 设置会影响在线 student 训练

student 脚本通常 `grace_steps=0` 且启用全部 termination。

如果 DOTPG online 阶段探索噪声较大，可能更频繁 reset。

排查顺序：

1. 先确认 teacher checkpoint 质量
2. 再确认 student 脚本 termination 是否过严
3. 再检查 `reset_dist_threshold`
4. 再检查 `two_finger_gate` 是否导致正向 reward 太稀疏
5. 最后再调 DOTPG buffer / warmup / exploration noise

### 14.4 Reward 分布不能和 XHand screwdriver 直接对比

原因：

- Dexh13 开启 penalty normalization
- reset/proximity 距离尺度不同
- two-finger gate 会压缩 rotate reward
- object scale/tilt 随机化范围不同

所以 DOTPG 曲线对比时应优先看趋势和任务指标，而不是直接比较 reward 绝对值。

建议关注：

- `screw/angular_position`
- `screw/positive_vel_ratio`
- `two_finger/gate`
- `two_finger/thumb_contact_w`
- `two_finger/other_contact_w`
- reset 频率
- episode length

## 15. 快速差异表

| 类别 | 差异程度 | 主要变化 |
|---|---|---|
| controller | 小 | pgain/dgain/action_scale/torque_limit 基本不变 |
| action dim | 大 | XHand 12 -> Dexh13 16 |
| object | 大 | `screw_driver` -> `screw_lightbulb` |
| reset distance | 大 | `0.05` -> `0.20` |
| reward 主体 | 小到中 | 公式基本保留 |
| reward gate | 大 | 新增 `two_finger_gate` |
| penalty scale | 中到大 | 启用 DOF normalization 后重调 scale |
| domain randomization | 中 | mass/COM/friction/noise 保留，scale 收窄，tilt 关闭 |
| termination | 中到大 | yaml debug-friendly，脚本训练时严格覆盖 |
| observation/privInfo | 中 | 框架保留，维度和语义因 hand/object 改变 |
| sim/physx | 小 | 大体一致 |

## 16. 一句话总结

`Dexh13HoraLightbulb` 相对 `XHandHoraScrewDriver` 的环境不是“重写版”，而是“继承 HORA screw 框架后的迁移适配版”。

真正需要重点关注的是：

- `reset_dist_threshold` 从 `0.05` 放宽到 `0.20`
- `two_finger_gate` 让正向旋转奖励依赖有效两指接触
- Dexh13 对 pose/torque/work penalty 做了 DOF 归一化
- lightbulb 的 scale randomization 更窄，object tilt 关闭
- 训练脚本会覆盖 yaml 默认 termination
- DOTPG student 的输入、teacher checkpoint 质量和 termination 设置必须与这些环境差异对齐
