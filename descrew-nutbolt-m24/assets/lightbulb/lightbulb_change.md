# 新灯泡 URDF 迁移指南

适用于当前 DexScrew / HORA 衍生项目，覆盖 `teacher(PPO) + PADAPT + DOTPG` 的完整迁移链路。  
目标读者是已经会修改 task yaml、运行训练脚本、排查 Isaac Gym 任务的项目开发者。

---

## 1. 背景与适用范围

当前项目采用的是一条“最小侵入式”的灯泡迁移路线：

- 不重写整套环境逻辑
- 尽量复用 `dexscrew/tasks/xhand_hora.py`
- 通过 `URDF + asset 路径 + task yaml + 训练脚本` 接入新灯泡

这条路线的核心假设是：

- 新灯泡仍然属于当前项目支持的 `screw-like object`
- 环境仍然可以把它理解成“可旋拧物体”
- reward、termination、DOF 读取和接触逻辑仍然大体成立

所以本指南适用于：

- 想替换一个新的灯泡 URDF
- 继续用现有灵巧手（例如 `RBHand`、`Dexh13`）去训练 teacher / student
- 希望尽量不修改共享环境主逻辑

本指南不适用于：

- 物体不是旋拧类对象
- 物体结构完全不符合当前 `screw_*` 任务范式
- 需要重新设计 object loader、reward 结构或任务拓扑

---

## 2. 当前项目中已验证的迁移路径

目前仓库已经验证了两条基于 lightbulb 的迁移样例：

- `RBHandHoraLightbulb`
- `Dexh13HoraLightbulb`

这两条任务都没有新写一套 object 环境，而是走相同的接入方式：

- object 通过 `object.type: screw_lightbulb` 接入
- 灯泡 URDF 放在 `assets/screw/lightbulb/`
- task yaml 放在 `configs/task/`
- teacher / student 训练沿用现有 `train.py` 与对应 shell 脚本

当前依赖的核心复用逻辑在：

- `dexscrew/tasks/xhand_hora.py`

这意味着：

- 手型不同可以主要通过 yaml 适配
- 灯泡不同也尽量通过 asset 和 yaml 适配
- 只有在物理稳定性、旋拧轴解释、接触质量明显不兼容时，才需要额外改共享代码

---

## 3. 新灯泡 URDF 的放置位置与命名规则

推荐放置路径：

```text
assets/screw/lightbulb/<your_bulb>.urdf
```

当前项目的 `lightbulb` 资产目录就是：

```text
assets/screw/lightbulb/
```

当前仓库中的已有示例：

```text
assets/screw/lightbulb/0000_lightbulb.urdf
```

### 命名与目录建议

- 推荐保持在 `assets/screw/lightbulb/` 下，不要一开始单独新开 object 类别
- 文件名建议使用稳定的编号或语义名，例如：
  - `0001_lightbulb.urdf`
  - `e27_bulb_longneck.urdf`
  - `rb_bulb_v2.urdf`

### 如果想让旧灯泡和新灯泡共存

可以把多个 URDF 都放在同一个目录下，例如：

```text
assets/screw/lightbulb/0000_lightbulb.urdf
assets/screw/lightbulb/0001_lightbulb_long.urdf
assets/screw/lightbulb/0002_lightbulb_fat.urdf
```

然后通过 task yaml 中的：

- `sampleProb`

去管理不同 object 的采样概率。

### 如果不想共用现有目录

也是可以的，但那已经不是本指南推荐路线。  
如果你想把新灯泡放到别的目录，就需要同步调整：

- object type 命名
- object 资产扫描逻辑
- 可能还要改 `xhand_hora.py` 的 object 加载分支

本指南默认不走这条路线，而是优先复用现有 `screw_lightbulb` 路径。

---

## 4. 新灯泡 URDF 的结构要求

新灯泡不是“能显示出来”就算接入成功。  
它至少要满足当前环境对旋拧物体的结构假设。

### 必须满足的要求

- 有明确的旋拧关节，环境能够把它解释成当前的 `nut` 旋转 DOF
- 旋转轴方向要和 task yaml 中的 `rotation_axis` 以及 reward 方向保持一致
- 碰撞体必须可用，接触面要能支持手指稳定接触，而不只是有视觉 mesh
- 刚体划分要与当前 screw-like object 的读取方式兼容，通常应能映射到类似 `base / bolt / nut` 的结构
- 关节上下限要合理，尤其要和 `screw_upper_limit` 对齐
- URDF 中不要存在明显异常的惯性、质量、速度上限、collision 配置

### 不是只看 Viewer

即使它在 Viewer 里看起来“加载成功”，仍然可能无法训练，因为训练还依赖：

- object DOF 被正确读取
- reward 能正确感知旋转方向
- termination 能正确判断停滞或异常
- 接触在 PhysX 下数值稳定
- 初始位姿不会导致严重穿模或无接触

### 推荐项

- 优先使用简洁稳定的 collision，而不是复杂视觉网格
- 如果视觉 mesh 很复杂，collision 应单独简化
- 质量和惯性尽量物理合理，不要极端偏小或偏大
- 对需要频繁接触的部位，尽量使用平滑、连续的接触几何

### 点云说明

当前代码会尝试为 screw-like object 加载对应点云。  
如果没有 `.npy` 点云文件，当前实现会退回到 cylinder fallback 点云。

这意味着：

- 没有点云文件时通常不会直接报错
- 但 fallback 只是兼容策略，不代表对新灯泡是最优表示
- 如果后续 teacher/student 很依赖几何观测，建议补充对应点云资产

---

## 5. 如何新建一个灯泡 task yaml

推荐做法不是从零开始写，而是：

- 复制最接近的现有 lightbulb task yaml
- 再局部修改

不要从 `screwdriver` 或 `nutbolt` task 直接起草。  
优先从当前已验证的 lightbulb 任务复制。

### 推荐基底

如果你要给 `RBHand` 迁移新灯泡，优先复制：

- `configs/task/RBHandHoraLightbulb.yaml`

如果你要给 `Dexh13` 迁移新灯泡，优先复制：

- `configs/task/Dexh13HoraLightbulb.yaml`

### 必须修改的字段

新建 yaml 时，下面这些字段必须逐项检查：

- `name`
- `env.numActions`
- `env.rotation_axis`
- `env.object.type`
- `env.object.init_pos`
- `env.object.init_pos_noise`
- `env.object.screw_upper_limit`
- `env.asset.handAsset`
- `env.asset.fingertipBodies`
- `env.asset.handRootPos`
- `env.asset.handRootQuat`
- `env.asset.handRootRPY`
- `env.asset.handInitPose`
- `env.apply_action_mask`
- `env.wrist.*`（仅当该手型需要 wrist 运动学辅助）

### 必须理解的含义

- `name`
  - 这是新的 task 名称，后续 `train.py task=<TaskName>` 会直接用到

- `env.numActions`
  - 必须和 hand asset 的 DOF 数一致
  - 如果和 hand URDF 不一致，环境初始化会直接失败

- `env.rotation_axis`
  - 定义旋拧方向的参考轴
  - 必须与新灯泡 URDF 的旋转关节方向一致

- `env.object.type`
  - 如果你仍沿用现有加载路线，保持 `screw_lightbulb`

- `env.object.init_pos` / `init_pos_noise`
  - 控制灯泡初始位置
  - 新灯泡尺寸或外形变化后，这一项通常最先需要调整

- `env.object.screw_upper_limit`
  - 应与新灯泡 URDF 的旋拧关节上限保持一致或兼容

- `env.asset.handRoot*`
  - 决定手相对于灯泡的初始姿态
  - 常常是能否进入“可训状态”的第一关键项

- `env.asset.handInitPose`
  - 必须覆盖该 hand asset 的全部可控关节
  - 这决定了初始抓取/逼近姿态

- `env.apply_action_mask`
  - 如果当前手型不是 XHand 原始关节排列，通常要检查是否应关闭

- `env.wrist.*`
  - 只在某些手型本体自由度不足、需要借助手腕绕轴时开启

---

## 6. 与新灯泡最相关的 task yaml 调参项

新灯泡迁移时，最常需要调的不是算法，而是 task yaml 里的几何与接触参数。

### 1. object 初始位置与噪声

重点字段：

- `env.object.init_pos`
- `env.object.init_pos_noise`

什么时候优先改：

- 新灯泡更大
- 新灯泡更细
- 灯泡头部更长
- 初始就穿模
- 初始完全接触不到

### 2. 手根位置与姿态

重点字段：

- `env.asset.handRootPos`
- `env.asset.handRootQuat`
- `env.asset.handRootRPY`
- `env.asset.handInitPose`

什么时候优先改：

- 手指对不准灯泡
- 只能碰到灯泡一侧
- 抓取姿态明显不自然

### 3. `randomizeScale` 范围

重点字段：

- `randomizeScaleList`
- `randomizeScaleLower / Upper`
- `randomizeScaleMin / Max`

什么时候优先改：

- 新灯泡尺寸变化大
- 随机尺度后容易穿模
- teacher 在大部分 env 下都一开始失败

### 4. `two_finger_gate`

重点字段：

- `thumb_fingertip_index`
- `other_fingertip_indices`
- `target_offset`
- `near / far`
- `use_contact_force`

什么时候优先改：

- 手能碰到灯泡，但旋拧奖励上不去
- 两指抓取不足，正向旋拧不稳定
- 新灯泡头部位置和旧灯泡不同

### 5. `reset_dist_threshold`

什么时候优先改：

- 初始 reset 太频繁
- 稍微偏一点就被重置
- 新灯泡比旧灯泡大或远

### 6. termination 严格程度

重点字段：

- `termination.grace_steps`
- `enable_finger_dist`
- `enable_nut_stagnation`
- `enable_no_contact`
- `enable_screw_limit`

什么时候优先改：

- 调试时想先让场景稳定可视化
- 新灯泡接触建立较慢
- teacher 训练初期总是过早结束

### 7. `wrist.enable` 与 `wrist.action_index`

什么时候优先改：

- 仅靠手指无法有效驱动旋拧
- 手型本身对绕轴运动能力弱
- 明显需要额外腕部辅助

### 调参优先顺序建议

如果新灯泡比旧灯泡明显更大、更细或更长，建议优先按这个顺序排：

1. `object.init_pos / init_pos_noise`
2. `handRootPos / handRootRPY / handRootQuat`
3. `handInitPose`
4. `randomizeScale`
5. `two_finger_gate`
6. `wrist`
7. termination

---

## 7. teacher 迁移流程

teacher 是整个迁移链的起点。  
如果 teacher 本身训练不起来，后面的 PADAPT 和 DOTPG 都不会好。

### 步骤 1：准备新 task yaml

- 从现有 `*Lightbulb.yaml` 复制一份
- 改成新的 task 名称和对应 hand / bulb 参数

例如：

- `configs/task/RBHandHoraMyBulb.yaml`
- `configs/task/Dexh13HoraMyBulb.yaml`

### 步骤 2：准备或确认 hand asset 与 bulb URDF

至少确认：

- hand asset 路径正确
- new bulb URDF 已放在 `assets/screw/lightbulb/`
- 旋转关节方向与 `rotation_axis` 一致
- object 初始位姿不会一开始就明显异常

### 步骤 3：先做 debug / 可视化 sanity check

推荐先在小规模 env 下检查：

- 是否能正确加载
- 是否接触合理
- 是否没有大面积瞬间 reset

如果是新 task，建议先手工运行一个最小 teacher 命令模板：

```bash
python train.py task=<NewTaskName> \
  headless=False \
  seed=42 \
  train.algo=PPO \
  task.env.numEnvs=1 \
  wandb_activate=False
```

如果只想看快速 sanity check，也可以先用：

```bash
python train.py task=<NewTaskName> \
  headless=True \
  seed=42 \
  train.algo=PPO \
  task.env.numEnvs=64 \
  wandb_activate=False
```

### 步骤 4：训练 PPO teacher

通用模板：

```bash
python train.py task=<NewTaskName> \
  headless=True \
  seed=<SEED> \
  experiment=rl \
  train.algo=PPO \
  task.env.numEnvs=<NUM_ENVS> \
  wandb_activate=True \
  train.ppo.output_name=<TeacherOutputName>
```

### 更推荐的做法

优先参考现有脚本复制一个新的 teacher 脚本，而不是长期手写大串命令。

可以参考：

- `scripts/rbhand_lightbulb_teacher.sh`
- `scripts/dexh13_lightbulb_teacher.sh`

如果新灯泡仍然需要与旧 `RBHand` / `Dexh13` lightbulb 相近的训练策略，优先复制这些脚本再改：

- `task=<NewTaskName>`
- `train.ppo.output_name=...`
- 以及必要的 `wrist` / `termination` 参数

### teacher 成功标准

至少满足以下几个条件再进入 student 阶段：

- 灯泡旋转方向正确
- reset 不会大面积瞬间触发
- reward 有明显正向增长
- viewer 下接触姿态合理
- 能稳定看到手指接触并尝试驱动旋拧

---

## 8. student 迁移流程

student 迁移默认分成两条路线：

- `PADAPT / ProprioAdapt`
- `DOTPGStudent`

两者都依赖 teacher checkpoint。

### 8.1 PADAPT / ProprioAdapt

PADAPT 是当前项目里的 student 蒸馏路线之一。

必须注意：

- 依赖 teacher checkpoint
- 需要 `train.ppo.proprio_adapt=True`
- teacher 和 student 的输入设定必须一致

这里的一致，至少包括：

- `train.ppo.priv_info`
- `train.ppo.use_point_cloud_info`
- `task.env.hora.point_cloud_sampled_dim`

如果这些设置不一致，student 和 teacher 的输入维度或 latent 对齐关系就可能出问题。

### PADAPT 最小命令模板

```bash
python train.py task=<NewTaskName> \
  headless=True \
  seed=<SEED> \
  train.algo=ProprioAdapt \
  train.ppo.proprio_adapt=True \
  checkpoint=<TEACHER_CKPT> \
  task.env.numEnvs=<NUM_ENVS> \
  wandb_activate=False \
  train.ppo.output_name=<StudentPadaptOutput>
```

### 更推荐的做法

参考现有脚本复制出新的 PADAPT 脚本：

- `scripts/dexh13_lightbulb_student_padapt.sh`

如果未来补了 `RBHand` 的 PADAPT 路线，也建议按同样脚本风格保持一致。

---

### 8.2 DOTPGStudent

DOTPG 是当前项目里的另一路 student 蒸馏 / 模仿学习框架。

它区分两种状态模式：

- `teacher-state`
- `student-state`

默认推荐：

- `student-state`

因为它更贴近真实 student 设定，即 policy 不直接吃 privileged input。

### teacher ckpt 在 DOTPG 里的作用

teacher checkpoint 不只是“初始化一下”而已，它实际承担了：

- 生成 expert action
- 提供 teacher latent label
- 支持 student-state 的表征预热

### `dynamic_state=True` 的意义

如果用 `student-state`，推荐开启：

- `dynamic_state=True`

原因是：

- replay / expert buffer 中存 raw `obs + proprio_hist`
- 采样时用当前 encoder 动态重建 state
- 可以减少 student encoder 持续训练带来的表征漂移

### DOTPG 最小命令模板

推荐基于 `config_dotpg_student.yaml`：

```bash
python train.py --config-name=config_dotpg_student.yaml \
  task=<NewTaskName> \
  train=<NewDotpgTrainConfig> \
  checkpoint=<TEACHER_CKPT> \
  headless=True \
  seed=<SEED> \
  task.env.numEnvs=<NUM_ENVS> \
  wandb_activate=True
```

### 更推荐的做法

参考现有脚本复制新的 DOTPG student 脚本：

- `scripts/rbhand_lightbulb_student_dotpg_student.sh`
- `scripts/dexh13_lightbulb_student_dotpg_student.sh`

复制后优先修改：

- `task=<NewTaskName>`
- `train=<NewDotpgTrainConfig>`
- `checkpoint=<TEACHER_CKPT>`
- 必要时追加 `wrist`、termination、warmup 相关参数

---

## 9. 新建 train config / script 的建议路线

如果只是“新灯泡 + 已有手型”，不建议一开始就大改训练体系。

### 推荐路线

1. 先复制最接近的 `configs/task/*Lightbulb*.yaml`
2. 再复制最接近的 `configs/train/*Lightbulb*.yaml`
3. 如果只是小改参数，甚至可以先不新建 train config，直接通过命令行覆盖
4. 先让 teacher 跑通
5. teacher 稳定后，再固化 student config 和脚本

### train config 建议

如果是：

- `RBHand + 新灯泡`

优先参考：

- `configs/train/RBHandHoraLightbulb.yaml`
- `configs/train/RBHandHoraLightbulb_DOTPG_Student.yaml`

如果是：

- `Dexh13 + 新灯泡`

优先参考：

- `configs/train/Dexh13HoraLightbulb.yaml`
- `configs/train/Dexh13HoraLightbulb_DOTPG_Student.yaml`

### script 建议

teacher 脚本优先复制：

- `scripts/rbhand_lightbulb_teacher.sh`
- `scripts/dexh13_lightbulb_teacher.sh`

PADAPT 脚本优先复制：

- `scripts/dexh13_lightbulb_student_padapt.sh`

DOTPG student 脚本优先复制：

- `scripts/rbhand_lightbulb_student_dotpg_student.sh`
- `scripts/dexh13_lightbulb_student_dotpg_student.sh`

### 为什么不建议一开始新增太多脚本

因为新灯泡迁移初期真正不稳定的通常不是脚本层，而是：

- 手物位姿
- object 关节方向
- collision / contact
- reward / termination 是否合理

所以建议先少量覆盖、快速验证，等 teacher 跑通后再把脚本和 train config 固化下来。

---

## 10. 常见失败模式与排查顺序

新灯泡迁移失败时，建议固定按下面顺序排查，不要一上来就怀疑算法。

### 1. URDF 是否被正确加载

典型现象：

- 直接初始化失败
- 物体不显示
- 刚体或关节数量不符合预期

优先检查：

- asset 路径
- URDF 文件本身
- 关节和 collision 定义

### 2. DOF / 旋转轴方向是否正确

典型现象：

- reward 一直为负或接近零
- 旋拧方向和预期相反
- viewer 里能动，但训练目标始终不成立

优先检查：

- URDF 旋转关节轴
- task yaml 中 `rotation_axis`
- `screw_upper_limit`

### 3. 初始手物相对位姿是否合理

典型现象：

- 一开始穿模
- 一开始完全碰不到
- reset 非常频繁

优先改：

- `handRootPos`
- `handRootQuat / handRootRPY`
- `object.init_pos`
- `object.init_pos_noise`

### 4. collision / contact 是否稳定

典型现象：

- 抖动
- NaN
- 接触看似发生但无法稳定施力

优先检查：

- URDF collision 几何
- 质量 / 惯性
- 是否使用过于复杂的 mesh 直接做 collision

### 5. reward 是否能给出正反馈

典型现象：

- teacher 一直学不会
- 接触存在但没有正向旋拧趋势

优先改：

- `two_finger_gate`
- `rotation_axis`
- `reward` 相关 scale
- `target_offset`

### 6. termination 是否过严

典型现象：

- 训练一开始就大规模结束
- viewer 中还没建立抓取就 reset

优先改：

- `termination.grace_steps`
- `enable_finger_dist`
- `enable_no_contact`
- `enable_nut_stagnation`

### 7. 尺寸随机化是否过激

典型现象：

- 小部分 env 正常，大部分 env 初始异常
- scale 稍一打开就训练崩

优先改：

- `randomizeScaleList`
- `randomizeScaleLower / Upper`
- `randomizeScaleMin / Max`
- `handRootPosZScaleComp`

### 8. student 不收敛

如果 teacher 正常、student 不正常，优先排查：

- teacher checkpoint 质量是否足够
- PADAPT 的 teacher/student 输入设定是否一致
- DOTPG 的 `state_mode` 是否选对
- DOTPG 是否开启了合适的 warmup 与 `dynamic_state`
- expert buffer / replay buffer 是否在合理工作

---

## 11. 最终迁移 checklist

按下面顺序执行，最容易把新灯泡迁移做成：

1. 把新灯泡 URDF 放到 `assets/screw/lightbulb/`
2. 确认继续沿用 `object.type: screw_lightbulb`
3. 复制最接近的 `*Lightbulb.yaml` 作为新 task yaml
4. 修改手型参数、object 初始位姿、rotation axis、hand init pose
5. 做 viewer / debug sanity check
6. 小规模 env 跑 PPO teacher 检查是否稳定
7. 正式训练 PPO teacher
8. 可视化 teacher，确认旋拧方向与接触姿态正常
9. 基于 teacher checkpoint 跑 PADAPT 或 DOTPG student
10. 可视化 student，确认能继承 teacher 的基本行为
11. 训练稳定后，再补脚本和 train config 的正式固化

---

## 12. 结论

在当前项目中，替换一个新的灯泡 URDF，最佳实践不是重写环境，而是：

- 保持在 `screw_lightbulb` 路线内
- 尽量复用现有 object loader、task 结构和 teacher-student 训练链
- 优先通过 `URDF + task yaml + 脚本复制` 完成迁移

只要新灯泡仍然满足当前项目对 `screw-like object` 的基本结构假设，这条路线通常是成本最低、验证最快的方案。  
真正的迁移难点通常不在算法，而在：

- URDF 物理结构是否合理
- 手物初始位姿是否能形成可学接触
- reward / termination 是否仍然适配这个新灯泡
