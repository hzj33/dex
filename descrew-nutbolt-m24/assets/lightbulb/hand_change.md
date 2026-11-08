# 新灵巧手 URDF 迁移指南

适用于当前 DexScrew / HORA 衍生项目，面向新增 hand asset（例如新的 4/5 指灵巧手），覆盖从 hand asset 接入到 task yaml、teacher/student 训练链的完整迁移路径。

## 1. 背景与适用范围

当前项目采用的是一条“最小侵入式 hand 迁移路线”：

- 不优先重写环境。
- 优先复用 `dexscrew/tasks/xhand_hora.py` 的通用旋拧任务逻辑。
- 通过新的 hand asset、新的 task yaml，以及少量 hand 专项稳定性补丁，把新手型接到现有训练链中。

当前已经验证过的两条 hand 迁移案例是：

- `RBHandHoraLightbulb`
- `Dexh13HoraLightbulb`

这条路线适用于“仍符合当前手部旋拧任务范式”的新 hand asset，例如：

- 仍然是单手操作 screw-like object
- 仍然沿用当前 `obs_dict`、reward、termination、point cloud、teacher/student 训练链
- 新手型主要只是 DOF、指尖定义、根姿态、初始关节姿态不同

这条路线不适用于以下情况：

- 控制结构和当前任务完全不同
- 任务拓扑不是“灵巧手 + 旋拧物体”
- 新手型需要全新的动作解释或奖励结构

## 2. 当前项目中已验证的 hand 迁移路径

当前仓库的新手型接入方式，不是为每个 hand 重写一套 task class，而是：

- `dexscrew/tasks/rbhand_hora.py` 直接继承 `XHandHora`
- `dexscrew/tasks/dexh13_hora.py` 直接继承 `XHandHora`

也就是说，新的 hand 主要不是通过“新环境实现”接入，而是通过 task yaml 注入 hand 差异。最关键的 hand 相关字段集中在：

- `env.asset.handAsset`
- `env.asset.fingertipBodies`
- `env.asset.handRootPos`
- `env.asset.handRootQuat`
- `env.asset.handRootRPY`
- `env.asset.handInitPose`
- `env.asset.dofLowerLimits`
- `env.asset.dofUpperLimits`
- `env.numActions`
- `env.apply_action_mask`
- `env.wrist.*`

共享逻辑主要在 `dexscrew/tasks/xhand_hora.py`。文档不要求你先读懂整个文件，但迁移时必须遵守它的 hand 接口契约。

## 3. 新 hand URDF 的放置位置与推荐目录结构

推荐把新的 hand 资产放到独立目录中，不要直接覆盖已有 hand 目录。

推荐目录结构：

```text
assets/<new_hand_name>/
  urdf/
    <new_hand>_sim.urdf
  meshes/
    ...
  config/
    ...
```

建议：

- 用稳定目录名保存 hand 家族，例如 `assets/myhand_v1/`
- 用具体 URDF 文件名区分版本，例如：
  - `<new_hand>_sim.urdf`
  - `<new_hand>_with_tips.urdf`
  - `<new_hand>_export.urdf`
- 不要直接覆盖：
  - `assets/xhand_left/`
  - `assets/RBHand_right/`
  - `assets/dexh13_hand/`

最终 task yaml 真正引用的是：

- `env.asset.handAsset`

所以最重要的是这个路径最终能被 Isaac Gym 正确加载。

## 4. 新 hand URDF 的结构要求

新 hand URDF 至少要满足以下 checklist。

### 4.1 必须满足

- DOF 数明确，且能与 `env.numActions` 一一对应
- 关节名称稳定可读，便于 `handInitPose` 按 joint name 配置
- 有可作为 fingertip 的 rigid body 名称，供 `fingertipBodies` 指定
- 根链接与整体朝向清晰，便于通过 `handRootQuat` 或 `handRootRPY` 把手对准物体
- `joint limit`、`velocity limit`、`mass/inertia`、`collision` 配置不能明显异常
- collision 结构适合接触任务，不能只保留视觉 mesh

### 4.2 必须明确知道

新 hand 不是“能加载就能训练”。

除了能被 asset loader 读进去之外，它还必须满足：

- DOF 读取正确
- 初始姿态可构成可学抓取
- 指尖接触稳定
- controller 不会因为 joint limit 或 velocity limit 异常而退化
- reward / termination 对这只手仍然成立

### 4.3 推荐项

- 非 XHand 手型尽量提供稳定的关节上下限
- 如果 URDF 中 velocity limit 缺失或写成 0，当前代码会补默认值，但这只是兜底，不是最佳实践
- 如果新手几何复杂、惯性差异大、接触不稳定，可能需要像 Dexh13 一样做 hand 专项 PhysX 稳定性参数调整

## 5. 当前 `xhand_hora.py` 对 hand asset 的硬约束

基于当前代码，hand 接入至少要满足以下真实约束。

### 5.1 DOF 数必须严格匹配

当前环境会检查：

- `hand asset DOF count == env.numActions`

如果 asset 里的 hand DOF 数和 yaml 里的 `env.numActions` 不一致，环境会直接报错。

### 5.2 非 XHand 必须显式提供 `handInitPose`

当前代码对非 XHand hand 要求更严格：

- 必须显式给 `env.asset.handInitPose`
- 如果按 dict 写法配置，必须覆盖所有 hand DOF 名称

否则会在初始化阶段失败。

### 5.3 `fingertipBodies` 必须可解析

当前代码要求：

- `env.asset.fingertipBodies` 是 rigid-body name 列表
- 列表长度必须与当前任务认为的手指数一致
- 名称必须能在 hand asset 中找到

这不是可选项。它直接影响：

- 指尖观测
- reward
- two-finger gate
- 接触相关判断

### 5.4 非 XHand 通常需要显式配置 root pose

对于非 XHand，新手型通常都要显式配置：

- `env.asset.handRootPos`
- `env.asset.handRootQuat` 或 `env.asset.handRootRPY`

其中：

- 如果 `handRootQuat` 和 `handRootRPY` 同时给出，当前代码以 `handRootQuat` 为主
- `handRootRPY` 更适合人工微调
- `handRootQuat` 更适合确定最终稳定姿态

### 5.5 自定义关节上下限是允许的

如果 URDF 自带的关节上下限不适合训练，可以通过：

- `env.asset.dofLowerLimits`
- `env.asset.dofUpperLimits`

进行覆盖。

注意：

- 这两个字段要么都不设，要么一起设
- 长度必须和 hand DOF 数一致

### 5.6 某些手型应关闭 `apply_action_mask`

当前 `apply_action_mask` 带有明显 XHand 任务假设。

因此像 Dexh13、RBHand 这样的非 XHand 手型，通常应该：

- `env.apply_action_mask: False`

否则动作掩码可能会对错误的关节生效。

### 5.7 手指 DOF 不够时可以考虑 wrist

当前代码已经提供了一个可选的：

- `env.wrist.*`

它本质上是一个“围绕旋拧轴的额外运动学 wrist yaw 辅助”。当手指本身难以驱动旋拧时，可以考虑启用。

### 5.8 重要结论

这套项目里，“能继承 `XHandHora`”不代表新手的 DOF 必须和 XHand 一样。

真正的要求是：

- 新 hand 必须满足 `xhand_hora.py` 的动态解析契约

只要：

- DOF 能解析
- fingertip 能解析
- init pose 能解析
- root pose 能对上

就有机会在不重写环境的前提下复用 `XHandHora`。

## 6. 如何新建一个 hand task yaml

不要从头手写一个新 yaml。最稳妥的路线是：

1. 先找最接近的新手型基底
2. 复制现有 task yaml
3. 在复制出的文件上局部修改

### 6.1 推荐基底

- 如果新 hand 更接近“5 指 + 可考虑 wrist 辅助”的风格，优先复制 `configs/task/RBHandHoraLightbulb.yaml`
- 如果新 hand 更接近“固定手腕、非 XHand 关节顺序、完全靠 root pose 对齐”的风格，优先复制 `configs/task/Dexh13HoraLightbulb.yaml`
- 如果新 hand 实际上就是复用 XHand 的原始结构，才考虑从 `configs/task/XHandHoraScrewDriver.yaml` 或 `configs/task/XHandHoraNutBolt.yaml` 起步

### 6.2 必须修改的字段

复制出新的 task yaml 后，至少检查并修改以下字段：

- `name`
- `env.numActions`
- `env.apply_action_mask`
- `env.asset.handAsset`
- `env.asset.fingertipBodies`
- `env.asset.handRootPos`
- `env.asset.handRootQuat`
- `env.asset.handRootRPY`
- `env.asset.handRootPosZScaleComp`
- `env.asset.handInitPose`
- `env.asset.dofLowerLimits`
- `env.asset.dofUpperLimits`
- `env.wrist.*`
- 与手型接触密切相关的 reward / termination / randomization 项

### 6.3 推荐流程

推荐按下面的顺序改：

1. 先让 `handAsset` 路径正确
2. 再让 `numActions` 对齐 URDF DOF
3. 再写完整 `handInitPose`
4. 再配置 `fingertipBodies`
5. 再调 `handRootPos` 和 `handRootQuat` / `handRootRPY`
6. 最后再调 reward、termination、randomization

## 7. hand task yaml 的规范与字段语义

这一节定义 hand yaml 中关键字段的职责，便于统一写法。

### 7.1 `env.numActions`

含义：

- 当前 hand 的动作维度

规范：

- 必须严格等于 hand asset 的 DOF 数

### 7.2 `env.asset.handAsset`

含义：

- Isaac Gym 要加载的 hand URDF 路径

规范：

- 必须指向一个可稳定加载的 URDF
- 不建议写成会和旧 hand 混淆的路径

### 7.3 `env.asset.fingertipBodies`

含义：

- 指定哪些 rigid body 作为指尖

规范：

- 必须是 rigid-body name 列表
- 顺序必须和 reward / gate 设计一致
- 例如 RBHand 当前顺序是 `index, middle, pinky, ring, thumb`
- 例如 Dexh13 当前顺序是 `index, middle, ring, thumb`

### 7.4 `env.asset.handInitPose`

含义：

- hand 初始关节姿态

规范：

- 对非 XHand，优先使用“按 joint name 的 dict”
- 必须覆盖所有 hand DOF
- 不建议只写部分 joint，再期待代码自动补齐

### 7.5 `env.asset.handRootPos`

含义：

- hand 根位置

规范：

- 用来把手整体移动到合适的抓取/旋拧起点
- 新手尺寸明显变大或变小时，经常需要和 `handRootPosZScaleComp` 一起调

### 7.6 `env.asset.handRootQuat`

含义：

- hand 根姿态的四元数表示

规范：

- 适合在姿态已经确定后做精确配置
- 如果设置了它，当前代码会优先使用它，而不是 `handRootRPY`

### 7.7 `env.asset.handRootRPY`

含义：

- hand 根姿态的欧拉角表示

规范：

- 更适合人工微调
- 不应与 `handRootQuat` 同时作为主配置来源

### 7.8 `env.asset.dofLowerLimits` / `env.asset.dofUpperLimits`

含义：

- 覆盖 hand 的关节上下限

规范：

- 一般只在 URDF 自带 limit 不适合训练时使用
- 两者应成对出现
- 长度必须匹配 DOF 数

### 7.9 `env.apply_action_mask`

含义：

- 是否启用当前环境中的动作掩码逻辑

规范：

- 非 XHand 默认优先考虑关闭
- 只有明确验证 mask 对新手型仍然正确时再打开

### 7.10 `env.wrist.*`

含义：

- 为 hand 提供一个围绕旋拧轴的额外运动学 wrist yaw 辅助

关键字段：

- `env.wrist.enable`
- `env.wrist.action_index`
- `env.wrist.yaw_scale`
- `env.wrist.limit_deg`
- `env.wrist.smooth`
- `env.wrist.pivot_mode`
- `env.wrist.write_to_obs`

规范：

- 只有当手指本体难以完成绕轴旋拧时再打开
- `action_index` 必须落在 `[0, env.numActions - 1]`

### 7.11 哪些字段是“基本必填”

对非 XHand 来说，下面这些字段基本可以认为是必填：

- `env.numActions`
- `env.asset.handAsset`
- `env.asset.fingertipBodies`
- `env.asset.handInitPose`
- `env.asset.handRootPos`
- `env.asset.handRootQuat` 或 `env.asset.handRootRPY`
- `env.apply_action_mask`

以下字段通常是“建议配置”，不是每次都必须改：

- `env.asset.handRootPosZScaleComp`
- `env.asset.dofLowerLimits`
- `env.asset.dofUpperLimits`
- `env.wrist.*`

## 8. 与新 hand 最相关的调参项

新 hand 迁移时，最常需要调的是下面这些项：

- `handRootPos`
- `handRootQuat` / `handRootRPY`
- `handInitPose`
- `fingertipBodies`
- `apply_action_mask`
- `wrist.enable`
- `wrist.action_index`
- `dofLowerLimits` / `dofUpperLimits`
- 与接触相关的 reward / gate 项

推荐调参顺序：

1. 先调根姿态
2. 再调初始关节姿态
3. 再调 `fingertipBodies` 与接触目标
4. 再决定 `apply_action_mask` / `wrist`
5. 最后调 reward 和 termination

如果新手尺寸或根链接偏移明显，`handRootPosZScaleComp` 也经常要一起调整。

## 9. 新 hand 如何适配原有 HORA 训练 task

高效迁移的关键不是新写一个 hand 环境，而是让新的 hand 继续满足现有 task 契约。

当前项目中的默认做法是：

- 新 hand 复用原有 task
- 原有 task 仍输出同样结构的 `obs_dict`
- teacher 和 student 算法都不直接感知 hand asset 本身，它们只消费 task 提供的输入张量

因此，新 hand 想真正适配原 task，至少要保证：

- 动作维度能对齐
- 指尖接触点定义合理
- 初始姿态能形成可学抓取
- reward 和 termination 对该手仍然有意义

换句话说，“高效迁移”的本质是：

- 让新 hand 满足现有 task 契约

而不是：

- 先创建一个新的 hand class 再重写所有逻辑

## 10. teacher 迁移流程

推荐按以下步骤迁移 teacher。

### 10.1 步骤

1. 准备新的 hand asset
2. 基于最接近的 yaml 复制出一个新 task yaml
3. 先做 viewer/debug sanity check
4. 再跑 PPO teacher

### 10.2 通用 teacher 命令模板

```bash
python train.py \
  task=<NewTaskName> \
  train.algo=PPO \
  headless=True \
  seed=<SEED> \
  experiment=rl \
  train.ppo.output_name=<NewTaskName>_teacher/<cache>
```

更推荐的做法不是长期手写长命令，而是：

- 先复制最接近的 teacher 脚本
- 例如参考 `scripts/rbhand_lightbulb_teacher.sh`
- 或 `scripts/dexh13_lightbulb_teacher.sh`

### 10.3 teacher 成功标准

teacher 至少要达到下面这些基本标准：

- 环境能稳定加载
- 大多数 env 不会在一开始就 reset
- 手能够建立合理接触
- 旋拧 reward 有正向增长趋势
- viewer 下姿态和接触方向看起来合理

如果这些基本条件都不满足，不要急着进入 student 阶段，先把 hand task 调稳。

## 11. student 迁移流程

当前项目已有两条 student 路线：

- `PADAPT / ProprioAdapt`
- `DOTPGStudent`

新的 hand 一般不需要因为 asset 换了就重写 student 算法，但必须确认：

- task 输出维度没有破坏 student 假设
- train / test / vis / load 约定仍然成立

### 11.1 PADAPT / ProprioAdapt

PADAPT 路线的关键点：

- `train.algo=ProprioAdapt`
- `train.ppo.proprio_adapt=True`
- student 训练依赖 teacher checkpoint
- teacher/student 的输入设定必须一致

通用命令模板：

```bash
python train.py \
  task=<NewTaskName> \
  train.algo=ProprioAdapt \
  train.ppo.proprio_adapt=True \
  checkpoint=<teacher_ckpt> \
  headless=True \
  seed=<SEED> \
  experiment=student_sim \
  train.ppo.output_name=<NewTaskName>_student_padapt/<cache>
```

脚本风格可参考：

- `scripts/dexh13_lightbulb_student_padapt.sh`

### 11.2 DOTPGStudent

DOTPG 路线的关键点：

- 区分 `teacher-state` 与 `student-state`
- 当前默认更推荐 `student-state`
- teacher checkpoint 不只是“初始化”，还会参与 expert action / latent / online supervision
- `dynamic_state=True`、expert buffer、warmup 等配置会影响 student 接管方式

通用命令模板：

```bash
python train.py \
  --config-name=config_dotpg_student.yaml \
  task=<NewTaskName> \
  train=<NewTaskTrainConfig> \
  checkpoint=<teacher_ckpt> \
  headless=True \
  seed=<SEED>
```

脚本风格可参考：

- `scripts/dexh13_lightbulb_student_dotpg_student.sh`

### 11.3 关键结论

换 hand 后，student 通常不需要“因为 hand 不同就改算法主体”，但必须检查：

- task 输出维度是否仍符合 student 输入假设
- PADAPT 的 teacher/student 输入语义是否一致
- DOTPG 的 state mode、buffer、checkpoint 约定是否仍成立

## 12. 新建 train config / script 的建议路线

如果只是“新 hand + 旧 task 结构”，推荐优先复制最接近的已有配置，而不是一开始就设计太多新分支。

### 12.1 train config

推荐路线：

- 先复制最接近的 `configs/train/*`
- 如果只是 task 参数变化，先不新建 train config，也可以直接命令行覆盖
- 确认能跑通后，再固化成独立 train config

### 12.2 scripts

推荐路线：

1. 先复制最接近的 teacher 脚本
2. 再复制 PADAPT student 脚本
3. 再复制 DOTPG student 脚本

命名建议：

- `scripts/<new_task>_teacher.sh`
- `scripts/<new_task>_student_padapt.sh`
- `scripts/<new_task>_student_dotpg_student.sh`

不建议一开始就创建很多 task 和脚本分支。更稳妥的做法是：

- 先在一个 task 上把 teacher 跑通
- 再把这条链扩展到 student

## 13. 何时只改 yaml，何时必须改共享代码

这一节很重要，因为不是所有 hand 都能只靠 yaml 接进来。

### 13.1 通常只改 yaml 就够的情况

如果下面这些条件都满足，通常只改 yaml 就够：

- DOF 数和关节名可正常读取
- `fingertipBodies` 可正常指定
- `handRootPos` 和 `handRootQuat` / `handRootRPY` 能解决接触问题
- `handInitPose` 能构造出合理的初始抓取
- 关节上下限和 controller 行为基本合理

### 13.2 往往需要改共享代码的情况

以下情况通常意味着要同步改 `xhand_hora.py` 或相关共享逻辑：

- hand collision 结构复杂，PhysX 数值不稳定
- hand 的 base pose 或动作解释需要特殊逻辑
- 手指数或 fingertip 组织方式让当前 reward / gate 假设失效
- 控制模式和现有 torque/position 控制假设不兼容

### 13.3 当前仓库中的现成例子

Dexh13 就是“主要复用 yaml，但仍然需要少量共享代码适配”的现有案例。

这类适配通常不是重写整个 task，而是加少量 hand 专项稳定性补丁，例如：

- hand 相关 PhysX 选项调整
- inertia / COM 覆盖
- 更保守的碰撞处理
- 更合适的 armature

所以要避免一个误区：

- 不是“任何新 hand 都只靠 yaml 就能接入”

更准确的说法是：

- 先尽量用 yaml 接入
- 只有在数值稳定性或结构假设不兼容时，再做小范围共享代码修补

## 14. 常见失败模式与排查顺序

建议严格按下面顺序排查，不要一上来就怀疑算法。

### 14.1 hand asset 无法加载

优先检查：

- `assets/<hand>/` 目录结构
- `env.asset.handAsset` 路径
- mesh 相对路径
- URDF 中引用的文件名

### 14.2 DOF 数与 `numActions` 不一致

优先检查：

- `env.numActions`
- hand URDF 实际 DOF 数

### 14.3 `handInitPose` 不完整

优先检查：

- `env.asset.handInitPose`
- joint name 是否与 URDF 完全一致

### 14.4 `fingertipBodies` 无法解析

优先检查：

- `env.asset.fingertipBodies`
- rigid body 名称是否真的存在于 asset 中
- fingertip 顺序是否和 reward / gate 假设一致

### 14.5 手根姿态不合理

优先检查：

- `env.asset.handRootPos`
- `env.asset.handRootQuat`
- `env.asset.handRootRPY`
- `env.asset.handRootPosZScaleComp`

### 14.6 初始关节姿态不合理

优先检查：

- `env.asset.handInitPose`

典型症状：

- 手指一开始完全张开，抓不到物体
- 手指一开始强烈穿透物体
- reset 后大面积立刻失败

### 14.7 action mask / wrist 设定错误

优先检查：

- `env.apply_action_mask`
- `env.wrist.enable`
- `env.wrist.action_index`

### 14.8 collision / inertia / velocity limit 异常

优先检查：

- URDF 中 collision 是否合理
- inertia / mass 是否极端
- velocity limit 是否缺失或异常

### 14.9 teacher 能跑但 student 不收敛

先不要立刻怀疑 hand 本体，优先检查：

- teacher 质量是否足够
- PADAPT 的输入一致性是否被破坏
- DOTPG 的 state mode / buffer / checkpoint 约定是否仍成立

## 15. 最终迁移 checklist

建议按下面顺序执行。

1. 放入新 hand asset
2. 确认 URDF 可被 Isaac Gym 正确加载
3. 复制并修改最接近的 task yaml
4. 填完整 `handInitPose`
5. 配置 `fingertipBodies`
6. 调整 `handRootPos` 和 `handRootQuat` / `handRootRPY`
7. 做 viewer/debug 检查
8. 训练 PPO teacher
9. 可视化 teacher
10. 跑 PADAPT 或 DOTPG student
11. 可视化与评测 student

## 16. 推荐的最小 hand yaml 骨架

下面给一个非 XHand 新手型的最小骨架，实际使用时建议从现有 task yaml 复制，而不是从这份骨架手写。

```yaml
name: MyHandHoraLightbulb
physics_engine: 'physx'

env:
  numEnvs: 1
  numActions: <DOF_COUNT>
  initPose: 'screwdriver_inclined'
  rotation_axis: '+z'
  apply_action_mask: False

  wrist:
    enable: False
    action_index: 0
    yaw_scale: -0.05
    limit_deg: 25.0
    smooth: True
    pivot_mode: nut_pos
    write_to_obs: True

  object:
    type: 'screw_lightbulb'
    init_pos: [0.0, 0.0, 0.0]
    init_pos_noise: [0.005, 0.005, 0.0]
    screw_upper_limit: 628.3185

  asset:
    handAsset: "assets/<new_hand_name>/urdf/<new_hand>_sim.urdf"
    fingertipBodies: ["<tip1>", "<tip2>", "<tip3>", "<tip4>"]
    handRootPos: [0.0, 0.0, 0.0]
    handRootRPY: [0.0, 0.0, 0.0]
    handRootPosZScaleComp: 0.0
    handInitPose:
      <joint_0>: 0.0
      <joint_1>: 0.0
      <joint_2>: 0.0
      <joint_3>: 0.0

  viewer:
    camera_pos: [0.35, -0.25, 0.18]
    camera_target: [0.0, 0.0, 0.06]
```

## 17. 本文与现有仓库文件的对应关系

如果后续需要回到代码核对，优先查看：

- `dexscrew/tasks/xhand_hora.py`
- `dexscrew/tasks/rbhand_hora.py`
- `dexscrew/tasks/dexh13_hora.py`
- `configs/task/RBHandHoraLightbulb.yaml`
- `configs/task/Dexh13HoraLightbulb.yaml`
- `scripts/rbhand_lightbulb_teacher.sh`
- `scripts/dexh13_lightbulb_teacher.sh`
- `scripts/dexh13_lightbulb_student_padapt.sh`
- `scripts/dexh13_lightbulb_student_dotpg_student.sh`
- `train.py`

关于 student 导出和部署，还要注意当前仓库的现实边界：

- `student_eval.py` 当前只直接支持 `ProprioAdapt`
- `scripts/convert_student_jit.sh` 目前也是按 `ProprioAdapt` 路线写的

因此，如果未来新增 hand 后还要导出新的 student policy，需要同时确认导出链是否仍兼容。
