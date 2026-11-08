# dexh13_right_description2 修改记录（参考 MuJoCo XML/URDF）

## 对比范围
- 原始手模：`assets/dexh13_right_description/urdf/dexh13_right.urdf`
- 参考文件夹：`assets/dexh13_right_description2/urdf/`
- 重点关注：`dexh13_right*.xml`（MuJoCo）与 `dexh13_right1.urdf`

## URDF 层面改动
- `assets/dexh13_right_description2/urdf/dexh13_right.urdf` 与原始 URDF 内容一致。
- `assets/dexh13_right_description2/urdf/dexh13_right1.urdf` 仅将所有 `mesh` 路径从 `package://dexh13_right_description/...` 改为相对路径 `../meshes/...`，便于 MuJoCo/本地加载。
- 未发现基于 URDF 的 collision 几何缩放或替换（collision 仍使用原 STL）。

## MuJoCo XML 结构性变化（与 URDF 不同）
- 固定关节的触觉片段（`*_tactile_link_*`）在 XML 中不再作为独立 body，而是作为同一 body 的额外 `geom` 挂载；对应的质量/惯量并入父 link。
- `dexh13_right.xml` 里 `right_palm` 是 world 级 `geom`，手指根部 body 直接挂在 world；而 `dexh13_right1/2/3.xml` 与 `dexh13_right_fixed.xml` 使用 `right_palm` 作为根 body，把手指挂在 palm 下面。

## 碰撞/接触相关改动（重点）
### 全局接触参数（所有 XML 变体）
- `dexh13_right1.xml` 使用 `geom solimp="0.95 0.99 0.001 0.5 2" solref="0.002 1" friction="1.0 0.005 0.0001" condim="3"`。
- `dexh13_right2.xml`、`dexh13_right3.xml`、`dexh13_right_fixed.xml` 使用更“软”的 `solimp="0.5 0.99 0.0001" solref="0.005 1"`，其余摩擦与 `condim` 相同。

### 显式自碰撞屏蔽（contact exclude）
- `dexh13_right2.xml`：屏蔽相邻手指的近端关节（Link 1）之间的碰撞。
  - `right_index_link_1` × `right_middle_link_1`
  - `right_index_link_1` × `right_ring_link_1`
  - `right_middle_link_1` × `right_ring_link_1`
- `dexh13_right3.xml` 与 `dexh13_right_fixed.xml`：屏蔽 palm 与各手指的 `link_0` / `link_1` 碰撞。
  - `right_palm` × `right_index_link_0` / `right_index_link_1`
  - `right_palm` × `right_middle_link_0` / `right_middle_link_1`
  - `right_palm` × `right_ring_link_0` / `right_ring_link_1`
  - `right_palm` × `right_thumb_link_0` / `right_thumb_link_1`

## 物理参数与驱动方式差异
- `dexh13_right1.xml` 的 `default` 关节参数偏大（`armature`/`damping` 值更高），且使用 `position` 执行器（带 `kp`）。
- `dexh13_right2/3.xml` 与 `dexh13_right_fixed.xml` 使用较小的 `armature`，并加入 `frictionloss`、`stiffness`、`springref`；执行器改为 `motor`（直接力矩/电机控制）。
- 所有 `dexh13_right1/2/3.xml` 与 `dexh13_right_fixed.xml` 都添加了 `tendon`，把每个手指的 `joint_2` 与 `joint_3` 进行耦合（`right_*_J0`）。

## 环境/固定方式
- `dexh13_right3.xml` 与 `dexh13_right_fixed.xml` 显式设置 `gravity="0 0 -9.81"`。
- `dexh13_right_fixed.xml` 额外加入：
  - 地面平面 `ground`（带 `condim="3"`）。
  - `right_palm` 使用 `mocap="true"`，用于固定/外部控制手掌位姿。
  - 一盏 `light`。

## 已移植到当前训练 URDF（`dexh13_right_fix_path.urdf`）
### 已做的碰撞改动
- 将所有 `*_tactile_link_*` 的 **collision** 从子 link 挪到父 link（按固定关节的 `origin` 添加），并移除触觉 link 自身的 collision。
  - 作用：等价于 MuJoCo XML 中“同一 body 内多 geom”的处理，减少固定关节之间的自碰撞，但保留触觉贴片的对外接触形状。
- 对 `right_index_link_1` / `right_middle_link_1` / `right_ring_link_1` 的 collision mesh 增加轻微缩放 `scale="0.98 0.98 0.98"`。
  - 作用：近似减少相邻手指近端 Link 1 的互相穿插，尽量降低改动幅度以避免影响抓取真实性。

### 保留/未做的项（原因）
- **显式 contact exclude**（指定 link 对不碰撞）在标准 URDF 中无法表达。
  - 若需要 1:1 对齐 MuJoCo XML 的排除规则，需要在仿真层做 collision filtering 或使用支持 contact exclude 的格式。
- `right_palm_link` 的 collision 目前已在此 URDF 中缩放到 `0.1`，本质上等效于 XML 中“palm vs link_0/1”排除思路；因此本次未再额外处理。

## 小结（与“简单缩放 collision”不同的处理思路）
- 参考模型没有直接缩放/替换 collision mesh，而是通过 **contact exclude** 关闭特定自碰撞对、并调整接触参数（`solimp/solref`）来缓解自碰撞问题。
- 另外通过 **固定关节合并**、**关节阻尼/摩擦设置**、**手指耦合 tendon** 与 **执行器模式** 改善整体动力学稳定性。

## Pasini Screwdriver Teacher vs 原生 Screwdriver Teacher（奖励与碰撞）
### 对比范围
- 训练脚本：`scripts/pasini_screwdriver_teacher.sh`、`scripts/screwdriver_teacher.sh`
- 任务配置：`configs/task/XHandPasiniScrewDriver.yaml`、`configs/task/XHandHoraScrewDriver.yaml`
- 奖励实现：`dexscrew/tasks/xhand_pasini.py`、`dexscrew/tasks/xhand_hora.py`

### 奖励项与计算方式
- 两个任务的奖励公式完全一致：同一套项（旋转奖励、姿态差惩罚、扭矩惩罚、功惩罚、点云 Z 距离惩罚、旋转惩罚、接近奖励），相同的 `compute_hand_reward` 组合方式。
- 差异来自 **scale 参数** 和 **接近阈值**（仅影响 proximity reward）。

### 你在 Pasini 里做的奖励微调（相对原生 Hora）
（历史记录）早期为了绕开 Pasini 自碰撞导致的训练不稳定，确实做过一些 reward scale 的加大/减小（例如提高 rotate / proximity、降低 pose penalty 等）。

（当前状态）目前 Pasini Screwdriver 的主 reward scale 已回到与 Hora 一致：
- `rotate_reward_scale`: 2.5
- `pose_diff_penalty_scale`: -0.1
- `rotate_penalty_scale`: -0.3
- `proximity_reward_scale`: 2.0

为改善“点戳/抽搐式”旋转策略，目前额外做了两类**可控的**微调（尽量不破坏 Hora 语义）：
- `work_penalty_scale`: **-0.005**（Hora 默认为 -0.01）
  - 动机：降低持续接触旋转时的能量惩罚动机，减少策略倾向于“短时碰一下就走”的行为。
- 新增可选平滑项（默认关闭）：
  - `action_rate_penalty_scale`: 0.0（可试 `-0.001 ~ -0.01`）
  - `env.controller.target_smoothing_alpha`: 0.0（可试 `0.1 ~ 0.3`）

### 碰撞检测是否一致（除奖励外）
- 物理引擎侧的**接触检测配置一致**：两者都使用 PhysX GPU contact、相同的 `contact_offset`/`contact_collection` 等仿真参数。
- **实际碰撞几何不一致**：Pasini 使用 `assets/dexh13_right_description/urdf/dexh13_right_fix_path.urdf`（含你为自碰撞做的碰撞几何调整），原生 Hora 使用 `assets/xhand_left/urdf/xhand_left.urdf`。因此接触对、接触法向/力的分布会不同。
- Pasini 手模里 `right_palm_link` 等 collision 的缩放/合并会直接改变自碰撞与碰撞检测结果，这一点与原生模型并不一致。

### 若要回到“原项目奖励”对齐
- 将 Pasini 的奖励 scale 改回 Hora 的值（上面 4 项）。
- 让 proximity 的阈值逻辑与 Hora 一致：删除/忽略 `proximity_dist_threshold`，直接用 `reset_dist_threshold`。

## 已执行：Pasini 奖励对齐 Hora（Screwdriver Teacher）
- `configs/task/XHandPasiniScrewDriver.yaml` 已更新为 Hora 版本的奖励 scale：
  - `rotate_reward_scale`: 2.5
  - `pose_diff_penalty_scale`: -0.1
  - `rotate_penalty_scale`: -0.3
  - `proximity_reward_scale`: 2.0
- `reset_dist_threshold` 已对齐 Hora teacher 脚本为 `0.1`。
- `proximity_dist_threshold` 设为 `0.1`，用于让 proximity reward 的归一化尺度与终止阈值一致（更直观）。
