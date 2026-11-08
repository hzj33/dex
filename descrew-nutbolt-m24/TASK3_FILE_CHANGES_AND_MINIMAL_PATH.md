# Task 3：文件新增/修改清单与最小可运行路径

> 这份文档明确指出：哪些文件必须新增、哪些必须改、哪些可以暂时不动。  
> 最后给出"Pasini 手 + 最小路径"的执行清单。

---

## Task 3.1：必须新增的文件清单

### 分析

根据 Task 2 的结论，Pasini Hand 是一个**新的硬件配置**（不同的 DOF 数、连接拓扑、控制参数）。因此需要：

1. **新的环境类** - 继承 XHandHora 或新建 XHandPasini
2. **新的任务配置** - 定义 Pasini 的参数
3. **资产文件** - URDF / 网格（如果还没有）

### 新增文件清单

| 文件路径                                          | 是否必须 | 优先级 | 原因                                          | 备注                                                 |
| ------------------------------------------------- | -------- | ------ | --------------------------------------------- | ---------------------------------------------------- |
| `dexscrew/tasks/xhand_pasini.py`                  | ✅ 是     | 🔴 高   | 定义环境类、重写针对 Pasini 的函数            | 可复用 XHandHora 70%，改 DOF 计算                    |
| `configs/task/XHandPasiniLightbulb.yaml`          | ✅ 是     | 🔴 高   | 定义任务参数（numActions=?, reward, reset等） | 必须与 xhand_pasini.py 的 DOF 一致                   |
| `configs/train/XHandPasiniLightbulb.yaml`         | ✅ 是     | 🔴 高   | 定义训练参数（learning_rate, network 等）     | 可从 XHandHoraScrewDriver.yaml 复制 + 改 output_name |
| `assets/xhand_left_pasini/xhand_left_pasini.urdf` | ✅ 是     | 🔴 高   | Pasini Hand 的 URDF                           | 描述手部结构、关节、link                             |
| `assets/xhand_left_pasini/meshes/`                | ✅ 是     | 🔴 高   | Pasini Hand 的网格文件                        | .obj 或 .stl（供 URDF 引用）                         |
| `assets/lightbulb/lightbulb.urdf`                 | ✅ 是     | 🔴 高   | 灯泡的 URDF                                   | 物体模型                                             |
| `assets/lightbulb/meshes/lightbulb.obj`           | ✅ 是     | 🔴 高   | 灯泡网格                                      | 可视化 + 物理                                        |
| `scripts/screwdriver_teacher_pasini.sh`           | ⚠️ 可选   | 🟡 中   | Pasini 专用的训练脚本                         | 为了方便，可复制并改参数                             |
| `scripts/vis_screwdriver_teacher_pasini.sh`       | ⚠️ 可选   | 🟡 中   | Pasini 专用的推理脚本                         | 为了方便                                             |

### 最小可运行版本（先跳过）

如果想快速验证"Pasini 手 + PPO 能训练"，可以**暂时不新增**：
- ✅ 灯泡资产（lightbulb.urdf / .obj）- **用一个简单的球体替代**
- ✅ 灯泡特定的奖励 - **先用原 screwdriver 的旋转奖励**

但**必须新增**：
- ❌ Pasini 的 URDF + 网格（否则无法加载手部）
- ❌ xhand_pasini.py（否则 DOF 数对不上）
- ❌ XHandPasiniLightbulb.yaml（否则参数错误）

---

## Task 3.2：必须修改的已有文件清单

### 分析

从 Task 2 反推，以下文件**必须改**，否则会报错或维度不匹配：

### 修改文件清单

| 文件                                            | 函数/字段                       | 改动原因                                       | 不改会怎样                                                | 优先级 |
| ----------------------------------------------- | ------------------------------- | ---------------------------------------------- | --------------------------------------------------------- | ------ |
| `dexscrew/tasks/__init__.py`                    | `isaacgym_task_map`             | 注册 `'XHandPasiniLightbulb': XHandPasini`     | KeyError: 'XHandPasiniLightbulb' not in isaacgym_task_map | 🔴 高   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `self.numActions`               | 改为 Pasini 的 DOF 数（如 22）                 | action 维度错误，apply_actions 崩溃                       | 🔴 高   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `self.num_xhand_hand_dofs`      | 改为 Pasini 的 DOF 数                          | 关节缓冲区大小错误                                        | 🔴 高   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `compute_observations()`        | 改 joint pos / target pos 的维度（从 12 → 22） | obs 维度错误，网络输入崩溃                                | 🔴 高   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `_setup_object_info()`          | 改物体初始位置、奖励函数（如果物体是灯泡）     | 物体位置不合理，task 无意义                               | 🟡 中   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `compute_reward()`              | 改为灯泡任务的奖励（接触奖励 vs 旋转奖励）     | 奖励函数不匹配，学不到策略                                | 🟡 中   |
| `dexscrew/tasks/xhand_pasini.py` (新建)         | `_setup_hand_default_dof_pos()` | 改初始手部姿态（Pasini 的舒适位置）            | 手部可能超出范围或奇怪姿态                                | 🟡 中   |
| `configs/task/XHandPasiniLightbulb.yaml` (新建) | `env.numActions`                | 改为 Pasini 的 DOF 数                          | numActions 与代码 self.numActions 不一致                  | 🔴 高   |
| `configs/task/XHandPasiniLightbulb.yaml` (新建) | `env.numObs`                    | 改为新的观测维度（关键！）                     | obs buffer 大小错误，RuntimeError                         | 🔴 高   |
| `configs/task/XHandPasiniLightbulb.yaml` (新建) | `env.reward.*`                  | 改为灯泡任务的奖励参数                         | 奖励定义不对                                              | 🟡 中   |
| `configs/task/XHandPasiniLightbulb.yaml` (新建) | `env.controller.*`              | 改为 Pasini 的控制参数（pgain, dgain 等）      | 控制不稳定                                                | 🟡 中   |
| `dexscrew/algo/models/models.py`                | 无需改（动态适配）              | ActorCritic 会从 env.obs_shape 读取            | 网络自动调大小                                            | ✅ 无   |
| `dexscrew/algo/ppo/ppo.py`                      | 无需改                          | 通用算法，不依赖手部结构                       | 算法无需改                                                | ✅ 无   |

### 关键维度计算

**XHand（原）:**
```
numActions: 12
obs_buf 结构:
  - joint_pos_history: 12 DOF × 3 frames = 36
  - target_pos_history: 12 DOF × 3 frames = 36
  - padding/other: 24
  总计：obs_dim = 96

priv_info_dim: ~100 (object state + fingertip info + etc.)
```

**Pasini（新，假设 22 DOF）:**
```
numActions: 22
obs_buf 结构:
  - joint_pos_history: 22 DOF × 3 frames = 66
  - target_pos_history: 22 DOF × 3 frames = 66
  - padding/other: ? (需要计算)
  总计：obs_dim = 132 + ?

priv_info_dim: ~110 (更多关节 + object state)
```

---

## Task 3.3：暂时不需要动的部分（安全区清单）

### ✅ 完全不用动的代码

| 组件                  | 文件                                       | 原因                                               | 验证命令                                              |
| --------------------- | ------------------------------------------ | -------------------------------------------------- | ----------------------------------------------------- |
| **PPO 训练器**        | `dexscrew/algo/ppo/ppo.py`                 | 通用强化学习算法，不依赖手部结构                   | `grep -n "numActions\|numObs\|xhand" ppo.py` → 无结果 |
| **ProprioAdapt 学生** | `dexscrew/algo/ppo/padapt.py`              | 通用蒸馏算法，自动适配维度                         | `grep -n "numActions\|xhand" padapt.py` → 无结果      |
| **ActorCritic 网络**  | `dexscrew/algo/models/models.py`           | 动态读取 `input_shape` / `actions_num`，自动调大小 | 网络初始化时自动计算维度                              |
| **RunningMeanStd**    | `dexscrew/algo/models/running_mean_std.py` | 通用归一化，不依赖任何具体参数                     | 直接复用                                              |
| **Hydra 配置系统**    | `configs/config.yaml`                      | 全局配置框架，无需改                               | `defaults:` 列表不变                                  |
| **主训练脚本**        | `train.py`                                 | 通用入口，不依赖手部                               | 无需改                                                |
| **VecTask 基类**      | `dexscrew/tasks/base/vec_task.py`          | 通用环境基类                                       | 无需改                                                |

### ✅ 会自动适配的网络结构

```python
# dexscrew/algo/models/models.py 中的 ActorCritic 初始化：

def __init__(self, kwargs):
    input_shape = kwargs.get('input_shape')  # ← 从 env.obs_shape 读取
    actions_num = kwargs.get('actions_num')  # ← 从 env.action_space 读取
    
    # 网络自动调大小，无需手动改
    self.actor_mlp = MLP(units=self.units, input_size=input_shape[0])  # 自动用新的 input_shape
    self.mu = torch.nn.Linear(out_size, actions_num)  # 自动用新的 actions_num
```

**验证方式：**
```bash
python train.py task=XHandPasiniLightbulb --cfg=all | grep -A5 "input_shape\|actions_num"
# 应该看到新的值（自动从 env 读取）
```

### ✅ 可以直接复用的超参

| 超参                     | 默认值              | 原因                       | 是否改   |
| ------------------------ | ------------------- | -------------------------- | -------- |
| `learning_rate`          | 5e-3                | RL 通用学习率              | ❌ 否     |
| `gamma`                  | 0.99                | 折扣因子，通用             | ❌ 否     |
| `tau`                    | 0.95                | GAE 参数，通用             | ❌ 否     |
| `entropy_coef`           | 0.0                 | 熵正则，通用               | ❌ 否     |
| `e_clip`                 | 0.2                 | PPO 裁剪范围，通用         | ❌ 否     |
| `critic_coef`            | 4                   | 价值函数权重，通用         | ❌ 否     |
| `network.mlp.units`      | [512, 256, 128]     | 网络大小，不依赖手部       | ❌ 否     |
| `network.priv_mlp.units` | [256, 128, 8]       | 特权信息 MLP，不依赖手部   | ❌ 否     |
| `ppo.horizon_length`     | 12                  | 交互步数，可能需调         | ⚠️ 看情况 |
| `ppo.minibatch_size`     | 16384               | 批大小，依赖显存和 numEnvs | ⚠️ 看情况 |
| `ppo.num_actors`         | ${task.env.numEnvs} | 自动同步                   | ✅ 自动   |

### ✅ 会自动同步的参数

```yaml
# configs/train/XHandPasiniLightbulb.yaml

ppo:
  num_actors: ${...task.env.numEnvs}  # ← 自动读取 task.env.numEnvs
  # 无需手动改，Hydra 会自动解析
```

---

## Task 3.4：Pasini 最小可运行路径（Dry Run）

### 目标

设计一个**最小化改动**的路径，使得：
1. ✅ Pasini 手 + 灯泡（或球体）能在 Isaac Gym 中加载
2. ✅ PPO 能开始训练（即使没有学到策略也没关系）
3. ✅ 可视化能运行（推理一个 episode）

### 最小路径清单

#### 🟢 第1步：新建最小化的 Pasini 环境类

**文件：** `dexscrew/tasks/xhand_pasini.py`

**策略：** 复制 `xhand_hora.py`，改以下部分：

```python
# xhand_pasini.py (约 90% 复用 xhand_hora.py)

class XHandPasini(XHandHora):  # ← 继承而不是重写全部
    def __init__(self, config, sim_device, graphics_device_id, headless):
        # 暂时先不改：物体初始化、奖励函数
        # 只改：DOF 相关
        super().__init__(config, sim_device, graphics_device_id, headless)
    
    # 重写：计算 DOF 数
    def _allocate_buffers(self):
        # ← 关键：改 self.numActions, self.num_xhand_hand_dofs
        super()._allocate_buffers()  # 先调用父类
        
        # 然后手动修正 DOF 相关
        # self.numActions = 22  (从 config 读取)
        # self.num_xhand_hand_dofs = 22
    
    # 重写：compute_observations() 中的 obs 维度
    def compute_observations(self):
        # 改 joint_pos / target_pos 的维度（12 → 22）
        # 其他保持不变
        super().compute_observations()
```

**改动最少化清单：**
- [ ] 只改 `self.numActions` 的赋值（从 config 读取）
- [ ] 只改 `compute_observations()` 中关节维度的计算
- [ ] 暂时保留原 screwdriver 的奖励函数（验证训练能运行）
- [ ] 暂时用 **球体代替灯泡**（物理更简单）

**预期代码量：** ~50 行（大部分复用父类）

---

#### 🟢 第2步：新建 Pasini 的 Task 配置

**文件：** `configs/task/XHandPasiniLightbulb.yaml`

**策略：** 复制 `XHandHoraScrewDriver.yaml`，改以下部分：

```yaml
# configs/task/XHandPasiniLightbulb.yaml

name: XHandPasiniLightbulb  # ← 关键：必须与 isaacgym_task_map 的 key 一致

env:
  numActions: 22  # ← 改为 Pasini 的 DOF（假设 22）
  numObs: 184  # ← 改为新的观测维度
              # 计算：(22 + 22) × 3 + 96 + padding = 220？
              # 需要准确计算！
  
  # 暂时保留原配置（物体、奖励等）
  # object: screwdriver  (后续再改为灯泡)
  # reward: 原配置
  # ...rest same as XHandHoraScrewDriver.yaml
```

**改动最少化清单：**
- [ ] 只改 `numActions` 和 `numObs`
- [ ] 其他复制原配置（暂时用原奖励和物体）
- [ ] 改 `name` 字段（必须）

**预期代码量：** ~150 行（大部分复用）

---

#### 🟢 第3步：新建 Pasini 的训练配置

**文件：** `configs/train/XHandPasiniLightbulb.yaml`

**策略：** 完全复制 `XHandHoraScrewDriver.yaml`，只改一行：

```yaml
# configs/train/XHandPasiniLightbulb.yaml

# 全部复制 XHandHoraScrewDriver.yaml，除了：
ppo:
  output_name: 'debug_pasini'  # ← 改输出目录名，避免覆盖原实验
  # ...rest completely same
```

**改动最少化清单：**
- [ ] 只改 `output_name`（避免覆盖原数据）
- [ ] 其他全部复制（网络、学习率等完全通用）

**预期代码量：** 0 行改动（纯复制 + 改一行）

---

#### 🟢 第4步：注册新任务

**文件：** `dexscrew/tasks/__init__.py`

**改动：**

```python
# 原代码：
from dexscrew.tasks.xhand_hora import XHandHora
isaacgym_task_map = {
    'XHandHoraScrewDriver': XHandHora,
}

# 改为：
from dexscrew.tasks.xhand_hora import XHandHora
from dexscrew.tasks.xhand_pasini import XHandPasini  # ← 新增导入

isaacgym_task_map = {
    'XHandHoraScrewDriver': XHandHora,
    'XHandPasiniLightbulb': XHandPasini,  # ← 新增注册
}
```

**改动最少化清单：**
- [ ] 新增 1 行 import
- [ ] 新增 1 行 map entry

**预期代码量：** 2 行

---

#### 🟢 第5步：新建 Pasini 的 URDF（必需）

**文件：** `assets/xhand_left_pasini/xhand_left_pasini.urdf`

**策略：** 从真实 Pasini Hand 的 URDF 复制，或用简化版本

**最小化版本：** 如果没有真实 URDF，可以用**参数化 URDF 生成脚本**（暂时跳过，用占位符）

---

#### 🟢 第6步：创建启动脚本

**文件：** `scripts/screwdriver_teacher_pasini.sh`

**内容：**

```bash
#!/bin/bash
GPUS=$1
SEED=$2
CACHE=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

CUDA_VISIBLE_DEVICES=${GPUS} \
python train.py task=XHandPasiniLightbulb headless=True seed=${SEED} \
experiment=rl \
train.algo=PPO \
task.env.reset_dist_threshold=0.1 \
wandb_activate=False \
train.ppo.output_name=XHandPasiniLightbulb_teacher/${CACHE} \
${EXTRA_ARGS}
```

**改动最少化清单：**
- [ ] 复制 `screwdriver_teacher.sh`
- [ ] 改 `task=XHandPasiniLightbulb`
- [ ] 改 `train.ppo.output_name=XHandPasiniLightbulb_teacher/...`

---

### 完整检查清单

| 步骤 | 文件                                              | 操作                            | 优先级 | 状态 |
| ---- | ------------------------------------------------- | ------------------------------- | ------ | ---- |
| 1    | `dexscrew/tasks/xhand_pasini.py`                  | 🆕 新建（继承 XHandHora）        | 🔴 高   | [ ]  |
| 2    | `configs/task/XHandPasiniLightbulb.yaml`          | 🆕 新建（复制 + 改 DOF）         | 🔴 高   | [ ]  |
| 3    | `configs/train/XHandPasiniLightbulb.yaml`         | 🆕 新建（复制 + 改 output_name） | 🔴 高   | [ ]  |
| 4    | `dexscrew/tasks/__init__.py`                      | ✏️ 改（新增 import + map）       | 🔴 高   | [ ]  |
| 5    | `assets/xhand_left_pasini/xhand_left_pasini.urdf` | 🆕 新建（或占位符）              | 🔴 高   | [ ]  |
| 6    | `scripts/screwdriver_teacher_pasini.sh`           | 🆕 新建（复制 + 改参数）         | 🟡 中   | [ ]  |

---

### 执行步骤

```bash
# 步骤 1：改 __init__.py（注册新任务）
# vim dexscrew/tasks/__init__.py

# 步骤 2：新建 xhand_pasini.py（环境类）
# cp dexscrew/tasks/xhand_hora.py dexscrew/tasks/xhand_pasini.py
# vim dexscrew/tasks/xhand_pasini.py  # 改 DOF 相关

# 步骤 3：新建配置文件
# cp configs/task/XHandHoraScrewDriver.yaml configs/task/XHandPasiniLightbulb.yaml
# vim configs/task/XHandPasiniLightbulb.yaml  # 改 numActions / numObs / name

# 步骤 4：复制训练配置
# cp configs/train/XHandHoraScrewDriver.yaml configs/train/XHandPasiniLightbulb.yaml
# vim configs/train/XHandPasiniLightbulb.yaml  # 改 output_name

# 步骤 5：创建 URDF（或占位符）
# mkdir -p assets/xhand_left_pasini/meshes
# # 放入 URDF + 网格文件（或先用原 xhand URDF 代替）

# 步骤 6：创建脚本
# cp scripts/screwdriver_teacher.sh scripts/screwdriver_teacher_pasini.sh
# vim scripts/screwdriver_teacher_pasini.sh  # 改 task + output_name

# 步骤 7：验证配置（不训练）
python train.py task=XHandPasiniLightbulb --cfg=all | head -50

# 步骤 8：尝试初始化环境（运行 1 step）
python train.py task=XHandPasiniLightbulb \
  headless=True \
  task.env.numEnvs=1 \
  train.ppo.max_agent_steps=1 \
  test=False

# 步骤 9：如果成功，尝试训练（10M steps）
./scripts/screwdriver_teacher_pasini.sh 0 42 dry_run \
  train.ppo.max_agent_steps=10000000
```

---

### 预期现象

#### ✅ 成功（预期看到）

```
Start Building the Environment  # ← 环境加载成功
Environment created with num_envs=48, action_dim=22, obs_dim=???
Episode 1 | Step 10000 | Reward: 0.x  # ← 训练开始
Episode 2 | Step 20000 | Reward: 0.y
...
```

#### ❌ 失败场景 1（缺少 URDF）

```
FileNotFoundError: assets/xhand_left_pasini/xhand_left_pasini.urdf not found
```

**解决：** 复制原 xhand URDF 或创建占位符

#### ❌ 失败场景 2（DOF 不匹配）

```
RuntimeError: Expected action shape (48, 12) but got (48, 22)
```

**解决：** 检查 `config.env.numActions` vs `xhand_pasini.py` 的 `self.numActions`

#### ❌ 失败场景 3（Obs 维度错）

```
RuntimeError: Expected obs shape (48, 96) but got (48, 220)
```

**解决：** 改 `config.task.numObs` 或调整 `compute_observations()` 的计算

#### ❌ 失败场景 4（Task 未注册）

```
KeyError: 'XHandPasiniLightbulb' not found in isaacgym_task_map
```

**解决：** 检查 `__init__.py` 是否添加了新的 map entry

---

### 验证命令集

```bash
# 1. 检查任务是否注册
python -c "from dexscrew.tasks import isaacgym_task_map; print(isaacgym_task_map.keys())"
# 应该看到 'XHandPasiniLightbulb'

# 2. 检查配置是否正确加载
python train.py task=XHandPasiniLightbulb --cfg=job | grep -E "numActions|numObs|name"

# 3. 检查网络是否正确初始化（不训练，只初始化）
python train.py task=XHandPasiniLightbulb \
  headless=True \
  task.env.numEnvs=1 \
  train.ppo.max_agent_steps=0 \
  test=False 2>&1 | head -100

# 4. 运行一个 step（最小验证）
python train.py task=XHandPasiniLightbulb \
  headless=True \
  task.env.numEnvs=1 \
  train.ppo.max_agent_steps=1 \
  test=False 2>&1 | grep -E "Episode|Step|Reward"

# 5. 运行推理测试
python train.py task=XHandPasiniLightbulb \
  headless=False \
  task.env.numEnvs=1 \
  test=True \
  checkpoint=outputs/XHandPasiniLightbulb_teacher/dry_run/stage1_nn/best_reward_*.pth
```

---

## 总结表

| 任务                  | 新增文件数 | 改动文件数 | 总改动行数 | 预计时间   |
| --------------------- | ---------- | ---------- | ---------- | ---------- |
| Task 3.1 - 文件清单   | -          | -          | -          | 5 min      |
| Task 3.2 - 修改清单   | -          | -          | -          | 5 min      |
| Task 3.3 - 安全区清单 | -          | -          | -          | 5 min      |
| Task 3.4 - 最小路径   | 5          | 1          | ~200       | **30 min** |

---

## 🚨 Task 3 的三个重要风险点（Task 4 必须解决）

### ⚠️ 问题 1：numObs "先猜一个" 风险太高

**当前风险：**
```
config.task.numObs: 184  # ← 这是估算，不是精确值！
# 实际运行时会报：
RuntimeError: Expected obs shape (48, 184) but got (48, 220)
# 然后你会花 2 小时追踪"到底哪里多了 36 维"
```

**Task 4 必须做：** 从代码精确反推 obs 结构

```python
# dexscrew/tasks/xhand_hora.py 的 compute_observations()

# Step 1: 找出所有 obs 构成部分
t_buf = self.obs_buf_lag_history[:, -3:, :self.obs_buf.shape[1]//3]  # 最后 3 帧
#        ↑ 历史缓冲的后 3 帧，维度 = (N, 3, obs_buf.shape[1]//3)

cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # 当前关节位置
#             ↑ self.xhand_hand_dof_pos，维度 = (N, 1, numActions)

cur_tar_buf = self.cur_targets[:, None, :self.num_actions]  # 当前目标位置
#             ↑ 维度 = (N, 1, numActions)

cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # 拼接
#             ↑ 维度 = (N, 1, 2*numActions)

self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
#                              ↑ 总历史 = (N, 30, 2*numActions) for proprio_adapt mode

self.obs_buf[:, :t_buf.shape[1]] = t_buf.reshape(...)  # 放入 obs_buf
#             ↑ obs_buf 的前几维是历史的后 3 帧

# Step 2: 加入其他信息
if self.use_point_cloud_info:
    point_cloud = obs_dict['point_cloud_info']  # (N, 100, 3) = 300
    self.obs_buf = torch.cat([self.obs_buf, point_cloud.reshape(N, -1)], dim=-1)
```

**精确 obs 公式（需要从代码逆向）：**

| 字段                               | 维度        | 计算                                                | 备注           |
| ---------------------------------- | ----------- | --------------------------------------------------- | -------------- |
| joint_pos_history (last 3 frames)  | 12×3 = 36   | `obs_buf_lag_history[-3:, :numActions]`             | XHand (12 DOF) |
| target_pos_history (last 3 frames) | 12×3 = 36   | `obs_buf_lag_history[-3:, numActions:2*numActions]` | XHand (12 DOF) |
| padding/other                      | 24          | ?                                                   | 需要确认       |
| point_cloud                        | 100×3 = 300 | 点云采样                                            | 可选           |
| **总计**                           | **96**      | 36+36+24 = 96（可能还有 priv_info padding）         | XHand 已验证   |

**Task 4 行动：**
- [ ] 在 `compute_observations()` 中找出 obs_buf 的每一部分
- [ ] 逆推公式：`obs_dim = history_frame * 2 * numActions + padding + point_cloud_dim`
- [ ] 对 XHand 验证：`obs_dim = 3 * 2 * 12 + 24 + 300 = 96` ✓
- [ ] 对 Pasini 计算：`obs_dim = 3 * 2 * 22 + ? + 300 = 432 + ?`

**精确结果应该输出：**
```yaml
# configs/task/XHandPasiniLightbulb.yaml
env:
  numActions: 22  # ← 已知
  numObs: 468  # ← 精确计算而非估算
  # 计算过程说明：
  # - proprio history (last 3 frames, pos+target): 3 * 2 * 22 = 132
  # - padding: 24
  # - point_cloud: 100 * 3 = 300
  # - total: 132 + 24 + 300 = 456  (or 468 if priv padding)
```

---

### ⚠️ 问题 2：继承时 DOF 初始化的时序问题

**当前风险：**
```python
# 错误做法（继承 XHandHora）
class XHandPasini(XHandHora):
    def __init__(self, config, ...):
        super().__init__(config, ...)  # ← 此时 self.numActions = 12（旧值）
        
        # 然后在 super() 后改：
        self.numActions = 22  # ← 太晚了！父类已经用旧值初始化了
        
        # 结果：
        # - self.xhand_hand_dof_pos 的 shape 已经是 (N, 12)
        # - self.hand_asset 的 DOF 数已经锁定
        # - gym tensor 已经按旧维度建好
        # → 改了这个变量但下面一堆地方没跟上
```

**Task 4 必须做：** 精确找出 DOF 的初始化关键路径

```python
# dexscrew/tasks/xhand_hora.py 中的初始化顺序：

def __init__(self, config, sim_device, ...):
    # Line 35-36: DOF 第一次确定
    self.numActions = config['env']['numActions']  # ← 关键：这里第一次读取
    
    # Line 37: 根据 numActions 创建 default pos buffer
    self.xhand_hand_default_dof_pos = torch.zeros(
        self.num_xhand_hand_dofs, ...  # ← 需要知道 num_xhand_hand_dofs 的来源
    )
    
    # Line 58+: 调用 super().__init__()，这里会：
    super().__init__(config, sim_device, ...)
    #     ↓
    #  VecTask.__init__() 中:
    #    - 加载 URDF：self.hand_asset = load_asset(hand_urdf)
    #      ↑ URDF 里的关节数必须 == self.numActions
    #    - 创建 gym actors
    #    - 获取 DOF state tensor
    
    # Line 75+: 创建 dof state wrapper
    self.xhand_hand_dof_state = self.dof_state.view(...)[:, :self.num_xhand_hand_dofs]
    #                                                    ↑ 这个数字必须精确
```

**关键问题：**
- `self.num_xhand_hand_dofs` 是从哪里来的？是常数（12）还是动态的？
- URDF 的加载是在哪一行？
- gym tensor 的创建是在哪一行？

**Task 4 行动：**
- [ ] 搜索 `self.num_xhand_hand_dofs` 的定义
  ```bash
  grep -n "num_xhand_hand_dofs" dexscrew/tasks/xhand_hora.py
  ```
- [ ] 搜索 URDF 加载的位置
  ```bash
  grep -n "hand_asset\|load.*urdf" dexscrew/tasks/xhand_hora.py
  ```
- [ ] 搜索 gym tensor 创建的位置
  ```bash
  grep -n "dof_state\|acquire_dof_state" dexscrew/tasks/xhand_hora.py
  ```

**两种解决方案对比：**

| 方案                        | 优点             | 缺点                                  | 推荐度               |
| --------------------------- | ---------------- | ------------------------------------- | -------------------- |
| **继承 + super() 前改 DOF** | 代码复用         | 需要确保所有初始化都在 super() 前完成 | 🟡 中                 |
| **不继承，复制全部代码**    | 时序清晰，不踩坑 | 代码重复，后期难维护                  | 🟢 高（用于 dry run） |

**推荐做法（Task 4 采用）：**
```python
# 方案 A：复制 xhand_hora.py → xhand_pasini.py
# 改的部分：
#   1. class 名改为 XHandPasini
#   2. self.numActions = 22  (从 config 读)
#   3. self.num_xhand_hand_dofs = 22  (如果是常数，改这里)
#   4. URDF 路径改为 pasini 的 URDF
#   5. 其他保持不变

# 优点：确保初始化顺序正确，不会遗漏
# 后期再抽象继承（等代码稳定后）
```

---

### ⚠️ 问题 3：资产替代"先用球体"的具体位置不明

**当前风险：**
```
你说"先用球体替代灯泡"，但没说：
- 球体在哪个函数里创建？
- reward 里哪个变量表示"旋转角/旋转速度"？
- reset 逻辑是否依赖螺钉的形状？
```

**Task 4 必须做：** 明确 object 加载、reward、reset 的最小 stub

```python
# dexscrew/tasks/xhand_hora.py 中的三个关键部分：

# Part 1: Object 加载（通常在 _setup_object_info() 或 __init__）
def _setup_object_info(self, config):
    self.object_asset = self.gym.load_asset(
        self.sim, 
        asset_root="assets",
        filename="screwdriver/0000_stripe.urdf"  # ← 这里改为球体
        # filename="sphere/sphere.urdf"  # ← 替代：简单球体
    )

# Part 2: Reward 计算（通常在 compute_reward()）
def compute_reward(self):
    # 原逻辑（针对旋转）：
    self.nut_dof_vel = ...  # 螺钉的旋转速度
    reward = -torch.abs(self.nut_dof_vel - target_vel)  # 奖励旋转
    
    # Stub 逻辑（仅为验证）：
    reward = torch.ones(self.num_envs) * 0.1  # 常数奖励，确保不会 NaN
    
# Part 3: Reset 逻辑（通常在 reset_idx()）
def reset_idx(self, env_ids):
    # 原逻辑（依赖螺钉的旋转状态）：
    if self.nut_dof_pos[env_id] > threshold:
        reset = True
    
    # Stub 逻辑（任何物体都适用）：
    if self.object_pos[env_id, 2] < -0.5:  # 物体掉了
        reset = True
```

**Task 4 行动：**
- [ ] 找出 object URDF 加载的位置（通常在 `_setup_object_info()` 或构造函数）
  ```bash
  grep -n "gym.load_asset\|\.urdf" dexscrew/tasks/xhand_hora.py | grep -v hand
  ```
- [ ] 找出 reward 计算中的关键变量（`nut_dof_vel`, `nut_dof_pos` 等）
  ```bash
  grep -n "nut_dof\|object_.*vel\|object_.*pos" dexscrew/tasks/xhand_hora.py
  ```
- [ ] 找出 reset 逻辑中的物体相关条件
  ```bash
  grep -n "reset.*nut\|reset.*object" dexscrew/tasks/xhand_hora.py
  ```

**最小 stub 模板：**

```python
# dexscrew/tasks/xhand_pasini.py (继承 XHandHora)

class XHandPasini(XHandHora):
    
    def _setup_object_info(self, config):
        """改为加载球体而不是灯泡"""
        # 暂时不用灯泡 URDF，用球体
        self.object_asset = self.gym.load_asset(
            self.sim,
            asset_root="assets",
            filename="sphere/unit_sphere.urdf"  # ← 简单球体，大多数 Isaac Gym 都有
        )
        # 其他保持不变（object mass, friction 等）
    
    def compute_reward(self):
        """暂时返回常数奖励，确保不 NaN"""
        # 先不实现灯泡特定的奖励
        # 只要 reward 是有效数字就行
        reward = torch.ones(self.num_envs, device=self.device) * 0.1
        return reward, {}  # 如果原函数返回 dict
    
    def reset_idx(self, env_ids):
        """暂时用简单的物理重置条件"""
        # 先不依赖螺钉的旋转状态
        # 只要物体没有飞出去就不重置
        super().reset_idx(env_ids)
        # 自定义 reset 条件：如果物体掉了就重置
        if len(env_ids) > 0:
            bad_env_ids = self.object_pos[env_ids, 2] < -1.0  # 掉出底部
            if bad_env_ids.any():
                self.reset_idx(env_ids[bad_env_ids])
```

**验证 stub 是否正确的命令：**
```bash
# 运行一个 step，看 reward 是否有效（不 NaN，不 inf）
python train.py task=XHandPasiniLightbulb \
  headless=True \
  task.env.numEnvs=1 \
  train.ppo.max_agent_steps=1 \
  test=False 2>&1 | grep -E "reward|nan|inf"
# 应该看到数字，而不是 nan
```

---

## 🎯 Task 4 的精确目标

基于以上三个风险点，**Task 4 必须输出：**

### Task 4.1：精确的 obs 维度公式

**输出格式：**
```
XHand obs 维度（验证）:
  - joint_pos_history (last 3 frames): 12 × 3 = 36
  - target_pos_history (last 3 frames): 12 × 3 = 36
  - padding: 24
  - point_cloud: 100 × 3 = 300
  - TOTAL: 96 ✓ (matches expected)

Pasini obs 维度（推算）:
  - joint_pos_history (last 3 frames): 22 × 3 = 66
  - target_pos_history (last 3 frames): 22 × 3 = 66
  - padding: ? (same as XHand, assume 24)
  - point_cloud: 100 × 3 = 300
  - TOTAL: 456 (需要在代码中验证 padding 是否固定)
```

### Task 4.2：DOF 初始化的关键路径

**输出格式：**
```
DOF 初始化关键点：
  - Line 35-36: self.numActions = config['env']['numActions']  ← 在这里确定
  - Line 45-48: self.xhand_hand_default_dof_pos = torch.zeros(self.num_xhand_hand_dofs, ...)
  - Line 58: super().__init__() → VecTask 会加载 URDF、创建 gym tensor
  - Line 75: self.xhand_hand_dof_state = self.dof_state[..., :self.num_xhand_hand_dofs]
  
URDF 加载位置：
  - Line X: self.hand_asset = self.gym.load_asset(..., hand_urdf)
  - URDF 中的 DOF 数必须 == self.numActions
  
解决方案：复制 xhand_hora.py → xhand_pasini.py，改 config 的 hand_urdf path
```

### Task 4.3：Object / Reward / Reset 的依赖关系

**输出格式：**
```
Object 加载（第 XYZ 行）:
  - self.object_asset = gym.load_asset(..., "screwdriver/0000.urdf")
  - Stub: 改为 "sphere/unit_sphere.urdf"

Reward 关键变量（第 ABC 行）:
  - self.nut_dof_vel：螺钉旋转速度 → 对球体无意义
  - 改为：reward = 0.1（常数，仅验证 not NaN）

Reset 逻辑（第 DEF 行）:
  - self.nut_dof_pos > threshold：螺钉旋转角超限 → 对球体无意义
  - 改为：self.object_pos[..., 2] < -1.0：物体掉了就重置
```

**最后，汇总生成：**
- ✅ 精确的 obs 维度值
- ✅ xhand_pasini.py 的改动清单（第 X 行改什么）
- ✅ 球体 stub reward 和 reset 的代码
- ✅ 验证命令（确保 step 1 不崩溃）

---

**下一步：** 等待 Task 4 的完整执行。现在你有了清晰的三个目标，Task 4 应该输出精确的数值而不是估算。



