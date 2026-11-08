# Copilot Task 2：核心文件深度分析（必须做）

> 这份文档列出**下一阶段要分析的核心文件**及其关键问题。  
> 每个文件都是移植 Pasini Hand 和改物体（灯泡）时的**必读**。

---

## 📋 分析清单（按优先级）

### Task 2.1：Task 注册与映射机制

**文件：** [dexscrew/tasks/__init__.py](dexscrew/tasks/__init__.py)

**目标：** 确认 isaacgym_task_map 如何工作、新任务如何注册

**要分析的问题：**

1. `isaacgym_task_map` 字典的 key-value 结构是什么？
   - key 应该是什么？（来自 task.name）
   - value 应该是什么？（环境类，如 XHandHora）
   - 现在注册了哪些任务？

2. 新增 Pasini 任务的方式？
   - 是复用 `XHandHora` 类（写子类或参数化）？
   - 还是新建 `XHandPasini` 类？

3. 导入语句如何工作？
   ```python
   from dexscrew.tasks.xhand_hora import XHandHora
   ```
   如果新建 `xhand_pasini.py`，如何导入？

**预期输出：**

一个"注册清单"表格：
| 任务名（task.name）  | 类名        | 文件            | 备注         |
| -------------------- | ----------- | --------------- | ------------ |
| XHandHoraScrewDriver | XHandHora   | xhand_hora.py   | 当前         |
| XHandPasiniLightbulb | XHandPasini | xhand_pasini.py | 新增（待建） |

---

### Task 2.2：观测空间的详细结构

**文件：** [dexscrew/tasks/xhand_hora.py](dexscrew/tasks/xhand_hora.py)

**目标：** 列出 obs_dict 的所有字段来源，以及如何计算

**要分析的问题：**

1. `compute_observations()` 函数输出什么？

   环境每一步返回的是：
   ```python
   obs_dict = {
       'obs': [...],              # 本体感受觉（关键！）
       'priv_info': [...],        # 特权信息
       'point_cloud_info': [...], # 点云
       'proprio_hist': [...]      # 历史本体
   }
   ```
   
   请详细列出各字段的**来源、维度、含义**：

   | 字段               | 来源                             | 维度  | 含义                  | 移植时需改吗           |
   | ------------------ | -------------------------------- | ----- | --------------------- | ---------------------- |
   | obs[joint pos]     | `self.xhand_hand_dof_pos`        | 12    | 手部12个关节位置      | ✅ 改为 Pasini 自由度   |
   | obs[target pos]    | `self.cur_targets`               | 12    | 目标关节位置          | ✅ 改为 Pasini 自由度   |
   | proprio_hist       | `obs_buf_lag_history[-30:, :24]` | 30×24 | 过去30帧×(pos+target) | ✅ 改为维度变化         |
   | priv_info[obj_pos] | `self.object_pos`                | 3     | 物体位置              | ❌ 无需改               |
   | priv_info[obj_rot] | `self.object_rot`                | 4     | 物体四元数旋转        | ❌ 无需改               |
   | point_cloud_info   | 采样点云                         | 100×3 | 物体表面点云          | ✅ 如果物体变了可能需改 |

2. 动作 (action) 如何解释？

   ```python
   def apply_actions(self, actions):
       # actions 是什么？
       # - 绝对目标位置？
       # - 相对位移？
       # - 速度命令？
       # - 力矩命令？
   ```

   移植时需要知道：Pasini 的动作空间是什么（位置控制？力控？）

3. 关键缓冲区的初始化维度：

   找出以下几行，看它们的形状：
   ```python
   self.obs_buf = ...  # shape?
   self.obs_buf_lag_history = ...  # shape?
   self.proprio_hist_buf = ...  # shape?
   self.priv_info_buf = ...  # shape?
   ```

**预期输出：**

一个"观测空间图"：
```
obs_dict（环境返回）
├── obs: [96]
│   ├── joint pos history (last 3 frames): 12×3 = 36
│   ├── target pos history (last 3 frames): 12×3 = 36
│   └── padding: 24
├── priv_info: [120]
│   ├── object_position: 3
│   ├── object_rotation: 4
│   ├── object_linvel: 3
│   ├── object_angvel: 3
│   └── ... (其他特权信息)
├── point_cloud_info: [100, 3] = 300
└── proprio_hist: [30, 24]  # 历史缓冲（最后维度 = pos + target）
```

---

### Task 2.3：Task 配置参数分组

**文件：** [configs/task/XHandHoraScrewDriver.yaml](configs/task/XHandHoraScrewDriver.yaml)

**目标：** 把所有参数分为"换手必改""换物体必改""可复用"三类

**要分析的问题：**

1. **手部相关参数**（Pasini 手时必改）：
   ```yaml
   env.numActions: 12  # ← 必改（Pasini 可能 22）
   env.controller.*  # pgain, dgain, action_scale 等
   ```

2. **物体相关参数**（灯泡任务时必改）：
   ```yaml
   env.rotation_axis: '+z'  # ← 灯泡可能不需要旋转？
   env.reward.*  # 奖励函数（目前针对旋转）
   ```

3. **物理参数**（可能复用）：
   ```yaml
   env.episodeLength: 800  # 可能保持
   env.initPose: 'screwdriver_inclined'  # 需要新增 init pose 配置
   ```

4. **重置条件**（换物体需改）：
   ```yaml
   env.reset_dist_threshold: 0.05  # ← 灯泡的重置阈值不同
   ```

**预期输出：**

一个参数分类表：

| 参数                   | 默认值 | 类别         | 原因                   | 改到什么                |
| ---------------------- | ------ | ------------ | ---------------------- | ----------------------- |
| `numActions`           | 12     | 🔴 换手必改   | 手部自由度不同         | Pasini DOF              |
| `numObs`               | N/A    | 🔴 换手必改   | 观测空间维度变         | 重算                    |
| `controller.pgain`     | 3      | 🟡 可能需改   | 控制参数（控制器强度） | Pasini 的控制参数       |
| `rotation_axis`        | '+z'   | 🟡 换物体可改 | 如果灯泡不旋转，可删除 | 视灯泡模型              |
| `reward.*`             | 各种   | 🔴 换物体必改 | 奖励函数针对旋转       | 改为接触/导向灯泡的奖励 |
| `reset_dist_threshold` | 0.05   | 🟡 换物体可改 | 重置条件               | 灯泡的合理阈值          |
| `episodeLength`        | 800    | 🟢 可复用     | 任务长度               | 保持不变                |
| `clipObservations`     | 5.0    | 🟢 可复用     | 观测裁剪范围           | 保持不变                |

---

### Task 2.4：训练超参的依赖关系

**文件：** [configs/train/XHandHoraScrewDriver.yaml](configs/train/XHandHoraScrewDriver.yaml)

**目标：** 区分哪些超参和形态学 (morphology) 无关、哪些依赖 action/obs 维度

**要分析的问题：**

1. **Morphology-independent（无需改）：**
   ```yaml
   learning_rate: 5e-3  # ← 学习率，一般通用
   gamma: 0.99  # ← 折扣因子，通用
   tau: 0.95  # ← GAE 参数，通用
   ```

2. **Morphology-dependent（需要同步改）：**
   ```yaml
   network:
     mlp:
       units: [512, 256, 128]  # ← 网络大小（通常不改）
     priv_mlp:
       units: [256, 128, 8]  # ← 特权信息 MLP（通常不改）
   ```
   
   这些不需要改，因为 ActorCritic 会从 env 的 obs_shape / priv_info_dim 动态读取。

3. **Action/Obs 相关（需要校验）：**
   ```yaml
   ppo.num_actors: ${...task.env.numEnvs}  # ← 自动从 task 读取
   ppo.horizon_length: 12  # ← 交互步数，可能需要调
   ppo.minibatch_size: 16384  # ← 批大小（依赖显存）
   ```

**预期输出：**

一个超参分类表：

| 超参                | 默认值              | 类别 | 是否需改   | 原因                         |
| ------------------- | ------------------- | ---- | ---------- | ---------------------------- |
| `learning_rate`     | 5e-3                | 通用 | ❌ 否       | 学习率和手无关               |
| `gamma`             | 0.99                | 通用 | ❌ 否       | 折扣因子通用                 |
| `num_actors`        | ${task.env.numEnvs} | 自动 | ✅ 自动同步 | 来自 task config             |
| `horizon_length`    | 12                  | 可调 | ✅ 可能     | 动作频率相关                 |
| `minibatch_size`    | 16384               | 显存 | ✅ 可能     | 根据 numEnvs 调              |
| `network.mlp.units` | [512,256,128]       | 通用 | ❌ 否       | ActorCritic 动态读 obs_shape |

---

### Task 2.5：Checkpoint 保存与加载规则

**文件：** [dexscrew/algo/ppo/ppo.py](dexscrew/algo/ppo/ppo.py) 和 [dexscrew/algo/ppo/padapt.py](dexscrew/algo/ppo/padapt.py)

**目标：** 确认 restore_test() 期待的路径、保存频率、文件名规则

**要分析的问题：**

1. **PPO restore_test() 期待什么？**

   找到 [ppo.py#L269-L290](dexscrew/algo/ppo/ppo.py#L269)：
   ```python
   def restore_test(self, fn):
       if not fn:
           return
       checkpoint = torch.load(fn)
       # 加载什么内容？
       self.model.load_state_dict(...)
       self.running_mean_std.load_state_dict(...)
       if self.normalize_priv:
           self.priv_mean_std.load_state_dict(...)
       # ... 等等
   ```

   问题：
   - `fn` 期望是什么格式？绝对路径还是相对路径？
   - 如果 `fn` 包含通配符（如 `best_reward_*.pth`），如何处理？
   - 如果 `fn` 为空，程序会怎样？

2. **PPO 的保存频率与文件名规则？**

   找到 [ppo.py#L235-L251](dexscrew/algo/ppo/ppo.py#L235)：
   ```python
   # 保存频率
   if self.save_freq > 0:
       if (self.epoch_num % self.save_freq == 0) and ...
           self.save(...)  # 什么频率？
   
   # 文件名规则
   checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'
   self.save(os.path.join(self.nn_dir, checkpoint_name))
   ```

   问题：
   - save_freq 默认值是多少？（来自 train.ppo.save_frequency）
   - 保存目录是 `stage1_nn/`（来自 self.nn_dir）
   - 文件名包含 reward，所以同一次训练可能有多个文件

3. **ProprioAdapt 的保存规则（学生）与 PPO 有什么不同？**

   找到 [padapt.py#L240-L253](dexscrew/algo/ppo/padapt.py#L240)：
   ```python
   def save(self, name):
       weights = {
           'model': ...,
           'running_mean_std': ...,
           'sa_mean_std': ...,  # ← 学生特有
           ...
       }
       torch.save(weights, f'{name}.ckpt')  # ← .ckpt 而不是 .pth
   ```

   问题：
   - 学生保存为 `.ckpt`（checkpoint）而不是 `.pth`
   - 学生保存目录是 `stage2_nn/`
   - 学生的保存逻辑是什么？

4. **vis 脚本期望的默认 checkpoint 路径是什么？**

   在各个 `vis_*.sh` 脚本中搜索 `checkpoint=` 或 `train.load_path=`：
   - `vis_screwdriver_teacher.sh`：期望什么？
   - `vis_screwdriver_student_padapt.sh`：期望什么？

**预期输出：**

一个"checkpoint 流向图"：

```
训练过程：
  train.py main()
  → agent = PPO(...)
  → agent.train()
    ├─ 每 save_frequency epochs：
    │  ├─ save(ep_X_step_XYZm_reward_R.pth)  → outputs/.../stage1_nn/
    │  └─ save(last.pth)  → outputs/.../stage1_nn/
    └─ 每当 reward > best_reward：
       └─ save(best_reward_R.pth)  → outputs/.../stage1_nn/

推理过程：
  vis_screwdriver_teacher.sh
  → train.py test=True checkpoint=...
  → agent.restore_test(checkpoint)
    └─ torch.load(checkpoint) 读取 .pth 文件
  → agent.test()
    └─ 推理循环，实时渲染
```

---

## 🎯 综合输出格式

对于每个 Task，请输出：

### Task 2.X：[文件名] - [目标]

**关键发现：**
- 点 1
- 点 2
- ...

**代码片段：**
```python
# 关键代码行 + 行号
```

**修复/迁移建议：**
- [ ] 检查项 1
- [ ] 检查项 2

**对应的测试命令：**
```bash
# 验证该 task 的命令
```

---

## 📅 执行顺序

1. **Task 2.1** → 理解任务注册机制
2. **Task 2.2** → 理解观测空间（决定 obs 维度）
3. **Task 2.3** → 理解任务参数（决定改什么 YAML）
4. **Task 2.4** → 理解超参（决定改什么超参）
5. **Task 2.5** → 理解 checkpoint（决定如何保存/加载）

完成这五个 Task 后，就可以开始**Task 3：实际迁移代码**了。

---

## 🔗 相关文件导航

| 问题                  | 相关文件                                                                     |
| --------------------- | ---------------------------------------------------------------------------- |
| 任务如何注册？        | `dexscrew/tasks/__init__.py`                                                 |
| 观测空间如何定义？    | `dexscrew/tasks/xhand_hora.py` + `xhand_hora.py` 的 `compute_observations()` |
| 任务参数如何配置？    | `configs/task/XHandHoraScrewDriver.yaml`                                     |
| 训练超参如何配置？    | `configs/train/XHandHoraScrewDriver.yaml`                                    |
| Checkpoint 如何保存？ | `dexscrew/algo/ppo/ppo.py` 的 `save()` 方法                                  |
| Checkpoint 如何加载？ | `dexscrew/algo/ppo/ppo.py` / `padapt.py` 的 `restore_test()` 方法            |
| 如何运行推理？        | `scripts/vis_*.sh` + `train.py` 的 test 分支                                 |

