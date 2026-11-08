# Hydra 配置与输出路径规则分析

## 1️⃣ Hydra 的默认 Config 组合

### defaults 链路（优先级从高到低）
```yaml
# configs/config.yaml
defaults:
  - _self_                           # 优先级1：config.yaml 本身
  - task: XHandHoraScrewDriver       # 优先级2：configs/task/XHandHoraScrewDriver.yaml
  - train: ${task}                   # 优先级3：configs/train/XHandHoraScrewDriver.yaml
  - override hydra/job_logging: disabled
```

**合并结果：**
```
config (base)
├── task.* (from XHandHoraScrewDriver.yaml)
├── train.* (from train/XHandHoraScrewDriver.yaml)
└── hydra.* (logging disabled)
```

### 关键 Config 对象结构

```
config
├── task_name: 'XHandHoraScrewDriver' (← ${task.name})
├── test: False                      (← 推理模式开关)
├── checkpoint: ''                   (← 加载checkpoint路径)
├── headless: False                  (← 是否显示渲染)
├── seed: 42                         (← 随机种子)
├── sim_device: 'cuda:0'             (← 物理引擎GPU)
├── rl_device: 'cuda:0'              (← 训练GPU)
├── graphics_device_id: 7            (← 渲染GPU)
│
├── task.*                           (from configs/task/...)
│   ├── name: 'XHandHoraScrewDriver'
│   ├── env.numEnvs: 8192 (可override)
│   ├── env.episodeLength: 800
│   └── ... (reward, controller, physics etc.)
│
└── train.*                          (from configs/train/...)
    ├── algo: 'PPO'                  (← 算法选择)
    ├── load_path: ${..checkpoint}   (← 从全局checkpoint解析)
    └── ppo.*
        ├── output_name: 'debug'     (← **输出目录关键参数**)
        ├── priv_info: True
        ├── proprio_adapt: False
        └── ... (network, learning_rate, etc.)
```

---

## 2️⃣ test=True 时的函数调用流程（可视化模式）

### 执行路径
```python
train.py main()
└── env = isaacgym_task_map[config.task_name](...)  # 创建环境（需要完整 env）
    ├── obs_dict = env.reset()  # 获取初始观察
    └── → agent.step() 读取观察，不更新权重
    
├── agent = eval(config.train.algo)(env, output_dir, config)
│   ├── self.model = ActorCritic(...)
│   ├── 冻结权重（不加载 optim）
│   └── self.model.eval()  # 设置为评估模式
│
├── if config.test:  # ← test=True 进入此分支
│   ├── agent.restore_test(config.train.load_path)  # 加载权重
│   │   └── torch.load(fn) → 读取 state_dict
│   │
│   └── agent.test()  # 推理循环（不计算梯度）
│       ├── obs_dict = self.env.reset()
│       ├── while True:
│       │   ├── mu = self.model.act_inference(...)
│       │   ├── obs_dict, r, done, info = self.env.step(mu)
│       │   ├── if done[0]: break  # 单个 episode 后停止
│       │   └── → 实时渲染（如 headless=False）
│       └── [推理结束，主程序退出]
│
else:  # ← test=False（训练模式）
    └── agent.train()  # 无限循环训练（直到 max_agent_steps）
```

**关键区别：**
- ✅ test=True：单 episode，无梯度计算，立即渲染
- ✅ test=False：无限循环，梯度计算，定期保存 checkpoint

---

## 3️⃣ 输出目录命名规则

### 根目录结构
```
outputs/
└── {config.train.ppo.output_name}/          # ← 由 config 指定
    ├── stage1_nn/    (PPO 权重)
    ├── stage1_tb/    (TensorBoard 日志)
    ├── stage2_nn/    (ProprioAdapt 权重)
    ├── stage2_tb/    (TensorBoard 日志)
    ├── gitdiff.patch (训练开始时记录)
    └── config_*.yaml (训练时的配置备份)
```

### output_name 的设定路径
```yaml
# configs/train/XHandHoraScrewDriver.yaml (默认)
ppo:
  output_name: 'debug'  # ← 默认值

# scripts/screwdriver_teacher.sh (覆盖)
train.ppo.output_name=XHandHoraScrewDriver_teacher/${CACHE}

# scripts/screwdriver_student_padapt.sh (覆盖)
train.ppo.output_name=XHandHoraScrewDriver_student_padapt/${CACHE}
```

### 具体例子
```bash
# 教师训练
./scripts/screwdriver_teacher.sh 0 42 Reproduction
# → outputs/XHandHoraScrewDriver_teacher/Reproduction/stage1_nn/

# 学生训练
./scripts/screwdriver_student_padapt.sh 0 42 Reproduction
# → outputs/XHandHoraScrewDriver_student_padapt/Reproduction/stage2_nn/

# 可视化
./scripts/vis_screwdriver_teacher.sh 0 42 Reproduction
# → 加载 outputs/XHandHoraScrewDriver_teacher/Reproduction/stage1_nn/best_reward_*.pth
```

---

## 4️⃣ Checkpoint 文件命名规则

### PPO 保存的文件

**文件格式：** `.pth`（PyTorch model）

**保存位置：** `stage1_nn/`

**文件名规则：**
```python
# 定期保存（每 save_frequency epochs）
checkpoint_name = f'ep_{epoch}_step_{steps}m_reward_{reward:.2f}.pth'
# 例: ep_100_step_0098m_reward_1912.78.pth

# 最后一次保存
last.pth

# 最佳奖励模型（每当 reward > best_reward 时）
best_reward_{reward:.2f}.pth
# 例: best_reward_2093.48.pth
```

**内容：** 包含 model + running_mean_std + priv_mean_std + point_cloud_mean_std

### ProprioAdapt 保存的文件

**文件格式：** `.ckpt`（PyTorch checkpoint）

**保存位置：** `stage2_nn/`

**文件名规则：**
```python
# 定期保存（每 1e8 steps）
{agent_steps // 1e8}00m.ckpt
# 例: 100m.ckpt, 200m.ckpt

# 最后保存
model_last.ckpt

# 最佳奖励模型
model_best.ckpt
```

**内容：** model + running_mean_std + sa_mean_std + priv_mean_std + point_cloud_mean_std

---

## 5️⃣ 配置表：修改影响范围

| 配置项                      | 位置                       | 默认值                   | 作用           | 修改影响                 | 移植需改                           |
| --------------------------- | -------------------------- | ------------------------ | -------------- | ------------------------ | ---------------------------------- |
| **test**                    | config.yaml                | `False`                  | 推理/训练模式  | 决定 test() vs train()   | ✅ 无需改                           |
| **headless**                | config.yaml                | `False`                  | 显示/无头渲染  | 环境是否显示窗口         | ❌ 需改为 True（通常训练）          |
| **checkpoint**              | config.yaml                | `''`                     | 加载预训练模型 | 决定从零开始还是继续训练 | ✅ 无需改（由脚本传）               |
| **seed**                    | config.yaml                | `42`                     | 随机种子       | 复现性                   | ✅ 可保持                           |
| **sim_device**              | config.yaml                | `'cuda:0'`               | 物理模拟GPU    | 性能                     | ❌ 可改为需要的GPU                  |
| **rl_device**               | config.yaml                | `'cuda:0'`               | 训练GPU        | 性能                     | ❌ 可改为需要的GPU                  |
| **graphics_device_id**      | config.yaml                | `7`                      | 渲染GPU        | 可视化性能               | ❌ 可改（或注释掉）                 |
| **task_name**               | config.yaml (${task.name}) | `'XHandHoraScrewDriver'` | 任务环境类     | 决定创建哪个任务         | ✅ **需新增 Pasini 任务**           |
| **train.algo**              | train/*.yaml               | `'PPO'`                  | 算法选择       | PPO vs ProprioAdapt      | ✅ 脚本自动切换                     |
| **train.ppo.output_name**   | train/*.yaml               | `'debug'`                | 输出目录名     | 模型保存位置             | ❌ 脚本中 override                  |
| **train.ppo.priv_info**     | train/*.yaml               | `True`                   | 使用特权信息   | 教师完整性               | ❌ 可调整                           |
| **train.ppo.proprio_adapt** | train/*.yaml               | `False`                  | 自适应学生模式 | 激活蒸馏                 | ✅ 脚本自动切换                     |
| **train.ppo.num_actors**    | train/*.yaml               | ${task.env.numEnvs}      | 环境数         | 批量大小/显存            | ❌ 可改（${task.env.numEnvs} 动态） |
| **task.env.numEnvs**        | task/*.yaml                | 8192                     | 并行环境数     | 显存占用、速度           | ❌ 可改                             |
| **task.env.numActions**     | task/*.yaml                | 12                       | 动作维度       | 手部自由度               | ✅ **需改为 Pasini 自由度**         |

---

## 6️⃣ 关键配置项精确名称

### 全局配置（config.yaml）
```yaml
test                # 推理开关
checkpoint          # 模型加载路径（绝对或相对）
headless            # 无头模式
seed                # 随机种子
sim_device          # 物理 GPU
rl_device           # 训练 GPU
graphics_device_id  # 渲染 GPU
```

### 任务配置（task/*.yaml）
```yaml
task_name                    # 环境类名称
task.env.numEnvs            # 并行环境数
task.env.numActions         # 动作维度（关键！）
task.env.episodeLength      # 任务长度
task.env.controller.*       # 控制器参数
task.env.reset_dist_threshold  # 重置阈值
```

### 训练配置（train/*.yaml）
```yaml
train.algo                          # 算法名（PPO / ProprioAdapt）
train.load_path                     # 加载路径（来自 ${..checkpoint}）
train.ppo.output_name              # 输出目录
train.ppo.priv_info                # 特权信息启用
train.ppo.proprio_adapt            # 学生自适应模式
train.ppo.num_actors               # 并行环境数（推荐 = numEnvs）
train.ppo.learning_rate            # 学习率
train.ppo.max_agent_steps          # 最大训练步数
```

---

## 7️⃣ 三个最关键 Override 示例

### 示例 1：单 GPU + 可视化测试
```bash
python train.py \
  task=XHandHoraScrewDriver \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  graphics_device_id=0 \
  headless=False \
  test=True \
  checkpoint=outputs/XHandHoraScrewDriver_teacher/Reproduction/stage1_nn/best_reward_2093.48.pth
```

**效果：**
- 单卡运行（display:0 GPU）
- 实时渲染
- 加载最佳模型
- 推理一个 episode 后退出

---

### 示例 2：Headless 训练（无渲染）
```bash
python train.py \
  task=XHandHoraScrewDriver \
  headless=True \
  seed=42 \
  sim_device=cuda:0 \
  rl_device=cuda:0 \
  train.ppo.output_name=exp_v1 \
  train.ppo.max_agent_steps=10000000000
```

**效果：**
- 无窗口（节省显存）
- 存储到 outputs/exp_v1/stage1_nn/
- 从零开始训练（checkpoint=''）
- 内存占用更少，速度更快

---

### 示例 3：指定 Checkpoint 继续训练
```bash
python train.py \
  task=XHandHoraScrewDriver \
  headless=True \
  checkpoint=outputs/XHandHoraScrewDriver_teacher/v1/stage1_nn/best_reward_2093.48.pth \
  train.ppo.output_name=exp_v1_continue \
  test=False
```

**效果：**
- 从已有的最佳模型继续训练
- 保存到新的目录 exp_v1_continue
- 不覆盖原有模型

---

## 8️⃣ 移植检查清单

### 需要修改的配置

- [ ] `task.env.numActions` - 改为 **Pasini Hand 的自由度**
- [ ] `task.env.numEnvs` - 根据显存调整（可保持默认）
- [ ] `task.env.controller.*` - 改为 Pasini 的控制参数
- [ ] `task.env.reset_dist_threshold` - 调整为灯泡任务的阈值
- [ ] `train.ppo.output_name` - 改为 `XHandPasiniLightbulb_teacher/v1` 等

### 无需修改的配置

- ✅ `test`, `checkpoint`, `headless` - 由脚本动态传递
- ✅ `seed`, `sim_device`, `rl_device` - 保持默认或脚本传递
- ✅ `train.algo`, `train.ppo.priv_info` - 算法选择

### 需要新增的配置

- 🆕 `configs/task/XHandPasiniLightbulb.yaml` - 新任务配置
- 🆕 `configs/train/XHandPasiniLightbulb.yaml` - 新训练配置
- 🆕 `dexscrew/tasks/xhand_pasini_lightbulb.py` - 新环境实现

---

## 9️⃣ 快速调试命令

```bash
# 1. 快速测试配置是否正确（1 env，1 step）
python train.py task=XHandHoraScrewDriver \
  task.env.numEnvs=1 \
  headless=False \
  test=True \
  checkpoint=outputs/XHandHoraScrewDriver_teacher/Reproduction/stage1_nn/best_reward_2093.48.pth

# 2. 训练 10M steps（快速验证）
python train.py task=XHandHoraScrewDriver \
  headless=True \
  train.ppo.output_name=debug_test \
  train.ppo.max_agent_steps=10000000

# 3. 列出所有可用 config 参数
python train.py --cfg=job task=XHandHoraScrewDriver

# 4. 打印完整合并后的配置
python train.py --cfg=all task=XHandHoraScrewDriver
```

---

## 🚨 三个最常见的移植错误（务必避免）

### ❌ 错误 1：task_name / task / isaacgym_task_map 三者不一致

**发生场景：** 移植到 Pasini 手时

**错误表现：**
```
KeyError: 'XHandPasiniLightbulb' not found in isaacgym_task_map
```

**根本原因：** 三者没有同步

**修复清单（必须全做）：**

1️⃣ 在 [configs/task/XHandPasiniLightbulb.yaml](configs/task/XHandPasiniLightbulb.yaml) 顶部设置：
```yaml
name: XHandPasiniLightbulb  # ← 这是 task.name
```

2️⃣ [config.yaml](configs/config.yaml) 中自动解析（无需改）：
```yaml
task_name: ${task.name}  # ← 自动读取 task.name → task_name
```

3️⃣ 在 [dexscrew/tasks/__init__.py](dexscrew/tasks/__init__.py) 注册新类：
```python
from dexscrew.tasks.xhand_hora import XHandHora  # 复用或新建子类
from dexscrew.tasks.xhand_pasini import XHandPasini  # 如果新建

isaacgym_task_map = {
    'XHandHoraScrewDriver': XHandHora,
    'XHandPasiniLightbulb': XHandPasini,  # ← 添加这行，key 必须与 task.name 一致！
}
```

**验证命令：**
```bash
python train.py task=XHandPasiniLightbulb --cfg=job | grep -A5 "task_name\|task.name"
# 应该看到：
# task_name: XHandPasiniLightbulb
# task.name: XHandPasiniLightbulb
```

---

### ❌ 错误 2：只改了 numActions，没改 obs 维度

**发生场景：** 从 XHand (12 DOF) 迁移到 Pasini (可能 22 DOF) 时

**错误表现：**
```
RuntimeError: Expected shape (batch, 96) but got (batch, 76)
# 或
RuntimeError: num_obs=96 but observation buffer has 76 dimensions
```

**根本原因：** `compute_observations()` 的输出维度也变了，不只是 action

**修复清单（必须全做）：**

1️⃣ 统计 Pasini 的观测维度：
```python
# dexscrew/tasks/xhand_pasini.py (假设新建该文件)

# 原 XHand:
self.numActions = 12  # 12 DOF
# obs_buf 结构：
#   - joint pos history: 12 * 30 = 360 (历史30帧，每帧12维)
#   - target pos history: 12 * 30 = 360
#   - 其他: point cloud (100*3=300), privileged info (N)
# 总计：config.yaml 中 network.input_shape = [obs_dim]

# 新 Pasini:
self.numActions = 22  # 假设 22 DOF
# obs_buf 需要重新计算：
#   - joint pos history: 22 * 30 = 660
#   - target pos history: 22 * 30 = 660
#   - 其他: point cloud (100*3=300), privileged info (M)
# 总计：需要在 config 中修改 input_shape
```

2️⃣ 在 `compute_observations()` 中修改：
```python
def compute_observations(self):
    # ... 省略前面的代码
    
    # XHand 原逻辑（12 DOF）:
    # cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # shape: [N, 1, 12]
    # cur_tar_buf = self.cur_targets[:, None, :self.num_actions]  # shape: [N, 1, 12]
    
    # Pasini 新逻辑（22 DOF）:
    cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # shape: [N, 1, 22]
    cur_tar_buf = self.cur_targets[:, None, :self.num_actions]  # shape: [N, 1, 22]
    cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # shape: [N, 1, 44]
    
    self.obs_buf_lag_history[:] = torch.cat([prev_obs_buf, cur_obs_buf], dim=1)
    # 现在历史缓冲的最后一维是 44（而不是 24）
```

3️⃣ 在配置中更新 `num_obs`：
```yaml
# configs/task/XHandPasiniLightbulb.yaml

env:
  numActions: 22  # ← 改这里
  # 还需要添加
  numObs: ???  # ← 计算 obs_buf 的总维度
  # obs_buf = [22*30 joint pos历史, 22*30 target pos历史, 100*3 点云, priv_info维度]
```

4️⃣ 在网络配置中同步：
```yaml
# configs/train/XHandPasiniLightbulb.yaml

network:
  mlp:
    units: [512, 256, 128]
  priv_mlp:
    units: [256, 128, 8]
  point_mlp:
    units: [32, 32, 32]
  # ← 以上都不需要改，ActorCritic 会从 env 读 input_shape
```

**验证命令：**
```bash
python train.py task=XHandPasiniLightbulb test=True checkpoint=... headless=False 2>&1 | grep -i "shape\|dimension"
# 不应该出现 shape mismatch 错误
```

---

### ❌ 错误 3：Checkpoint 文件格式与搜索规则混淆

**发生场景：** 训练完成后，运行 vis 脚本找不到模型

**错误表现：**
```
FileNotFoundError: outputs/XHandPasini_teacher/v1/stage1_nn/best_reward_*.pth [Errno 2]
```

**根本原因：** 
- 教师只产生了 `ep_*.pth` 或 `last.pth`，没有 `best_reward_*.pth`
- vis 脚本期望的文件名和实际保存的不一致

**修复清单（必须全做）：**

1️⃣ 理解保存规则（[ppo.py#L235-L251](dexscrew/algo/ppo/ppo.py#L235)）：
```python
# PPO 保存策略（stage1）：
# 定期保存（根据 save_freq）：
#   checkpoint_name = f'ep_{epoch}_step_{steps}m_reward_{reward:.2f}.pth'
#   + last.pth（每次 save_freq 都覆盖）

# 最佳模型保存（只在 reward > best_reward 时）：
#   best_reward_{reward:.2f}.pth

# ProprioAdapt 保存策略（stage2）：
#   model_last.ckpt, model_best.ckpt（不同的命名！）
```

2️⃣ vis 脚本期望的默认路径：
```bash
# vis_screwdriver_teacher.sh 第 16 行：
checkpoint=$(find outputs/XHandHoraScrewDriver_teacher/output_name/stage1_nn \
  -name "best_reward_*.pth" | head -1)
# ↑ 明确指定找 best_reward_*.pth

# vis_screwdriver_student_padapt.sh 第 18 行：
train.load_path=outputs/XHandHoraScrewDriver_student_padapt/${CACHE}/stage2_nn/last.pth
# ↑ 找 model_last.ckpt（但这里还是用 last.pth...需要检查）
```

3️⃣ 如果训练中途中断（没有产生 best_reward）：
```bash
# 方案 A：使用最后一个 epoch checkpoint
checkpoint=outputs/XHandPasini_teacher/v1/stage1_nn/last.pth

# 方案 B：使用任意 ep_*.pth（选最新的）
checkpoint=$(ls -t outputs/XHandPasini_teacher/v1/stage1_nn/ep_*.pth | head -1)

# 方案 C：改 vis 脚本，改为搜索 last.pth
# 在 scripts/vis_screwdriver_teacher.sh 中改为：
checkpoint=outputs/XHandPasini_teacher/${CACHE}/stage1_nn/last.pth
```

4️⃣ 验证当前保存了什么：
```bash
ls -lh outputs/XHandPasini_teacher/v1/stage1_nn/
# 看输出里有没有 best_reward_*.pth / last.pth / ep_*.pth
```

**常见场景排查：**

| 现象                                 | 原因                              | 解决方案                        |
| ------------------------------------ | --------------------------------- | ------------------------------- |
| 只有 `last.pth`                      | 训练中断或 reward 从未超过初始值  | 用 `last.pth` 或继续训练        |
| 有 `best_reward_2093.48.pth`         | 正常（reward 曾达到 2093.48）     | 直接用这个文件                  |
| 有很多 `ep_*.pth` 但没 `best_reward` | 可能 save_freq=0（disabled）      | 检查 `train.ppo.save_frequency` |
| `.pth` vs `.ckpt` 混淆               | 混淆了教师（.pth）和学生（.ckpt） | 教师用 `.pth`，学生用 `.ckpt`   |

---

**总结：移植时的三个关键修改点**

1. **任务定义** - 创建 XHandPasiniLightbulb 的 YAML 配置和 env 类
   - ✅ 同步 task.name / task_name / isaacgym_task_map 三者
2. **动作维度** - numActions 改为 Pasini 的自由度
   - ✅ 同时改 obs 维度（不只是 action，观测也变了）
3. **脚本参数** - output_name 改为新的实验名（脚本中 override）
   - ✅ 注意 checkpoint 文件格式：教师 `.pth`，学生 `.ckpt`
