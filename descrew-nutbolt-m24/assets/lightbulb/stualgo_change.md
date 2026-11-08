# 新 Student 蒸馏算法迁移指南

适用于当前 DexScrew / HORA 衍生项目，面向未来新增一种新的 student 蒸馏算法，例如 diffusion 类 student。  
本文目标不是实现算法，而是沉淀一条“从算法定义到训练、评测、导出”的完整迁移方法，让后续开发者能够把新的 student 算法准确接入现有 task 的 student 阶段，并保证完整训练流程跑通。

当前仓库里还没有现成 diffusion student 实现，因此本文会以 diffusion 作为典型例子，但不把内容限制为某一种 diffusion 变体。

---

## 1. 背景与适用范围

当前项目的整体训练结构是：

- `teacher(PPO)`
- `student algorithm`

也就是说，环境和 task 先服务于 teacher 训练，再基于 teacher 的监督或 expert 数据训练 student。

目前仓库中已落地的 student 路线只有两条：

- `ProprioAdapt / PADAPT`
- `DOTPGStudent`

本指南面向“新增第三种 student 算法”的场景，例如：

- diffusion action policy
- diffusion latent policy
- diffusion + BC
- diffusion + teacher latent distillation

本指南不面向：

- 替换 PPO teacher
- 重写 task / env 架构
- 新建一套和当前 `train.py` 完全独立的训练系统

默认前提是：

- 现有 task 继续复用
- 现有 hand/object 环境继续复用
- 新算法只作为新的 student 路线接入

---

## 2. 当前项目中 student 算法的真实接入位置

student 算法在当前项目中的统一调度入口是：

- `train.py`

这里的关键逻辑是：

- `agent = eval(config.train.algo)(env, output_dif, full_config=config)`

这说明：

- `train.algo` 决定具体使用哪个 student 类
- 只要类被正确导入到 `train.py`，并且接口兼容，就能接入训练入口

### 现有 student 算法定义位置

- `ProprioAdapt`
  - `dexscrew/algo/ppo/padapt.py`

- `DOTPGStudent`
  - `dexscrew/algo/dotpg/dotpg.py`

### 新增 student 算法至少会涉及的接入点

如果你要新增一种 student 算法，例如 `DiffusionStudent`，通常至少要进入：

- 算法类定义
- 训练配置
- 脚本入口
- 可视化 / 评测入口
- 如需导出，还要进入：
  - `student_eval.py`
  - `scripts/convert_student_jit.sh`

### 当前仓库中的现实边界

要特别注意：

- `train.py` 已支持多 student 算法的统一调度
- 但 `student_eval.py` 当前只支持 `ProprioAdapt`
- `convert_student_jit.sh` 也默认走 `ProprioAdapt`

这意味着：

- 新 student 算法不是只要能训练就算接入完成
- 如果需要完整链路，还要考虑 eval / vis / export 的兼容性

---

## 3. 当前 student 算法的共性接口契约

从项目现状来看，一个新的 student 算法如果想被 `train.py` 正确接入，至少应满足以下接口契约。

### 3.1 构造函数形态

推荐保持与当前 student 算法一致：

```python
__init__(self, env, output_dir, full_config, ...)
```

原因：

- `train.py` 直接按这个形态实例化
- 现有 `ProprioAdapt` 和 `DOTPGStudent` 都遵循这套模式

### 3.2 必须提供的方法

新 student 算法至少应实现：

- `train()`
- `test()`
- `restore_train()`
- `restore_test()`

推荐也保留：

- `save()`
- `set_eval()`
- `set_train()`

### 3.3 `train.py` 当前如何调用这些方法

训练时：

- `agent.restore_train(config.train.load_path)`
- `agent.train()`

测试时：

- `agent.restore_train(config.checkpoint)`（仅在某些算法需要 teacher ckpt 时）
- `agent.restore_test(config.train.load_path)`
- `agent.test()`

所以新算法必须提前明确：

- `restore_train()` 是加载 teacher、还是加载 student warm start、还是两者之一
- `restore_test()` 只加载 student，还是还需要 teacher 参与

### 3.4 输出目录约定

当前 student 算法有自己的输出目录风格：

- `ProprioAdapt`
  - `stage2_nn/`
  - `stage2_tb/`

- `DOTPGStudent`
  - `student_output/dotpg_nn/`
  - `student_output/dotpg_tb/`

新增 student 算法必须在文档和实现里统一输出目录规则，否则会导致：

- vis 脚本不好写
- ckpt 路径难以约定
- 导出和恢复逻辑容易断链

### 3.5 如果新算法不兼容这套接口怎么办

如果新 student 算法天然不适合当前接口，例如：

- test 阶段必须同时加载多个模型对象
- 推理入口不是单一 `test()`
- 训练流程和 `checkpoint / train.load_path` 分工完全不同

那就不能只写算法类，还必须同步修改：

- `train.py`

不要默认任何新算法都天然兼容现有调度逻辑。

---

## 4. 算法定义位置与推荐目录结构

新增 student 算法时，不建议继续把代码塞进已有的 `padapt.py` 或 `dotpg.py`。  
推荐为新算法建立独立目录。

### 推荐目录结构

以 diffusion 类算法为例，推荐新增：

```text
dexscrew/algo/diffusion/
```

目录内建议至少包含：

- 主算法实现文件
- 网络定义
- buffer / sampler / diffusion schedule 等辅助模块
- `__init__.py`

例如：

```text
dexscrew/algo/diffusion/__init__.py
dexscrew/algo/diffusion/diffusion_student.py
dexscrew/algo/diffusion/networks.py
dexscrew/algo/diffusion/schedule.py
dexscrew/algo/diffusion/buffer.py
```

### 推荐算法类名

例如：

- `DiffusionStudent`

类名需要满足两个条件：

- 在 `train.py` 中被导入
- 能通过 `train.algo=DiffusionStudent` 正确实例化

### 不推荐的做法

如果新算法只是“受 PADAPT 启发”，也不要继续往：

- `dexscrew/algo/ppo/padapt.py`

里硬塞一个新分支。  
优先独立目录、独立类、独立 config，这样后续维护最清晰。

---

## 5. 如何把新 student 算法接入现有训练调度

这一步的目标不是“先写出算法”，而是先保证训练入口能正确找到它。

### 步骤 1：在算法目录新增类

例如新增：

- `DiffusionStudent`

### 步骤 2：在 `train.py` 中导入

让 `train.py` 能解析：

- `train.algo=DiffusionStudent`

### 步骤 3：保持调用契约兼容

确认以下行为和现有入口兼容：

- `restore_train()`
- `restore_test()`
- `train()`
- `test()`

### 步骤 4：明确 `checkpoint` 与 `train.load_path` 的职责

在当前项目中，建议延续以下约定：

- teacher ckpt 通常走：
  - `checkpoint`

- student 自身恢复通常走：
  - `train.load_path`

如果新算法同时依赖：

- teacher ckpt
- student ckpt

那么文档和实现都必须把这层约定写清楚，并保证 `train.py` 的 test 分支能正确工作。

### 推荐做法

优先让新算法的接口尽量贴近现有 student 算法，而不是先改 `train.py`。  
只有当算法本身确实不兼容时，再修改统一调度逻辑。

---

## 6. 如何把新算法准确无误迁移到 task 的 student 阶段

这是新增 student 算法时最关键的一步。

### 6.1 默认不需要新建 task class

通常情况下：

- 不需要新建新的 task class
- 不需要改 `task_name -> env` 映射

默认继续复用现有 task，例如：

- `XHandHoraScrewDriver`
- `RBHandHoraLightbulb`
- `Dexh13HoraLightbulb`

task 侧继续通过 `obs_dict` 提供输入，新算法自己决定消费哪些字段。

### 6.2 student 输入设计必须先锁定

在真正写代码前，必须先决定新算法属于哪一种输入范式：

- teacher-state 型
- student-state 型
- diffusion-state / latent-state 型

这一步不能模糊处理，因为它会直接影响：

- policy 输入
- teacher 监督目标
- config 字段
- train/test 是否一致

### 6.3 必须先回答的输入问题

新算法在设计时必须先明确：

- policy 输入是什么
- teacher 监督目标是什么
- 是否使用 `priv_info`
- 是否使用 `proprio_hist`
- 是否使用 `point_cloud_info`

这些决策不能只停留在算法注释里，必须对应到 config 字段和脚本用法。

### 6.4 以 diffusion 为例，必须先锁定的设计问题

如果要新增 diffusion 类 student，至少要先回答：

- diffusion 是生成 `action` 还是 `latent`
- 去噪目标是什么
- 条件输入是什么
- 推理时是否还需要 teacher
- 训练时 teacher 提供的是：
  - action
  - latent
  - 还是两者都有

### 6.5 推荐的迁移顺序

建议按下面顺序把新算法接到 task 的 student 阶段：

1. 先选一个已稳定的 task 做最小落地
2. 明确 student 输入字段
3. 明确 teacher 提供的监督
4. 定义 train config
5. 跑通单 task 的 train
6. 再扩展到其他 hand / task

不要一开始就试图同时支持所有 hand 和所有 task。

---

## 7. teacher 依赖与蒸馏监督来源

这是 student 迁移最容易出错的地方，必须单独写清楚。

对每一个新增 student 算法，都必须明确三件事：

- teacher 在训练时提供什么
- 新 student 学什么
- 推理时还剩下什么

### 7.1 必须显式回答的问题

实现者必须先回答以下问题：

- 是否依赖 PPO teacher checkpoint
- 是否依赖 teacher latent
- 是否只做 action imitation
- 是否需要在线 teacher rollout
- 是否要像 DOTPG 一样保存 expert buffer

### 7.2 为什么这一步很关键

如果这部分没有先锁定，很容易出现：

- train 能跑
- 但 test 需要 teacher 却没加载
- 或者 export 时根本不知道推理阶段是否还依赖 teacher

### 7.3 典型三种监督来源

当前项目中的 student 蒸馏大致有三种监督来源形式：

- teacher action
- teacher latent
- teacher rollout / expert buffer

新增算法必须说明自己属于哪一种，或者几种的组合。

### 7.4 对 diffusion 类算法的特别提醒

如果你做的是 diffusion student，建议尽早明确：

- diffusion 训练目标是直接重建 teacher action
- 还是重建 teacher latent 再 decode 成 action
- 还是同时使用 action loss 和 latent loss

这会直接决定：

- 是否复用 PADAPT 风格输入
- 是否复用 DOTPG 风格 expert buffer
- export 阶段是否只导出 denoiser 还是导出完整 student policy

---

## 8. 训练配置接入方式

当前项目的 config 组织方式很明确：

- 顶层主配置在：
  - `configs/config*.yaml`

- 每个 task 对应 train config 在：
  - `configs/train/*.yaml`

### 8.1 新 student 算法推荐的 config 组织方式

新增 student 算法时，推荐新增：

- 一个或多个 train config
- 必要时新增主配置文件

### 8.2 推荐组织原则

如果新算法是 task 无关的 student 框架，推荐仿照 DOTPG 的组织方式：

- 顶层主配置控制整体入口
- 每个 task 各有对应 train config

如果算法存在明显不同的工作模式，例如：

- teacher-state
- student-state
- latent-diffusion
- action-diffusion

可以拆成多个 train config，不要强行塞在一个文件里用大量 if/flag 维持。

### 8.3 train config 中必须明确的内容

新增 train config 时，至少要明确：

- `algo`
- `load_path`
- teacher 依赖
- model/network 超参数
- replay / buffer / diffusion schedule 等专属超参数
- 输出目录命名规范

### 8.4 参数命名建议

如果是 diffusion 类 student，建议新参数走独立命名空间，例如：

- `train.diffusion.*`

或者：

- `train.<algo_name>.*`

不要把新算法参数硬塞进：

- `train.ppo.*`

除非该字段本来就属于 teacher / shared observation contract。

---

## 9. 脚本接入方式

当前仓库的脚本组织原则是：

- teacher、student、vis 脚本都放在 `scripts/`
- 同一 task 的脚本命名尽量对齐

### 9.1 新 student 算法的脚本建议

推荐至少新增：

- 训练脚本：
  - `scripts/<task>_student_<algo>.sh`

- 评测/可视化脚本：
  - `scripts/vis_<task>_student_<algo>.sh`

### 9.2 脚本中至少要覆盖的内容

脚本中建议显式覆盖：

- `task=...`
- `train=...`
- `train.algo=...` 或主配置中的默认算法
- teacher ckpt 传递
- `train.load_path` 传递
- 常用 env 覆盖

### 9.3 不建议长期只靠手写长命令

新算法初期可以先手写命令验证，但一旦确认训练能跑通，建议尽快固化脚本。  
否则后续很容易在：

- config-name
- teacher ckpt 传参
- train.load_path
- eval 模式

这些细节上出现不一致。

---

## 10. 完整训练流程保证项

新增 student 算法时，不应只保证“能开始训练”，还必须保证整条 student 链不断。

推荐按以下顺序逐项确认：

1. teacher 已训练并可加载
2. student train config 能实例化新算法
3. `restore_train()` 能正确处理 teacher ckpt
4. `train()` 能输出 student ckpt 到约定目录
5. `restore_test()` 能正确只加载 student 或同时加载 teacher+student
6. `test()` 与 vis 脚本能跑通
7. 如需导出，`student_eval.py` 与 `convert_student_jit.sh` 能兼容

### 10.1 当前项目里最容易被忽略的断链点

最常见的问题不是算法主体，而是链路后半段：

- ckpt 保存结构不统一
- test 阶段加载约定不清晰
- vis 脚本和 train config 不匹配
- export 假设 student 是单步 actor，但 diffusion 推理不是

所以新增 student 算法必须从一开始就把：

- train
- test
- vis
- export

看成一条完整链，而不是只看训练主体。

---

## 11. 导出与部署路径

由于本指南覆盖完整 student 链，必须把导出路径一起考虑进去。

### 11.1 当前项目已有的导出入口

当前仓库已有：

- `student_eval.py`
- `scripts/convert_student_jit.sh`

### 11.2 当前导出链的现实限制

当前 `student_eval.py` 实际只支持：

- `ProprioAdapt`

也就是说，如果新增：

- `DiffusionStudent`

那么不能默认导出链自动可用，必须显式评估：

- 是否能导出成统一 student policy
- 是否需要额外 wrapper
- 是否只导出 actor / decoder / denoiser 的一部分

### 11.3 diffusion 类算法的导出特别提醒

如果 diffusion 推理依赖多步采样，必须提前回答：

- 导出与部署链是否允许多步采样
- 是否要把 diffusion student 再蒸馏成单步 policy
- 是否需要额外的 decode-only 推理模式

如果不先回答这些问题，后面很容易出现：

- 训练效果不错
- 但无法用现有导出链部署

---

## 12. 常见失败模式与排查顺序

新增 student 算法时，建议固定按下面顺序排查。

### 1. `train.algo` 无法实例化

优先改：

- `train.py` 的 import
- 算法类名
- config 中的 `train.algo`

### 2. `restore_train()` 与 teacher ckpt 不兼容

优先改：

- 新算法的 `restore_train()`
- teacher ckpt 读取逻辑
- `checkpoint` 与 `train.load_path` 约定

### 3. student 输入字段与 task 输出不匹配

优先改：

- algorithm class
- task obs contract
- config 中关于 `priv_info / proprio_hist / point_cloud_info` 的设置

### 4. config 路径和 train config 组织错误

优先改：

- `configs/config*.yaml`
- `configs/train/*.yaml`
- 脚本中的 `--config-name` 与 `train=...`

### 5. train 能跑但 test / vis 加载失败

优先改：

- `restore_test()`
- vis 脚本中的 ckpt 传递方式
- teacher / student 是否都应在 test 阶段加载

### 6. student ckpt 保存结构与 `restore_test()` 不一致

优先改：

- `save()`
- `restore_test()`

### 7. 导出路径与 student forward 形式不兼容

优先改：

- `student_eval.py`
- convert 脚本
- wrapper
- 推理接口定义

---

## 13. 最终迁移 checklist

按下面顺序做，最容易把一个新的 student 算法稳妥接进当前项目。

1. 定义新算法目录和类
2. 接入 `train.py`
3. 明确 teacher 监督来源
4. 新增 train config
5. 新增训练脚本
6. 在一个现有 task 上跑通 train
7. 跑通 test / vis
8. 确认 ckpt 约定
9. 如需部署，跑通 export

---

## 14. 结论

在当前项目中，新增一种 student 蒸馏算法，最稳妥的路线不是“先写一个模型再说”，而是先明确四件事：

- 训练入口如何接入
- teacher 提供什么监督
- task 输出哪些输入字段
- train / test / vis / export 是否形成完整闭环

对于 diffusion 类 student，尤其要先锁定：

- diffusion 是生成 action 还是 latent
- teacher 在训练时提供 action、latent 还是两者
- 推理时是否还需要 teacher
- 导出时是否允许多步采样

只要这些问题先锁定，再按当前项目已有的 `PADAPT` 和 `DOTPGStudent` 接入规律来组织算法、config 和脚本，就能够以最小风险把新的 student 算法迁移到现有灵巧手 task 的 student 阶段。
