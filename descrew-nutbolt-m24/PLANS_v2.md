# PLANS_v2.md

## 1. 当前阶段定义
当前阶段不是继续扩大 diffusion 路线，而是：

**以“证据硬化 -> latent gap closing -> latent diffusion”作为当前唯一主线，并把 residual/corrective diffusion 设为明确保底主线。**

本阶段的核心目标不是“证明 diffusion 一定优于当前 student”，而是：
- 用可追溯证据确认哪条 diffusion 路线最适合本科毕设继续推进
- 避免继续被 action diffusion 的不确定信号和文档印象驱动
- 在可验证、可复现、可写论文的前提下完成阶段收敛

---

## 2. 本轮修订中的保留 / 修订 / 降级

### 2.1 保留
以下内容继续保留，不改动其治理地位：

- canonical path 仍是：
  `XHandHoraScrewDriver -> PPO teacher -> current student -> diffusion student -> unified evaluation`
- current student 的准确定位保持不变：
  - action BC
  - latent distillation
  - adapter-only adaptation
- baseline 组继续保留：
  - PPO teacher
  - current student
  - pure BC
- 当前阶段仍坚持：
  - 本科毕设优先
  - 路径可控
  - 实验可验证
  - 统一评测口径优先于新方法扩张

### 2.2 修订
以下内容是本轮正式修订点：

- diffusion 主线从“action diffusion first”修订为：
  - **Mainline A：evidence hardening -> latent gap closing -> latent diffusion**
  - **Mainline B：若 latent gate 未过，则转 residual/corrective diffusion**
- 为 latent 主线新增强制 gate：
  - AE/latent reconstruction gate
  - decode-only rollout stability gate
- 为 residual 主线新增强制 gate：
  - residual magnitude distribution gate
  - normalization stability gate
- evidence hardening 上升为阶段验收要求，而非建议
- robustness harness 上升为高优先级，与 nominal evaluation 并行
- sampling efficiency 保留，但仅作为 supporting axis，不得抢占主线

### 2.3 降级 / 移出
以下内容从“当前主推进项”降级：

- action diffusion：
  - 降级为 baseline / appendix / exploratory branch
  - 允许保留结果与低成本维护
  - 不再允许大规模 sweep 作为当前阶段中心工作
- diffusion export parity：
  - 当前仍记录缺口
  - 不升格为本阶段必达 milestone
- closed-loop correction：
  - 只保留 minimal wrapper 级别探索
  - 不得扩展成新的大架构项目

---

## 3. 当前阶段目标

### 目标 A：完成证据硬化
把当前阶段所有会影响治理判断的实验结果，从“文档结论”升级成“可追溯证据结论”。

### 目标 B：完成 latent gap closing 判断
先验证 latent 表达本身是否能保留 teacher 行为，再决定是否继续把 latent diffusion 作为主线推进。

### 目标 C：完成 latent diffusion 的最小可信比较
在统一 protocol 下，把 latent diffusion 与 current student / pure BC 做规范化比较。

### 目标 D：建立 residual/corrective diffusion 的保底主线地位
如果 latent route 在 gap-closing 阶段失败，立即切换到 residual diffusion，而不是重新回到 action diffusion 主线。

### 目标 E：把 action diffusion 固定为比较对象而非默认主线
保留其结果用于：
- 对比
- 附录
- 负结果/中性结果证据
- 说明为什么 Plan v2 转向 latent / residual

---

## 4. 当前阶段主线结构

### Mainline A：Evidence hardening -> latent gap closing -> latent diffusion
当前正式主线顺序：

1. baseline + evaluation evidence hardening
2. latent action/chunk representation learning
3. latent gap closing 验证
4. latent diffusion training
5. latent diffusion vs current student / BC comparison
6. 仅在主行为正确后，再考虑 sampling efficiency

### Mainline B：Residual / corrective diffusion fallback
若 Mainline A 在 gate 上失败，则自动切换为：

1. current student 作为 base policy
2. diffusion 学习 residual correction
3. 先做 nominal + robustness incremental improvement
4. 若 residual 明确有效，则以 residual diffusion 作为阶段主贡献收敛

### Exploratory branch：Action diffusion
action diffusion 保留，但地位明确为：
- exploratory branch
- appendix direction
- comparative baseline

只有当出现新的强证据表明其在当前任务上存在明确优势，才允许重新请求治理层讨论是否回升主线。

---

## 5. 当前 milestone

### M0：治理对齐
完成条件：
- `PLANS_v2.md` 生效
- latent-first + residual-fallback 成为正式主线
- action diffusion 正式降级为 exploratory/baseline
- evidence block 成为阶段验收硬要求

### M1：Evidence hardening baseline pack
完成条件：
- teacher / current student / pure BC 的 baseline comparison table 建立
- nominal + robustness protocol 明确
- 每个接受结果都附带 evidence block
- summary / handoff / acceptance 三类文档数值完成一次 canonical 对齐

### M2：Latent gap closing gate
完成条件：
- action 或 action-chunk AE / latent representation 完成训练
- reconstruction metrics 明确
- decode-only rollout stability 结果明确
- 能回答：latent 表达是否足以保留 teacher 关键行为

若 M2 未通过，则不进入 latent diffusion 主训练，直接转 M4。

### M3：Latent diffusion minimum credible run
完成条件：
- latent diffusion 在统一 protocol 下完成最小训练与评测
- 至少完成：
  - nominal
  - robustness
  - multiseed / repeated runs（按当前资源取最小可行）
- 与 current student / BC 的比较进入统一 table
- 能明确回答 latent diffusion 的收益点或失败点

### M4：Residual diffusion fallback gate
触发条件：
- latent gap-closing 未通过
或
- latent diffusion 无法形成可接受收益且主要瓶颈不适合继续投入

完成条件：
- residual target 定义明确
- residual magnitude distribution 已分析
- normalization / loss scaling 稳定
- nominal 下至少可运行
- robustness 下能判断是否存在边缘增益

### M5：阶段收敛
完成条件：
- 明确当前最适合继续推进的是：
  - latent diffusion
  - residual diffusion
  - 或保持 non-diffusion baseline + 结束 generative 主线扩张
- 已能写清楚：
  - 为什么 action diffusion 不再是主线
  - 哪些结论是 artifact-backed
  - 下一阶段只聚焦一个 supporting axis

---

## 6. Gate 与强制验收条件

### Gate G1：Evidence block gate
任何进入阶段结论的实验，都必须附带 evidence block。
缺少 evidence block 的结果只能标记为：
- provisional
- partial
- scaffold-only

### Gate G2：Latent gap closing gate
进入 latent diffusion 主训练前，必须同时满足：

- latent reconstruction 已完成
- reconstruction error 已报告
- decode-only rollout 可运行
- decode-only rollout 在关键任务指标上未出现明显不可接受崩坏
- 已能说明 latent gap 是否可接受

### Gate G3：Residual readiness gate
进入 residual diffusion 比较前，必须同时满足：

- residual target 定义明确
- residual magnitude distribution 已统计
- normalization / scaling 方案明确
- residual 不是纯噪声主导
- 已说明 residual 学习在当前 baseline 上有无实际信号

### Gate G4：Decision gate
任何“主线切换”“路线否决”“阶段验收”都必须基于：
- comparison table
- evidence block
- 统一 protocol
- 至少一个 artifact pointer 集合

不得再基于单篇 summary 的印象性结论直接下判断。

---

## 7. Evidence block 要求
从本阶段开始，每个关键实验条目必须提供以下 evidence block：

- run ID
- git commit hash
- config snapshot
- dataset version / dataset hash（如果使用 rollout dataset）
- seeds
- evaluation episodes 数量
- primary metrics
- dispersion 指标（std / CI / 至少一种）
- artifact pointers
  - table path
  - plot/log path
  - checkpoint/output path
- 一句话结论：
  - support
  - not support
  - inconclusive

若缺少上述大部分字段，则不得作为阶段关闭依据。

---

## 8. 验收标准

### 8.1 工程验收
- canonical Hora path 仍然可用
- teacher/current student/baseline evaluation 入口未失效
- latent 或 residual 主线中至少一条可执行
- artifact 指针可追溯

### 8.2 baseline 验收
- 至少有 teacher / current student / pure BC
- nominal 与 robustness protocol 一致
- comparison table 中可回答：
  - current student 比 pure BC 强在哪里
  - diffusion 到底要替代什么、补什么缺口

### 8.3 latent 主线验收
若 latent 作为当前阶段主线收敛，必须满足：

1. G2 已通过
2. latent diffusion 有统一 comparison table
3. 至少有 nominal + robustness 结果
4. 结果不是只靠单次可视化
5. 能明确说明 latent route 的有效点或失败点

### 8.4 residual 主线验收
若 residual 作为当前阶段收敛方向，必须满足：

1. G3 已通过
2. current student 作为 base policy 保持可用
3. residual 改动是可增量比较的
4. 至少有 nominal + robustness 的改变量判断
5. 能明确说明 residual 是否比 latent 更适合毕设继续推进

### 8.5 action diffusion 分支验收
本阶段对 action diffusion 的要求仅为：

- 保留其已有结果和可比性
- 明确记录其当前定位：
  - exploratory
  - appendix
  - baseline
- 不再要求其承担主线级里程碑

### 8.6 论文验收
本阶段结束时，至少能清楚写出：

- 当前任务与 teacher-student 设定
- current student 的准确方法学定位
- 为什么 action diffusion 被降级
- 为什么 latent / residual 更适合当前阶段
- 哪些结论是 artifact-backed
- 下一阶段只保留哪个 supporting axis

---

## 9. 暂不处理的内容
以下内容不纳入本阶段主交付：

- action diffusion 大规模重新 sweep
- 新的 raw-action diffusion 架构扩展
- closed-loop correction 的完整研究化展开
- offline RL / critic / Diffusion-QL
- multimodal diffusion transformer
- Pasini stage-2 纳入当前主线
- sim-to-real / external repo 深度联动
- diffusion export parity 完整打通
- 以工程重构为目的的大改动

---

## 10. 当前阶段优先级

### P0
治理对齐：
- latent-first
- residual-fallback
- action diffusion 降级
- evidence block 生效

### P1
evidence hardening：
- baseline table
- nominal + robustness protocol
- artifact traceability
- 文档数值 canonical 对齐

### P2
latent gap closing：
- AE / latent representation
- reconstruction metrics
- decode-only rollout stability

### P3
latent diffusion 最小可信比较：
- nominal
- robustness
- repeated evaluation / multiseed
- 与 current student / BC 同口径比较

### P4
residual fallback（若触发）：
- residual target
- residual magnitude distribution
- normalization stability
- incremental robustness test

### P5
仅在主行为稳定后，再进入 supporting axis：
- sampling efficiency
- 或 minimal closed-loop wrapper

---

## 11. CodeAgent 可自行推进的问题
以下问题可由 CodeAgent 直接推进：

- evidence block 模板化与补齐
- baseline comparison table 的生成与对齐
- nominal / robustness protocol 的脚本化
- latent gap closing 所需最小实验
- decode-only rollout 稳定性验证
- residual target 构建、统计与最小训练验证
- action diffusion 分支的低成本维护性检查
- 与当前阶段严格一致的文档更新

前提：
- 不改变当前主线
- 不新增大型方法分支
- 不重写整体工程结构
- 不擅自把 action diffusion 升回主线

---

## 12. 必须生成 codeagent_issue.md 回馈 GPT 的问题
出现以下任一情况，必须升级：

### A. 主线冲突
- latent gap closing 无法判断是否通过
- latent diffusion 与 residual diffusion 的优先级需要重新排序
- 需要把 action diffusion 重新升为主线候选

### B. 证据失效
- 关键结论无法绑定 artifact pointers
- 不同文档中的 canonical metrics 冲突严重
- robustness / multiseed 结果推翻当前治理判断

### C. 路线转向
- latent route 明确失败
- residual route 明确无有效信号
- 必须引入 closed-loop/reactive wrapper 才能继续
- 需要把 sampling efficiency 升为主主线

### D. 工程边界失控
- 为完成本阶段必须进行大范围重构
- 当前 repo 不足以支撑 latent 或 residual 主线
- Pasini / 外部 repo 成为当前阶段不可绕过依赖

---

## 13. 当前阶段退出条件
当同时满足以下条件时，可结束 Plan v2 阶段：

- evidence hardening 已完成
- canonical comparison table 已建立
- G2 或 G3 至少一条完整走通
- 已明确当前最适合继续推进的是：
  - latent diffusion
  - residual diffusion
  - 或停止 generative 主线扩张
- action diffusion 的地位已稳定为：
  - appendix / baseline / exploratory
- 已明确下一阶段只保留一个 supporting axis：
  - sampling efficiency
  - 或 minimal closed-loop correction