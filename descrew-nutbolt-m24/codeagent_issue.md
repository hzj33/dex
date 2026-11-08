# codeagent_issue.md

## 2026-09-20 — 局部适配恢复模拟完整循环；实机与严格受力验收仍待验证

### Task background
用户目标为sim2real真实M24完整循环，明确允许原部署基础上的小幅辅助；撤销历史条目中代理擅自禁止一切辅助的限制。显式张开只完成诊断，不能替代任务。

### Current blocker
真实接触闭环未验证；匹配执行学生的原执行五轮与5N验收未过。不是现有teacher/student接口不兼容，也未触发算法重设计。

### Evidence
同指令空载主动关节RMSE从1.787/2.561°降至0.332/0.683°，支持近似执行响应；不证明接触校准。新a819学生匹配执行+被动DIP固定五轮1/1，四组扰动467/512且全部至少四轮；原执行0/128，固定200Hz峰值50.29/22.14N。详情见docs/e003_sim2real_adapter_2026-09-20.md。

### What has been tried
真实关节release guard与encoder小修未通过被动耦合；在原PPO结构/奖励/门禁下120+600s有界适配教师，再以原学生管线120+600s适配encoder，按物理循环选优。部署入口/关节顺序/SDK接口不变，无privileged或人工阶段输入。软件/录像/包校验完成。

### Local conclusion
已达完整循环候选短试里程碑，尚未达到真实M24或严格受力验收。新包e003_matched_cycle_trial_20260920，不需要R；原d42/a7和旧部署保留。此更新替代下面历史条目的“教师也不胜任/模型未核对”的当前判断，不删除失败证据。

### Recommended next action
使用新包默认短时真实闭环测试，记录实际松开、回位重抓和螺母净转动及日志。依据真实反馈定位残余接触/执行误差；不要把release标签、教师成绩或单独张开当任务完成。

## 2026-09-20 — 人工张开诊断结束，自主sim2real仍为唯一交付目标

### Task background
用户确认显式张开可见，同时重申最终目标是sim2real复现仿真效果，不是让真机单独松开。人工接管仅是已完成的诊断。

### Current blocker
自主完整循环未迁移；近似真机执行模型下学生和冻结教师都存在回位失败。当前一阶过滤+PD模型的组合响应尚未用同一真机命令序列直接验证，不能据此提前确定唯一根因或直接重设计教师/学生。

### Evidence
两次新实机空载日志均200步，R step23/43后相关关节实际约−17.2°，最终主动关节目标误差<0.05°。这排除空载下“合理目标也无法张开”的解释，不排除接触问题。旧阶段提示不产生明确张开；自主修正执行评估仍0return，教师也未完整循环。见最新session_handoff_v2与explicit_open结果文档。

### What has been tried
入口契约修复、有界蒸馏、阶段oracle/人工提示、显式目标/恢复策略对照及新真机空载验证；没有将辅助控制计入自主成功。

### Local conclusion
停止扩展人工开手功能。先核对实际执行响应再确定教师或学生修正，不能用阶段标签、空载张开或更多同类蒸馏替代sim2real验收。本次没有进行兼容性重设计。

### Recommended next action
以新增真机日志的同一发送目标，在仿真中做执行响应对照，保持真实关节/发送目标契约、被动耦合与20Hz时序；根据模型有效性复核教师完整循环胜任性，再开展有依据的策略适配。无人工接管的接触闭环及既定门槛才是正式验收。

## 2026-09-20 — 自主部署仍阻塞，人工松开假设已有可测试入口

### Task background
用户要求推进到部署测试，并提出人为触发松开。当前PPO教师d42冻结、候选a7c00a27；不能把离线标签当物理事件。

### Current blocker
执行器包装的发送目标观测不一致已修复，但执行近似模型下自主return仍失败，教师给定真实阶段也失败。尚未建立此执行模型下的完整胜任监督；不能通过简单续蒸馏承诺自主循环。

### Evidence
修正后a7候选step131 workspace、0循环；教师step110 workspace。600s新蒸馏原执行128/126，而真机执行固定/扰动各0/128return。旧候选人工release20步及时触发恢复3轮，晚0.5/1秒均0完整循环；5N未过。详见docs/e003_manual_release_trial_2026-09-20.md。

### What has been tried
旧执行三倍时限无效；观测契约修复及实际runtime一致性；一次有界重训；阶段/阶段+clock oracle；10/20步release提示及触发延迟对照；同权重200Hz与20Hz录像；部署辅助入口/封包假设备验证。没有运行真机，没有改奖励或成功门禁。

### Local conclusion
阶段误判是瓶颈之一，人工提示可以帮助松开，但不能认定自主迁移或时机鲁棒已经解决。暂停重复同类蒸馏，不进行未经验证的教师/学生重设计；人工辅助诊断包明确保留失败项。

### Recommended next action
用户用outputs/local_m24_deployment/e003_manual_release_diagnostic_20260920做一次≤10秒人工触发松开诊断，按真实离触/关节目标/按键记录判断；之后才确定需要修正执行模型、建立胜任教师还是继续学生适配。

## 2026-09-17 — 真机不松开根因已定位：主动关节响应形态（更新下方2026-09-15条目）

### Task background
用户确认真机失败模式为：手指真正夹住螺母后立刻卡住，从未触发松开。继续执行因果诊断，不启动训练或硬件。

### Current blocker
根因已定位为学生阶段估计器对**主动关节响应形态**的依赖：它从仿真Kp2.4 PD的跟踪误差形态读取阶段推进证据，真机伺服响应形态不同（i1/i2跟踪更紧、i0更松）使推断崩溃为旋拧/回位跳转、无重抓。修正需要在蒸馏中加入执行模型变化（如PD滞后/伺服响应随机化）并以真机日志重放签名为验收——这超出入口小修，属训练目标变更，需用户授权后执行。

### Evidence
有接触卡死复核（outputs/local_pipeline_e003_jam_diagnostic_20260917）：螺母DOF摩擦0.2→2.0/20.0Nm完全锁死（0.01°）且夹持保持时，学生仍周期松开[76,51,30,43]/第16步，物理阻力单独不构成失败。离线反事实重放（同目录counterfactual_replay.json）：真机流修复index_3仍崩溃[90,19,107,0]；仿真流破坏index_3不影响循环——index_3双向排除；全通道零跟踪误差两流均坍缩纯旋拧[200,0,0,0]；仿真主动关节滞后均匀减半即复现真机签名[65,12,123,0]/36步无重抓。基线与2026-09-15接触组逐位一致（0rad）。

### What has been tried
被动耦合弹性近似（k2.4/4.8）、空拧（螺母移远1m）、有接触卡死（2/20Nm）三组物理单因素对照均未复现真机失败；index_3通道反事实双向排除；主动关节滞后剂量实验复现签名。全部只读/诊断，未改生产任务、权重、部署，未操作硬件。

### Local conclusion
真机不松开不是index_3耦合、螺母阻力或缺少接触造成，而是sim2real执行形态差异破坏了本体感受学生的阶段推断。当前部署学生不能通过入口级修补恢复实机循环。

### Recommended next action（2026-09-17第二次更新）
执行器随机化蒸馏已完成三轮并选出真机试运行候选a7c00a27（真机日志流签名首次完整循环通过；真机执行仿真drive/release可靠，旧学生drive 0/128）。剩余取舍：原执行扰动鲁棒108/128（旧124）、真机执行return撞PD节奏门禁。请用户三选一：(a)真机短时试运行(≤10s采集日志)；(b)补全完整验收打包；(c)第四轮针对性再训练。未决定前不重训不上真机。

### Superseded recommendation（2026-09-17更新）
部署侧响应匹配对照已完成：辨识真机执行器模型单独进环即复现卡住签名，而目标低通塑形三档全部恶化——部署侧滤波修复被否定。请用户授权**执行器随机化蒸馏**（学生侧执行混入一阶模型，参数在辨识区间随机：a∈[0.2,1]、延迟{0,1}步、vmax[0.5,1.2]、i3被动耦合；教师冻结；验收=真机日志流反事实重放恢复周期签名+原仿真五轮不退化）。可选：真机1-2分钟空载阶梯测试细化延迟/加速度参数（协议见docs/e003_actuator_identification_2026-09-17.md）。授权前不重训、不上真机。5N与力矩平衡目标保留。

## 2026-09-15 — 从动耦合单因素对照未复现首轮松开失败

用户授权继续因果诊断。手册确认普指DIP被动，仿真独立PD是执行模型错误，不能将此前DIP目标误差称堵转证据。前5条实机日志拟合二次耦合，第6条留出RMSE0.00524°。同一79610a9c学生/seed42/1env原配置本地CPU推理+GPU PhysX：独立DIP五轮；删独立PD并加双向弹性耦合k2.4与4.8，两组均4轮，第五轮重抓超时，但各完成5次松开/回位。首次松开完成24/25/26步，未复现真机首轮不分开。耦合误差中位0.554°/0.350°，P95仍6.8°/3.78°，不冒称精确硬件复现。

新增独立诊断评估脚本与日志拟合脚本，生产任务/权重/部署未改。三阶段180s独立timeout均exit0；两次启动配置失败单独保留，非物理失败。全部本轮Docker已退出；无训练或硬件执行。详细证据docs/e003_passive_coupling_diagnostic_2026-09-15.md及outputs/local_pipeline_e003_passive_diagnostic_20260915；AST、留出拟合及baseline与原208步五轮录像数值一致检查通过。结论为耦合影响可靠性，但尚不支持其为首轮不分开的主要原因。

唯一下一步：保留耦合诊断，结合实机主动关节响应和初始装夹/接触差异定位阶段推断偏离，不直接重训；装夹条件待用户补充。


## 2026-09-15 — E003实机松开失败，执行契约待核实

### Task background
用户报告部署student实机完全没有分开趋势，与CPU仿真录像不符。目标是定位失败，不启动硬件或训练。

### Current blocker
实机食指index_3反馈未跟随独立目标，并与index_2高度相关；耦合/SDK约束、机械接触阻挡或反馈换算尚未区分。模型阶段推断同时未形成完整循环。目前无法继续宣称实机循环可用。若确认被动耦合，需修正仿真执行模型并重新评估教师兼容性，超出入口小修。

### Evidence
读取交付目录6次非空真实运行日志，同一79610a9c权重；原模型离线重放所有下发目标最大误差1.20e-7rad。最新20260915T091216_063367Z：216步，实测20.055Hz；index_3反馈相对上一目标平均绝对误差35.13°、最大57.70°，48步上一目标为0时反馈仍39.21–41.76°。index_3对index_2反馈线性拟合斜率1.17926、截距0.02845rad、RMSE0.411°、相关系数0.999291；强烈提示耦合，但不足以单独确定硬件结构。

同一历史离线恢复的阶段argmax计数[86,23,107,0]（旋拧/松开/回位/重抓），第78步首次松开，随后反复跳转，未判重抓；这些是模型估计，不是实机真实接触标签。仿真录像评分阶段第14步松开、第24步回位、第32步重抓；评分阶段与模型估计不可当作同类准确率指标。

### What has been tried
只读检查日志、部署源码及本地SDK声明；离线重放，无SDK导入/硬件命令。原模型输出与日志一致，排除明显错权重/步幅/推理接口问题；SDK串口成功不证明每关节目标可执行。证据及可复跑脚本位于outputs/e003_real_diagnostic_20260915/{report.json,replay.py}。未以调大力/增益或阶段定时覆盖掩盖问题。

### Local conclusion
当前student实机循环失败。最突出的可测异常是index_3执行/反馈契约不匹配，阶段泛化失败可能为后果或独立问题，尚不确认为唯一根因。不是靠延长蒸馏就能假定解决。

### Recommended next action
先结合用户空载/夹持条件及DexH13实际版本的主动/被动关节映射核实index_3可控性，再决定修正执行模型或控制接口。遵循AGENTS.md中teacher-student compatibility非局部变化需写issue的边界，在核实前暂停训练及硬件试跑，不自行改关节映射。


## 2026-08-04 Resolved Escalation: M24 Pasini PPO Migration

- status: `resolved_authorized`
- resolved_on: `2026-08-04`
- resolution: the user explicitly authorized changing the diffusion branch
  objective to `XHandPasiniM24NutBolt -> PPO teacher` and updating governance.

### Task Background
- The user paused MuJoCo sim2sim work, saved commit `159d994` on the
  `sim2sim` branch, and switched the active worktree to `diffusion`.
- The requested next task is to migrate the M24x3, 160 mm nut-and-bolt PPO
  candidate from `sim2sim` into `diffusion` and prepare local PPO training.
- The M24 task uses the 16-DOF Paxini/Pasini hand and a user-confirmed keyboard
  init pose.

### Current Blocker
- The `diffusion` branch `AGENTS.md` defines
  `XHandHoraScrewDriver -> PPO teacher -> current student -> evaluation/export`
  as the canonical path.
- The same file requires codeagent to stop and write `codeagent_issue.md` when
  “Pasini or external repos become necessary for the current stage.”
- A correct M24 migration necessarily modifies the Pasini runtime, task map,
  task/train configs, and assets. It cannot be implemented as an asset-only
  copy while preserving the saved init-pose semantics.

### Evidence
- Source snapshot: `sim2sim` commit `159d994`, already pushed to
  `origin/sim2sim`.
- Required new M24 files include:
  - `assets/meshes/m24_hex_bolt_head_36x15.stl`
  - `assets/meshes/m24x3_bolt_thread_160.stl`
  - `assets/meshes/m24x3_hex_nut_36x21p5.stl`
  - `assets/screw/m24hex/0000_m24x3_160.{urdf,npy}`
  - `configs/task/XHandPasini{,M24}NutBolt.yaml`
  - `configs/train/XHandPasini{,M24}NutBolt.yaml`
  - M24 asset generator and keyboard-tuner launchers.
- Required shared runtime changes are visible in
  `diffusion..sim2sim`:
  - `dexscrew/tasks/__init__.py`: register `XHandPasiniNutBolt` and
    `XHandPasiniM24NutBolt`;
  - `dexscrew/tasks/xhand_pasini.py`: 123 changed lines so
    `asset.handInitPose`, `handRootPos`, and `handRootRPY` are consumed during
    actor creation and reset;
  - `scripts/tune_dexh13_lightbulb_initpose.py`: generic Paxini-compatible
    numeric keyboard controls.
- The active `diffusion` worktree was clean before this issue entry and is at
  commit `2864631`, synchronized with `origin/diffusion`.

### What Has Been Tried
- Updated and pushed the full sim2sim snapshot without losing moved DOTPG
  checkpoints.
- Switched to the clean `diffusion` branch and pulled it fast-forward.
- Read the branch-specific `AGENTS.md`, `PLANS_v2.md`, current session handoff,
  and stage acceptance summary.
- Audited the exact M24 files and shared runtime diff against `sim2sim`.
- Did not copy M24 files, modify Pasini runtime, or launch PPO after confirming
  the escalation boundary.

### Local Conclusion
- The migration is technically bounded and the source artifacts are preserved,
  but it changes the active diffusion stage from the canonical Hora student
  path to a new Pasini M24 PPO teacher task.
- This is a governance/plan transition, not a missing-asset or implementation
  blocker. Continuing without revising the active plan would violate the
  branch's explicit escalation rule.

### Recommended Next Action
1. Explicitly authorize a new training-stage objective on `diffusion` for
   `XHandPasiniM24NutBolt -> PPO teacher` and update the active plan/governance
   files accordingly.
2. Then selectively import only the M24 assets/configs, Pasini pose support,
   task registration, and tuner utilities from commit `159d994`; do not import
   `sim2sim/` or MuJoCo documentation.
3. Run compile/Hydra/one-env GPU PhysX checks, followed by a monitored
   8192-environment allocation probe on the local RTX 4080 SUPER before
   starting the long PPO run.

### Resolution Evidence
- `AGENTS.md` now names M24 Pasini PPO as the canonical path.
- `PLANS_m24_ppo.md` defines migration, smoke, capacity, and persistent-run
  milestones.
- The selective import contains no `sim2sim/` directory or MuJoCo files.
- Static checks and a one-environment Docker GPU PhysX smoke passed with 16
  hand DOFs, three object bodies, one nut DOF, finite reset state, and the
  user-saved M24 init pose.

## Status
- status_now: `open`
- opened_on: `2026-04-16`
- last_updated_on: `2026-04-17`
- trigger: `flow_after_v8_not_continue_worthy_under_current_scope`

## Task Background
- 当前执行计划已推进到：`PLANS_v8.md`。
- V8 的定位不是替代 `padapt`，而是：
  - 给 `FlowMatchingLatentStudent` 做一次 bounded recovery sprint；
  - 先判断它是否有资格继续进入扩时或 multiseed，而不是直接冲 accepted branch。
- 已完成：
  - `V8-M0` flow 工程闭环补丁（latent init / rollout helper / restore-save closure）
  - `V8-M1` 旧 flow artifact 的 `infer2 / infer4` eval-only probe
  - `V8-M2` 5 个 bounded single-seed fresh candidates
- 统一评测口径：`seed=42`, `steps=256`, `nominal + light_v2 + hard`。

## Current Blocker
- 在 V8 的 recovery sprint 与 bounded sweep 内，仍没有任何 flow candidate 达到“继续投入”的 single-seed continue gate：
  - `nominal_reward >= 1.95`
  - `light_v2_reward >= 1.70`
  - `hard_reward >= 1.55`
  - `hard_done <= 0.002372`
- 因此：
  - `V8-M2.5` 30min 扩时未触发；
  - `V8-M3` multiseed 未触发；
  - 当前范围内仍无法证明 Flow Matching diffusion 值得继续投入。

## Evidence
- `PLANS_v8.md`
- `docs/plansv8_final_verdict.md`
- `docs/stage_acceptance_summary.md`
- 关键结果（seed42, steps256）：
  - M1 eval-only probes:
    - `infer2`: nominal `1.805502`, light_v2 `1.681982`, hard `1.428555`
    - `infer4`: nominal `1.885794`, light_v2 `1.640372`, hard `1.189338`
  - M2 fresh candidates:
    - `zeroinit_bc12`: nominal `0.992769`, light_v2 `0.921589`, hard `1.020256`
    - `align2_rollout`: nominal `1.082436`, light_v2 `1.108353`, hard `0.750190`
    - `align2_anchor`: nominal `1.000670`, light_v2 `0.923009`, hard `0.953265`
    - `align2_bcheavy`: nominal `1.124580`, light_v2 `1.055061`, hard `1.053348`
    - `align4_rollout`: nominal `1.114895`, light_v2 `1.152107`, hard `1.079496`
- 参考 baseline：
  - current flow baseline:
    - nominal `1.782656`
    - light_v2 `1.587523`
    - hard `1.367413`
  - strongest accepted diffusion reference:
    - `V5.5 consistency_boundary_bc_tuned`

## What Has Been Tried
- V8 已尝试：
  - `flow_train_init_mode`
  - train-time rollout BC (`flow_train_align_infer + flow_rollout_bc_coef`)
  - `infer_steps=2 / 4`
  - heavier BC weighting
  - small base action anchor
  - restore/save closure for `sa_mean_std` and `agent_steps`
- 每个候选均完成必要评测，并记录 run_dir / ckpt sha1 / verdict 文档。

## Local Conclusion
- 在当前 V8 范围与预算下，Flow Matching 已完成 recovery sprint，但仍未证明自己具备继续投入的价值。
- 按 AGENTS escalation 边界，“diffusion 无法展现超越当前 student baseline 的价值”条件继续成立。
- 当前最合理的本地定位是：
  - `padapt` 继续作为主线 baseline；
  - `V5.5 consistency_boundary_bc_tuned` 保留为最强已接受 diffusion 参考；
  - `V8` 作为 flow recovery sprint 的负结果收口。

## Recommended Next Action
1. 治理层确认：接受 `PLANS_v8 completed_conclude`，冻结当前 flow matching 扩张线。
2. 若继续：必须新开计划，并明确授权**超出 V8 边界**的新假设；仅重复当前 bounded sweep、继续加时长、或在 `infer2/infer4 + rollout BC` family 内换小系数，不建议再做。
3. 若不继续：保留 `padapt` 为主线，`V5.5 consistency_boundary_bc_tuned` 为 diffusion 次优参考，`V8` 作为论文中的负结果与边界说明。
