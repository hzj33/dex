# E003 sim2real执行匹配与策略适配（2026-09-20）

用户目标：真机获得仿真的完整拧转效果，允许直接部署基础上的小幅辅助。不能把单独张开、阶段标签或仅有教师成功当最终交付；也不额外强加“禁止一切部署补偿”的限制。

## 同指令实机/仿真对照

两次新真实空载日志074626_718191Z、074644_976818Z作为独立回放数据，发送同样200步目标。仿真仅初始设置到日志首帧实测角，之后完全由物理推进；螺母移远1m并逐步断言接触力0。关闭自动重置仅用于此空载响应诊断，不作为任务验收。原执行与已辨识执行模型都加入食指DIP被动耦合k4.8/d0.1，未用新日志重新拟合参数。

7主动关节合并RMSE，单位度：

| 日志 | 原执行，200步 | 已辨识执行，200步 | 原执行，前80步 | 已辨识执行，前80步 |
|---|---:|---:|---:|---:|
| step23张开 | 1.787 | 0.332 | 2.825 | 0.524 |
| step43张开 | 2.561 | 0.683 | 4.050 | 1.080 |

已辨识执行瞬态最大误差4.19°/6.54°，并非精确模型。结果支持保留近似执行模型开展诊断；只约束空载执行，不证明接触力/接触中耦合刚度已校准。产物outputs/local_pipeline_m24_e003_command_match_20260920，各.command.sh及status.json记录有界240s命令和起止/退出状态；有效组exit0。新增evaluate_m24_command_replay.py和analyze_m24_command_replay.py可复现。

## 部署小修对照，未进入交付

共享ReleaseGuard只读真实index_0/index_1，迟滞触发20步release阶段提示，保留原actor、latent、真实反馈及发送目标历史。假设是纠正学生迟到的阶段提示；不读取螺母角/接触/仿真事件。阈值index_0≤−0.10rad、index_1≥0.90rad、回程index_0≥0.10重置触发资格。单独角阈值会在第二次drive完成前触发，故加入屈曲门槛；该调整只使用seed42开发轨迹。

- 原近似执行但DIP独立驱动：固定姿态3完整循环；seed43±1°/128组，循环数0/1/2/3/4/5分别1/2/36/51/28/10，首次松开/回位128，至少一轮127、至少两轮125。五轮只有10，不能说完整验收通过。
- 加DIP被动耦合：固定姿态1轮；同seed43/128组，循环数0/1/2/3/4/5分别86/33/5/3/1/0。旧卡住日志也经常无法触发此阈值。部署小修未达到真机候选要求，不打包。
- 阈值−0.05/−0.15、脉冲10步、屈曲0.95/1.0及回程重新初始化估计历史的有限对照均未完成五轮。不要重复无明确修正假设的扫描。
- 仅变换encoder的主动关节跟踪误差特征，actor仍读取真实关节/目标：倍率0.5/1/1.5/2在被动执行下分别0/0/1/0完整循环；1.5叠加release guard仍0。不是有效部署补偿。默认倍率1直接调用原预测，保持精确原行为。

产物outputs/local_pipeline_m24_e003_release_guard_20260920与...encoder_compensation_20260920，所有物理任务成功/工作空间/阶段门禁保留。控制器与评估脚本仅研究用，未接入用户部署包。历史summary的completed_two_cycles字段实际上按required_cycles=5统计，本文均使用first_episodes中的真实cycles或termination5计数。

## 有界原策略适配，原教师文件保留

小修不能通过带被动耦合的验证，因此检验在同一PPO结构、奖励、动作接口和原任务门禁下，仅适配执行动力学能否恢复胜任性。新候选从d42教师warm start，原教师文件不改。不是重新定义任务或删除失败门禁。

云实例pro-786095bb6e03处于shutdown，开机失败InternalError“当前算力规格暂无库存”，request_id d3ac458b45cd90bb09990233c4414a2c。回执保存到outputs/local_pipeline_m24_e003_policy_adapter_20260920/cloud_power_on.json，未读到运行中的云仓库交接/未同步云。用户已授权无库存本地回退。

本地RTX3060/12GiB：teacher1024env、minibatch12288，实测显存约4.7GiB、约3.5k—4.5k env steps/s；同时进行64env学生管线smoke时约6.7GiB。每阶段独立timeout及持久日志。

- teacher smoke：120s，exit124且正常完成当前更新保存；36轮更新、454656 agent steps。冻结检查点smoke/stage1_nn/last.pth，SHA68b9ac7d7d73e24e015350dd97bd9cd50373501d31abe788e766505d7b225315。
- 相同seed43±1°/128组、辨识执行+被动耦合：原教师release128/return0/五轮0；smoke教师release128/return112/regrasp112/五轮47。smoke的0/1/2/3/4/5轮计数16/1/3/45/16/47。**这是教师结果，使用真实仿真阶段和privileged输入，不能直接上机。**
- teacher formal：已启动600s独立上限，从smoke续训，周期快照120s。运行路径...policy_adapter_20260920/formal，runner/status/command均保存；完成后按物理循环而非训练奖励选择。
- student pipeline smoke：120s，使用smoke教师、64env、相同执行+被动耦合，仅warm start原a7的encoder，固定新的teacher actor。此阶段用于验证兼容管线，不能把它当最终学生通过。最终学生必须重新进行无privileged输入的闭环测试。

## 代码与兼容性

新增train_m24_e003_real_teacher.py，复用现有有界PPO入口；新增共享e003_passive_coupling.py，原被动评估改为调用它。同指令200步物理重放与抽取前逐位一致（最大误差0）。real_exec评估可选相同被动耦合，默认行为保留。

train_m24_e003_deploy_student.py保留默认d42指纹和完整warm-start检查，仅新增显式指定teacher SHA、encoder-only warm start和被动执行配置；新的teacher actor/normalizer由指定checkpoint装载，不能被旧student覆盖。软件验证脚本同样允许显式teacher路径+指纹，默认不变。没有改变PPO/DAgger算法、损失权重、模型维度或20Hz/0.05rad部署契约。

## 执行收尾：完整循环学生短试候选已交付

上面的运行中状态已结束。teacher formal独立600s（08:15:16—08:25:17 UTC），student formal独立600s（08:28:16—08:38:17 UTC），均exit124正常保存；所有起止/精确命令/status在同pipeline。学生正式阶段6465向量步/413760样本，teacher冻结，新teacher actor与归一化没有被旧encoder warm start覆盖。没有运行真实硬件，云仍无库存/无任务。

### 按物理表现选择教师与学生

teacher epoch74在同seed43±1°/128组五轮115，其余13组四轮；last五轮94、best reward五轮113且14组0轮，因此选择epoch74而非最后/最高奖励。冻结副本selected_teacher/matched_teacher.pth，SHAd98f2ce66923369150aad1e97831afe032b7669984ba327848b3fc164a71a745。seed44复核五轮123/128；固定耦合k4.8与k2.4均五轮，200Hz峰值分别50.87/20.12N和54.05/23.57N，不是低力合格教师。

student smoke（120s）只使用本体观测，seed43的128组全至少两轮，五轮4；不是最终选型。其encoder作为warm start，在新d98教师/相同执行和被动耦合下进行600s有界蒸馏。只训练encoder，奖励/损失/阶段门禁/SDK映射不改。

| 学生检查点 | seed43五轮/128 | 最少完整循环 | 选择 |
|---|---:|---:|---|
| candidate_0001462 | 122 | 4 | 短试候选 |
| candidate_0002984 | 110 | 4 | 不选 |
| formal/student_last（6465步） | 107 | 1 | 不选 |

选型文件student_selection.json。候选SHA **a8196702d6664e31d65ce921695008357a0fb3e28c3683f185defdcf12079c3f**。没有使用未评测的后期快照，也没有按损失/训练时长自动覆盖。

### 同一待部署学生的闭环验证

| 执行模型/扰动 | 五轮 | 最少完整循环 | 螺母净转角中位数 |
|---|---:|---:|---:|
| 辨识执行+被动DIP，固定姿态，1env | 1/1 | 5 | 178.75° |
| 同模型，seed43±1° | 122/128 | 4 | 183.20° |
| 同模型，seed44±1° | 117/128 | 4 | 184.10° |
| 同模型，seed45±1° | 112/128 | 4 | 183.27° |
| 同模型，seed46±1° | 116/128 | 4 | 184.17° |
| 原执行（无辨识/被动包装），seed43±1° | 0/128 | 0 | 83.41° |

匹配执行四组共467/512五轮，512组全至少四轮；所有阶段来自模型自己的关节/发送目标历史推断，仿真阶段仅用于评分。固定姿态首轮重抓step45、五轮step251；252帧20Hz原生录像，角度/阶段/循环/回位误差/动作/力与独立数值评估逐位一致。动作/目标观测契约误差均0。旧a7在修正匹配模型的0完整循环，不能与旧观测错误版本计数混用。

**失败项不隐藏**：原执行五轮0/128（0/1/2轮分别68/51/9），不是原仿真与辨识执行兼容的鲁棒模型；空载响应近似吻合不能保证接触响应匹配。固定姿态200Hz食指/拇指法向力峰值50.29/22.14N，5N未过，完整逐组110%受力保持未验收。接触力没有真机校准；这些是模拟力，不是实机力预测。未证明无限循环、真实螺母转动或拧紧扭矩。严格转移验收保持false。

### 交付与验证

新包：outputs/local_m24_deployment/e003_matched_cycle_trial_20260920/，入口run_cycle_trial.sh，另有同名zip（SHAd93c5a4f1eaf35e11f77d7ebb4cfe78baad89cff4b8cf0692ae6428adc4c7646）。包内student.pt、原SDK入口、README、manifest、软件/受力/录像证据齐全；不是原full-acceptance packager的通过包，明确matched_execution_cycle_trial;strict_transfer_not_passed。原d42/a7、旧部署包保留。

运行：`bash outputs/local_m24_deployment/e003_matched_cycle_trial_20260920/run_cycle_trial.sh`。不需R，无阶段脉冲/显式张开/encoder小修；默认20Hz、0.05rad/步、200步约10s、右手/dev/ttyUSB_right、禁用机械臂键盘。默认短试结束关闭/禁用；Ctrl+C原停止流程会先发零位再禁用（入口RETURN_TO_ZERO_ON_STOP=True），详见README。

验证命令：verify_m24_e003_deploy_artifact.py指定d98 teacher及SHA，实际a819模型CPU native/JIT/仿真/部署200步逐位一致，privileged输入NaN，外部cwd假设备20步，CPU中位0.503ms/p990.847ms。封包后再从/运行run_cycle_trial.sh --dry-run --max-policy-steps 20，通过；全部文件SHA及zip CRC通过。compile/diff检查通过。扩展record_m24_e003_real_exec.py的SHA/耦合元数据，package_m24_e003_matched_trial.py明确保留未验收状态；未改实际部署控制入口。

**唯一下一步**：用户用新包进行默认短时真实闭环测试，判据是实际松开→回位→重抓并继续带动螺母净拧转，保存policy_steps.jsonl/run_metadata.json；仅关节摆动或release标签不算。根据该真实反馈决定剩余接触/执行补偿，不继续盲目扩时训练，也不重复交付单独开手练习。
