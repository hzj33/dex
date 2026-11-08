# E003 人工触发松开部署诊断（2026-09-20）

后续实机反馈：用户已进行200步空拧，step46的20步阶段提示确实生效，但没有观察到明显张开。详见[e003_manual_release_air_result_2026-09-20.md](e003_manual_release_air_result_2026-09-20.md)。下文记录交付时的仿真与软件验证，不代表实机松开通过。

## 当前可交付范围

用户提出真机夹紧后人为触发策略松开。现已提供可执行的人工辅助短时诊断包，目标是验证一次真实松开；不是稳定自主拧M24或5N验收。没有运行真机。

包：outputs/local_m24_deployment/e003_manual_release_diagnostic_20260920/，同名.zip。模型仍candidate_0040857，SHA a7c00a275002946a7c4a40311d269a801c8676ba64236b9a2db482bd98819913。压缩包SHA cd038df66c65efc534acfebadbec8d63fdfa5b35a61f7a86236b09829325ae5c。

## 实现与对照

直接指定actor的release阶段，保留模型预测clock/latent和真实关节、上一发送目标、历史。默认入口不变，--manual-release才开启；交互终端r+回车启动20步（约1秒）提示，随后恢复模型阶段。脉冲中重复触发不延长。SDK关节限位、20Hz、0.05rad/步、中指无名指动作屏蔽及原停止行为不变。main入口只在该模式加辅助日志，不改变legacy源码。此实现与伪造关节角不同：不会把假角度当真实状态输入动作网络。

物理诊断仍是含接触的近似执行模型，并非实机校准的力模型。每次仿真drive完成事件代替理想人工按键时刻；除了延迟对照，该时刻不是随机人类反应。

| 同一a7c00a27、一环境CPU策略 | 完整循环 | 终止 |
|---|---:|---|
| 原始自主（修正执行目标观测） | 0 | workspace，step131 |
| 全程oracle阶段，保留预测clock/latent | 2 | workspace，step184 |
| 全程oracle阶段+clock，保留预测latent | 1 | workspace，step179 |
| 每次drive后release提示10步 | 1 | workspace，step144 |
| 每次drive后release提示20步 | 3 | 第4次drive超时，step415 |
| 20步提示，触发延迟10步 | 0 | workspace，step144；首个release完成 |
| 20步提示，触发延迟20步 | 0 | workspace，step148；首个release完成 |

因此阶段信息是瓶颈之一，但及时触发也只恢复有限循环，人工时机鲁棒性失败。20步提示的前10秒完成2轮；第三轮完成step255。200Hz食指/拇指峰值65.9929/29.7574N，whole-finger与tip同值，5N明确失败。不是硬件力预测，不把仿真力数值解释成硬件安全限幅。

正确观测契约重训后的student_last：原执行固定128、s43=126；真机执行模型固定/扰动各0return。新模型加20步release或全程oracle阶段仍0循环（均step112 workspace）。不以训练更久为由替换人工诊断包里的旧候选。

## 验证与证据

- scripts/test_m24_e003_manual_release.py：真实a7权重，100步不触发与旧路径逐位一致；20步触发+期间重复按键+恢复推断；关节反馈不变；步幅/屏蔽不变；原入口22步假设备+按键+日志通过。
- 同一辅助物理轨迹416步以控制端点关节反馈重放，动作与目标最大误差0。最初误用physics substep0作为策略观测产生差异，改为真实观测时序（初始pose、随后前一控制端点）即逐位一致；未改变模型弥合差异。
- scripts/verify_m24_e003_deploy_artifact.py：CPU native/JIT精确，200步部署接口一致，不可用输入污染、异cwd20步默认包假设备通过。CPU中位0.405ms、P99 1.361ms（默认模式）。
- scripts/record_m24_e003_phase_intervention.py：416帧20Hz，角度/循环/回位误差与数值轨迹逐位一致。已查看12帧接触图，能见旋拧、松开、回位/重抓，以及末段失效；录像叠字明确辅助触发。
- scripts/package_m24_e003_manual_release.py：核对模型/入口/helper/legacy/software/录像SHA和判据，manifest明确自主五轮/时机/5N均未通过。新包另以绝对路径从cwd=/运行22步假设备，20步提示和日志验证，zip CRC与所有manifest文件SHA验证。
- 所有命令/原始JSON/200HzNPZ/视频/未成功启动记录位于outputs/local_pipeline_m24_e003_return_audit_20260920。没有硬件动作。云无库存后本地训练/诊断已全部结束。

## 使用与单一下一步

在包目录交互终端执行bash run_manual_release.sh；入口右手/dev/ttyUSB_right，禁用机械臂键盘，最多200策略步。r+回车触发一次release提示。若未真正离触或动作异常，结束该短试，不靠持续重复触发追求完整循环。Ctrl+C默认发全关节零位后禁用，用户可按原习惯编辑RETURN_TO_ZERO_ON_STOP。

下一步：用户进行一次人工提示松开的≤10秒实机诊断，保留policy_steps.jsonl、manual_release.jsonl、run_metadata.json及真实动作观察，用实际离触/回位判断效果。自主版本仍不合格；进一步学习需解决执行模型下教师回位胜任性，不能重复旧条件继续堆蒸馏。
