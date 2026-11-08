## 2026-10-08 — 最新里程碑：回位修复两次真机功能复测成功

用户报告新候选部署两次效果成功。独立包e003_return_progress_recovery_20261008真实日志880步/43.90s与572步/28.52s，a61/0.05rad/20Hz确认；动作和前后目标逐步重放误差0。结束前仍持续关节运动与阶段切换，最长连续return仅15/22步，未再现旧195步静止陷阱。记为本次功能修复两次复测通过；原权重保持，仅预测return时修正actor控制进度，没有重新训练教师。详见docs/e003_recovery_hardware_review_2026-10-08.md。

唯一下一步：固定现有a61包与参数，后续同装夹收集较长日志/录像核对真实螺母净进展和重复性；真实力矩/长期成功率未量化，仿真5N仍未过，无新增失败证据不继续改策略或重训。下方“待真机复测”为此前交付时记录。

## 2026-10-08 — 当前修复交付：回位控制进度候选，待真机复测

原20260920模型能循环但有回位陷阱，当前目标是恢复长程旋拧→松开→回位→重抓。长程仿真跨圈回弹已在独立入口修正，不放宽物理门禁。最终保留原a819教师/学生参数，仅将学生预测return时的actor控制clock固定0：固定20轮1/1，s43扰动12/16、独立s44 56/64，优于原5/16、42/64。自动时机补偿及补训快照未优于此候选，不采用。

独立包outputs/local_m24_deployment/e003_return_progress_recovery_20261008，run_recovery_trial.sh默认50秒，run_deploy.sh持续；旧包/20Hz/0.05rad/SDK/用户当前退出复位True保留。400步软件一致与假设备启动/连续验证通过，真实硬件尚未测；5N仍未过。详见docs/e003_long_cycle_recovery_2026-10-08.md及最新handoff。

唯一下一步：新包真机复测并采集logs，以实际回位、重抓和螺母净进展判断；不只数标签/张合，不未经新证据继续训练教师。

## 2026-09-20 — 当前交付：完整持续部署包

用户明确要求完整部署。outputs/local_m24_deployment/e003_matched_full_20260920/run_deploy.sh使用已反馈好的a819模型，默认持续到Ctrl+C，取消200步上限；Ctrl+C/异常不复位，禁用电机并断开。原控制/SDK与20Hz/0.05rad不变。假设备210步后真实SIGINT验证零位发送0次且关闭设备，外部cwd限步20自检、SHA/zipCRC通过。详见最新handoff与包内README。下一步使用完整包收集真实多轮稳定性及物体净进展；长期可靠性/拧紧扭矩/5N仍不能宣称通过。

## 2026-09-20 — 已有里程碑：首次真实周期运动正反馈，固定候选核对任务与重复性

用户最新反馈效果好；16:50真实200步日志已核对a819、20Hz、全序列动作/目标精确重放，实测出现四次明显往返回程，末段继续运动。不能从关节/阶段标签直接确认螺母净进展；带载情况已补问。较早16:44有效运行有后段停滞，不能忽略。详细报告docs/e003_first_matched_hardware_review_2026-09-20.md。

下一步固定a819/现有控制，确认真实M24松开重抓后继续同向转动，再在相同装夹核对重复性并保留录像/日志；不继续改策略或用单独张开替代。5N/严格转移未过状态不变。

## 2026-09-20 — 已交付里程碑：完整循环策略已封包，等待真实闭环短试

匹配执行适配完成，teacher d98/学生a819选优见最新handoff与docs/e003_sim2real_adapter_2026-09-20.md。实际待部署student在辨识执行+被动DIP下固定五轮1/1，四组±1°五轮467/512，全部至少四轮；无需R或阶段oracle。原执行五轮0/128、200Hz峰值50.29/22.14N，严格转移/5N仍未通过，不能改写为正式合格。

新包outputs/local_m24_deployment/e003_matched_cycle_trial_20260920/run_cycle_trial.sh，默认200步约10秒；模型/软件一致性、独立扰动、200Hz审计、20Hz同权重录像与包校验完成。实际真机尚未验证。下一步只做真实完整循环短试并采集日志，以物理松开/回位/重抓/螺母净进展判断；不是再练一次单独张开。先获得此证据再决定是否需要用户允许的小幅部署辅助，不盲目续训。

## 2026-09-20 — 已完成：匹配执行后的有界策略适配（下为当时启动记录）

同指令空载回放支持现有辨识执行模型（主动关节RMSE0.332/0.683° vs 原执行1.787/2.561°）。关节触发提示在独立DIP模型恢复部分循环，但被动DIP与真机旧日志检查不通过；encoder特征小修也未完整循环，不交付。小幅辅助仍获用户授权，不把失败方案继续包装。

原PPO架构/奖励/物理门禁不变，在辨识执行+被动DIP上120s适配已将seed43±1°的五轮从0/128提升到47/128、回位0→112。这是教师胜任性进展，不能直接部署。600s teacher formal及120s学生兼容管线smoke正在本地有界运行，云无库存回执已记录；见最新handoff与docs/e003_sim2real_adapter_2026-09-20.md。下一里程碑：按物理结果选择teacher，再让仅本体观测student恢复完整循环，并完成部署验证。原d42/a7文件不覆盖。

## 2026-09-20 — 用户明确允许小幅部署补偿，完整任务效果为准

用户要求继续解决sim2real不松开，允许在直接部署基础上加小幅辅助。下方“无外部阶段/预设轨迹”的一刀切限制是此前代理自行加严，现撤销。允许有界、可解释的执行补偿或松开触发/衔接修正，但必须验证能恢复原策略后续回位、重抓与持续拧转，不能以独立开手代替任务完成。模型/SDK默认入口保留；所有辅助如实标注，既定物理成功与受力指标保持。

当前执行：已有两次实机空载日志与原仿真/近似真机执行模型同指令回放，先验证响应差异，再优先检查可逆部署侧补偿。无需再向用户确认已授权的本地分析/小修或必要云训练。

## 2026-09-20 — 用户纠正目标：恢复自主sim2real主线，结束人工张开功能扩展

最终目标始终是策略在真实M24任务上复现仿真的自主旋拧→松开→回位→重抓循环。R阶段提示和显式张开仅是诊断，不是交付目标或成功替代物。用户已确认两次空载显式张开可见，相关关节实测约−17.2°，末端主动关节目标误差<0.05°；诊断子目标结束，不再开发人工张开/定时状态机以代替自主策略。

**当前唯一下一步**：用同一发送目标序列，对比新增真机日志与仿真执行响应（包含延迟、速度限制、被动耦合、真实发送目标观测），验证或修正近似执行模型。此前一阶辨识参数外加PD的组合尚不能当作已校准的真机动力学；“能复现卡住”不够证明模型有效。空载日志只约束空载执行，不能据此宣称接触已校准。

随后在经核对的执行模型中复核教师完整循环胜任性：若教师不胜任，先定位并修复/适配教师侧局部问题，再建立有效学生监督；若教师通过而学生失败，则聚焦阶段/latent推断和分布适配。遵守现有兼容性升级边界，不在未验证执行模型上无限重复原蒸馏。

正式验收：同一待部署策略无R/外部阶段/预设开手轨迹，在仿真及真机有接触闭环中自主完成完整循环，并核对螺母实际净进展、回位/重抓及既定扰动/受力门槛；阶段预测标签、离线重放、单次开手均不能替代。云训练授权与无资源本地回退保留，只有训练假设与监督条件明确后启动。

## 2026-09-20 — 显式张开空载测试可执行

共享有界关节目标控制器与新包e003_explicit_open_air_20260920已验证。R基于真实角度减少index_1/thumb_2各0.30rad，限0.025rad/步，随后保持而不自动恢复。两种接触执行模型能离触，但恢复策略仍workspace/0循环、200Hz 5N未过；空载代理部分起点有短暂间距缩小。因此交付范围仅空载张开测试，下一步用实机反馈确认方向和幅度。详见docs/e003_explicit_open_result_2026-09-20.md，训练/模型/门禁不改。

## 2026-09-20 — 实机空拧反馈：阶段提示不足以保证张开

用户已完成200步空拧，step46—65的release覆盖确实有效，但未见明显张开。精确重放与同状态敏感性分析见docs/e003_manual_release_air_result_2026-09-20.md；R只改阶段，预测clock/latent与历史仍控制动作。当前优先项改为基于真实状态离线验证有界显式张开及返回策略的衔接；不把阶段标签计数当物理松开，不未经验证延长脉冲或重复同类训练。旧诊断包仍保留作证据，不能标为实机通过。

## 2026-09-20 — 可进行人工触发松开的短时部署诊断

人工辅助包已准备并验证，详见docs/e003_manual_release_trial_2026-09-20.md；不是自主/5N合格。正确观测契约的有界蒸馏已结束，原执行128/126，真机执行近似模型两组0return，不续同类训练。下一步用户短试≤10s、r+回车release提示并采集真实反馈；阶段提示不伪造关节信息。后续必须依据实机离触和循环，不能以阶段标签验收。

## 2026-09-20 — 当前优先项：修复发送目标观测契约后的闭环验收

用户授权继续至可部署测试，云无资源可本地。AR包装的滤波cur_targets误入观测已修复；原a7c00a27在修正执行模型中仍0循环。先完成一次120s smoke+600s formal的student-only对照，原d42教师/权重/门禁冻结，不以离线阶段计数充当实机事件。云无库存回执与本地预算已记录。详细假设/停机判定docs/e003_return_contract_fix_2026-09-20.md、最新handoff。此前9月17日“第19步松开/第23步重抓/248步终止”的摘要由9月20日原始数据复核覆盖，不继续引用。



## 2026-09-17 — AR蒸馏R3完成：真机日志流签名全通，选出真机试运行候选a7c00a27

R3（warm-start R2、锚定0.5、42072步/269万样本）完成：真机日志流离线签名**首次完整循环通过**（real_raw [89,19,69,23]/松开19步/重抓23步；hybrid [40,28,112,20]亦过）；原执行仿真固定128/128；真机执行仿真drive 126-128、release 121-128（旧学生drive 0/128）。30个周期快照扫描命中0039237与0040857双通过，选**0040857**（s43更高108 vs 105、净转角更大）为hardware_release_trial_candidate，最小交付包outputs/local_m24_deployment/e003_ar_student_candidate_20260917（SHA a7c00a275002946a7c4a40311d269a801c8676ba64236b9a2db482bd98819913，含README/验收证据）。

如实标注的取舍/未过项：(1)原执行±1°扰动五轮108/128低于旧学生124（鲁棒性换执行形态覆盖）；(2)真机执行仿真return/regrasp 0-43/0-16、五轮完成0——return撞按PD节奏整定的阶段截止/工作空间门禁（中位248步早终止）；(3)未做s44-46/200Hz受力/录像/完整package打包。完整证据docs/e003_ar_student_distillation_2026-09-17.md与outputs/local_pipeline_m24_e003_ar_student_r3_20260917。

唯一下一步（用户三选一）：(a)真机短时试运行本候选（≤10s采集日志，判据=周期性松开）；(b)补全完整验收打包（s44-46+200Hz+录像+package脚本）；(c)第四轮针对return门禁取舍再训练。未选前不重训不上真机；5N与力矩平衡保留。

## 2026-09-17 — 执行器随机化蒸馏进行中：R2已打通真机执行四阶段，R3训练中

用户要求"最推荐方案让真机学会松开"并授权执行器随机化蒸馏；云端两次开机无库存（request 0fcc3612/f5d73cc6），按用户指示本地执行。新增默认关闭的逐关节执行器随机化（dexscrew/tasks/e003_actuator_randomization.py，pre_physics_step一阶预滤波+延迟+限速+Kp倍率，prev_targets保持未滤波部署语义；契约测试全过）与真机执行仿真评估入口（evaluate_m24_e003_real_exec.py：全部env=辨识参数+接触物理）。教师冻结，不改奖励/门禁/物理。

R1（逐env统一，1800s）：仿真胜任保持（固定128/128、s43=124/128），但真机签名仍无重抓。R2（逐关节+30%真机锚定，warm-start R1）：仿真127/128；真机日志流松开恢复到第30步且充分出现（旧77步崩溃）；**真机执行仿真下旧学生drive 0/128（复现真机卡住，校准测试有效），R2 drive 128、release/return/regrasp 91/91/91**——四阶段全通，剩余失败为慢执行撞门禁（工作空间57、drive超时40，中位248步早终止）。R3（warm-start R2、锚定0.5、3600s）训练中，outputs/local_pipeline_m24_e003_ar_student_r3_20260917（nohup，runner.log），后续自动跑原执行双评估+真机执行双评估+离线签名验收。

详细方法/结果表/预声明验收标准见docs/e003_ar_student_distillation_2026-09-17.md。代码已提交6934e21。

唯一下一步：读R3结果按验收标准判定；通过则整理部署候选并建议短时真机试运行，未通过则诊断剩余失败模式再定下一轮。不放宽门禁，不上真机除非验收通过。

## 2026-09-17 — 真机执行器辨识完成，部署侧滤波匹配被否定，修正路线已具体化

用户选择"先提供真机伺服PD/响应参数，做部署侧响应匹配对照"。调研+SDK手册确认：DexH13位置模式是梯形速度规划点到点运动（手册3.5.1），无速度查询接口，经典PD增益不可读；标准做法是日志黑盒辨识（FOPDT）与执行器网络（Hwangbo 2019）。从既有6条真机日志离线辨识（无需新真机数据）：i0/i1/t0/t1每步增益0.22（τ≈0.2s）+50ms延迟+食指vmax0.5-0.65rad/s，远端i2/t3近似瞬时（a0.94/0.81），i3拟合失败佐证被动；全部R²≥0.92。仿真流同模型R²≈0——仿真反馈是PD+接触交互形态，非一阶执行器，两者本质不同。

混合重放对照：辨识真机执行器模型单独进估计闭环即复现卡住签名[110,32,58,0]/87步无重抓（真机[86,23,107,0]/78步），执行形态是充分原因；目标低通塑形τ=0.05/0.1/0.2全部恶化为纯旋拧——部署侧滤波修复被证据否定（真机缺的不是滞后而是训练分布的接触交互形态）。用户假说"真机关节角未达松开触发条件"获定量支持（松开形态证据推迟至87步且永不完整）。

有证据支持的修正路线（待授权）：执行器随机化蒸馏——学生侧执行混入一阶模型，参数在辨识区间随机（a∈[0.2,1]、延迟{0,1}步、vmax[0.5,1.2]、i3被动耦合），验收=真机日志流反事实重放恢复周期签名+原仿真五轮不退化；教师冻结。可选真机1-2分钟阶梯测试细化延迟/加速度（协议见docs/e003_actuator_identification_2026-09-17.md）。

产物outputs/local_pipeline_e003_jam_diagnostic_20260917/actuator_identification.json与scripts/analyze_m24_e003_actuator_identify.py；详细docs/e003_actuator_identification_2026-09-17.md。未训练、未操作硬件、云未启用。

唯一下一步：用户授权执行器随机化蒸馏（或先做可选真机阶梯测试细化参数）；授权前不重训不上真机。5N与力矩平衡保留。

## 2026-09-17 — 有接触卡死复核与反事实重放定位真机不松开根因

用户确认真机为真正夹持后立刻卡住、从未触发松开。据此执行既定"有接触复核"：同一79610a9c/seed42/1env/k4.8耦合，`create_actor`后把螺母actor DOF摩擦0.2→2.0/20.0Nm（首次`set_asset_dof_properties`API不存在失败，日志保留；改用生产同款`set_actor_dof_properties`）。基线与2026-09-15接触组200步逐位一致（0rad）。两档卡死螺母0.01°、夹持保持（147/200步>1N），但学生仍[76,51,30,43]周期循环、第16步估计松开、指尖张开至103mm——物理阻力单独不构成真机失败。

新增离线反事实重放（冻结TorchScript开环、自生成目标回填）：real_raw精确复现[86,23,107,0]/77步；真机修好index_3仍崩溃[90,19,107,0]/78步；仿真流改i3不破坏循环——index_3双向排除。全通道零误差两流均坍缩[200,0,0,0]纯旋拧；主动关节滞后减半复现真机签名[65,12,123,0]/36步无重抓，缩至1/4过度坍缩。结论：根因是**主动关节响应形态差异**——阶段估计器从仿真Kp2.4 PD跟踪误差形态读取阶段证据，真机伺服响应形态（i1/i2更紧、i0更松）使推断崩溃为旋拧/回位跳转。与index_3、螺母阻力、接触与否均无关（单因素对照全部排除）。

未训练、未操作硬件、云未启用；产物outputs/local_pipeline_e003_jam_diagnostic_20260917与脚本evaluate_m24_e003_jam_diagnostic.py/analyze_m24_e003_counterfactual_replay.py，详细docs/e003_jam_contact_diagnostic_2026-09-17.md。

唯一下一步：用户决定是否授权"执行模型随机化蒸馏+真机日志重放签名验收"的学生修正路线（或提供真机伺服参数先做匹配对照）；授权前不重训、不上真机。5N与力矩平衡保留。

## 2026-09-15 — 空拧对照复现无松开模式

用户询问无螺母接触时是否还会张开。保留k4.8从动耦合/同一79610a9c/seed42/1env，匹配reset后将对象沿Y移远1m；两组均抑制自动reset，仅诊断200步，不更改学生输入。原位接触阶段估计72/51/31/46，指尖参考间距53.1–110.7mm；移远后两指螺母法向力全0，阶段200/0/0/0，间距由52.8降到约33.7mm，没有周期大幅张开。不能保证真机空拧复现有接触视频，也不在原装夹条件未确认前声称根因解决。

新增独立air诊断脚本，生产任务/部署/权重不改。两阶段本地RTX3060、各timeout180s均exit0，无训练/硬件启动；原位组前200步关节与前次k4.8逐位一致（0rad差）。AST/无接触断言/图检查通过，容器已退出。产物outputs/local_pipeline_e003_air_diagnostic_20260915；详细docs/e003_air_diagnostic_2026-09-15.md。初始化共用原reset，弹性耦合非硬件标定；指尖参考距离不是表面间隙。

唯一下一步：待用户确认此前是真正夹持还是空拧，再决定有接触复核或主动关节响应诊断。已通过异步问题询问，未擅自认为原试验空载。最新实机有松开/回位估计，与此次全旋拧不完全一致。

## 2026-09-15 — 从动耦合单因素对照未复现首轮松开失败

用户授权继续因果诊断。手册确认普指DIP被动，仿真独立PD是执行模型错误，不能将此前DIP目标误差称堵转证据。前5条实机日志拟合二次耦合，第6条留出RMSE0.00524°。同一79610a9c学生/seed42/1env原配置本地CPU推理+GPU PhysX：独立DIP五轮；删独立PD并加双向弹性耦合k2.4与4.8，两组均4轮，第五轮重抓超时，但各完成5次松开/回位。首次松开完成24/25/26步，未复现真机首轮不分开。耦合误差中位0.554°/0.350°，P95仍6.8°/3.78°，不冒称精确硬件复现。

新增独立诊断评估脚本与日志拟合脚本，生产任务/权重/部署未改。三阶段180s独立timeout均exit0；两次启动配置失败单独保留，非物理失败。全部本轮Docker已退出；无训练或硬件执行。详细证据docs/e003_passive_coupling_diagnostic_2026-09-15.md及outputs/local_pipeline_e003_passive_diagnostic_20260915；AST、留出拟合及baseline与原208步五轮录像数值一致检查通过。结论为耦合影响可靠性，但尚不支持其为首轮不分开的主要原因。

唯一下一步：保留耦合诊断，结合实机主动关节响应和初始装夹/接触差异定位阶段推断偏离，不直接重训；装夹条件待用户补充。

## 2026-09-15 — 实机反馈更新

交付student未复现实机分开，当前实机循环判失败。先解决食指index_3目标/反馈严重不一致及疑似耦合，参见codeagent_issue.md与最新handoff；暂停PPO/学生续训，不能以仿真五轮通过代替实机验收。5N及力矩平衡目标保留。

## 2026-09-15 — E003真机输入student部署包交付，受力保持未通过

完成里程碑：原合格d42教师冻结蒸馏出的student现在只用关节/上一目标/80帧历史，阶段与时钟在模型内推断，0.05rad/步、20Hz；原M24 SDK/串口/映射/初始化/历史/归一化接口已适配，不需要外部阶段逻辑。独立包outputs/local_m24_deployment/e003_proprio_student_20260915及同名zip；最终模型79610a9cab049abf08c283352e9c4ccb57569fc60d2020f3c7ec79ddb0a85c9a，zip SHA418470fd1f105f6fa635e157a099265e674d122d3c96a0fd6c3e2d553d7fba54。包状态deployable_runtime_candidate;strict_transfer_not_passed，不能标完整低力转移成功。

选型：交付smoke固定128/128、扰动493/512，原教师472/512；首轮正式最优483，教师锚点对照最优490。交付模型全640例20Hz峰值24.2267/12.6685N（食指/拇指）；固定19.4402/12.3248N，对照教师固定16.2940/9.7489N，所以原逐组110%受力标准仍失败。拇指drive失触各组均值0s。选择smoke的完成数与食指峰值更好，不按训练时长选择；详细三候选表/局限见docs/e003_student_deployment_result_2026-09-15.md。5N、力矩平衡仍保留后续，未优化PPO、未改变成功teacher标签。

原sim2real PyTorch2.5.1环境通过实际权重CPU native/JIT逐位一致、200步sim/deploy一致及不可用输入污染；最终包从其他cwd默认路径20步fake-device运行exit0，模型SHA/0.05/profile正确；zip CRC和模型/原源码SHA通过。SDK单独仅导入检查通过，未创建真实控制器/启用电机。CPU推理+原GPU PhysX单例五轮完成，200Hz法向峰值20.5453/12.1342N、整指与指尖一致；同权重CPU录像208帧/20Hz五轮，代表阶段画面已看。源凸体交集内切球半径0.2929mm与教师0.2907mm同量级，不冒称PhysX穿透深度。额外20轮测试学生8–9轮失败、教师12轮jump失败，不能承诺无限循环；README用200步作为约10秒试运行上限，不是外部阶段切换。

新CPU推理开关只改变模型计算设备，GPU物理/门禁保持；默认GPU行为保留，CPU contract和实际闭环/录像通过。一次诊断命令全局替换误改checkpoint及一次随主容器退出被中断录像均标记排除（cpu_probe_corrections.json）。有效CPU证据为scale005/smoke_cpu_checked_physics_nominal和smoke_cpu_final_native_nominal。封包dt检查修正为容纳已读取float32的0.04999999888s，不改物理/受力门槛。

保留示范正式600s到时exit124、8618步/551552样本、teacher_unchanged=true，25%锚点16384样本覆盖四阶段，仍未解决受力保持。本地所有本轮训练/评估/录像容器已退出，nvidia-smi compute apps为空，无排队续训；云本轮无库存未启用，云侧交接无法同步，本条为本地实测。当前方法文档已改写为最终0.05契约，避免旧0.02/开机步骤被当成当前下一步。

唯一下一步：等待用户对交付候选的短时实机试运行指令，按包内README/原M24环境执行并采集关节与目标日志。本次没有操作硬件，不自动续训PPO/学生、开机或运行实机。相关代码/文档自动Git提交，不推送；大模型/视频/日志留在outputs中不入Git。

## 2026-09-15 — 首轮student最终对照完成，保留示范短训已启动

首轮scale005 pipeline正常完成全部评估，选中formal/candidate_0025727.pt：固定128；43/44/45/46为124/121/120/118，合计483/512。固定峰值19.5968/11.6092N；扰动全组最大26.0060/12.4959N。五轮可靠性/拇指失触通过，但逐组110%受力保持失败；比smoke493/512且全组最大24.2267/12.6685N没有综合优势，旧候选均保存，不能默认选最后模型。原Docker已退出，无旧训练残留。

保留教师样本的新launcher首次把主机绝对路径传入容器，legacy source断言失败，exit1且未进入训练；原失败目录/anchor_launch_status.json保留。修正为容器工作目录下相对路径，使用新目录outputs/local_pipeline_m24_e003_deploy_student_anchor_pathfix_20260915，Docker m24_e003_deploy_student_anchor_pathfix_20260915。120s smoke+600s formal，warm-start严格为79610a9c短训学生，25%不可变教师示范/75%近期学生状态，64env；start_local_anchor_pathfix.command.sh与anchor_pathfix_launch_status.json记录完整命令及时间。当前smoke已进入IsaacGym启动，不更改PPO/奖励/物理/原五轮门禁。

唯一下一步：确认教师锚点四阶段覆盖并完成这次有界对照，按实际分数选择学生、验证同一权重的录像/200Hz力/原sim2real入口，交付保留真实限制的部署包。不开真机；云仍无库存关机，本地交接待同步。

## 2026-09-15 — 保留教师示范的有界student对照准备完成

目标：修复蒸馏中的原教师受力保持，不优化PPO。首轮formal提前SIGTERM保存结束，26080向量步/1669120样本、teacher_unchanged=true，正式exit0；未跑满原3600s上限。formal_stop_decision.json记录原因；原pipeline仍按既定固定/43/44候选、45/46选定评估收尾，待退出后才启动新训练，不并行两轮训练。

证据：先前4337/9376/19358纯学生固定均128/128，但峰值约20/12N，未接近原16.3/9.75N；现有FIFO容量16384，beta降至0后最初教师示范完全被覆盖。初步重建教师轨迹离线动作误差对照未验证首帧精确重放，仅作诊断、不作验收/因果结论。可检验假设是持续保留原教师状态监督能改善受力保持。来源DAgger Algorithm1在https://proceedings.mlr.press/v15/ross11a/ross11a.pdf明确聚合历次数据；固定16384教师缓冲、每batch25%教师锚点/75%近期学生状态、从已验证smoke权重初始化均为本地适配，不是论文保证。不改损失系数、actor、物理、奖励或任何门禁。

新增默认关闭expert replay选项，保存初始500步纯教师阶段的最后16384样本，要求四阶段均覆盖，随后不可变并保存expert_anchor.pt；训练记录fraction/count/来源SHA。warm-start严格校验非encoder所有张量与原教师/控制常量一致。CPU测试验证fraction0与旧抽样逐位一致、25%标签/输入对齐、锚点不可变、未就绪回退，原0.05完整contract也通过。拟新pipeline outputs/local_pipeline_m24_e003_deploy_student_anchor_20260915，64env、120s smoke+600s formal各独立timeout、warm-start为scale005/smoke/student_last.pt（79610a9c），不无限续训同类设置。

部署工程已可交付候选包：新增package_m24_e003_deploy_student.py把真实权重、原部署源码、新入口、验证和录像收齐，严格force失败保留为false，不更改10%标准。已测preview_smoke为deployable_runtime_candidate;strict_transfer_not_passed，不标记成功student。原sim2real Python环境（PyTorch2.5.1）的SDK仅导入通过，未创建controller/连接/启用电机。保守十秒试运行上限是进程上限，不作为阶段切换。原五轮之外的补充20轮16env测试：smoke各完成8–9轮后重抓超时/工作空间/旋拧超时；原教师12轮后全部角度jump终止。两者均不是无限循环合格，后者不能当蒸馏退化的完全解释；不混入原五轮结果。

唯一下一步：完成原pipeline收尾并检查候选，再按假设运行一次保留示范对照，最终选型、完整同权重验证和部署包交付。云因无库存仍关机、此次仅本地，交接待同步。5N、力矩平衡保留后续。

## 2026-09-15 — 无外部阶段输入的student闭环与原sim2real环境验证通过，受力保持待验收

当前子目标仍为可直接部署student，不优化PPO。0.05/20Hz短训1552步的固定128/128完成原五轮；扰动43/44/45/46为124/122/125/122，合计493/512，原本地教师472/512。各组拇指失触累计均值0s。固定食指/拇指法向力峰值19.4402/12.3248N，教师16.2940/9.7489N；扰动全组最大24.2267/12.6685N。不能把完成率提升当作原低力全部保留，预先声明的逐组110%力保持检查尚未通过。

原生20Hz录像smoke_native_nominal/student_20hz.mp4共207帧，实际五轮、末reason5、净角138.77°；已查看旋拧/离触/回位/重抓代表帧，未见明显大范围视觉穿透。trace中历史goal_completed=false是录制器固定保守字段，真实数值见rows末cycles5，不据此篡改旧记录。200Hz独立审计smoke_physics_nominal已完成，源碰撞凸体抽样交集内切球半径最大0.288mm（不能称PhysX穿透深度）；教师同条件审计在teacher_physics_nominal。完整接触摩擦wrench仍不合格，不宣称力矩平衡。

新增verify_m24_e003_deploy_artifact.py验证实际训练JIT而非随机fixture：原teacher全部actor/归一化及控制常量逐位一致，CPU native/JIT逐位一致，200步模拟适配/部署一致，不可用输入NaN污染通过；将模型和原部署源码复制独立临时目录、从其他cwd用默认模型路径运行20步fake-device通过。在原/home/vision/software/miniconda3/envs/sim2real/bin/python中同样通过，未导入SDK/未操作硬件。修复新入口libtiff在NumPy/Torch前加载的原SDK ABI顺序；默认模型与日志定位bundle旁，串口/关节映射/旧源码不变。可选阶段估计日志用于诊断，默认关闭；0.05实际teacher+SDK限位对照128/128、16.1767/9.5433N，限位不是当前力增大的主要解释。

正式3600s蒸馏仍运行，Docker m24_e003_deploy_student_scale005_20260915，pipeline outputs/local_pipeline_m24_e003_deploy_student_scale005_20260915；03:42:53Z启动，预计04:42:53Z到时保存。当前约2.1万向量步、教师冻结、beta0，全四阶段覆盖。9376/19358快照固定128，但力约19.93/11.92、20.25/12.15N，尚未完整转移验收。watch_candidates.py依次检查9000/18000/27000/36000门槛快照，实际命令与结果持久化；不随意追加PPO/放宽原任务。

本次代码/交接里程碑自动Git提交，不推送。云无库存保持关机，本条待云可达再同步。唯一下一步：完成当前有界蒸馏并选择实际最优候选，汇总完整受力/视频/原sim2real环境验证，交付明确标注验收结果的部署包；5N、力矩平衡仍为后续目标。

## 2026-09-15 — 用户选择0.05rad/步，按原合格教师配置重新蒸馏

用户明确选择“0.05rad/步，对齐合格E003教师”。新student训练/纯学生评估/导出控制步统一0.05、20Hz、SDK限位与E003初始位；旧sim2real原脚本仍原样保留，新部署入口只对E003所选profile从artifact读取步幅，日志如实记录。默认新蒸馏入口同步改0.05，防止误启已失败0.02设置。不操作实机；PPO教师仍冻结d42e98a2。

此前0.02短训纯学生固定127drive/127release/0return。监督可行性诊断：冻结教师0.02且动作×2.5再clip，0/128drive；只将phase clock放慢2.5倍，113drive/release但0return；所有动作同比减速并将phase时间预算同步×2.5，128drive/release但0return，单例最小回位误差6.511°（门槛5°），随后偏离。均明确control_diagnostic、非验收，不修改原教师权重/摩擦/奖励。数据outputs/local_pipeline_m24_e003_deploy_student_devicefix_20260915/oracle_*和outputs/local_pipeline_m24_e003_uniform_slow_teacher_20260915。两次诊断包装命令错误（Hydra路径、重复入口参数）保留日志，实际有效结果为checked与uniform目录。

因此提前结束不完整监督的0.02正式训练，5329步/341056样本，SIGTERM完成last checkpoint保存且teacher_unchanged=true；随后停止该Docker，stop_decision.json记录原因，未跑满1小时、未宣称目标完成。云依旧无库存关机，无本地旧job残留。

新增明确action-scale profile；修正初次history用已发送初始化目标填充旧帧，仅当前帧追加测量，匹配原M24/仿真reset时序（此前用当前测量填满旧帧）。CPU0.05完整契约测试通过，含实际teacher绑定/冻结梯度、初始历史、60步sim/deploy逐位一致及旧SDK流程fake-device运行。新pipeline outputs/local_pipeline_m24_e003_deploy_student_scale005_20260915，Docker m24_e003_deploy_student_scale005_20260915，64env，独立120s smoke/3600s formal；启动信息见该status.json及outputs/e003_deploy_preparation_20260915/start_local_scale005.command.sh。原本地teacher472/512报告经验证复用。正式来源快照逐文件随模型保存。

唯一下一步：检查0.05训练四阶段监督覆盖及纯学生闭环成绩，完成数值/物理峰值/视频和完整部署包验收；当前仍无新合格student。5N、力矩平衡继续保留后续，不进行PPO优化。

## 2026-09-15 — 本地 student smoke 通过，正式蒸馏运行中

AutoDL无库存后按既有授权使用本地RTX3060/12GiB、dexscrew:ig20-py38 Docker/PyTorch2.3.1+cu121。原合格教师在本地固定128/128，43/44/45/46分别112/120/121/119（472/512），各评估exit0；数据在outputs/local_pipeline_m24_e003_deploy_student_20260915，不能将与云468/512的差异解释为模型变化。

首次student smoke因环境tensor-device=cpu而RL观测=cuda引发device mismatch，正式未启动；原失败保存。修复学生/缓冲/抽样采用cfg.rl_device，仿真限位张量仍在env.device；不改变GPU PhysX或CPU tensor pipeline。新pipeline outputs/local_pipeline_m24_e003_deploy_student_devicefix_20260915，Docker同名去掉local_pipeline前缀为m24_e003_deploy_student_devicefix_20260915；经SHA/seed/门禁/执行地点校验复用已完成teacher报告，避免重跑。

修复后120s smoke exit124正常，1531步/97984样本，约921样本/s，action MSE0.00752，phase CE0.01772；原teacher冻结检查通过。正式3600s上限于2026-09-15T03:27:55Z启动（本地11:27:55），64env/512batch、2次监督更新，GPU显存初测约3GB（含桌面）。正式启动前仅补充phase监督覆盖/准确率和实际runtime_source快照，不改优化计算；每120s保存独立JIT候选及last checkpoint，尚无合格学生。

同时已开始一次短训权重纯学生nominal闭环诊断（相同本地GPU独立进程，显存仍有余量；其重叠期间吞吐不作容量比较），smoke_pure_nominal.command.sh/log。录制入口新增record_deploy_student开关，使用同一部署控制步且20Hz播放，不依赖真实阶段输入。唯一下一步：确认纯学生真实循环及阶段监督覆盖，监控有界正式蒸馏，随后完整数值/物理峰值/视频/部署验收；不优化PPO，不启动实机，云仍关闭。

## 2026-09-15 — 云端无库存，转入已授权本地 student 训练

用户确认开机pro-786095bb6e03，但AutoDL exit1/InternalError“当前算力规格暂无库存”，request b7cbf094f631c338d05b5f66980d0444；再次status确认shutdown，request b8803da4a143957391f6df827fcc0260。按用户已授权无云资源可本地训练，使用本地RTX3060/12GiB与既有dexscrew:ig20-py38 Docker，仅运行student目标，不启动PPO或实机；云保持关闭且无法读取实际云交接，本条为本地待同步。

当前子目标：部署输入student的本地启动/容量验证和首轮蒸馏。初始64env，smoke120s、正式3600s独立timeout；每组原五轮评估240s。原教师d42e98a2冻结，原实机0.02rad/步，详细方案见docs/e003_deploy_student_2026-09-15.md。管线新增execution-location/provider-failure参数，避免本地结果误记为云。provider失败/计划记录outputs/e003_deploy_preparation_20260915/provider_failure.json；拟输出outputs/local_pipeline_m24_e003_deploy_student_20260915。

唯一下一步：完成本地原教师复现和student smoke，确认真实GPU容量/吞吐后继续有界蒸馏及纯学生闭环评估；尚无新合格学生。不开云端并行任务。

## 2026-09-15 — E003 真机输入 student 适配准备完成，尚未训练

用户明确改为：暂停PPO优化，仅从原合格E003教师d42e98a2蒸馏可直接部署student。旧phase101学生依赖仿真阶段，不作为此次交付。新增独立history→阶段/时钟/latent估计器，冻结原teacher actor；导出包含SDK限位、20Hz、原实机0.02rad/步和目标历史的完整控制步。训练采用有界DAgger监督，教师不更新；sim2real原串口/SDK/日志/停止流程复用4736d94版本。5N与完整力矩平衡保留为后续目标。

新增模型、train/pipeline、deploy/runtime、CPU contract测试入口；原cycle评估通过显式开关可调用仅本体输入的导出学生，仿真phase只作评分。CPU实际teacher绑定、冻结梯度、TorchScript、不可用输入污染、连续60步部署/模拟目标一致、原M24假设备3步运行全部通过；旧student contract与编译通过。测试只用未训练临时模型，没有优化器更新、IsaacGym训练或实机动作。没有可交付的新权重，部署任务尚未完成。

详细方案/来源/验证与风险：docs/e003_deploy_student_2026-09-15.md。初次云120s smoke+3600s student，各评估240s独立timeout；64env起步，原teacher完整5轮作参考。0.05教师→0.02实机步幅差异与阶段可观测性必须实测，不能用接口通过替代行为验收。准备脚本/原部署源码/供应商状态记录在outputs/e003_deploy_preparation_20260915；预期pipeline outputs/cloud_pipeline_m24_e003_deploy_student_20260915。

本次provider只查询状态：shutdown，request06be950538c21a032a90b7535c29510f；本地RTX3060/12GiB只做开发测试。AutoDL技能明确要求针对pro-786095bb6e03本次开机单独确认，尚未开机；本条尚未同步云端。唯一下一步：取得精确实例开机确认后，读取实际云端交接，再同步并启动student-only smoke/训练/闭环验证。代码准备里程碑自动Git提交，不推送；不得宣称学生部署完成。


## 2026-09-14 — 已按用户条件停止优化目标（未完成）

用户授权放宽/取消门禁，若仍不行就停止。最后E003RelaxedGates试验已完成：60s smoke exit124，正式1548288steps/epoch83 exit0，固定/43/44各128env、BEST与LAST均128/128完成两轮，但两候选各自0/384满足全过程5N；仅20Hz螺母法向力端点已全部超限。BEST周期成本信号峰值16.1437/8.3941N，LAST16.1526/8.4767N；端点峰值分别16.0728/8.3349、16.1398/8.4070N。放宽完成率不作原五轮验收，完整优化未完成。

LAST a3e6e9744edcc5a583a1d15e575af346a2b116a88a6a7fe597c9c9e7e840d968；BEST3cb6aaa0eb63aa1be13467a46ad7c4d6a4d344cd6dab092588d253462e640fd7，路径见pipeline status.json。outputs/cloud_pipeline_m24_e003_relaxed_gates_final_20260914完整自动同步完成，本地两checkpoint SHA与云manifest匹配。final_gate_analysis.py/json及goal_stop_decision.json保存可复现计算和用户停止条件。

已核验云端无tmux会话、无训练/评估进程、无GPU计算进程；没有排队后续任务。云机保持开机（本轮未收到关机指令），未操作实机。原e003_low_force_cycles_success标签仍指向5e7e965、原教师SHA d42e98a2不变。此前GPU完整wrench未校准/CPU物理不等价的问题也未解决，不能作完整力矩平衡验收。

收敛审查 docs/e003_optimization_review_2026-09-14.md已补齐最后结果。停止是未达到目标后的用户条件执行，不是宣布5N不可能，也不是将优化标记成功。唯一下一步：等待用户明确新指令；本目标禁止自动续训或启动其他优化实验。

## 2026-09-14 — 用户授权最后一轮放宽门禁，失败即停止

用户先指出可能陷入循环，随后明确“可以降低或取消门禁。如果还不行，就停止目标”。本轮按阶段/验收门禁理解（已提供异步纠正入口），不继续追加同类调参短训。已完成BoundedMeanStrong：LAST固定128、43/44=108/117（225/256）；BEST固定128、115/120（235/256），无已通过5N候选。

食指分阶段刚度冻结策略诊断已完成，三个180s独立上限exit0：Kp1.8净43.6219°但drive10（配对有效进展不足），1.2净12.5852°，0.8净8.6323°均drive10；200Hz峰值分别14.7185/8.4567、11.4613/7.1222、6.7827/8.5762N。只在drive/release改食指，return/regrasp仍2.4，thumb2.4；没有完整低力循环。实际wrapper测试确认只影响指定关节/阶段、base值为exact no-op；报告明确control_diagnostic/acceptance_eligible=false，汇总排除这类控制覆盖。

最后放宽试验E003RelaxedGates继承较稳定BoundedMean（边界coef0.01，不是已拒绝的0.1）：goal_cycles5→2，return容差5°→15°，deadline双倍[320,160,240,160]且clock_scale0.25保持已学80/40/60/40观测时钟，dwell2→1；关闭配对累计进展门禁，但仍要求实际净旋转达到原20°且有真实接触。释放仍需两指离触，回位仍需实际几何误差在新容差内。奖励公式/5N超限成本/物理摩擦/Kp2.4/固定螺栓不变，降低门禁不等同降低测力标准。事件奖励的触发条件是本次用户授权的本地适配；原Tac2Motion接触/旋转/回位结构保留。测试原完整cycle、原force成本、relaxed contact与phase-clock逐位一致通过。

评估新增required_cycles、stroke_target_degrees、return_tolerance_degrees、paired_drive_progress_required、event_dwell_steps，放宽报告不混入原五轮scoreboard。最后pipeline outputs/cloud_pipeline_m24_e003_relaxed_gates_final_20260914 / tmux m24_e003_relaxed_gates_final，原d42新warm-start、OMP/MKL1、1536/3072，独立60s smoke/600s上限/1548288steps，再固定/43/44与必要200Hz检查。云无其他job。结束后若仍不能建立低力旋拧结果，按用户要求停止目标：保存真实失败结论、自动Git提交、不再启动训练或其他目标，不标记原优化完成。用户未要求关机，不操作实机或电源。

收敛审查草稿 docs/e003_optimization_review_2026-09-14.md，最后结果补齐。唯一下一步：执行且收尾这一轮放宽试验。

## 2026-09-14 — BoundedMean完成，固定输入确认正则化不足

BoundedMean LAST c2f1e18ee8e631ed129119e93d8ff4fe47d4b40871023061388c980fa5fbe2fb 固定128、43/44=120/115（235/256）；BEST固定128、118/120（238/256）。LAST seed43周期峰值25.8778/12.2101N，未5N。全部云job已结束，原成功基线不变。

新增默认关闭raw审计的policy_inputs.npz（仅单env），保存实际模型输入，不重新计算状态；原teacher完整轨迹共同字段保持exact，CPU离线重放rawmean在1e-5容差内一致。compare_m24_e003_fixed_state_actions.py对后续候选强制输入统计逐tensor相同，严格载入模型；输出按相同61个drive/57release/46return/33regrasp观测比较，不把不同轨迹的阶段分布当成网络改进。

相同原teacher输入上：未正则Strong5 LAST drive bound_loss2.84598，BoundedMean LAST2.59675（-8.76%），原teacher2.77846。最大raw mean3.235→3.134，饱和比例仍37.295%（原同值）；当前0.01正则只轻微改变越界，并未到达可执行范围。完整证据 outputs/cloud_pipeline_m24_e003_reference_inputs_20260914/fixed_state_comparison.json。基于此明确不足而非盲目重复，下一BoundedMeanStrong仅coef0.01→0.1，其原drive辅助项约0.278（公式/phase mask同上一轮），检验越界损失/饱和比例能否显著下降同时维持五轮与降力。系数本地适配，公式来源rl_games官方链接见上节。

拟pipeline outputs/cloud_pipeline_m24_e003_bounded_mean_strong_20260914 / tmux m24_e003_bounded_mean_strong，原d42、原物理/gates、独立60s smoke/600s上限/1548288同采样量，随后固定/43/44；下一步完整验证此较强动作正则化，而非接受当前5N结果。

## 2026-09-14 — 动作饱和审计通过；BoundedMean假设

SmallNoiseStrong5同量正式exit0，LAST固定128、43/44=116/115（231/256），BEST固定128、111/119（230/256），LAST seed43控制周期峰值24.4069/12.4871N，仍拒绝5N验收。CPU任务设备修正后exit0但首驱动超时，仅5.2987°，200Hz法向峰值122.3846/4.7811N；与GPU原五轮144.71°明显不等价。CPU传感器已知rig通过不能转作GPU任务完整wrench验收，保持GPU物理。

新增可选cycle_eval_raw_actions，只hook实际ActorCritic.act_inference返回值，动作/状态不变；GPU原teacher197条共同trace逐项精确等于此前control，raw clamp后与active实际动作精确相等。绑定d42 checkpoint的审计：原始mean最大3.236，而执行限幅±1；单例驱动37.295% active动作饱和，mean越界正则sum平方均值2.7785，缩小σ后这些饱和动作平均采到[-1,1]的概率由8.482%降到2.355%。全phase单例饱和40.799%，批量此前36.635%，不混统计。证据 outputs/cloud_pipeline_m24_e003_raw_action_20260914/raw_action_analysis.json（云端已生成，分析器绑定SHA）。这支持存在难以探索的区间，不冒称唯一因果。

下一BoundedMean继承SmallNoiseStrong5，在原PPO已有soft_bound=1.1平方损失上开启coef0.01，仅active8关节的非return阶段应用，return保留原KL避免与原始饱和参考直接冲突。公式源自rl_games官方 https://github.com/Denys88/rl_games/blob/master/rl_games/algos_torch/a2c_continuous.py ，原repo已有同式但coef0关闭；phase/action mask与0.01系数是本地适配。原单例drive辅助项约0.0278/步、release0.00527，不改变物理限幅或5N目标，检验越界程度/可执行探索概率是否改善并同时降低受力。新增函数与标准公式等价、return/inactive/interior梯度零、越界恢复方向测试通过。默认all旧分支保持。

拟outputs/cloud_pipeline_m24_e003_bounded_mean_20260914 / tmux m24_e003_bounded_mean，原d42重新开始、1536/3072、OMP/MKL1、60s独立smoke/600s上限且1548288同量PPO，固定/43/44比较，训练后必须审计raw输出不能只看成功率。前面所有云job已完成，无并发。下一步执行该对照，原成功基线不变，5N/平衡仍未完成。

## 2026-09-14 — SmallNoise完整对照完成，降力/卸载未通过

SmallNoise formal1548288steps/epoch83 exit0，BEST 801b05a799d912525a31ce424997a00623a0f6208a175765e9bd4bde0a321c39（best_reward_173.75.pth）固定128、43/44=115/119（234/256）；LAST dec0c072a573ef4dabae694cabcde55bed430a0b3deede7a5c24dfde19d98edc 固定128、112/113。BEST实际checkpoint所有输入统计与原teacher逐tensor相同，actor均值/value统计改变，验证脚本通过。

seed43共同成功110例，原/新力平方积分709.928/707.905N²s，食指超5持续4.54045/4.55s，拇指1.365/1.37136s；峰值23.7530/11.8029→21.3811/12.5985N。稳定离触lag0.33828→0.33984s、单侧冲量1.71707→1.73190Ns，仍全拇指先离开。只有训练稳定性改善，不作降力或卸载验收。证据pipeline下paired_best_s43.json/unload_best_s43.json，完整checkpoint和评估已自动同步。

CPU任务审计首次exit1：sim_device=cpu但rl_device=cuda导致RunningMeanStd张量设备不一致，尚无有效轨迹。修正仅将rl_device=cpu，保留失败记录，新tmux m24_e003_cpu_task_audit_fixed/pipeline outputs/cloud_pipeline_m24_e003_cpu_task_audit_fixed_20260914，等待Strong5完整结束后单独180s执行。

SmallNoiseStrong5当前正式PPO运行（A0c/115前检通过、smoke124），约3.1GB VRAM，未改正在运行任务。下一步完成较强5N惩罚结果与CPU修正版审计，仍保留原成功基线。

## 2026-09-14 — SmallNoiseStrong5受控后续

SmallNoise LAST固定128、43/44=112/113（225/256），相较FreezeNorm LAST181改善但未及原234；seed43端点21.49865/11.56957N、周期峰值24.18537/12.74764N，仍远超5N。BEST尚未完成所有组，不提前验收。无噪声诊断wrapper的first_episodes、27279条first_episode_steps、initial_joint_offsets均与原前检逐项相同。

下一假设：冻结输入统计与较小探索已分别获得归一化因果/冻结策略噪声证据，重新检验force_excess_scale=0.25是否能在此稳定条件下降低超限。SmallNoiseStrong5仅将SmallNoise的force0.05→0.25，原d42重新warm-start、std×0.25一次、normal0.125/peak/KL1/LRcap1e-5/1536env全部保留。不同于失败Strong5KL的输入统计漂移与大噪声；不重复相同失败条件。超限线性公式参考官方legged_gym https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py ，系数为本地适配；阶段KL参考 https://arxiv.org/abs/1803.03835 。仍独立60s smoke、600s正式上限/1548288采样同量、固定/43/44，先等待CPU任务审计结束再串行启动。拟tmux m24_e003_small_noise_strong5，outputs/cloud_pipeline_m24_e003_small_noise_strong5_20260914。

下一步完成SmallNoise和CPU任务审计，执行此强超限学习对照，不提升基线。

## 2026-09-14 — CPU完整wrench已知动态rig通过，任务迁移待验

SmallNoise已通过实际restore/freeze/KL测试、A0c和seed43前检115/128；60s smoke正常124，正式PPO正在运行，tmux m24_e003_small_noise。自动同步监控已启动。

CPU PhysX四组μ0/1、y/z动态rig分析完成：原始sensor不缩放时，独立惯性接触参考与力最大误差5.95e-6N，倾覆moment最大0.001844Nm，全部低于事先界F峰0.1N/RMS0.05N、moment峰0.01Nm/RMS0.005Nm。μ0 null参考有效、μ1切向激励成立。CPU raw/8失败；GPU raw/8也失败，不混用后端或宣称任务测量已通过。证据 outputs/cloud_pipeline_m24_e003_dynamic_wrench_cpu_20260914/calibration_unity_analysis.json；分析器显式sensor-divisor=1，旧默认8及旧报告保留。此为观察CPU原始力吻合后的后续分析，不冒称预注册unity假设。

新增probe_m24_e003_dynamic_wrench_cloud.py和analyze_m24_e003_dynamic_wrench.py，独立已知外力/自由球惯性参考、创建前材质、CPU/GPU显式选择；未改训练物理。创建前材质参考官方legged_gym代码，后创建材质μ0/1完全同轨迹的失败rig保留。

已排队tmux m24_e003_cpu_task_audit，pipeline outputs/cloud_pipeline_m24_e003_cpu_task_audit_20260914，等待SmallNoise完整结束（1800s等待上限）后独立180s运行原d42教师n1/seed43/200Hz审计，唯一物理后端改CPU用于诊断，不用于替换GPU验收。下一步读取SmallNoise五轮/5N指标并检查CPU任务轨迹与完整wrench迁移；原成功基线不变，整体优化尚未完成。

## 2026-09-14 — FreezeNorm早期候选恢复鲁棒性；探索噪声诊断

FreezeNorm同量1548288steps/epoch83正常exit0（独立600s上限仍在，实际约4分钟），吞吐约6360steps/s；最终checkpoint e9651148已验证obs/priv/point统计逐tensor与d42e98a2相同，mu与value统计改变，证据frozen_statistics_verification.json。BEST固定128、43/44=123/123（246/256），LAST固定128、89/92（181/256），故不能默认采用最后模型。BEST固定峰值16.5673/9.4895N、43/44合并21.5063/11.9403N，仍未5N，拇指峰值略高于原11.8029N，不作完整优化验收。

已完成独立冻结策略噪声对照（不学习、单独随机流seed43，现BalanceFirstPeak物理/目标）：scale0/1/0.5/0.25五轮115/39/95/107。scale1有64 return12、21 drive10、4 workspace；0.25为20 return12、1 drive10。原始噪声每控制步作用于增量目标，其±1°量级会破坏已学轨迹，初始扰动鲁棒不等于每步探索鲁棒。最初wrapper重复注册Hydra resolver退出，已复用calibrate bootstrap并在新exploration_fixed pipeline四组全部exit0，原失败记录保留。所有混入噪声的报告明确diagnostic/acceptance_eligible=false。

新增可选init_action_std_scale，只有restore_train对log_std加log(scale)，默认1逐位保留；restore_test不改变保存分布。实际完整restore方法测试证明初始mu推理精确不变、源checkpoint文件不变、非sigma模型参数不变，冻结return参考使用显式缩小后的std。SmallNoise继承FreezeNorm，唯一训练改动std初始×0.25；这个操作相对于所选warm-start checkpoint，不可在连续恢复同一已缩放checkpoint时重复缩放。原teacher文件仍d42e98a2，reference均值/输入统计保持，KL的参考方差按上述噪声设计调整。

拟pipeline outputs/cloud_pipeline_m24_e003_small_noise_20260914 / tmux m24_e003_small_noise，OMP/MKL1、1536env/3072mini，独立60s smoke、正式600s上限且1548288steps同量对照、原固定/43/44评估。5N阈值、force0.05/normal_balance0.125、Kp2.4/物理/gates与FreezeNorm相同，不通过力裁剪改善。下一步验收低探索微调是否维持可靠性并降低真实载荷，再推进更强5N学习。

完整wrench仍未通过：创建后设置μ0/1的rig轨迹完全相同；改为官方legged_gym同样的创建前asset材质设置后，μ0的独立惯性力与normal readback误差<2.1e-6N，而μ1产生显著切向差，校验参考成立。仍不能使用raw/8，含接触瞬间最大F误差54.29N，留出窗口峰值0.328N。仅改校验rig，未改当前教师物理；CPU PhysX相同rig对照已完成待分析。创建前材质参考 https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py 。

## 2026-09-14 — 归一化漂移因果证据与线程吞吐改善

BalanceFirstPeak最终LAST固定122、43/44=81/83（164/256）；BEST固定128、77/80（157/256），未通过，正式1548288steps/84轮更新左右。完整表待同步后更新，不以较短失败轨迹接受载荷下降。

归一化2×2诊断完成：原actor+新input norms固定128、seed43=74（42 return12/11 drive10/1 workspace），相同原actor+原norm原115；新actor+原norm固定128、seed43=94（33 return12/1 drive10），而新actor+新norm原76（29 return12/23 drive10）。只有输入统计改变已能导致41个成功数净损失；恢复原统计能消除大部分新增drive失败，但网络回位退化仍存在。二者仅诊断，不作策略验收。

新增默认关闭freeze_input_stats，PPO模型继续train，obs/priv/point RMS切eval冻结，value RMS仍train适应新奖励。实际set_train方法与真实RunningMeanStd执行3轮验证：三类input状态逐tensor不变、value统计增长、actor有有限梯度，默认模式更新行为保持。ProprioAdapt独立实现未受影响。

独立60s OMP_NUM_THREADS=1/MKL_NUM_THREADS=1容量probe正常124退出，实际Torch线程1、稳态约6090steps/s（原峰值版约2700），约2.25倍吞吐，checkpoint只是容量产物。后续采用明确线程环境；不把更多步数造成的训练差异当单因素学习证据。

下一FreezeNorm对照保留BalanceFirstPeak全部奖励/物理/峰值读取/回位KL，仅冻结输入统计；云OMP/MKL1，原d42e98a2，独立60s smoke、正式600s硬上限且max_agent_steps=1548288匹配上轮采样量（上限以先到为准），随后原固定/43/44评估。新增runner可选--max-agent-steps不替代独立timeout。拟pipeline outputs/cloud_pipeline_m24_e003_freeze_norm_20260914 / tmux m24_e003_freeze_norm，等当前校验rig完成后串行启动。

动态wrench首轮探针在step249–262前滚离螺母，留出窗口无接触，数据判invalid、不能证明任何缩放通过。原源码/SHA/数据保留。已在校验rig加入已知且全量记录的切向/轴向外部居中PD(500N/m,10Ns/m)，螺母原关节不锁定；这修复探针逃逸的明确假设，未改训练物理。新pipeline outputs/cloud_pipeline_m24_e003_dynamic_wrench_stable_20260914正在运行，四组每项120s；预设误差界仍F峰0.1N/RMS0.05N、倾覆moment峰0.01Nm/RMS0.005Nm，raw/8仍仅假设。下一步读此校验并执行FreezeNorm对照，5N/完整平衡未完成。

## 2026-09-14 — KL完整拒绝；原资产横向负载影响与BalanceFirstPeak

Strong5KL600s完整完成/自动同步：LAST固定128、43/44=76/74（150/256），BEST固定128、70/72（142/256）；固定LAST峰值17.6967/9.1419N。KL修复了MSE版本固定0/128的回位遗忘，但扰动仍远差于原234/256，拒绝提升基线。

原资产known-load sweep45组已独立120s上限exit0，无手/无接触，原URDF/friction0.2/damping0.01/重力/8-0迭代/真实朝向不变。0.1Nm、横向0N一秒转1.16210rad（66.584°），横向5N几乎0；5N横向时0.2Nm转0.91443rad。说明当前模型横向负载会显著增加转动阻力，不等同于真机标定或双指5N可行性证明。原get_actor_dof_states报告速度近零而q持续每步变化，分析必须用位置差分；原始字段保留，修正版脚本单列两种速度，分析在joint_load_analysis.json，不能误判“转后停下”。

原教师action_scale=0.0375单例完成五轮、200Hz峰值16.3293/7.9740N；0.025和0.0125均return12，峰值15.3203/4.1011、14.4611/3.3431N。拒绝速度方案，不能因拇指<5接受。全档案已同步，无训练并发。

下一有界工程假设：先改善横向加载效率，再收紧总载荷。原teacher驱动法向横向负载均值约6.78N、index/thumb法向均值11.06/4.98N；已知外力对照支持避免单侧挤压的机械动机。BalanceFirstPeak保留5N目标、Kp2.4与原gates/原初始化：force_excess_scale采用较温和Force5值0.05，normal_balance_scale由0.025提高至0.125；原轨迹normal成本约5.64→28.22（约总221.18的12.8%），让平衡项不再远弱于驱动收益。并启用200Hz周期峰值信号（原轨迹超限成本信号111.54→128.82、+15.5%，3个窗口端点<5而中间>5），return KL系数1、LR上限1e-5。这是联合工程策略，不声称单因素因果隔离或完整摩擦wrench平衡。奖励范数参考官方legged_gym、阶段教师指导参考Kickstarting，权重/几何投影为上述任务诊断支持的本地适配。

拟云pipeline `outputs/cloud_pipeline_m24_e003_balance_first_peak_20260914` / tmux m24_e003_balance_first_peak，独立60s smoke/600s PPO/每组180s评估，1536env/3072minibatch。上一批已全部完成，GPU串行；无实机操作。下一步验证峰值信号在批量PPO中生效、回位与低力/负载平衡联合结果；完整计划仍未达标，原成功教师不变。

## 2026-09-14 — KL smoke标准差广播修复

首次Strong5KL在独立smoke中exit1、未启动正式PPO：实际ActorCritic返回共享16维sigma，而辅助实现按batch×16索引。已用expand_as(mu)兼容共享/批量sigma，新增真实形状测试与PyTorch解析KL一致性通过。原失败pipeline保留，修复后新pipeline `outputs/cloud_pipeline_m24_e003_strong5_kl_fixed_20260914`、tmux m24_e003_strong5_kl_fixed；原教师重新开始独立60s/600s，未沿用失败状态。下一步核对正式启动与return KL日志，再按原五轮/5N评估。

## 2026-09-14 — 200Hz峰值奖励信号验证通过；KL600s已启动

原教师seed43开关对照各独立180s上限推理exit0、均完成五轮；197条控制trace所有共同字段逐项严格相同。每个10物理步窗口的整指净接触峰值与新增force_cost_peak误差最大9.54e-7N；峰值17.09995/8.84922N，控制端峰值16.71490/8.44840N。证据 `outputs/cloud_pipeline_m24_e003_peak_audit_20260914/verification.json`，原始1970步/组完整同步。测量子目标通过，不代表5N通过。

云Strong5KL已启动，pipeline `outputs/cloud_pipeline_m24_e003_strong5_kl_20260914`，tmux m24_e003_strong5_kl；独立60s smoke/600s正式/180s逐组评估，1536env/3072minibatch、LR上限1e-5，回位KL系数1。仅MSE→KL单因素比较，峰值开关保持关闭。下一步检查辅助loss与BEST/LAST/周期候选五轮成功率，若保护有效再将已验证峰值成本加入独立比较；不接受失败截断产生的低力。

## 2026-09-14 — Strong5回位退化已定位，KL保护进入对照

Strong5Anchor600s完成并自动同步：LAST固定0/128、43/44=42/32，BEST固定0、46/34；固定全部return12，拒绝选用。return MSE约0.0068并不等于足够准确。独立诊断pipeline `outputs/cloud_pipeline_m24_e003_return_ref_20260914` 仅phase2使用原d42e98a2，其余用Strong5 LAST，固定恢复128/128、seed43=64/128（41 drive10、23 return12）；两次180s上限推理exit0、档案已同步。明确标记single_policy=false/acceptance_eligible=false，不用混合结果冒充单策略验收。

新增可选return_anchor_loss=kl：冻结原教师均值/标准差，对8有效关节的return样本计算解析KL(teacher||student)均值，保留MSE默认；与torch.distributions逐项一致、冻结目标和非return梯度屏蔽测试通过。依据Kickstarting https://arxiv.org/abs/1803.03835 的辅助教师指导；阶段限制是本地适配。原教师active sigma约0.146–0.438，原MSE未按动作分布尺度约束，也未保持方差；可检验假设是相同系数1的KL能减少已被诊断证明的回位漂移。Strong5KL仅改变辅助loss形式，force0.25、其它奖励/物理/LR1e-5上限/原初始化均保留，拟独立60s smoke/600s云PPO，不能预称成功。

另完成默认关闭的物理周期峰值成本接口：整指逐body净接触向量范数和，在控制周期内取最大，nut-only控制端法向力作为下限。它包含非nut接触且不能解除body内部向量抵消，非完整摩擦wrench。单元验证中间8N不会被末尾1N隐藏、每周期清零、门槛/gates保持；目前云 `m24_e003_peak_audit` 正以原教师进行开关对照与200Hz审计，不在KL训练中启用，避免混淆归因。首次上传driver目录不存在已补建后重试，未发生训练失败。

当前子目标是修复回位保护和验证峰值测量；完整5N/卸载协调/法向平衡尚未联合通过。下一步先确认峰值审计轨迹一致，再执行Strong5KL受控对照，保留原成功教师。无实机操作。

## 2026-09-14 — Capped600s拒绝，转入更强限力与回位保护

Capped600s实际LR始终<=1e-5，完整训练/评估完成。BEST固定128、43/44=104/108（212/256），扰动峰值18.1252/12.5012N；LAST固定128、100/108（208/256）。LAST seed43新增22回位超时、6workspace失败，不能以低力/较低积分接受。周期模型保存epoch1至约146，完整正式目录自动同步；未替换原教师。

下一假设由两条独立诊断支撑：原5N代价0.05只扣原轨迹总奖励约2.55%，而600s后仍高力；持续微调仍遗忘回位。Strong5Anchor在Balance5基础上仅提高force_excess_scale至0.25（原轨迹约28.21、总奖励约221.18的12.75%），同时启用已测试的return_anchor_coef1.0辅助MSE，限定raw phase2/8有效关节；单位系数是初始工程比较值，不称论文最优值。此轮为联合工程改进，不宣称分别识别两个改动的因果效应。方法依据与测试见前节Kickstarting本地适配说明，仍是单个PPO网络，部署不需要教师运行时混合。

保持Kp2.4、物理/摩擦/5N门槛/五轮gates、LR上限1e-5/KL0.005，原d42e98a2初始化。云独立60s smoke/600s PPO，1536env/3072minibatch，每60s保存周期候选；pipeline`outputs/cloud_pipeline_m24_e003_strong5_anchor_20260914`，tmux m24_e003_strong5_anchor。云任务切换前上轮已完成；不执行实机。下一步验证云端冻结参考目标、非零辅助loss与实际force/成功率，继续直到联合目标通过或发现必须升级解决的实际阻塞。

## 2026-09-14 — Balance5短程结果及真实学习率上限

Balance5短训完成：LAST固定128/128，43/44=113/121，总234/256与原相同；扰动峰值20.8594/11.7584N（原四种子24.2553/11.8029），但驱动失联均值0.1281/0.1023s，不接受5N完成。BEST固定128、123/121，固定食指峰值21.5844N且驱动失联升高，不能仅按成功数选择。保留候选与原教师，计划仍未完成。

诊断early120s TensorBoard info/last_lr共30记录：初始1e-5、实际最大7.59375e-5、末5.0625e-5。自适应scheduler的默认上限1e-2导致“初始LR保守”不能限制后续漂移。新增可选max_learning_rate，Capped配置固定上限1e-5且保留调度下限1e-6；实际scheduler抽取测试100次升/降都遵守界限。下一轮600s观察受控学习率下的持续学习，其他Balance5奖励和物理不变。每60s额外保存周期模型，并同步完整正式目录，避免只靠training BEST选择。

已准备默认关闭的return_anchor辅助模块：复制冻结教师及全部观测/priv/point归一化，只在原始phase2回位数据上计算8个有效关节动作MSE，其他阶段无辅助梯度；真实ActorCritic fixture验证目标精确、教师/归一化冻结、非回位和锁定指无梯度通过。依据Kickstarting(https://arxiv.org/abs/1803.03835)的辅助教师指导动机；阶段限制和连续动作MSE是本地适配，并非论文CE公式。当前Capped实验不启用此项，先区分学习率效果，未改变PPO基本算法/默认行为。

云pipeline`outputs/cloud_pipeline_m24_e003_balance5_capped_20260914`、tmux m24_e003_balance5_capped，独立60s smoke/600s PPO，1536env/3072minibatch，原教师d42e98a2。下一步核对实际LR上限、周期模型与完整五轮/力指标，再决定是否需要回位保护或更强5N超限代价。完整wrench传感器仍未验证：已知约5N接触在4/8位置迭代下读约25/45N，禁止直接作物理验收。

## 2026-09-14 — Balance5法向平衡局部目标实现

用户持续目标仍active，5N和完整摩擦wrench未验收。柔顺性诊断显示Kp2.0单例五轮但200Hz峰值17.78/8.91N未全面优于2.4，1.6/1.2/0.8初次drive超时，故不选降低刚度方案。

新增Balance5继承Force5，只加入0.025*(||F_normal_transverse||/5N + ||M_normal_tilt||/0.05Nm)，使用已有nut-specific接触记录，不使用未通过标定的solver传感器。投影排除螺杆轴向，保留有效轴向驱动力矩；法向代理不等同完整摩擦wrench。代价仍为官方legged_gym力范数惩罚的本任务几何适配，cross product/轴向投影不是论文奖励公式。依据教师200Hz峰值netnormal14.95N、normalmoment0.145Nm；新代价均值0.02865/step，峰值0.12449，policy等效五轮累计5.644，与Force5基线累计5.64同量级。假设减少不对称加载可同时减少固定约束负担并改善低力驱动，而非仅同高力卸载。

单元验证通过：相反法向力抵消、同向轴向扭矩不惩罚、横向/倾覆代价、旋转轴变换一致、原gates不变；Force5回归通过。task默认关闭兼容所有旧配置；不改质量/摩擦/控制/观测。新A0c+60s smoke+120s PPO，原d42e98a2、Kp2.4、1536env/3072minibatch、LR1e-5/KL0.005不变。pipeline`outputs/cloud_pipeline_m24_e003_balance5_20260914`，tmux m24_e003_balance5；下一步BEST/LAST固定/43/44和逐物理步联合筛选。

## 2026-09-14 — 传感器已跑通但支撑反力校验未通过；继续柔顺性诊断

Wrench probe两组均完成五轮。教师200Hz峰值17.09995/8.84922N，与原基线一致；early LAST env15偏移重放峰值19.94481/13.05164N，但n1轨迹没有复现n128的13.9588N端点峰值，不能当精确轨迹复现。新增sensor不改变教师法向峰值；完整trace字段差异仍需逐项核查。

base solver sensor出现约130N，而nut横向约0、bolt全0；独立无手/无接触/零重力的M24资产施加5N横向外力，solver-only传感器仍约0，未通过支撑反力标定，因此不得用于完整平衡奖励或验收。首次校验缺libpython3.8，显式LD_LIBRARY_PATH后完成，保留失败与重试log。此结论只说明当前API/配置不隔离所需反力，不宣称物理力矩为0。官方force sensor包含内力说明参见本地IsaacGym文档，完整wrench仍待解决。

当前下一子目标：云端冻结d42e98a2，Kp2.0/1.6/1.2/0.8/0.4单环境seed43各独立180s推理，Kd0.01、摩擦/gates/目标积分均不变，200Hz载荷audit。选取比例递减覆盖约1/6原刚度，检验高刚度是否使5N短训受限，不作为实机标定；以五轮与力峰值共同判断，低力不动直接拒绝。pipeline`outputs/cloud_pipeline_m24_e003_gainlow_20260914`、tmux m24_e003_gainlow，无新训练。每完成测量/实现子目标自动提交；下一步根据柔顺性结果决定受力PPO配置，并继续建立可验证的非驱动方向测量。

## 2026-09-14 — Force5短训未达标，启动200Hz传感器核验

Force5 60s smoke/120s PPO及全部评估完成，LAST固定128、43/44=114/118，峰值24.6978/12.5160N；BEST固定128、120/123，峰值23.3472/15.5667N。均远高于5N，不接受，不以reward添加完成冒充计划完成。完整pipeline`outputs/cloud_pipeline_m24_e003_force5_20260914`，checkpoint和自动同步保留。

下一测量子目标已启动：`outputs/cloud_pipeline_m24_e003_wrench_probe_20260914`，tmux m24_e003_wrench；两个独立180s上限的推理（非训练）：教师seed43，以及early LAST的seed43/env15初始偏移单环境重放。真实偏移逐项检查，保留教师priv初始参考；n128→n1的重放不保证轨迹逐位相同。新增e003_wrench_audit默认关闭，仅创建base/bolt/nut三个world-frame solver-only力传感器。原200Hz audit保存solver_wrench，明确含joint reaction，不预先称完整接触或支撑wrench。依据本地官方IsaacGym docs/_sources/programming/forcesensors.rst.txt。新分析器同时报告20Hz/200Hz每指超限峰值/时长，不按均值判通过。

下一步：验证传感器读数和轨迹不受监测影响，再据结果选择控制柔顺性/5N惩罚改进。物理摩擦0.2不是直接已标定0.2Nm抵抗，不能据此声称5N不可行；不通过降低摩擦达标。最新PhysX文档仅作为解释参考，不把新版能力假定在旧IsaacGym可用。

## 2026-09-14 — 用户要求持续完成计划；5N超限目标开始实现

用户明确要求继续直到优化计划完成；已建立持续目标，不把一次短训完成视作计划通过。当前目标：先修正双指同高力的奖励缺口，再建立完整反力矩诊断；每目标验证后自动Git提交，原成功教师保留，无实机操作。

定位early LAST seed43/env15/step158/cycle3：phase_before=release，两指16.0410/13.9588N，但力差只有2.0822N。单纯差值代价对共同高力不敏感。新增默认关闭force_excess_scale，Force5配置在全部阶段扣0.05*sum(max(F_i-5,0)/5)，保留卸载差值0.1。参考官方legged_gym `_reward_feet_contact_forces` 的超阈值线性形式：https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py 。每指法向合力/归一化和本地权重为任务适配；没有裁剪仿真力，也不是硬约束。

权重证据：原nominal每episode归一化超限累计112.83（drive85.91/release26.90），0.05约扣5.64，与五轮驱动事件总奖励50相比有限，避免原全阶段平方载荷成本造成的严重回位退化。假设短程微调能降低共同加载而保住成功率。实际reward单元验证两指高力、5N边界、零力及全部phase/gates不变通过，卸载原测试回归通过。

云已读live handoff并确认RTX4090/49140MiB空闲；新A0c，独立60s smoke/120s PPO，1536env/3072minibatch，LR1e-5/KL0.005，原d42e98a2起点；pipeline`outputs/cloud_pipeline_m24_e003_force5_20260914`，tmux`m24_e003_force5`。完成后同种子BEST/LAST评估；5N需逐物理步验收，20Hz筛选绝不代替200Hz峰值。下一步分析此轮并补充横向合力/倾覆反力矩及卸载同步诊断。

## 2026-09-14 — 短程PPO评估完成，形成协调/峰值取舍证据

早期实验`outputs/cloud_pipeline_m24_e003_unload_early_20260914`独立60s smoke/120s PPO均正常限时退出；A0c及BEST/LAST各固定/43/44评估全部完成。云GPU0MiB/0%，实例开机空闲，未操作实机。未继续扩展种子或延长训练，因为当前联合门槛未通过。

|候选|固定五轮|43/44五轮|固定离触时间差s|固定单指冲量Ns|43/44峰值食指/拇指N|
|---|---|---|---|---|---|
|原成功教师|128|115/119|0.33414|1.67074|需按原两种子核对；四种子最大24.2553/11.8029|
|120s LAST|128|116/119|0.28180|1.20934|22.1629/13.9588|
|120s BEST|128|125/125|0.33633|1.61510|23.6003/11.9598|

LAST固定单指冲量下降27.6%、时间差下降15.7%，但拇指峰值上升；BEST扰动250/256但离触时间差不改善，且拇指峰值仍高于原四种子最大值。BEST扰动F²积分710.29/699.33N²s，也不能只凭成功率称低力更好。两个候选均未通过“成功率保持+卸载改善+峰值不升高”的联合验收，原成功教师不替换。20Hz的第一离触指标仍非严格同步或完整摩擦力矩，缺失/重接触需进一步诊断。短程样本和相同种子比较只支持本轮工程结论，不宣称统计显著普遍提升。

证据：pipeline内`unload_last.json`、`unload_best.json`、`loads_last.json`、`loads_best.json`及原始evaluation.json；模型路径/SHA在status.json；本地后台监控自动同步全部pipeline及候选模型。上轮300s退化和本轮120s结果均保留，按用户规则自动提交本目标结果。

唯一推荐下一步：定位LAST seed43拇指13.96N峰值的阶段/接触状态并核对200Hz接触反力矩，再设计以5N超限惩罚补足“等力但共同大力”的缺口；保持早停/回位能力约束。不能直接给双指差值奖励加大权重或放宽成功判据。5N和横向/倾覆力矩仍未实现。

## 2026-09-14 — 卸载300s候选拒绝，转入短程保持性验证

已完成首轮卸载PPO评估及本地同步。BEST固定128/128，但seed43/44仅83/89（172/256），原教师115/119（234/256）；LAST固定70/128、扰动81/85。主要退化为回位超时，BEST两种子81次回位超时及3次卸载超时。BEST固定卸载滞后0.27836s（原0.33414），单指冲量1.04987Ns（原1.67074），改善约37%；扰动峰值25.0505/13.0327N也不满足不升高门槛。不能用减少的成功回合数或缩短的失败轨迹伪装低载荷改善。两候选均拒绝，不替换d42e98a2+Kp2.4。

证据：`outputs/e003_unload_candidate_20260914.json`、`outputs/e003_unload_loads_20260914.json`及pipeline各evaluation.json；自动同步receipt已完成。已完成分析目标，按用户要求自动提交。

下一实验假设：300s微调导致回位策略漂移，120s较早候选可能保留原能力。只改正式训练时间，LR1e-5/KL0.005/卸载权重0.1/物理/gates均不变；从同一冻结原教师重启，非续训失败模型。runner新增--formal-seconds并持久记录。独立60s smoke、120s PPO，1536env/3072minibatch，自动BEST/LAST固定/43/44评估，root`outputs/cloud_pipeline_m24_e003_unload_early_20260914`，tmux`m24_e003_unload_early`。云实时GPU0MiB/0%，无本地训练。下一步按相同验收条件判断早期候选，未通过则继续定位回位退化而非盲目延长。

## 2026-09-14 — PPO卸载协调有界实验

已提交学生基线c59a89b，用户授权切回PPO并要求每个完成目标自动提交。诊断已完成：nominal640次卸载全部拇指先离开，平均差0.3341s；seed43/44为0.3385/0.3371s，单指期峰值23.0999N。证据`outputs/e003_unload_baseline_20260914.json`，分析脚本`analyze_m24_e003_unload.py`，20Hz采样且缺失离触不按零处理。

假设：仅卸载阶段惩罚0.1*|F_index-F_thumb|/5可减少单指残余冲量和卸载时间差。5N为用户目标的归一化参考，不是假称硬约束；按基线单指冲量1.67Ns，离散累计惩罚约0.67，与原卸载事件奖励1同量级。采用线性力代价形式参考legged_gym官方`_reward_feet_contact_forces`：https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot.py 。阶段限定和双指绝对差是本任务局部改写，不是论文原式。它不能保证双指共同轻压，不能表示完整力矩；5N上限和横向/倾覆力矩目标继续保留。

新E003Unload继承成功Kp2.4配置，只新增默认关闭的卸载代价；事件、物理、摩擦、观测不变。与已拒绝的全阶段平方载荷成本H2不同。真实reward测试通过：阶段选择、两指对称、等力零成本、原门限不变；旧load测试亦通过。云4090/49140MiB当前空闲；沿用已稳定1536env/3072minibatch，保守LR1e-5/KL0.005以控制冻结基线漂移。独立60s smoke/300s PPO，timeout124预期，BEST/LAST各固定/43/44评估。驱动`scripts/run_m24_e003_unload_probe.py`，tmux`m24_e003_unload`，root`outputs/cloud_pipeline_m24_e003_unload_20260914`。

验收：固定128/128，扰动至少不劣于教师相同种子，且卸载时间差/单指冲量下降、峰值不升高才扩展验证。失败候选不覆盖成功基线。下一步：云端新A0c、smoke/PPO及候选评估，按实际结果决定后续优化；未操作硬件。

## 2026-09-14 训练完成 — E003学生已导出，低力保持尚未通过

本节取代下方“正式训练运行中”的历史状态。云端正式训练3600s已完成，35158次更新、2250176 agent steps；smoke/formal退出124均为独立时限的预期结果。pipeline于北京时间13:16完成评估/导出，13:19完整产物自动同步本地。

- 选择正式LAST为学生候选（SHA b9f20a4326937be146ce786b1252686a71e0baef88a6677dfd6ea257ae9bc55d）；BEST按训练奖励保存但评估退化，未采用。
- 固定五轮：教师128/128，学生128/128。±1°扰动四种子：学生110/110/110/113，共443/512（86.52%），教师468/512（91.41%）。
- 同口径20Hz法向合力峰值：扰动教师食指/拇指24.2553/11.8029N，学生38.6106/12.5845N；学生低力保持未通过，不能仅凭低蒸馏loss或固定成功率接受。5N与双指受力/力矩平衡仍为后续必做目标。
- 原生录像记录210帧、20Hz、五轮完成；视频编码和视觉审核仍待完成。TorchScript导出及210步原生动作一致性通过（最大误差0），但101维观测仍需外部阶段命令，原M24实机适配尚未完成；未操作硬件。
- 学生：`outputs/cloud_pipeline_m24_e003_student_20260914/selected_student/student_candidate.ckpt`；导出：同pipeline的`export/student_phase101.pt`；状态/评估/同步证据均已在pipeline本地保存。冻结教师及原成功标签保持不变。
- 本次进度核查：读最终status、评估、export manifest和automatic_sync_receipt，并实时检查云端无tmux、GPU 0MiB/0%；实例仍开机空闲。此次仅更新报告、计划和交接/学生状态，没有新训练或Git提交。
- 唯一推荐下一步：定位学生高力峰值和扰动失败，比较已保存的中间学生checkpoint，再决定蒸馏修正；优先恢复教师低力与鲁棒性。

## 2026-09-14 — E003学生smoke完成，正式训练运行中

- 模型契约测试、新A0c、冻结教师128环境固定五轮全部通过；教师固定128/128，食指/拇指峰值16.6735/9.3674N。
- 独立300s学生smoke已按时退出124，2882次更新/184512steps，最后latent loss0.005534、BC loss0.021637、总loss0.027171；每次检查冻结教师不变，LAST完成更新后保存。smoke学生SHA`c8b0c18272c8e8fc5b363820360fdb5a4514933717595480b670a2ebadd5b74d`，路径`outputs/XHandPasiniM24NutBolt_student/cloud_pipeline_m24_e003_student_20260914_smoke/stage2_nn/model_last.ckpt`。
- 独立smoke学生固定起点评估128/128完整五轮，驱动拇指失联均值0s；但法向峰值20.5028/11.7688N、F²时间积分815.12N²s高于教师668.92。只确认训练和固定动作可行，**不标记完整低力保持或学生sim2real验收通过**。
- 正式训练已于2026-09-14 12:10左右（UTC+8）开始：独立3600s上限，64env/seed42，从同一冻结教师重新开始PAdapt，仅适配网络，不从smoke继续累计预算。预计13:10左右到达训练时限，随后自动BEST/LAST评估、额外种子/200Hz/native/导出；这些不是已完成结果。
- 当前云tmux`m24_e003_student`，状态`outputs/cloud_pipeline_m24_e003_student_20260914/status.json`。本地独立监控`scripts/monitor_m24_e003_student.py`（PID见root/local_monitor.json，日志`/tmp/m24_e003_student_monitor.log`）每45s记录状态、每5min复制正式学生BEST/LAST，pipeline结束后自动同步完整产物并写automatic_sync_receipt。后台监控不启动训练/硬件，不自动关机。
- 本轮新增E003Student配置、有界学生wrapper、精确教师点云latent目标开关（默认关闭）、学生评估/录像选择、phase101导出和自动同步。CPU真实模型方法fixture、Python语法、git diff --check通过；云端契约/A0c/教师/学生smoke评估退出0，smoke124为预期。未提交Git；原成功标签不变。
- 101维学生仍显式需要阶段命令；仿真状态机供给不等于实机阶段生成已实现，导出会明确标注，旧96维入口仍需适配。当前不加入5N/平衡新奖励或轴向进给，不操作硬件。
- 唯一下一步：等待正式学生及自动评估结果，优先比较完整五轮和受力保持，再进行原生录像审核与实机接口适配。不能凭smoke128/128直接采用，不能隐藏力峰值上升。

## 2026-09-14 — 用户授权开机并开始E003学生训练

当前子目标：从已成功的E003Compliant/d42e98a2教师训练第一版ProprioAdapt学生；用户已明确切换到学生阶段并确认开启pro-786095bb6e03。后续5N与双指平衡目标保留，当前不改教师/奖励/物理。

云端RTX4090/49140MiB，开机API成功且running，启动检查无tmux/GPU占用。读取了实际cloud handoff及原验收路径。采用原M24 PAdapt的64env、32维proprio×30历史、seed42、仅adapt_tconv训练；独立300s smoke和3600s正式训练，各自timeout，124为预期。每60s原子保存LAST和冻结教师检查，保留BEST；正式后自动BEST/LAST固定/43/44评估，选候选后45/46/200Hz/native及TorchScript推理一致性。

学生接口明确保留101维观测：96维历史加外部阶段命令5维；仿真评估由原状态机供给。学生免去priv_info/点云输入依赖，不自动免去阶段命令。导出接口显式要求阶段输入，未接实机状态生成，不称即插即用。原实机位置模式与仿真Kp2.4的映射仍单列，未操作硬件。

兼容性诊断发现旧PAdapt教师目标对点云latent未执行与PPO相同的tanh；新增默认关闭teacher_point_tanh开关，仅E003Student打开，以匹配冻结教师真实动作。精确动作/latent目标、无priv/点云学生动作、旧目标差异fixture通过；不是奖励改动或算法替换。通用评估/录像仅新增ProprioAdapt分支，PPO默认保留。

pipeline：`outputs/cloud_pipeline_m24_e003_student_20260914`；driver：`outputs/cloud_pipeline_m24_e003_student_20260914_driver/run.sh`；tmux：`m24_e003_student`。开工准备中，尚无通过验收的E003学生。下一步：完成云端预检与smoke，确认loss/载荷和存档有效后继续正式学生训练及自动评估。

## 2026-09-14 — 已找到sim2real分支M24入口；建议先建立E003学生基线

只读核对本地`sim2real`分支commit`4736d94bd4634240b2dc93ff2a5b45d272936440`，未切换/合并分支：
- `scripts/deploy_m24_student.py`：专用DexH13右手16DOF，20Hz，关节顺序与已存M24初始位、SDK限位、动作mask、日志和dry-run；原PAdapt学生TorchScript部署。
- `scripts/export_m24_student.py`：原生与TorchScript在线输入一致性校验。
- `scripts/cloud_pasini_m24_student_padapt.sh`：原M24 ProprioAdapt云训练流程。旧脚本固定原任务/旧评估和时限，不能未经适配直接用于E003。
- 更正先前信息不足：原M24专用部署确实存在，当前分支的12DOF通用XHand脚本不代表原M24能力，不需要重建整套硬件流程。

建议下一阶段顺序（本轮为建议，尚未指令启动学生训练）：冻结成功E003教师及Kp2.4 → 复用旧M24链路核对新增接口 → 训练/验收一版E003 ProprioAdapt学生并完成导出与离线部署检查 → 在实机验证任务中收集差异 → 继续5N与力矩平衡教师优化，再更新学生。5N/平衡仍是必须实现的后续目标，不作为本次已接受成功基线开始学生验证的前置条件。

训练前需明确两个实际差异：原M24部署OBS_DIM96，E003为101（新增阶段/时钟）；现有PAdapt不能被假定自动移除这些观测，必须明确阶段信息在实机端的来源或另行验证学生输入适配。原实机入口通过SDK发送位置目标，仿真Kp2.4不会随模型权重自动传到硬件；需沿用并核对已有控制接口/动作缩放，而不是声称权重导出即保留低力。学生需验证完整五轮、失联与受力相对教师的保持，而不只看蒸馏loss。

依据：项目原流程也是oracle→PAdapt学生→旋转策略部署，见[DexScrew官方README](https://github.com/x-robotics-lab/dexscrew/blob/main/README.md)。本次学生优先的排序是针对当前成功基线与尚未闭合部署链路的工程判断，不是论文规定。

## 2026-09-14 — 用户确认后续5N与受力平衡目标，排除螺母进给

本节优先于下方历史讨论。保留`e003_low_force_cycles`成功及sim2real资格，Git基线`5e7e965`、标签`e003_low_force_cycles_success`不变。

### 后续必须实现的目标

| 目标 | 实现和评价口径 |
|---|---|
| 接触力5N以内 | 以每个手指对螺母的法向力峰值<=5N为当前明确工作口径；不是两指之和或全接触合力。后续以逐物理步峰值、超限时长、P95及完整五轮完成率共同验收，保留旋拧/释放/离触回位要求。当前尚未实现，不通过截断物理求解器反力来满足数值。若改为真实传感器总力口径，须显式更新定义。 |
| 双指受力与力矩平衡 | 减小非期望横向合力和倾覆力矩；保留两指绕螺杆轴同向驱动的有效力矩，避免互相抵消。驱动阶段协调加载、释放阶段协调卸载，记录单指持续加载与脱离时间差。螺栓仍固定，用接触/基座反力及反力矩评估，不能用螺母不晃替代测量。完整摩擦wrench尚未建立，具体阈值需由诊断确定，不能把法向力矩代理当完整平衡。 |

### sim2real范围与学生状态

- 用户明确螺母轴向进给在真机任务中不用考虑：从后续目标与迁移前置项移除，不启动螺旋关节、轴向跟踪或进给补偿开发。已有简化模型说明保留为事实，不能再将它作为当前任务障碍。
- 优先参考并复用原M24的实机流程、已有关节映射/初始化/控制和保护设置；先由agent核对实际入口与E003增量兼容性，不要求用户重新提供整套硬件信息。
- 已检查的本仓库通用`xhand-deploy/xhand_deploy.py`是12DOF/96维观测，不能当成已验证的16DOF M24入口；E003配置为16DOF/101维观测（含阶段命令）。这只说明该通用脚本不直接匹配，尚未证明原M24实机链路缺少相应功能；不得臆造新的传感器采购要求。
- **E003没有训练/蒸馏得到对应学生。** 选中的是原d42e98a2 PPO教师和Kp2.4配置，两轮新增学习仍是PPO教师实验且未采用。历史其他任务的学生不能标作E003学生；用户本轮是在询问状态和安排后续目标，未指令启动学生训练。

唯一下一步：在保留成功基线的前提下设计5N与双指平衡的测量/有界验证方案，迁移时先核对原M24入口的16DOF及阶段输入兼容。本轮仅更新计划与记录，未改运行代码、奖励或门槛，未训练、未开云、未执行硬件。

## 2026-09-14 — e003_low_force_cycles成功，具备sim2real验证资格

用户明确接受当前`e003_low_force_cycles`为成功里程碑，并授予进入sim2real验证的资格。正式记录：`docs/e003_low_force_cycles_success.json`，绑定E003Compliant、Kp2.4、原d42e98a2模型与原生录像SHA。这个决定不改写既有测量；实机执行效果/负载安全尚未实测。后续改进不撤销本次成功标记，不要求先消除全部仿真失败才允许开展迁移验证。

当前讨论优先项：每指5N低力目标的物理可行性和控制限载；双指合力/非旋转轴力矩的协调；真实螺纹轴向进给与负载、实机阶段状态获取；剩余42回位超时/2工作空间失败。5N暂为研究提议，尚未作为已实现的指标或改入奖励；法向力与总接触力须区分。现有奖励鼓励双指接触驱动和最终双指离触，不直接最小化完整合力/倾覆力矩，也不约束同一时刻脱离。

本次仅做Git里程碑保存与代码/物理解释，未训练、未部署、未开云实例；云端最后确认状态为2026-09-11的shutdown。唯一下一步：从这个成功基线制定实机载荷与阶段观测映射，并验证5N目标所需扭矩/摩擦余量；新训练或硬件执行另按具体任务推进。

## 2026-09-11 — E003首轮低力改善已验证，保留为当前仿真配置

本节更新下方阶段安排。当前选用 **`XHandPasiniM24NutBoltE003Compliant` + 原d42e98a2 PPO**，唯一生效差异是Kp从3降至2.4；摩擦、Kd、动作、奖励和成功门槛保持。原E002任务/模型继续作为固定对照。

- E003-0：已增加可选200Hz控制/接触监测并核验轨迹不变。资产effort从1降至0.001不改变直接力矩轨迹，不能视为限矩保护。源凸包重叠与接触运动代理量已记录，尚未获得实际PhysX cooked形状穿透上限或硬件安全限值。
- E003-1首轮结果：固定五轮128/128；四组独立±1°偏移432→468/512（84.38%→91.41%）。配对恢复56、退化20；驱动拇指失联均值0.2743→0.1102s。20Hz扰动批次食指/拇指峰值34.123/14.882→24.255/11.803N，平均受力平方时间积分下降46.5%。这证明相对改善，不代表已找到最小力或零滑移。
- 单环境seed43的200Hz峰值26.660/12.522→17.100/8.849N；原生197帧、20fps、9.85s连续五轮录像及20个事件关键帧已核验。所有批次成功均保留20°有效驱动和离触回位条件。
- 目标误差硬投影和两轮力成本PPO均未通过完整动作比较，保留为否定证据，不采用。每轮独立60s smoke/300s PPO；没有扩大训练时长。
- 当前剩余44/512失败：42回位超时、2工作空间违规。E003整体尚未完成；接触切向滑移、实际穿透深度、实机负载保护仍需继续。

验收与精确模型绑定：`outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/acceptance.json`、`selection.json`；报告：`docs/m24_e003_low_force_2026-09-11.md`。云端工作已结束，本地/云端保留结果；未部署实机。

**唯一下一步：以E003Compliant为对照，诊断42个回位超时及20个配对退化的实际状态，优先做兼顾低力与回位恢复的有界比较；同步保留接触滑移/穿透诊断，不继续盲降刚度或重跑已拒绝的力成本设置。**

## 2026-09-11 — E002初步完成，进入E003完善与改进（阶段起始记录）

本节是当前阶段安排，优先于下方历史阶段优先级。E002以“学会完整重复旋拧”初步收尾；E003进入计划与诊断准备，改善受力、接触配合和恢复能力。本次阶段划分没有启动PPO，也没有改变奖励、控制器或硬件限值。

### 保留E002成果作为固定对照

- 代码提交：`cb61dc6`。任务：`XHandPasiniM24NutBoltE002RobustTime`。不重命名或覆盖既有入口、模型和录像。
- checkpoint：`outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`，SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`。
- 已验证：单一PPO完成“拧动→释放→无接触回位→再抓取”连续五轮；固定姿态128/128，独立±1°关节偏移324/384；真实仿真录像验证五轮、151.78°净转角。验收证据：`outputs/cloud_pipeline_m24_e002_robust_time_20260911/acceptance.json` 与 `visual_verification.json`。
- 阶段完成范围是上述仿真运动能力；不包含实机安全认证、任意初始姿态或超过五轮的可靠性。现有受力与回位问题转入E003，不要求先把E002做到零失败。

### E003目标和顺序

| 子目标 | 本任务已有证据 | 实施与评价 |
|---|---|---|
| E003-0：受力监测与控制能力核验 | 食指法向合力峰值34.12N；现有20Hz日志不覆盖子步冲击、全手碰撞和实际关节负载 | 先增加可选的逐物理步PD请求/实际施加或可观测关节力矩、全部接触载荷、持续受力、饱和及关节行程日志，区分请求值、测量值和估计值；核验力矩控制下的effort限幅行为。记录固定基线，暂不猜测硬件安全上限。 |
| E003-1：滑脱恢复与协调卸载 | 拇指失联约1.65秒时食指仍加载而螺母停滞；第五轮释放时食指21.55→25.76N | 诊断驱动中的接触丢失、继续加载及目标积累；比较恢复双指接触/卸载行为，减少无效持续挤压与释放加压。主动释放允许离触，不统一惩罚所有失联。 |
| E003-2：回位恢复与稳定重复 | 384次中47个回位失败、10个工作空间失败、3个驱动超时 | 从实测失败状态诊断并学习回位恢复，保留原始起点与已有成功动作。随后逐步扩大独立种子、初始偏移和循环次数，分别报告各条件结果。 |

### 评价和推进原则

- 同时报告完整五轮完成率、净转角、驱动失联时长、无效加载时间、释放阶段力的增幅、力峰值/P95/P99、受力持续时间及可获取的关节负载。20Hz法向合力与更完整的新测量分开列示，不能混合比较不同统计口径。
- 用相同checkpoint、起点、种子及成功定义作配对对照。先完成一个有诊断依据的局部改变，再做独立评估；需要学习时保持每次独立60秒smoke和300秒PPO。早期改善不以短训达到某个成熟成功率为推进门槛；同时披露运动能力和受力之间的取舍。
- 不把当前官方15N指尖输出参数直接作为损坏阈值。真实限力/限矩保护需要具体硬件版本、方向、持续时间及厂家允许值。限值未确认不阻止仿真监测和相对改善，但不能宣称实机安全或直接部署。
- 保留真实净转动、驱动接触配合、释放/回位、可重复性以及模型/资产/源代码完整性。不要重新引入未经验证的法向力矩一票否决，也不要凭“成功了”忽略明显持续挤压。
- PPO教师仍是主线，螺栓基座固定。训练优先云端；资源不可用时按用户已授权例外记录原因后使用本地。无自动30分钟训练、学生蒸馏或MuJoCo阶段扩展。
- 学习设计依据：[Tac2Motion II-B/II-C](https://arxiv.org/html/2509.17812v1)的接触、释放、旋转及控制代价结构；[DemoStart III-A](https://arxiv.org/html/2409.06613v1)的起始状态课程与原任务分布评价。它们为方法提供依据，不提供本硬件的力限值；每个本地公式/权重变化需另列实测假设与对照。

**唯一下一步：E003-0，增加受力与控制饱和的诊断记录并核验现有力矩限幅行为，使用冻结的E002模型建立更完整的力基线。** 同时等待具体实机允许负载信息；完成诊断前不因阶段改名而直接扩大训练时长。详细既有证据见 `docs/m24_e002_force_audit_2026-09-11.md`。

## 2026-09-11 -- Timing correction verified; five-cycle robustness84.4% across384 starts

- Current selected combination: **task `XHandPasiniM24NutBoltE002RobustTime` + unchanged PPO** `outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`, SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. Older tasks/checkpoints remain preserved. No PPO trained this session; no reward-weight or physical-success-threshold change.
- Diagnosis of31 old drive timeouts:27already reached20degrees actual net stroke but none reached20degrees counted pair-contact progress. Doubling execution time with unchanged policy-clock units recovers24of31 into actual fullfive-cycle successes. Do not count mere20degree net angle as a success; the original paired-contact stroke and release/return conditions remain required.
- Optional `E002Cycle.phase_clock_scale` defaults1 and preserves legacy behavior. New task uses phase execution budgets160/80/120/80steps (8/4/6/4s),clock scale0.5 preserves original80/40/60/40input units and saturation. Goal5cycles,episodeLength2200. Exact-clock, delayed-release, reset, full-cycle/no-recycled-progress tests pass. Evaluator adds opt-in joint-state/target recording and timeout-only diagnostic controls.
- Frozen-policy formal evaluation: nominalfive128/128. Independent +/-1degree joint errors: seed43 **94→108/128**, seed44 **99→109/128**, unseen-at-budget-selection seed45 **91→107/128**. Combined284→324/384 (73.96→84.38%),all284previous successes retained,40recovered,no regressions. Original two validation seeds alone193→217/256 (75.39→84.77%). All successful episodes meet the original20degree pair-contact stroke and fully off-contact return criteria. Exact sampled reset arrays match. One seed44 diagnostic prefix had numerical divergence (env64,action0.05362/angle0.511degree); do not describe all batched trajectories as exactly identical.
- Native seed43,previously268frames/fourcycles/fifth-release timeout, now290frames/14.5s/fivecycles and151.780degrees net. All first268action/angle samples match the failed recording exactly;22additional steps finish release→off-contact return→regrasp. All20event keyframes and force/angle/phase plot inspected; native process exits0. MP4 under new pipeline `native_seed43/learned_perturbed_cycles.mp4`, SHA256 `be1c58c53805cddc5427d250b77bca7fa0a86c048291f962fcadbd4c02840db4`.
- Remaining60/384failures:47return timeouts,10workspace violations,3drive timeouts. In the original two seeds,29extended-time return failures are physically off-contact but21.75–67.88degrees from target (median24.16); none within6degrees. More time did not fix those geometric errors. Wider perturbations,beyondfivecycles and real-hand transfer remain unverified.
- Cloud outputs: diagnostic `outputs/cloud_pipeline_m24_e002_timeout_diagnostic_20260911`; formal `outputs/cloud_pipeline_m24_e002_robust_time_20260911`. Four diagnostic evals,five formal evals,native recording and fresh A0c all exit0/finite; newA0c JSON explicitly `passed`,source snapshot refreshed. Each cloud process bounded180s. Acceptance,selected-policy binding,visual receipt,source snapshots,commands,timestamps and raw artifacts preserved locally/cloud. Full report `docs/m24_e002_timeout_correction_2026-09-11.md`.
- Cloud instance pro-786095bb6e03 remains ON,all task jobs finished after final verification. Git still94e8007 plus previous robustness/current timing changes,uncommitted; no push. **Single recommended next step:** collect and diagnose the47late off-contact return failures,then teach recovery from those measured states with a bounded cloud continuation using this explicit task/checkpoint. Keep nominal and fixed held-out comparisons; do not keep increasing deadlines or rerun rejected two-cycle training. The older robustness runner hardcodes the old task; adapt its task/horizon/preflight wiring before using it for this next curriculum.

## 2026-09-11 -- Five-cycle robustness improved; bounded cloud work complete

- Git request completed first: `94e8007` (`feat: validate learned return and repeated M24 twist cycles`), 19 files; no push. Initial-state robustness changes were developed afterward and remain uncommitted.
- New selected PPO: `outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`, SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. Original repeated-cycle baseline SHA7742e40d remains preserved.
- Nominal candidate743: two cycles **512/512** (previous501/512), five cycles **128/128** (previous128/128). Independent index/thumb initial joint perturbations within +/-1degree, 128 environments each: seed43 **70→94**, seed44 **76→99** complete five cycles. Combined146→193/256 (57.03→75.39%). Exact baseline/post reset-offset hashes match; every successful return stays off-contact throughout. Paired outcomes:75 new successes,28 regressions,118 successes retained,35 failures retained. These two evaluation seeds were used for selection, not a final untouched test set.
- First two-cycle training comparison regressed and was rejected. Diagnosis found28/32 nominal failures after the two-cycle training boundary. Controlled follow-up extended training to five cycles/1100steps; same baseline start, +/-1degree reset noise,50%nominal mixture,seed42,1536envs/3072minibatch,LR0.0001. Each comparison had its own60s smoke and300s PPO; all four phases exit124 as expected. FreshA0c explicitly passed; six baseline and16 post-training evals exit0/finite. Reward weights and physical phase criteria unchanged. No local training,30minute run or student.
- Native cloud recordings both exit0, actual one-env first episodes at20Hz: seed44 completesfive cycles/20ordered events in189frames (9.45s),net150.458degrees; seed43 completesfour cycles then times out during fifth release in268frames (13.4s),net151.780degrees. Both preserved under `selected_perturbed_video_s43`/`s44`; MP4, event keyframes and force/angle/phase plots verified. Seed43 ends off-contact but without completing the required release dwell before timeout; it is not counted as a five-cycle success. One-env reset draws differ from128-env eval batches.
- Remaining selected-model perturbed batch failures:31 drive timeouts (19third-drive,12fifth-drive),27 return timeouts,5workspace violations. Full initial-state robustness, more thanfive cycles,full160mmtravel and real-hand transfer remain unverified. Independent reset helper tests, configured-seed integrity and source syntax checks pass. Local GPU driver mismatch remains; cloud EGL rendering works.
- Artifacts root `outputs/cloud_pipeline_m24_e002_robustness_20260911`: `robustness_acceptance.json`, `selection.json`, `five_cycle_training_comparison.json`, `selected_failure_analysis.json`, source snapshots, commands, timestamps and SHA-verified baseline/first/second/native archives. Successful workflow `..._five_cycles1536/workflow_status.json`; first negative comparison `..._one_degree1536/workflow_status.json`. Full report `docs/m24_e002_robustness_2026-09-11.md`.
- Cloud instance `pro-786095bb6e03` remains ON; all task tmux jobs finished. **Single recommended next step:** diagnose the31 third/fifth-drive timeouts from saved action/contact/joint-travel trajectories, distinguishing stalled motion from deadline truncation, then run one evidence-backed bounded continuation from the new selected model. Preserve nominal and paired perturbation comparisons; do not repeat the rejected two-cycle adaptation.

## 2026-09-11 -- Git committed; initial-state robustness baseline active

- User requested Git commit then continuation. Committed`94e8007`(19files),preserving verified learned return/regrasp/five-cycle work. No push. Next target: robustness to independently perturbed index/thumb initial joints,starting selected repeated-twist SHA`7742e40ddf1d55a48167f011307c77c5ca943306a7754298e433f7bdbc586c7c`.
- Add opt-in independent per-environment joint perturbations with inactive fingers fixed and a nominal mixture. Legacy broadcast reset noise is untouched. Evaluator now honors configured torch/NumPyseed and logs exact sampled reset offsets;42pathpreserved. Actual-helper CPUtests and syntax checks pass. Rewards/phasegates unchanged.
- Cloud reachable/idle at bootstrap,RTX4090/49140MiB. Read cloudhandoff andverifiedbaselineSHA. Active tmux`m24_robust_baseline`,driver`outputs/cloud_pipeline_m24_e002_robustness_20260911/preflight.sh`:freshA0c,then128env0/0.5/1/2degreebaselines andheld-out seeds43/44at1degree,each180s. Explicitconda+NVIDIAEGLVulkanICD retained. No new PPO yet.
- Single recommended next step: choose an initial-joint perturbation width from actual baseline failures/successes,then one bounded60s smoke/300s cloud PPO and nominal/held-out evaluations. Detailed hypothesis/primaryliteraturein`docs/m24_e002_robustness_2026-09-11.md`.

## 2026-09-11 -- Fixed-pose learned repetition achieved and visually verified

- **Current result:** selected BEST completes5continuous drive/release/off-contact-return/regrasp cycles in128/128originalcandidate743episodes. Net rotation148.542–150.743degrees,median149.747; minimum measured contact-supported drive stroke20.527degrees. Every return trajectory stays off-contact throughout. One PPO,normal inputs,no demonstrated resets/action overrides/policy composition in acceptance.
- Separate512env two-cycle evaluation:501/512complete (97.85%),also with both returns off throughout. Remaining11failures:8first-drive timeouts,3regrasp timeouts. The128and512runs have different reset batches; do not claim universal100%robustness. LAST comparison:464/512two cycles,109/128five cycles. Select BEST by repeated-cycle behavior.
- Selected checkpoint`outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911/selected_teacher/learned_repeated_twist.pth`,SHA256`7742e40ddf1d55a48167f011307c77c5ca943306a7754298e433f7bdbc586c7c`. Source training file`...closure1536_ppo/stage1_nn/best_reward_54.97.pth`. Previous learned-return/release models preserved. Only one new300scloud PPO with separate60s smoke (both124),1536/3072,LR0.0001; **reward weights and phase gates unchanged**. No local training,30minute run,student or deployment.
- Native cloud camera verified selected BEST in a197frame/20Hzsingle episode (9.85s),20ordered phase events,five cycles,149.331degrees net. Inspected all20event keyframes and angle/force/sector plot. Actual video`outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911/best_five_cycle_video/learned_five_cycles.mp4`; GIFalsoavailable. No episode concatenation; GIF looping is only playback.
- Execution receipts: initial post-JSON native shutdown segfaults and one missing-conda retry are preserved. With conda activation and task-local NVIDIA EGL Vulkan ICD, all4BEST/LAST original/curriculum retries and bothfive-cycle evaluations plusnative recordings exit0/finite. Clean receipt`...closure1536/retry_completion_v3.json`,longer receiptroot`longer_evaluation_status.json`. Originalfailedworkflow remainshistorical; authoritative acceptanceisroot`repetition_acceptance.json`,status`passed:fixed_pose_learned_repeated_cycle`,full_cycle_goal_completed=truewithinthisfixed-pose scope.
- Local/cloud source,checkpoint,seed/eval/recording artifacts are preserved withSHA256. Git remains`f91ec27`plus uncommitted return/regrasp continuation; no new commit/push this session. Cloud work complete; instance remainsON,idle after final verification. LocalGPUvisualization isblockedbykernel580.173.02/userspace580.178.04mismatch; no workstation driver changes/reboot. CloudnativeEGLrecording is the validated route.
- **Single recommended next step:** preserve this learned-five-cycle model and expand initial-state/seed robustness around the measured11/512failures. Do not restart solved drive/release/return curricula. More than5cycles,full160mmtravel,randompose/domainrobustness and realhand transfer remain unverified.

## 2026-09-11 active target: productive regrasp and second drive

Controlled closure followed by the unchanged PPO demonstrates8/66next-drive successes; learned original-start repetition is still incomplete. Use measured regrasp-state curricula with50%originalstarts and unchanged reward weights. FreshA0c passed; each cloud comparison remains60s smoke plus300s PPO followed by best/last original512 evaluation. Detailed evidence and current execution are in `docs/m24_e002_regrasp_curriculum_2026-09-11.md` and the latest handoff.

## 2026-09-10 result: return achieved; next target is productive regrasp

Single PPO from original743:512/512drive,release,and off-contact return; net at return>=25.7498degrees,sector error<=2.4081degrees,normal inputs and no demo resets in acceptance. Current contact-only regrasp event512/512drifts back to~26.77degrees sector error;0second drives and0repeated cycles. Next: retain the returned start sector while establishing the next grip and preserve usable joint travel,then learn the second stroke. Return acceptance,checkpoint SHA and current artifacts are in the latest handoff. The full repeated-screwing goal remains incomplete.

## 2026-09-10 continuation: learn return from actual learned release

Current target: both fingers return to their captured initial radial sectors while off the nut, retaining the achieved net rotation. Preserve the95.3% original743 drive/release baseline (`f91ec27`). Use actual learned released-state initialization, physical reachability/policy/zero diagnostics, a bounded cloud learning comparison, and original-start evaluation. This advances the next phase; it does not claim regrasp or repeated cycles. No reward weights changed by adding this curriculum.

# M24 Paxini PPO Plan

## 2026-09-10 active amendment — user-authorized gate revision and local fallback

This amendment supersedes conflicting historical E002 stage/promotion rules below.
The user approved removing unvalidated normal-moment vetoes and the requirement
that a five-minute initial policy reach mature success rates before continuing.
Cloud remains preferred, but the user explicitly permits local training when
cloud resources are unavailable. Cloud capacity has returned; all training in
the release-learning session ran on cloud. Local execution was development,
analysis and authorized inference visualization only.

- Final objective remains cooperative index/thumb rotation, release, return,
  regrasp and repetition. A drive-only pass is never full-task completion.
- `acceptance_mode=contact_progress`: nut-only physical pair contact and new net
  rotation govern drive milestones/progress. Normal torque, balance and old
  pair gates are diagnostic metrics; `strict_normal_proxy` preserves comparison.
- A0c still checks actual fixed-pose execution, finite/reset integrity, stable
  nut contact and useful contact-associated motion. Its normal-moment conjunction
  is no longer the active veto. A sign retained in this mode is a diagnostic
  convention, not a validated full-friction force sign.
- Fixed evaluation records goal progress and old thresholds separately from
  execution health. Zero early success does not prohibit a further bounded
  learning experiment. Neither an evaluator exit0 nor an A0c pass proves the goal.
- Keep source/pose/asset correctness, no NaN, no reward from bolt-only contact or
  high-water revisits, real net rotation and recoverable contact behavior.
- Exact same-50ms-bin release/regrasp is no longer an acceptance requirement.
  Measure coordination lag and repeatable cycles directly. Future phase learning
  must allow intentional release; the current drive-only loss timeout is not a
  universal cycle rule. Do not count timed-out phase transitions as success.
- First revised experiment: fresh A0c -> one bounded smoke -> five-minute
  warm-start PPO -> best/last fixed evaluation. Preserve checkpoints and exact
  settings. No automatic 30-minute extension or student training.
- Primary evidence, hypotheses and validation: `docs/m24_e002_gate_revision_2026-09-10.md`.

### Current milestone: learned release achieved (September10)

One PPO now performs drive512/512 then release488/512 (95.3125%) from fixed
original candidate743 with canonical inputs. No scripted actions, demonstration
resets or two-policy composition enter this acceptance. Release retains at least
27.489degrees of net nut rotation; both final forces0N.24release timeouts remain.
A verified35frame/20Hzsingle-episode recording confirms actual drive and release;
thumb/index opening lag0.60s. This is not full-cycle completion.

Selected checkpoint:
`outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/selected_teacher/learned_drive_release.pth`
(SHA256 `aa0e55b06ca7fd864b2ccf717bd73bd95c173e9eface25be93c3328450b9f471`).
Reward-best is worse on release; select the evaluated LAST. Preserve it as the
baseline while learning the next phase.

Next subgoal: collect actual learned released states and teach return to the
initial grasp sector with fingers off and net nut angle preserved, then evaluate
from original743. Keep nominal episode initial-hand-pose context consistent in
intermediate resets, restore critic value-normalization on resume, and do not
reintroduce the corrected nut-DOF reset defect. Return/regrasp/repeated cycles
remain unachieved. No automatic30minute run or student training.

Detailed evidence, four separately bounded5minute cloud comparisons and limitations:
`docs/m24_e002_release_curriculum_2026-09-10.md` and the latest handoff.

## 1. Current objective

Iteratively train and evaluate an IsaacGym PPO teacher for:

`XHandPasiniM24NutBolt -> PPO teacher -> checkpoint/evaluation`

The active optimization target is a human-like two-finger twist cycle: index
and thumb must both contribute during the drive stroke, unload and release in a
short synchronized window, return without dragging the nut, and regrasp. The
immediate campaign evaluates this motion in simulation. Contact-load reward
tuning and real DexH13 motor-current work are explicitly deferred until the
motion reward succeeds.

The task uses:

- Paxini/Pasini 16-DOF hand;
- M24x3 black-oxide hex nut;
- 160 mm M24 bolt;
- 36 mm nut across flats;
- 21.5 mm nut thickness;
- user-saved keyboard init pose;
- 20 Hz policy control and 200 Hz physics.

MuJoCo sim2sim and previous Hora diffusion/student work are paused. They remain
historical context and must not be mixed into the active training run.

## 2. Source and branch contract

- Parent branch: `diffusion`.
- Active branch: `feat/m24-humanlike-twist`.
- Persistent experiment ledger:
  `docs/m24_humanlike_twist_experiments.md`.
- M24 source snapshot: `sim2sim@159d994`.
- Import only M24 assets/configs, Pasini pose support, task registration, and
  init-pose utilities.
- Do not import `sim2sim/`, MuJoCo assets, or sim2sim-specific documentation.
- Preserve existing diffusion/Hora entrypoints.
- Keep the bolt base fixed in all training and evaluation variants. Use
  reaction-force/reaction-moment metrics to measure destabilizing lateral load.
- Do not modify DexH13 deployment, motor-current telemetry, or current-aware
  observations during the immediate motion-reward experiments.
- Train the motion candidate with only index and thumb active. Make middle and
  ring action masking (`action_mask_indices=[4,5,6,7,8,9,10,11]`) explicit in
  the task/config used for the run.

## 3. Training configuration

- Task: `XHandPasiniM24NutBolt`.
- Algorithm: PPO.
- All training, including PPO smokes/capacity probes and any future explicitly
  authorized student training, runs on the cloud. Local execution is limited
  to development, static/deterministic checks, analysis, and authorized
  visualization. Cloud unavailability does not authorize local training.
- Execution target: AutoDL Pro instance `pro-786095bb6e03` via
  `ssh -p 15233 root@connect.westd.seetacloud.com` (verified 2026-09-08;
  refresh AutoDL snapshot if the mapped port changes).
- The 2026-09-03 AutoDL snapshot advertises `vGPU-48GB`, but the instance was
  `shutdown`. After an explicitly confirmed power-on, verify the actual GPU,
  VRAM, CUDA visibility, free storage, and active processes with cloud-side
  commands before selecting capacity.
- Before any cloud sync, code change, training, or evaluation, read
  `/root/code/dexscrew-repro/docs/cloud_session_handoff.md` and every active
  status/manifest/log it references.
- If the actual GPU has at least 24 GB usable VRAM, probe in this order:
  1. 12288 environments / 24576 minibatch;
  2. 8192 environments / 16384 minibatch;
  3. 6144 environments / 12288 minibatch.
- Use the first pair that allocates, reaches stable PPO startup, and leaves
  useful wall-clock time. Record actual GPU model, peak VRAM, startup FPS, and
  the reason for any backoff.
- Do not spend another 30-minute budget until a 5-minute candidate passes a
  fixed-step net-progress evaluation. Wrap the short gate in `timeout 300` and
  any later 30-minute extension in its own `timeout 1800`, both in a named
  `tmux` session with exact commands and statuses under
  `outputs/cloud_pipeline_m24_humantwist_e002_*`.
- Treat timeout status `124` as expected, preserve the best checkpoint, and
  update both local and cloud session handoffs after the run or manual stop.
- Never store the SSH password, AutoDL token, or Jupyter token in the repo,
  command log, experiment ledger, shell history helper, or handoff.

## 4. Milestones

### M0: Governance and selective migration

Completion conditions:

- `AGENTS.md` points to this plan and M24 PPO path.
- Required M24 files are present on `diffusion`.
- No MuJoCo sim2sim directory is imported.

### M1: Configuration and scene smoke

Completion conditions:

- Python and shell syntax checks pass.
- Hydra resolves task/train configs and all 16 saved joint values.
- One-environment GPU PhysX reset loads:
  - 16 hand DOFs;
  - three object rigid bodies;
  - one nut DOF;
  - no NaN/non-finite reset state.

### M2: Capacity probe

Completion conditions:

- On a cloud GPU with at least 24 GB usable VRAM, attempt 12288 environments /
  24576 minibatch first, then follow the documented backoff order only on
  allocation or unstable-startup evidence.
- Confirm startup reaches PPO iteration output with finite metrics.
- Record GPU memory, selected environment count, minibatch size, and initial
  FPS.
- Reduce capacity only with explicit OOM/PhysX/contact allocation evidence.

### M3: Persistent PPO run

Completion conditions:

- Run in a persistent cloud `tmux` session with exact command and log path.
- Record start timestamp, PID/session, git commit, output directory, and active
  capacity.
- Verify checkpoints begin appearing under `stage1_nn/`.

### M4: Training monitoring and first verdict

Completion conditions:

- Track reward, episode length, done/reset rate, FPS, NaN/errors, and GPU use.
- Preserve the best PPO checkpoint.
- Run headed evaluation after a meaningful checkpoint exists.
- Decide whether pose/reward/contact behavior is acceptable before any student
  or diffusion follow-up.

## 5. Acceptance criteria

- M24 asset and saved pose are exactly the configured training inputs.
- Training runs without NaN, repeated simulation reset explosions, or silent
  fallback to a different object/task.
- The chosen environment count is the largest cloud-demonstrated stable
  setting from the defined probe order.
- At least one valid PPO checkpoint and reproducible log are preserved.
- Success is judged by both reward trends and headed nut-rotation behavior.
- Success additionally requires the current-stage motion gates defined in
  `docs/m24_humanlike_twist_experiments.md`; total return alone cannot select a
  model.
- Gross positive angular velocity is not progress. Selection must use
  unwrapped final net angle, new-high-water angle, reverse travel, gross travel,
  and virtual M24x3 screw advance (`3 mm` per full revolution).
- A checkpoint fails immediately if forward and reverse travel cancel, if the
  nut does not establish positive net advance, or if a release timeout is
  counted as a successful phase transition.
- Every run that informs a reward decision must record its exact reward scales,
  command, checkpoint, quantitative metrics, manual video verdict, and one next
  experiment in the ledger.

## 6. Current next step

September10 supersedes the historical next steps below: the revised local run
has achieved stable20degree pair-contact drive in512/512 fixed evaluations.
Introduce release/return/regrasp from that stable stroke; do not wait for30degree
or90% maturity. See `docs/m24_e002_gate_revision_2026-09-10.md` for matched
baseline results and the selected last checkpoint. Cloud is off due to resource
shortage; local fallback is explicitly authorized. No experiment is active.


## 2026-09-08 -- Nut-contact reward comparison completed; strict goal unmet

- User explicitly authorized the pending reward-multiplier exception. This comparison is completed; do NOT ask for its approval again. All training was cloud-only. Goal remains uncompleted; no B/C, student or30-minute training was launched.
- Report: `docs/m24_e002_nutcontact_comparison_2026-09-08.md`.
- One production change: `forward_source=nut_contact_quality`, positive high-water progress weighted by minimum opposing NUT-specific contact confidence. Legacy pair-quality mode retained. All weights and strict gates unchanged. Nut-only force prevents bolt-contact credit. Actual-expression and A0 checks pass.
- Pipeline: `outputs/cloud_pipeline_m24_e002_nutcontact_20260908_175230`. A0c explicitly passed4/27, selected19; physical traces match previous A0c exactly. SHA256 `6e344b6ccf47fa9331a8b3a7d7af2887ee43a99b9512dad293ae8bbe21b5e6b7`. Candidate19 identical-trajectory reward−0.15029 ->+9.23656, not a learned pass.
- Smoke17:54:05–17:55:06+08, expected124. PPO12288/24576,17:55:06–18:00:13+08, timeout300 plus7s completed-update grace, expected124. Best update10; last update11/counter1,769,472 (initial-batch offset included). Checkpoints are inference snapshots, not optimizer state.
- Best/last fixed512-env,100-step first-episode evaluation both0/512 success and0/512 reached30deg. Median net26.0711deg /18.3390deg; mostly monotonic, but stalls around3.7deg under increasing thumb pressure then loses contact. Best469/512 and last512/512 terminate on workspace. Strict credited ratios41.2%/50.45%. `phase.txt` / `acceptance.json`: `rejected:fixed_step_numeric_gate`. No headed pass.
- Checkpoint directory: `outputs/XHandPasiniM24NutBolt_teacher/m24_e002_nutcontact_s42_20260908_175230_probe_12288_24576/stage1_nn`. Best SHA `b1e8123447fc31340373a93af8ac93cc7d7e988a6b67e8a4594b91bfbad8ef31`; last SHA `9859db2503758d11b0c92e0c1f56c27ce65270d00e89d177d528a67356fadd65`. Files are synced and hashes match.
- Frozen-policy target-clamp and6-step startup-settling diagnostics both fail to solve behavior; neither is applied to production. Contact-point gait checks do not establish a direction-only reward fix. `gait_audit` first attempt failed a tensor-view indexing check; corrected diagnostic is `gait_audit_retry`.
- New reference substep audit:800 physics samples still gate0 everywhere. Angle31.54448deg but sampled qdot integral0.79903deg; diagnostic velocity-iterations1 gives30.02368deg vs0.00296deg integral, still gate0. Thumb normal moment negative throughout baseline. Full friction remains unmeasured. Do not mistake geometric reachability or endpoint velocity for validated torque-driven motion. PhysX internal position/velocity iteration semantics may explain differences; no simulator bug or all-motion artifact is established. No physics setting was changed in production.
- Current cloud: no tmux/train/eval/probe job; GPU idle. Source, commands, logs, timings, plots, full raw diagnostics and checkpoints preserved.
- Single next action: diagnostic-only known nut torque, with and without finger contact, to distinguish actual angle, sampled velocity and constraint response before further reward design. This next diagnostic is NOT yet executed. No repeated PPO until an evidenced fix and fresh A0c; strict gates and weights remain pinned.


Treat `E001 / R1-motion` as rejected and execute the staged E002 recovery plan.
Do not train all behaviors simultaneously again. The complete implementation,
reward, curriculum, evaluation, and stop contract is
`docs/m24_e002_training_plan_2026-09-04.md` and takes precedence over the
earlier abbreviated E002 notes below.

The first E002-A0/A1 probe was invalid: the nominal reset had no stable pair
contact, and the old termination incorrectly coupled physical contact to
positive torque. A0b/A0c are required before any bounded E002-A1b smoke or
`timeout 300` probe. The diagnostic launcher never starts training.

The initial 2026-09-08 A0b/A0c cloud run found stable reset candidate 743 but
no accepted pair-drive streak. Its unrelated global maxima were insufficient
for a physical diagnosis; the diagnostic work below supersedes that summary.

The 2026-09-08 diagnostic and single-fix rerun are now complete. Both cloud
JSONs explicitly failed: 67/67 stable holds, 637/670 probe steps with pair
contact, but zero simultaneous torque, pair-gate, or full-condition passes
under either force sign. The only behavioral fix was E002 `rotation_axis=-z`
to match the M24 URDF nut joint's positive DOF axis; no weights or thresholds
changed. See `docs/m24_e002_a0c_diagnostic_2026-09-08.md` for correlated traces.
The subsequent one-environment cloud contact audit is also complete:
`docs/m24_e002_contact_audit_2026-09-08.md`. Six paired GPU/CPU tensor runs
saved 3840 physics samples. In all 600 contact-audit probe samples the index
link-origin proxy and actual contact-point normal moment have opposite signs.
Normal records reconstruct the reported tensor within 3.48e-6 N, but no full
friction wrench is established; actual shape friction is 1.0 despite zero
raw contact friction fields. The audit is not an A0c pass.
The subsequent cloud recovery is recorded in
`docs/m24_e002_cloud_recovery_2026-09-08.md`. Sensor negative controls rejected
raw solver sensors as a contact-only source. The prior requirement to obtain
a full friction wrench before correcting the existing normal-force proxy was
overbroad and is superseded: the measured normal forces now use their actual
contact points. Full friction wrench remains explicitly unmeasured. A real
two-env validation passed. Paired CPU-tensor/GPU-PhysX A0c traces match exactly
in force, distance, pose and motion while only torque geometry changes.
E002 alone selects `contact_point_normal`; weights and all gates are unchanged.
The physical-sign torque passes improve from 4 to 28 / 1460 probe steps, but
that paired bank had no accepted A0c candidate. Contact-based action refinement
reaches gate 0.674 and 0.125 rad net motion in one candidate, but only one
consecutive accepted step before thumb contact loss. Extending the same action
bank from 20 to 30 steps does not solve the loss. A focused late-increment taper bank now explicitly passed A0c:
`outputs/cloud_pipeline_m24_e002_taper_refine_20260908_154240/a0c`.
Four candidates pass; selected physical-sign candidate has streak 7 and net
0.12568 rad, with unchanged gate and weights. Accepted-window progress is only
0.000760 rad, so the A0c pass is not a useful-stroke or policy-behavior pass.
Cloud smoke and one five-minute PPO completed in
`outputs/cloud_pipeline_m24_e002_passed_a1_20260908_154343` at 12288/24576.
Both timed phases exited expected 124. Selected best-reward checkpoint failed
512-env / 100-step evaluation: 0/512 successes, 512/512 contact loss at 13–14
steps, zero credited progress. The pipeline is rejected and cloud is idle;
no headed pass or longer training is claimed. The best checkpoint is from
initial epoch 1; no final last.pth was saved. The single next cloud action is a
diagnostic comparison of early PPO actions against zero hold and the accepted
A0c schedule to isolate thumb loss. Do not repeat the failed training settings
or change weights/gate without a newly demonstrated bounded issue.
Thirty-minute training and E002-B/C remain prohibited.

### E002-A: monotonic two-finger drive

- Keep the bolt base, M24 asset, saved pose, 20 Hz policy rate, 200 Hz physics,
  and index+thumb-only mask fixed.
- Train one approximately 30 degree pair-contributed stroke, rather than asking
  the two fingers to rotate continuously without a release.
- Remove the R1 phase machine from reward and selection for this stage.
- Replace positive-velocity accumulation with unwrapped high-water progress:
  only a new maximum nut angle earns progress reward.
- Apply reverse-travel penalty on every step with a coefficient at least as
  large as the forward-progress coefficient. Remove the currently dead
  `reverse_penalty` path.
- Disable constant per-step proximity reward after contact. Approach shaping,
  if retained, must be potential-based and stop paying for static contact.
- Keep pair contact/positive torque as a gate on new progress, but separately
  log ungated net angle so reward-sign/contact errors cannot hide non-motion.
- Add an opt-in M24-only four-state phase command plus normalized phase elapsed
  to the policy observation now, held at `drive` in E002-A, so E002-B can resume
  the same network without an input-shape change. Do not expose simulator-only
  contact or nut truth to the actor.
- Make distance, no-contact, and stagnation termination phase-aware before B;
  planned release cannot be reset as if it were a failure, and no timeout can
  become a successful transition.
- Select by fixed-step behavior metrics, not `best_reward_*.pth`: final net
  angle positive in at least 95% of evaluation environments, net/gross travel
  efficiency at least 0.80, reverse/gross at most 0.10, and pair-drive ratio at
  least 0.90.
- Run deterministic reward tests, one-env smoke, then a `timeout 300` cloud
  probe. Reject immediately if the numeric gate or headed motion fails.

### E002-B: strict unload and return

- Start only from an E002-A checkpoint that passed monotonic-drive acceptance.
- Enter unload only after genuine new-high-water stroke progress; a drive
  timeout is failure/reset, never a phase transition.
- Leave unload only after both contacts are off for consecutive steps. An
  unload timeout is failure/reset with penalty, never a transition to return.
- Keep both fingers off throughout return; persistent contact cannot return to
  grasp. Require an off-dwell plus return-pose criterion before regrasp.
- Record valid releases separately from timeouts. Missing releases are invalid,
  never zero-lag successes.
- Run another 5-minute gate and headed evaluation before any extension.

### E002-C: repeated full cycles

- Resume the accepted E002-B checkpoint and train repeated
  `drive -> unload -> return -> regrasp` cycles.
- Give a cycle-completion bonus only when monotonic progress, valid double
  release, off-dwell return, and double regrasp all occurred in order.
- Extend to 30 minutes only after the 5-minute checkpoint shows positive net
  M24x3 virtual advance and at least three valid visual cycles without reverse
  oscillation or continuous-contact shortcuts.
