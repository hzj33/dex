# Cloud Session Handoff

Scope: live cloud execution state for `/root/code/dexscrew-repro`.

Use this file as the first stop when switching between Ubuntu-side and Windows-side Codex sessions. It records what the cloud machine has already done and what should happen next.

## 2026-09-14 — 用户明确确认云端关机

- 本次唯一操作：关闭实例 `pro-786095bb6e03`。关机前已读实际云端交接；未发现tmux会话或GPU计算进程，最近实验与checkpoint已由上一会话完整同步保存。
- 用户授权关机，现将通过AutoDL管理脚本请求关机。本条在关机前同步到云端；最终供应商状态确认保存在本地 `docs/session_handoff_v2.md` 与关机回执目录 `outputs/cloud_shutdown_20260914_184514`。本条本身不是供应商已完成关机的确认。
- 不启动PPO、不追加优化、不操作实机。E003优化依旧按用户条件停止且未完成；已有基线、标签和产物保持不变。
- 唯一下一步：保持停机，等待用户新的明确指令，不自动开机或续训。

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

## 2026-09-14 — 峰值PPO批量有效；归一化与线程对照排队

BalanceFirstPeak批量preflight仍115/128；128env控制端峰值23.7530/11.8029N、周期峰值25.3993/12.3798N，累计超限信号比值1.1362。独立smoke124完成，正式PPO运行。当前峰值采样版约2700steps/s，低于此前约4800；云PyTorch默认intra-op104、interop208线程，尚未断言线程是唯一瓶颈。

新增build_m24_e003_normalization_ablation.py，用已完成Strong5KL LAST bf515758与原d42e98a2构造“原actor+新input norms”“新actor+原input norms”。模型/全部输入归一化逐tensor精确断言通过，SHA/来源在outputs/e003_normalization_ablation_20260914/manifest.json；保留各自critic value norm，不作为验收策略。cloud_pipeline_m24_e003_normalization_20260914 / tmux m24_e003_normalization排队等待BalanceFirstPeak完整结束，再4组固定/43各180s上限推理，检查归一化漂移贡献。

其后cloud_pipeline_m24_e003_threads1_20260914 / tmux m24_e003_threads1排队独立60s/1536env/3072mini PPO容量probe，沿用BalanceFirstPeak/原教师，仅OMP_NUM_THREADS=1、MKL_NUM_THREADS=1；记录实际Torch线程、命令、起止、退出码，124正常，不将probe模型当候选。各队列有明确等待上限并串行，不改进行中训练。

辅助只读分析：原成功五轮周期38/37/38/39/45控制步，现rollout12步；GAE允许value bootstrap跨越rollout，不能据此宣称bug。如后续回位退化仍持续，可据GAE原论文 https://arxiv.org/abs/1506.02438 做更长rollout有界对照，尚未改horizon或lambda。下一步完成上述串行队列，依据实际受力/回位/吞吐结果选择下一PPO设置；5N与完整平衡仍未验收。

## 2026-09-14 — KL完整拒绝；原资产横向负载影响与BalanceFirstPeak

Strong5KL600s完整完成/自动同步：LAST固定128、43/44=76/74（150/256），BEST固定128、70/72（142/256）；固定LAST峰值17.6967/9.1419N。KL修复了MSE版本固定0/128的回位遗忘，但扰动仍远差于原234/256，拒绝提升基线。

原资产known-load sweep45组已独立120s上限exit0，无手/无接触，原URDF/friction0.2/damping0.01/重力/8-0迭代/真实朝向不变。0.1Nm、横向0N一秒转1.16210rad（66.584°），横向5N几乎0；5N横向时0.2Nm转0.91443rad。说明当前模型横向负载会显著增加转动阻力，不等同于真机标定或双指5N可行性证明。原get_actor_dof_states报告速度近零而q持续每步变化，分析必须用位置差分；原始字段保留，修正版脚本单列两种速度，分析在joint_load_analysis.json，不能误判“转后停下”。

原教师action_scale=0.0375单例完成五轮、200Hz峰值16.3293/7.9740N；0.025和0.0125均return12，峰值15.3203/4.1011、14.4611/3.3431N。拒绝速度方案，不能因拇指<5接受。全档案已同步，无训练并发。

下一有界工程假设：先改善横向加载效率，再收紧总载荷。原teacher驱动法向横向负载均值约6.78N、index/thumb法向均值11.06/4.98N；已知外力对照支持避免单侧挤压的机械动机。BalanceFirstPeak保留5N目标、Kp2.4与原gates/原初始化：force_excess_scale采用较温和Force5值0.05，normal_balance_scale由0.025提高至0.125；原轨迹normal成本约5.64→28.22（约总221.18的12.8%），让平衡项不再远弱于驱动收益。并启用200Hz周期峰值信号（原轨迹超限成本信号111.54→128.82、+15.5%，3个窗口端点<5而中间>5），return KL系数1、LR上限1e-5。这是联合工程策略，不声称单因素因果隔离或完整摩擦wrench平衡。奖励范数参考官方legged_gym、阶段教师指导参考Kickstarting，权重/几何投影为上述任务诊断支持的本地适配。

拟云pipeline `outputs/cloud_pipeline_m24_e003_balance_first_peak_20260914` / tmux m24_e003_balance_first_peak，独立60s smoke/600s PPO/每组180s评估，1536env/3072minibatch。上一批已全部完成，GPU串行；无实机操作。下一步验证峰值信号在批量PPO中生效、回位与低力/负载平衡联合结果；完整计划仍未达标，原成功教师不变。

## 2026-09-14 — 动作增量诊断排队

KL训练仍独占GPU（约228万环境steps，正式600s未结束）。在其完整评估结束后，先执行已知joint-load sweep，再依次执行原教师action_scale=0.0375/0.025/0.0125单环境seed43/200Hz审计，每个180s独立上限。相对原0.05递减，检验增量目标推进速度与载荷，不改Kp2.4、摩擦或phase成功/超时门槛；低力停转直接拒绝。没有重复已失败的目标误差硬投影或低刚度设置。排队pipeline `outputs/cloud_pipeline_m24_e003_action_speed_20260914` / tmux m24_e003_action_speed，显式等待joint-load pipeline退出；所有排队都有1200s等待上限，不与正式训练并发仿真。下一步仍为KL和物理对照的实际结果，尚不改变成功基线。

## 2026-09-14 — 当前训练与排队物理负载诊断

Strong5KL修复后已进入正式PPO：实际1,529,856steps时return KL约0.00787/max0.02045，实际LR始终1e-5，RTX4090约3.1GB、约4.8k环境steps/s；仍不能用训练loss替代回位验收。新增无手原M24已知横向力/轴向力矩sweep脚本，匹配原资产/重力/8-0求解迭代/实际朝向，测量转动响应，不改训练物理或证明双指可行性。云pipeline `outputs/cloud_pipeline_m24_e003_joint_load_20260914` / tmux m24_e003_joint_load 正排队，明确等待KL完整pipeline结束再单独120s上限执行，避免与限时训练抢GPU。

新增单PPO对照汇总脚本，排除学生/混合策略，实际12候选均未通过20Hz每指5N筛选；原始数据仍为权威，汇总不作200Hz验收或基线提升。监控器增加PPO last.training_state.json进度兼容（原来只读学生progress.json，最终同步功能已正常）。下一步是KL五轮/受力验收后读取原资产扭矩响应，再决定下一学习比较；整体计划未完成。

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

## 2026-09-14 — 其他力传感器开关与摩擦字段核验

同一已知5N接触rig、8/0迭代设置：forward-dynamics-only根读数约0，both约-40.04N，与solver-only相同；所有官方开关组合都未得到可直接使用的固定支撑物理wrench。两次独立90s上限命令、起止、exit0在sensor_modes_execution.json，JSON已同步本地。原教师12873条实际contacts的lambdaFriction两分量及reported friction字段全0，尽管实际shape friction=1；原net_contact与逐指normal向量相反项残差<=1e-6。不能把0摩擦字段解读为仿真没有摩擦，也不能据此称完整力矩平衡。

当前Strong5Anchor600s训练正常，return辅助loss约0.0068/max0.0083，LR始终<=1e-5；完整计划未通过。下一步依然是独立五轮/5N与法向平衡评估，完整摩擦wrench测量限制单列。

## 2026-09-14 — 整根手指接触覆盖诊断

接触wrench函数新增body组支持（整数接口保持），200Hz audit可分别聚合食指/拇指全部link与tactile link，分析器单列whole_finger_peak_N及5N筛选。避免策略把载荷转移到近端指节而只降低末端指标。用原教师1970物理步真实contacts档案验证：整指与原末端法向矢量差严格0，峰值17.09995/8.84922N；单元素body组与原整数接口一致。首次通过package导入触发缺IsaacGym的本地依赖，改用文件模块导入进行纯NumPy分析，无本地仿真/训练。新诊断尚未用于新候选验收，不改当前云训练源码或5N定义；后续候选必须检查整指覆盖。

当前云Strong5Anchor600s训练正在运行，smoke冻结目标测试通过，辅助loss有非零记录；下一步仍为最终/周期模型成功率与载荷验证。

## 2026-09-14 — Capped600s拒绝，转入更强限力与回位保护

Capped600s实际LR始终<=1e-5，完整训练/评估完成。BEST固定128、43/44=104/108（212/256），扰动峰值18.1252/12.5012N；LAST固定128、100/108（208/256）。LAST seed43新增22回位超时、6workspace失败，不能以低力/较低积分接受。周期模型保存epoch1至约146，完整正式目录自动同步；未替换原教师。

下一假设由两条独立诊断支撑：原5N代价0.05只扣原轨迹总奖励约2.55%，而600s后仍高力；持续微调仍遗忘回位。Strong5Anchor在Balance5基础上仅提高force_excess_scale至0.25（原轨迹约28.21、总奖励约221.18的12.75%），同时启用已测试的return_anchor_coef1.0辅助MSE，限定raw phase2/8有效关节；单位系数是初始工程比较值，不称论文最优值。此轮为联合工程改进，不宣称分别识别两个改动的因果效应。方法依据与测试见前节Kickstarting本地适配说明，仍是单个PPO网络，部署不需要教师运行时混合。

保持Kp2.4、物理/摩擦/5N门槛/五轮gates、LR上限1e-5/KL0.005，原d42e98a2初始化。云独立60s smoke/600s PPO，1536env/3072minibatch，每60s保存周期候选；pipeline`outputs/cloud_pipeline_m24_e003_strong5_anchor_20260914`，tmux m24_e003_strong5_anchor。云任务切换前上轮已完成；不执行实机。下一步验证云端冻结参考目标、非零辅助loss与实际force/成功率，继续直到联合目标通过或发现必须升级解决的实际阻塞。

## 2026-09-14 — 配对共同成功回合的受力复核

新增compare_m24_e003_force_candidates.py，强制两组初始关节偏移逐项一致，并分别报告全体/共同成功回合、恢复/退化ID。Balance5 LAST seed43原115→113，其中恢复6、退化8，共同成功107；共同成功的食指峰值23.7530→20.8594N、F²积分709.57→695.71N²s，但拇指>5N时长1.3514→1.4477s，不能笼统称每项载荷改善。证据outputs/e003_balance5_paired_s43.json，未改变验收门槛。

同时重算原nominal奖励量级：每episode drive奖励200.90、release6.37、return6.54、regrasp7.36，当前5N超限成本0.05累计约5.64，仅约原总奖励2.55%。若受控LR长训仍持续超限，下一明确假设是超限代价相对旋转收益过弱；可在保留其它设置的对照中提高至0.25（原轨迹约28.21，仍明显低于完成收益），不能声称软惩罚保证硬限力。

当前云Capped600s训练继续，周期模型保存正常；下一步最终和中途模型联合筛选，后续依据上述量级选择单因素代价对照。

## 2026-09-14 — 完全匹配任务求解参数的传感器校验

已补测position iterations8/velocity iterations0（与任务配置一致）：已知5N接触、normal lambda均值5.00000061N，base solver Fx均值-40.04145N。结果与4/1和8/1共同证明原始支撑读数依赖求解迭代，不可直接用于物理wrench验收。第三组数据/SHA已加入sensor_calibration_result.json。测试脚本显式支持--vel0并保存参数。云Capped600s PPO继续正常更新，当前>100万steps，未改其运行源码；下一步为最终/周期模型联合评估。

## 2026-09-14 — 传感器校验结论归档

`outputs/cloud_pipeline_m24_e003_wrench_probe_20260914/sensor_calibration_result.json`记录已知5N接触的两个稳定窗口：normal lambda均值5.00000083/4.99999973N，base solver Fx均值-25.2194/-45.0076N，对应position iterations4/8（velocity1）。两组JSON及SHA已同步本地，传感器验收状态failed:physical_support_wrench_calibration，不能对动态峰值简单除迭代数。教师新增传感器与原audit197步所有共同字段完全相同，原独有joint_state/joint_targets为日志选项差异，已确认监测不影响动作。

云端Balance5Capped10分钟训练仍运行；这里完成的是测量有效性子目标，不是5N或完整平衡达标。下一步继续检查真实LR上限和训练后联合评估。

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

### 启动核验补充

学生基线commit c59a89b；卸载诊断/代价/runner commit badf4cd。云A0c和教师seed43预检通过，60s smoke按SIGTERM在完整更新后存档，正式1536env PPO已开始（约5800 steps/s，显存3085MiB/49140MiB）。首批episode结束后mean_rewards恢复为有限值，当前约175.19、best176.26；速度约5200 steps/s，未发现启动非有限错误，后续仍以独立评估为准。自动同步监控PID3457266，日志/tmp/m24_e003_unload_monitor.log，已扩展同步最终PPO候选checkpoint；不控制电源/硬件。正式阶段300s后自动评估BEST/LAST，结果尚未产生，不标记策略改进通过。

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

## 2026-09-14 — Student smoke nominal128/128; formal1h running

As of this update, model contract/A0c/frozen teacher nominal all passed. Separate300s PAdapt smoke exited124 with2882updates/184512steps, finalBC0.021637 andlatent0.005534; teacher weights unchanged. Smoke student checkpointSHAc8b0c18272c8e8fc5b363820360fdb5a4514933717595480b670a2ebadd5b74d at `outputs/XHandPasiniM24NutBolt_student/cloud_pipeline_m24_e003_student_20260914_smoke/stage2_nn/model_last.ckpt`.

Smoke numeric nominal fivecycles128/128, drive thumb-loss0; force peaks20.5028/11.7688N are HIGHER than teacher16.6735/9.3674N. This is not final low-force student acceptance. Formal3600s began about12:10Asia/Shanghai (04:10UTC),64env,seed42, fresh same frozen teacher. No new PPO. Expect training limit around13:10local, followed by automated BEST/LAST nominal43/44, extra45/46,audit,native,phase101JIT export. Read `outputs/cloud_pipeline_m24_e003_student_20260914/status.json` for actual current phase, not this timestamped snapshot. tmux`m24_e003_student`.

A detached LOCAL read-only monitor mirrors progress every45s, copies formalBEST/LAST every5min, and synchronizes all outputs after pipeline ends; receiptroot/automatic_sync_receipt.json then appears. Monitor script`monitor_m24_e003_student.py`, localPIDinroot/local_monitor.json, log/tmp/m24_e003_student_monitor.log. No automatic power-off, no real hardware. Source/metrics details in docs/m24_e003_student_2026-09-14.md and localhandoff.

Next action: review formal completion/load/multi-seed/native results before accepting a final student. Preserve smoke and periodic copies in case later checkpoints regress. Phase command remains explicit external input; real adapter and position-modeKp mapping are still pending. Do not equate nominal success with retained force or hardware readiness.

## 2026-09-14 — User-authorized E003 student training startup

User confirmed GPU power-on and explicitly requested student training. pro-786095bb6e03 running, RTX4090/49140MiB idle at bootstrap; all previous shutdown notes are historical. No teacher optimization or hardware actuation. Current target: first E003Compliant ProprioAdapt student from selected d42e98a2 frozen teacher, Kp2.4 unchanged. Original M24 deployment source reference is sim2real branch4736d94.

Pipeline `outputs/cloud_pipeline_m24_e003_student_20260914/status.json`; driver `outputs/cloud_pipeline_m24_e003_student_20260914_driver/run.sh`; tmux `m24_e003_student`. Separate300s smoke/3600s formal student limits (124 expected),64env/seed42, onlyadapt_tconv, periodicatomicLAST and BEST. FreshA0c and frozen nominal128 before training; smoke numeric eval, formal BEST/LAST nominal/43/44, selected45/46,audit,native and export. Exact commands/timestamps/status saved per phase. All source mutations backed up under driver/pre_sync_backup.

Interface:101D observation includes external phase command; stage2 removes privileged/point-cloud action dependence, not the phase command. Export explicitly requires phase input and is not yet a drop-in old96D real policy. Real position-mode control does not inherit simulatedKp2.4 from weights. No5N or zero-slip claim. E003-only teacher_point_tanh opt-in corrects exact PPO target; synthetic parity and no-privileged-inference tests pass, legacy default unchanged.

Next action: verify physical preflight/smoke, then allow bounded formal student and automatic cycle/load/native/export evaluation. Do not restart PPO or operate hardware. Keep selected E003 success baseline and user5N/balance follow-up goals; axialfeed excluded by user.

## 2026-09-11 — User-confirmed shutdown requested

At 2026-09-11T15:54:24.196488+00:00, user explicitly confirmed power-off of `pro-786095bb6e03`. Pre-shutdown check: no tmux sessions, GPU0MiB/0%; E003 artifacts already synchronized and preserved. The AutoDL power-off API will now be called; final shutdown status receipt is recorded locally in `outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/power_off_receipt.json` because SSH becomes unavailable after shutdown. Do not start more cloud work automatically. Next action: leave instance off until further user instructions; retain E003Compliant selection and all checkpoints.

## 2026-09-11 — E003 compliant profile selected; all cloud jobs finished

- Current selected simulation task: `XHandPasiniM24NutBoltE003Compliant`, frozen E002 PPO SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. Only resolved task difference is Kp3→2.4; Kd0.01, friction, rewards and success gates unchanged. Explicit binding/copy: `outputs/cloud_pipeline_m24_e003_compliant_validate_20260911/selection.json` and `selected_teacher/e003_compliant_frozen_teacher.pth`. Preserve E002 baseline.
- Formal root `outputs/cloud_pipeline_m24_e003_compliant_20260911`: nominal128/128, seed43/44=115/119. Extra root `outputs/cloud_pipeline_m24_e003_compliant_validate_20260911`: seed45/46=115/119, total468/512 vs432/512. Paired identical starts;56recovered,20regressed,44remaining failures (42return timeouts,2workspace). All successful ordered20events,>=20degree effective drive and off-contact returns checked. Acceptance scope is relative improvement, not universal success.
- Pooled20Hz index/thumb normal peaks34.123/14.882→24.255/11.803N; mean drive thumb-loss0.2743→0.1102s; force-squared integral1272.00→680.10N²s. Native seed43 continuousfivecycles197frames20fps9.85s144.71degrees;200Hz peaks26.660/12.522→17.100/8.849N. All20event keyframes, peak frame and curves visually reviewed. MP4 `native_s43/e003_low_force_cycles.mp4`, plus `native_event_sheet.png`/`force_comparison.png`.
- H1 target-error clamps rejected. H2/H2b final models rejected; H2b midpoint nominal128 but428/512perturbed and native43only4cycles, also rejected. Two learning comparisons each had separate60s smoke/300s PPO (expected124), cloud1536env/3072minibatch. All learning artifacts and negative results retained. No local training, student,30minute expansion or real-hand deployment.
- Raw extra-validation and H2b candidate `status.json` retain `failed:execution`: missing cloud ffmpeg caused encoding127 only; all numeric/audit/native simulation exits0. The already-running processes had loaded old code; actual version preserved as `executed_validation_source.py`. Local existing ffmpeg encoded videos successfully; `native_s43/encoding_completion.json` records exit0 and ffprobe197frames20fps9.85s. Updated runner marks missing ffmpeg as pending local encoding. Final authority is `acceptance.json`, referencing `numeric_acceptance.json`, `visual_verification.json`, `encoding_completion.json` and `selection.json`; do not misread raw encoding failure as simulation failure or overwrite it.
- Final cloud check: no tmux sessions, RTX4090/49140MiB with0MiB/0% GPU. Instance remains ON. Local/cloud code, report, handoffs and final SHA snapshot synchronized. No Git commit/push. Full report `docs/m24_e003_low_force_2026-09-11.md`.
- Limits: zero tangential slip and actual PhysX cooked-shape penetration bound unverified; source-convex intersection is only a proxy. PD commands/total DOF sensors are not motor current. Softwaretorque_limit300 and asseteffort are not hardware-calibrated protection. No claim of real-hand safety.
- **Single recommended next cloud action:** diagnose saved42return failures and20paired regressions using the selected compliant profile as control, then a bounded low-force return-recovery comparison supported by those states. Keep contact-slip/penetration diagnostics; do not blindly lower friction/gain or rerun rejected clamp/force-cost settings.

## 2026-09-11 — H3 compliant frozen-policy batch active alongside H2b validation

Two independent inference jobs, all PPO finished. `m24_e003_candidate`: E003Load/midpointSHAab385ce1 additionalseeds/audit/video; status outputs/cloud_pipeline_m24_e003_candidate_20260911/status.json. `m24_e003_compliant`: new E003Compliant/Kp2.4 + unchanged E002d42e98a2, nominal/43/44 each128; status outputs/cloud_pipeline_m24_e003_compliant_20260911/status.json. One-env43 H3: fivecycles, index/thumbpeak25.76/11.78->16.71/8.45N, thumb-loss1.75->0s. Not yet promoted; compare batched success and load. Cloud4090/48GB has ample memory for two inference processes; each180s. Next: choose based on paired results and complete extra seeds/native/200Hz validation.

## 2026-09-11 — Mid-training low-force candidate at128/128; validation queued

Conservative finalBEST/LAST remain rejected27/10nominal. SavedmidpointSHAab385ce1 restores128/128fivecycles, indexpeak22.714->20.695N, thumbpeak12.724->13.169N, thumb-lossmean0.01133->0s, release-force-increasemean2.927->0.657N. Frozenmidpoint/gain probes in tmux`m24_e003_midpoint`; then tmux`m24_e003_candidate` starts candidate43/44/45/46,baseline46,physics audit andnative43video. Driver`outputs/cloud_pipeline_m24_e003_load_20260911/validate_candidate.sh`; status`outputs/cloud_pipeline_m24_e003_candidate_20260911/status.json`. Everyinference180s. No modelpromoted until broaderloads/completion andvisual checked; E002d42e98a2 preserved. Next: validate this explicit midpointcheckpoint, disclose thumbpeak increase and remaining failures.

## 2026-09-11 — H2 rejected; conservative-update comparison queued

First low-force model has0/128 nominal full cycles (all first return timeout),46/128seed43 and41/128seed44; force reduction is not accepted. Measured firstKL0.05838, LRpeak0.0003375 despite initial0.0001. H2b keeps samecost/controller/physics and originald42e98a2; initialLR1e-5 and KLtarget0.002. tmux`m24_e003_conservative` waits for first comparison to finish, then freshA0c/frozenpreflight/separate60s smoke+300s PPO1536/3072 and fixedevals. Newstatus`outputs/cloud_pipeline_m24_e003_conservative_20260911/status.json`; driver`outputs/cloud_pipeline_m24_e003_load_20260911/conservative.sh`. No parallel GPU training, no selectedE003. Next: compare retained return, contact load and thumb-loss, reject regressions.

## 2026-09-11 — E003 low-force bounded learning active

Load audit proved identical290step baseline, measuredPD1.892Nm/target error0.631rad. Asseteffort0.001 vs1 identical; all target-error clamp cases failed first stroke and are rejected. NextH2: originalcontroller plus explicit static force/unilateral drive cost in new E003Load only. tmux`m24_e003_learn`; script`outputs/cloud_pipeline_m24_e003_load_20260911/learn.sh`; status`outputs/cloud_pipeline_m24_e003_learn_20260911/status.json`. FreshA0c, frozen128preflight, independent60s smoke/300s PPO1536/3072, then BEST/LAST nominal/s43/s44. Each inference180s. FrozeninitialSHA d42e98a2; no E003 selected. Next: inspect paired completion/load/contact loss.

## 2026-09-11 — E003-0 active load audit

User authorized autonomous low-force and thumb-loss improvements. Frozen E002 d42e98a2; no reward/physics/control change. Cloud RTX4090/49140MiB idle at bootstrap. tmux `m24_e003_load`; pipeline `outputs/cloud_pipeline_m24_e003_load_20260911`, `status.json`, two 180s inference limits. Optional physics endpoint diagnostics record PD requests, submitted torques, DOF total sensors, contact geometry and body loads. Next: verify instrumentation preserves baseline, then diagnose thumb loss and test one bounded control change.

## 2026-09-11 -- Timing correction verified; five-cycle robustness84.4% across384 starts

- Current selected combination: **task `XHandPasiniM24NutBoltE002RobustTime` + unchanged PPO** `outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`, SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. Older tasks/checkpoints remain preserved. No PPO trained this session; no reward-weight or physical-success-threshold change.
- Diagnosis of31 old drive timeouts:27already reached20degrees actual net stroke but none reached20degrees counted pair-contact progress. Doubling execution time with unchanged policy-clock units recovers24of31 into actual fullfive-cycle successes. Do not count mere20degree net angle as a success; the original paired-contact stroke and release/return conditions remain required.
- Optional `E002Cycle.phase_clock_scale` defaults1 and preserves legacy behavior. New task uses phase execution budgets160/80/120/80steps (8/4/6/4s),clock scale0.5 preserves original80/40/60/40input units and saturation. Goal5cycles,episodeLength2200. Exact-clock, delayed-release, reset, full-cycle/no-recycled-progress tests pass. Evaluator adds opt-in joint-state/target recording and timeout-only diagnostic controls.
- Frozen-policy formal evaluation: nominalfive128/128. Independent +/-1degree joint errors: seed43 **94→108/128**, seed44 **99→109/128**, unseen-at-budget-selection seed45 **91→107/128**. Combined284→324/384 (73.96→84.38%),all284previous successes retained,40recovered,no regressions. Original two validation seeds alone193→217/256 (75.39→84.77%). All successful episodes meet the original20degree pair-contact stroke and fully off-contact return criteria. Exact sampled reset arrays match. One seed44 diagnostic prefix had numerical divergence (env64,action0.05362/angle0.511degree); do not describe all batched trajectories as exactly identical.
- Native seed43,previously268frames/fourcycles/fifth-release timeout, now290frames/14.5s/fivecycles and151.780degrees net. All first268action/angle samples match the failed recording exactly;22additional steps finish release→off-contact return→regrasp. All20event keyframes and force/angle/phase plot inspected; native process exits0. MP4 under new pipeline `native_seed43/learned_perturbed_cycles.mp4`, SHA256 `be1c58c53805cddc5427d250b77bca7fa0a86c048291f962fcadbd4c02840db4`.
- Remaining60/384failures:47return timeouts,10workspace violations,3drive timeouts. In the original two seeds,29extended-time return failures are physically off-contact but21.75–67.88degrees from target (median24.16); none within6degrees. More time did not fix those geometric errors. Wider perturbations,beyondfivecycles and real-hand transfer remain unverified.
- Cloud outputs: diagnostic `outputs/cloud_pipeline_m24_e002_timeout_diagnostic_20260911`; formal `outputs/cloud_pipeline_m24_e002_robust_time_20260911`. Four diagnostic evals,five formal evals,native recording and fresh A0c all exit0/finite; newA0c JSON explicitly `passed`,source snapshot refreshed. Each cloud process bounded180s. Acceptance,selected-policy binding,visual receipt,source snapshots,commands,timestamps and raw artifacts preserved locally/cloud. Full report `docs/m24_e002_timeout_correction_2026-09-11.md`.
- Cloud instance pro-786095bb6e03 remains ON,all task jobs finished after final verification. Git still94e8007 plus previous robustness/current timing changes,uncommitted; no push. **Single recommended next step:** collect and diagnose the47late off-contact return failures,then teach recovery from those measured states with a bounded cloud continuation using this explicit task/checkpoint. Keep nominal and fixed held-out comparisons; do not keep increasing deadlines or rerun rejected two-cycle training. The older robustness runner hardcodes the old task; adapt its task/horizon/preflight wiring before using it for this next curriculum.

## 2026-09-11 -- Timeout diagnosis recovers24 failures; explicit task validation active

- Completed four inference-only diagnostic evaluations, all exit0/finite. Preserving checkpoint and policy clock while doubling termination budgets gives seed43 94→108 and seed44 99→109 complete five cycles (193→217/256). All24 newly successful timeout cases were originally drive timeouts; no success condition or reward weight was relaxed.
- Exact reset arrays match. Among diagnosed timeout endpoints, seed43 actions/angles match before the old deadline; seed44 has one nonidentical trajectory (env64, action max difference0.05362, nut angle0.511degrees). Record this numerical divergence; do not claim a perfectly identical counterfactual prefix for all256 samples.
- Implemented optional `E002Cycle.phase_clock_scale` (default1, legacy behavior preserved) and separate task/train `XHandPasiniM24NutBoltE002RobustTime`: execution deadlines160/80/120/80 with clock scale0.5 retain learned80/40/60/40 observation units. Physical20degree/contact/release/return criteria, rewards and PPO weights remain unchanged. Actual-cycle CPUtests plus exact-clock/delayed-release regression pass.
- Active tmux `m24_robust_time_validate`, pipeline `outputs/cloud_pipeline_m24_e002_robust_time_20260911`, sibling `_driver/run.sh`. Frozen-policy formal task evals: nominal128, seeds43/44, newseed45 original-versus-extended; then native seed43 recording. Every process has180s timeout. No new PPO yet; no local training or power-state change.
- Single next action: verify formal configuration reproduces the measured improvement and check the unseen seed and native failure replay; select the model-plus-task configuration only with explicit scope and remaining failures.

## 2026-09-11 -- Diagnose late drive timeouts before further PPO

- Active target: distinguish stalled physical motion from deadline truncation in31 third/fifth-drive failures of selected robustness checkpoint SHA d42e98a2. Original traces show27/31 have actual net strokes >=20degrees, while0/31 have20degrees of counted pair-contact progress;15/31 gain <0.1degree in their final10steps. These are diagnostics, not a revised success declaration.
- Cloud actual handoff/completion checked; instance pro-786095bb6e03 ON, GPU idle at bootstrap. Prior temporary SSH helpers were absent, so refreshed AutoDL snapshot privately and restored the connection helper. No power-state change.
- Inference-only paired comparison active in tmux `m24_timeout_diagnostic`, output `outputs/cloud_pipeline_m24_e002_timeout_diagnostic_20260911`, driver in sibling `_driver/run.sh`. Four128env evaluations: seeds43/44, original deadlines and doubled deadlines. Each bounded180s; unchanged selected policy, original sampled starts and original policy phase-clock normalization. No new PPO/reward/gate changes. New optional evaluator joint-state/target logs and diagnostic timeout multiplier are recorded explicitly.
- Existing physical gate requires both actual net stroke and pair-contact-counted stroke >=20degrees. Tac2Motion II-B Eq3 uses contact-weighted rotation shaping; this exact conjunction is a local rule, not a literature requirement. No conclusion about relaxing it before paired diagnostics.
- Single next action: finish timeout-only comparison, verify paired starts/policy-input prefix, inspect recovered versus stalled failures, then choose one evidence-backed bounded continuation. Preserve prior selected model and strict success counts.

## 2026-09-11 -- Five-cycle robustness improved; bounded cloud work complete

- Git request completed first: `94e8007` (`feat: validate learned return and repeated M24 twist cycles`), 19 files; no push. Initial-state robustness changes were developed afterward and remain uncommitted.
- New selected PPO: `outputs/cloud_pipeline_m24_e002_robustness_20260911/selected_teacher/learned_repeated_twist_robust_1deg.pth`, SHA256 `d42e98a260be396689e0b0bd9dae421021e63cd169da1ecb50efbb710e2a5cb0`. Original repeated-cycle baseline SHA7742e40d remains preserved.
- Nominal candidate743: two cycles **512/512** (previous501/512), five cycles **128/128** (previous128/128). Independent index/thumb initial joint perturbations within +/-1degree, 128 environments each: seed43 **70→94**, seed44 **76→99** complete five cycles. Combined146→193/256 (57.03→75.39%). Exact baseline/post reset-offset hashes match; every successful return stays off-contact throughout. Paired outcomes:75 new successes,28 regressions,118 successes retained,35 failures retained. These two evaluation seeds were used for selection, not a final untouched test set.
- First two-cycle training comparison regressed and was rejected. Diagnosis found28/32 nominal failures after the two-cycle training boundary. Controlled follow-up extended training to five cycles/1100steps; same baseline start, +/-1degree reset noise,50%nominal mixture,seed42,1536envs/3072minibatch,LR0.0001. Each comparison had its own60s smoke and300s PPO; all four phases exit124 as expected. FreshA0c explicitly passed; six baseline and16 post-training evals exit0/finite. Reward weights and physical phase criteria unchanged. No local training,30minute run or student.
- Native cloud recordings both exit0, actual one-env first episodes at20Hz: seed44 completesfive cycles/20ordered events in189frames (9.45s),net150.458degrees; seed43 completesfour cycles then times out during fifth release in268frames (13.4s),net151.780degrees. Both preserved under `selected_perturbed_video_s43`/`s44`; MP4, event keyframes and force/angle/phase plots verified. Seed43 ends off-contact but without completing the required release dwell before timeout; it is not counted as a five-cycle success. One-env reset draws differ from128-env eval batches.
- Remaining selected-model perturbed batch failures:31 drive timeouts (19third-drive,12fifth-drive),27 return timeouts,5workspace violations. Full initial-state robustness, more thanfive cycles,full160mmtravel and real-hand transfer remain unverified. Independent reset helper tests, configured-seed integrity and source syntax checks pass. Local GPU driver mismatch remains; cloud EGL rendering works.
- Artifacts root `outputs/cloud_pipeline_m24_e002_robustness_20260911`: `robustness_acceptance.json`, `selection.json`, `five_cycle_training_comparison.json`, `selected_failure_analysis.json`, source snapshots, commands, timestamps and SHA-verified baseline/first/second/native archives. Successful workflow `..._five_cycles1536/workflow_status.json`; first negative comparison `..._one_degree1536/workflow_status.json`. Full report `docs/m24_e002_robustness_2026-09-11.md`.
- Cloud instance `pro-786095bb6e03` remains ON; all task tmux jobs finished. **Single recommended next step:** diagnose the31 third/fifth-drive timeouts from saved action/contact/joint-travel trajectories, distinguishing stalled motion from deadline truncation, then run one evidence-backed bounded continuation from the new selected model. Preserve nominal and paired perturbation comparisons; do not repeat the rejected two-cycle adaptation.

## 2026-09-11 -- Two-cycle adaptation rejected; five-cycle training comparison active

- First robustness run `_one_degree1536` completed smoke/PPO124 and all8evaluations0. LAST nominaltwo/five475/512and96/128;held-out±1degreefive68/128(seed43),60/128(seed44),both below baseline70/76. BEST nominaltwo/five457/512and91/128also regress. No new checkpoint promoted; original learned-five-cycle baseline preserved.
- LAST nominalfive failure diagnosis:28/32failures occur in third drive/release,after the two-cycle training termination;4second-return failures. New comparison isolates training horizon: start again from originalSHA7742e40d, same±1degreeindependentperturbations,50%nominal,seed42,LR0.0001,1536/3072,rewardweights/phasephysicalthresholds unchanged; extend training fromtwo tofive cycles,episode440→1100. Each phase remains60ssmoke/300sPPO.
- Active cloud tmux`m24_robust_five_cycles`,driverroot`learn_five_cycles.sh`,output`outputs/cloud_pipeline_m24_e002_robustness_20260911_five_cycles1536`. First-run artifacts and exact paired baseline/post comparisons preserved inroot`two_cycle_training_comparison.json`. Evaluation seeds43/44 and nominal checks remain the same for comparison.
- Single next action: complete the horizon comparison and inspect nominal plus held-out actual five-cycle results; do not substitute higher episode reward for goal progress. Gitcommit94e8007preserves prior milestone; robustness work follows it.

## 2026-09-11 -- Measured one-degree robustness gap; bounded PPO active

- Baseline selected five-cycle model under independent index/thumb reset perturbations (128env,5cycles,seed42):0degrees128successes,0.5degrees101,1degree66,2degrees43. Held-out1degree seeds43/44yield70/76successes. FreshA0c passed;all6baselineevaluations exit0/finite. Each perturbed batch has128distinct initial offsets,inactivefinger offsets0,bounds verified;nominaloffsets0.
- Initial1degree failures include37beforefinishingtwo cycles and25later failures. First bounded comparison keeps two-cycle training objective unchanged and evaluates fullfive-cycle behavior. Sample50%nominal plus50%independentuniform±1degree initialjoint perturbations. This targets the measured gap without arbitraryrewardchanges or additional hardgates.
- Active tmux`m24_robust_one_degree`,driverroot`learn_one_degree.sh`,output`outputs/cloud_pipeline_m24_e002_robustness_20260911_one_degree1536`. Separate60ssmoke/300sPPO,1536/3072,LR0.0001,trainingseed42,warm-start selectedSHA7742e40d. BEST/LAST evaluations include nominal512two/128five andheld-outseeds43/44perturbed128five,each180s. No localtraining/student/30minuterun.
- Single next action: finish this boundedcomparison and select only a model with measured held-out robustness gain while preserving nominal learnedrepetition. Git milestone before newrobustness work remains94e8007;continuation changes not yetcommitted.

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

## 2026-09-11 -- Learned two-cycle breakthrough; longer inference verification

- One60s smoke and300s cloud PPO completed (both124),unchanged reward weights/gates,1536/3072,LR0.0001,regrasp curriculum50%original743. LAST original512:512second effective drives,464two complete cycles; BEST original512:501two complete cycles,8first-drive timeouts,3regrasp timeouts. BEST SHA256 `7742e40ddf1d55a48167f011307c77c5ca943306a7754298e433f7bdbc586c7c`; LAST `3d3efe9a8d054f613f34d90cb215c4142d3746fa09b94dac921b885dd02d0cd3`. Prior learned-return baseline preserved.
- Original numeric evaluations initially wrote finite JSON then native shutdown segfault(-11). First retry also missed conda activation (libpython missing),preserved as failed. Activateconda and explicitly select NVIDIA EGL Vulkan ICD per official headless-library guidance; retry3 original/curriculum evaluations now exit0. All failed attempts remain in logs/retry records; do not silently label the original failed pipeline clean.
- Local visualization blocked by hostkernel580.173.02/userspace580.178.04 mismatch and inactive persistenced; no workstation restart or driver changes. Cloud native-camera LAST recording exits0 with the explicit EGLICD. Actual68frame/20Hzuninterrupted episode events1/2/3/4/1/2/3/4,final60.922degrees. Both returns physically off-contact. Keyframes0/13/24/28/34/42/54/61/67visually inspected. GIFloop is playback,not evidence of extra cycles.
- Active next step is inference only:128env five-cycle comparison for BEST/LAST and native-cloud BEST five-cycle recording;tmux`m24_regrasp_longer_eval`,driver root`evaluate_longer.py`,each180s. No additional PPO. Basic learned repetition now demonstrated; maximum sustainable repetition/remaining failure distribution still being measured.

## 2026-09-11 -- Regrasp reachability verified; bounded PPO continuation

- Controlled66candidate diagnostic:9pair recontacts,8next20degree drives under unchanged PPO after controlled closure. Bestcandidate60recontacts within0.252degrees of starting sector and reaches46.980degrees net. Feasibility only; no learned-goal acceptance. Original baseline remains512drive/release/return and0second drives.
- Fresh current-sourceA0c JSON explicitly passed;128env policy boundary and zero preflights exit0/finite,canonicalhandreferenceerror0. Seed0(19/19) and26(16/16) allow next drive;6/12/18/21/24/actualreturn30fail next drive. Zero128/128regrasp timeouts,no free event.
- Train the measured mixture0/6/12/18/21/24/26/30,50%original743,from selected learned-return checkpoint. Each phase independently limited:60s smoke,300s PPO,1536envs/3072minibatch,LR0.0001. No reward weights/phase gates changed. DemoStart-inspired state curriculum is a local adaptation,not demonstration action imitation.
- Cloud tmux `m24_regrasp_closure`,driver root`learn_closure.sh`,output`outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911_closure1536`. Best/last512curriculum and original-start evaluations follow. Report `docs/m24_e002_regrasp_curriculum_2026-09-11.md`.
- Single next action: assess actual second-drive learning and preservation of the original chain; diagnose any remaining gap from the current policy's actual returned state. Repeated screwing remains incomplete.

## 2026-09-11 -- User-confirmed power-on; productive regrasp diagnostic

- User explicitly authorized power-on and continuation. Instance `pro-786095bb6e03` provider running (power-on request `0f913eb33f98a254cba447cfb8185585`, status request `65fc9cf0a7e2d25d86944ee123da7906`); SSH reachable, RTX4090/49140MiB, initially no GPU/tmux jobs. Selected learned-return SHA256 `7bba93e1ada571995b5db6f452530d625b264b6771c6c62e90c6496b0799b4d3` verified.
- Target: productive regrasp and a second effective drive. New root `outputs/cloud_pipeline_m24_e002_regraspcurriculum_20260911`. Add opt-in phase3 full-state initialization using actual learned return frame30. No reward/threshold change. Diagnostic compares66 low-amplitude joint/sector-preserving closure candidates, followed by unchanged PPO drive; controlled results are feasibility evidence, not learned completion.
- Cloud driver `diagnose.sh`, tmux `m24_regrasp_diagnostic`, each diagnostic bounded180s. No PPO launched yet. Prior drive/release/return baseline preserved.
- Single recommended next step: inspect actual recontact sector and next-drive success, then make only an evidence-backed learning correction and bounded cloud validation.

## 2026-09-11 -- User-confirmed cloud shutdown

- Provider subsequently confirmed `shutdown` (request `b4b30177b109af3cf233833d8ec89037`); power-off request `3a6cbc1b371bf3d8e0f0897bc8f519e0` succeeded. This confirmation is local-only because SSH is unavailable after shutdown; the cloud handoff already contains the authorized shutdown intent and idle-state check.
- User explicitly confirmed shutdown of `pro-786095bb6e03`. Pre-shutdown check: provider running, no tmux sessions or GPU compute jobs. Selected learned-return checkpoint SHA256 remains `7bba93e1ada571995b5db6f452530d625b264b6771c6c62e90c6496b0799b4d3`; results are preserved locally and on cloud.
- Shutdown is authorized and being submitted to the provider after this handoff sync. The final provider response/status will be recorded locally under `outputs/cloud_shutdown_20260911/`; verify provider status before any future execution. No training is requested in this shutdown session.
- Single recommended next step: keep the instance off until the user requests continuation; then resume productive-regrasp diagnostics from the saved return model. Repeated screwing remains incomplete.

## 2026-09-10 -- Learned original-start return verified; productive regrasp remains

- **Return achieved:** selected LAST from`_currentpath1536` completes drive/release/return512/512from original candidate743 with one PPO,normal inputs,no demo resets or action/policy-input overrides. All512return trajectories stay off-contact throughout,retain at least25.7498degrees (median26.1594),and end within2.4081degrees of the initial radial sectors. All four final-run evaluations exit0/finite. This is a fixed-pose result,not broad pose robustness.
- The policy then makes pair-contact events512/512,but recontact drifts back to26.7664degrees median sector error. Every second drive times out (0second-drive milestones,0two-cycle completions). **Do not equate the permissive pair-contact/regrasp event with productive regrasp or a completed repeated cycle.** This is the next concrete learning problem; return itself is now verified.
- Selected checkpoint`outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/selected_teacher/learned_drive_release_return.pth`,SHA256`7bba93e1ada571995b5db6f452530d625b264b6771c6c62e90c6496b0799b4d3`. Originalfile`...currentpath1536_ppo/stage1_nn/last.pth`. Baseline drive/release model preserved. Git milestone before this continuation:`f91ec27`; subsequent return work is not included in that commit.
- Four completed cloud PPO comparisons,each separately bounded300s with own60s smoke:near1536,progress1536,earlier1536,currentpath1536. All completed phases training124/eval0. One additional farther1536was intentionally stopped after its60s smoke and2PPOupdates when preflight showed its proposed states already solved; excluded from completed comparisons and preserved with stop evidence. No local training,long-run launch or student. Final curriculum current-path9/12/15,fraction0.5,1536/3072,LR0.0001. Reward weights/physical thresholds unchanged; freshA0c source remains matching.
- Actual local inference playback verified at20Hz with the SHA-matched model.41frame single episode:driveevent13,release23,return30(retaintheta26.407degrees,error1.729degrees,force0/0),recontact40(error27.002degrees,force3.326/2.055N). Inspected frames0/13/23/30/40and angle/force/error plot. Video`outputs/local_pipeline_m24_e002_returncurriculum_20260910/learned_first_cycle_video/learned_return_then_recontact.gif` is one episode; GIF looping is not learned repetition.31frame return-only clip is also saved. Full tensor states at return/recontact are available in these traces for follow-up diagnostics.
- Authoritative acceptance:`outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/return_acceptance.json`,return_subgoal_completed=true,full_cycle_goal_completed=false,repeatable_regrasp_completed=false. Report:`docs/m24_e002_return_curriculum_2026-09-10.md`. Cloud training/eval and local recording are complete; server remains ON. Artifacts/checkpoints/source are archived locally and on cloud; final_execution_state.json records the final process state.
- **Single recommended next step:** use the verified returned state to diagnose/reward a regrasp that stays near the returned start sector and leaves travel for the next stroke; then validate a second effective drive. Do not add a mature-success-rate veto or accept contact alone as repeatability. Preserve the learned return baseline.

## 2026-09-10 -- Current-release trajectory boundary identified; bounded adaptation active

- Current-release controlled return has3endpoint-success candidates, but only62stays off throughout; use62for curriculum,not61/57which recontact. Old starts0/3/34fail,6passes; actual current released26 policy/zero both0/128returns. Current-path early0/3/6/9andactual26allfail under PPO,despite old-path27.8degree state succeeding.
- Boundary preflight128 on current path9/12/15/18/21/24/27/30:9/12fail,15andlater allreturn. Canonicalreferenceerror0,finite. This supplies a measured successful/failed starting-state boundary,not an endpoint-threshold problem. `current_boundary_seed_candidates.json` SHA256`064ce5287ad23824b094cd2d136809b101e0fd27beb005ae0f97102dd212d4d3`.
- Active cloud tmux`m24_return_currentpath`,driver root`learn_currentpath_return.sh`,output`_currentpath1536`. Start third-run LAST; curriculum current-path9/12/15 plus50%original743,physical preflight then separate60s smoke/300s PPO,1536/3072,LR0.0001. Same validatedA0c/source,reward weights and thresholds. Earlier proposedbridge[0,3,6,9,26]was only evaluated (allfail),not trained; earlier all-solved farther run remains intentionally interrupted and excluded.
- Single next action: finish this current-state adaptation and evaluate the original chain; if needed extend towards current actual released26using the measured bridge. Full original return/regrasp/repetition remain incomplete.

## 2026-09-10 -- Discarded already-solved curriculum; diagnose actual release distribution

- Fourth `_farther1536` preflight showed the third model already returns72/72from6/9/12,so these proposed states do not test the remaining full-start gap. Stopped that owned pipeline after its60s smoke and2completed PPO updates; this interrupted run is NOT an accepted5minute comparison. `workflow_status.json=stopped:curriculum_already_mastered`,exact prior phase/pid evidence in`intentional_stop.json`; interrupted checkpoints preserved but not selected. No residual GPU task after stop. Continue from third-run LAST, not the interrupted model.
- Original third-model trace starts return at32.17degrees,improves to24.42then recontacts/fails; old demonstrated states at27.8degrees already return. Angle alone therefore does not characterize the remaining distribution gap. Need actual current released pose/targets/history,not another solved old-state run.
- Captured current model's actual original743release: step26,33.354degree sector error,27.888degree nut angle,force0/0. Archive`current_return_seed_candidates.json`,SHA256`9f52ee811aa427cf7ca4dc797773c9ab35910cbe904783780278982d13758270`; complete source trace saved,normal single-PPO inputs, no controlled actions.
- Active tmux`m24_return_current_diagnostic`,driver root`diagnose_current_context.sh`: compare old states0/3/6/34,policy/zero from current actual release,and65controlled current-state return candidates. Eachphase180s,diagnostic only; no PPO currently running.
- Single next action: identify a learnable bridge from the actual current release distribution and validate its state reset,then bounded PPO with original-start evaluation. Reward weights/thresholds unchanged; full return/regrasp/repetition incomplete.

## 2026-09-10 -- Learned 20degree return segment; fourth bounded continuation

- Third `_earlier1536` complete: smoke/PPO124,all4evals0. Best/last264/264demonstrated starts12/15/18return; original512/512drive+release,0return. LAST SHA256`7b4e95ef4af911a3d95cf8a3ff7019ba4d9627babf6cb4bc9d36578faf61912a`,downloaded/hashverified. All264return trajectories stay off throughout. Return competence now covers the20degree-error intermediate state,not yet actual release at~35degrees.
- Fourth continuation uses earlier controlled states6/9/12 (~27.8/24/20degreeerrors),50%originalstarts; archiveSHA`d39c689cde6e14b3e0c1d1353e3acb91cf17dde3745f86e4cd2ab6057df23de3`. Active tmux`m24_return_farther`,driver root`learn_farther_return.sh`,output`_farther1536`. Each phase is independently bounded: physical preflight180s,smoke60s,PPO300s,then best/last curriculum/original evaluations. Same1536/3072,LR0.0001,verifiedA0c,unchanged reward weights and thresholds.
- Single next action: evaluate this earlier return segment and original start; keep advancing only from measured learning progress. Full original drive-release-return chain and regrasp/repetition remain incomplete.

## 2026-09-10 -- Learned 11degree return segment; third bounded continuation

- Second `_progress1536` run complete: smoke/PPO124 and all4evaluations0. Bothbest/last264/264demonstrated starts18/19/20/21return; original512/512drive+release,0return. All264return trajectories stay off contact throughout,retain27.742degrees,finish at<=2.575degrees sector error. This is autonomous PPO from intermediate states, not a full original-start return.
- LAST SHA256`1b8147979949fb886683f9982022b7c00a94058e6659abe7134d3ba0a4bb9ed4`,downloaded and verified. Diagnostics: `progress_last_return_analysis.json`. Baseline and all prior models preserved.
- Third continuation moves to controlled-trajectory states12/15/18 (~20/15.7/11.1degreeerrors),50%originalstarts,archiveSHA`67cc1a816431aa7a6f460ffdc11f4d552269892bc95f94b90a5fefa96e077d0a`. Active tmux `m24_return_earlier`,driver root`learn_earlier_return.sh`,output`_earlier1536`; preflight then separate60s smoke/300s PPO at1536/3072,LR0.0001. Same verifiedA0c/source,weights and thresholds unchanged.
- Single next action: finish this run and inspect return from earlier/original states; do not confuse successful short return segments with the full drive-release-return chain or repeated cycles.

## 2026-09-10 -- Learned return endpoint; earlier-state continuation active

- First return run `_near1536` completed: smoke/PPO124; all4best/last curriculum/original evaluations exit0. LAST learns return89/89fromseed21; seeds15(67envs)/18(108envs)0returns. Bothbest/last retain original512/512drive+release but0return. The endpoint succeeds with0/0Ncontact force; this is learned partial return, not the original-start chain.
- LAST SHA256`b18400d6f30072c94e2613f516c5b1511e9048637ed141abf061631938630a44`. Root source/commands/checkpoints preserved. First-run failed seeds approach9.54/6.22degree minimum error then recontact; seed21stays off and reaches4.59degrees. No reward/threshold change is indicated by these data.
- Continue progressively earlier states18/19/20/21,50%originalstarts,from this measured LAST; archiveSHA`25c94bec57223b4625a6fc9b3b34a8c7b073a7990882fb4a0e54999bfbda8c29`. Active tmux `m24_return_progress`; driver root`learn_progress_return.sh`; output`_progress1536`. Physical policy preflight then independent60s smoke/300s PPO,1536envs/3072minibatch,LR0.0001. Existing passedA0c remains source-identical. No local/long/student training.
- Single next action: assess this continuation's curriculum and original743 return traces, including recontact during return. Full return/regrasp/repetition remain incomplete.

## 2026-09-10 -- Bounded return curriculum training active on cloud

- Git commit before continuation: `f91ec27`. Physical65candidate diagnostic finds2off-contact returns preserving27.742degrees; best candidate62takes1.40s,allforces0,finalsectorerror4.523degrees. This is controlled-action feasibility, not learned return. Old-policy/zero actual released-state baselines both0returns.
- Near-return curriculum uses controlled-trajectory states15/18/21 (15.73/11.11/7.47degreeerrors),SHA256`1f0834ee18bd022a7396cc3ac07f8250b991cc94d0161dcc31f153c5e0621daf`,50%mixture with original743. Both128env policy/zero preflights exit0,finite,0returns,canonicalreferenceerror0; zero has no free success. Actual-helper CPU/cloud checks pass. Fresh source A0c `a0c_near/manifests/a0c_diagnostic.json` explicitly passed.
- Active cloud tmux `m24_return_near`; driver `outputs/cloud_pipeline_m24_e002_returncurriculum_20260910/learn_near_return.sh`; runoutput root plus`_near1536`. Separate60s smoke and300s PPO,1536envs/3072minibatch,LR0.0001,starting unchanged selecteddrive/releasecheckpoint. Reward weights,contact thresholds,return tolerance unchanged; phase statistics already have healthyvariance and were not reset. No local training,long run or student.
- Single recommended next action: monitor this boundedrun,then inspect best/last512env curriculum and original743 evaluations. Preserve original baseline even if the newest checkpoint forgets drive/release. Learned return/full repeated cycle are not yet demonstrated.

## 2026-09-10 -- Git milestone committed; return curriculum diagnostics active

- User requested one Git commit then continuation. Local commit `f91ec27` preserves the verified learned drive/release milestone, diagnostics, fixes, tests and handoffs (63 files); no push or binary checkpoints in Git. Post-commit work targets actual return, not another release-only run.
- Cloud reachable/idle before this session; RTX4090/49140MiB. Headless single-PPO original743 state capture exited0: release atstep34,27.742degrees retained,35.575degrees return-sector error. Continued old policy recontacts the nut and times out returning. Root: `outputs/cloud_pipeline_m24_e002_returncurriculum_20260910`.
- Added opt-in full tensor-state capture to cycle evaluator and phase2 seed resets with imported event counts cleared and canonical initial-hand context preserved. Return subtask code7 requires an actual off-contact return event with retained progress; full-cycle code5 remains separate. New Return task/train configuration. Existing release/default entrypoints preserved; reward weights and physical thresholds unchanged.
- Actual-helper CPU subset/history/context/no-credit checks pass for release and return states. Cloud action diagnostic65candidates and policy/zero preflights run under individual180s limits, tmux `m24_return_diagnose`; no PPO launched yet. This uses DemoStart III-A intermediate-state initialization as a local fixed-state adaptation, not its full automatic curriculum.
- Single recommended next action: inspect actual return reachability and restored-state preflights, then fresh current-source A0c and a separately bounded60s smoke/300s cloud PPO, evaluating original743 as well as demonstrated starts. Baseline learned drive/release checkpoint remains preserved. Full return/regrasp/repetition is incomplete.

## 2026-09-10 -- Release subgoal achieved from original743 with one PPO

- **Current result:** selected LAST from `_context1536` completes learned drive512/512 and actual two-finger release488/512 (95.3125%) from original candidate743, with normal task inputs and no demonstration resets, scripted actions or policy composition. All accepted releases follow a measured drive event, retain at least27.489degrees (median27.822), and have both contacts off/forces<0.15N for two endpoints; final force0/0N. Initial-hand-reference error0. No workspace/jump termination.24release timeouts remain. Return/regrasp/repetition is NOT learned.
- Checkpoint `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/selected_teacher/learned_drive_release.pth`,SHA256 `aa0e55b06ca7fd864b2ccf717bd73bd95c173e9eface25be93c3328450b9f471`; original training file is `_context1536_ppo/stage1_nn/last.pth`. Reward-best18.50only20/512releases; LAST is deliberately selected by behavior. Both models and source are preserved,downloaded,hash-verified.
- Four separately bounded cloud PPO runs this session (each300s plus bounded completion grace),with separate60s smokes: `_near6144`14updates,`_value1536`52,`_early1536`61,`_context1536`57. Last run1536envs/3072minibatch,seeds18/28/36,demofraction0.5,initialLR0.0001. Reward weights/actual release criteria unchanged. No local training,30minute run or student. All final-run phases exited expected124(training)/0(evaluation).
- Key corrections: reset the articulated nut's position/velocity; restore saved critic value-normalization statistics; preserve canonical episode initial-hand-pose context during intermediate-state resets. The last mismatch was introduced by this session's new demo helper and is now fixed. CPU actual-method/helper tests and physical cloud preflights pass. Earlier context-override512/512result remains diagnostic-only and is not the selected solution.
- Actual **single-episode** playback verified at20Hz: `outputs/local_pipeline_m24_e002_releasecurriculum_20260910/learned_release_video/learned_drive_then_release.gif`,35frames,net27.749degrees,finalforces0/0N. Inspected initial,drive-complete and released frames. Thumb opens before index by0.60s; this is not perfectly simultaneous release or a repeated cycle. Cloud graphics initialization crashed; the identical SHA-verified model was recorded locally with test=True/inference only. Video/trace/plot are mirrored to cloud.
- Authoritative acceptance: root pipeline `release_acceptance.json` (`release_subgoal_completed=true`, `full_cycle_goal_completed=false`). Final source: `final_source/SHA256SUMS`. Report: `docs/m24_e002_release_curriculum_2026-09-10.md`. Source,commands,evaluations and best/last checkpoints preserved locally and on cloud. The second run's best-eval shutdown crash has clean separate retries in `retry_completion.json`.
- Current execution: cloud training/evaluation complete; local visualization complete. Server remains ON. Final process/GPU state is recorded in `final_execution_state.json`.
- **Single recommended next step:** use the selected single-PPO checkpoint to collect actual released-state seeds, then learn return while keeping both fingers off and preserving net nut rotation; evaluate from original743 and retain this drive→release checkpoint as the baseline. Do not treat release as completion of return,regrasp or repeated cycles.

## 2026-09-10 -- Confirmed curriculum context mismatch; corrected short continuation

- Third `_early1536` run completed all phases: smoke/PPO124; best/last demo and original evaluations exit0. Last mixture264/264demonstrated states release (18:67,28:108,36:89); all248originalstarts drive but fail release. Dedicated original512/512drive,0release,median40.183degrees. This is not an original-start solution.
- Two learned-policy inference diagnostic (original stable drive + latest release) also0/512release,retaining23.319degrees. First wrapper attempt failed CPU/CUDA mask device mismatch before evaluation; fixed selector mask device, retained failed log, retry exit0. This is optional inference only, not PPO redesign/student training.
- **Causal one-feature diagnostic:** keep both models,physical scene,phase events and thresholds unchanged; feed only the release expert the demonstrated step18 hand-initial-pose reference. Result512/512drive+release,median23.305degrees,exit0. No scripted joint actions, but the feature override is DIAGNOSTIC ONLY, not accepted as the completed goal. Artifacts `learned_skill_chain_retry/` vs `hand_reference_diagnostic/`.
- Root cause in this session's new demo helper: it overwrote privileged `hand_joint_pos` with the intermediate physical pose. Base reset writes that field only once, before noise, so it represents nominal episode initial pose; normal full-start rollouts keep743 there. Fix removes the overwrite, preserving canonical task context while restoring actual physical DOFs/history. New actual-helper context regression fails before,passes after; cloud test passes. No reward/threshold change.
- Fourth bounded continuation: tmux `m24_release_context`,root `learn_release_context.sh`,fresh A0c `a0c_context`,same1536envs,seeds18/28/36,fraction0.5,LR0.0001,starting early-last. Preflight explicitly reports canonical initial-hand-reference error; smoke60s/PPO300s then best/last evaluations. Output suffix `_context1536`.
- The second run's best evaluation shutdown crash was retried cleanly: best-near512release and best-original0release, both exit0. `retry_completion.json` preserves original failed workflow status plus successful retries.
- Single next action: verify context-consistent learned release from the real original start; record it without any policy-input override. Full release/return/regrasp/repetition is not yet achieved. Cloud ON; no local training or automatic30minute/student run.

## 2026-09-10 -- Learned near-release512/512; earlier/full-start mixture now training

- Corrected-value continuation `_value1536`: smoke60s/PPO300s exit124. Last checkpoint deterministic512/512 releases from seeds40/43 (256each),all code6,median23.791degrees; evaluator exit0,finite. Original743 last0drive/0release,median17.974degrees,all drive deadlines. This is learned near-release, NOT a full original-start solution.
- Reward-best near-state JSON also contains512releases, but process crashed with SIGSEGV after writing/printing the report, during shutdown. Overall workflow accurately remains `failed:execution`; best result requires a clean retry and best-original eval was not run. Preserve failure evidence. Continued from independently validated LAST, not the runtime-failed best evaluation.
- Earlier-state preflight128:72demo starts (18:30,28:23,36:19) plus56originalstarts. Zero-action0release; policy releases19/19from36,0from18/28 and0originaldrive. Both modes exit0,finite. This supports advancing the curriculum into early unloading while mixing original drive practice.
- Cloud tmux `m24_release_early`,driver root `learn_release_early.sh`; third independently bounded smoke60s/PPO300s,1536envs,3072minibatch,seeds18/28/36,demofraction0.5,LR0.0001. Same reward weights, force thresholds and validated source/A0c. Output suffix `_early1536`. Cloud is on; local machines only development/analysis.
- Single next action: finish mixed training, evaluate actual original-start drive→release, retry the second run's best eval separately, and record learned motion without scripted actions. Full return/regrasp/repetition remains unachieved.

## 2026-09-10 -- Near-release first run complete; corrected value normalization continuation

- First cloud near-state run completed: fresh passed A0c,smoke60s/PPO300s both124,6144envs/12288minibatch. Best/last512-env evaluations each0release from demo seeds40/43; original743 each512drive/0release,median23.707/22.943degrees. Pressure traces show continued squeezing, not unloading (last thumb32–38N). All evaluations finite; no deterministic jump/workspace failure. Original drive remains preserved; release is NOT learned.
- New bounded fix: `PPO.restore_train` loaded the critic weights but omitted saved `value_mean_std`, so rollout values used different units after resume. Restore that statistic when present; legacy checkpoints without it still load. Actual-method CPU test using the saved checkpoint: max denormalized value error1.455477->0. This proves the restore defect, not its share of the learning failure. Reward weights unchanged.
- Cloud continuation: tmux `m24_release_value`,driver `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/learn_release_value_restore.sh`,fresh A0c under `a0c_value_restore`,results suffix `_value1536`. Start the first run's last.pth,keep seeds40/43,LR0.0001.1536envs/3072minibatch increases completed updates in a5minute budget; this is not a one-factor reward ablation. Each smoke/PPO phase independently bounded60/300seconds.
- Single next step: inspect this continuation's best/last demo/original evaluations and advance the initialization curriculum only from measured release progress. Preserve original stable-drive checkpoint. Cloud is ON; no local training,30minute run or student.

## 2026-09-10 -- Autonomous release curriculum in progress on cloud

- User explicitly requested the next subgoal: learn release. AutoDL power-on now succeeds (request `8e99b32b5905767ea80b8847264534df`); cloud RTX4090/49140MiB, initially1MiB/no tmux job. Read cloud-side handoff before syncing; old cloud source files preserved in `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910/source_before/`. Prior no-inventory blocker no longer applies. Training is cloud-only this session.
- Target: DemoStart-style intermediate-state initialization inside existing PPO, first validated states40/43. New opt-in release config terminates on actual two-finger release with retained net angle (code6, distinct from full-cycle success5). Imported drive/release/cycle counters clear; angle high-water and history persist without replay credit. Final evaluation also uses original743 without demo resets.
- **New confirmed reset bug:** original reset_idx leaves object nut angle/velocity untouched. Actual4env subset check: reset env0retains0.4rad/0.5rad/s, env2retains0.3rad/0.3rad/s; untouched envs unchanged. Cloud `reset_before.json` records failure. Previous first-episode evaluations remain valid; repeated-episode training was contaminated, so prior late jump failures cannot be attributed solely to reward/learning-rate choices.
- E002-only fix clears nut DOF state and submits both hand/object actors to indexed DOF reset. The identical cloud subset check now passes: reset error0, untouched change0. Demo reconstruction then explicitly restores selected articulated-object state. No reward-weight or force/contact threshold change.
- Active pipeline `outputs/cloud_pipeline_m24_e002_releasecurriculum_20260910`; local evidence mirror `outputs/local_pipeline_m24_e002_releasecurriculum_20260910`. Physical reset and curriculum/zero-action baselines precede fresh A0c, smoke60s and PPO300s. No30minute/student run.
- Preflight completed: policy and zero-action baselines each0/64releases from seeds40/43, finite, all release deadlines. Fresh current-source A0c explicitly passed. Cloud tmux `m24_release` is running smoke60s followed by PPO300s,6144envs/12288minibatch,initialLR0.0001,unchanged reward weights. Driver `learn_release.sh`; learning results under the root pipeline plus suffix `_near6144`.
- Single next action: monitor the bounded release curriculum, then inspect best/last512-env demo-state and original743 evaluations; use actual learned release and retained drive to choose the next curriculum step.

## 2026-09-10 -- Cycle learning and validated release-state seeds; full goal still incomplete

- Target: learned drive -> release -> return -> regrasp -> repeat. Implemented opt-in `XHandPasiniM24NutBoltE002Cycle`,101Dphase observation, measured event transitions, global high-water across cycles, separate phase failures and fixed first-episode cycle evaluator. Original E002 remains available. Report: `docs/m24_e002_cycle_learning_2026-09-10.md`.
- Cloud remains shutdown after prior no-inventory power-on failure; user-authorized local RTX3060fallback used. **All local training/eval/record/replay jobs are finished; no GPU compute process remains.** Final provider status verified `shutdown`, request `67efaba785cfcdad4900eb02d1b68baa`. These updates are local-only and pending cloud sync. No30minute or student run.
- Three separately bounded comparisons, each fresh passed A0c ->60s smoke ->300s PPO ->best/last512-env evaluation: `outputs/local_pipeline_m24_e002_cycle_20260910_probe3072`, `...cycle_lr1e4_20260910_probe3072`, `...relativeforce_20260910_probe3072`. Each phase records exact command/timestamps/checkpoints; training exit124 expected, all six evals exit0 and finite. Goal incomplete in every JSON.
- First cycle run: last505drive/0release,median56.99deg; reward-best97drive/0release,median1.33deg,allworkspace failures. InitialKL4.335against0.02 prompted an isolated initialLR0.005->0.0001comparison. That reduces firstKL to0.0107; best/last512drive/0release,median23.25/58.52deg,all release timeouts. Late stochastic angle-jump failures persist.
- Third comparison changes only release potential to recorded-entry-relative force; keeps0.5scale and all actual event criteria. Supported by successful-trajectory counterfactual (early reward0.00356->0.12659,total0.48949->0.5). Best512drive/0release,median24.89deg; last0drive/0release,median2.99deg,512drive timeouts. This did not solve learning and is retained as an opt-in comparison, not promoted as a solution. No autonomous release/return/regrasp/repeated cycle has been demonstrated.
- Physical action diagnostics:55low-amplitude candidates identify outward unloading; directed8candidate test yields6true pair releases retaining23–24deg. Video `outputs/local_pipeline_m24_e002_cycle_20260910/release_reference_video/drive_then_scripted_release.gif` records one47frame/20Hz episode: learned drive plus SCRIPTED release, finalforces0/0N,net23.796deg. Inspected loaded/released frames. This is feasibility evidence only.
- New `replay_m24_e002_release_states.py` reconstructs5recorded intermediate states and continues recorded actions:5/5release. Early states have transientforce differences up to4.12N; near-release states40/43have maximum angle error0.00649/0deg and force error0.00990/0.02688N. Seed archive and replay results are under the video and `release_state_replay/` directories. Archive itself retains unvalidated provenance; separate replay JSON supplies validation evidence. Solver caches are not serialized.
- Actual tensor tests, relative unloading/force-oscillation/reset tests, existing learning and A0c contract checks, syntax checks all pass. Normalization preparation changes only constant phase-bit statistics, not modelweights. Recorder/export/source snapshot fixes are included; train/task config snapshots now use distinct names. Full source and consolidated results: `outputs/local_pipeline_m24_e002_cycle_20260910/final_source` and `session_results.json`.
- Preserve the original stable-drive default: `outputs/XHandPasiniM24NutBolt_teacher/local_pipeline_m24_e002_contactprogress_20260910_retry3072_ppo/stage1_nn/last.pth` (SHA `ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`). Do not automatically select the newest cycle checkpoint.
- **Single recommended next step:** implement an opt-in DemoStart-style release-state reset curriculum using validated states40/43 first, within existing PPO. Preserve physically verified target/history/cycle state reconstruction and no free angle credit; gradually move to earlier states. Evaluate full behavior from original candidate743 with no scripted actions or demonstration resets. Primary-source basis and exact seed/replay paths are in the report. Do not repeat the failed full-start settings unchanged.

## 2026-09-10 -- Revised gates produced stable learned20degree drive locally

- User authorized relaxing normal-moment/mature-success gates and local training when cloud has no resources. AutoDL power-on failed with no inventory (`InternalError`, request `5dc32034b9c3ed3712ae56d689677ce8`); instance remains shutdown. This entry is local-only until cloud becomes reachable. No local/cloud training or evaluation job remains active.
- Report: `docs/m24_e002_gate_revision_2026-09-10.md`. Active `acceptance_mode=contact_progress` uses nut-only contact and actual new rotation for drive progress/milestones. Normal moment/balance remain diagnostics. Early90%/95% thresholds and exact50ms synchrony no longer have veto power. All prior conflicting plan restrictions are superseded by the September10 amendment.
- Actual reward CPU-tensor tests, A0c false-pass tests and existing expression checks pass. Fresh local candidate743 A0c explicitly passed, candidate22,17 contact/direction steps,7.362deg new motion under pair contact. SHA `e67e3626ae141bf083241edfa67bc959016970b2ae6ee5b11a005fede8a58d49`.
- Local RTX3060/12GiB; existing IsaacGym Docker, torch2.3.1+cu121. Smoke60s ended expected124. A launcher falsely matched historical diff text containing Traceback; fixed anchored matching, reused the verified smoke with identical A0c/starting weights. Original evidence retained in `outputs/local_pipeline_m24_e002_contactprogress_20260910_probe3072`.
- PPO pipeline `outputs/local_pipeline_m24_e002_contactprogress_20260910_retry3072`:3072envs/6144minibatch,15:46:39–15:51:46 +08,300s timeout plus7s completion grace,expected124.43completed updates, counter1,622,016 includes initial-batch offset. Warm-started September8 best policy/normalization with fresh optimizer. Sampled peak GPU4724MiB, stable5.2–5.5k environment steps/s.
- **Matched local revised-contract evaluations:** old starting policy0/512 hold20deg for4pair-contact steps,0successes,510workspace+2contact losses. Latest policy512/512 hold20deg,159/512 meet25–35deg drive band+4step hold,353drive deadlines,0workspace/contact-loss/angle-jump terminations; median24.64deg.0/512 reaches30deg. Reward-best has median9.30deg and0successes, so select `last.pth` by behavior.
- Selected directory `outputs/XHandPasiniM24NutBolt_teacher/local_pipeline_m24_e002_contactprogress_20260910_retry3072_ppo/stage1_nn`; `last.pth` SHA `ab0c2fe729372fb70b9e8f711a1d94fefe2ac54f1597f2c5b2462191305c630c`; `best_reward_5.44.pth` SHA `d91fcd9a39c0e7ce3bb388f1f1a6cc0f03e7101bfd8c4831671a597f948e50dc`. Full source/commands/checkpoints/TensorBoard/evaluation traces retained locally.
- A0c/analysis pipeline `outputs/local_pipeline_m24_e002_contactprogress_20260910`; `results_summary.json` and `policy_video/learned_drive.gif` document this result. One actual81frame/20fps first episode reaches23.67deg then holds until deadline. Inspected rendered initial/held frames; no resets are concatenated as cycles.
- Known applied nut-torque diagnostic is now complete:12cases with/without hand,200physics samples each. No-contact0.05/0.1Nm yields20.03/66.58deg; endpoint velocity integrals stay near0 even without finger contact. The grasp resists0.1Nm (2.02deg). This invalidates using endpoint qdot alone as a motion veto, not a proof of complete physical fidelity. Artificial nut torque is diagnostic only; production physics/actuation unchanged.
- Remaining limits: full release/return/regrasp/repetition is not trained; stochastic exploration still exhibits frequent large-angle terminations; normal-force median11.68/28.90N and maxima15.81/33.90N are not load-optimized. Full goal remains incomplete; drive success is not cycle success.
- **Single next action:** add and test release/return/regrasp learning using the already demonstrated approximately20degree stable stroke. Preserve net rotation across cycles; do not wait for30degree/90% maturity, and do not count phase timeouts as success. No automatic30minute run or student stage.

## 2026-09-09 -- User-requested cloud shutdown

- Verified locally at 20:30:37 +08:00: provider status is `shutdown` (`Success`, request ID `9328472ae2afde6c53c13adc82de1914`), after a successful `power-off` command at 20:30:25. Evidence: `outputs/cloud_shutdown_20260909/verified_shutdown.json`. This final verification line is local-only until a future authorized startup; the pre-shutdown notice below was synced and hash-verified on both machines.
- Target: shut down AutoDL instance `pro-786095bb6e03`, explicitly confirmed by the user. Cloud execution is paused; do not automatically restart the instance or launch training.
- Preflight at 20:28:51 +08:00: remote hostname matches the instance; no tmux sessions or training/evaluation/probe processes; GPU 1 MiB, 0% utilization, no compute processes. Only platform Jupyter/TensorBoard services remain.
- Latest best/last checkpoint SHA256 values were rechecked on both machines and match the values in the September 8 entry below. Both handoffs also matched before this update; existing results remain preserved.
- This shutdown notice is synced to the cloud before power-off. The local `outputs/cloud_shutdown_20260909/` directory records the power-off response and subsequent provider status; final verification cannot be synced to a powered-off instance.
- No reward, gate, physics, or training change. The learned twisting goal remains unmet.
- Single next action: keep the instance off until the user requests resumption; then reread the current handoff before continuing the pending diagnostic.

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

## 2026-09-08 -- Reset fixed; PPO rejected; 31.54-degree cooperative reference found

### Target and current result
- User asked continuous execution toward actual learned twisting and primary-
  literature-grounded reward work. Goal is NOT complete. No weight or gate
  changes, B/C, student or 30-minute run were made. All training is cloud-only.
- Full report: `docs/m24_e002_reset_and_reachability_2026-09-08.md`.
  Primary-source comparison and unapplied reward proposal:
  `docs/m24_e002_literature_alignment_2026-09-08.md`.
- One production fix in `dexscrew/tasks/xhand_pasini.py`: initialize E002 reset
  history before actor observation assembly. Real cold/subset-reset checks have
  errors0, nonreset change0 and finite observations. Unrelated tasks unchanged.
- Pipeline `outputs/cloud_pipeline_m24_e002_reset_history_20260908_171000`.
  Fixed candidate743 A0c passed with unchanged bank/gates; SHA256
  `8e40add7eae7b95349439e216eb5018aa454bd1142d4eb850f54fe176498ca68`.
- Existing models reevaluated: both fail. New cloud smoke passed124; PPO
  12288/24576 ran17:14:08–17:19:26 +08, timeout300 plus18-second completed-update
  grace, expected124. Best now update11 (-0.90356), last update12 / counter
  1,916,928 (includes batch offset). Both saved, synced and hash verified.
- New best/last each fail 512-env/100-step first-episode evaluation:0/512 success,
  effectively zero angle, no credited steps. Contact loss best atstep10, last
  atsteps10–18. `phase.txt` / `acceptance.json` remain numeric-gate rejection.
  No headed acceptance is claimed. Checkpoint names/hashes are in the report.

### New physical evidence, not a learned policy
- Diagnostic `probe_m24_e002_contact_control.py` maps contact-point force targets
  through J.T and the unchanged PD/action interface. No virtual nut torque is
  applied. Termination suppression is recorded; no diagnostic is a policy pass.
- Same candidate743, 16N normal target, both tangent ratios0.5:31.54 degrees,
  all80 endpoints with pair contact, normal-moment gate0. Remove either finger's
  tangential command: about0.001 degrees with pair contact retained. This
  supports cooperative driving under these commands, not measured full wrench.
- Raw GPU contact audit confirms only nut contacts; all fingertip resultants
  reconstruct from normal records within9.54e-7N. Friction remains unmeasured.
- Explicit CPU-PhysX reference reaches28.63 degrees but differs physically.
  Extra resultant force is thumb-bolt contact, not recovered friction. Zero
  lambdaFriction fields on both backends. No production source/backend change.
  Misnamed `contact_ablation_cpu` actually retains GPU PhysX and is marked as
  a GPU replicate; `contact_ablation_cpu_physx` is the real CPU reference.
- Figure: pipeline `contact_gate_evidence.png`; complete commands, raw records,
  control ablations, training/eval data and local/cloud artifacts are retained.

### Single next action and required input
- The user was asked for one explicit exception: change only the training
  progress multiplier to literature-style opposing nut-contact quality, keeping
  all weights and strict A0c/acceptance pair gate0.50 unchanged. The concrete
  patch is prepared but NOT applied. No answer has arrived; do not infer consent.
- Once authorized, one A0c -> smoke -> five-minute PPO comparison, with best/last
  strict evaluation. Nut-only contact must exclude bolt contact. The strict
  normal-only contribution metric still requires validated friction treatment
  before claiming final learned behavior. Do not hide that limitation.
- Do not repeat the rejected PPO unchanged. No active cloud tmux/train/eval/probe
  process remains; GPU idle1/49140 MiB,0%. Goal remains unmet, pending the explicit
  training-gate decision. The approved reset fix is complete and preserved.

## 2026-09-08 -- Reward conflict corrected; A0c passed; best/last PPO rejected

### Target, correction and outcome
- The user's reward-audit request superseded the earlier next-step suggestion.
  Read the supplied review as reference and checked Tac2Motion/DexScrew primary
  papers. Report: `docs/m24_e002_reward_audit_2026-09-08.md`.
- One reward expression changed: invalid-forward cost now uses opposing
  contact-confidence deficit, not the complement of torque quality. Old motion
  component `4*n*(2*g-1)` punished improving motion for `g<0.5`; positive reward
  remains `4*g*n`. All weights, pair gate 0.50, thresholds, physics, pose and
  strict valid-progress/milestone/success gates are unchanged.
- Actual-expression regression checks and A0/diagnostic checks passed.
  New A0c explicitly passed, 4/27 candidates; physical traces match old A0c
  exactly. Candidate 19 motion-component replay: -9.5371 -> -0.15029.
  Accepted-window movement is still only 0.000760 rad, not a useful stroke.
- Pipeline: `outputs/cloud_pipeline_m24_e002_reward_contactdeficit_20260908_161444`.
  A0c SHA256 `18bc2d493db8387096038eda4ef50a8680862f8e1918e2a8dee4d514f6a4ec42`.
  Initial `a1` smoke failed before training because imported Hydra could not
  resolve configs. Absolute config path was fixed; original logs preserved.
- `a1_retry`: smoke passed expected timeout 124; PPO 12288 / 24576 ran
  16:21:37–16:26:52 +08 with timeout300 plus 15-second completed-update TERM
  grace; expected exit124. Completed-update last model is epoch11 / counter
  1,769,472; best reward is epoch1 / counter294,912. Counter has a batch offset.
  Policy snapshots are inference weights, not optimizer-resume checkpoints.
- Fixed evaluation: each model, 512 envs / 100 steps, first episode before
  reset. Both exit4, 0/512 successes, no credited progress, max gate0.
  Best median net1.61574e-7rad, contact-loss termination step15; last median
  9.11651e-8rad, termination step10. Both lose the thumb, index stays in contact.
  `phase.txt` / `acceptance.json`: `rejected:fixed_step_numeric_gate`.
- Best/last checkpoint directory (local/cloud hashes verified):
  `outputs/XHandPasiniM24NutBolt_teacher/m24_e002_contactdeficit_retry_s42_20260908_162100_probe_12288_24576/stage1_nn`.
  Full paths, SHA256s, source/config snapshots, commands, timings, TensorBoard,
  replay and full evaluations are saved in the report and pipeline.

### New diagnostic evidence, limits and single next action
- A final read-only one-env reset-observation counterfactual saves
  `reset_observation_audit/audit.json`. It performs no training or post-reset
  physical rollout. The initial actor history segment is all zero although
  the initialized history contains candidate-743. Task code assembles actor
  observations before refilling reset history. Replacing only that segment at
  the same physical state changes the best model's maximum active-joint action
  from 1.0 to about 0.1902. The last model still has one saturated active joint.
  This proves an observation-order mismatch and action effect, not successful
  behavior or that it is the sole cause of contact loss. No second task fix made.
- Next: one E002-only reset-history initialization-order fix, rerun A0c with
  unchanged gates, then reevaluate existing best/last models before any new PPO.
  Do not repeat the failed training configuration blindly.
- Goal remains unmet. Agent owns cloud execution; all training is cloud-only.
  No 30-minute training, E002-B/C, student or headed acceptance was launched.
  Current cloud: no tmux/train/eval job; GPU idle 1/49140 MiB, 0% utilization.

## 2026-09-08 -- A0c passed; cloud five-minute PPO completed, behavior rejected

### Target and result
- Agent owns cloud A0c -> smoke -> five-minute PPO -> evaluation. This chain
  was executed. The user's learned 30-degree pair-drive goal remains unmet.
- All training is cloud-only. No 30-minute run, E002-B/C or student was launched.
- Current cloud state: no tmux sessions or training/eval processes; GPU idle,
  1 MiB / 49140 MiB. Read-only result pipeline:
  `outputs/cloud_pipeline_m24_e002_passed_a1_20260908_154343`.
  `phase.txt` / `acceptance.json`: `rejected:fixed_step_numeric_gate`.

### Changes and verified evidence
- Full report: `docs/m24_e002_cloud_recovery_2026-09-08.md`.
- One evidenced torque-source correction uses actual nut normal-contact points,
  validated against independent reconstruction and same-step tensors. E002 CPU
  tensors enable per-env contact API; GPU PhysX and CUDA PPO remain. Existing
  weights, contact/torque/pair thresholds, pose and control/physics rates are fixed.
  Full friction wrench remains unmeasured. The simulation-wide contact API
  returned invalid indices and is excluded; production uses copied env records.
- Paired old/new CPU A0c physics traces match exactly; only torque geometry
  differs. Contact-aware action proposals and scheduled late-increment taper
  then produced a passed A0c, without another task/reward change.
- Passed A0c:
  `outputs/cloud_pipeline_m24_e002_taper_refine_20260908_154240/a0c/manifests/a0c_diagnostic.json`;
  SHA256 `9a2ee6a0e46281a0a12631ead90138b38ea311f90ff45096cff48891f7ed8d5b`.
  4/27 candidates pass with sign -1. Candidate 19 has 7 consecutive accepted
  steps, net 0.1256776452 rad, maximum gate 0.6420834; full schedule is saved.
- Guarded import, A0 checks and trace exited 0. One-env smoke exited expected
  124. PPO 12288 envs / minibatch 24576 ran 15:44:59–15:50:00 (+08:00), expected
  124, stable ~5.8k actor steps/s; persisted step counter reaches 1,622,016.
- Selected checkpoint (local/cloud SHA verified):
  `outputs/XHandPasiniM24NutBolt_teacher/m24_e002_contactpoint_s42_20260908_154343_probe_12288_24576/stage1_nn/best_reward_-1.07.pth`;
  SHA256 `699cee72ec4b1c6d33037577f13368ed0dacbf30089301e16eb50e97589fa21b`.
- `fixed_eval/evaluation.json`: 512-env / 100-step deterministic inference,
  exit 4. Success 0/512; contact loss 512/512 at 13–14 steps; median net
  1.63298e-7 rad; zero credited progress; hard-speed terminations 0/512.
  Per-env first episodes and steps are captured before reset. No headed pass
  claimed; immediate numeric rejection prevents further promotion.
- Static/deterministic checks, real two-env reconstruction, A0c and training
  startup passed. Full diagnostics, code/config hashes, command/status/timing
  records, TensorBoard, evaluation and checkpoint are synced locally/cloud.

### Remaining limits and single recommended next step
- Best checkpoint is epoch 1 / counter 294,912; no final `last.pth` was saved.
  Evaluation does not represent an unavailable final optimizer state.
- Accepted A0c steps 17–23 have only 0.0007598922 rad summed progress, including
  near-static increments. Existing A0c criteria pass, but useful sustained
  motion is not established. Do not confuse that pass with E002-A success.
- Next: cloud diagnostic comparison of the PPO policy's first 14 steps versus
  zero hold and the accepted A0c schedule, isolating early thumb contact loss
  before another bounded fix. Do not rerun the failed training configuration,
  change reward weights, lower pair gate, or start 30-minute/B/C training.

## 2026-09-08 -- Candidate-743 Contact-Point Audit Completed

### Target milestone/subgoal
- Execute the previous handoff's one-environment cloud contact-point/normal/
  wrench audit. Validate the link-origin torque proxy before any further
  single task fix. All training remains cloud-only and gated by explicit A0c.

### What changed (files + behavior impact)
- Added `scripts/audit_m24_e002_contacts.py`, no-training cloud wrapper
  `scripts/cloud_run_m24_e002_contact_audit.sh`, offline analyzer
  `scripts/analyze_m24_e002_contacts.py`, and independent geometry check
  `scripts/test_m24_e002_contact_wrench.py`.
- Each selected candidate saves all 200 Hz physics endpoints, raw contacts,
  body poses/velocities, COM/shape friction metadata, and original diagnostics.
  Paired GPU/CPU tensor modes retain GPU PhysX; analyzer reports their actual
  differences and reconstructed normal moments separately from full wrench.
- Task code, reward weights, gate, fixed candidate-743 pose, source action
  vectors and physics settings are unchanged. The prior axis correction is
  still the only task-behavior fix; no second speculative fix or A0c rerun.
- Evidence: `docs/m24_e002_contact_audit_2026-09-08.md`; plan/ledger updated.

### What was verified (commands + key outcomes)
- Read live cloud handoff and prior fixed-pose manifests before execution.
- Authoritative pipeline (full local/cloud copies):
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_contactbank_20260908_140118`.
  Start/end `2026-09-08T14:01:18+08:00` / `2026-09-08T14:02:54+08:00`;
  `phase.txt=completed:contact_audit_only`; 12/12 simulator exits 0,
  each with its own timeout 180 in tmux. Exact commands, configs, timestamps,
  statuses and source hashes are under that pipeline.
- Six fixed actions: IDs 0/34/36/46/63/64 (zero, three +0.1 single joints,
  both equal-ratio tangent directions). 3840 total physics samples, including
  1920 with 23088 raw contact records; no unexpected reset or nonfinite data.
- Normal contact forces reconstruct the reported same-step tensor within
  3.4752e-6 N. Only index-nut and thumb-nut pairs have active normal forces.
- Index proxy sign differs from contact-point normal moment on 600/600 probe
  samples (thumb 0/600). Zero-action last index moment is +0.00269912 Nm by
  proxy but -0.00439155 Nm at the actual contact points, both using -Z/-1.
- Both signs have zero simultaneous normal-torque/balance/full-condition
  passes. Maximum normal-only gate is 0.053007, below unchanged 0.50;
  greatest audited net angle is 8.09177e-7 rad, below pi/180.
- Actual hand/object shape friction is 1.0; contact friction fields are all
  zero. This does not establish zero physical friction or total contact wrench.
- Local/cloud offline geometry checks and analysis passed; Python/shell syntax
  checked. No local simulator or training ran. Current cloud has no tmux or
  training/audit process; RTX 4090 idle at 1/49140 MiB and 0% utilization.

### Remaining blocked/risky
- Full tangential/contact torque is unverified. Do not substitute normal-only
  torque, a guessed contact offset or an unvalidated body sensor for it.
  Solver force sensors can include joint constraints.
- GPU/CPU tensor trajectories differ: settled-hold force difference <=0.00494
  N, but reset transient <=0.660 N and probe difference <=0.518 N. Per-phase
  differences are saved; same-step contact/proxy comparison is within one mode.
- Startup pilot `..._m24_e002_contactpilot_20260908_135419` failed in COM metadata
  extraction before samples and is excluded. The corrected zero-action pilot
  `..._m24_e002_contactpilot_20260908_135509` passed; main bank is authoritative.
- `manifests/contact_audit_summary.json` says `a0c_status=not_evaluated` and
  `training_eligible=false`. Latest actual A0c is still axis-fix pipeline
  `..._m24_e002_a0c_axisfix_s42_20260908_113745`, status
  `failed:no_pair_positive_drive`. No sign/action/checkpoint is selected;
  smoke, five-minute PPO, 30-minute training and E002-B/C remain blocked.

### Single recommended next step
- Validate a per-finger total contact-wrench source in a bounded one-environment
  cloud instrumentation test, including tangential forces/application points
  and separation from joint reaction. Only then make one evidenced E002 source
  fix and rerun A0c with unchanged thresholds. If the installed GPU API needs
  a major refactor, record evidence and the smallest alternative in
  `codeagent_issue.md` rather than silently changing the gate.

## 2026-09-08 -- Candidate-743 A0c Diagnostics And One Axis Fix Completed

### Target milestone/subgoal
- Implement and run diagnostic-only A0c from fixed candidate-743, identify
  one demonstrated problem, fix only that problem and rerun with unchanged
  reward weights and acceptance thresholds. No PPO unless A0c explicitly passes.
- User added the persistent convention: **all training runs on the cloud**,
  including training smokes/capacity probes and future authorized students.
  Local/cloud `AGENTS.md` and `PLANS_m24_ppo.md` now record this rule.

### What changed (files + behavior impact)
- Added `scripts/diagnose_m24_e002_a0c.py`, offline helper
  `scripts/m24_e002_a0c_diagnostics.py`, synthetic regression check
  `scripts/test_m24_e002_a0c_diagnostics.py`, and a no-training cloud wrapper
  `scripts/cloud_run_m24_e002_a0c_diagnostic.sh`.
- JSONs save every candidate/action and every hold/settle/probe policy-step
  force vector, distance, both signed torques, contact, pair gate/balance,
  angle, individual condition flags, counts and correlated top-k rows.
  Actual joint positions/targets, all-body contact forces, reset audits and
  tangent Jacobians are included. No A0b pose search was repeated.
- The only behavioral fix: E002-only `rotation_axis: '-z'` now matches the
  M24 `nut_joint` URDF axis `[0,0,-1]`; previously the trace recorded +Z while
  progress used positive nut DOF displacement. Force sign remains uncalibrated.
  No reward weight, pair threshold, action amplitude, pose or physics parameter
  changed. Historical M24/E001 configuration remains intact.
- Detailed result: `docs/m24_e002_a0c_diagnostic_2026-09-08.md`; ledger updated.

### What was verified (commands + key outcomes)
- AutoDL API reports running; snapshot revealed new SSH port **15233** on
  `connect.westd.seetacloud.com`. Old 39088 is stale. Read live cloud handoff
  and prior pipeline manifest/command/log before syncing; cloud source was
  clean at `224bccd`, and runtime task/calibrator/E002 YAML hashes matched local.
- Baseline pipeline (cloud and full local copy):
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_diag_s42_20260908_113348`.
  Axis-fix rerun:
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a0c_axisfix_s42_20260908_113745`.
  Each used its own timeout 300 in tmux and exited 4 in ten seconds;
  both JSONs are `failed:no_pair_positive_drive`, phases `failed:a0c_gate`.
- Each run: 67 candidates (zero + 48 low-amplitude signed one-joint + 18
  coupled tangent proposals), amplitudes 0.025/0.05/0.10, 2144 finite samples,
  zero unexpected resets, 67/67 stable holds, 637/670 pair-contact probe steps.
  Both signs have zero simultaneous torque, balance-weighted pair-gate and
  full-conjunction passes, hence maximum accepted streak 0/5.
- Corrected maximum net angle `8.28395855e-7 rad` is far below `pi/180`.
  Top net-angle candidate 64 has contact 6/10 and pair gate 0; it is not a
  near-pass. No sign/action/checkpoint was selected, no smoke/PPO/video exists.
- Local/cloud deterministic A0 and diagnostic synthetic checks passed;
  Python compilation, shell syntax and `git diff --check` passed. Source
  snapshots/hashes, exact commands, resolved configs, timestamps and exit
  status are preserved under each pipeline.
- Final cloud check: no tmux or Python workload, RTX 4090 idle at
  `1 MiB / 49140 MiB`, 0%. No local simulator or training job was launched.

### Remaining blocked/risky
- Axis consistency is repaired, but useful pair drive remains unproven.
  Corrected probe torques are opposite-sign on 660/670 steps and one torque
  is zero on the other 10; flipping force_sign cannot resolve this.
- All other bodies have zero net contact force in this bounded trace and the
  sum-of-contact-forces residual is <=1.2e-7 N. This supports nut contact here
  but cannot establish the true per-finger contact-point torque from
  `link origin x resultant force`; no contact application points were measured.
- Both failed JSONs continue to block smoke, five-minute PPO, 30-minute
  training and E002-B/C. Do not retry the same action bank or lower the gate.

### Single recommended next step
- Run a cloud one-environment candidate-743 contact-point/normal/wrench audit
  to establish whether the link-origin resultant-force torque proxy represents
  actual nut-drive contribution. Use its evidence before any further single
  fix; preserve the current thresholds and require a new explicit A0c pass.

## 2026-09-08 -- E002-A0c post-run audit: add diagnostics before another run

### Current cloud state

- Instance is running, but the RTX 4090 was idle at audit time (`1 MiB / 49140
  MiB`, `0%`); no tmux server and no E002/train process were active.
- Cloud branch `feat/m24-humanlike-twist` is clean at `e7ce8b3`.
- Terminal pipeline remains
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1b_s42_20260908_103643`:
  deterministic A0 passed, A0 calibration exited `4`, and terminal phase is
  `failed:a0_calibration`. No smoke, PPO, checkpoint, or video exists.

### Result and unresolved problem

- A0b found stable fixed reset candidate 743. A0c found no candidate with the
  required five-step pair-positive-drive streak (`best_probe_streak=0`) among
  1024 candidates. The reported maximum directed displacement is
  `0.0551694781 rad`; the reported maximum mean pair gate is `0.0539684556`.
- Those two values are independent maxima and may be from different
  candidates/signs. The failed JSON contains no candidate actions, correlated
  top-k rows, per-step raw force/distance/torque/gate trace, or per-condition
  pass counts. It proves only that the full A0c conjunction failed; it cannot
  identify which term failed.
- The current eight-DOF constant action is accumulated into joint targets over
  ten steps, so wrong tangential basis/contact loss is plausible. The
  resultant fingertip-force proxy may also contain non-nut collisions. These
  are hypotheses, not yet measured conclusions.

### Single recommended next cloud action

- Do not rerun the same search, lower thresholds, change PPO reward weights, or
  launch PPO. First patch A0c to run diagnostic-only from candidate 743 and
  persist every candidate action with per-step index/thumb force, distance,
  signed torque under both signs, contact, pair gate, directed angle, and each
  acceptance-condition flag. Report condition counts and correlated top-k
  candidates. Use structured low-amplitude one-joint and coupled tangential
  candidates first. After that evidence identifies one failing condition,
  change only the demonstrated action family/contact source and repeat A0c.

---

## 2026-08-09 -- Local R3/S5 Isaac Gym viewer utility

### Current Cloud State

- No cloud job, GPU workload, artifact mutation, export, or hardware action was performed for this
  utility.  The simulation-only delivery state below remains authoritative.
- The browser-video helper was deliberately removed.  Local source uses the existing
  `scripts/vis_pasini_m24_latest_ppo.sh` with a fixed R3 `RUN_ID`, plus the new
  `scripts/vis_pasini_m24_best_student.sh`, which launches only a headed local Docker/Isaac Gym
  viewer for verified S5 `best_deploy`.  Neither path trains, exports, publishes, or contacts
  hardware.

### Recommended Next Cloud Action

- **None.** Delivery remains complete; keep the versioned simulation-only bundle immutable.  Local
  visual review should use the headed Isaac Gym teacher/student viewer commands from the workstation.

## 2026-08-09 -- S5 simulation-only candidate is exported and published

### Current Cloud State

- No M24 tmux session or GPU workload is active.  The only student run remains
  terminal; its S5 completed checkpoint was explicitly selected in a separate,
  simulation-only review.  S6 remains `S6_interrupted` evidence only and was
  not used.
- Source and published candidate:
  `outputs/XHandPasiniM24NutBolt_student_padapt/m24_padapt_s42_20260808_235300_r3_formal/stage2_nn/S5.ckpt`
  and sibling `best_deploy.ckpt` both SHA-256
  `442f673b62efcce470788201269212c74065479a9e70820b88cfc0943bad9334`.
  `best_deploy_manifest.json` records the copy provenance.
- Export evidence is in
  `outputs/cloud_m24_student_exports/m24_padapt_s42_20260808_235300_r3/S5/`:
  `export_manifest.json` is `ok`; `student_policy.pt` SHA-256 is
  `dca0d430a423b78259d54d599a8ef7e242b8e9699de5851a660bea8854561d56`;
  `torchscript_parity.json` records finite `[1,16]` live JIT execution and
  zero maximum native/JIT output differences (limit `1e-5`).
- The original deploy review with `max_work=2.50` correctly refused the
  canonical S5 `work_mean=2.982767` and remains preserved at
  `outputs/cloud_m24_student_deploy_reviews/m24_padapt_s42_20260808_235300_r3/S5/`.
  Explicit review authorized one fresh retry changing only `max_work` to 3.10;
  it passed at
  `outputs/cloud_m24_student_deploy_reviews/m24_padapt_s42_20260808_235300_r3/S5_maxwork3p10/`.
  Its `student_deploy_selection.json`/TSV are `approved`, require video, and
  use thresholds rotation >=10, positive velocity >=0.90, reset <=0.01,
  action <=0.85, torque <=1.50, work <=3.10.  S5 measured
  13.13880/.93038/.004609/.805675/1.467491/2.982767 respectively.
- `manual_review_audit.md` in that fresh review records p95 action/torque/work
  of 1.55649/2.32831/8.49325 and the camera limitation: the 256 px video
  demonstrates continuity but cannot independently prove physical direction.
  This is simulated nut-DOF evidence only.  No hardware command was run or
  authorized, and `best_deploy.ckpt` does not authorize a real-robot action.
- A new immutable delivery copy is at
  `outputs/cloud_m24_student_deliveries/m24_padapt_s42_20260808_235300_r3_S5_simulation_only_20260809_v1/`,
  with sibling 13 MB tarball SHA-256
  `43c54f70e61dcf1b547a9c6e6976900b572dbe09854f049fc6b18da9e1dc0644`.
  It contains S0--S5 checkpoints plus full evaluation manifest/episodes/MP4
  evidence, an explicitly non-deployable S6 evidence directory, S5 export and
  live-parity proof, approval/review/audit/teacher records, formal provenance,
  `delivery_manifest.json`, and `SHA256SUMS`.  The copied-artifact check was
  80 OK / 0 failed; tar listing checks passed with no GPU compute process.

### Recommended Next Cloud Action

- **Delivery is complete.** Keep all originals and the SHA-256-verified,
  versioned simulation-only delivery bundle immutable.  Do not launch
  training, export again, or contact hardware without separate authorization.
  Any real-robot integration requires an independent interface/safety review.

## 2026-08-09 -- M24 student terminal partial-final evidence is ready for review

### Current Cloud State

- There is no active M24 tmux or GPU process.  The sole student pipeline
  `outputs/cloud_pipeline_m24_student/m24_padapt_s42_20260808_235300_r3/`
  is terminal at `completed_with_failures_or_incomplete_stages`, because its
  unchanged one-hour formal timeout arrived before smoke-derived
  `U=53,480` updates—not because of a runtime fault.
- Formal recorded exit `124` and
  `interrupted:expected_timeout_incomplete_stages`.  S0--S5 are completed
  stages; S6 is explicitly `S6_interrupted` at `45,932/53,480` updates with
  reason `SIGTERM(15)` and remains non-deployable teaching evidence.
- All S0--S5 plus `S6_interrupted` completed the required common 16-env,
  800-step, masked fixed-camera evaluator with an 800-frame non-empty MP4 and
  zero nonfinite state/action values.  The authoritative pipeline summary,
  manifests, videos, and episodes remain under the terminal pipeline path.
- The valid, read-only partial-final audit is
  `outputs/cloud_pipeline_m24_post_student_supervisor/m24_padapt_s42_20260808_235300_r3_partial_final_normalized_exit_status/`.
  Its `decision.json` is `review_required_partial_final`, `valid=true`, with
  simulation candidate allowlist S0--S5 and evidence-only `S6_interrupted`.
  It approves no student, publishes no checkpoint, attempts no export, and
  authorizes no hardware.  An earlier failed audit directory remains preserved
  solely as evidence of the repaired string-vs-int exit-status comparison.
- Source commit `6ae8a0a` adds the narrow audit status normalization and a
  regression test; cloud `dexscrew-ig` `py_compile` and synthetic test passed
  with GPU visibility disabled.  No evaluator/training asset was modified.

### Recommended Next Cloud Action

- Perform the explicit simulation-only semantic review of the audit table,
  fixed videos, signed rotation/episode distributions, reset and risk metrics.
  Choose or reject exactly one completed S0--S5 checkpoint only in a separate
  gate.  Do not use S6_interrupted, export anything, or authorize hardware
  without a later explicit decision.

---

## 2026-08-09 -- M24 PAdapt formal run active; S0--S4 preserved

### Current Cloud State

- The one authorized, simulation-only PAdapt student pipeline remains active in
  tmux `m24_student_s42_20260808_235300` at
  `outputs/cloud_pipeline_m24_student/m24_padapt_s42_20260808_235300_r3/`.
  Its only formal training child is the bounded `timeout 3600` run from the
  approved R3 teacher; no second student run has been started.
- The five-minute smoke completed its expected timeout and measured
  `14.85546875` updates/s, yielding the fixed formal target `U=53,480`.
  The formal run is stable at about 13 updates/s, so its configured one-hour
  timeout may arrive before the 100% S6 target.  Do not extend the timeout or
  revise `U` after the fact.
- Atomic stage checkpoints/manifests S0--S4 are complete and trace to the
  approved R3 checkpoint and formal config hash
  `b049e239516688be443e764208a00fc4d079f0bab18526599f009fb63d1539aa`.
  S4 is `16,044/53,480` updates and `1,026,816` agent steps, SHA-256
  `95dac6776bf2397ca5284dca99553e9a980a1def6809c6387bacaacea153bd59`.
  Runtime monitoring has found no OOM, non-finite, PhysX, traceback, or core
  dump marker.
- The cloud source also contains the read-only post-student evidence auditor
  and its synthetic test.  The auditor has passed `py_compile` and the
  simulator-free test with GPU visibility disabled, but it has **not** been
  invoked against this live pipeline.  Its strict default still requires full
  terminal evidence; any partial-final review must be explicitly opted in only
  after the recorded normal timeout and must exclude `S6_interrupted` from
  candidate selection.
- No export, deployment, hardware command, or real-robot action is authorized
  by this run or by its post-run auditor.

### Recommended Next Cloud Action

- Let the existing timeout-bounded formal run finish unchanged.  Preserve and
  fixed-evaluate every complete stage; if the timeout interrupts S6, retain it
  only as clearly labelled, non-deployable teaching evidence while preserving
  S0--S5.  Do not invoke the real auditor until the pipeline has written its
  terminal phase, and do not select/export/deploy a student without a separate
  simulation-only review.

---

## 2026-08-08 -- R3 approved for one simulation-only PAdapt student run

### Current Cloud State

- The reward-teaching PPO pipeline is terminal at
  `done:teacher_review_required`; its separate unarmed supervisor is terminal
  at `teacher_review_required`.  Their original evidence artifacts remain
  unchanged under
  `outputs/cloud_pipeline_m24_reward_ppo/m24_reward_teaching_s42_20260808_220000/`
  and
  `outputs/cloud_pipeline_m24_post_teacher_supervisor/m24_reward_teaching_s42_20260808_220000/`.
- An explicit simulation-only review approved R3 through
  `scripts/select_m24_teacher.py --require-video --approve-checkpoint`, not by
  training reward.  The approved aggregate record is
  `outputs/cloud_teacher_approvals/m24_reward_teaching_s42_20260808_220000_R3_simulation_approval/teacher_selection.json`;
  the selected checkpoint is
  `outputs/XHandPasiniM24NutBolt_teacher/m24_reward_teaching_s42_20260808_220000_R3_formal/stage1_nn/best_reward_4189.31.pth`,
  SHA-256
  `fedd7e330734ac592a6ed3155442538c9a80fc4bd9579745db2819259bcd25a5`.
  `approval_audit.md` alongside the record preserves the protocol, metric, and
  simulation-only caveats.
- Fresh preflight passed immediately before launch: clean source, idle RTX
  4090, 55 GB free disk, `dexscrew-ig` Python/Isaac Gym/CUDA/imageio-ffmpeg,
  and the tracked EGL Vulkan ICD all succeeded.  The student launcher's own
  required teacher/video dry run also completed with exit status 0.
- Active tmux session: `m24_student_s42_20260808_235300`.
  Active student pipeline:
  `outputs/cloud_pipeline_m24_student/m24_padapt_s42_20260808_235300_r3/`.
  It is the sole PAdapt run, with `EVAL_MODE=required`, seed 42, 64 envs/actors,
  proprio `32x30`, a 5-minute smoke followed only on its measured-throughput
  gate by the one-hour formal run with S0--S6 fixed-video evidence.
- This approval and pipeline remain simulation-only.  They do not authorize
  export, deployment, hardware commands, or real-robot use.  Fixed 256 px
  video supports review but does not independently prove physical rotation
  direction; the task reports simulated nut-DOF evidence only.

### Recommended Next Cloud Action

- Monitor the active smoke log and pipeline status.  If the finite-throughput
  gate passes, allow this one formal run to continue and inspect every saved
  stage's fixed evaluation before any separate simulation-only deployment
  decision.  Do not start another student run or any hardware action.

---

## 2026-08-08 -- M24 teacher review supervisor active (unarmed)

### Current Cloud State

- The reward-teaching teacher tmux session remains
  `m24_teacher_s42_20260808_220000`; at this handoff update it is in the
  serial R3 formal PPO phase.  Its pipeline remains
  `outputs/cloud_pipeline_m24_reward_ppo/m24_reward_teaching_s42_20260808_220000/`.
- A separate, read-only tmux session
  `m24_post_teacher_s42_20260808_220000` runs
  `scripts/cloud_m24_post_teacher_supervisor.py` with a 30-second poll and a
  900-second teacher-stall threshold.  Its durable output is
  `outputs/cloud_pipeline_m24_post_teacher_supervisor/m24_reward_teaching_s42_20260808_220000/`.
- The supervisor source is commit `39167f7`.  It resumes safely after a
  restart and waits only for the teacher pipeline's exact
  `done:teacher_review_required` terminal phase.  It verifies protocol,
  checkpoint SHA, fixed-evaluator manifests, non-empty recorded video frames,
  and zero nonfinite state/action values before producing a review table.
- This supervisor is deliberately **unarmed**: it never approves a teacher,
  starts a student tmux session, exports a student, signals foreign processes,
  or authorizes hardware.  Semantic video and action-risk review remain an
  explicit human gate.

### Recommended Next Cloud Action

- Let the active R3 formal run finish its bounded timeout and fixed evaluation.
  Once the teacher pipeline reaches `done:teacher_review_required`, inspect
  the supervisor's evidence table plus representative fixed-camera videos and
  action-risk metrics, then make a separate explicit teacher-selection
  decision.  Do not launch a student before that decision.

---

## 2026-08-08 -- M24 reward-teaching PPO pipeline active

### Current Cloud State

- The tracked AutoDL EGL ICD at
  `configs/runtime/autodl_headless_vulkan_icd.json` is now validated.  The
  R2 1-env/2-step required-video check completed with `status=ok`, two fixed
  camera frames, and a non-empty MP4.  `imageio-ffmpeg` is installed only in
  the `dexscrew-ig` Conda environment for this video path.
- Active tmux session: `m24_teacher_s42_20260808_220000`.
- Active teacher pipeline:
  `outputs/cloud_pipeline_m24_reward_ppo/m24_reward_teaching_s42_20260808_220000/`.
  Its tmux stdout/stderr is also retained at the sibling
  `m24_reward_teaching_s42_20260808_220000.tmux_launch.log`.
- The pipeline started at `2026-08-08T21:45:31+08:00` from source commit
  `dc30cc55513bc6d8318f690cdf77c353f80f9f6c`, with `EVAL_MODE=required`.
  It first began the required R2 unmasked OOD video evaluation at
  `2026-08-08T21:45:33+08:00`; the common masked R2 evaluation and serial
  R0/R1/R3 probe/formal phases follow only if this evidence gate passes.
- The pipeline is the only cloud GPU process (initially about 3.1 GiB on the
  48 GiB RTX 4090).  No M24 student job is running or authorized yet.

### Recommended Next Cloud Action

- Monitor this tmux session and the pipeline `phase.txt`/status manifests;
  do not start a student until R0--R3 complete and a teacher is explicitly
  selected by the fixed physical/video gate.

---

## 2026-08-08 -- M24 execution layer ready; do not launch formal runs until cloud Vulkan is repaired

### Current Cloud State

- The unrelated `SmolVLA_Dexhand_tactile` job has finished; the RTX 4090 is
  idle and there is no M24 tmux session.
- Native `dexscrew-ig` remains usable for numerical Isaac Gym work: Python
  3.8.20, PyTorch 2.3.1+cu121, Preview 4 `gym_38.so`, and the verified M24 R2
  checkpoint are present.  Import order remains mandatory: import `isaacgym`
  before `torch`.
- A 1-env/2-step M24 PPO fixed-evaluator run without video completed with
  `status=ok`, including checkpoint/schema and physical metric manifests.
- The synced evaluator's required-video dry run was also exercised against the
  verified R2 checkpoint at `outputs/cloud_preflight_m24_vulkan_20260808`:
  exit status `1`, persisted `status=failed` Vulkan manifest, and zero new
  `core` files.  This confirms the gate fails safely before camera creation;
  it did not launch training or consume a rollout budget.
- New local execution sources are ready to sync/commit: native R0/R1/R3 PPO
  runner; required R2 unmasked/common evaluator gate; M24 PAdapt smoke/formal
  runner; S0--S6 callback; teacher/student non-reward gates; and exporter.
  No R0/R1/R3 or M24 student training has been started.

### Blocking Video Gate

- `vulkaninfo --summary` fails before instance creation with
  `ERROR_INCOMPATIBLE_DRIVER`.  Isaac Gym camera-sensor creation then native
  core-dumps, including the upstream multi-camera example.
- The running kernel/userspace NVIDIA version is 580.105.08.  The cloud image
  has an NVIDIA ICD JSON referencing `libGLX_nvidia.so.0`, but the ICD lookup
  does not expose `vkCreateInstance`.  Apt offers only incompatible 580.173.02
  NVIDIA packages; do **not** install or mix them into this running image.
- `libvulkan1` and `vulkan-tools` were installed only to diagnose/preflight;
  no NVIDIA package or driver was changed.  The evaluator now records a clear
  failed/video-unavailable manifest before it constructs a camera, avoiding
  coredumps.

### Required Provider Action

- Repair or replace the cloud image so a matching NVIDIA 580.105.08 Vulkan
  ICD/userspace stack is exposed to this instance, then verify
  `vulkaninfo --summary` succeeds.  A provider reboot/image repair is safer
  than installing mismatched NVIDIA packages in the guest.

### Recommended Next Cloud Action

- After Vulkan verification succeeds, sync the committed M24 execution layer,
  run the required R2 unmasked fixed-camera video first, then use
  `scripts/cloud_pasini_m24_reward_ppo.sh` for the serial R0/R1/R3 pipeline.

---

## 2026-08-07 -- SeeTaCloud M24 Student Runtime Ready; GPU Reserved By Another Project

### Current Cloud State

- Instance: `pro-786095bb6e03`, RTX 4090 (48 GB).
- An unrelated user project, `SmolVLA_Dexhand_tactile`, is actively using nearly
  all GPU memory.  It is outside this M24 scope and must not be stopped,
  modified, or shared with an M24 job.  There is no M24 training or M24 tmux
  session active.
- Fresh source checkout at `/root/code/dexscrew-repro`, commit
  `bc282b289c3b93155d6f09d1e42bf13c9ee6b44f` (`diffusion`).  GitHub access on
  this instance was unusually slow, so the identical local Git bundle was
  transferred and SHA-checked instead.
- Isaac Gym Preview 4 Python runtime was copied to
  `/root/code/isaacgym/python`; its Python 3.8 binding `gym_38.so` is present.
- A dedicated `dexscrew-ig` Conda environment was created with Python 3.8.20.
- The environment contains PyTorch 2.3.1+cu121, the repository requirements,
  and the editable Isaac Gym package.  `torch.cuda.is_available()`,
  `isaacgym`, and `gymtorch` imports all pass; `pip check` reports no broken
  requirements.  Persistent Conda variables point `PYTHONPATH` at
  `/root/code/isaacgym/python`; `tmux` is installed.
- M24 teacher artifact was copied and verified:
  `outputs/XHandPasiniM24NutBolt_teacher/m24_usersaved_contactpose_s42_20260805_153516_ppo30m/stage1_nn/best_reward_3656.81.pth`
  SHA-256 `874d2b5b330b8a5d244a4e2d922831c0e9841a9a83cf3d06f0729f0f21ca30eb`.
- Direct cloud package downloads were measured at only about 0.7 MB/s, so the
  required Git bundle and CUDA wheels were transferred from the local
  workstation with SHA verification.  This completed successfully; slow cloud
  download is no longer blocking this configured runtime.
- The M24 headless scene smoke completed successfully with four environments:
  `scripts/debug_pasini_env.py task=XHandPasiniM24NutBolt ... +steps=96
  +random_actions=False`.  It reached `[DEBUG] Done.` with exit status 0 and
  no `Traceback`, non-finite, or anomaly findings.  Logs are under
  `outputs/cloud_setup_m24/logs/`.
- No M24 student training has been launched.

### Important M24 Boundary

- Do not use `scripts/nutbolt_student_padapt.sh`; it targets the legacy
  `XHandHoraNutBolt` task.  The future M24 launch must explicitly use
  `task=XHandPasiniM24NutBolt`, `train.algo=ProprioAdapt`, and the verified
  teacher path above.
- The M24 plan still requires an unmasked teacher replay before actual student
  distillation.  This session configures only the cloud runtime.

### Recommended Next Cloud Action

- Leave the unrelated GPU job untouched.  After it releases the GPU, first
  perform the required unmasked headed M24 teacher replay and create/verify
  the M24-specific student launcher and checkpoint protocol; only then launch
  an explicitly time-limited student run from the verified teacher.

---

## 2026-05-18 -- Middle1 PPO3799 Students Synced To Ubuntu Local

### Current Cloud State
- GPU remains idle.
- The completed Middle1 PPO3799 PAdapt/DOTPG cloud outputs have been synced to Ubuntu local.

### Synced Outputs
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`
- Pipeline logs/status:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`

### Recommended Next Action
- Headed-visualize Middle1 PPO/PAdapt/DOTPG locally and decide which policy is worth deploying or further distilling.

---

## 2026-05-18 -- Middle1 PPO3799 PAdapt/DOTPG 1h Parallel Complete

### Current Cloud State
- GPU is idle:
  `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
- The tmux session `middle1_students_ppo3799_20260518_160530` has exited.
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`
- `status/phase.txt=done`

### Teacher Checkpoint
- Both students used the latest Middle1 NoInitNoise cloud PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`

### Results
- PAdapt:
  - `padapt_exit_status=124`
  - `padapt_timeout_status=expected_1h_timeout`
  - max parsed `Current Best: 3236.33`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/stage2_nn/model_best.ckpt`
- DOTPG:
  - `dotpg_exit_status=124`
  - `dotpg_timeout_status=expected_1h_timeout`
  - max parsed `Current Best: 2901.99`
  - checkpoint:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Recommended Next Cloud Action
- Sync both student output dirs and this pipeline directory to Ubuntu local, then headed-visualize the two students on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`.

---

## 2026-05-18 -- Middle1 PPO3799 PAdapt/DOTPG 1h Parallel Active

### Current Cloud State
- Active tmux session:
  `middle1_students_ppo3799_20260518_160530`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/`
- `status/phase.txt`: `running_parallel`
- Cloud GPU startup check after launch:
  `NVIDIA GeForce RTX 4090 D, 9162 MiB / 24564 MiB, 95%`
- Active jobs:
  - PAdapt 1h student distillation, `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`
  - DOTPG 1h student distillation, `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`

### Teacher Checkpoint
- Both students use the latest Middle1 NoInitNoise cloud PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`

### Output Targets
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_padapt1h_from_ppo3799/`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530_dotpg1h_from_ppo3799_dual_bc5/`

### Logs And Status
- PAdapt log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/padapt.log`
- DOTPG log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/dotpg.log`
- GPU log:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_160530/logs/gpu_usage.csv`

### Recommended Next Cloud Action
- Let both jobs reach their expected `timeout 3600` wall-clock completion (`exit_status=124`), then sync the two student output dirs and pipeline status back to Ubuntu local for headed visualization.

---

## 2026-05-18 -- Cloud Status Check: No Active Cloud Training

### Current Cloud State
- GPU is idle:
  `NVIDIA GeForce RTX 4090 D, 1 MiB / 24564 MiB, 0%`.
- No active `train.py` / `timeout 3600` cloud training processes were found.

### Middle1 Status
- `Middle1 KeyboardLatest NoInitNoise PPO1h` completed:
  - pipeline:
    `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/`
  - `status/phase.txt=done`
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - final teacher:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_3799.45.pth`
- The planned follow-up script path exists locally in prior notes, but this cloud path did not contain a launched student-after-PPO status directory:
  `outputs/cloud_pipeline_codrive_middle1_students_after_noinitnoise1h/middle1_keyboardlatest_noinitnoise_students_s42_20260518_115617/`.
- Older Middle1 students from the earlier `middle1_ppo1h_s42_20260515_120420` teacher are complete:
  - DOTPG:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`
  - PAdapt exists locally from the paired local run:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_middle1/middle1_padapt_s42_20260515_131253_from_ppo1h/stage2_nn/model_best.ckpt`

### Small Status
- `CoDriveSmall` PPO and students were run locally on Ubuntu, not on cloud:
  - PPO final best:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_small/codrive_small_keyboardlatest_s42_20260518_120558_ppo1h/stage1_nn/best_reward_3985.40.pth`
  - PAdapt 1h:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_padapt1h/stage2_nn/model_best.ckpt`
  - DOTPG 1h:
    `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_small/codrive_small_students_from_ppo3985_s42_20260518_131156_dotpg1h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Recommended Next Cloud Action
- If continuing Middle1, sync the final cloud PPO output back to local before visualization:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`.
- If continuing Small, visualize the local PAdapt/DOTPG checkpoints first; cloud is currently free for a longer follow-up run.

---

## 2026-05-18 -- Middle1 KeyboardLatest NoInitNoise PPO1h Active

### Current Cloud State
- Active tmux session:
  `middle1_keyboardlatest_noinitnoise_ppo1h_20260518_111917`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`
- Confirmed cloud task YAML values:
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosZScaleComp: 0.0`
  - `asset.handRootPos: [0.092000, 0.020000, 0.245000]`
  - `right_index_joint_3: 0.4492996037`
  - `right_thumb_joint_2: 0.2999999821`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
- `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_keyboardlatest_noinitnoise_s42_20260518_111917/run_middle1_keyboardlatest_noinitnoise_ppo1h_cloud.sh`

### Verification
- The previous `middle1_keyboardlatest_s42_20260518_111658` cloud run was stopped/restarted because init-pose noise was not zero.
- Cloud startup verified:
  - Python active under `timeout 3600`.
  - GPU around `14313 MiB / 24564 MiB`.
  - First best checkpoint appeared:
    `best_reward_61.34.pth` at about 2.6 minutes elapsed.
- Mid-run sync for local visualization:
  - Cloud was still active: `phase=ppo`.
  - GPU around `14313 MiB / 24564 MiB`.
  - Synced current output to Ubuntu local.
  - Latest synced checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/stage1_nn/best_reward_2807.89.pth`.

### Recommended Next Cloud Action
- Let this no-noise Middle1 PPO run to expected 1h timeout (`ppo_exit_status=124`), then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_keyboardlatest_noinitnoise_s42_20260518_111917_ppo1h/`
- Visualize the final `stage1_nn/best_reward_*.pth`.

---

## 2026-05-18 -- Pure CoDrive Reprobe PPO1h Active

### Current Cloud State
- Active tmux session:
  `codrive_reprobe_ppo1h_20260518_105210`
- Pipeline:
  `outputs/cloud_pipeline_codrive_reprobe_ppo1h/codrive_reprobe_s42_20260518_105210/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDrive`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
- `outputs/cloud_pipeline_codrive_reprobe_ppo1h/codrive_reprobe_s42_20260518_105210/run_codrive_reprobe_ppo1h_cloud.sh`

### Verification
- Previous `Index112 NoInitNoise` cloud PPO was manually stopped by user request.
  - Stop time: `2026-05-18T02:51:59+00:00`.
  - Last observed best before stop:
    `best_reward_3360.56.pth`.
  - `status/phase.txt`: `stopped_manual`.
- Pure CoDrive YAML was compared against the deployment copy under local `configs/codrive/`.
  - Task YAML SHA256 matched exactly.
  - Train YAML SHA256 matched exactly.
- Pure CoDrive cloud startup verified:
  - Python active under `timeout 3600`.
  - GPU around `14329 MiB / 24564 MiB`.
  - First best checkpoint appeared:
    `best_reward_65.72.pth` at about 3 minutes elapsed.

### Recommended Next Cloud Action
- Let the pure CoDrive reprobe run to the expected 1h timeout (`ppo_exit_status=124`), then sync:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive/codrive_reprobe_s42_20260518_105210_ppo1h/`
- Compare the final best against the historical deploy teacher:
  `configs/codrive/best_reward_4159.37.pth`.

---

## 2026-05-18 -- CoDriveMiddle1 Index112 NoInitNoise PPO1h Active

### Current Cloud State
- Active tmux session:
  `middle1_index112_noinitnoise_ppo1h_20260518_101206`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.num_actors=12288`
  - `train.ppo.minibatch_size=24576`
  - `num_threads=22`
- Isolation variables relative to current Middle1:
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
  - `right_index_joint_3: 1.1200000000`
  - `asset.handRootPosZScaleComp` remains `0.0`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1Index112NoInitNoise.yaml`
- `outputs/cloud_pipeline_codrive_middle1_index112_noinitnoise_ppo1h/middle1_index112_noinitnoise_s42_20260518_101206/run_middle1_index112_noinitnoise_ppo1h_cloud.sh`

### Verification
- Cloud handoff was read before launch.
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Startup health verified after launch:
  - Python active under `timeout 3600`
  - GPU around `14339 MiB / 24564 MiB`
  - first best checkpoint appeared and continued updating:
    `best_reward_117.90.pth` at about 3.6 minutes elapsed.
- 30min visual-check sync:
  - `phase=ppo`, cloud training still active.
  - GPU around `14339 MiB / 24564 MiB`.
  - best at about 30.5 minutes:
    `best_reward_3237.63.pth`.
  - local sync pulled the slightly newer checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/stage1_nn/best_reward_3255.81.pth`.

### Recommended Next Cloud Action
- Local 30min checkpoint is ready for headed visualization:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1_index112_noinitnoise/middle1_index112_noinitnoise_s42_20260518_101206_ppo1h/`
- If waiting for full completion, expect `ppo_exit_status=124` as the correct 1h timeout result.

---

## 2026-05-15 -- CoDriveMiddle1 Latest Initpose PPO1h Completed And Synced

### Current Cloud State
- Completed tmux session:
  `middle1_latestinit_ppo1h_20260515_150457`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/`
- `status/phase.txt`: `done`
- Completed command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - `train.ppo.num_actors=12288`
  - `num_threads=22`
- Latest init pose synced from Ubuntu:
  - `handRootPos: [0.080000, 0.020000, 0.239000]`
  - `handRootRPY: [3.141500, 0.439627, 3.141500]`
  - index joints: `[0.3499999940, 0.9852794409, 0.2628971040, 0.5692995787]`
  - thumb joints: `[0.1299999952, 1.5700000525, 0.0399999991, 0.5719662905]`
- Completion:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - final best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/stage1_nn/best_reward_3309.63.pth`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `outputs/initpose_tuning/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1_current.yaml`
- `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/run_middle1_latestinit_ppo1h_cloud.sh`
- Cloud pipeline logs/status were synced back to Ubuntu local:
  `outputs/cloud_pipeline_codrive_middle1_ppo1h/middle1_latestinit_s42_20260515_150457/`
- PPO output/checkpoint/events were synced back to Ubuntu local:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_latestinit_s42_20260515_150457_ppo1h/`

### Verification
- Cloud was idle before launch: RTX 4090 D around `1 MiB / 24564 MiB`.
- Launched in tmux and confirmed active Python process under `timeout 3600`.
- GPU after startup around `14273 MiB / 24564 MiB`.
- During training, best reached `3309.63`.
- Final status file confirms expected timeout `124`.
- Cloud after completion is idle: RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.

### Recommended Next Cloud Action
- No active cloud action is required for this run.
- Next useful action is local headed visualization of `best_reward_3309.63.pth`.

---

## 2026-05-15 -- CoDriveMiddle1 DOTPG1h Completed And Synced

### Current Cloud State
- Completed tmux session:
  `middle1_dotpg1h_20260515_131253`
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/`
- `status/phase.txt`: `done`
- Completed command:
  DOTPG student distillation on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1`
- Teacher checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle1/middle1_ppo1h_s42_20260515_120420/stage1_nn/best_reward_3783.10.pth`
- Output name:
  `Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=1024`
  - `train.ppo.minibatch_size=12288`
  - `train.ppo.num_actors=1024`
  - `num_threads=10`
- DOTPG variant:
  - `policy_arch=teacher_actor`
  - `policy_loss_mode=dual`
  - `bc_coef=5.0`
  - `bc_alpha_max=20.0`
- Best student checkpoint:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/student_output/dotpg_nn/model_best.ckpt`
- Completion:
  - `dotpg_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - observed final current best plateau around `2397.86`

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle1.yaml`
- Middle1 PPO teacher checkpoint `best_reward_3783.10.pth`
- `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/run_middle1_dotpg1h_cloud.sh`
- Cloud DOTPG pipeline logs/status were synced back to Ubuntu local:
  `outputs/cloud_pipeline_codrive_middle1_dotpg1h/middle1_dotpg_s42_20260515_131253/`
- Cloud DOTPG output/checkpoint/events were synced back to Ubuntu local:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_middle1/middle1_dotpg_s42_20260515_131253_dual_bc5_from_ppo1h/`

### Verification
- Cloud was idle before launch.
- DOTPG process used GPU during training, around `6.1GB / 24GB`.
- Early training reached current best around `32.18` and saved `student_output/dotpg_nn/model_best.ckpt`.
- Final status file confirms expected timeout `124`.
- Cloud after completion is idle: RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.

### Recommended Next Cloud Action
- No active cloud action is required for this DOTPG run.
- Next useful step is local visualization against the paired Middle1 PAdapt checkpoint.

---

## 2026-05-15 -- CoDriveMiddle2 PPO1h Completed And Synced

### Current Cloud State
- No active Middle2 training process.
- GPU idle after completion:
  RTX 4090 D around `1 MiB / 24564 MiB`, `0%`.
- Pipeline:
  `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/`
- `status/phase.txt`: `done`
- Completed command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2`
- Output name:
  `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h`
- Runtime budget:
  `timeout 3600` wall-clock seconds.
- Cloud resource settings:
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - `train.ppo.num_actors=12288`
  - `num_threads=22`
- GPU during startup:
  RTX 4090 D around `14.3GB / 24GB`, active utilization.

### Synced Files
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveMiddle2.yaml`
- `outputs/cloud_pipeline_codrive_middle2_ppo1h/middle2_s42_20260515_120420/run_middle2_ppo1h_cloud.sh`

### Verification
- Active retry uses the existing cloud conda IsaacGym environment (`dexscrew-ig`), matching earlier successful cloud pipelines.
- PPO progress confirmed after startup:
  - `Agent Steps: 0004M`
  - FPS about `29k`
  - current best reward `230.80`
- Completion confirmed:
  - `ppo_exit_status=124`
  - `timeout_status=expected_1h_timeout`
  - best checkpoint:
    `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_middle2/middle2_s42_20260515_120420_ppo1h/stage1_nn/best_reward_3503.45.pth`
- Results synced back to Ubuntu local:
  - cloud pipeline logs/status
  - PPO output directory and best checkpoint

### Notes
- First launch attempted Docker and failed immediately because this cloud machine does not have image `dexscrew:ig20-py38`.

### Recommended Next Cloud Action
- No immediate cloud action required for Middle2.
- Next useful step is local visualization/comparison against Middle1:
  - Middle1 local best: `best_reward_3783.10.pth`
  - Middle2 cloud-synced best: `best_reward_3503.45.pth`

---

## 2026-05-13 -- RealBulb PPO2h + PAdapt/DOTPG2h Pipeline Active

### Current Cloud State
- Active tmux session:
  `realbulb_ppo2h_students2h_20260513_031105`
- Pipeline:
  `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/realbulb_s42_20260513_031105/`
- Convenience link:
  `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest`
- `status/phase.txt`: `ppo`
- Active command:
  PPO teacher on `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb`
- GPU at startup confirmation:
  RTX 4090 D around `14359 MiB / 24564 MiB`, roughly `80%` utilization.

### Synced Files
- `assets/screw/realbulb/0000_lightbulb.urdf`
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
- `outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/run_realbulb_ppo2h_padapt_dotpg2h.sh`

### Pipeline Plan
- PPO teacher:
  - `timeout 7200`
  - `task.env.numEnvs=12288`
  - `train.ppo.minibatch_size=24576`
  - output:
    `outputs/Dexh13HoraLightbulb_teacher_realbulb/realbulb_s42_20260513_031105_ppo2h/`
- After PPO timeout, the script explicitly selects newest/best:
  `stage1_nn/best_reward_*.pth`
- Student phase starts automatically afterward:
  - PAdapt: `timeout 7200`, `task.env.numEnvs=512`, `train.ppo.minibatch_size=6144`
  - DOTPG: `timeout 7200`, `task.env.numEnvs=1024`, `train.ppo.minibatch_size=12288`
  - PAdapt and DOTPG run in parallel.

### Verification
- Cloud smoke passed before launch:
  - loaded `screw_realbulb`
  - generated initial poses at scales `0.975` and `1.025`
  - exited with `max steps achieved`
- PPO live progress confirmed:
  - `Agent Steps: 0005M ... Current Best: 57.18`

### Recommended Next Cloud Action
- Check:
  `cat outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest/status/phase.txt`
- While phase is `ppo`, monitor:
  `grep -a "Agent Steps" outputs/cloud_pipeline_realbulb_ppo2h_padapt_dotpg2h/latest/logs/ppo.log | tail`
- After PPO reaches expected timeout `124`, confirm:
  - `selected_teacher_ckpt.txt` exists
  - `status/phase.txt` becomes `students_parallel`
  - both PAdapt and DOTPG logs show training progress

---

## 2026-05-12 -- Real-Size CoDrive Bulb Added Local

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local workspace added a real-size bulb asset and a matching CoDrive task variant.

### Local Files
- `assets/screw/realbulb/0000_lightbulb.urdf`
  - visual and collision both use `contact0.stl` / `contact1.stl`.
  - geometry scaled to about `140 mm` length and `60 mm` max diameter.
  - viewer display now matches the actual contact surface.
- `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - `object.type: screw_realbulb`
  - `baseObjScale: 1.00`
  - object scale randomization covers approximately `0.95-1.05`.
- `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveRealBulb.yaml`
  - copied from the original CoDrive train YAML for Hydra compatibility.

### Verification
- XML/YAML parse checks passed.
- Local IsaacGym smoke loaded the new asset and exited with `max steps achieved`.

### Recommended Next Cloud Action
- No cloud action is required yet.
- Before any cloud training, sync the new asset/task/train files and visually confirm the inherited hand init pose still fits the taller real-size bulb.

---

## 2026-05-12 -- Original CoDrive Student Deploy Packages Completed Local

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local deploy packages for original non-noise CoDrive were completed under `sim2real/deploy/codrive`.

### Local Deploy Packages
- Original CoDrive deploy packages now available:
  - `sim2real/deploy/codrive/padapt_deploy`
  - `sim2real/deploy/codrive/dotpg_deployv1`
  - `sim2real/deploy/codrive/diffusion_latent_deploy`
  - `sim2real/deploy/codrive/consistency_latent_deploy`
  - `sim2real/deploy/codrive/flow_matching_deploy`
  - `sim2real/deploy/codrive/diffusion_action_chunk_deploy`
  - `sim2real/deploy/codrive/bc_latentbc_deploy`
  - `sim2real/deploy/codrive/dagger_deploy`
- Each contains task YAML, train YAML, teacher `best_reward_4159.37.pth`, and one student `model_best.ckpt`.
- Teacher/YAML files in every package hash-match the frozen original CoDrive package under `sim2real/codrive`.

### Recommended Next Cloud Action
- No cloud action is required for this local packaging step.
- If evaluating deployment candidates on cloud later, use `sim2real/deploy/codrive` and remember these are the original no-init-noise CoDrive students, not CoDriveNoise.

---

## 2026-05-12 -- Local Deploy Packages Grouped By Task

### Current Cloud/Local State
- No cloud training/eval job was launched.
- Ubuntu local deploy packages were reorganized by task family.

### Local Deploy Layout
- `sim2real/deploy/codrive/`
  - non-thesis CoDrive deploy package, currently `dotpg_deployv1`
- `sim2real/deploy/codrive_thesis/`
  - CoDriveThesis deploy packages: BC/LatentBC, DAgger, DOTPG v2, PAdapt, diffusion latent, consistency, flow matching, action chunk, and PureBC
- `sim2real/deploy/codrive_noise/`
  - CoDriveNoise deploy packages: PAdapt, DOTPG, diffusion latent, consistency, flow matching, action chunk, PureBC, DAgger, and BC/LatentBC
- Verified every nested deploy package still has exactly four files.

### Recommended Next Cloud Action
- No cloud action is required for this local file organization change.
- If using a deploy package from another OS/session, use the task-family folder prefix.

---

## 2026-05-12 -- CoDriveNoise Student Deploy Packages Grouped Local

### Current Cloud/Local State
- Cloud training remains complete and idle; no new training job was launched.
- Ubuntu local workspace synced the cloud CoDriveNoise student outputs for:
  - diffusion latent
  - consistency latent
  - flow matching
  - diffusion action chunk
  - PureBC
  - DAgger
- Existing local/cloud-synced PAdapt, DOTPG, teacher pth, task YAML, and train YAML were verified.

### Local Deploy Packages
- The following local folders now exist under `sim2real/deploy/codrive_noise`, each with exactly four deploy files: task YAML, train YAML, PPO teacher pth, and student `model_best.ckpt`.
  - `codrive_noise_padapt_deploy`
  - `codrive_noise_dotpg_deploy`
  - `codrive_noise_diffusion_latent_deploy`
  - `codrive_noise_consistency_latent_deploy`
  - `codrive_noise_flow_matching_deploy`
  - `codrive_noise_diffusion_action_chunk_deploy`
  - `codrive_noise_purebc_deploy`
  - `codrive_noise_dagger_deploy`
  - `codrive_noise_bc_latentbc_deploy`
- Verified no `sim2real/deploy/codrive_noise_*_deploy` directories remain directly under `sim2real/deploy`.

### Recommended Next Cloud Action
- No cloud action is required for packaging.
- Next useful cloud action is unified fixed-step eval over these CoDriveNoise deploy candidates.

---

## 2026-05-12 -- CoDriveNoise Remaining Baselines 3h Completed

### Current Cloud State
- Completed run tag:
  `codrive_noise_remaining_s42_3h_20260512_013000`
- Main cloud pipeline:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/codrive_noise_remaining_s42_3h_20260512_013000/`
- `status/phase.txt`: `done`
- No active `python train.py` process.
- No active CoDriveNoise tmux sessions; only the old stale paper supervisor session remains.
- Cloud GPU after completion:
  RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Time-Budget Allocation
- User allowed up to 10h total wall-clock budget.
- Chosen schedule keeps the formal remaining-baseline comparison at 3h per method, matching the existing PAdapt/DOTPG 3h CoDriveNoise runs.
- Remaining wall-clock budget is intentionally reserved for failure recovery, sync, visualization, and fixed-step eval rather than blindly extending all baselines and making them unfair against PAdapt/DOTPG.

### Cloud Jobs And Exit Status
- Cloud script:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_diffusion_purebc_3h.sh`
- Ran 5 jobs in parallel with `timeout 10800`, `task.env.numEnvs=192`, `train.ppo.minibatch_size=2304`:
  - `DiffusionLatentStudent`
  - `ConsistencyLatentStudent`
  - `FlowMatchingLatentStudent`
  - `DiffusionActionChunkStudent`
  - `PureBC`
- Separate cloud DAgger script:
  `outputs/cloud_pipeline_codrive_noise_remaining_baselines_3h/run_cloud_dagger_3h.sh`
- Ran DAgger with `timeout 10800`, `task.env.numEnvs=128`, `train.ppo.minibatch_size=1536`, CPU replay buffer.
- Exit statuses were all `124`, expected for requested 3h timeouts:
  - `diffusion_latent_exit_status = 124`
  - `consistency_latent_exit_status = 124`
  - `flow_matching_latent_exit_status = 124`
  - `diffusion_action_chunk_exit_status = 124`
  - `purebc_exit_status = 124`
  - `dagger_exit_status = 124`

### Teacher And Task
- Task:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise`
- Teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`

### Parsed Cloud Rewards
- `DiffusionLatentStudent`: current-best max `4019.67`
- `ConsistencyLatentStudent`: current-best max `3840.09`
- `FlowMatchingLatentStudent`: current-best max `3850.96`
- `DiffusionActionChunkStudent`: current-best max `1341.73`
- `PureBC`: current-best max `3305.94`
- `DAgger`: no `Current Best` line; train return last `1655.33`, best student eval `979.06`
- Tail scans of real training logs found no `Traceback`, CUDA OOM, missing-key/size-mismatch, or segmentation-fault patterns.

### Cloud Checkpoints
- Diffusion latent:
  `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_nn/model_best.ckpt`
- Consistency:
  `outputs/Dexh13HoraLightbulb_student_consistency_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_consistency_nn/model_best.ckpt`
- Flow matching:
  `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_flow_nn/model_best.ckpt`
- Diffusion action chunk:
  `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_diffusion_action_chunk_nn/model_best.ckpt`
- PureBC:
  `outputs/Dexh13HoraLightbulb_student_purebc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000/stage2_bc_nn/model_best.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_dagger_pure_replay/dagger_nn/model_best.ckpt`

### Local Companion Job
- Ubuntu local machine completed `BCStudent`/LatentBC for the same run tag and teacher.
- Local wrapper status file remained stale at `running`, but the Docker/train process is gone and checkpoint files exist.
- Local BC/LatentBC checkpoint:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_noise/codrive_noise_remaining_s42_3h_20260512_013000_latentbc/bc_nn/model_best.ckpt`
- Local BC eval-select history:
  - max `avg_reward = 3.47085`
  - max `score = 2.98257`

### Recommended Next Cloud Action
- Sync the six cloud baseline outputs locally.
- Run a unified fixed-step eval and visualization over CoDriveNoise: PPO teacher, PAdapt, DOTPG, diffusion latent, consistency, flow matching, action chunk, PureBC, DAgger, and BC/LatentBC.

---

## 2026-05-12 -- CoDriveNoise PPO/PAdapt/DOTPG Pipeline Completed

### Current Cloud State
- Pipeline completed:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/`
- Convenience symlink:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest`
- `status/phase.txt`: `done`
- No active `python train.py` process.
- GPU after completion: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.
- The launch tmux session has exited; only the old stale paper-eval tmux session remains.

### Exit Status
- PPO teacher: `status/ppo_exit_status = 124`
- PAdapt student: `status/padapt_exit_status = 124`
- DOTPG student: `status/dotpg_exit_status = 124`
- All three `124` statuses are expected timeout completions for the requested 3h phases.

### Result Checkpoints
- Selected PPO teacher:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_4076.32.pth`
- PAdapt student:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h/stage2_nn/model_best.ckpt`
- DOTPG student:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5/student_output/dotpg_nn/model_best.ckpt`

### Parsed Training Rewards
- PPO actual training lines: max/current best `4076.32`.
- PAdapt actual training lines: max/current best `3501.40`.
- DOTPG actual training lines: max/current best `2631.82`.
- Important log note: these logs include startup dirty-git-diff text. Broad greps may find unrelated historical `Current Best` or error words from embedded docs/diffs. Use only real progress lines matching `Agent Steps: ... Current Best`.
- Tail scans of the final 300 log lines for each phase found no `Traceback`, `RuntimeError`, CUDA OOM, missing-key/size-mismatch, or segmentation-fault patterns.

### Recommended Next Cloud Action
- Sync the three result checkpoints plus the two CoDriveNoise YAML files to local if the user wants visualization or deploy packaging.
- For final paper-style comparison, run the same fixed-step eval protocol as the existing CoDriveThesis table rather than relying only on training reward.

---

## 2026-05-11 -- CoDriveNoise PPO3h + PAdapt/DOTPG3h Pipeline Active

### Current Cloud State
- Active tmux session:
  `codrive_noise_ppo3h_students3h_20260511_174738`
- Active pipeline:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/codrive_noise_s42_20260511_094738/`
- Convenience symlink:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest`
- Current phase file:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/status/phase.txt`
  currently showed `ppo` at launch validation.
- PPO process confirmed active after launch:
  `python train.py task=Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise ...`
- GPU after PPO startup:
  RTX 4090 D, about `14327 MiB / 24564 MiB`, about `67%` utilization.
- Latest startup check observed PPO checkpoint:
  `outputs/Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h/stage1_nn/best_reward_70.57.pth`

### Config/Code State
- Original CoDrive task restored to no init-pose noise:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDrive.yaml`
  - `object.init_pos_noise: [0.0, 0.0, 0.0]`
  - `asset.handRootPosNoise: [0.0, 0.0, 0.0]`
- New noise variant added and synced:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise.yaml`
  - `eval_cache_name: sim2real_twofinger_codrive_noise`
  - `object.init_pos_noise: [0.005, 0.005, 0.0]`
  - `asset.handRootPosNoise: [0.001, 0.001, 0.001]`
- Pipeline script:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/run_codrive_noise_ppo3h_padapt_dotpg3h.sh`

### Pipeline Plan
- PPO teacher:
  - timeout: `10800s` (3h)
  - task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveNoise`
  - output: `Dexh13HoraLightbulb_teacher_sim2real_twofinger_codrive_noise/codrive_noise_s42_20260511_094738_ppo3h`
  - aggressive 24GB GPU setting: `task.env.numEnvs=12288`, `train.ppo.minibatch_size=24576`
- After PPO exits, the script selects the max-reward `stage1_nn/best_reward_*.pth` and writes:
  `selected_teacher_ckpt.txt`
- PAdapt student:
  - timeout: `10800s` (3h)
  - output: `Dexh13HoraLightbulb_student_padapt_codrive_noise/codrive_noise_s42_20260511_094738_padapt3h_from_ppo3h`
- DOTPG student:
  - timeout: `10800s` (3h)
  - output: `Dexh13HoraLightbulb_student_dotpg_codrive_noise/codrive_noise_s42_20260511_094738_dotpg3h_from_ppo3h_dual_bc5`
  - DOTPG recipe: current dual actor loss + teacher actor init + BC anchor `bc_coef=5.0`.
- Expected timeout exit status for completed wall-clock phases: `124`.

### Logs And Status Files
- Pipeline log:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/pipeline.log`
- PPO log:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/ppo.log`
- Student logs:
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/padapt.log`
  `outputs/cloud_pipeline_codrive_noise_ppo3h_padapt_dotpg3h/latest/logs/dotpg.log`
- Exact commands:
  `commands_ppo.txt`, `commands_padapt.txt`, `commands_dotpg.txt`
- Exit statuses:
  `status/ppo_exit_status`, `status/padapt_exit_status`, `status/dotpg_exit_status`

### Recommended Next Cloud Action
- Monitor the active tmux/pipeline until PPO reaches the 3h timeout, then verify:
  - `status/ppo_exit_status` is `124` or clean `0`
  - `selected_teacher_ckpt.txt` exists and points to the intended best PPO checkpoint
  - PAdapt and DOTPG both start and complete their own 3h timeouts
- Do not modify the original CoDrive YAML for this ablation; use the `CoDriveNoise` task name for the init-noise experiment.

---

## 2026-05-11 -- DOTPG Normal-Speed Visualization Support Synced

### Current Cloud State
- No training or eval was launched for this update.
- Purpose: keep future Ubuntu/Windows/cloud visual checks aligned with the new local default that headed visualization should run at normal speed unless fast eval is explicitly requested.

### Synced/Expected Source Behavior
- `AGENTS.md` records that headed IsaacGym visualization intended for human inspection should default to real-time playback.
- `dexscrew/dotpg/dotpg.py` supports:
  - `++train.dotpg.test_realtime=True`
  - `++train.dotpg.test_realtime_factor=1.0`
  - optional fixed override `++train.dotpg.test_sleep_sec=<seconds>`
- For CoDriveThesis, normal speed is about one policy step per `sim.dt * controlFrequencyInv = 0.005 * 10 = 0.05s`.

### Recommended Next Cloud Action
- For DOTPG headed visualization, include:
  `++train.dotpg.test_realtime=True ++train.dotpg.test_realtime_factor=1.0`.
- Keep headless numeric eval fast unless the user asks for wall-clock playback.

---

## 2026-05-06 -- CoDriveThesis Extra Paper Eval Completed

### Current Cloud State
- No active `tmux` sessions after completion.
- No active `python train.py` process after completion.
- GPU after completion: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Eval Run
- Additional eval run:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/`
- Previous clean baseline eval reused for combined tables:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- Protocol:
  - task: `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`
  - seeds: `42,43,44`
  - fixed steps: `2048`
  - num envs: `48`
  - termination/noise aligned with previous paper eval.
- Startup note:
  - An initial tmux launch used `./run_eval.sh` and resolved `RUN_DIR` to the repo root, so it exited before starting real eval.
  - The eval was relaunched with an absolute script path and completed cleanly.

### Validation
- `validation_summary.txt`:
  - `raw_rows=18`
  - `expected_rows=18`
  - `all_ok=True`
  - `combined_raw_rows=30`
  - `combined_expected_rows=30`
  - `combined_all_ok=True`
- Error scan over logs found no:
  `Traceback`, `RuntimeError`, `Error executing job`, `CUDA out of memory`,
  `FileNotFoundError`, `Missing key`, `Unexpected key`, or `size mismatch`.

### Added Methods
| Method | Fixed-Step Reward mean/std | Done Rate mean/std | Status |
|---|---:|---:|---|
| `padapt` | `4.831464 +/- 0.077221` | `0.001061 +/- 0.000025` | OK |
| `diffusion_latent` | `3.504940 +/- 0.152013` | `0.001193 +/- 0.000006` | OK |
| `consistency_latent` | `5.310003 +/- 0.030259` | `0.001038 +/- 0.000011` | OK |
| `flow_matching` | `5.078809 +/- 0.134603` | `0.001078 +/- 0.000041` | OK |
| `diffusion_action_chunk` | `-1.231769 +/- 0.053342` | `0.016602 +/- 0.000000` | OK run, poor result |
| `purebc` | `4.969141 +/- 0.056104` | `0.001065 +/- 0.000038` | OK |

### Combined Table Artifacts
- Additional raw:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_raw.csv`
- Additional aggregate:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_aggregate.csv`
- Combined raw:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_combined_raw.csv`
- Combined aggregate:
  `outputs/paper_eval_codrive_thesis_extra_s42_s43_s44_20260506_161503/fixed_eval_combined_aggregate.csv`

### Local Conclusion
- The uploaded current CoDriveThesis PAdapt/diffusion/PureBC checkpoints are evaluable under the same paper protocol.
- `consistency_latent` is the strongest student in this fixed-step table and is close to teacher PPO.
- `flow_matching` and `purebc` are also strong.
- `diffusion_latent` is valid but below DOTPG/PAdapt/PureBC/DAgger in this metric.
- `diffusion_action_chunk` is a valid run but fails behaviorally under this protocol and should not be presented as competitive without explanation.

### Recommended Next Cloud Action
- Treat `fixed_eval_combined_aggregate.csv` as the current cloud-side main-table source.
- Before final paper export, decide whether to include `diffusion_action_chunk` in the main table as a negative result or move it to an ablation/appendix table.
- Do not rerun these 30 rows unless the paper protocol changes.

## 2026-05-06 -- Windows Handoff: Eval Checkpoints Verified On Cloud

### Current Cloud State
- No new eval/training was launched.
- Final checkpoint availability check was run on cloud before switching to Windows-side development.
- GPU at check time: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`.

### Verified Eval Checkpoints Present
All checkpoint paths needed for the next unified CoDriveThesis paper eval are present on cloud:

- PPO teacher:
  `sim2real/codrive_thesis/best_reward_3655.17.pth`
- LatentBC:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_best_student_eval.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_best_student_eval.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`
- PAdapt:
  `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_nn/model_best_train.ckpt`
- diffusion latent:
  `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
- consistency latent:
  `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
- flow matching:
  `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_flow_nn/model_best_train.ckpt`
- diffusion action chunk:
  `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt`
- PureBC:
  `outputs/Dexh13HoraLightbulb_student_purebc_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_bc_nn/model_best.ckpt`

### Existing Clean Eval
- Existing paper eval:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- `validation_summary.txt`:
  `raw_rows=12`, `expected_rows=12`, `all_ok=True`
- This clean table covers:
  PPO teacher, LatentBC, DAgger, DOTPG.

### Recommended Next Cloud Action
- From Windows Codex, first read this file.
- Extend the existing paper eval pipeline to add:
  PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, and PureBC.
- Keep protocol aligned with the existing clean run unless intentionally changing the paper protocol:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`, seeds `42,43,44`, `2048` steps.

---

## 2026-05-06 -- DOTPG/Diffusion Local Files Synced, Eval Deferred

### Current Cloud State
- User explicitly deferred new eval execution; no eval or training was launched in this step.
- Cloud-visible handoff file is now present at:
  `docs/cloud_session_handoff.md`
- `AGENTS.md` now requires agents to read cloud-side docs as needed before cloud execution or preparing cloud commands. At minimum, reachable cloud sessions should check this file first.

### Synced From Local To Cloud
- Workflow/source:
  - `AGENTS.md`
  - `train.py`
  - `dexscrew/dotpg/`
  - `dexscrew/algo/eval_select.py`
  - diffusion-class student files under `dexscrew/algo/ppo/`
  - student wrappers/baseline files under `dexscrew/algo/student/`
- Scripts:
  - `scripts/cloud_codrive_diffusion4_student_1h.sh`
  - `scripts/cloud_codrive_diffusion4_student_continue2h.sh`
  - diffusion/DOTPG launch and visualizer scripts
- Config/package:
  - `configs/task/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - `configs/train/Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis.yaml`
  - `sim2real/codrive_thesis/`
- Docs/theory:
  - `docs/diffusion_algorithm.md`
  - `docs/dotpg_codrive_optimization.md`
  - `docs/dotpg_env.md`
  - `thesis_reference/DOTPG-draft.md`
- Selected local CoDriveThesis ckpts now available on cloud:
  - PAdapt continued:
    `outputs/Dexh13HoraLightbulb_student_padapt_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_nn/model_best_train.ckpt`
  - diffusion latent 3h:
    `outputs/Dexh13HoraLightbulb_student_diffusion_latent_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_nn/model_best_train.ckpt`
  - consistency latent 3h:
    `outputs/Dexh13HoraLightbulb_student_consistency_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_consistency_nn/model_best_train.ckpt`
  - flow matching continued:
    `outputs/Dexh13HoraLightbulb_student_flow_matching_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_flow_nn/model_best_train.ckpt`
  - diffusion action chunk 3h:
    `outputs/Dexh13HoraLightbulb_student_diffusion_action_chunk_codrive_thesis/codrive_thesis_teacher3655_s42_3h_final_20260504/stage2_diffusion_action_chunk_nn/model_best_train.ckpt`
  - PureBC continued:
    `outputs/Dexh13HoraLightbulb_student_purebc_codrive_thesis/codrive_thesis_continue2h_rising_s42_20260505_from3h/stage2_bc_nn/model_best.ckpt`

### Existing Paper Eval To Reuse
- Existing cloud paper-eval run:
  `outputs/paper_eval_codrive_thesis_cloud_s42_s43_s44_20260506_092231/`
- It already has clean `s42,s43,s44`, `2048`-step fixed eval for:
  - PPO teacher
  - LatentBC
  - DAgger
  - DOTPG
- `validation_summary.txt` reports:
  `all_ok=True`

### Recommended Next Cloud Action
- From Windows or Ubuntu, start by reading this file.
- Do not rerun the already-clean teacher/LatentBC/DAgger/DOTPG table unless the protocol changes.
- Extend the existing paper eval pipeline to add PAdapt, diffusion latent, consistency latent, flow matching, diffusion action chunk, and PureBC under the same task/seeds/steps protocol.

---

## 2026-05-06 -- CoDriveThesis Classic Baselines Completed, Eval Needs Repair

### Current Cloud State
- Cloud host: `cloud-training` / `/root/code/dexscrew-repro`
- GPU at last check: RTX 4090 D idle, about `1 MiB / 24564 MiB`, `0%`
- No active `tmux` sessions and no active `train.py` process at last check.

### Latest Pipeline
- Pipeline:
  `outputs/cloud_pipeline_codrive_thesis_classic/codrive_thesis_classic_s42_20260505_140934/`
- Phase:
  `done`
- Purpose:
  Train missing CoDriveThesis classic baselines from the frozen teacher:
  `sim2real/codrive_thesis/best_reward_3655.17.pth`
- Task:
  `Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis`
- Selected cloud resource profile:
  `aggressive`

### Training Status
All formal training phases reached expected wall-clock timeout:

| Phase | status | note |
|---|---:|---|
| `formal_latentbc` | `124` | expected 3h timeout |
| `formal_dagger` | `124` | expected 3h timeout |
| `formal_dotpg` | `124` | expected 12h timeout |

Training summary from `training_summary.tsv`:

| Algorithm | Training signal |
|---|---|
| `LatentBC` | `best_student_eval=1009.37`, `last_student_eval=673.23` |
| `DAgger pure_replay` | `best_student_eval=1358.47`, `last_student_eval=1100.66` |
| `DOTPG dual_bc5` | `Current Best=2774.05` |

### Existing Fixed Eval Result
Current `fixed_eval_summary.csv` is only partially valid:

| Checkpoint | status | avg_reward | avg_done_rate | validity |
|---|---:|---:|---:|---|
| `latentbc_model_last` | `0` | `4.068575` | `0.000163` | valid |
| `dagger_model_last` | `0` | `3.808516` | `0.000651` | valid |
| `latentbc_model_best_deploy` | `1` | `NA` | `NA` | invalid: checkpoint missing |
| `dagger_model_best_deploy` | `1` | `NA` | `NA` | invalid: checkpoint missing |
| `dotpg_model_best` | `1` | `NA` | `NA` | invalid eval command: missing DOTPG restore overrides |
| `dotpg_model_last` | `1` | `NA` | `NA` | invalid: checkpoint missing |

DOTPG eval failed because the eval command restored a `teacher_actor`/`dual` checkpoint with default DOTPG policy settings. Re-eval must include:

```text
+train.dotpg.state_mode=student
+train.dotpg.dynamic_state=True
+train.dotpg.policy_arch=teacher_actor
+train.dotpg.policy_output_mode=clamp
+train.dotpg.policy_loss_mode=dual
```

### Relevant Checkpoints
- LatentBC:
  `outputs/Dexh13HoraLightbulb_student_bc_codrive_thesis/codrive_thesis_formal_latentbc_s42/bc_nn/model_last.ckpt`
- DAgger:
  `outputs/Dexh13HoraLightbulb_student_dagger_codrive_thesis/codrive_thesis_formal_dagger_pure_replay_s42/dagger_nn/model_last.ckpt`
- DOTPG:
  `outputs/Dexh13HoraLightbulb_student_dotpg_codrive_thesis/codrive_thesis_formal_dotpg_dual_bc5_s42/student_output/dotpg_nn/model_best.ckpt`

### Known Pitfalls
- `outputs/local_pipeline_codrive_thesis_students_6x3h/.../summary.tsv` contains polluted `5001.07` values; use `strict_summary.tsv` instead.
- CoDriveThesis diffusion/PAdapt/PureBC training rewards are not final paper numbers; they still need identical fixed-step eval.
- DOTPG `model_last.ckpt` is absent in the latest classic cloud run; use `model_best.ckpt`.
- LatentBC/DAgger `model_best_deploy.ckpt` were absent in the latest classic cloud run; use `model_last.ckpt` unless a later selector run creates deploy aliases.

### Recommended Next Cloud Action
Run a unified CoDriveThesis paper-eval pipeline on the cloud for:

- PPO teacher
- PAdapt continued
- diffusion latent 3h
- consistency latent 3h
- flow matching continued
- diffusion action chunk 3h
- PureBC continued
- LatentBC cloud `model_last`
- DAgger cloud `model_last`
- DOTPG cloud `model_best`

Use one fixed protocol first:

```text
task = Dexh13HoraLightbulbSim2RealTwoFingerCoDriveThesis
seed = 42
num_envs = 48 or 256
steps = 256
termination enabled
obs_noise_t_scale = 0.01
obs_noise_e_scale = 0.02
```

Then extend to multiseed/robustness only after the single-seed table is clean.

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210423`
- tmux/session owner: `paper_codrive_thesis_20260506_210423`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210423/status/phase.txt`

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210536`
- tmux/session owner: `paper_codrive_thesis_20260506_210536`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210536/status/phase.txt`

### Correction: 20260506_210423 launch aborted
- Run dir: `outputs/paper_codrive_thesis_full_20260506_210423`
- Reason: initial launch did not attach to tmux due shell quoting; processes were stopped and `status/phase.txt` was set to `aborted_bad_tmux_launch`.
- Active run is `outputs/paper_codrive_thesis_full_20260506_210536` in tmux `paper_codrive_thesis_20260506_210536`.


### Correction: 20260506_210536 launch aborted
- Run dir: `outputs/paper_codrive_thesis_full_20260506_210536`
- Reason: evaluator restore commands were missing `train.ppo.proprio_adapt=True` for student checkpoints; script was fixed before any long training completed.
- This run was stopped and `status/phase.txt` was set to `aborted_eval_restore_arg_fix`.

## 2026-05-06 -- CoDriveThesis Full Paper Experiment Started

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210723`
- tmux/session owner: `paper_codrive_thesis_20260506_210723`
- Plan: formal 5-method 3-train-seed suite, unified eval, NFE/latency, representation ablation, robustness.
- GPU: NVIDIA GeForce RTX 4090 D, 24564 MiB
- Status file: `outputs/paper_codrive_thesis_full_20260506_210723/status/phase.txt`

## 2026-05-07 -- Checkpoint Selection Hotfix

- Run dir: outputs/paper_codrive_thesis_full_20260506_210723
- Added checkpoint aliaser session: paper_codrive_ckpt_aliaser_20260506_210723
- Reason: active 5h timeout runs do not reach eval_select 20M interval, so deploy checkpoints were not emitted.
- Action: model_best_deploy.ckpt symlinked to model_best_train.ckpt; see status/checkpoint_selection_hotfix.txt and status/checkpoint_aliases.tsv.

## 2026-05-07 -- Active Guard Started

- Session: paper_codrive_active_guard_20260506_210723
- Interval: 300 seconds
- Status: outputs/paper_codrive_thesis_full_20260506_210723/status/active_guard_status.txt
- Progress snapshot: outputs/paper_codrive_thesis_full_20260506_210723/status/training_progress.tsv
- Control alert: outputs/paper_codrive_thesis_full_20260506_210723/status/CONTROL_REQUIRED.txt

## 2026-05-08 -- CoDriveThesis Full Paper Experiment Pipeline Finished

- Run dir: `outputs/paper_codrive_thesis_full_20260506_210723`
- Validation: `outputs/paper_codrive_thesis_full_20260506_210723/validation_summary.txt`
- Main aggregate: `outputs/paper_codrive_thesis_full_20260506_210723/aggregate_csv/main_aggregate.csv`

---

## 2026-09-03 -- E001 / R1-motion cloud run active

### Current Cloud State

- Active tmux: m24_humantwist_e001_s42.
- Pipeline: outputs/cloud_pipeline_m24_humantwist_e001_m24_humantwist_e001_r1motion_s42_20260903_160559/.
- Cloud branch/commit: feat/m24-humanlike-twist / 8d729a217049b11272df5611ec11ff5d5eeecf2a.
- The R1 source adds index+thumb-only masking, dual positive-torque-gated drive, explicit grasp/drive/unload/return/regrasp state, and diagnostic-only force/wrench proxies. No current/deployment/load-threshold change was made.
- One-env GPU smoke completed 2026-09-03T16:07:00+08:00 with expected timeout exit 124, finite rewards, best_reward_6.98.pth, and no traceback/OOM/PhysX marker.
- The ordered 12288 env / 24576 minibatch capacity probe is active. First monitor sample used about 12443 MiB of the 49140 MiB RTX 4090.

### Recommended Next Cloud Action

- Do not start another job. Let this pipeline select the first healthy ordered capacity and finish its own timeout 1800 formal PPO phase; then inspect its selected checkpoint, logs, R1 metrics, and headed real-time behavior before any reward decision.

### E001 capacity update -- 2026-09-03T16:10:00+08:00

- The ordered 12288 env / 24576 minibatch probe ended with expected timeout 124 after 180 s, around 30.1k FPS, finite best reward 39.37, and no allocation/PhysX/NaN marker.
- The first capacity was selected without fallback. Its separate formal timeout 1800 phase is active from scratch; first monitored rewards rose 0.49 to 1.14, using about 12.5 GiB VRAM.
- Do not start a second job or alter this active formal run.

### E001 completion update -- 2026-09-03T16:40:01+08:00

- The sole formal 12288 env / 24576 minibatch PPO phase ended through expected timeout 124 after 1800 s, at about 51M agent steps and 28.7k final aggregate FPS. No traceback, CUDA OOM, PhysX, illegal-memory-access, or NaN marker was found.
- Selected checkpoint: outputs/XHandPasiniM24NutBolt_teacher/m24_humantwist_e001_r1motion_s42_20260903_160559_formal/stage1_nn/best_reward_549.29.pth; SHA-256 eb3a3f9f3cece152ad15f2b28ae3dca316e7ed04757a2011a4fe86c38a88585b.
- The tmux session is gone and GPU is idle. The entire pipeline and formal output were synchronized to the local repository with matching SHA-256.

### Recommended Next Cloud Action

- No cloud job is active. Run normal-real-time headed evaluation of this exact R1 checkpoint and assess the two-finger drive/release gates before changing a reward scale or launching another run.
# 2026-09-04 -- M24 E002-A1 bounded probe ended; do not extend

## Current cloud state

- Repo / branch: `/root/code/dexscrew-repro`, `feat/m24-humanlike-twist`.
- No active `tmux`, `train.py`, or GPU workload after the probe; RTX 4090 is
  idle.
- Current E002 source commits: `3945540b351f6605a885ddc8f7787ebf994c9358`
  (E002-A implementation) and `5ef588ea667f9bf328fd759ef593fc8e28704441`
  (E002 train config).

## E002 result

- Invalid config-only startup (exclude from results):
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1_s42_20260904_150357`.
  Hydra lacked the E002 train config; no simulator/GPU work occurred.  The
  config was added and composed successfully before the fresh run.
- Valid bounded pipeline:
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1_s42_20260904_150834`;
  tmux `m24_e002_a1_s42`; started `2026-09-04T15:08:34+08:00`, finished
  `2026-09-04T15:14:35+08:00`.
  - A0 deterministic math: passed.
  - 1-env smoke: `timeout 60`, exit `124` expected, no numerical/PhysX/OOM
    marker, transient checkpoint `..._smoke_1env/stage1_nn/best_reward_-2.00.pth`.
  - capacity probe: `12288/24576`, `timeout 300`, exit `124` expected, no
    numerical/PhysX/OOM marker, about 39.8k FPS and 11.8 GiB VRAM; transient
    checkpoint `..._probe_12288_24576/stage1_nn/best_reward_-2.00.pth`.
- This is a resource pass only, **not** an E002 behavioral pass or teacher
  selection.  Probe `pair_drive_ratio=0` and success `=0` throughout;
  pair gate max `3.57e-05`; final index/thumb contact states `0.00724` /
  `0.000163`; index signed torque stayed negative (`-0.00187` to
  `-6.71e-06`); contact-loss termination reached `0.01489`.  The 1-env smoke
  also reached contact-loss `1` with zero success.  Direct-drive contact loss
  after the grace period makes this A1 learning objective invalid.

## Preserved local artifacts

- Full pipeline/logs/config/status are synchronized locally at
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1_s42_20260904_150834/`.
- Smoke checkpoint SHA-256:
  `880db2ade085a112b48c3265342d14397b80b84017bd05accb029254e1053436`.
- Probe checkpoint SHA-256:
  `bb5be140f10f0fe9e01fa0fdfeff70cc67bca8e00a30cd7f54f67b5b36a94a37`.

## Single recommended next cloud action

Do not launch `timeout 1800`, E002-B/C, a student, or a headed gate from these
checkpoints.  First add/run a one-environment controlled-action E002-A0 trace
that logs raw index/thumb force, distance, torque vector, nut-axis projection,
contact state, and nut response for positive and negative actions.  Correct
only the demonstrated fingertip mapping, sign convention, force source, or
saved-grasp issue; then rerun deterministic checks, smoke, and one bounded
`timeout 300` A1 probe.
# 2026-09-04 -- E002-A0 controlled trace: reset lacks stable pair contact

## Controlled physical diagnostic

- Added `scripts/trace_m24_e002_a0.py` (cloud commits
  `a26afd322a430e5b5c326b4c796bcf8dccf7cfb9` and
  `36d82fa5f8f180ef13caa9684d4e69be34629174`).  It is no-PPO, one-env,
  bounded controlled action tracing; it logs raw index/thumb force, distance,
  torque vector/axis projection, contact state, reward/done, and nut response.
- Valid trace:
  `outputs/cloud_pipeline_m24_humantwist_e002_a0trace_s42_20260904_152300`,
  `timeout 180`, exit `0`, start/end `15:36:08` / `15:36:19+08:00`, 107 finite
  CSV rows.  Full trace artifact is synchronized locally.
- Finding: the saved pose has index/thumb centre distances about `0.0333` /
  `0.0432 m` but zero resultant force under zero action, so it is not a stable
  double-contact reset.  No candidate obtained two valid pair contributions.
  Same-direction pair action moved the nut `-0.351 rad` with index/thumb
  projected torques `+0.091/-0.431`; thumb-only positive action moved it
  `+0.351 rad` but lacks an index contribution; opposed actions lose a finger
  or hit workspace termination.

## Cloud state and required next action

- No active tmux, train process, or GPU workload.
- Do **not** flip a sign, relax the pair gate, extend A1, or select a transient
  checkpoint.  First use the Pasini pose-tuning workflow to record a fixed M24
  reset sustaining both index and thumb force/distance contact-on through the
  A0 pre-drive window.  Repeat this trace on that reset; only then make any
  evidence-backed sign/mapping correction and rerun smoke/probe.
# 2026-09-08 -- E002-A1b stopped at required A0c pair-drive calibration gate

## Current cloud state

- Branch: `feat/m24-humanlike-twist`; no active tmux, `train.py`, or GPU load.
- Local source `694a1b0` changed-file set was synced as cloud commit `973fa42`.
  Two pre-measurement NumPy-to-Torch limit fixes followed as `16dcfa7` and
  `8c6e96a`; neither changes an E002 gate, threshold, action mask, force/current
  setting, or stage definition.

## E002-A0b/A0c result

- Final pipeline:
  `outputs/cloud_pipeline_m24_humantwist_e002_m24_e002_a1b_s42_20260908_103643`.
  Deterministic A0 passed; A0 calibration exited `4`; terminal phase is
  `failed:a0_calibration`.
- Full calibration JSON:
  ```json
  {
    "best_directed_delta": 0.055169478058815,
    "best_mean_pair_gate": 0.05396845564246178,
    "best_probe_streak": 0,
    "candidate_count": 1024,
    "custom_init_dof_pos": [0.2141115516424179, 1.0670015811920166, 0.2418796271085739, 0.4392591416835785, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.1261191964149475, 1.4251307249069214, 0.74101722240448, 0.47267526388168335],
    "pose_index": 743,
    "required_probe_streak": 5,
    "status": "failed:no_pair_positive_drive"
  }
  ```
- A0b found a stable pair reset (it would otherwise exit 3); A0c could not
  find one candidate with five consecutive simultaneous positive pair-drive
  steps.  No force sign or drive action was selected.  No smoke, PPO A1b,
  checkpoint, fixed-step metric, or headed video exists.  The full log is
  `logs/a0_calibration.log`; the exact invocation is
  `commands/a0_calibration.sh`.

## Single recommended next cloud action

Do not relax the pair gate or start PPO.  Inspect the selected-reset raw
action basis and change only one evidence-backed calibration action-family
parameter, then rerun A0b/A0c.  Proceed to smoke/PPO only if a new calibration
JSON says `passed`.
