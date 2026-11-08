# 2026-10-10 三指触觉显示修正

已实现默认三指指尖布局：拇指S11、食指S3、中指S6，各10列×12行。旧完整四指布局通过窗口按钮或启动参数切换。没有操作真实手或机械臂，没有训练。

## 运行

```bash
bash outputs/local_m24_deployment/e003_return_progress_recovery_20261008/run_deploy.sh --show-tactile
```

完整布局：

```bash
DEXH13_TACTILE_LAYOUT=full bash outputs/local_m24_deployment/e003_return_progress_recovery_20261008/run_deploy.sh --show-tactile
```

默认颜色范围0–10 N。按用户两次明确说明，“实际传感器数值的0到1是指0到10N”，显示进程将既有采集转换后的sensor_force乘10，0.1/0.5/1分别显示1/5/10 N。保留SDK原始帧和采集值，不让显示换算进入策略。这是用户提供的显示标定关系，并非本轮重新测出的硬件标定，也不能直接拿去改变其他SDK数据接口的单位。

## 箭头和过滤

箭头保持SDK局部Fx/Fy，不新增法向箭头，也不直接套用仿真的剪切符号转换。仿真M24显示用velocity_only_display_field：颜色是原始总力，箭头排除法向，只使用滑动剪切项。SDK的Fx/Fy与仿真计算的滑动剪切来源不同，本轮只统一显示含义，真实物理方向仍需滑动核对。

颜色顶满不移除有效切向箭头；颜色范围与箭长范围独立。新三指布局画出所有符合条件的可见点，最多360个，黄色箭头加深色描边、最大0.85格，避免四指共享72个名额的覆盖不足。完整旧布局保留原稀疏筛选。

默认总力过滤阈值0.2 N，≤阈值显示零且不画箭头；有效接触再以独立切向门槛0.02 N筛选，箭长满量程1 N。纯法向且切向为零不画虚假方向。过滤只作用显示，不校零、不改原始采集；阈值是可调初值，并非实测食指噪声上限。浮点边界使用1e-12 N容差。

窗口支持颜色上限、噪声阈值、箭头门槛、箭长量程输入，点击显示采集三分量、对应N值、总力、切向、过滤状态以及超颜色量程提示。Fx/Fy/Fz单分量模式保留正负量程。

启动覆盖：DEXH13_TACTILE_MAX_N=10、DEXH13_TACTILE_LAYOUT=fingertips、DEXH13_TACTILE_NOISE_N=0.2、DEXH13_TACTILE_ARROW_MIN_N=0.02、DEXH13_TACTILE_ARROW_SCALE_N=1。直接Python CLI对应--force-max-n、--layout、--noise-floor-n、--arrow-min-n、--arrow-scale-n。

## 验证与限制

- display_tests.json：三指映射/点序、0–1到0–10 N换算、360点同时有箭头、法向超过10 N和颜色范围变化不改变切向箭头、纯法向无虚假箭头、过滤边界与独立切向门槛、点击、布局切换、signed量程、非法输入及两种截图均通过。
- 原触觉集成测试按本次开始前的策略/SDK/模型SHA核对，通过23项，覆盖假SDK单连接/互斥/控制优先、GUI关闭/启动失败、读错误、SIGINT/SIGTERM及清理。旧测试初次仅“所有原文件不变”的历史SHA断言失败：它假定部署入口仍为9月20日且共享显示不可修改；改为本会话开始前的不变策略文件基线后通过，未放宽行为断言。
- 本次当前入口普通/触觉各20步dry run，SDK导入阻断、无DISPLAY，q/动作/前后目标逐值一致。分别核对用户设置0.02与随后自行调整的0.04，两组普通/触觉输出均一致。模型内部0.05、a61及legacy保持；部署Python入口仅用户ACTION_SCALE一行发生变化，本轮显示实现未修改该入口。
- 两种截图已目视核对，新布局最高颜色上的黄色描边箭头清晰；M24 --snapshot CLI也通过。

验证命令使用sim2real环境Python，运行outputs/local_pipeline_m24_tactile_display_fix_20261010中的verify_display.py、test_integration.py、verify_current_deployment.py；产物和修改前备份同目录。保留原ZIP历史快照，当前目录脚本与manifest已更新。原包内历史训练/仿真验收记录不随显示改动改写。

全部操作均为本地软件检查/假设备。真实噪声阈值、方向和触觉读取对控制周期的影响仍需用户真机复核。下一步使用原--show-tactile命令，观察无接触底色和轻接触，按需要调阈值，并用滑动核对箭头方向。

源码归档[m24_tactile_display_fix_2026-10-10.patch](m24_tactile_display_fix_2026-10-10.patch)以/home/vision为根，可用git apply --unidiff-zero应用；外部SDK目录无Git、outputs被忽略。项目里其他用户文档编辑不纳入本次提交。
