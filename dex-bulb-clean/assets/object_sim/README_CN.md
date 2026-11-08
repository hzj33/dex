# object_sim
日常物品的 MuJoCo 模型库
（点击下方图片查看完整预览视频）
[![视频预览](https://img.youtube.com/vi/3uxuXqnf3YA/maxresdefault.jpg)](https://youtu.be/3uxuXqnf3YA)


# 安装

```bash
pip install mujoco click scikit-video numpy
```

# 使用方法

## 预览物体模型

使用 `preview.py` 可视化 3D 模型：

```bash
# 预览指定物体（自动旋转 2 秒）
python preview.py -o lightbulb

# 自由浏览模式 - 手动控制相机，关闭窗口退出
python preview.py -o lightbulb -f

# 自定义时长（5 秒）
python preview.py -o lightbulb -d 5.0

# 生成视频（离屏渲染）
python preview.py -o lightbulb -r offscreen

# 预览所有物体
python preview.py
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `-o, --object_name` | 要预览的物体文件夹名称 |
| `-f, --free` | 自由浏览模式 - 使用鼠标控制相机 |
| `-d, --duration` | 预览时长，单位秒（默认：2.0） |
| `-r, --render_mode` | `onscreen`（默认）或 `offscreen` 生成视频 |

### 操作说明（自由浏览模式）

- **鼠标左键拖拽**：旋转视角
- **鼠标右键拖拽**：平移视角
- **鼠标滚轮**：缩放
- **关闭窗口**：退出程序

# 引用

如果您在研究中使用了本代码，请引用我们的论文：
```
@inproceedings{dasari2023pgdm,
  title={Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps},
  author={Dasari, Sudeep and Gupta, Abhinav and Kumar, Vikash},
  booktitle={IEEE International Conference on Robotics and Automation 2023},
  year={2023}
}
```
