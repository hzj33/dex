# object_sim
MuJoCo Models for objects of daily living
(click image below to preview the whole set in details)
[![Video Preview](https://img.youtube.com/vi/3uxuXqnf3YA/maxresdefault.jpg)](https://youtu.be/3uxuXqnf3YA)


# Installation

```bash
pip install mujoco click scikit-video numpy
```

# Usage

## Preview Objects

Use `preview.py` to visualize 3D models:

```bash
# Preview a specific object (auto-rotate for 2 seconds)
python preview.py -o lightbulb

# Free browse mode - manual control until window closed
python preview.py -o lightbulb -f

# Custom duration (5 seconds)
python preview.py -o lightbulb -d 5.0

# Generate video (offline rendering)
python preview.py -o lightbulb -r offscreen

# Preview all objects
python preview.py
```

### Options

| Option | Description |
|--------|-------------|
| `-o, --object_name` | Name of the object folder to preview |
| `-f, --free` | Free browse mode - use mouse to control camera |
| `-d, --duration` | Preview duration in seconds (default: 2.0) |
| `-r, --render_mode` | `onscreen` (default) or `offscreen` for video |

### Controls (Free Browse Mode)

- **Left mouse drag**: Rotate view
- **Right mouse drag**: Pan view
- **Scroll wheel**: Zoom in/out
- **Close window**: Exit program

# Citation

If you found this code useful in any way, please cite our paper:
```
@inproceedings{dasari2023pgdm,
  title={Learning Dexterous Manipulation from Exemplar Object Trajectories and Pre-Grasps},
  author={Dasari, Sudeep and Gupta, Abhinav and Kumar, Vikash},
  booktitle={IEEE International Conference on Robotics and Automation 2023},
  year={2023}
}
```
