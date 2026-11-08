# --------------------------------------------------------
# Your Hand + Hora Task (Inheritance from XHandHora)
# --------------------------------------------------------

import numpy as np
import torch
from .xhand_hora import XHandHora
from isaacgym.torch_utils import to_torch


class XHandPasini(XHandHora):
    """
    Your Hand manipulation task (Nut-Bolt / Screwdriver)

    继承自XHandHora，只需覆盖以下方法：
    1. __init__() - 初始化时修改DOF配置和初始姿态
    """

    def __init__(self, config, sim_device, graphics_device_id, headless):
        """
        初始化Your Hand任务

        关键改动点：
        1. 初始姿态 (joint_values) - 关节名称和数值
        2. Fingertip位置 - 指尖rigid body名称
        3. DOF限制 - 下限、上限、力矩限制
        4. DOF名称列表 - 用于PD增益随机化
        """
        # 在调用父类__init__之前，保存原始的_parse_hand_dof_props方法
        # 然后在init中覆盖
        super().__init__(config, sim_device, graphics_device_id, headless)

    def _parse_hand_dof_props(self):
        """
        覆盖DOF属性配置

        这是核心改动点！修改这里的DOF限制
        """
        self.num_xhand_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        xhand_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.xhand_hand_dof_lower_limits = []
        self.xhand_hand_dof_upper_limits = []

        # ========== 改这里：YOUR HAND的DOF限制 ==========
        # TODO: 根据你的灵巧手URDF，填入12个关节的上下限
        xhand_dof_lower_limits = [
            -0.175,  # DOF 0 - 修改为你的灵巧手
            0.0,  # DOF 1
            0.0,  # DOF 2
            0.0,  # DOF 3
            0.0,  # DOF 4
            0.0,  # DOF 5
            0.0,  # DOF 6
            0.0,  # DOF 7
            0.0,  # DOF 8
            0.0,  # DOF 9
            -1.05,  # DOF 10
            -0.17,  # DOF 11
        ]
        self.xhand_dof_lower_limits = np.array(xhand_dof_lower_limits)

        xhand_dof_upper_limits = [
            0.175,  # DOF 0 - 修改为你的灵巧手
            1.92,  # DOF 1
            1.92,  # DOF 2
            1.92,  # DOF 3
            1.92,  # DOF 4
            1.92,  # DOF 5
            1.92,  # DOF 6
            1.92,  # DOF 7
            1.92,  # DOF 8
            1.83,  # DOF 9
            1.57,  # DOF 10
            1.83,  # DOF 11
        ]

        if self.config["env"]["object"]["thumb_range_limit"]:
            xhand_dof_upper_limits[9] = 1.73
            xhand_dof_lower_limits[9] = 0.6
        self.xhand_dof_upper_limits = np.array(xhand_dof_upper_limits)

        # ========== 改这里：YOUR HAND的力矩限制 ==========
        # TODO: 根据你的灵巧手，填入力矩限制
        xhand_effort_limits = [
            0.4,
            1.1,
            0.4,
            1.1,
            0.4,
            1.1,
            0.4,
            1.1,
            1.1,
            0.4,
            1.1,
            1.1,
            0.4,
        ]

        for i in range(self.num_xhand_hand_dofs):
            xhand_hand_dof_props["lower"][i] = xhand_dof_lower_limits[i]
            xhand_hand_dof_props["upper"][i] = xhand_dof_upper_limits[i]
            self.xhand_hand_dof_lower_limits.append(xhand_dof_lower_limits[i])
            self.xhand_hand_dof_upper_limits.append(xhand_dof_upper_limits[i])
            xhand_hand_dof_props["effort"][i] = xhand_effort_limits[i]

            if self.torque_control:
                xhand_hand_dof_props["stiffness"][i] = 0.0
                xhand_hand_dof_props["damping"][i] = 0.0
            else:
                xhand_hand_dof_props["stiffness"][i] = self.config["env"]["controller"][
                    "pgain"
                ]
                xhand_hand_dof_props["damping"][i] = self.config["env"]["controller"][
                    "dgain"
                ]

            xhand_hand_dof_props["friction"][i] = 0.01
            xhand_hand_dof_props["armature"][i] = 0.001

        self.xhand_hand_dof_lower_limits = to_torch(
            self.xhand_hand_dof_lower_limits, device=self.device
        )
        self.xhand_hand_dof_upper_limits = to_torch(
            self.xhand_hand_dof_upper_limits, device=self.device
        )

        return xhand_hand_dof_props

    def _create_object_asset(self):
        """
        覆盖物体资源创建

        关键改动：Fingertip位置定义
        """
        # 调用父类的物体资源加载
        super()._create_object_asset()

        # ========== 改这里：YOUR HAND的Fingertip位置 ==========
        # TODO: 根据你的灵巧手URDF，修改fingertip的名称
        # 顺序很重要！必须是：[index, mid, pinky, ring, thumb]
        # 注意：这里会覆盖父类设置的fingertip_handles

        # 重新获取手资源
        hand_asset_file = self.config["env"]["asset"]["handAsset"]
        asset_root = self.config["env"].get("asset_root", "../../")

        # 重新设置fingertip
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name)
            for name in [
                "your_hand_index_rota_tip",  # TODO: 改为你的灵巧手
                "your_hand_mid_tip",  # TODO: 改为你的灵巧手
                "your_hand_pinky_tip",  # TODO: 改为你的灵巧手
                "your_hand_ring_tip",  # TODO: 改为你的灵巧手
                "your_hand_thumb_rota_tip",
            ]
        ]  # TODO: 改为你的灵巧手


# ========== 注意：初始姿态的覆盖 ==========
#
# 如果你需要修改初始姿态（joint_values），有两种方式：
#
# 方式1：在配置文件中新增initPose配置
#        然后在XHandHora.__init__中添加判断分支
#
# 方式2：在YourHandHora中添加新的方法，在__init__中调用
#
# 建议使用方式2，代码如下：

# def _init_grasping_states(self):
#     """初始化YOUR HAND特定的抓取姿态"""
#     self.saved_grasping_states = {}
#     num_random_poses = 5000
#
#     if self.config['env']['initPose'] == 'nutbolt_inclined':
#         joint_values = {
#             "your_hand_index_bend_joint": -0.17,      # TODO: 改为你的灵巧手
#             "your_hand_index_joint1": 1.1,
#             "your_hand_index_joint2": 0.4,
#             "your_hand_mid_joint1": 1.1,
#             "your_hand_mid_joint2": 0.4,
#             "your_hand_pinky_joint1": 0,
#             "your_hand_pinky_joint2": 0,
#             "your_hand_ring_joint1": 0,
#             "your_hand_ring_joint2": 0,
#             "your_hand_thumb_bend_joint": 1.3,
#             "your_hand_thumb_rota_joint1": 0.5,
#             "your_hand_thumb_rota_joint2": 0.45,
#         }
#     elif self.config['env']['initPose'] == 'screwdriver_inclined':
#         joint_values = {
#             "your_hand_index_bend_joint": -0.036,
#             "your_hand_index_joint1": 1.15,
#             "your_hand_index_joint2": 0.5,
#             "your_hand_mid_joint1": 0.925,
#             "your_hand_mid_joint2": 0.58,
#             "your_hand_pinky_joint1": 0,
#             "your_hand_pinky_joint2": 0,
#             "your_hand_ring_joint1": 1.3,
#             "your_hand_ring_joint2": 0.43,
#             "your_hand_thumb_bend_joint": 1.455,
#             "your_hand_thumb_rota_joint1": 0.817,
#             "your_hand_thumb_rota_joint2": 0.154,
#         }
#     else:
#         raise ValueError(f"Unknown initPose: {self.config['env']['initPose']}")
#
#     # 保存关节顺序
#     self.joint_values_lst = list(joint_values.values())
#
#     # 生成随机初始姿态
#     for s in self.randomize_scale_list:
#         scale_key = str(s)
#         random_pose_data = torch.zeros((num_random_poses, 19), device=self.device, dtype=torch.float)
#
#         for i in range(12):  # 12 DOFs
#             random_pose_data[:, i] = torch.ones(num_random_poses, device=self.device) * self.joint_values_lst[i]
#
#         random_pose_data[:, 12] = torch.zeros(num_random_poses, device=self.device)  # x
#         random_pose_data[:, 13] = torch.zeros(num_random_poses, device=self.device)  # y
#         random_pose_data[:, 14] = self.reset_z_threshold + 0.1  # z
#         random_pose_data[:, 15:18] = 0.0
#         random_pose_data[:, 18] = 1.0
#         quat_norm = torch.norm(random_pose_data[:, 15:19], dim=1, keepdim=True)
#         random_pose_data[:, 15:19] = random_pose_data[:, 15:19] / quat_norm
#
#         self.saved_grasping_states[scale_key] = random_pose_data
#         print(f"Generated {num_random_poses} random initial poses for YourHand at scale {s}")
