# --------------------------------------------------------
# Learning Dexterous Manipulation Skills from Imperfect Simulations
# Written by Paper Authors
# Copyright (c) 2025 All Authors
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Based on: Lessons from Learning to Spin "Pens"
# Copyright (c) 2024 All Authors
# Licensed under MIT License
# https://github.com/HaozhiQi/penspin/
# --------------------------------------------------------

import os
from typing import Optional
import torch
import omegaconf
import numpy as np
import time

from glob import glob
from collections import OrderedDict

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import (
    quat_conjugate,
    quat_mul,
    to_torch,
    quat_apply,
    tensor_clamp,
    torch_rand_float,
    quat_from_euler_xyz,
)

from ..utils.point_cloud_prep import sample_cylinder
from .base.vec_task import VecTask
from dexscrew.utils.misc import tprint
import torch.nn.functional as F


class XHandHora(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config
        env_cfg = self.config.get("env", {})
        self.hand_asset_cfg = env_cfg.get("asset", {})
        self.object_cfg = env_cfg.get("object", {})
        fingertip_cfg = self.hand_asset_cfg.get("fingertipBodies")
        if fingertip_cfg is None:
            self.fingertip_body_names = self._default_fingertip_body_names()
        else:
            self.fingertip_body_names = [str(x) for x in list(fingertip_cfg)]
        # before calling init in VecTask, need to do
        # 0. setup the dim info
        self.numActions = config["env"]["numActions"]
        self.fingers_num = len(self.fingertip_body_names)
        self.apply_action_mask = self._cfg_bool(
            env_cfg.get("apply_action_mask", self._default_apply_action_mask()),
            default=self._default_apply_action_mask(),
        )
        self.custom_action_mask_indices = env_cfg.get("action_mask_indices")
        # 1. setup randomization
        self._setup_domain_rand_config(config["env"]["randomization"])
        # 2. setup privileged information
        self._setup_priv_option_config(config["env"]["privInfo"])
        # 3. setup object assets
        self._setup_object_info(config["env"]["object"])
        # 4. setup rewards
        self._setup_reward_config(config["env"]["reward"])
        # unclassified config
        self.base_obj_scale = config["env"]["baseObjScale"]
        self.aggregate_mode = self.config["env"]["aggregateMode"]
        self.up_axis = "z"
        self.rotation_axis = config["env"]["rotation_axis"]
        self.reset_z_threshold = self.config["env"]["reset_z_threshold"]
        self.reset_dist_threshold = self.config["env"]["reset_dist_threshold"]
        self.normalize_penalties_by_num_actions = self._cfg_bool(
            env_cfg.get("normalize_penalties_by_num_actions", False), default=False
        )
        self._setup_finger_object_contact_config(
            env_cfg.get("finger_object_contact", {})
        )
        self._setup_termination_config(env_cfg.get("termination", {}))
        self._setup_two_finger_gate_config(env_cfg.get("two_finger_gate", {}))
        self._setup_fingertip_tangent_reward_config(
            env_cfg.get("fingertip_tangent_reward", {})
        )
        self._setup_fingertip_torque_reward_config(
            env_cfg.get("fingertip_torque_reward", {})
        )
        self._setup_active_two_finger_contact_config(
            env_cfg.get("active_two_finger_contact", {})
        )
        self._setup_opposition_grip_reward_config(
            env_cfg.get("opposition_grip_reward", {})
        )
        self._setup_thumb_slip_diagnostics_config(
            env_cfg.get("thumb_slip_diagnostics", {})
        )
        self._setup_thumb_slip_penalty_config(env_cfg.get("thumb_slip_penalty", {}))
        self.with_camera = config["env"]["enableCameraSensors"]
        self.nut_termination_history_len = config["env"]["object"][
            "nut_termination_history_len"
        ]
        self.nut_stagnation_eps = config["env"]["object"]["nut_stagnation_eps"]
        self.screw_upper_limit = config["env"]["object"]["screw_upper_limit"]

        # Important: map CUDA device IDs to Vulkan ones.
        graphics_device_id = 0

        super().__init__(config, sim_device, graphics_device_id, headless)
        self.eval_done_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.max_episode_length = self.config["env"]["episodeLength"]
        self.dt = self.sim_params.dt

        if self.viewer:
            cam_pos = gymapi.Vec3(0.0, 0.4, 1.5)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        net_contact_forces = self.gym.acquire_net_contact_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.xhand_hand_default_dof_pos = torch.zeros(
            self.num_xhand_hand_dofs, dtype=torch.float, device=self.device
        )
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.xhand_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, : self.num_xhand_hand_dofs
        ]
        self.xhand_hand_dof_pos = self.xhand_hand_dof_state[..., 0]
        self.xhand_hand_dof_vel = self.xhand_hand_dof_state[..., 1]
        self.pre_state = torch.zeros_like(self.xhand_hand_dof_pos)
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        thumb_indices = [i for i, name in enumerate(dof_names) if "thumb" in name]
        pose_penalty_cfg = self.config["env"].get("pose_diff_penalty", {})
        self.pose_diff_penalty_thumb_weight = float(
            pose_penalty_cfg.get("thumb_weight", 0.0)
        )
        self.pose_diff_penalty_mask = torch.ones(
            self.num_actions, device=self.device, dtype=torch.float
        )
        if len(thumb_indices) > 0:
            self.pose_diff_penalty_mask[thumb_indices] = (
                self.pose_diff_penalty_thumb_weight
            )
        self.pose_diff_penalty_thumb_indices = torch.as_tensor(
            thumb_indices, dtype=torch.long, device=self.device
        )
        self.contact_forces = gymtorch.wrap_tensor(net_contact_forces).view(
            self.num_envs, -1, 3
        )

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(
            self.num_envs, -1, 13
        )
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(
            -1, 13
        )

        self._refresh_gym()

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        self.prev_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        self.cur_targets = torch.zeros(
            (self.num_envs, self.num_dofs), dtype=torch.float, device=self.device
        )
        # object apply random forces parameters
        self.force_scale = self.config["env"].get("forceScale", 0.0)
        self.random_force_prob_scalar = self.config["env"].get(
            "randomForceProbScalar", 0.0
        )
        self.force_decay = self.config["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.config["env"].get("forceDecayInterval", 0.08)
        self.force_decay = to_torch(
            self.force_decay, dtype=torch.float, device=self.device
        )
        self.rb_forces = torch.zeros(
            (self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device
        )
        self.contact_thresh = torch.zeros(
            (self.num_envs, self.num_contacts), dtype=torch.float, device=self.device
        )

        self.saved_grasping_states = {}
        num_random_poses = 5000
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        self.hand_dof_names = list(dof_names)
        joint_values = self._resolve_hand_init_pose(dof_names)
        self.joint_values_lst = list(joint_values.values())
        hand_dof_dim = self.num_xhand_hand_dofs
        obj_pose_start = hand_dof_dim
        for s in self.randomize_scale_list:
            scale_key = str(s)
            random_pose_data = torch.zeros(
                (num_random_poses, hand_dof_dim + 7),
                device=self.device,
                dtype=torch.float,
            )

            # Pre-defined pose
            for i in range(hand_dof_dim):
                random_pose_data[:, i] = (
                    torch.ones(num_random_poses, device=self.device)
                    * self.joint_values_lst[i]
                )
            random_pose_data[:, obj_pose_start + 0] = torch.zeros(
                num_random_poses, device=self.device
            )  # x
            random_pose_data[:, obj_pose_start + 1] = torch.zeros(
                num_random_poses, device=self.device
            )  # y
            random_pose_data[:, obj_pose_start + 2] = (
                self.reset_z_threshold + 0.1
            )  # z - slightly above threshold
            random_pose_data[:, obj_pose_start + 3 : obj_pose_start + 6] = 0.0
            random_pose_data[:, obj_pose_start + 6] = 1.0
            quat_norm = torch.norm(
                random_pose_data[:, obj_pose_start + 3 : obj_pose_start + 7],
                dim=1,
                keepdim=True,
            )
            random_pose_data[:, obj_pose_start + 3 : obj_pose_start + 7] = (
                random_pose_data[:, obj_pose_start + 3 : obj_pose_start + 7]
                / quat_norm
            )

            self.saved_grasping_states[scale_key] = random_pose_data
            print(
                f"Generated {num_random_poses} random initial poses for XHand at scale {s}"
            )

        self.rot_axis_buf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        sign, axis = self.rotation_axis[0], self.rotation_axis[1]
        axis_index = ["x", "y", "z"].index(axis)
        self.rot_axis_buf[:, axis_index] = 1
        self.rot_axis_buf[:, axis_index] = (
            -self.rot_axis_buf[:, axis_index]
            if sign == "-"
            else self.rot_axis_buf[:, axis_index]
        )

        # useful buffers
        self.init_pose_buf = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )  # +1 for screw
        self.actions = torch.zeros(
            (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
        )
        # there is an extra dim [self.control_freq_inv] because we want to get a mean over multiple control steps
        self.torques = torch.zeros(
            (self.num_envs, self.control_freq_inv, self.num_dofs),
            device=self.device,
            dtype=torch.float,
        )
        self.dof_vel_finite_diff = torch.zeros(
            (self.num_envs, self.control_freq_inv, self.num_actions),
            device=self.device,
            dtype=torch.float,
        )

        # calculate velocity at control frequency instead of simulated frequency
        self.object_pos_prev = self.object_pos.clone()
        self.object_rot_prev = self.object_rot.clone()
        self.ft_pos_prev = self.fingertip_pos.clone()
        self.ft_rot_prev = self.fingertip_orientation.clone()
        self.dof_vel_prev = self.dof_vel_finite_diff.clone()
        self.nut_dof_pos_prev = self.nut_dof_pos.clone()

        self.obj_linvel_at_cf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.obj_angvel_at_cf = torch.zeros(
            (self.num_envs, 3), device=self.device, dtype=torch.float
        )
        self.ft_linvel_at_cf = torch.zeros(
            (self.num_envs, 4 * 3), device=self.device, dtype=torch.float
        )
        self.ft_angvel_at_cf = torch.zeros(
            (self.num_envs, 4 * 3), device=self.device, dtype=torch.float
        )
        self.nut_dof_vel_cf = torch.zeros(
            (self.num_envs, 1), device=self.device, dtype=torch.float
        )
        self.prev_thumb_drive_w = torch.zeros(
            (self.num_envs,), device=self.device, dtype=torch.float
        )
        self.dof_acc = torch.zeros(
            (self.num_envs, self.num_dofs), device=self.device, dtype=torch.float
        )

        assert type(self.p_gain) in [int, float] and type(self.d_gain) in [
            int,
            float,
        ], "assume p_gain and d_gain are only scalars"
        self.p_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.p_gain
        )
        self.d_gain = (
            torch.ones(
                (self.num_envs, self.num_actions), device=self.device, dtype=torch.float
            )
            * self.d_gain
        )

        # debug and statistics
        self.evaluate = self.config["on_evaluation"]
        self.evaluate_cache_name = self.config["eval_cache_name"]
        self.stat_sum_rewards = [0 for _ in self.object_type_list]  # all episode reward
        self.stat_sum_episode_length = [
            0 for _ in self.object_type_list
        ]  # average episode length
        self.stat_sum_rotate_rewards = [
            0 for _ in self.object_type_list
        ]  # rotate reward, with clipping
        self.stat_sum_rotate_penalty = [
            0 for _ in self.object_type_list
        ]  # rotate penalty with clipping
        self.stat_sum_unclip_rotate_rewards = [
            0 for _ in self.object_type_list
        ]  # rotate reward, with clipping
        self.stat_sum_unclip_rotate_penalty = [
            0 for _ in self.object_type_list
        ]  # rotate penalty with clipping
        self.extrin_log = []
        self.env_evaluated = [0 for _ in self.object_type_list]
        self.evaluate_iter = 0

        self.x_unit_tensor = to_torch(
            [1, 0, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch(
            [0, 1, 0], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch(
            [0, 0, 1], dtype=torch.float, device=self.device
        ).repeat((self.num_envs, 1))

    def _create_envs(self, num_envs, spacing, num_per_row):
        self._create_ground_plane()
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        self._create_object_asset()
        xhand_hand_dof_props = self._parse_hand_dof_props()
        hand_pose, obj_pose = self._init_object_pose()

        # compute aggregate size
        self.num_xhand_hand_bodies = self.gym.get_asset_rigid_body_count(
            self.hand_asset
        )
        self.num_xhand_hand_shapes = self.gym.get_asset_rigid_shape_count(
            self.hand_asset
        )
        max_agg_bodies = self.num_xhand_hand_bodies + 2
        max_agg_shapes = self.num_xhand_hand_shapes + 2

        self.envs = []
        self.vid_record_tensor = (
            None  # Used for record video during training, NOT FOR POLICY OBSERVATION
        )
        self.object_init_state = []

        self.hand_indices = []
        self.hand_actors = []
        self.object_indices = []
        self.object_type_at_env = []

        self.obj_point_clouds = []

        xhand_hand_rb_count = self.gym.get_asset_rigid_body_count(self.hand_asset)
        self.screw_base_rb_handle = xhand_hand_rb_count + 0  # base
        self.screw_bolt_rb_handle = xhand_hand_rb_count + 1  # bolt
        self.screw_nut_rb_handle = xhand_hand_rb_count + 2  # nut
        object_rb_count = 3

        self.object_rb_handles = list(
            range(xhand_hand_rb_count, xhand_hand_rb_count + object_rb_count)
        )

        for i in range(num_envs):
            tprint(f"{i} / {num_envs}")
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(
                    env_ptr, max_agg_bodies * 20, max_agg_shapes * 20, True
                )

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(
                env_ptr, self.hand_asset, hand_pose, "hand", i, -1, 1
            )
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, xhand_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)
            self.hand_actors.append(hand_actor)

            # add object
            eval_object_type = self.config["env"]["object"]["evalObjectType"]
            if eval_object_type is None:
                object_type_id = np.random.choice(
                    len(self.object_type_list), p=self.object_type_prob
                )
            else:
                object_type_id = self.object_type_list.index(eval_object_type)

            self.object_type_at_env.append(object_type_id)
            object_asset = self.object_asset_list[object_type_id]
            print(f"env {i} object_asset id: {object_type_id}")

            object_handle = self.gym.create_actor(
                env_ptr, object_asset, obj_pose, "object", i, 0, 2
            )
            self.object_init_state.append(
                [
                    obj_pose.p.x,
                    obj_pose.p.y,
                    obj_pose.p.z,
                    obj_pose.r.x,
                    obj_pose.r.y,
                    obj_pose.r.z,
                    obj_pose.r.w,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ]
            )
            object_idx = self.gym.get_actor_index(
                env_ptr, object_handle, gymapi.DOMAIN_SIM
            )
            self.object_indices.append(object_idx)
            self._update_priv_buf(env_id=i, name="screw_joint_friction", value=0.2)

            self.obj_scale = self.base_obj_scale
            if self.randomize_scale:
                num_scales = len(self.randomize_scale_list)
                self.obj_scale = np.random.uniform(
                    self.randomize_scale_list[i % num_scales] - 0.025,
                    self.randomize_scale_list[i % num_scales] + 0.025,
                )
            self.gym.set_actor_scale(env_ptr, object_handle, self.obj_scale)
            if hasattr(self, "object_scale_buf"):
                self.object_scale_buf[i] = float(self.obj_scale)
            self._update_priv_buf(env_id=i, name="obj_scale", value=self.obj_scale)

            obj_com = [0, 0, 0]
            if self.randomize_com:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                if "screw" in self.config["env"]["object"]["type"]:
                    assert len(prop) == 3, "Screw object should have 2 rigid bodies"
                    # Base COM randomization (smaller range since it's the stabilizing part)
                    base_com = [
                        np.random.uniform(
                            self.randomize_com_lower / 2, self.randomize_com_upper / 2
                        ),
                        np.random.uniform(
                            self.randomize_com_lower / 2, self.randomize_com_upper / 2
                        ),
                        np.random.uniform(
                            self.randomize_com_lower / 2, self.randomize_com_upper / 2
                        ),
                    ]

                    nut_com = [
                        np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                        np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                        0.35
                        + np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                    ]

                    prop[0].com.x, prop[0].com.y, prop[0].com.z = base_com
                    prop[2].com.x, prop[2].com.y, prop[2].com.z = nut_com
                    obj_com = base_com
                    ro_hand_rb_count = self.gym.get_asset_rigid_body_count(
                        self.hand_asset
                    )
                else:
                    assert len(prop) == 1
                    obj_com = [
                        np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                        np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                        np.random.uniform(
                            self.randomize_com_lower, self.randomize_com_upper
                        ),
                    ]
                    prop[0].com.x, prop[0].com.y, prop[0].com.z = obj_com
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
            self._update_priv_buf(env_id=i, name="obj_com", value=obj_com)

            obj_friction = 1.0
            obj_restitution = 0.0
            # Friction randomization
            if self.randomize_friction:
                rand_friction = np.random.uniform(
                    self.randomize_friction_lower, self.randomize_friction_upper
                )
                obj_restitution = np.random.uniform(
                    self.randomize_restitution_lower,
                    self.randomize_restitution_upper,
                )

                hand_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, hand_actor
                )
                for p in hand_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, hand_actor, hand_props
                )

                object_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, object_handle
                )
                for p in object_props:
                    p.friction = rand_friction
                    p.restitution = obj_restitution
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, object_handle, object_props
                )
                obj_friction = rand_friction
            self._update_priv_buf(env_id=i, name="obj_friction", value=obj_friction)
            self._update_priv_buf(
                env_id=i, name="obj_restitution", value=obj_restitution
            )

            if self.randomize_mass:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                for p in prop:
                    p.mass = np.random.uniform(
                        self.randomize_mass_lower, self.randomize_mass_upper
                    )
                self.gym.set_actor_rigid_body_properties(env_ptr, object_handle, prop)
                self._update_priv_buf(env_id=i, name="obj_mass", value=prop[0].mass)
            else:
                prop = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
                self._update_priv_buf(env_id=i, name="obj_mass", value=prop[0].mass)

            if self.point_cloud_sampled_dim > 0:
                self.obj_point_clouds.append(
                    self.asset_point_clouds[object_type_id] * self.obj_scale
                )

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # for training record, visualized in tensorboard
            if self.with_camera:
                self.vid_record_tensor = self._create_camera(env_ptr)

            self.envs.append(env_ptr)

        self.obj_point_clouds = to_torch(
            np.array(self.obj_point_clouds), device=self.device, dtype=torch.float
        )
        self.object_init_state = to_torch(
            self.object_init_state, device=self.device, dtype=torch.float
        ).view(self.num_envs, 13)
        self.object_rb_handles = to_torch(
            self.object_rb_handles, dtype=torch.long, device=self.device
        )
        self.hand_indices = to_torch(
            self.hand_indices, dtype=torch.long, device=self.device
        )
        self.object_indices = to_torch(
            self.object_indices, dtype=torch.long, device=self.device
        )
        self.object_type_at_env = to_torch(
            self.object_type_at_env, dtype=torch.long, device=self.device
        )

    def _create_camera(self, env_ptr) -> torch.Tensor:
        """Create a camera in a particular environment. Should be called in _create_envs."""
        camera_props = gymapi.CameraProperties()
        camera_props.width = 256
        camera_props.height = 256
        camera_props.enable_tensors = True

        camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

        cam_pos = gymapi.Vec3(0.0, 0.2, 0.75)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)

        self.gym.set_camera_location(camera_handle, env_ptr, cam_pos, cam_target)
        # obtain camera tensor
        vid_record_tensor = self.gym.get_camera_image_gpu_tensor(
            self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR
        )
        # wrap camera tensor in a pytorch tensor
        vid_record_tensor.device = 0
        torch_vid_record_tensor = gymtorch.wrap_tensor(vid_record_tensor)
        assert torch_vid_record_tensor.shape == (
            camera_props.height,
            camera_props.width,
            4,
        )

        return torch_vid_record_tensor

    def reset_idx(self, env_ids):
        if self.randomize_pd_gains:
            dof_names = self.gym.get_asset_dof_names(self.hand_asset)
            pgain_priv = torch.zeros(
                (len(env_ids), self.num_xhand_hand_dofs),
                device=self.device,
                dtype=torch.float,
            )
            dgain_priv = torch.zeros(
                (len(env_ids), self.num_xhand_hand_dofs),
                device=self.device,
                dtype=torch.float,
            )
            for joint_idx, joint_name in enumerate(dof_names):
                p_lower = self.randomize_p_gain_lower
                p_upper = self.randomize_p_gain_upper
                d_lower = self.randomize_d_gain_lower
                d_upper = self.randomize_d_gain_upper
                pgain_priv[:, joint_idx] = torch_rand_float(
                    p_lower, p_upper, (len(env_ids), 1), device=self.device
                ).squeeze(-1)
                dgain_priv[:, joint_idx] = torch_rand_float(
                    d_lower, d_upper, (len(env_ids), 1), device=self.device
                ).squeeze(-1)

                self.p_gain[env_ids, joint_idx] = pgain_priv[:, joint_idx]
                self.d_gain[env_ids, joint_idx] = dgain_priv[:, joint_idx]

            self._update_priv_buf(env_ids, "pgain", pgain_priv)
            self._update_priv_buf(env_ids, "dgain", dgain_priv)

        self.random_obs_noise_e[env_ids] = torch.normal(
            0,
            self.random_obs_noise_e_scale,
            size=(len(env_ids), self.num_actions),
            device=self.device,
            dtype=torch.float,
        )
        self.random_action_noise_e[env_ids] = torch.normal(
            0,
            self.random_action_noise_e_scale,
            size=(len(env_ids), self.num_actions),
            device=self.device,
            dtype=torch.float,
        )
        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        num_scales = len(self.randomize_scale_list)
        for n_s in range(num_scales):
            s_ids = env_ids[
                (env_ids % num_scales == n_s).nonzero(as_tuple=False).squeeze(-1)
            ]
            if len(s_ids) == 0:
                continue
            obj_scale = self.randomize_scale_list[n_s]
            scale_key = str(obj_scale)
            # single object (category) case:
            sampled_pose_idx = np.random.randint(
                self.saved_grasping_states[scale_key].shape[0], size=len(s_ids)
            )
            sampled_pose = self.saved_grasping_states[scale_key][
                sampled_pose_idx
            ].clone()
            sampled_object_pos = self._sample_object_init_positions(len(s_ids))
            sampled_pose[:, self.numActions : self.numActions + 3] = sampled_object_pos
            self.object_z = float(sampled_object_pos[0, 2].item())

            object_pose_noise = torch.normal(
                0,
                self.random_pose_noise,
                size=(sampled_pose.shape[0], 7),
                device=self.device,
                dtype=torch.float,
            )
            object_pose_noise[:, 3:] = 0
            self.root_state_tensor[self.object_indices[s_ids], :7] = (
                sampled_pose[:, self.numActions :] + object_pose_noise
            )

            self.root_state_tensor[self.object_indices[s_ids], 7:13] = 0
            pos = sampled_pose[:, : self.numActions]
            self._update_priv_buf(
                env_id=s_ids, name="hand_joint_pos", value=pos
            )  # give the policy correct hand joint position
            half_widths = torch.tensor(
                0.1 * (self.xhand_dof_upper_limits - self.xhand_dof_lower_limits) / 2.0,
                device=self.device,
                dtype=torch.float,
            )
            hand_joint_noise = (torch.rand_like(half_widths) * 2 - 1) * half_widths
            pos = pos + hand_joint_noise
            self.sampled_pose = sampled_pose[:, : self.numActions]
            self.xhand_hand_dof_pos[s_ids, :] = pos
            self.xhand_hand_dof_vel[s_ids, :] = 0
            self.prev_targets[s_ids, : self.num_xhand_hand_dofs] = pos
            self.cur_targets[s_ids, : self.num_xhand_hand_dofs] = pos

            screw_joint_init = torch.zeros(
                (len(s_ids), self.num_dofs - self.num_xhand_hand_dofs),
                device=self.device,
            )
            self.prev_targets[s_ids, self.num_xhand_hand_dofs :] = screw_joint_init
            self.cur_targets[s_ids, self.num_xhand_hand_dofs :] = screw_joint_init
            full_dof_state = torch.cat([pos, screw_joint_init], dim=1)
            self.init_pose_buf[s_ids, :] = full_dof_state

        # X-axis rotation
        random_degrees_x = (
            torch.rand(len(env_ids), device=self.device) * 10 + 20
        )  # randomize orientation bewteen 20-30 degrees
        angles_x = np.pi / 2 - random_degrees_x * (np.pi / 180)
        cos_half_x, sin_half_x = torch.cos(angles_x / 2), torch.sin(angles_x / 2)

        # Y-axis rotation
        random_degrees_y = (
            torch.rand(len(env_ids), device=self.device) * 10 - 5 + 90
        )  # randomize orientation bewteen -5-5 degrees, 90 is the correction term for y
        angles_y = np.pi / 2 - random_degrees_y * (np.pi / 180)
        cos_half_y, sin_half_y = torch.cos(angles_y / 2), torch.sin(angles_y / 2)

        # Create quaternions for x and y rotations
        quat_x = torch.stack(
            [
                sin_half_x,
                torch.zeros_like(sin_half_x),
                torch.zeros_like(sin_half_x),
                cos_half_x,
            ],
            dim=1,
        )
        quat_y = torch.stack(
            [
                torch.zeros_like(sin_half_y),
                sin_half_y,
                torch.zeros_like(sin_half_y),
                cos_half_y,
            ],
            dim=1,
        )

        quats = torch.stack(
            [
                quat_y[:, 3] * quat_x[:, 0]
                + quat_y[:, 0] * quat_x[:, 3]
                + quat_y[:, 1] * quat_x[:, 2]
                - quat_y[:, 2] * quat_x[:, 1],  # x
                quat_y[:, 3] * quat_x[:, 1]
                - quat_y[:, 0] * quat_x[:, 2]
                + quat_y[:, 1] * quat_x[:, 3]
                + quat_y[:, 2] * quat_x[:, 0],  # y
                quat_y[:, 3] * quat_x[:, 2]
                + quat_y[:, 0] * quat_x[:, 1]
                - quat_y[:, 1] * quat_x[:, 0]
                + quat_y[:, 2] * quat_x[:, 3],  # z
                quat_y[:, 3] * quat_x[:, 3]
                - quat_y[:, 0] * quat_x[:, 0]
                - quat_y[:, 1] * quat_x[:, 1]
                - quat_y[:, 2] * quat_x[:, 2],  # w
            ],
            dim=1,
        )

        hand_env_indices = self.hand_indices[env_ids]
        hand_root_quats = self._sample_hand_root_quats(len(env_ids))
        if hand_root_quats is None:
            hand_root_quats = quats
        self.root_state_tensor[hand_env_indices, 3:7] = hand_root_quats
        self._update_priv_buf(
            env_id=env_ids, name="hand_orientation", value=hand_root_quats
        )

        # position randomization
        pos = self._sample_hand_root_positions(env_ids)
        self.root_state_tensor[hand_env_indices, :3] = pos
        self._update_priv_buf(env_id=env_ids, name="hand_position", value=pos)

        # Object inclination randomization (±5 degrees tilt from vertical Z-axis)
        if self.object_tilt_enabled:
            tilt_angle = (
                torch.rand(len(env_ids), device=self.device) * 5.0 * (np.pi / 180)
            )  # 0-5 degrees in radians
            tilt_direction = torch.rand(len(env_ids), device=self.device) * 2 * np.pi

            # Convert to tilt around X and Y axes
            x_rotation = tilt_angle * torch.sin(tilt_direction)
            y_rotation = tilt_angle * torch.cos(tilt_direction)

            # Create quaternions for small rotations (small angle approximation)
            cos_half_x, sin_half_x = torch.cos(x_rotation / 2), torch.sin(
                x_rotation / 2
            )
            cos_half_y, sin_half_y = torch.cos(y_rotation / 2), torch.sin(
                y_rotation / 2
            )

            quat_x_obj = torch.stack(
                [
                    sin_half_x,
                    torch.zeros_like(sin_half_x),
                    torch.zeros_like(sin_half_x),
                    cos_half_x,
                ],
                dim=1,
            )
            quat_y_obj = torch.stack(
                [
                    torch.zeros_like(sin_half_y),
                    sin_half_y,
                    torch.zeros_like(sin_half_y),
                    cos_half_y,
                ],
                dim=1,
            )

            # Combine rotations by multiplying quaternions (quat_y * quat_x)
            object_quats = torch.stack(
                [
                    quat_y_obj[:, 3] * quat_x_obj[:, 0]
                    + quat_y_obj[:, 0] * quat_x_obj[:, 3]
                    + quat_y_obj[:, 1] * quat_x_obj[:, 2]
                    - quat_y_obj[:, 2] * quat_x_obj[:, 1],  # x
                    quat_y_obj[:, 3] * quat_x_obj[:, 1]
                    - quat_y_obj[:, 0] * quat_x_obj[:, 2]
                    + quat_y_obj[:, 1] * quat_x_obj[:, 3]
                    + quat_y_obj[:, 2] * quat_x_obj[:, 0],  # y
                    quat_y_obj[:, 3] * quat_x_obj[:, 2]
                    + quat_y_obj[:, 0] * quat_x_obj[:, 1]
                    - quat_y_obj[:, 1] * quat_x_obj[:, 0]
                    + quat_y_obj[:, 2] * quat_x_obj[:, 3],  # z
                    quat_y_obj[:, 3] * quat_x_obj[:, 3]
                    - quat_y_obj[:, 0] * quat_x_obj[:, 0]
                    - quat_y_obj[:, 1] * quat_x_obj[:, 1]
                    - quat_y_obj[:, 2] * quat_x_obj[:, 2],  # w
                ],
                dim=1,
            )

            # Apply to object
            object_env_indices = self.object_indices[env_ids]
            self.root_state_tensor[object_env_indices, 3:7] = object_quats
            self._update_priv_buf(
                env_id=env_ids, name="obj_orientation", value=object_quats
            )

            # Compensate Z position for tilt to prevent collision with ground
            shaft_radius = 0.00625  # From URDF: shaft radius = 0.00625m
            z_compensation = shaft_radius * torch.tan(tilt_angle)
            current_z = self.root_state_tensor[object_env_indices, 2]
            self.root_state_tensor[object_env_indices, 2] = current_z + z_compensation

        else:
            # No tilt - set object to perfectly vertical orientation (identity quaternion)
            object_env_indices = self.object_indices[env_ids]
            identity_quats = torch.zeros(
                (len(env_ids), 4), device=self.device, dtype=torch.float
            )
            identity_quats[:, 3] = 1.0  # w = 1, x = y = z = 0 (identity quaternion)

            self.root_state_tensor[object_env_indices, 3:7] = identity_quats
            self._update_priv_buf(
                env_id=env_ids, name="obj_orientation", value=identity_quats
            )

        object_indices = torch.unique(self.object_indices[env_ids]).to(torch.int32)
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_actor_indices = torch.cat([object_indices, hand_indices])
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.root_state_tensor),
            gymtorch.unwrap_tensor(all_actor_indices),
            len(all_actor_indices),
        )
        if not self.torque_control:
            self.gym.set_dof_position_target_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.prev_targets),
                gymtorch.unwrap_tensor(hand_indices),
                len(env_ids),
            )
        self.gym.set_dof_state_tensor_indexed(
            self.sim,
            gymtorch.unwrap_tensor(self.dof_state),
            gymtorch.unwrap_tensor(hand_indices),
            len(env_ids),
        )

        # reset tactile
        self.contact_thresh[env_ids] = 0.05

        self.progress_buf[env_ids] = 0
        self.obs_buf[env_ids] = 0
        self.rb_forces[env_ids] = 0
        self.priv_info_buf[env_ids, 0:3] = 0
        self.proprio_hist_buf[env_ids] = 0
        self.dof_vel_finite_diff[:] = 0
        self.nut_dof_pos_history[env_ids] = 0
        self.nut_contact_history[env_ids] = 0
        self.prev_thumb_drive_w[env_ids] = 0

        self.at_reset_buf[env_ids] = 1

    def compute_observations(self):
        self._refresh_gym()
        # observation noise
        random_obs_noise_t = torch.normal(
            0,
            self.random_obs_noise_t_scale,
            size=self.xhand_hand_dof_pos.shape,
            device=self.device,
            dtype=torch.float,
        )
        noisy_joint_pos = (
            random_obs_noise_t + self.random_obs_noise_e + self.xhand_hand_dof_pos
        )

        t_buf = (
            self.obs_buf_lag_history[:, -3:, : self.obs_buf.shape[1] // 3].reshape(
                self.num_envs, -1
            )
        ).clone()
        self.obs_buf[:, : t_buf.shape[1]] = t_buf  # [1, 96]

        # deal with normal observation, do sliding windows
        prev_obs_buf = self.obs_buf_lag_history[:, 1:].clone()
        cur_obs_buf = noisy_joint_pos.clone().unsqueeze(1)  # xhand dim [1, 1, 12]
        cur_tar_buf = self.cur_targets[:, None, : self.num_actions]  # [1, 1, 12]
        cur_obs_buf = torch.cat([cur_obs_buf, cur_tar_buf], dim=-1)  # [1, 1, 24]

        self.obs_buf_lag_history[:] = torch.cat(
            [prev_obs_buf, cur_obs_buf], dim=1
        )  # torch.Size([48, 80, 24])

        # refill the initialized buffers
        at_reset_env_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
        self.obs_buf_lag_history[at_reset_env_ids, :, 0 : self.numActions] = (
            self.init_pose_buf[at_reset_env_ids, : self.num_actions].unsqueeze(1)
        )
        self.obs_buf_lag_history[
            at_reset_env_ids, :, self.numActions : self.numActions * 2
        ] = self.init_pose_buf[at_reset_env_ids, : self.num_actions].unsqueeze(1)

        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]
        if len(at_reset_env_ids) > 0:
            self.nut_dof_vel_cf[at_reset_env_ids] = self.nut_dof_vel[at_reset_env_ids]

        self.at_reset_buf[at_reset_env_ids] = 0
        rand_rpy = torch.normal(
            0,
            self.noisy_rpy_scale,
            size=(self.num_envs, 3),
            device=self.device,
            dtype=torch.float,
        )
        rand_quat = quat_from_euler_xyz(rand_rpy[:, 0], rand_rpy[:, 1], rand_rpy[:, 2])
        _noisy_quat = quat_mul(rand_quat, self.object_rot)
        _noisy_position = (
            torch.normal(
                0,
                self.noisy_pos_scale,
                size=(self.num_envs, 3),
                device=self.device,
                dtype=torch.float,
            )
            + self.object_pos
        )

        # Update nut history buffers for termination conditions
        prev_nut_dof_pos_history = self.nut_dof_pos_history[:, 1:].clone()
        cur_nut_dof_pos = self.nut_dof_pos.clone().unsqueeze(1)
        self.nut_dof_pos_history[:] = torch.cat(
            [prev_nut_dof_pos_history, cur_nut_dof_pos], dim=1
        )

        prev_nut_contact_history = self.nut_contact_history[:, 1:].clone()
        cur_nut_contact = self.nut_contact.clone().unsqueeze(1)
        self.nut_contact_history[:] = torch.cat(
            [prev_nut_contact_history, cur_nut_contact], dim=1
        )

        if len(at_reset_env_ids) > 0:
            self.nut_dof_pos_history[at_reset_env_ids] = (
                self.nut_dof_pos[at_reset_env_ids]
                .unsqueeze(1)
                .repeat(1, self.nut_termination_history_len, 1)
            )
            self.nut_contact_history[at_reset_env_ids] = (
                self.nut_contact[at_reset_env_ids]
                .unsqueeze(1)
                .repeat(1, self.nut_termination_history_len, 1)
            )

        self.proprio_hist_buf[:] = self.obs_buf_lag_history[
            :, -self.prop_hist_len :, : self.numActions * 2
        ]  # [1, 30, 32]
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="obj_position",
            value=self.object_pos.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="obj_orientation",
            value=self.object_rot.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="obj_linvel",
            value=self.obj_linvel_at_cf.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="fingertip_orientation",
            value=self.fingertip_orientation.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="fingertip_linvel",
            value=self.ft_linvel_at_cf.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="fingertip_angvel",
            value=self.ft_angvel_at_cf.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs), name="nut_pos", value=self.nut_pos.clone()
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="nut_dof_pos",
            value=self.nut_dof_pos.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="nut_dof_vel",
            value=self.nut_dof_vel_cf.clone(),
        )
        self._update_priv_buf(
            env_id=range(self.num_envs),
            name="fingertip_position",
            value=self.fingertip_pos.clone(),
        )

        if self.point_cloud_sampled_dim > 0:
            # for collecting bc data
            self.point_cloud_buf[:, : self.point_cloud_sampled_dim] = (
                quat_apply(
                    self.object_rot[:, None].repeat(1, self.point_cloud_sampled_dim, 1),
                    self.obj_point_clouds,
                )
                + self.object_pos[:, None]
            )  # [1, 100, 3]

    def _get_reward_scale_by_name(self, name):
        env_steps = self.gym.get_frame_count(self.sim) * len(self.envs)
        agent_steps = env_steps // self.control_freq_inv
        init_scale, final_scale, curr_start, curr_end = self.reward_scale_dict[name]
        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to [0, 0.05, 1.0] instead of continuous value
            # during batch collection, avoid reward confusion
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1
        if self.evaluate:
            curr_progress = 1
        return init_scale + (final_scale - init_scale) * curr_progress

    def _get_current_angvel_penalty_threshold(self):
        """Get the current angular velocity penalty threshold based on curriculum progress."""
        env_steps = self.gym.get_frame_count(self.sim) * len(self.envs)
        agent_steps = env_steps // self.control_freq_inv
        init_threshold, final_threshold, curr_start, curr_end = (
            self.angvel_penalty_threshold_curriculum
        )

        if curr_end > 0:
            curr_progress = (agent_steps - curr_start) / (curr_end - curr_start)
            curr_progress = min(max(curr_progress, 0), 1)
            # discretize to avoid confusion during batch collection
            curr_progress = round(curr_progress * 20) / 20
        else:
            curr_progress = 1

        if self.evaluate:
            curr_progress = 1

        current_threshold = (
            init_threshold + (final_threshold - init_threshold) * curr_progress
        )
        return current_threshold

    def compute_reward(self, actions):
        # Update current angular velocity penalty threshold based on curriculum
        current_angvel_penalty_threshold = self._get_current_angvel_penalty_threshold()

        pose_diff_sq = (
            self.xhand_hand_dof_pos - self.init_pose_buf[..., : self.num_actions]
        ) ** 2
        pose_diff_penalty = (pose_diff_sq * self.pose_diff_penalty_mask).sum(-1)
        thumb_pose_diff_penalty = torch.zeros_like(pose_diff_penalty)
        if self.pose_diff_penalty_thumb_indices.numel() > 0:
            thumb_pose_diff_penalty = pose_diff_sq.index_select(
                -1, self.pose_diff_penalty_thumb_indices
            ).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques[:, -1, : self.num_actions] ** 2).sum(-1)
        work_penalty = (
            (
                torch.abs(self.torques[:, -1, : self.num_actions])
                * torch.abs(self.dof_vel_finite_diff[:, -1])
            ).sum(-1)
        ) ** 2
        if self.normalize_penalties_by_num_actions:
            denom = float(max(int(self.num_actions), 1))
            pose_diff_penalty = pose_diff_penalty / denom
            thumb_pose_diff_penalty = thumb_pose_diff_penalty / denom
            torque_penalty = torque_penalty / denom
            work_penalty = work_penalty / (denom * denom)

        angdiff = self.quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        )
        object_angvel = angdiff / (self.control_freq_inv * self.dt)

        # Calculate proximity reward
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3]
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        thumb_pos = fingertip_pos[:, self.finger_contact_thumb_index, :]
        other_pos = fingertip_pos[:, self.finger_contact_other_indices, :]
        finger_target_pos = self._get_finger_contact_target_pos(
            nut_pos=nut_pos,
            nut_rot=nut_states[..., 3:7],
        )
        finger_threshold = self._get_finger_contact_threshold()
        thumb_dist = torch.norm(thumb_pos - finger_target_pos, dim=-1)
        other_dist_all = torch.norm(
            other_pos - finger_target_pos.unsqueeze(1), dim=-1
        )
        other_dist = other_dist_all.mean(dim=-1)
        mean_dist = 0.5 * (thumb_dist + other_dist)
        ratio = mean_dist / finger_threshold
        proximity_reward = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

        nut_dof_linvel = (
            (self.nut_dof_pos.squeeze(-1) - self.nut_dof_pos_prev.squeeze(-1))
            / (self.control_freq_inv * self.dt)
        ).squeeze(-1)
        self.nut_dof_vel_cf = nut_dof_linvel.unsqueeze(-1)
        rotate_reward_raw = torch.clip(
            nut_dof_linvel, max=self.angvel_clip_max, min=self.angvel_clip_min
        )
        rotate_reward = rotate_reward_raw
        two_finger_extra = torch.zeros_like(rotate_reward)
        gate_stats = None
        if self.two_finger_gate_enable:
            rotate_reward, two_finger_extra, gate_stats = self._apply_two_finger_gate(
                rotate_reward_raw=rotate_reward_raw,
                nut_dof_linvel=nut_dof_linvel,
                nut_pos=nut_pos,
            )

        rotate_penalty_raw = torch.where(
            nut_dof_linvel > current_angvel_penalty_threshold,
            nut_dof_linvel - current_angvel_penalty_threshold,
            0,
        )
        reverse_penalty = torch.where(
            nut_dof_linvel < 0,
            torch.abs(nut_dof_linvel) * 2.0,
            torch.zeros_like(nut_dof_linvel),
        )
        rotate_penalty = rotate_penalty_raw

        object_linvel = (
            (self.object_pos - self.object_pos_prev) / (self.control_freq_inv * self.dt)
        ).clone()
        self.obj_angvel_at_cf = object_angvel
        self.obj_linvel_at_cf = object_linvel
        ft_angdiff = self.quat_to_axis_angle(
            quat_mul(
                self.fingertip_orientation.reshape(-1, 4),
                quat_conjugate(self.ft_rot_prev.reshape(-1, 4)),
            )
        ).reshape(-1, self.fingers_num * 3)
        self.ft_angvel_at_cf = ft_angdiff / (self.control_freq_inv * self.dt)
        self.ft_linvel_at_cf = (self.fingertip_pos - self.ft_pos_prev) / (
            self.control_freq_inv * self.dt
        )
        fingertip_tangent_reward = torch.zeros_like(rotate_reward)
        fingertip_tangent_stats = None
        if self.fingertip_tangent_reward_enable:
            fingertip_tangent_reward, fingertip_tangent_stats = (
                self._compute_fingertip_tangent_reward(nut_pos)
            )
        fingertip_torque_reward = torch.zeros_like(rotate_reward)
        fingertip_torque_stats = None
        if self.fingertip_torque_reward_enable:
            fingertip_torque_reward, fingertip_torque_stats = (
                self._compute_fingertip_torque_reward(nut_pos)
            )
        active_contact_penalty = torch.zeros_like(rotate_reward)
        active_contact_stats = None
        if self.active_two_finger_contact_enable:
            active_contact_penalty, active_contact_stats = (
                self._compute_active_two_finger_contact_penalty(nut_dof_linvel)
            )
        opposition_grip_reward = torch.zeros_like(rotate_reward)
        opposition_grip_stats = None
        if self.opposition_grip_reward_enable:
            opposition_grip_reward, opposition_grip_stats = (
                self._compute_opposition_grip_reward(nut_pos)
            )
        thumb_slip_penalty = torch.zeros_like(rotate_reward)
        thumb_slip_penalty_stats = None
        if self.thumb_slip_penalty_enable:
            thumb_slip_penalty, thumb_slip_penalty_stats = (
                self._compute_thumb_slip_penalty(
                    thumb_dist=thumb_dist,
                    nut_dof_linvel=nut_dof_linvel,
                    nut_pos=nut_pos,
                )
            )
        finger_diag_stats = self._compute_finger_diagnostics(nut_pos)
        self.z_dist_penalty = (self.object_pos[:, 2] - self.object_z) ** 2

        if self.point_cloud_sampled_dim > 0:
            point_cloud_z = self.point_cloud_buf[:, : self.point_cloud_sampled_dim, -1]
            z_dist_penalty = point_cloud_z.max(axis=1)[0] - point_cloud_z.min(axis=1)[0]
            z_dist_penalty[z_dist_penalty <= 0.03] = 0
        else:
            z_dist_penalty = to_torch([0], device=self.device)

        self.rew_buf[:] = compute_hand_reward(
            rotate_reward,
            self._get_reward_scale_by_name("rotate_reward"),
            pose_diff_penalty,
            self._get_reward_scale_by_name("pose_diff_penalty"),
            torque_penalty,
            self._get_reward_scale_by_name("torque_penalty"),
            work_penalty,
            self._get_reward_scale_by_name("work_penalty"),
            z_dist_penalty,
            self._get_reward_scale_by_name("pc_z_dist_penalty"),
            rotate_penalty,
            self._get_reward_scale_by_name("rotate_penalty"),
            proximity_reward,
            self._get_reward_scale_by_name("proximity_reward"),
        )
        self.rew_buf[:] = self.rew_buf + two_finger_extra
        if self.fingertip_tangent_reward_enable:
            self.rew_buf[:] = self.rew_buf + fingertip_tangent_reward * (
                self._get_reward_scale_by_name("fingertip_tangent_reward")
            )
        if self.fingertip_torque_reward_enable:
            self.rew_buf[:] = self.rew_buf + fingertip_torque_reward * (
                self._get_reward_scale_by_name("fingertip_torque_reward")
            )
        if self.active_two_finger_contact_enable:
            self.rew_buf[:] = self.rew_buf + active_contact_penalty
        if self.opposition_grip_reward_enable:
            self.rew_buf[:] = self.rew_buf + (
                opposition_grip_reward * self.opposition_grip_reward_scale
            )
        if self.thumb_slip_penalty_enable:
            self.rew_buf[:] = self.rew_buf + thumb_slip_penalty

        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras["step_all_reward"] = self.rew_buf.mean()
        self.extras["rotation_reward"] = rotate_reward.mean()
        self.extras["pose_diff_penalty"] = pose_diff_penalty.mean()
        self.extras["work_done"] = work_penalty.mean()
        self.extras["torques"] = torque_penalty.mean()
        self.extras["roll"] = torch.abs(object_angvel[:, 0]).mean()
        self.extras["pitch"] = torch.abs(object_angvel[:, 1]).mean()
        self.extras["yaw"] = torch.abs(object_angvel[:, 2]).mean()
        self.extras["z_dist_penalty"] = z_dist_penalty.mean()
        self.extras["pose_diff_penalty/thumb_raw"] = thumb_pose_diff_penalty.mean()
        self.extras["pose_diff_penalty/thumb_weighted"] = (
            thumb_pose_diff_penalty * self.pose_diff_penalty_thumb_weight
        ).mean()
        self.extras["rotate_penalty"] = rotate_penalty.mean()

        # curriculum tracking
        self.extras["curriculum/angvel_penalty_threshold"] = (
            current_angvel_penalty_threshold
        )

        # screw-specific metrics
        self.extras["screw/angular_velocity"] = self.nut_dof_vel.mean()
        self.extras["screw/angular_position"] = self.nut_dof_pos.mean()
        self.extras["screw/positive_vel_ratio"] = (self.nut_dof_vel > 0).float().mean()
        if gate_stats is not None:
            self.extras["two_finger/gate"] = gate_stats["gate"].mean()
            self.extras["two_finger/thumb_contact_w"] = gate_stats[
                "thumb_weight"
            ].mean()
            self.extras["two_finger/other_contact_w"] = gate_stats[
                "other_weight"
            ].mean()
            self.extras["two_finger/other_mean_w"] = gate_stats[
                "other_mean_weight"
            ].mean()
            self.extras["two_finger/other_min_w"] = gate_stats[
                "other_min_weight"
            ].mean()
            self.extras["two_finger/thumb_dist"] = gate_stats["thumb_dist"].mean()
            self.extras["two_finger/other_dist"] = gate_stats["other_dist"].mean()
            self.extras["two_finger/other_mean_dist"] = gate_stats[
                "other_mean_dist"
            ].mean()
            self.extras["two_finger/other_max_dist"] = gate_stats[
                "other_max_dist"
            ].mean()
            self.extras["two_finger/extra_penalty"] = two_finger_extra.mean()
        if fingertip_tangent_stats is not None:
            self.extras["fingertip_tangent/reward"] = (
                fingertip_tangent_reward.mean()
            )
            self.extras["fingertip_tangent/tangent_vel"] = (
                fingertip_tangent_stats["tangent_vel"].mean()
            )
            self.extras["fingertip_tangent/positive_vel"] = (
                fingertip_tangent_stats["positive_tangent_vel"].mean()
            )
            self.extras["fingertip_tangent/dist_w"] = fingertip_tangent_stats[
                "dist_weight"
            ].mean()
            self.extras["fingertip_tangent/contact_w"] = fingertip_tangent_stats[
                "contact_weight"
            ].mean()
            self.extras["fingertip_tangent/reward_min"] = (
                fingertip_tangent_stats["reward_all"].min(dim=-1).values.mean()
            )
        if fingertip_torque_stats is not None:
            self.extras["fingertip_torque/reward"] = (
                fingertip_torque_reward.mean()
            )
            self.extras["fingertip_torque/signed_torque"] = (
                fingertip_torque_stats["signed_torque"].mean()
            )
            self.extras["fingertip_torque/positive_torque"] = (
                fingertip_torque_stats["positive_torque"].mean()
            )
            self.extras["fingertip_torque/negative_torque"] = (
                fingertip_torque_stats["negative_torque"].mean()
            )
            self.extras["fingertip_torque/abs_torque"] = fingertip_torque_stats[
                "abs_torque"
            ].mean()
            self.extras["fingertip_torque/dist_w"] = fingertip_torque_stats[
                "dist_weight"
            ].mean()
            self.extras["fingertip_torque/contact_w"] = fingertip_torque_stats[
                "contact_weight"
            ].mean()
            self.extras["fingertip_torque/reward_min"] = (
                fingertip_torque_stats["reward_all"].min(dim=-1).values.mean()
            )
        if active_contact_stats is not None:
            self.extras["active_two_finger/penalty"] = active_contact_stats[
                "penalty"
            ].mean()
            self.extras["active_two_finger/active_frac"] = active_contact_stats[
                "active_frac"
            ].mean()
            self.extras["active_two_finger/pair_contact_w"] = active_contact_stats[
                "pair_contact_w"
            ].mean()
            self.extras["active_two_finger/thumb_contact_w"] = active_contact_stats[
                "thumb_contact_w"
            ].mean()
            self.extras["active_two_finger/other_contact_w"] = active_contact_stats[
                "other_contact_w"
            ].mean()
        if opposition_grip_stats is not None:
            self.extras["opposition_grip/reward"] = opposition_grip_reward.mean()
            self.extras["opposition_grip/reward_scaled"] = (
                opposition_grip_reward * self.opposition_grip_reward_scale
            ).mean()
            self.extras["opposition_grip/oppositeness"] = opposition_grip_stats[
                "oppositeness"
            ].mean()
            self.extras["opposition_grip/radial_dot"] = opposition_grip_stats[
                "radial_dot"
            ].mean()
            self.extras["opposition_grip/pair_dist_w"] = opposition_grip_stats[
                "pair_dist_w"
            ].mean()
            self.extras["opposition_grip/pair_inward_w"] = opposition_grip_stats[
                "pair_inward_w"
            ].mean()
            self.extras["opposition_grip/thumb_inward_force"] = opposition_grip_stats[
                "thumb_inward_force"
            ].mean()
            self.extras["opposition_grip/other_inward_force"] = opposition_grip_stats[
                "other_inward_force"
            ].mean()
        if thumb_slip_penalty_stats is not None:
            self.extras["thumb_slip_penalty/penalty"] = thumb_slip_penalty.mean()
            self.extras["thumb_slip_penalty/contact_loss"] = (
                thumb_slip_penalty_stats["contact_loss"].mean()
            )
            self.extras["thumb_slip_penalty/far_loss"] = thumb_slip_penalty_stats[
                "far_loss"
            ].mean()
            self.extras["thumb_slip_penalty/ejection_loss"] = (
                thumb_slip_penalty_stats["ejection_loss"].mean()
            )
            self.extras["thumb_slip_penalty/after_drive_loss"] = (
                thumb_slip_penalty_stats["after_drive_loss"].mean()
            )
            self.extras["thumb_slip_penalty/terminal_ease_loss"] = (
                thumb_slip_penalty_stats["terminal_ease_loss"].mean()
            )
            self.extras["thumb_slip_penalty/return_contact_loss"] = (
                thumb_slip_penalty_stats["return_contact_loss"].mean()
            )
            self.extras["thumb_slip_penalty/return_phase_w"] = (
                thumb_slip_penalty_stats["return_phase_w"].mean()
            )
            self.extras["thumb_slip_penalty/return_context_w"] = (
                thumb_slip_penalty_stats["return_context_w"].mean()
            )
            self.extras["thumb_slip_penalty/thumb_tangent_vel"] = (
                thumb_slip_penalty_stats["thumb_tangent_vel"].mean()
            )
            self.extras["thumb_slip_penalty/thumb_contact_w"] = (
                thumb_slip_penalty_stats["thumb_contact_w"].mean()
            )
            self.extras["thumb_slip_penalty/thumb_dist"] = (
                thumb_slip_penalty_stats["thumb_dist"].mean()
            )
            self.extras["thumb_slip_penalty/thumb_tip_speed"] = (
                thumb_slip_penalty_stats["thumb_tip_speed"].mean()
            )
            self.extras["thumb_slip_penalty/active_frac"] = (
                thumb_slip_penalty_stats["active"].mean()
            )
            self.extras["thumb_slip_penalty/velocity_weight"] = (
                thumb_slip_penalty_stats["velocity_weight"].mean()
            )
            self.extras["thumb_slip_penalty/speed_weight"] = (
                thumb_slip_penalty_stats["speed_weight"].mean()
            )
            self.extras["thumb_slip_penalty/prev_drive_w"] = (
                thumb_slip_penalty_stats["prev_drive_w"].mean()
            )
            self.extras["thumb_slip_penalty/current_drive_w"] = (
                thumb_slip_penalty_stats["current_drive_w"].mean()
            )
        self._write_finger_diagnostic_extras(finger_diag_stats)
        if self.termination_log:
            reasons = getattr(self, "_last_termination_reasons", None)
            if isinstance(reasons, dict):
                for k, v in reasons.items():
                    self.extras[f"term/{k}_frac"] = v.float().mean()
            self.extras["term/any_reset_frac"] = self.reset_buf.float().mean()

        if self.evaluate:
            vec_dot = (object_angvel * self.rot_axis_buf).sum(-1)
            for i in range(len(self.object_type_list)):
                env_ids = torch.where(self.object_type_at_env == i)
                if len(env_ids[0]) > 0:
                    running_mask = 1 - self.eval_done_buf[env_ids]
                    self.stat_sum_rewards[i] += (
                        running_mask * self.rew_buf[env_ids]
                    ).sum()
                    self.stat_sum_episode_length[i] += running_mask.sum()
                    self.stat_sum_rotate_rewards[i] += (
                        running_mask * rotate_reward[env_ids]
                    ).sum()
                    self.stat_sum_unclip_rotate_rewards[i] += (
                        running_mask * vec_dot[env_ids]
                    ).sum()

                    if self.config["env"]["object"]["evalObjectType"] is not None:
                        flip = running_mask * self.reset_buf[env_ids]
                        self.env_evaluated[i] += flip.sum()
                        self.eval_done_buf[env_ids] += flip

                    info = f"Progress: {self.evaluate_iter} / {self.max_episode_length}"
                    tprint(info)
            self.evaluate_iter += 1

    def post_physics_step(self):
        self.progress_buf += 1
        self.reset_buf[:] = 0
        self._refresh_gym()
        self.compute_reward(self.actions)

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        self.compute_observations()

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def pre_physics_step(self, actions):
        self.actions = actions.clone().to(self.device)

        targets = self.prev_targets + self.action_scale * self.actions
        self.cur_targets[:, : self.num_xhand_hand_dofs] = tensor_clamp(
            targets[:, : self.num_xhand_hand_dofs],
            self.xhand_hand_dof_lower_limits,
            self.xhand_hand_dof_upper_limits,
        )

        # get prev* buffer here
        self.prev_targets[:] = self.cur_targets
        self.object_rot_prev[:] = self.object_rot
        self.object_pos_prev[:] = self.object_pos
        self.nut_dof_pos_prev[:] = self.nut_dof_pos
        self.ft_rot_prev[:] = self.fingertip_orientation
        self.ft_pos_prev[:] = self.fingertip_pos
        self.dof_vel_prev[:] = self.dof_vel_finite_diff

    def reset(self):
        super().reset()
        self.obs_dict["priv_info"] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict["point_cloud_info"] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict["rot_axis_buf"] = self.rot_axis_buf.to(self.rl_device)
        return self.obs_dict

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        action_mask = torch.ones_like(actions)
        if self.apply_action_mask:
            if self.custom_action_mask_indices is not None:
                mask_indices = [int(i) for i in list(self.custom_action_mask_indices)]
                action_mask[:, mask_indices] = 0.0
            elif self.config["env"]["initPose"] == "screwdriver_inclined":
                action_mask[:, 5:7] = 0.0
            else:
                action_mask[:, 5:9] = 0.0  # mask out pinky, and ring finger actions
        actions = actions * action_mask
        actions = F.pad(
            actions, (0, 1), value=0.0
        )  # pad the last dim with 0.0 for the nut joint

        if (
            extrin_record is not None
            and self.config["env"]["object"]["evalObjectType"] is not None
        ):
            # Put a (z vectors, is done) tuple into the log.
            self.extrin_log.append(
                (
                    extrin_record.detach().cpu().numpy().copy(),
                    self.eval_done_buf.detach().cpu().numpy().copy(),
                )
            )

        self.pre_state = self.xhand_hand_dof_pos[0]
        super().step(actions)
        self.obs_dict["priv_info"] = self.priv_info_buf.to(self.rl_device)
        # stage 2 buffer
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict["point_cloud_info"] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict["rot_axis_buf"] = self.rot_axis_buf.to(self.rl_device)
        return self.obs_dict, self.rew_buf, self.reset_buf, self.extras

    def capture_frame(self) -> np.ndarray:
        assert self.enable_camera_sensors  # camera sensors should be enabled
        assert self.vid_record_tensor is not None
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.start_access_image_tensors(self.sim)
        frame = self.vid_record_tensor.cpu().numpy()
        self.gym.end_access_image_tensors(self.sim)

        return frame

    def update_low_level_control(self, step_id):
        # Refresh DOF state first; on startup the wrapped tensors may contain
        # uninitialized values. Using them before a refresh can inject NaNs into torques.
        self.gym.refresh_dof_state_tensor(self.sim)
        random_action_noise_t = torch.normal(
            0,
            self.random_action_noise_t_scale,
            size=self.xhand_hand_dof_pos.shape,
            device=self.device,
            dtype=torch.float,
        )
        noise_action = (
            self.cur_targets[..., : self.num_xhand_hand_dofs]
            + self.random_action_noise_e
            + random_action_noise_t
        )

        if self.torque_control:
            dof_pos = self.xhand_hand_dof_pos
            # Prefer simulator-provided velocities (finite and consistent).
            dof_vel = self.xhand_hand_dof_vel
            self.dof_vel_finite_diff[:, step_id] = dof_vel.clone()
            torques = self.p_gain * (noise_action - dof_pos) - self.d_gain * dof_vel
            torques = torch.clip(torques, -self.torque_limit, self.torque_limit).clone()
            self.torques[:, step_id, : self.num_xhand_hand_dofs] = torques.clone()
            torques = self.torques[:, step_id, :].clone()
            self.gym.set_dof_actuation_force_tensor(
                self.sim, gymtorch.unwrap_tensor(torques)
            )
        else:
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(noise_action)
            )

    def update_rigid_body_force(self):
        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(
                self.force_decay, self.dt / self.force_decay_interval
            )
            # apply new forces
            obj_mass = [
                self.gym.get_actor_rigid_body_properties(
                    env, self.gym.find_actor_handle(env, "object")
                )[0].mass
                for env in self.envs
            ]
            obj_mass = to_torch(obj_mass, device=self.device)
            prob = self.random_force_prob_scalar
            force_indices = (
                torch.less(torch.rand(self.num_envs, device=self.device), prob)
            ).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = (
                torch.randn(
                    self.rb_forces[force_indices, self.object_rb_handles, :].shape,
                    device=self.device,
                )
                * obj_mass[force_indices, None]
                * self.force_scale
            )
            self.gym.apply_rigid_body_force_tensors(
                self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.ENV_SPACE
            )

    def check_termination(self, object_pos):
        term_by_max_eps = torch.greater_equal(
            self.progress_buf, self.max_episode_length
        )
        resets = term_by_max_eps
        grace_ready = self.progress_buf >= self.termination_grace_steps

        # finger nut distance check
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3]

        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        thumb_pos = fingertip_pos[:, self.finger_contact_thumb_index, :]
        other_pos = fingertip_pos[:, self.finger_contact_other_indices, :]
        finger_target_pos = self._get_finger_contact_target_pos(
            nut_pos=nut_pos,
            nut_rot=nut_states[..., 3:7],
        )
        finger_threshold = self._get_finger_contact_threshold()
        thumb_dist = torch.norm(thumb_pos - finger_target_pos, dim=-1)
        other_dist_all = torch.norm(
            other_pos - finger_target_pos.unsqueeze(1), dim=-1
        )
        # Reset logging
        tracked_dists = torch.cat(
            [thumb_dist.unsqueeze(-1), other_dist_all], dim=-1
        )
        finger_dist_condition = torch.any(
            tracked_dists > finger_threshold.unsqueeze(-1), dim=-1
        )
        finger_dist_reset = torch.zeros_like(resets)
        if self.termination_enable_finger_dist:
            finger_dist_reset = grace_ready & finger_dist_condition
        resets = torch.logical_or(resets, finger_dist_reset)

        # nut constant pos check - terminate if nut position is similar over 10 timesteps
        nut_pos_history_filled = self.progress_buf >= self.nut_termination_history_len
        nut_pos_variance = torch.var(self.nut_dof_pos_history, dim=1).squeeze(
            -1
        )  # (num_envs,)
        nut_pos_stagnant_condition = (
            nut_pos_variance < self.nut_stagnation_eps
        ) & nut_pos_history_filled
        nut_pos_stagnant = torch.zeros_like(resets)
        if self.termination_enable_nut_stagnation:
            nut_pos_stagnant = grace_ready & nut_pos_stagnant_condition
        resets = torch.logical_or(resets, nut_pos_stagnant)

        # nut contact check - terminate if nut has 0 contact force over 10 timesteps
        contact_history_filled = self.progress_buf >= self.nut_termination_history_len
        no_contact = torch.all(self.nut_contact_history <= 1e-3, dim=1).squeeze(-1)
        no_contact_condition = no_contact & contact_history_filled
        no_contact_reset = torch.zeros_like(resets)
        if self.termination_enable_no_contact:
            no_contact_reset = grace_ready & no_contact_condition
        resets = torch.logical_or(resets, no_contact_reset)

        # screw joint limit check for automatic reset
        screw_at_limit = torch.zeros_like(resets)
        if hasattr(self, "dof_state"):
            current_screw_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
                :, self.num_xhand_hand_dofs :
            ]
            current_screw_pos = current_screw_dof_state[:, 0, 0]
            screw_upper_limit = 628.3185
            reset_threshold = (
                screw_upper_limit - 5.0
            )  # Reset when within 5 radians of limit

            screw_at_limit_condition = current_screw_pos > reset_threshold
            if self.termination_enable_screw_limit:
                screw_at_limit = grace_ready & screw_at_limit_condition
            resets = torch.logical_or(resets, screw_at_limit)
        if self.termination_log:
            self._last_termination_reasons = {
                "max_eps": term_by_max_eps,
                "finger_dist": finger_dist_reset,
                "nut_stagnant": nut_pos_stagnant,
                "no_contact": no_contact_reset,
                "screw_limit": screw_at_limit,
            }
        return resets

    def _refresh_gym(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        self.fingertip_states = self.rigid_body_states[:, self.fingertip_handles]
        self.fingertip_pos = self.fingertip_states[:, :, :3].reshape(self.num_envs, -1)
        self.fingertip_orientation = self.fingertip_states[:, :, 3:7].reshape(
            self.num_envs, -1
        )
        self.fingertip_linvel = self.fingertip_states[:, :, 7:10].reshape(
            self.num_envs, -1
        )
        self.fingertip_angvel = self.fingertip_states[:, :, 10:13].reshape(
            self.num_envs, -1
        )
        self.nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        self.nut_pos = self.nut_states[:, :3]
        all_contact_forces = torch.norm(self.contact_forces.clone(), dim=-1)
        self.nut_contact = all_contact_forces[:, self.screw_nut_rb_handle].unsqueeze(-1)
        self.nut_dof_state = self.dof_state.view(self.num_envs, -1, 2)[
            :, self.num_xhand_hand_dofs :
        ]
        self.nut_dof_vel = self.nut_dof_state[:, 0, 1].unsqueeze(-1)
        self.nut_dof_pos = self.nut_dof_state[:, 0, 0].unsqueeze(-1)

    def _setup_termination_config(self, termination_cfg):
        if termination_cfg is None:
            termination_cfg = {}
        self.termination_grace_steps = int(termination_cfg.get("grace_steps", 0))
        self.termination_enable_finger_dist = self._cfg_bool(
            termination_cfg.get("enable_finger_dist", True), default=True
        )
        self.termination_enable_nut_stagnation = self._cfg_bool(
            termination_cfg.get("enable_nut_stagnation", True), default=True
        )
        self.termination_enable_no_contact = self._cfg_bool(
            termination_cfg.get("enable_no_contact", True), default=True
        )
        self.termination_enable_screw_limit = self._cfg_bool(
            termination_cfg.get("enable_screw_limit", True), default=True
        )
        self.termination_log = self._cfg_bool(
            termination_cfg.get("log", False), default=False
        )
        self._last_termination_reasons = None

    def _setup_finger_object_contact_config(self, contact_cfg):
        if contact_cfg is None:
            contact_cfg = {}
        self.finger_contact_thumb_index = int(
            contact_cfg.get("thumb_fingertip_index", max(self.fingers_num - 1, 0))
        )
        raw_other_indices = contact_cfg.get("other_fingertip_indices", [0])
        if isinstance(raw_other_indices, (int, float)):
            raw_other_indices = [raw_other_indices]
        self.finger_contact_other_indices = [
            int(i) for i in list(raw_other_indices)
        ]
        if not (0 <= self.finger_contact_thumb_index < self.fingers_num):
            raise ValueError(
                "env.finger_object_contact.thumb_fingertip_index is out of range "
                f"for {self.fingers_num} fingertips: {self.finger_contact_thumb_index}"
            )
        if len(self.finger_contact_other_indices) == 0:
            raise ValueError(
                "env.finger_object_contact.other_fingertip_indices must not be empty"
            )
        for idx in self.finger_contact_other_indices:
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    "env.finger_object_contact.other_fingertip_indices contains an "
                    f"out-of-range index for {self.fingers_num} fingertips: {idx}"
                )
        self.finger_contact_target = str(contact_cfg.get("target", "nut_pos"))
        if self.finger_contact_target not in {"nut_pos", "object_pos"}:
            raise ValueError(
                "env.finger_object_contact.target must be 'nut_pos' or "
                f"'object_pos'; got {self.finger_contact_target}"
            )
        self.finger_contact_target_offset = self._resolve_numeric_vector(
            contact_cfg.get("target_offset", [0.0, 0.0, 0.0]),
            3,
            "env.finger_object_contact.target_offset",
        )
        self.finger_contact_scale_with_object = self._cfg_bool(
            contact_cfg.get("scale_with_object", False), default=False
        )
        self.finger_contact_threshold_scale_with_object = self._cfg_bool(
            contact_cfg.get("threshold_scale_with_object", False), default=False
        )

    def _setup_two_finger_gate_config(self, gate_cfg):
        if gate_cfg is None:
            gate_cfg = {}
        self.two_finger_gate_enable = self._cfg_bool(
            gate_cfg.get("enable", False), default=False
        )
        self.two_finger_gate_thumb_index = int(
            gate_cfg.get("thumb_fingertip_index", max(self.fingers_num - 1, 0))
        )
        raw_other_indices = gate_cfg.get("other_fingertip_indices", [0, 1])
        if isinstance(raw_other_indices, (int, float)):
            raw_other_indices = [raw_other_indices]
        self.two_finger_gate_other_indices = [int(i) for i in list(raw_other_indices)]
        self.two_finger_gate_other_aggregation = str(
            gate_cfg.get("other_aggregation", "max")
        )
        self.two_finger_gate_other_mean_weight = float(
            gate_cfg.get("other_mean_weight", 0.5)
        )
        self.two_finger_gate_other_min_weight = float(
            gate_cfg.get("other_min_weight", 0.5)
        )
        self.two_finger_gate_target = str(gate_cfg.get("target", "nut_pos"))
        self.two_finger_gate_target_offset = self._resolve_numeric_vector(
            gate_cfg.get("target_offset", [0.0, 0.0, 0.0]),
            3,
            "env.two_finger_gate.target_offset",
        )
        self.two_finger_gate_near = float(gate_cfg.get("near", 0.08))
        self.two_finger_gate_far = float(gate_cfg.get("far", 0.13))
        self.two_finger_gate_min_mult = float(gate_cfg.get("min_mult", 0.2))
        self.two_finger_gate_power = float(gate_cfg.get("power", 1.0))
        self.two_finger_gate_scale_with_object = self._cfg_bool(
            gate_cfg.get("scale_with_object", False), default=False
        )
        self.two_finger_gate_apply_positive_vel_only = self._cfg_bool(
            gate_cfg.get("apply_positive_vel_only", True), default=True
        )
        self.two_finger_gate_use_contact_force = self._cfg_bool(
            gate_cfg.get("use_contact_force", False), default=False
        )
        self.two_finger_gate_contact_force_min = float(
            gate_cfg.get("contact_force_min", 0.5)
        )
        self.two_finger_gate_contact_force_max = float(
            gate_cfg.get("contact_force_max", 2.0)
        )
        self.two_finger_gate_no_grasp_penalty_scale = float(
            gate_cfg.get("no_grasp_penalty_scale", 0.0)
        )
        if not self.two_finger_gate_enable:
            return
        if not (0 <= self.two_finger_gate_thumb_index < self.fingers_num):
            raise ValueError(
                "env.two_finger_gate.thumb_fingertip_index is out of range "
                f"for {self.fingers_num} fingertips: {self.two_finger_gate_thumb_index}"
            )
        if len(self.two_finger_gate_other_indices) == 0:
            raise ValueError(
                "env.two_finger_gate.other_fingertip_indices must not be empty"
            )
        for idx in self.two_finger_gate_other_indices:
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    "env.two_finger_gate.other_fingertip_indices contains an out-of-range "
                    f"index for {self.fingers_num} fingertips: {idx}"
                )
        supported_aggregations = {"max", "mean", "min", "mean_min"}
        if self.two_finger_gate_other_aggregation not in supported_aggregations:
            raise ValueError(
                "env.two_finger_gate.other_aggregation must be one of "
                f"{sorted(supported_aggregations)}; got "
                f"{self.two_finger_gate_other_aggregation}"
            )
        if self.two_finger_gate_other_aggregation == "mean_min":
            if (
                self.two_finger_gate_other_mean_weight
                + self.two_finger_gate_other_min_weight
            ) <= 0.0:
                raise ValueError(
                    "env.two_finger_gate.other_mean_weight + "
                    "other_min_weight must be > 0 for mean_min aggregation"
                )
        if self.two_finger_gate_far <= self.two_finger_gate_near:
            raise ValueError(
                "env.two_finger_gate.far must be greater than near; "
                f"got near={self.two_finger_gate_near}, far={self.two_finger_gate_far}"
            )

    def _setup_fingertip_tangent_reward_config(self, tangent_cfg):
        if tangent_cfg is None:
            tangent_cfg = {}
        self.fingertip_tangent_reward_enable = self._cfg_bool(
            tangent_cfg.get("enable", False), default=False
        )
        raw_indices = tangent_cfg.get("fingertip_indices", [1])
        if isinstance(raw_indices, (int, float)):
            raw_indices = [raw_indices]
        self.fingertip_tangent_reward_indices = [int(i) for i in list(raw_indices)]
        self.fingertip_tangent_reward_target = str(
            tangent_cfg.get("target", "nut_pos")
        )
        self.fingertip_tangent_reward_target_offset = self._resolve_numeric_vector(
            tangent_cfg.get("target_offset", [0.0, 0.0, 0.0]),
            3,
            "env.fingertip_tangent_reward.target_offset",
        )
        self.fingertip_tangent_reward_near = float(tangent_cfg.get("near", 0.08))
        self.fingertip_tangent_reward_far = float(tangent_cfg.get("far", 0.13))
        self.fingertip_tangent_reward_velocity_clip = float(
            tangent_cfg.get("velocity_clip", 0.5)
        )
        self.fingertip_tangent_reward_scale_with_object = self._cfg_bool(
            tangent_cfg.get("scale_with_object", False), default=False
        )
        self.fingertip_tangent_reward_positive_only = self._cfg_bool(
            tangent_cfg.get("positive_only", True), default=True
        )
        self.fingertip_tangent_reward_use_contact_force = self._cfg_bool(
            tangent_cfg.get("use_contact_force", True), default=True
        )
        self.fingertip_tangent_reward_contact_force_min = float(
            tangent_cfg.get("contact_force_min", 0.3)
        )
        self.fingertip_tangent_reward_contact_force_max = float(
            tangent_cfg.get("contact_force_max", 2.0)
        )
        self.fingertip_tangent_reward_aggregation = str(
            tangent_cfg.get("aggregation", "mean")
        )
        if not self.fingertip_tangent_reward_enable:
            return
        if len(self.fingertip_tangent_reward_indices) == 0:
            raise ValueError(
                "env.fingertip_tangent_reward.fingertip_indices must not be empty"
            )
        for idx in self.fingertip_tangent_reward_indices:
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    "env.fingertip_tangent_reward.fingertip_indices contains an "
                    f"out-of-range index for {self.fingers_num} fingertips: {idx}"
                )
        if self.fingertip_tangent_reward_target not in {"nut_pos", "object_pos"}:
            raise ValueError(
                "env.fingertip_tangent_reward.target must be 'nut_pos' or "
                f"'object_pos'; got {self.fingertip_tangent_reward_target}"
            )
        if self.fingertip_tangent_reward_far <= self.fingertip_tangent_reward_near:
            raise ValueError(
                "env.fingertip_tangent_reward.far must be greater than near; "
                f"got near={self.fingertip_tangent_reward_near}, "
                f"far={self.fingertip_tangent_reward_far}"
            )
        if self.fingertip_tangent_reward_velocity_clip <= 0.0:
            raise ValueError(
                "env.fingertip_tangent_reward.velocity_clip must be > 0"
            )
        if self.fingertip_tangent_reward_aggregation not in {"mean", "min"}:
            raise ValueError(
                "env.fingertip_tangent_reward.aggregation must be one of "
                "['mean', 'min']; got "
                f"{self.fingertip_tangent_reward_aggregation}"
            )

    def _setup_thumb_slip_diagnostics_config(self, slip_cfg):
        if slip_cfg is None:
            slip_cfg = {}
        self.thumb_slip_contact_drop_w = float(
            slip_cfg.get("contact_drop_w", 0.2)
        )
        self.thumb_slip_far_dist = float(slip_cfg.get("far_dist", 0.09))
        self.thumb_slip_high_tip_speed = float(
            slip_cfg.get("high_tip_speed", 0.25)
        )
        self.thumb_slip_active_screw_vel = float(
            slip_cfg.get("active_screw_vel", 0.2)
        )

    def _setup_thumb_slip_penalty_config(self, slip_cfg):
        if slip_cfg is None:
            slip_cfg = {}
        self.thumb_slip_penalty_enable = self._cfg_bool(
            slip_cfg.get("enable", False), default=False
        )
        self.thumb_slip_penalty_thumb_index = int(
            slip_cfg.get("thumb_fingertip_index", self.finger_contact_thumb_index)
        )
        self.thumb_slip_penalty_active_screw_vel = float(
            slip_cfg.get("active_screw_vel", 0.15)
        )
        self.thumb_slip_penalty_contact_force_min = float(
            slip_cfg.get("contact_force_min", 0.3)
        )
        self.thumb_slip_penalty_contact_force_max = float(
            slip_cfg.get("contact_force_max", 2.0)
        )
        self.thumb_slip_penalty_far_dist = float(slip_cfg.get("far_dist", 0.09))
        self.thumb_slip_penalty_high_tip_speed = float(
            slip_cfg.get("high_tip_speed", 0.25)
        )
        self.thumb_slip_penalty_high_tip_speed_span = float(
            slip_cfg.get("high_tip_speed_span", 0.2)
        )
        self.thumb_slip_penalty_scale_with_object = self._cfg_bool(
            slip_cfg.get("scale_with_object", False), default=False
        )
        self.thumb_slip_contact_penalty_scale = float(
            slip_cfg.get("contact_penalty_scale", 0.0)
        )
        self.thumb_slip_far_penalty_scale = float(
            slip_cfg.get("far_penalty_scale", 0.0)
        )
        self.thumb_slip_ejection_penalty_scale = float(
            slip_cfg.get("ejection_penalty_scale", 0.0)
        )
        self.thumb_slip_after_drive_penalty_scale = float(
            slip_cfg.get("after_drive_penalty_scale", 0.0)
        )
        self.thumb_slip_terminal_ease_penalty_scale = float(
            slip_cfg.get("terminal_ease_penalty_scale", 0.0)
        )
        self.thumb_slip_terminal_ease_near_limit = float(
            slip_cfg.get("terminal_ease_near_limit", 0.75)
        )
        self.thumb_slip_terminal_ease_limit_span = float(
            slip_cfg.get("terminal_ease_limit_span", 0.2)
        )
        self.thumb_slip_terminal_ease_vel_clip = float(
            slip_cfg.get("terminal_ease_vel_clip", 1.5)
        )
        self.thumb_slip_return_contact_penalty_scale = float(
            slip_cfg.get("return_contact_penalty_scale", 0.0)
        )
        self.thumb_slip_return_tangent_vel_threshold = float(
            slip_cfg.get("return_tangent_vel_threshold", 0.02)
        )
        self.thumb_slip_return_tangent_vel_span = float(
            slip_cfg.get("return_tangent_vel_span", 0.10)
        )
        if not self.thumb_slip_penalty_enable:
            return
        if not (0 <= self.thumb_slip_penalty_thumb_index < self.fingers_num):
            raise ValueError(
                "env.thumb_slip_penalty.thumb_fingertip_index is out of range "
                f"for {self.fingers_num} fingertips: "
                f"{self.thumb_slip_penalty_thumb_index}"
            )
        if self.thumb_slip_penalty_contact_force_max <= (
            self.thumb_slip_penalty_contact_force_min
        ):
            raise ValueError(
                "env.thumb_slip_penalty.contact_force_max must be greater than "
                "contact_force_min"
            )
        if self.thumb_slip_penalty_far_dist <= 0.0:
            raise ValueError("env.thumb_slip_penalty.far_dist must be > 0")
        if self.thumb_slip_penalty_high_tip_speed_span <= 0.0:
            raise ValueError(
                "env.thumb_slip_penalty.high_tip_speed_span must be > 0"
            )
        if not (0.0 <= self.thumb_slip_terminal_ease_near_limit < 1.0):
            raise ValueError(
                "env.thumb_slip_penalty.terminal_ease_near_limit must be in [0, 1)"
            )
        if self.thumb_slip_terminal_ease_limit_span <= 0.0:
            raise ValueError(
                "env.thumb_slip_penalty.terminal_ease_limit_span must be > 0"
            )
        if self.thumb_slip_terminal_ease_vel_clip <= 0.0:
            raise ValueError(
                "env.thumb_slip_penalty.terminal_ease_vel_clip must be > 0"
            )
        if self.thumb_slip_return_tangent_vel_threshold < 0.0:
            raise ValueError(
                "env.thumb_slip_penalty.return_tangent_vel_threshold must be >= 0"
            )
        if self.thumb_slip_return_tangent_vel_span <= 0.0:
            raise ValueError(
                "env.thumb_slip_penalty.return_tangent_vel_span must be > 0"
            )

    def _setup_fingertip_torque_reward_config(self, torque_cfg):
        if torque_cfg is None:
            torque_cfg = {}
        self.fingertip_torque_reward_enable = self._cfg_bool(
            torque_cfg.get("enable", False), default=False
        )
        raw_indices = torque_cfg.get("fingertip_indices", [1])
        if isinstance(raw_indices, (int, float)):
            raw_indices = [raw_indices]
        self.fingertip_torque_reward_indices = [int(i) for i in list(raw_indices)]
        self.fingertip_torque_reward_target = str(
            torque_cfg.get("target", "nut_pos")
        )
        self.fingertip_torque_reward_target_offset = self._resolve_numeric_vector(
            torque_cfg.get("target_offset", [0.0, 0.0, 0.0]),
            3,
            "env.fingertip_torque_reward.target_offset",
        )
        self.fingertip_torque_reward_near = float(torque_cfg.get("near", 0.08))
        self.fingertip_torque_reward_far = float(torque_cfg.get("far", 0.13))
        self.fingertip_torque_reward_clip = float(
            torque_cfg.get("torque_clip", 0.04)
        )
        self.fingertip_torque_reward_scale_with_object = self._cfg_bool(
            torque_cfg.get("scale_with_object", False), default=False
        )
        self.fingertip_torque_reward_positive_only = self._cfg_bool(
            torque_cfg.get("positive_only", True), default=True
        )
        self.fingertip_torque_reward_force_sign = float(
            torque_cfg.get("force_sign", -1.0)
        )
        self.fingertip_torque_reward_contact_force_min = float(
            torque_cfg.get("contact_force_min", 0.3)
        )
        self.fingertip_torque_reward_contact_force_max = float(
            torque_cfg.get("contact_force_max", 2.0)
        )
        self.fingertip_torque_reward_aggregation = str(
            torque_cfg.get("aggregation", "mean")
        )
        if not self.fingertip_torque_reward_enable:
            return
        if len(self.fingertip_torque_reward_indices) == 0:
            raise ValueError(
                "env.fingertip_torque_reward.fingertip_indices must not be empty"
            )
        for idx in self.fingertip_torque_reward_indices:
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    "env.fingertip_torque_reward.fingertip_indices contains an "
                    f"out-of-range index for {self.fingers_num} fingertips: {idx}"
                )
        if self.fingertip_torque_reward_target not in {"nut_pos", "object_pos"}:
            raise ValueError(
                "env.fingertip_torque_reward.target must be 'nut_pos' or "
                f"'object_pos'; got {self.fingertip_torque_reward_target}"
            )
        if self.fingertip_torque_reward_far <= self.fingertip_torque_reward_near:
            raise ValueError(
                "env.fingertip_torque_reward.far must be greater than near; "
                f"got near={self.fingertip_torque_reward_near}, "
                f"far={self.fingertip_torque_reward_far}"
            )
        if self.fingertip_torque_reward_clip <= 0.0:
            raise ValueError(
                "env.fingertip_torque_reward.torque_clip must be > 0"
            )
        if self.fingertip_torque_reward_aggregation not in {"mean", "min"}:
            raise ValueError(
                "env.fingertip_torque_reward.aggregation must be one of "
                "['mean', 'min']; got "
                f"{self.fingertip_torque_reward_aggregation}"
            )

    def _setup_active_two_finger_contact_config(self, contact_cfg):
        if contact_cfg is None:
            contact_cfg = {}
        self.active_two_finger_contact_enable = self._cfg_bool(
            contact_cfg.get("enable", False), default=False
        )
        self.active_two_finger_contact_thumb_index = int(
            contact_cfg.get("thumb_fingertip_index", max(self.fingers_num - 1, 0))
        )
        self.active_two_finger_contact_other_index = int(
            contact_cfg.get("other_fingertip_index", 0)
        )
        self.active_two_finger_contact_screw_vel = float(
            contact_cfg.get("active_screw_vel", 0.2)
        )
        self.active_two_finger_contact_force_min = float(
            contact_cfg.get("contact_force_min", 0.5)
        )
        self.active_two_finger_contact_force_max = float(
            contact_cfg.get("contact_force_max", 2.0)
        )
        self.active_two_finger_contact_penalty_scale = float(
            contact_cfg.get("penalty_scale", 0.0)
        )
        if not self.active_two_finger_contact_enable:
            return
        for name, idx in (
            ("thumb_fingertip_index", self.active_two_finger_contact_thumb_index),
            ("other_fingertip_index", self.active_two_finger_contact_other_index),
        ):
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    f"env.active_two_finger_contact.{name} is out of range "
                    f"for {self.fingers_num} fingertips: {idx}"
                )
        if self.active_two_finger_contact_force_max <= (
            self.active_two_finger_contact_force_min
        ):
            raise ValueError(
                "env.active_two_finger_contact.contact_force_max must be greater "
                "than contact_force_min"
            )

    def _setup_opposition_grip_reward_config(self, grip_cfg):
        if grip_cfg is None:
            grip_cfg = {}
        self.opposition_grip_reward_enable = self._cfg_bool(
            grip_cfg.get("enable", False), default=False
        )
        self.opposition_grip_thumb_index = int(
            grip_cfg.get("thumb_fingertip_index", max(self.fingers_num - 1, 0))
        )
        self.opposition_grip_other_index = int(
            grip_cfg.get("other_fingertip_index", 0)
        )
        self.opposition_grip_target = str(grip_cfg.get("target", "nut_pos"))
        self.opposition_grip_target_offset = self._resolve_numeric_vector(
            grip_cfg.get("target_offset", [0.0, 0.0, 0.0]),
            3,
            "env.opposition_grip_reward.target_offset",
        )
        self.opposition_grip_scale_with_object = self._cfg_bool(
            grip_cfg.get("scale_with_object", False), default=False
        )
        self.opposition_grip_near = float(grip_cfg.get("near", 0.08))
        self.opposition_grip_far = float(grip_cfg.get("far", 0.13))
        self.opposition_grip_opposite_cos_min = float(
            grip_cfg.get("opposite_cos_min", 0.2)
        )
        self.opposition_grip_contact_force_min = float(
            grip_cfg.get("contact_force_min", 0.5)
        )
        self.opposition_grip_contact_force_max = float(
            grip_cfg.get("contact_force_max", 3.0)
        )
        self.opposition_grip_force_sign = float(grip_cfg.get("force_sign", -1.0))
        self.opposition_grip_reward_scale = float(grip_cfg.get("reward_scale", 0.0))
        if not self.opposition_grip_reward_enable:
            return
        for name, idx in (
            ("thumb_fingertip_index", self.opposition_grip_thumb_index),
            ("other_fingertip_index", self.opposition_grip_other_index),
        ):
            if not (0 <= idx < self.fingers_num):
                raise ValueError(
                    f"env.opposition_grip_reward.{name} is out of range "
                    f"for {self.fingers_num} fingertips: {idx}"
                )
        if self.opposition_grip_target not in {"nut_pos", "object_pos"}:
            raise ValueError(
                "env.opposition_grip_reward.target must be 'nut_pos' or "
                f"'object_pos'; got {self.opposition_grip_target}"
            )
        if self.opposition_grip_far <= self.opposition_grip_near:
            raise ValueError(
                "env.opposition_grip_reward.far must be greater than near; "
                f"got near={self.opposition_grip_near}, far={self.opposition_grip_far}"
            )
        if self.opposition_grip_contact_force_max <= (
            self.opposition_grip_contact_force_min
        ):
            raise ValueError(
                "env.opposition_grip_reward.contact_force_max must be greater "
                "than contact_force_min"
            )

    def _get_current_object_scale_tensor(self):
        if hasattr(self, "object_scale_buf"):
            return self.object_scale_buf
        return torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        ) * float(self.base_obj_scale)

    def _get_finger_contact_target_pos(self, nut_pos, nut_rot):
        target_offset = torch.tensor(
            self.finger_contact_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        if self.finger_contact_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.finger_contact_target == "nut_pos":
            target_base = nut_pos
            target_rot = nut_rot
        elif self.finger_contact_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            raise ValueError(
                "Unsupported env.finger_object_contact.target: "
                f"{self.finger_contact_target}"
            )

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        return target_base + quat_apply(target_rot, target_offset)

    def _get_finger_contact_threshold(self):
        threshold = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        ) * float(self.reset_dist_threshold)
        if self.finger_contact_threshold_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            threshold = threshold * scale_ratio
        return threshold

    def _compute_fingertip_tangent_reward(self, nut_pos):
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        fingertip_vel = self.ft_linvel_at_cf.reshape(
            self.num_envs, self.fingers_num, 3
        )
        target_offset = torch.tensor(
            self.fingertip_tangent_reward_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        scale_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.fingertip_tangent_reward_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.fingertip_tangent_reward_target == "nut_pos":
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]
        elif self.fingertip_tangent_reward_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            raise ValueError(
                "Unsupported env.fingertip_tangent_reward.target: "
                f"{self.fingertip_tangent_reward_target}"
            )

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        target_pos = target_base + quat_apply(target_rot, target_offset)

        near = (
            torch.ones_like(scale_ratio) * self.fingertip_tangent_reward_near
        )
        far = torch.ones_like(scale_ratio) * self.fingertip_tangent_reward_far
        if self.fingertip_tangent_reward_scale_with_object:
            near = near * scale_ratio
            far = far * scale_ratio
        span = torch.clamp(far - near, min=1e-6)

        tracked_pos = fingertip_pos[:, self.fingertip_tangent_reward_indices, :]
        tracked_vel = fingertip_vel[:, self.fingertip_tangent_reward_indices, :]
        radial = tracked_pos - target_pos.unsqueeze(1)
        radius = torch.norm(radial, dim=-1)
        tangent = torch.cross(
            self.rot_axis_buf.unsqueeze(1).expand_as(radial), radial, dim=-1
        )
        tangent_norm = torch.clamp(torch.norm(tangent, dim=-1, keepdim=True), min=1e-6)
        tangent_dir = tangent / tangent_norm
        tangent_vel = (tracked_vel * tangent_dir).sum(dim=-1)
        if self.fingertip_tangent_reward_positive_only:
            positive_tangent_vel = torch.clamp(tangent_vel, min=0.0)
        else:
            positive_tangent_vel = tangent_vel

        dist_weight = torch.clamp(
            (far.unsqueeze(-1) - radius) / span.unsqueeze(-1),
            min=0.0,
            max=1.0,
        )
        contact_weight = torch.ones_like(dist_weight)
        if self.fingertip_tangent_reward_use_contact_force:
            fingertip_force = torch.norm(
                self.contact_forces[:, self.fingertip_handles, :], dim=-1
            )
            tracked_force = fingertip_force[
                :, self.fingertip_tangent_reward_indices
            ]
            force_span = max(
                self.fingertip_tangent_reward_contact_force_max
                - self.fingertip_tangent_reward_contact_force_min,
                1e-6,
            )
            contact_weight = torch.clamp(
                (
                    tracked_force
                    - self.fingertip_tangent_reward_contact_force_min
                )
                / force_span,
                min=0.0,
                max=1.0,
            )

        normalized_vel = torch.clamp(
            positive_tangent_vel / self.fingertip_tangent_reward_velocity_clip,
            min=0.0,
            max=1.0,
        )
        reward_all = normalized_vel * dist_weight * contact_weight
        if self.fingertip_tangent_reward_aggregation == "min":
            reward = reward_all.min(dim=-1).values
        else:
            reward = reward_all.mean(dim=-1)
        stats = {
            "tangent_vel": tangent_vel,
            "positive_tangent_vel": positive_tangent_vel,
            "dist_weight": dist_weight,
            "contact_weight": contact_weight,
            "reward_all": reward_all,
        }
        return reward, stats

    def _compute_fingertip_torque_reward(self, nut_pos):
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        target_offset = torch.tensor(
            self.fingertip_torque_reward_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        scale_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.fingertip_torque_reward_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.fingertip_torque_reward_target == "nut_pos":
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]
        elif self.fingertip_torque_reward_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            raise ValueError(
                "Unsupported env.fingertip_torque_reward.target: "
                f"{self.fingertip_torque_reward_target}"
            )

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        target_pos = target_base + quat_apply(target_rot, target_offset)

        near = torch.ones_like(scale_ratio) * self.fingertip_torque_reward_near
        far = torch.ones_like(scale_ratio) * self.fingertip_torque_reward_far
        if self.fingertip_torque_reward_scale_with_object:
            near = near * scale_ratio
            far = far * scale_ratio
        span = torch.clamp(far - near, min=1e-6)

        tracked_pos = fingertip_pos[:, self.fingertip_torque_reward_indices, :]
        radial = tracked_pos - target_pos.unsqueeze(1)
        radius = torch.norm(radial, dim=-1)
        tracked_force = self.contact_forces[
            :, [self.fingertip_handles[i] for i in self.fingertip_torque_reward_indices], :
        ]
        force_on_object = tracked_force * self.fingertip_torque_reward_force_sign
        torque_vec = torch.cross(radial, force_on_object, dim=-1)
        signed_torque = (
            torque_vec * self.rot_axis_buf.unsqueeze(1).expand_as(torque_vec)
        ).sum(dim=-1)
        if self.fingertip_torque_reward_positive_only:
            positive_torque = torch.clamp(signed_torque, min=0.0)
        else:
            positive_torque = signed_torque
        negative_torque = torch.clamp(-signed_torque, min=0.0)

        dist_weight = torch.clamp(
            (far.unsqueeze(-1) - radius) / span.unsqueeze(-1),
            min=0.0,
            max=1.0,
        )
        force_mag = torch.norm(tracked_force, dim=-1)
        force_span = max(
            self.fingertip_torque_reward_contact_force_max
            - self.fingertip_torque_reward_contact_force_min,
            1e-6,
        )
        contact_weight = torch.clamp(
            (force_mag - self.fingertip_torque_reward_contact_force_min)
            / force_span,
            min=0.0,
            max=1.0,
        )
        normalized_torque = torch.clamp(
            positive_torque / self.fingertip_torque_reward_clip,
            min=0.0,
            max=1.0,
        )
        reward_all = normalized_torque * dist_weight * contact_weight
        if self.fingertip_torque_reward_aggregation == "min":
            reward = reward_all.min(dim=-1).values
        else:
            reward = reward_all.mean(dim=-1)
        stats = {
            "signed_torque": signed_torque,
            "positive_torque": torch.clamp(signed_torque, min=0.0),
            "negative_torque": negative_torque,
            "abs_torque": torch.abs(signed_torque),
            "dist_weight": dist_weight,
            "contact_weight": contact_weight,
            "reward_all": reward_all,
        }
        return reward, stats

    def _compute_active_two_finger_contact_penalty(self, nut_dof_linvel):
        fingertip_force = torch.norm(
            self.contact_forces[:, self.fingertip_handles, :], dim=-1
        )
        force_span = max(
            self.active_two_finger_contact_force_max
            - self.active_two_finger_contact_force_min,
            1e-6,
        )
        thumb_force = fingertip_force[:, self.active_two_finger_contact_thumb_index]
        other_force = fingertip_force[:, self.active_two_finger_contact_other_index]
        thumb_contact_w = torch.clamp(
            (thumb_force - self.active_two_finger_contact_force_min) / force_span,
            min=0.0,
            max=1.0,
        )
        other_contact_w = torch.clamp(
            (other_force - self.active_two_finger_contact_force_min) / force_span,
            min=0.0,
            max=1.0,
        )
        pair_contact_w = torch.minimum(thumb_contact_w, other_contact_w)
        positive_velocity = torch.clamp(nut_dof_linvel, min=0.0)
        active = (nut_dof_linvel > self.active_two_finger_contact_screw_vel).float()
        penalty = (
            self.active_two_finger_contact_penalty_scale
            * active
            * positive_velocity
            * (1.0 - pair_contact_w)
        )
        stats = {
            "penalty": penalty,
            "active_frac": active,
            "pair_contact_w": pair_contact_w,
            "thumb_contact_w": thumb_contact_w,
            "other_contact_w": other_contact_w,
        }
        return penalty, stats

    def _compute_thumb_slip_penalty(self, thumb_dist, nut_dof_linvel, nut_pos):
        fingertip_force = torch.norm(
            self.contact_forces[:, self.fingertip_handles, :], dim=-1
        )
        fingertip_vel = self.ft_linvel_at_cf.reshape(
            self.num_envs, self.fingers_num, 3
        )
        thumb_force = fingertip_force[:, self.thumb_slip_penalty_thumb_index]
        thumb_tip_speed = torch.norm(
            fingertip_vel[:, self.thumb_slip_penalty_thumb_index, :], dim=-1
        )
        force_span = max(
            self.thumb_slip_penalty_contact_force_max
            - self.thumb_slip_penalty_contact_force_min,
            1e-6,
        )
        thumb_contact_w = torch.clamp(
            (thumb_force - self.thumb_slip_penalty_contact_force_min) / force_span,
            min=0.0,
            max=1.0,
        )

        active = (nut_dof_linvel > self.thumb_slip_penalty_active_screw_vel).float()
        velocity_weight = torch.clamp(
            torch.clamp(nut_dof_linvel, min=0.0) / max(float(self.angvel_clip_max), 1e-6),
            min=0.0,
            max=1.0,
        )
        active_weight = active * velocity_weight
        speed_weight = torch.clamp(
            (
                thumb_tip_speed - self.thumb_slip_penalty_high_tip_speed
            )
            / self.thumb_slip_penalty_high_tip_speed_span,
            min=0.0,
            max=1.0,
        )

        far_dist = torch.ones_like(thumb_dist) * self.thumb_slip_penalty_far_dist
        if self.thumb_slip_penalty_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            far_dist = far_dist * scale_ratio

        contact_loss = active_weight * (1.0 - thumb_contact_w)
        far_loss = active_weight * torch.clamp(
            (thumb_dist - far_dist) / torch.clamp(far_dist, min=1e-6),
            min=0.0,
            max=1.0,
        )
        detach_loss = torch.maximum(contact_loss, far_loss)
        ejection_loss = detach_loss * speed_weight
        after_drive_loss = self.prev_thumb_drive_w * detach_loss * speed_weight

        terminal_ease_loss = torch.zeros_like(contact_loss)
        if self.pose_diff_penalty_thumb_indices.numel() > 0:
            thumb_indices = self.pose_diff_penalty_thumb_indices
            lower = self.xhand_hand_dof_lower_limits.index_select(0, thumb_indices)
            upper = self.xhand_hand_dof_upper_limits.index_select(0, thumb_indices)
            thumb_pos = self.xhand_hand_dof_pos.index_select(-1, thumb_indices)
            thumb_vel = self.xhand_hand_dof_vel.index_select(-1, thumb_indices)
            thumb_span = torch.clamp(upper - lower, min=1e-6)
            thumb_norm = torch.clamp((thumb_pos - lower) / thumb_span, 0.0, 1.0)
            thumb_edge = torch.abs(thumb_norm - 0.5) * 2.0
            edge_weight = torch.clamp(
                (
                    thumb_edge.max(dim=-1).values
                    - self.thumb_slip_terminal_ease_near_limit
                )
                / self.thumb_slip_terminal_ease_limit_span,
                min=0.0,
                max=1.0,
            )
            thumb_joint_vel_abs = torch.abs(thumb_vel).mean(dim=-1)
            joint_vel_weight = torch.clamp(
                thumb_joint_vel_abs / self.thumb_slip_terminal_ease_vel_clip,
                min=0.0,
                max=1.0,
            )
            terminal_ease_loss = active_weight * edge_weight * joint_vel_weight

        thumb_tangent_vel = torch.zeros_like(contact_loss)
        return_phase_w = torch.zeros_like(contact_loss)
        return_context_w = torch.zeros_like(contact_loss)
        return_contact_loss = torch.zeros_like(contact_loss)
        if self.thumb_slip_return_contact_penalty_scale != 0.0:
            fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
            thumb_pos = fingertip_pos[:, self.thumb_slip_penalty_thumb_index, :]
            thumb_vel_vec = fingertip_vel[:, self.thumb_slip_penalty_thumb_index, :]

            target_offset = torch.tensor(
                self.fingertip_torque_reward_target_offset,
                device=self.device,
                dtype=torch.float,
            ).unsqueeze(0)
            scale_ratio = torch.ones(
                self.num_envs, device=self.device, dtype=torch.float
            )
            if self.fingertip_torque_reward_scale_with_object:
                scale_ratio = self._get_current_object_scale_tensor() / max(
                    float(self.base_obj_scale), 1e-6
                )
                target_offset = target_offset * scale_ratio.unsqueeze(-1)

            if self.fingertip_torque_reward_target == "nut_pos":
                target_base = nut_pos
                target_rot = self.nut_states[:, 3:7]
            elif self.fingertip_torque_reward_target == "object_pos":
                target_base = self.object_pos
                target_rot = self.object_rot
            else:
                target_base = nut_pos
                target_rot = self.nut_states[:, 3:7]

            if target_offset.shape[0] == 1:
                target_offset = target_offset.expand(self.num_envs, -1)
            target_pos = target_base + quat_apply(target_rot, target_offset)

            radial = thumb_pos - target_pos
            tangent = torch.cross(self.rot_axis_buf, radial, dim=-1)
            tangent_norm = torch.clamp(
                torch.norm(tangent, dim=-1, keepdim=True), min=1e-6
            )
            tangent_dir = tangent / tangent_norm
            thumb_tangent_vel = (thumb_vel_vec * tangent_dir).sum(dim=-1)
            return_phase_w = torch.clamp(
                (
                    (-thumb_tangent_vel)
                    - self.thumb_slip_return_tangent_vel_threshold
                )
                / self.thumb_slip_return_tangent_vel_span,
                min=0.0,
                max=1.0,
            )
            return_context_w = torch.maximum(active, self.prev_thumb_drive_w)
            return_contact_loss = return_context_w * return_phase_w * thumb_contact_w

        current_thumb_drive_w = (active * thumb_contact_w).detach()
        penalty = (
            self.thumb_slip_contact_penalty_scale * contact_loss
            + self.thumb_slip_far_penalty_scale * far_loss
            + self.thumb_slip_ejection_penalty_scale * ejection_loss
            + self.thumb_slip_after_drive_penalty_scale * after_drive_loss
            + self.thumb_slip_terminal_ease_penalty_scale * terminal_ease_loss
            + self.thumb_slip_return_contact_penalty_scale * return_contact_loss
        )
        stats = {
            "contact_loss": contact_loss,
            "far_loss": far_loss,
            "ejection_loss": ejection_loss,
            "after_drive_loss": after_drive_loss,
            "terminal_ease_loss": terminal_ease_loss,
            "return_contact_loss": return_contact_loss,
            "return_phase_w": return_phase_w,
            "return_context_w": return_context_w,
            "thumb_tangent_vel": thumb_tangent_vel,
            "thumb_contact_w": thumb_contact_w,
            "thumb_dist": thumb_dist,
            "thumb_tip_speed": thumb_tip_speed,
            "active": active,
            "velocity_weight": velocity_weight,
            "speed_weight": speed_weight,
            "prev_drive_w": self.prev_thumb_drive_w.clone(),
            "current_drive_w": current_thumb_drive_w,
        }
        self.prev_thumb_drive_w[:] = current_thumb_drive_w
        return penalty, stats

    def _compute_opposition_grip_reward(self, nut_pos):
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        target_offset = torch.tensor(
            self.opposition_grip_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        scale_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.opposition_grip_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.opposition_grip_target == "nut_pos":
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]
        elif self.opposition_grip_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            raise ValueError(
                "Unsupported env.opposition_grip_reward.target: "
                f"{self.opposition_grip_target}"
            )

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        target_pos = target_base + quat_apply(target_rot, target_offset)

        thumb_pos = fingertip_pos[:, self.opposition_grip_thumb_index, :]
        other_pos = fingertip_pos[:, self.opposition_grip_other_index, :]
        thumb_radial = thumb_pos - target_pos
        other_radial = other_pos - target_pos
        thumb_radius = torch.norm(thumb_radial, dim=-1)
        other_radius = torch.norm(other_radial, dim=-1)
        thumb_dir = thumb_radial / torch.clamp(thumb_radius.unsqueeze(-1), min=1e-6)
        other_dir = other_radial / torch.clamp(other_radius.unsqueeze(-1), min=1e-6)

        near = torch.ones_like(scale_ratio) * self.opposition_grip_near
        far = torch.ones_like(scale_ratio) * self.opposition_grip_far
        if self.opposition_grip_scale_with_object:
            near = near * scale_ratio
            far = far * scale_ratio
        span = torch.clamp(far - near, min=1e-6)
        thumb_dist_w = torch.clamp((far - thumb_radius) / span, min=0.0, max=1.0)
        other_dist_w = torch.clamp((far - other_radius) / span, min=0.0, max=1.0)
        pair_dist_w = torch.minimum(thumb_dist_w, other_dist_w)

        radial_dot = (thumb_dir * other_dir).sum(dim=-1)
        opposite_span = max(1.0 - self.opposition_grip_opposite_cos_min, 1e-6)
        oppositeness = torch.clamp(
            ((-radial_dot) - self.opposition_grip_opposite_cos_min)
            / opposite_span,
            min=0.0,
            max=1.0,
        )

        thumb_force = self.contact_forces[
            :, self.fingertip_handles[self.opposition_grip_thumb_index], :
        ]
        other_force = self.contact_forces[
            :, self.fingertip_handles[self.opposition_grip_other_index], :
        ]
        thumb_force_on_object = thumb_force * self.opposition_grip_force_sign
        other_force_on_object = other_force * self.opposition_grip_force_sign
        thumb_inward_force = (thumb_force_on_object * (-thumb_dir)).sum(dim=-1)
        other_inward_force = (other_force_on_object * (-other_dir)).sum(dim=-1)
        force_span = max(
            self.opposition_grip_contact_force_max
            - self.opposition_grip_contact_force_min,
            1e-6,
        )
        thumb_inward_w = torch.clamp(
            (thumb_inward_force - self.opposition_grip_contact_force_min)
            / force_span,
            min=0.0,
            max=1.0,
        )
        other_inward_w = torch.clamp(
            (other_inward_force - self.opposition_grip_contact_force_min)
            / force_span,
            min=0.0,
            max=1.0,
        )
        pair_inward_w = torch.minimum(thumb_inward_w, other_inward_w)
        reward = oppositeness * pair_dist_w * pair_inward_w
        stats = {
            "reward": reward,
            "oppositeness": oppositeness,
            "radial_dot": radial_dot,
            "pair_dist_w": pair_dist_w,
            "pair_inward_w": pair_inward_w,
            "thumb_inward_w": thumb_inward_w,
            "other_inward_w": other_inward_w,
            "thumb_inward_force": thumb_inward_force,
            "other_inward_force": other_inward_force,
        }
        return reward, stats

    def _compute_finger_diagnostics(self, nut_pos):
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        fingertip_vel = self.ft_linvel_at_cf.reshape(
            self.num_envs, self.fingers_num, 3
        )
        target_offset = torch.tensor(
            self.fingertip_torque_reward_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        scale_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.fingertip_torque_reward_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.fingertip_torque_reward_target == "nut_pos":
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]
        elif self.fingertip_torque_reward_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        target_pos = target_base + quat_apply(target_rot, target_offset)

        radial = fingertip_pos - target_pos.unsqueeze(1)
        dist = torch.norm(radial, dim=-1)
        radial_dir = radial / torch.clamp(dist.unsqueeze(-1), min=1e-6)
        tangent = torch.cross(
            self.rot_axis_buf.unsqueeze(1).expand_as(radial), radial, dim=-1
        )
        tangent_norm = torch.clamp(torch.norm(tangent, dim=-1, keepdim=True), min=1e-6)
        tangent_dir = tangent / tangent_norm
        tangent_vel = (fingertip_vel * tangent_dir).sum(dim=-1)
        positive_tangent_vel = torch.clamp(tangent_vel, min=0.0)

        fingertip_force = self.contact_forces[:, self.fingertip_handles, :]
        force_mag = torch.norm(fingertip_force, dim=-1)
        force_span = max(
            self.fingertip_torque_reward_contact_force_max
            - self.fingertip_torque_reward_contact_force_min,
            1e-6,
        )
        force_weight = torch.clamp(
            (force_mag - self.fingertip_torque_reward_contact_force_min)
            / force_span,
            min=0.0,
            max=1.0,
        )

        force_on_object = fingertip_force * self.fingertip_torque_reward_force_sign
        inward_normal_force = torch.clamp(
            -(force_on_object * radial_dir).sum(dim=-1), min=0.0
        )
        tangent_force = (force_on_object * tangent_dir).sum(dim=-1)
        positive_tangent_force = torch.clamp(tangent_force, min=0.0)
        torque_vec = torch.cross(radial, force_on_object, dim=-1)
        signed_torque = (
            torque_vec * self.rot_axis_buf.unsqueeze(1).expand_as(torque_vec)
        ).sum(dim=-1)
        positive_torque = torch.clamp(signed_torque, min=0.0)

        contribution_indices = [idx for idx in [0, 1, 3] if idx < self.fingers_num]
        if contribution_indices:
            torque_denominator = positive_torque[:, contribution_indices].sum(
                dim=-1, keepdim=True
            )
        else:
            torque_denominator = positive_torque.sum(dim=-1, keepdim=True)
        torque_ratio = positive_torque / torch.clamp(torque_denominator, min=1e-6)

        middle_joint_vel_abs = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        middle_joint0_sign_flip = torch.zeros_like(middle_joint_vel_abs)
        if self.num_xhand_hand_dofs >= 8:
            middle_vel = self.xhand_hand_dof_vel[:, 4:8]
            middle_joint_vel_abs = torch.abs(middle_vel).mean(dim=-1)
            current_middle_joint0_vel = self.xhand_hand_dof_vel[:, 4]
            previous_middle_joint0_vel = self.dof_vel_prev[:, -1, 4]
            vel_threshold = 1e-4
            middle_joint0_sign_flip = (
                (current_middle_joint0_vel * previous_middle_joint0_vel < 0.0)
                & (torch.abs(current_middle_joint0_vel) > vel_threshold)
                & (torch.abs(previous_middle_joint0_vel) > vel_threshold)
            ).float()

        index_middle_tip_dist = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.fingers_num > 1:
            index_middle_tip_dist = torch.norm(
                fingertip_pos[:, 0, :] - fingertip_pos[:, 1, :], dim=-1
            )

        thumb_slip_stats = self._compute_thumb_slip_diagnostics(
            dist=dist,
            force_weight=force_weight,
            fingertip_vel=fingertip_vel,
            force_mag=force_mag,
            inward_normal_force=inward_normal_force,
            tangent_vel=tangent_vel,
        )

        return {
            "signed_torque": signed_torque,
            "positive_torque": positive_torque,
            "torque_ratio": torque_ratio,
            "positive_tangent_vel": positive_tangent_vel,
            "force_mag": force_mag,
            "inward_normal_force": inward_normal_force,
            "tangent_force": tangent_force,
            "positive_tangent_force": positive_tangent_force,
            "force_weight": force_weight,
            "dist": dist,
            "middle_joint_vel_abs": middle_joint_vel_abs,
            "middle_joint0_sign_flip": middle_joint0_sign_flip,
            "index_middle_tip_dist": index_middle_tip_dist,
            "thumb_slip": thumb_slip_stats,
        }

    def _compute_thumb_slip_diagnostics(
        self,
        dist,
        force_weight,
        fingertip_vel,
        force_mag=None,
        inward_normal_force=None,
        tangent_vel=None,
    ):
        zero = torch.zeros((), device=self.device, dtype=torch.float)
        thumb_idx = int(self.finger_contact_thumb_index)
        if thumb_idx >= self.fingers_num:
            return {
                "contact_drop_frac": zero,
                "far_frac": zero,
                "active_detach_frac": zero,
                "active_far_frac": zero,
                "ejection_frac": zero,
                "tip_speed_mean": zero,
                "tip_speed_p95": zero,
                "joint_vel_abs_mean": zero,
                "joint_vel_abs_p95": zero,
                "dist_p95": zero,
                "contact_w_p05": zero,
                "force_raw_mean": zero,
                "force_raw_p05": zero,
                "normal_force_mean": zero,
                "normal_force_p05": zero,
                "tangent_vel_abs_mean": zero,
                "tangent_vel_abs_p95": zero,
                "active_normal_drop_frac": zero,
                "active_screw_frac": zero,
                "score": zero,
            }

        thumb_contact_w = force_weight[:, thumb_idx]
        thumb_dist = dist[:, thumb_idx]
        thumb_tip_speed = torch.norm(fingertip_vel[:, thumb_idx, :], dim=-1)
        if force_mag is None:
            thumb_force_raw = torch.zeros_like(thumb_dist)
        else:
            thumb_force_raw = force_mag[:, thumb_idx]
        if inward_normal_force is None:
            thumb_normal_force = torch.zeros_like(thumb_dist)
        else:
            thumb_normal_force = inward_normal_force[:, thumb_idx]
        if tangent_vel is None:
            thumb_tangent_vel_abs = torch.zeros_like(thumb_dist)
        else:
            thumb_tangent_vel_abs = torch.abs(tangent_vel[:, thumb_idx])

        thumb_joint_vel_abs = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.num_xhand_hand_dofs >= 16:
            thumb_joint_vel_abs = torch.abs(self.xhand_hand_dof_vel[:, 12:16]).mean(
                dim=-1
            )

        contact_drop = thumb_contact_w < self.thumb_slip_contact_drop_w
        far = thumb_dist > self.thumb_slip_far_dist
        high_tip_speed = thumb_tip_speed > self.thumb_slip_high_tip_speed
        active_screw = self.nut_dof_vel.view(-1) > self.thumb_slip_active_screw_vel
        normal_drop = thumb_normal_force < self.fingertip_torque_reward_contact_force_min

        active_count = torch.clamp(active_screw.float().sum(), min=1.0)
        active_detach_frac = (contact_drop & active_screw).float().sum() / active_count
        active_far_frac = (far & active_screw).float().sum() / active_count
        active_normal_drop_frac = (
            (normal_drop & active_screw).float().sum() / active_count
        )
        ejection_frac = (far & high_tip_speed).float().mean()
        score = active_detach_frac + active_far_frac + ejection_frac

        return {
            "contact_drop_frac": contact_drop.float().mean(),
            "far_frac": far.float().mean(),
            "active_detach_frac": active_detach_frac,
            "active_far_frac": active_far_frac,
            "ejection_frac": ejection_frac,
            "tip_speed_mean": thumb_tip_speed.mean(),
            "tip_speed_p95": torch.quantile(thumb_tip_speed, 0.95),
            "joint_vel_abs_mean": thumb_joint_vel_abs.mean(),
            "joint_vel_abs_p95": torch.quantile(thumb_joint_vel_abs, 0.95),
            "dist_p95": torch.quantile(thumb_dist, 0.95),
            "contact_w_p05": torch.quantile(thumb_contact_w, 0.05),
            "force_raw_mean": thumb_force_raw.mean(),
            "force_raw_p05": torch.quantile(thumb_force_raw, 0.05),
            "normal_force_mean": thumb_normal_force.mean(),
            "normal_force_p05": torch.quantile(thumb_normal_force, 0.05),
            "tangent_vel_abs_mean": thumb_tangent_vel_abs.mean(),
            "tangent_vel_abs_p95": torch.quantile(thumb_tangent_vel_abs, 0.95),
            "active_normal_drop_frac": active_normal_drop_frac,
            "active_screw_frac": active_screw.float().mean(),
            "score": score,
        }

    def _write_finger_diagnostic_extras(self, stats):
        finger_names = {
            0: "index",
            1: "middle",
            3: "thumb",
        }
        for idx, name in finger_names.items():
            if idx >= self.fingers_num:
                continue
            self.extras[f"finger_torque/{name}/signed"] = stats[
                "signed_torque"
            ][:, idx].mean()
            self.extras[f"finger_torque/{name}/positive"] = stats[
                "positive_torque"
            ][:, idx].mean()
            self.extras[f"finger_torque/{name}/ratio_positive"] = stats[
                "torque_ratio"
            ][:, idx].mean()
            self.extras[f"finger_tangent/{name}/positive_vel"] = stats[
                "positive_tangent_vel"
            ][:, idx].mean()
            self.extras[f"finger_tangent/{name}/positive_force"] = stats[
                "positive_tangent_force"
            ][:, idx].mean()
            self.extras[f"finger_contact/{name}/force_raw"] = stats[
                "force_mag"
            ][:, idx].mean()
            self.extras[f"finger_contact/{name}/normal_force"] = stats[
                "inward_normal_force"
            ][:, idx].mean()
            self.extras[f"finger_contact/{name}/force_w"] = stats[
                "force_weight"
            ][:, idx].mean()
            self.extras[f"finger_dist/{name}"] = stats["dist"][:, idx].mean()

        self.extras["finger_motion/middle_joint_vel_abs"] = stats[
            "middle_joint_vel_abs"
        ].mean()
        self.extras["finger_motion/middle_joint0_sign_flip_rate"] = stats[
            "middle_joint0_sign_flip"
        ].mean()
        self.extras["finger_motion/index_middle_tip_dist"] = stats[
            "index_middle_tip_dist"
        ].mean()

        for name, value in stats.get("thumb_slip", {}).items():
            self.extras[f"thumb_slip/{name}"] = value

    def _apply_two_finger_gate(self, rotate_reward_raw, nut_dof_linvel, nut_pos):
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        target_offset = torch.tensor(
            self.two_finger_gate_target_offset,
            device=self.device,
            dtype=torch.float,
        ).unsqueeze(0)
        scale_ratio = torch.ones(
            self.num_envs, device=self.device, dtype=torch.float
        )
        if self.two_finger_gate_scale_with_object:
            scale_ratio = self._get_current_object_scale_tensor() / max(
                float(self.base_obj_scale), 1e-6
            )
            target_offset = target_offset * scale_ratio.unsqueeze(-1)

        if self.two_finger_gate_target == "nut_pos":
            target_base = nut_pos
            target_rot = self.nut_states[:, 3:7]
        elif self.two_finger_gate_target == "object_pos":
            target_base = self.object_pos
            target_rot = self.object_rot
        else:
            raise ValueError(
                "Unsupported env.two_finger_gate.target: "
                f"{self.two_finger_gate_target}"
            )

        if target_offset.shape[0] == 1:
            target_offset = target_offset.expand(self.num_envs, -1)
        target_pos = target_base + quat_apply(
            target_rot, target_offset
        )
        near = torch.ones_like(scale_ratio) * self.two_finger_gate_near
        far = torch.ones_like(scale_ratio) * self.two_finger_gate_far
        if self.two_finger_gate_scale_with_object:
            near = near * scale_ratio
            far = far * scale_ratio
        span = torch.clamp(far - near, min=1e-6)

        thumb_pos = fingertip_pos[:, self.two_finger_gate_thumb_index, :]
        other_pos = fingertip_pos[:, self.two_finger_gate_other_indices, :]
        thumb_dist = torch.norm(thumb_pos - target_pos, dim=-1)
        other_dist_all = torch.norm(
            other_pos - target_pos.unsqueeze(1), dim=-1
        )

        thumb_dist_w = torch.clamp((far - thumb_dist) / span, min=0.0, max=1.0)
        other_dist_w_all = torch.clamp(
            (far.unsqueeze(-1) - other_dist_all) / span.unsqueeze(-1),
            min=0.0,
            max=1.0,
        )

        thumb_weight = thumb_dist_w
        other_weight_all = other_dist_w_all
        if self.two_finger_gate_use_contact_force:
            fingertip_force = torch.norm(
                self.contact_forces[:, self.fingertip_handles, :], dim=-1
            )
            force_span = max(
                self.two_finger_gate_contact_force_max
                - self.two_finger_gate_contact_force_min,
                1e-6,
            )
            thumb_force = fingertip_force[:, self.two_finger_gate_thumb_index]
            other_force_all = fingertip_force[:, self.two_finger_gate_other_indices]
            thumb_force_w = torch.clamp(
                (thumb_force - self.two_finger_gate_contact_force_min) / force_span,
                min=0.0,
                max=1.0,
            )
            other_force_w_all = torch.clamp(
                (
                    other_force_all - self.two_finger_gate_contact_force_min
                )
                / force_span,
                min=0.0,
                max=1.0,
            )
            thumb_weight = thumb_weight * thumb_force_w
            other_weight_all = other_weight_all * other_force_w_all

        other_weight_mean = other_weight_all.mean(dim=-1)
        other_weight_min = other_weight_all.min(dim=-1).values
        other_dist_mean = other_dist_all.mean(dim=-1)
        other_dist_max = other_dist_all.max(dim=-1).values
        if self.two_finger_gate_other_aggregation == "max":
            other_weight, other_best_idx = other_weight_all.max(dim=-1)
            other_dist = other_dist_all.gather(
                1, other_best_idx.unsqueeze(-1)
            ).squeeze(-1)
        elif self.two_finger_gate_other_aggregation == "mean":
            other_weight = other_weight_mean
            other_dist = other_dist_mean
        elif self.two_finger_gate_other_aggregation == "min":
            other_weight = other_weight_min
            other_dist = other_dist_max
        elif self.two_finger_gate_other_aggregation == "mean_min":
            mean_w = self.two_finger_gate_other_mean_weight
            min_w = self.two_finger_gate_other_min_weight
            weight_sum = max(mean_w + min_w, 1e-6)
            other_weight = (
                mean_w * other_weight_mean + min_w * other_weight_min
            ) / weight_sum
            other_dist = other_dist_mean
        gate = torch.clamp(thumb_weight * other_weight, min=0.0, max=1.0)
        gate = torch.pow(gate, self.two_finger_gate_power)
        gate_mult = self.two_finger_gate_min_mult + (
            1.0 - self.two_finger_gate_min_mult
        ) * gate

        if self.two_finger_gate_apply_positive_vel_only:
            rotate_reward = torch.where(
                rotate_reward_raw > 0,
                rotate_reward_raw * gate_mult,
                rotate_reward_raw,
            )
        else:
            rotate_reward = rotate_reward_raw * gate_mult

        positive_velocity = torch.clamp(nut_dof_linvel, min=0.0)
        two_finger_extra = self.two_finger_gate_no_grasp_penalty_scale * (
            positive_velocity * (1.0 - gate)
        )
        gate_stats = {
            "gate": gate,
            "thumb_weight": thumb_weight,
            "other_weight": other_weight,
            "other_mean_weight": other_weight_mean,
            "other_min_weight": other_weight_min,
            "thumb_dist": thumb_dist,
            "other_dist": other_dist,
            "other_mean_dist": other_dist_mean,
            "other_max_dist": other_dist_max,
        }
        return rotate_reward, two_finger_extra, gate_stats

    def _setup_domain_rand_config(self, rand_config):
        self.randomize_mass = rand_config["randomizeMass"]
        self.randomize_mass_lower = rand_config["randomizeMassLower"]
        self.randomize_mass_upper = rand_config["randomizeMassUpper"]
        self.randomize_com = rand_config["randomizeCOM"]
        self.randomize_com_lower = rand_config["randomizeCOMLower"]
        self.randomize_com_upper = rand_config["randomizeCOMUpper"]
        self.randomize_friction = rand_config["randomizeFriction"]
        self.randomize_friction_lower = rand_config["randomizeFrictionLower"]
        self.randomize_friction_upper = rand_config["randomizeFrictionUpper"]
        self.randomize_restitution_lower = rand_config.get(
            "randomizeRestitutionLower", 0.0
        )
        self.randomize_restitution_upper = rand_config.get(
            "randomizeRestitutionUpper", 1.0
        )
        self.randomize_screw_joint_friction = rand_config.get(
            "randomizeScrewJointFriction", False
        )
        self.randomize_screw_joint_friction_lower = rand_config.get(
            "randomizeScrewJointFrictionLower", 0.2
        )
        self.randomize_screw_joint_friction_upper = rand_config.get(
            "randomizeScrewJointFrictionUpper", 0.2
        )
        self.randomize_scale = rand_config["randomizeScale"]
        self.randomize_hand_scale = rand_config["randomize_hand_scale"]
        self.scale_list_init = rand_config["scaleListInit"]
        self.randomize_scale_list = rand_config["randomizeScaleList"]
        self.randomize_scale_lower = rand_config["randomizeScaleLower"]
        self.randomize_scale_upper = rand_config["randomizeScaleUpper"]
        # Store joint-specific randomization parameters
        dof_names = [
            "left_hand_index_bend_joint",
            "left_hand_index_joint1",
            "left_hand_index_joint2",
            "left_hand_mid_joint1",
            "left_hand_mid_joint2",
            "left_hand_pinky_joint1",
            "left_hand_pinky_joint2",
            "left_hand_ring_joint1",
            "left_hand_ring_joint2",
            "left_hand_thumb_bend_joint",
            "left_hand_thumb_rota_joint1",
            "left_hand_thumb_rota_joint2",
        ]
        self.joint_p_gain_lower = {}
        self.joint_p_gain_upper = {}
        self.joint_d_gain_lower = {}
        self.joint_d_gain_upper = {}
        for joint_name in dof_names:
            p_lower_key = f"randomizePGainLower_{joint_name}"
            p_upper_key = f"randomizePGainUpper_{joint_name}"
            d_lower_key = f"randomizeDGainLower_{joint_name}"
            d_upper_key = f"randomizeDGainUpper_{joint_name}"
            self.joint_p_gain_lower[joint_name] = rand_config.get(
                p_lower_key, rand_config["randomizePGainLower"]
            )
            self.joint_p_gain_upper[joint_name] = rand_config.get(
                p_upper_key, rand_config["randomizePGainUpper"]
            )
            self.joint_d_gain_lower[joint_name] = rand_config.get(
                d_lower_key, rand_config["randomizeDGainLower"]
            )
            self.joint_d_gain_upper[joint_name] = rand_config.get(
                d_upper_key, rand_config["randomizeDGainUpper"]
            )

        self.randomize_pd_gains = rand_config["randomizePDGains"]
        self.randomize_p_gain_lower = rand_config["randomizePGainLower"]
        self.randomize_p_gain_upper = rand_config["randomizePGainUpper"]
        self.randomize_d_gain_lower = rand_config["randomizeDGainLower"]
        self.randomize_d_gain_upper = rand_config["randomizeDGainUpper"]
        self.random_obs_noise_e_scale = rand_config["obs_noise_e_scale"]
        self.random_obs_noise_t_scale = rand_config["obs_noise_t_scale"]
        self.random_pose_noise = rand_config["pose_noise_scale"]
        self.random_action_noise_e_scale = rand_config["action_noise_e_scale"]
        self.random_action_noise_t_scale = rand_config["action_noise_t_scale"]
        # stage 2 specific
        self.noisy_rpy_scale = rand_config["noisy_rpy_scale"]
        self.noisy_pos_scale = rand_config["noisy_pos_scale"]

    def _setup_priv_option_config(self, p_config):
        self.enable_priv_obj_position = p_config["enableObjPos"]
        self.enable_priv_obj_mass = p_config["enableObjMass"]
        self.enable_priv_obj_scale = p_config["enableObjScale"]
        self.enable_priv_obj_com = p_config["enableObjCOM"]
        self.enable_priv_obj_friction = p_config["enableObjFriction"]
        self.contact_input_dim = p_config["contact_input_dim"]
        self.contact_form = p_config["contact_form"]
        self.contact_input = p_config["contact_input"]
        self.contact_binarize_threshold = p_config["contact_binarize_threshold"]
        self.enable_priv_obj_orientation = p_config["enable_obj_orientation"]
        self.enable_priv_obj_linvel = p_config["enable_obj_linvel"]
        self.enable_priv_obj_angvel = p_config["enable_obj_angvel"]
        self.enable_priv_fingertip_position = p_config["enable_ft_pos"]
        self.enable_priv_fingertip_orientation = p_config["enable_ft_orientation"]
        self.enable_priv_fingertip_linvel = p_config["enable_ft_linvel"]
        self.enable_priv_fingertip_angvel = p_config["enable_ft_angvel"]
        self.enable_priv_hand_scale = p_config["enable_hand_scale"]
        self.enable_priv_obj_restitution = p_config["enable_obj_restitution"]
        self.enable_priv_tactile = p_config["enable_tactile"]
        self.enable_priv_nut_contact = p_config["enable_nut_contact"]
        self.enable_priv_nut_pos = p_config["enable_nut_pos"]
        self.enable_priv_nut_dof_vel = p_config["enable_nut_dof_vel"]
        self.enable_priv_nut_dof_pos = p_config["enable_nut_dof_pos"]
        self.enable_priv_hand_position = p_config["enable_hand_position"]
        self.enable_priv_hand_orientation = p_config["enable_hand_orientation"]
        self.enable_priv_hand_joint_pos = p_config["enable_hand_joint_pos"]
        self.enable_priv_pgain = p_config["enable_pgain"]
        self.enable_priv_dgain = p_config["enable_dgain"]
        self.enable_priv_screw_joint_friction = p_config.get(
            "enable_screw_joint_friction", False
        )
        self.num_contacts = 0

        self.priv_info_dict = {
            "obj_position": (0, 3),
            "obj_scale": (3, 4),
            "obj_mass": (4, 5),
            "obj_friction": (5, 6),
            "obj_com": (6, 9),
        }

        start_index = 0
        priv_dims = OrderedDict()
        priv_dims["obj_orientation"] = 4
        priv_dims["obj_linvel"] = 3
        priv_dims["obj_angvel"] = 3
        priv_dims["fingertip_position"] = 3 * self.fingers_num
        priv_dims["fingertip_orientation"] = 4 * self.fingers_num
        priv_dims["fingertip_linvel"] = self.fingers_num * 3
        priv_dims["fingertip_angvel"] = self.fingers_num * 3
        priv_dims["hand_scale"] = 1
        priv_dims["obj_restitution"] = 1
        priv_dims["tactile"] = self.num_contacts
        priv_dims["nut_contact"] = 1
        priv_dims["nut_pos"] = 3
        priv_dims["nut_dof_vel"] = 1
        priv_dims["nut_dof_pos"] = 1
        priv_dims["pgain"] = self.numActions
        priv_dims["dgain"] = self.numActions
        priv_dims["hand_joint_pos"] = self.numActions
        priv_dims["hand_orientation"] = 4
        priv_dims["hand_position"] = 3
        priv_dims["screw_joint_friction"] = 1
        for name, dim in priv_dims.items():
            if eval(f"self.enable_priv_{name}"):
                self.priv_info_dict[name] = (start_index, start_index + dim)
                start_index += dim

    def _update_priv_buf(self, env_id, name, value):
        # normalize to -1, 1
        if eval(f"self.enable_priv_{name}"):
            s, e = self.priv_info_dict[name]
            if type(value) is list:
                value = to_torch(value, dtype=torch.float, device=self.device)
            self.priv_info_buf[env_id, s:e] = value

    def _setup_object_info(self, o_config):
        self.object_type = o_config["type"]
        raw_prob = o_config["sampleProb"]
        assert sum(raw_prob) == 1

        # Load object tilt configuration (default to True for backward compatibility)
        self.object_tilt_enabled = o_config.get("object_tilt", True)
        print(f"---- Object Tilt Configuration ----")
        print(f"Object tilt enabled: {self.object_tilt_enabled}")

        primitive_list = self.object_type.split("+")
        print("---- Primitive List ----")
        print(primitive_list)
        self.object_type_prob = []
        self.object_type_list = []
        self.asset_files_dict = {
            "simple_tennis_ball": "assets/ball.urdf",
            "simple_cube": "assets/cube.urdf",
            "simple_cylin4cube": "assets/cylinder4cube.urdf",
        }
        for p_id, prim in enumerate(primitive_list):
            if "screw" in prim:
                subset_name = (
                    self.object_type.split("_")[-1]
                    if "_" in self.object_type
                    else "None"
                )
                screw_compounds = sorted(glob(f"assets/screw/{subset_name}/*.urdf"))
                screw_compound_list = [
                    f"screw_{i}" for i in range(len(screw_compounds))
                ]
                self.object_type_list += screw_compound_list
                for i, name in enumerate(screw_compounds):
                    self.asset_files_dict[f"screw_{i}"] = name.replace("../assets/", "")
                self.object_type_prob += [
                    raw_prob[p_id] / len(screw_compound_list)
                    for _ in screw_compound_list
                ]
            else:
                self.object_type_list += [prim]
                self.object_type_prob += [raw_prob[p_id]]
        print("---- Object List ----")
        print(f"using {len(self.object_type_list)} training objects")
        assert len(self.object_type_list) == len(self.object_type_prob)

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config["env"]["hora"]["propHistoryLen"]
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.point_cloud_sampled_dim = self.config["env"]["hora"][
            "point_cloud_sampled_dim"
        ]
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
        self.object_scale_buf = torch.ones(
            (num_envs,), device=self.device, dtype=torch.float
        ) * float(self.base_obj_scale)
        self.priv_info_buf = torch.zeros(
            (num_envs, self.priv_info_dim), device=self.device, dtype=torch.float
        )
        # for collecting bc data
        self.point_cloud_buf = torch.zeros(
            (num_envs, self.point_cloud_sampled_dim, 3),
            device=self.device,
            dtype=torch.float,
        )
        # fixed noise per-episode, for different hardware have different this value
        self.random_obs_noise_e = torch.zeros(
            (num_envs, self.config["env"]["numActions"]),
            device=self.device,
            dtype=torch.float,
        )
        self.random_action_noise_e = torch.zeros(
            (num_envs, self.config["env"]["numActions"]),
            device=self.device,
            dtype=torch.float,
        )
        # ---- stage 2 buffers
        # stage 2 related buffers
        self.proprio_hist_buf = torch.zeros(
            (num_envs, self.prop_hist_len, self.numActions * 2),
            device=self.device,
            dtype=torch.float,
        )

        # ---- nut termination buffers (10 timesteps history)
        self.nut_dof_pos_history = torch.zeros(
            (num_envs, self.nut_termination_history_len, 1),
            device=self.device,
            dtype=torch.float,
        )
        self.nut_contact_history = torch.zeros(
            (num_envs, self.nut_termination_history_len, 1),
            device=self.device,
            dtype=torch.float,
        )
        self.last_nut_pos = torch.zeros(
            (num_envs,), device=self.device, dtype=torch.float
        )

    def _setup_reward_config(self, r_config):
        # the list
        self.reward_scale_dict = {}
        for k, v in r_config.items():
            if "scale" in k:
                if type(v) is not omegaconf.listconfig.ListConfig:
                    v = [v, v, 0, 0]
                else:
                    assert len(v) == 4
                self.reward_scale_dict[k.replace("_scale", "")] = v
        self.angvel_clip_min = r_config["angvelClipMin"]
        self.angvel_clip_max = r_config["angvelClipMax"]

        # angular velocity penalty threshold curriculum
        angvel_threshold_config = r_config.get("angvelPenaltyThres", 10.0)
        if type(angvel_threshold_config) is not omegaconf.listconfig.ListConfig:
            # No curriculum - use fixed value
            self.angvel_penalty_threshold_curriculum = [
                angvel_threshold_config,
                angvel_threshold_config,
                0,
                0,
            ]
        else:
            # Curriculum enabled - [init_threshold, final_threshold, start_step, end_step]
            assert len(angvel_threshold_config) == 4
            self.angvel_penalty_threshold_curriculum = angvel_threshold_config

        # Initialize with the starting threshold
        self.angvel_penalty_threshold = self.angvel_penalty_threshold_curriculum[0]

    def _create_object_asset(self):
        # object file to asset
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../")
        hand_asset_file = self.config["env"]["asset"]["handAsset"]  # load hand asset

        # load hand asset
        hand_asset_options = gymapi.AssetOptions()
        hand_asset_options.flip_visual_attachments = False
        hand_asset_options.fix_base_link = True
        hand_asset_options.collapse_fixed_joints = False
        hand_asset_options.convex_decomposition_from_submeshes = True
        hand_asset_options.disable_gravity = True
        hand_asset_options.thickness = 0.001
        hand_asset_options.angular_damping = 0.01

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(
            self.sim, asset_root, hand_asset_file, hand_asset_options
        )
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name)
            for name in self.fingertip_body_names
        ]
        missing_fingertips = [
            name
            for name, handle in zip(self.fingertip_body_names, self.fingertip_handles)
            if handle == -1
        ]
        if missing_fingertips:
            raise ValueError(
                f"Missing fingertip rigid bodies in asset {hand_asset_file}: "
                f"{missing_fingertips}"
            )

        # load object asset
        self.object_asset_list = []
        self.asset_point_clouds = []
        for object_type in self.object_type_list:
            object_asset_file = self.asset_files_dict[object_type]
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.fix_base_link = True
            # If we've specified a specific eval object, we only need to load that object.
            eval_object_type = self.config["env"]["object"]["evalObjectType"]
            if eval_object_type is not None and object_type != eval_object_type:
                self.object_asset_list.append(None)
                self.asset_point_clouds.append(None)
                continue

            object_asset = self.gym.load_asset(
                self.sim, asset_root, object_asset_file, object_asset_options
            )
            self.object_asset_list.append(object_asset)
            if "screw" in object_type and self.point_cloud_sampled_dim > 0:
                # Load pre-generated point cloud from NPY file (following cylinder/cube pattern)
                pc_file = os.path.join(
                    asset_root, object_asset_file.replace(".urdf", ".npy")
                )
                if os.path.exists(pc_file):
                    screw_points = np.load(pc_file)

                    # Resample to desired number of points if needed
                    if len(screw_points) != self.point_cloud_sampled_dim:
                        if len(screw_points) > self.point_cloud_sampled_dim:
                            indices = np.random.choice(
                                len(screw_points),
                                self.point_cloud_sampled_dim,
                                replace=False,
                            )
                        else:
                            indices = np.random.choice(
                                len(screw_points),
                                self.point_cloud_sampled_dim,
                                replace=True,
                            )
                        screw_points = screw_points[indices]

                    self.asset_point_clouds.append(screw_points)
                else:
                    # Fallback: simple cylinder if NPY file missing (approximate nut size)
                    fallback_points = (
                        sample_cylinder(1.0) * 0.022
                    )  # Match typical nut radius
                    self.asset_point_clouds.append(fallback_points)
            else:
                # Default fallback for unknown object types
                if self.point_cloud_sampled_dim > 0:
                    fallback_points = sample_cylinder(1.0) * 0.05
                    self.asset_point_clouds.append(fallback_points)

        assert any([x is not None for x in self.object_asset_list])

    def _parse_hand_dof_props(self):
        self.num_xhand_hand_dofs = self.gym.get_asset_dof_count(self.hand_asset)
        xhand_hand_dof_props = self.gym.get_asset_dof_properties(self.hand_asset)

        self.xhand_hand_dof_lower_limits = []
        self.xhand_hand_dof_upper_limits = []

        (
            xhand_dof_lower_limits,
            xhand_dof_upper_limits,
            xhand_effort_limits,
            xhand_velocity_limits,
        ) = self._default_hand_dof_props()
        lower_override = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("dofLowerLimits"),
            self.num_xhand_hand_dofs,
            "env.asset.dofLowerLimits",
        )
        upper_override = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("dofUpperLimits"),
            self.num_xhand_hand_dofs,
            "env.asset.dofUpperLimits",
        )
        effort_override = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("dofEffortLimits"),
            self.num_xhand_hand_dofs,
            "env.asset.dofEffortLimits",
        )
        velocity_override = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("dofVelocityLimits"),
            self.num_xhand_hand_dofs,
            "env.asset.dofVelocityLimits",
        )
        if lower_override is not None:
            xhand_dof_lower_limits = lower_override
        if upper_override is not None:
            xhand_dof_upper_limits = upper_override
        if effort_override is not None:
            xhand_effort_limits = effort_override
        if velocity_override is not None:
            xhand_velocity_limits = velocity_override
        self.xhand_dof_lower_limits = np.array(xhand_dof_lower_limits)
        if self.config["env"]["object"]["thumb_range_limit"]:
            xhand_dof_upper_limits[9] = 1.73
            xhand_dof_lower_limits[9] = 0.6
        self.xhand_dof_upper_limits = np.array(xhand_dof_upper_limits)

        for i in range(self.num_xhand_hand_dofs):
            # Set the joint limits based on the URDF
            xhand_hand_dof_props["lower"][i] = xhand_dof_lower_limits[i]
            xhand_hand_dof_props["upper"][i] = xhand_dof_upper_limits[i]
            self.xhand_hand_dof_lower_limits.append(xhand_dof_lower_limits[i])
            self.xhand_hand_dof_upper_limits.append(xhand_dof_upper_limits[i])

            # Set the effort limit
            xhand_hand_dof_props["effort"][i] = xhand_effort_limits[i]
            if xhand_velocity_limits is not None:
                xhand_hand_dof_props["velocity"][i] = xhand_velocity_limits[i]

            # Set controller properties
            if self.torque_control:
                xhand_hand_dof_props["stiffness"][i] = 0.0
                xhand_hand_dof_props["damping"][i] = 0.0
                xhand_hand_dof_props["driveMode"][i] = gymapi.DOF_MODE_EFFORT
            else:
                xhand_hand_dof_props["stiffness"][i] = self.config["env"]["controller"][
                    "pgain"
                ]
                xhand_hand_dof_props["damping"][i] = self.config["env"]["controller"][
                    "dgain"
                ]

            # From URDF dynamics, all joints have damping=1 and friction=1
            xhand_hand_dof_props["friction"][i] = 0.01
            xhand_hand_dof_props["armature"][i] = 0.001

        self.xhand_hand_dof_lower_limits = to_torch(
            self.xhand_hand_dof_lower_limits, device=self.device
        )
        self.xhand_hand_dof_upper_limits = to_torch(
            self.xhand_hand_dof_upper_limits, device=self.device
        )
        return xhand_hand_dof_props

    def _init_object_pose(self):
        xhand_hand_start_pose = gymapi.Transform()
        hand_root_pos = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPos"), 3, "env.asset.handRootPos"
        )
        if hand_root_pos is None:
            hand_root_pos = [0.0, 0.0, 0.21]
        xhand_hand_start_pose.p = gymapi.Vec3(*hand_root_pos)

        hand_root_quat = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootQuat"), 4, "env.asset.handRootQuat"
        )
        hand_root_rpy = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootRPY"), 3, "env.asset.handRootRPY"
        )
        if hand_root_quat is not None:
            xhand_hand_start_pose.r = gymapi.Quat(*hand_root_quat)
        elif hand_root_rpy is not None:
            quat = quat_from_euler_xyz(
                torch.tensor([hand_root_rpy[0]], dtype=torch.float),
                torch.tensor([hand_root_rpy[1]], dtype=torch.float),
                torch.tensor([hand_root_rpy[2]], dtype=torch.float),
            )[0]
            xhand_hand_start_pose.r = gymapi.Quat(
                float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])
            )
        else:
            xhand_hand_start_pose.r = gymapi.Quat.from_axis_angle(
                gymapi.Vec3(1, 0, 0), np.pi / 2 - 25 * (np.pi / 180)
            )

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_init_pos = self._resolve_numeric_vector(
            self.object_cfg.get("init_pos"), 3, "env.object.init_pos"
        )
        if object_init_pos is None:
            object_init_pos = [0.0, 0.0, 0.0]
        object_start_pose.p.x = float(object_init_pos[0])
        object_start_pose.p.y = float(object_init_pos[1])
        object_start_pose.p.z = float(object_init_pos[2])

        return xhand_hand_start_pose, object_start_pose

    def _default_apply_action_mask(self):
        return True

    def _default_fingertip_body_names(self):
        return [
            "left_hand_index_rota_tip",
            "left_hand_mid_tip",
            "left_hand_pinky_tip",
            "left_hand_ring_tip",
            "left_hand_thumb_rota_tip",
        ]

    def _default_hand_dof_props(self):
        xhand_dof_lower_limits = [
            -0.175,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -1.05,
            -0.17,
        ]
        xhand_dof_upper_limits = [
            0.175,
            1.92,
            1.92,
            1.92,
            1.92,
            1.92,
            1.92,
            1.92,
            1.92,
            1.83,
            1.57,
            1.83,
        ]
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
        ]
        if len(xhand_effort_limits) < len(xhand_dof_lower_limits):
            xhand_effort_limits = xhand_effort_limits + [
                xhand_effort_limits[-1]
                for _ in range(len(xhand_dof_lower_limits) - len(xhand_effort_limits))
            ]
        return (
            xhand_dof_lower_limits,
            xhand_dof_upper_limits,
            xhand_effort_limits,
            None,
        )

    def _default_joint_values(self, dof_names):
        if self.config["env"]["initPose"] == "nutbolt_inclined":
            joint_values = OrderedDict(
                [
                    ("left_hand_index_bend_joint", -0.17),
                    ("left_hand_index_joint1", 1.1),
                    ("left_hand_index_joint2", 0.4),
                    ("left_hand_mid_joint1", 1.1),
                    ("left_hand_mid_joint2", 0.4),
                    ("left_hand_pinky_joint1", 0),
                    ("left_hand_pinky_joint2", 0),
                    ("left_hand_ring_joint1", 0),
                    ("left_hand_ring_joint2", 0),
                    ("left_hand_thumb_bend_joint", 1.3),
                    ("left_hand_thumb_rota_joint1", 0.5),
                    ("left_hand_thumb_rota_joint2", 0.45),
                ]
            )
        elif self.config["env"]["initPose"] in (
            "screwdriver_inclined",
            "lightbulb_inclined",
        ):
            joint_values = OrderedDict(
                [
                    ("left_hand_index_bend_joint", -0.036),
                    ("left_hand_index_joint1", 1.15),
                    ("left_hand_index_joint2", 0.5),
                    ("left_hand_mid_joint1", 0.925),
                    ("left_hand_mid_joint2", 0.58),
                    ("left_hand_pinky_joint1", 0),
                    ("left_hand_pinky_joint2", 0),
                    ("left_hand_ring_joint1", 1.3),
                    ("left_hand_ring_joint2", 0.43),
                    ("left_hand_thumb_bend_joint", 1.455),
                    ("left_hand_thumb_rota_joint1", 0.817),
                    ("left_hand_thumb_rota_joint2", 0.154),
                ]
            )
        else:
            raise ValueError(
                f"Unsupported initPose: {self.config['env']['initPose']} for XHandHora"
            )
        if list(dof_names) != list(joint_values.keys()):
            raise ValueError(
                f"Unexpected DOF order for init pose resolution.\n"
                f"Expected: {list(joint_values.keys())}\n"
                f"Got: {list(dof_names)}"
            )
        return joint_values

    def _resolve_hand_init_pose(self, dof_names):
        custom_init = self.config["env"].get("customInitDofPos", None)
        if custom_init is not None:
            custom_values = self._resolve_numeric_vector(
                custom_init, len(dof_names), "env.customInitDofPos"
            )
            return OrderedDict(zip(dof_names, custom_values))

        hand_init_pose = self.hand_asset_cfg.get("handInitPose")
        if hand_init_pose is not None:
            if hasattr(hand_init_pose, "items"):
                pose_map = {str(k): float(v) for k, v in hand_init_pose.items()}
                missing = [name for name in dof_names if name not in pose_map]
                extra = [name for name in pose_map.keys() if name not in dof_names]
                if missing or extra:
                    raise ValueError(
                        "env.asset.handInitPose keys must match hand DOF names. "
                        f"Missing={missing}, Extra={extra}"
                    )
                return OrderedDict((name, pose_map[name]) for name in dof_names)
            init_values = self._resolve_numeric_vector(
                hand_init_pose, len(dof_names), "env.asset.handInitPose"
            )
            return OrderedDict(zip(dof_names, init_values))

        return self._default_joint_values(dof_names)

    def _default_object_init_pos(self):
        if self.config["env"]["initPose"] == "nutbolt_inclined":
            return [0.0175, 0.06, 0.0]
        if self.config["env"]["initPose"] in (
            "screwdriver_inclined",
            "lightbulb_inclined",
        ):
            return [0.009, 0.06, 0.0]
        raise ValueError(
            f"Unsupported initPose for object initialization: {self.config['env']['initPose']}"
        )

    def _sample_object_init_positions(self, batch_size):
        base_pos = self._resolve_numeric_vector(
            self.object_cfg.get("init_pos"), 3, "env.object.init_pos"
        )
        if base_pos is None:
            base_pos = self._default_object_init_pos()
        noise_cfg = self.object_cfg.get("init_pos_noise", [0.0075, 0.0075, 0.0])
        noise_scale = self._resolve_numeric_vector(
            noise_cfg, 3, "env.object.init_pos_noise"
        )
        base = torch.tensor(base_pos, device=self.device, dtype=torch.float).unsqueeze(0)
        noise = (
            torch.rand((batch_size, 3), device=self.device, dtype=torch.float)
            * torch.tensor(noise_scale, device=self.device, dtype=torch.float)
        )
        return base.repeat(batch_size, 1) + noise

    def _sample_hand_root_quats(self, batch_size):
        hand_root_quat = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootQuat"), 4, "env.asset.handRootQuat"
        )
        if hand_root_quat is not None:
            quat = torch.tensor(hand_root_quat, device=self.device, dtype=torch.float)
            quat = quat / torch.norm(quat)
            return quat.unsqueeze(0).repeat(batch_size, 1)

        hand_root_rpy = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootRPY"), 3, "env.asset.handRootRPY"
        )
        if hand_root_rpy is not None:
            roll = torch.full(
                (batch_size,),
                float(hand_root_rpy[0]),
                device=self.device,
                dtype=torch.float,
            )
            pitch = torch.full(
                (batch_size,),
                float(hand_root_rpy[1]),
                device=self.device,
                dtype=torch.float,
            )
            yaw = torch.full(
                (batch_size,),
                float(hand_root_rpy[2]),
                device=self.device,
                dtype=torch.float,
            )
            return quat_from_euler_xyz(roll, pitch, yaw)
        return None

    def _sample_hand_root_positions(self, env_ids):
        hand_root_pos = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPos"), 3, "env.asset.handRootPos"
        )
        if hand_root_pos is None:
            original_pos = torch.zeros(
                (len(env_ids), 3), device=self.device, dtype=torch.float
            )
            original_pos[:, 2] = 0.21
            return original_pos + torch.rand_like(original_pos) * 0.001

        pos = torch.tensor(hand_root_pos, device=self.device, dtype=torch.float)
        pos = pos.unsqueeze(0).repeat(len(env_ids), 1)

        z_scale_comp = float(self.hand_asset_cfg.get("handRootPosZScaleComp", 0.0))
        if z_scale_comp != 0.0 and self.randomize_scale:
            num_scales = len(self.randomize_scale_list)
            scale_ids = (env_ids % num_scales).tolist()
            scale_tensor = torch.tensor(
                [self.randomize_scale_list[int(i)] for i in scale_ids],
                device=self.device,
                dtype=torch.float,
            )
            pos[:, 2] += z_scale_comp * (scale_tensor - 1.0)

        pos_noise = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPosNoise", [0.0, 0.0, 0.0]),
            3,
            "env.asset.handRootPosNoise",
        )
        if any(abs(v) > 0 for v in pos_noise):
            pos += (
                torch.rand((len(env_ids), 3), device=self.device, dtype=torch.float)
                * torch.tensor(pos_noise, device=self.device, dtype=torch.float)
            )
        return pos

    def _resolve_numeric_vector(self, value, expected_len, field_name):
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return [float(value) for _ in range(expected_len)]
        values = [float(x) for x in list(value)]
        if len(values) != expected_len:
            raise ValueError(
                f"{field_name} must have length {expected_len}; got {len(values)}"
            )
        return values

    def _cfg_bool(self, value, default=False):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            token = value.strip().lower()
            if token in ("true", "1", "yes", "y", "on"):
                return True
            if token in ("false", "0", "no", "n", "off"):
                return False
        return default

    @staticmethod
    def quat_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
        norms = torch.norm(quaternions[..., :3], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., 3:])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        return quaternions[..., :3] / sin_half_angles_over_angles


def compute_hand_reward(
    rotate_reward,
    rotate_reward_scale: float,
    pose_diff_penalty,
    pose_diff_penalty_scale: float,
    torque_penalty,
    torque_pscale: float,
    work_penalty,
    work_pscale: float,
    z_dist_penalty,
    z_dist_penalty_scale: float,
    rotate_penalty,
    rotate_penalty_scale: float,
    proximity_reward,
    proximity_reward_scale: float,
):
    reward = rotate_reward_scale * rotate_reward
    reward = reward + pose_diff_penalty * pose_diff_penalty_scale
    reward = reward + torque_penalty * torque_pscale
    reward = reward + work_penalty * work_pscale
    reward = reward + z_dist_penalty * z_dist_penalty_scale
    reward = reward + rotate_penalty * rotate_penalty_scale
    reward = reward + proximity_reward * proximity_reward_scale
    return reward
