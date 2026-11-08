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
from isaacgym import gymutil
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


class XHandPasini(VecTask):
    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.config = config

        # Debug toggles (Hydra overrides). Must be set before super().__init__
        # because VecTask.__init__ -> create_sim() -> _create_envs() uses them.
        env_cfg = self.config.get("env", {})
        self.hand_asset_cfg = env_cfg.get("asset", {})

        def _cfg_bool(v, default=False):
            if v is None:
                return default
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(v)
            if isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "1", "yes", "y", "on"):
                    return True
                if s in ("false", "0", "no", "n", "off"):
                    return False
            return default

        self.debug_disable_object_collisions = _cfg_bool(
            env_cfg.get("debug_disable_object_collisions", False), default=False
        )
        self.debug_use_asset_collision_filter = _cfg_bool(
            env_cfg.get("debug_use_asset_collision_filter", True), default=True
        )
        self.debug_print_top_hand_contacts = _cfg_bool(
            env_cfg.get("debug_print_top_hand_contacts", False), default=False
        )
        self.apply_action_mask = _cfg_bool(
            env_cfg.get("apply_action_mask", True), default=True
        )
        self.custom_action_mask_indices = env_cfg.get("action_mask_indices")
        # Historical Pasini resets add +/-5% of each joint range even when
        # domain randomization is disabled.  Keep that historical default, but
        # let nominal tasks (notably E002-A) request a truly fixed reset.
        self.hand_init_dof_noise_fraction = float(
            env_cfg.get("hand_init_dof_noise_fraction", 0.10)
        )
        if self.hand_init_dof_noise_fraction < 0.0:
            raise ValueError("env.hand_init_dof_noise_fraction must be non-negative")

        # When True, log extra debug metrics (lots of `debug/*` + `term/*`).
        # Keep this off by default to make retraining runs clean.
        self.log_debug_metrics = _cfg_bool(
            env_cfg.get("log_debug_metrics", False), default=False
        )
        # before calling init in VecTask, need to do
        # 0. setup the dim info
        self.numActions = config["env"]["numActions"]
        self.fingers_num = 4
        # 1. setup randomization
        self._setup_domain_rand_config(config["env"]["randomization"])
        # 2. setup privileged information
        self._setup_priv_option_config(config["env"]["privInfo"])
        # 3. setup object assets
        self._setup_object_info(config["env"]["object"])
        # 4. setup rewards
        self._setup_reward_config(config["env"]["reward"])
        # R1 is deliberately opt-in so the historical M24 reward remains
        # bit-for-bit usable for existing checkpoints and viewers.
        self._setup_r1_motion_config(env_cfg.get("r1_motion", {}))
        self._setup_e002_motion_config(env_cfg.get("e002_motion", {}))
        # unclassified config
        self.base_obj_scale = config["env"]["baseObjScale"]
        self.aggregate_mode = self.config["env"]["aggregateMode"]
        self.up_axis = "z"
        self.rotation_axis = config["env"]["rotation_axis"]
        self.reset_z_threshold = self.config["env"]["reset_z_threshold"]
        self.reset_dist_threshold = self.config["env"]["reset_dist_threshold"]
        # Proximity shaping threshold (can be independent from termination threshold).
        # If not provided, fall back to reset_dist_threshold for backward compatibility.
        self.proximity_dist_threshold = float(
            self.config["env"].get(
                "proximity_dist_threshold", self.reset_dist_threshold
            )
        )
        # Optional low-pass filtering on joint targets to reduce high-frequency “抽搐式”动作.
        # 0.0 disables; closer to 1.0 means heavier smoothing (more lag).
        ctrl_cfg = self.config["env"].get("controller", {})
        try:
            self.target_smoothing_alpha = float(ctrl_cfg.get("target_smoothing_alpha", 0.0))
        except Exception:
            self.target_smoothing_alpha = 0.0
        self.target_smoothing_alpha = float(
            max(0.0, min(1.0, self.target_smoothing_alpha))
        )
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
        # Optionally disable pose-diff penalty for thumb joints by masking them out.
        # Default behavior keeps the historical "Hora semantics": thumb is not penalized for deviating from init pose.
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        thumb_indices = [i for i, name in enumerate(dof_names) if "thumb" in name]
        self.pose_diff_penalty_mask = torch.ones(
            self.num_actions, device=self.device, dtype=torch.float
        )
        mask_thumb = bool(config["env"].get("mask_thumb_pose_diff_penalty", True))
        if mask_thumb and len(thumb_indices) > 0:
            self.pose_diff_penalty_mask[thumb_indices] = 0.0
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
        expected_dof_names = [
            "right_index_joint_0",
            "right_index_joint_1",
            "right_index_joint_2",
            "right_index_joint_3",
            "right_middle_joint_0",
            "right_middle_joint_1",
            "right_middle_joint_2",
            "right_middle_joint_3",
            "right_ring_joint_0",
            "right_ring_joint_1",
            "right_ring_joint_2",
            "right_ring_joint_3",
            "right_thumb_joint_0",
            "right_thumb_joint_1",
            "right_thumb_joint_2",
            "right_thumb_joint_3",
        ]
        dof_names = self.gym.get_asset_dof_names(self.hand_asset)
        self.hand_dof_names = list(dof_names)
        if list(dof_names) != expected_dof_names:
            raise ValueError(
                f"Unexpected Pasini DOF order.\nExpected: {expected_dof_names}\nGot: {list(dof_names)}"
            )

        if self.config["env"]["initPose"] == "nutbolt_inclined":
            joint_values = OrderedDict(
                zip(
                    dof_names,
                    [
                        -0.17,
                        1.40,
                        0.0,
                        0.4,
                        0.0,
                        1.4,
                        0.0,
                        0.4,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        1.4,
                        0.5,
                        0.45,
                    ],
                )
            )
        elif self.config["env"]["initPose"] in ("bulb_inclined", "lightbulb_inclined"):
            # Bulb twisting pose: keep a screwdriver-like pre-grasp by default.
            # This is intentionally separated from screwdriver/nutbolt so the bulb task can be tuned
            # independently (and can still be overridden by env.customInitDofPos).
            joint_values = OrderedDict(
                zip(
                    dof_names,
                    [
                        0.31805448815226555,
                        0.7768147802352905,
                        0.2902536576986313,
                        1.0633865308761597,
                        0.03999972581863403,
                        0.5151662063598633,
                        0.7383003902435303,
                        0.7970210313796997,
                        -0.17020170390605927,
                        0.8792879581451416,
                        0.1139976978302002,
                        1.4113634824752808,
                        -0.35274748712778091,
                        1.57,
                        0.07279872894287,
                        0.4473716020584106,
                    ],
                )
            )
        elif self.config["env"]["initPose"] == "screwdriver_inclined":
            joint_values = OrderedDict(
                zip(
                    dof_names,
[0.34266707360744476, 
  1.2325279521942139, 
  0.28289711773395538, 
  1.1292990016937256, 
  0.00888176321983337, 
  0.7502785110473633, 
  0.8105450248718262, 
  0.825642421245575, 
  -0.3587684601545334, 
  1.3127488565444946, 
  0.0030379545688629, 
  1.5381801319122314, 
  -0.35994077682495117, 
  1.5656383037567139, 
  0.5520286703109741, 
  0.531196631193161],
                )
            )
        else:
            raise ValueError(
                f"Unsupported initPose: {self.config['env']['initPose']} for Pasini"
            )

        # Optional: override the initial DOF pose from config to support pose bootstrapping.
        # `customInitDofPos` remains the highest-priority legacy override.  The
        # named `asset.handInitPose` form is what the interactive pose tuner
        # writes, and avoids silently depending on the URDF DOF order.
        custom_init = self.config["env"].get("customInitDofPos", None)
        if custom_init is not None:
            if len(custom_init) != len(dof_names):
                raise ValueError(
                    f"env.customInitDofPos must have length {len(dof_names)}; got {len(custom_init)}"
                )
            joint_values = OrderedDict(zip(dof_names, [float(x) for x in custom_init]))
        else:
            hand_init_pose = self.hand_asset_cfg.get("handInitPose")
            if hand_init_pose is not None:
                if hasattr(hand_init_pose, "items"):
                    pose_map = {str(k): float(v) for k, v in hand_init_pose.items()}
                    missing = [name for name in dof_names if name not in pose_map]
                    extra = [name for name in pose_map if name not in dof_names]
                    if missing or extra:
                        raise ValueError(
                            "env.asset.handInitPose keys must match Pasini DOF names. "
                            f"Missing={missing}, Extra={extra}"
                        )
                    joint_values = OrderedDict(
                        (name, pose_map[name]) for name in dof_names
                    )
                else:
                    values = self._resolve_numeric_vector(
                        hand_init_pose,
                        len(dof_names),
                        "env.asset.handInitPose",
                    )
                    joint_values = OrderedDict(zip(dof_names, values))

        self.joint_values_lst = list(joint_values.values())
        hand_dof_dim = self.num_xhand_hand_dofs
        obj_start = hand_dof_dim
        pose_dim = hand_dof_dim + 7

        for s in self.randomize_scale_list:
            scale_key = str(s)
            random_pose_data = torch.zeros(
                (num_random_poses, pose_dim), device=self.device, dtype=torch.float
            )

            for i in range(hand_dof_dim):
                random_pose_data[:, i] = (
                    torch.ones(num_random_poses, device=self.device)
                    * self.joint_values_lst[i]
                )

            random_pose_data[:, obj_start + 0] = 0.0  # x
            random_pose_data[:, obj_start + 1] = 0.0  # y
            random_pose_data[:, obj_start + 2] = (
                self.reset_z_threshold + 0.1
            )  # z - slightly above threshold
            random_pose_data[:, obj_start + 3 : obj_start + 6] = 0.0
            random_pose_data[:, obj_start + 6] = 1.0
            quat_norm = torch.norm(
                random_pose_data[:, obj_start + 3 : obj_start + 7], dim=1, keepdim=True
            )
            random_pose_data[:, obj_start + 3 : obj_start + 7] = (
                random_pose_data[:, obj_start + 3 : obj_start + 7] / quat_norm
            )

            self.saved_grasping_states[scale_key] = random_pose_data
            print(
                f"Generated {num_random_poses} random initial poses for Pasini at scale {s}"
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

        # Cache hand rigid body names (hand-only indices: 0..num_xhand_hand_bodies-1)
        self.hand_rb_names = list(self.gym.get_asset_rigid_body_names(self.hand_asset))

        # Useful groupings for self-collision debugging (substring match)
        self.thumb_rb_indices = [
            i for i, name in enumerate(self.hand_rb_names) if "thumb" in name
        ]
        self.index_rb_indices = [
            i for i, name in enumerate(self.hand_rb_names) if "index" in name
        ]
        self.middle_rb_indices = [
            i for i, name in enumerate(self.hand_rb_names) if "middle" in name
        ]
        self.ring_rb_indices = [
            i for i, name in enumerate(self.hand_rb_names) if "ring" in name
        ]

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

            # # add hand
            # # - group=-1 uses asset collision filters (usually enables finger-finger self-collision)
            # # - group=1, filter=1 disables self-collisions for bodies in the same group
            # if self.debug_use_asset_collision_filter:
            #     hand_group, hand_filter = -1, 1
            # else:
            #     hand_group, hand_filter = 1, 1

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
                obj_restitution = np.random.uniform(0, 1)

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

            # Debug: isolate pure hand self-collision by disabling all object collisions.
            # This keeps the object for observation/state, but removes contact constraints.
            if self.debug_disable_object_collisions:
                obj_shape_props = self.gym.get_actor_rigid_shape_properties(
                    env_ptr, object_handle
                )
                for p in obj_shape_props:
                    p.filter = 0
                self.gym.set_actor_rigid_shape_properties(
                    env_ptr, object_handle, obj_shape_props
                )

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
            if self.config["env"]["initPose"] in (
                "nutbolt_inclined",
                "bulb_inclined",
                "lightbulb_inclined",
                "screwdriver_inclined",
            ):
                self.object_x, self.object_y, self.object_z = (
                    self._sample_object_init_pos()
                )

            sampled_pose[:, self.numActions + 0] = (
                self.object_x
            )  # X (left-right: postive-nagative)
            sampled_pose[:, self.numActions + 1] = (
                self.object_y
            )  # Y (forward-backward: nagative-positive)
            sampled_pose[:, self.numActions + 2] = (
                self.object_z
            )  # Z (above palm, so it can fall downward)

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
            # Asset limits are stored as NumPy arrays for Gym property setup;
            # reset noise is sampled on the simulation device.
            dof_lower = torch.as_tensor(
                self.xhand_dof_lower_limits, device=self.device, dtype=pos.dtype
            )
            dof_upper = torch.as_tensor(
                self.xhand_dof_upper_limits, device=self.device, dtype=pos.dtype
            )
            half_widths = self.hand_init_dof_noise_fraction * (dof_upper - dof_lower) / 2.0
            hand_joint_noise = (torch.rand_like(half_widths) * 2 - 1) * half_widths
            pos = pos + hand_joint_noise
            initial_noise = float(self.config["env"].get("e002_init_joint_noise_rad", 0.0))
            if self.e002_motion_enabled and initial_noise > 0:
                from dexscrew.tasks.e002_initial_state import perturb_initial_joints
                pos = perturb_initial_joints(
                    pos, dof_lower, dof_upper, initial_noise,
                    float(self.config["env"].get("e002_nominal_reset_fraction", 0.5)))
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

        # X-axis rotation,红线逆时针为正
        random_degrees_x = (
            torch.rand(len(env_ids), device=self.device) * 10 + 90
        )  # randomize orientation bewteen 20-30 degrees
        angles_x = np.pi / 2 - random_degrees_x * (np.pi / 180)
        cos_half_x, sin_half_x = torch.cos(angles_x / 2), torch.sin(angles_x / 2)

        # Y-axis rotation
        random_degrees_y = (
            torch.rand(len(env_ids), device=self.device) * 10 - 5 - 90
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

        configured_quats = self._configured_hand_root_quats(len(env_ids))
        if configured_quats is not None:
            quats = configured_quats

        hand_env_indices = self.hand_indices[env_ids]
        self.root_state_tensor[hand_env_indices, 3:7] = quats
        self._update_priv_buf(env_id=env_ids, name="hand_orientation", value=quats)

        # # position randomization
        # # We want the "Palm Center" to be at (0, 0, 1.0), not the Wrist (Root).
        # # Based on URDF, fingers start at x=0.135. Let's define palm center at x=0.10 local.
        # target_pos = torch.zeros_like(self.root_state_tensor[hand_env_indices, :3])
        # target_pos[:, 2] = 0.57  # Target Z height for the palm center
        # target_pos[:, :3] = torch.tensor([0.13, -0.06, 0.57], device=self.device)
        # # 相对于isaacgym环境坐标系作偏移，绿y反红x正

        # position randomization
        pos = self._configured_hand_root_positions(len(env_ids))
        if pos is None:
            pos = torch.zeros_like(self.root_state_tensor[hand_env_indices, :3])
            pos[:, :3] = torch.as_tensor(
                [0.14, 0.072, 0.177], device=self.device, dtype=pos.dtype
            )

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

        if self.e002_motion_enabled:
            # The articulated object owns its nut DOF; resetting only the hand
            # actor leaves the previous episode's nut angle/velocity in PhysX.
            self.dof_state.view(self.num_envs, -1, 2)[env_ids, self.num_xhand_hand_dofs:, :] = 0
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
            gymtorch.unwrap_tensor(all_actor_indices if self.e002_motion_enabled else hand_indices),
            len(all_actor_indices) if self.e002_motion_enabled else len(env_ids),
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
        self.r1_phase[env_ids] = 0
        self.r1_phase_step[env_ids] = 0
        self.r1_phase_nut_pos[env_ids] = self.nut_dof_pos[env_ids].squeeze(-1)
        self.r1_contact_state[env_ids] = False
        self.r1_release_step[env_ids] = -1
        self.r1_release_lag[env_ids] = 0.0
        self.r1_transition_count[env_ids] = 0
        self.r1_last_transition_reason[env_ids] = 0
        self.e002_theta_start[env_ids] = 0.0
        self.e002_theta_prev[env_ids] = 0.0
        self.e002_high_water[env_ids] = 0.0
        self.e002_paid_high_water[env_ids] = 0.0
        self.e002_gross_travel[env_ids] = 0.0
        self.e002_reverse_travel[env_ids] = 0.0
        self.e002_contact_state[env_ids] = False
        self.e002_pair_contact_streak[env_ids] = 0
        self.e002_pair_contact_loss_streak[env_ids] = 0
        self.e002_contact_acquired[env_ids] = False
        self.e002_contact_potential_prev[env_ids] = 0.0
        self.e002_drive_step[env_ids] = 0
        self.e002_valid_pair_progress[env_ids] = 0.0
        self.e002_normal_proxy_progress[env_ids] = 0.0
        self.e002_target_validated[env_ids] = False
        self.e002_target_hold[env_ids] = 0
        self.e002_success[env_ids] = False
        self.e002_milestone_paid[env_ids] = False
        self.e002_termination_reason[env_ids] = 0
        if self.e002_cycle is not None:
            self.e002_cycle.reset(env_ids)

        self.at_reset_buf[env_ids] = 1
        if self.e002_demo is not None:
            self.e002_demo.apply(self, env_ids, gymtorch)

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

        if self.e002_motion_enabled:
            # Build reset observations from this episode's pose history, not
            # zeros at startup or the previous episode's history after a reset.
            reset_ids = self.at_reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if self.e002_demo is not None:
                reset_ids = reset_ids[~self.e002_demo.active[reset_ids]]
            reset_pose = self.init_pose_buf[reset_ids, : self.num_actions].unsqueeze(1)
            self.obs_buf_lag_history[reset_ids, :, : self.numActions] = reset_pose
            self.obs_buf_lag_history[reset_ids, :, self.numActions : self.numActions * 2] = reset_pose

        t_buf = (
            self.obs_buf_lag_history[:, -3:, : self.numActions * 2].reshape(
                self.num_envs, -1
            )
        ).clone()
        self.obs_buf[:, : t_buf.shape[1]] = t_buf  # [1, 96]
        if self.e002_motion_enabled:
            # Fixed `drive` command: [drive, release, return, regrasp] plus
            # normalized elapsed phase. B/C reuse this same actor shape.
            tail_start = self.numActions * 2 * 3
            self.obs_buf[:, tail_start : tail_start + 4] = 0.0
            self.obs_buf[:, tail_start] = 1.0
            self.obs_buf[:, tail_start + 4] = torch.clamp(
                self.progress_buf.float() / max(self.e002_drive_deadline_steps, 1),
                min=0.0,
                max=1.0,
            )
            if self.e002_cycle is not None:
                self.obs_buf[:, tail_start:tail_start + 5] = self.e002_cycle.command()

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
        history_reset_ids = at_reset_env_ids
        if self.e002_demo is not None:
            history_reset_ids = history_reset_ids[~self.e002_demo.active[history_reset_ids]]
        self.obs_buf_lag_history[history_reset_ids, :, 0 : self.numActions] = (
            self.init_pose_buf[history_reset_ids, : self.num_actions].unsqueeze(1)
        )
        self.obs_buf_lag_history[
            history_reset_ids, :, self.numActions : self.numActions * 2
        ] = self.init_pose_buf[history_reset_ids, : self.num_actions].unsqueeze(1)

        # velocity reset
        self.obj_linvel_at_cf[at_reset_env_ids] = self.object_linvel[at_reset_env_ids]
        self.obj_angvel_at_cf[at_reset_env_ids] = self.object_angvel[at_reset_env_ids]
        self.ft_linvel_at_cf[at_reset_env_ids] = self.fingertip_linvel[at_reset_env_ids]
        self.ft_angvel_at_cf[at_reset_env_ids] = self.fingertip_angvel[at_reset_env_ids]
        # Fix shape mismatch: nut_dof_vel_cf is (N, 1), nut_dof_vel is (N, 1)
        # But indexing with at_reset_env_ids might cause dimension issues if not handled carefully
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

    def _compute_r1_motion_reward(self, nut_pos, nut_dof_vel):
        """Return R1-gated rotation credit, phase shaping, and diagnostics.

        The torque sign is defined as the force *on the nut* about the configured
        nut axis. `force_sign=-1` converts Isaac Gym's force on the fingertip to
        this convention. Contact is hysteretic in both force and distance.
        """
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        fingertip_force = self.contact_forces[:, self.fingertip_handles, :]
        # Pasini ordering is index, middle, ring, thumb.
        pair_pos = fingertip_pos[:, [0, 3], :]
        pair_force = fingertip_force[:, [0, 3], :]
        radial = pair_pos - nut_pos.unsqueeze(1)
        force_mag = torch.norm(pair_force, dim=-1)
        dist = torch.norm(radial, dim=-1)
        force_on_nut = pair_force * self.r1_force_sign
        torque_vec = torch.cross(radial, force_on_nut, dim=-1)
        axis = self.rot_axis_buf.unsqueeze(1).expand_as(torque_vec)
        signed_torque = (torque_vec * axis).sum(dim=-1)
        positive_torque = torch.clamp(signed_torque, min=0.0)

        force_conf = torch.clamp(
            (force_mag - self.r1_contact_force_off)
            / (self.r1_contact_force_on - self.r1_contact_force_off),
            min=0.0,
            max=1.0,
        )
        dist_conf = torch.clamp(
            (self.r1_contact_dist_off - dist)
            / (self.r1_contact_dist_off - self.r1_contact_dist_on),
            min=0.0,
            max=1.0,
        )
        contact_conf = torch.minimum(force_conf, dist_conf)
        contact_on = (force_mag >= self.r1_contact_force_on) & (
            dist <= self.r1_contact_dist_on
        )
        contact_still_on = (force_mag >= self.r1_contact_force_off) & (
            dist <= self.r1_contact_dist_off
        )
        old_contact = self.r1_contact_state.clone()
        contact = torch.where(old_contact, contact_still_on, contact_on)
        self.r1_contact_state[:] = contact

        torque_conf = torch.clamp(
            positive_torque / self.r1_torque_clip, min=0.0, max=1.0
        )
        pair_gate = torch.minimum(
            contact_conf[:, 0] * torque_conf[:, 0],
            contact_conf[:, 1] * torque_conf[:, 1],
        )
        pair_contact = contact.all(dim=-1)
        pair_positive_torque = (signed_torque >= self.r1_torque_min).all(dim=-1)

        old_phase = self.r1_phase.clone()
        phase = old_phase.clone()
        phase_step = self.r1_phase_step + 1
        transition_reason = torch.zeros_like(phase)
        nut_pos_scalar = self.nut_dof_pos.squeeze(-1)

        grasp_to_drive = (old_phase == 0) & pair_contact & pair_positive_torque
        if torch.any(grasp_to_drive):
            phase[grasp_to_drive] = 1
            phase_step[grasp_to_drive] = 0
            self.r1_phase_nut_pos[grasp_to_drive] = nut_pos_scalar[grasp_to_drive]
            transition_reason[grasp_to_drive] = 1

        drive_progress = nut_pos_scalar - self.r1_phase_nut_pos
        drive_to_unload = (old_phase == 1) & (
            (drive_progress >= self.r1_drive_stroke)
            | (phase_step >= self.r1_drive_max_steps)
        )
        if torch.any(drive_to_unload):
            phase[drive_to_unload] = 2
            phase_step[drive_to_unload] = 0
            self.r1_release_step[drive_to_unload] = -1
            transition_reason[drive_to_unload] = 2

        # Record each finger's first contact-off event after unload begins.
        in_unload = old_phase == 2
        contact_off_event = in_unload.unsqueeze(-1) & old_contact & ~contact
        elapsed_unload = phase_step
        unset_release = self.r1_release_step < 0
        release_value = elapsed_unload.unsqueeze(-1).expand_as(self.r1_release_step)
        self.r1_release_step[:] = torch.where(
            contact_off_event & unset_release, release_value, self.r1_release_step
        )
        released_together = (~contact).all(dim=-1)
        release_timeout = elapsed_unload >= self.r1_release_window_steps
        unload_to_return = in_unload & (released_together | release_timeout)
        if torch.any(unload_to_return):
            phase[unload_to_return] = 3
            phase_step[unload_to_return] = 0
            idx_step = self.r1_release_step[:, 0]
            thumb_step = self.r1_release_step[:, 1]
            known_pair = (idx_step >= 0) & (thumb_step >= 0)
            lag = torch.abs(idx_step - thumb_step).float() * (
                self.control_freq_inv * self.dt
            )
            self.r1_release_lag[unload_to_return & known_pair] = lag[
                unload_to_return & known_pair
            ]
            transition_reason[unload_to_return] = torch.where(
                released_together[unload_to_return],
                torch.full_like(transition_reason[unload_to_return], 3),
                torch.full_like(transition_reason[unload_to_return], 4),
            )

        return_to_grasp = (old_phase == 3) & (
            (phase_step >= self.r1_return_min_steps) & pair_contact
        )
        if torch.any(return_to_grasp):
            phase[return_to_grasp] = 0
            phase_step[return_to_grasp] = 0
            transition_reason[return_to_grasp] = 5

        transitioned = phase != old_phase
        self.r1_phase[:] = phase
        self.r1_phase_step[:] = phase_step
        self.r1_transition_count += transitioned.long()
        self.r1_last_transition_reason[:] = torch.where(
            transitioned, transition_reason, self.r1_last_transition_reason
        )

        drive_mask = (phase == 1).float()
        positive_rotation = torch.clamp(nut_dof_vel, min=0.0)
        rotate_credit = torch.clamp(positive_rotation, max=self.angvel_clip_max)
        rotate_credit = rotate_credit * pair_gate * drive_mask
        unload_mask = phase == 2
        return_mask = phase == 3
        asymmetry = torch.logical_xor(contact[:, 0], contact[:, 1]).float()
        synchronized_release = unload_to_return & released_together
        release_bonus = synchronized_release.float() * self.r1_release_bonus_scale
        unload_penalty = unload_mask.float() * asymmetry * self.r1_unload_asymmetry_scale
        return_drag = return_mask.float() * contact.any(dim=-1).float()
        return_penalty = return_drag * self.r1_return_drag_scale + (
            return_mask.float()
            * torch.clamp(-nut_dof_vel, min=0.0)
            * self.r1_return_reverse_scale
        )
        phase_reward = release_bonus - unload_penalty - return_penalty

        pair_force = force_on_nut.sum(dim=1)
        fixed_base_lateral_proxy = torch.norm(pair_force[:, :2], dim=-1)
        fixed_base_moment_proxy = torch.norm(
            torch.cross(radial, force_on_nut, dim=-1).sum(dim=1)[:, :2], dim=-1
        )
        stats = {
            "force_mag": force_mag,
            "distance": dist,
            "contact_conf": contact_conf,
            "contact": contact,
            "signed_torque": signed_torque,
            "positive_torque": positive_torque,
            "pair_gate": pair_gate,
            "phase": phase,
            "release_lag": self.r1_release_lag,
            "return_drag": return_drag,
            "fixed_base_lateral_proxy": fixed_base_lateral_proxy,
            "fixed_base_moment_proxy": fixed_base_moment_proxy,
            "phase_reward": phase_reward,
        }
        return rotate_credit, phase_reward, stats

    def _compute_e002_a_reward(self, nut_pos, nut_dof_vel, actions):
        """E002-A drive: new nut progress with an explicit acceptance contract."""
        fingertip_pos = self.rigid_body_states[:, self.fingertip_handles, :3]
        fingertip_force = self.contact_forces[:, self.fingertip_handles, :]
        pair_pos = fingertip_pos[:, [0, 3], :]
        pair_force = fingertip_force[:, [0, 3], :]
        radial = pair_pos - nut_pos.unsqueeze(1)
        force_mag = torch.norm(pair_force, dim=-1)
        if self.e002_acceptance_mode == "contact_progress":
            # Net fingertip force can include the bolt. The active contact
            # contract must use the same nut-only source as forward reward.
            force_mag = torch.norm(self.e002_normal_contact_force, dim=-1)
        dist = torch.norm(radial, dim=-1)
        force_conf = torch.clamp(
            (force_mag - self.e002_contact_force_off)
            / (self.e002_contact_force_on - self.e002_contact_force_off),
            min=0.0,
            max=1.0,
        )
        dist_conf = torch.clamp(
            (self.e002_contact_dist_off - dist)
            / (self.e002_contact_dist_off - self.e002_contact_dist_on),
            min=0.0,
            max=1.0,
        )
        contact_conf = torch.minimum(force_conf, dist_conf)
        contact_on = (force_mag >= self.e002_contact_force_on) & (
            dist <= self.e002_contact_dist_on
        )
        contact_still_on = (force_mag >= self.e002_contact_force_off) & (
            dist <= self.e002_contact_dist_off
        )
        self.e002_contact_state[:] = torch.where(
            self.e002_contact_state, contact_still_on, contact_on
        )
        force_on_nut = pair_force * self.e002_force_sign
        if self.e002_torque_source == "contact_point_normal":
            # Stored moments act on the nut; force_sign retains its historical
            # convention relative to the force acting on the fingertip.
            torque_vec = -self.e002_force_sign * self.e002_normal_contact_moment
        else:
            torque_vec = torch.cross(radial, force_on_nut, dim=-1)
        signed_torque = self.e002_direction * (
            torque_vec * self.rot_axis_buf.unsqueeze(1).expand_as(torque_vec)
        ).sum(dim=-1)
        positive_torque = torch.clamp(signed_torque, min=0.0)
        torque_conf = torch.clamp(
            positive_torque / self.e002_torque_saturation, min=0.0, max=1.0
        )
        contribution = contact_conf * torque_conf
        torque_sum = positive_torque.sum(dim=-1)
        torque_balance = torch.where(
            torque_sum > 0.0,
            2.0 * torch.minimum(positive_torque[:, 0], positive_torque[:, 1])
            / torch.clamp(torque_sum, min=1e-6),
            torch.zeros_like(torque_sum),
        )
        pair_gate = (
            torch.minimum(contribution[:, 0], contribution[:, 1]) * torque_balance
        )
        pair_contact = self.e002_contact_state.all(dim=-1)
        pair_valid = (self.e002_contact_state.all(dim=-1)) & (
            signed_torque >= self.e002_torque_min
        ).all(dim=-1)
        normal_proxy_valid = pair_valid & (pair_gate >= self.e002_pair_gate_min)
        # User authorization, 2026-09-10: normal moments omit friction and
        # therefore have diagnostic status, not veto power over real progress.
        progress_valid = (pair_contact if self.e002_acceptance_mode == "contact_progress"
                          else normal_proxy_valid)

        # Contact acquisition is a prerequisite, not a per-step source of
        # reward.  A short consecutive-contact gate filters reset transients.
        self.e002_pair_contact_streak[:] = torch.where(
            pair_contact,
            self.e002_pair_contact_streak + 1,
            torch.zeros_like(self.e002_pair_contact_streak),
        )
        acquired_now = (
            self.e002_pair_contact_streak >= self.e002_contact_stable_steps
        ) & ~self.e002_contact_acquired
        self.e002_contact_acquired |= acquired_now
        self.e002_pair_contact_loss_streak[:] = torch.where(
            self.e002_contact_acquired & ~pair_contact,
            self.e002_pair_contact_loss_streak + 1,
            torch.zeros_like(self.e002_pair_contact_loss_streak),
        )
        self.e002_drive_step[:] = torch.where(
            self.e002_contact_acquired,
            self.e002_drive_step + 1,
            torch.zeros_like(self.e002_drive_step),
        )

        contact_potential = torch.minimum(contact_conf[:, 0], contact_conf[:, 1])
        contact_potential_delta = (
            contact_potential - self.e002_contact_potential_prev
        )
        approach_reward = torch.where(
            self.e002_contact_acquired,
            torch.zeros_like(contact_potential_delta),
            self.e002_approach_scale * contact_potential_delta,
        )
        self.e002_contact_potential_prev[:] = contact_potential
        contact_bonus = acquired_now.float() * self.e002_contact_bonus

        theta = self.nut_dof_pos.squeeze(-1)
        directed_theta = self.e002_direction * (theta - self.e002_theta_start)
        directed_step = self.e002_direction * (theta - self.e002_theta_prev)
        prev_paid = self.e002_paid_high_water.clone()
        prev_high_water = self.e002_high_water.clone()
        self.e002_high_water[:] = torch.maximum(self.e002_high_water, directed_theta)
        self.e002_paid_high_water[:] = torch.clamp(
            self.e002_high_water, min=0.0, max=self.e002_target_angle
        )
        delta_new = torch.clamp(self.e002_paid_high_water - prev_paid, min=0.0)
        self.e002_valid_pair_progress += delta_new * progress_valid.float()
        self.e002_normal_proxy_progress += delta_new * normal_proxy_valid.float()
        delta_reverse = torch.clamp(-directed_step, min=0.0)
        self.e002_gross_travel += torch.abs(directed_step)
        self.e002_reverse_travel += delta_reverse
        self.e002_theta_prev[:] = theta

        milestone_thresholds = torch.tensor(
            [self.e002_target_angle * 0.5, self.e002_target_angle],
            device=self.device,
        )
        crossed_now = (
            (prev_high_water.unsqueeze(-1) < milestone_thresholds)
            & (self.e002_high_water.unsqueeze(-1) >= milestone_thresholds)
        )
        milestone_now = (
            crossed_now
            & ~self.e002_milestone_paid
            & progress_valid.unsqueeze(-1)
            & (delta_new > 0.0).unsqueeze(-1)
        )
        self.e002_milestone_paid |= milestone_now
        milestone_reward = milestone_now.float().sum(dim=-1)

        in_target = (
            torch.abs(directed_theta - self.e002_target_angle)
            <= self.e002_target_band
        )
        self.e002_target_validated |= (
            self.e002_valid_pair_progress
            >= self.e002_target_angle - self.e002_target_band
        )
        hold = (
            in_target
            & pair_contact
            & self.e002_contact_acquired
            & self.e002_target_validated
            & (directed_step >= -self.e002_delta_ref)
        )
        self.e002_target_hold[:] = torch.where(
            hold, self.e002_target_hold + 1, torch.zeros_like(self.e002_target_hold)
        )
        success_now = (
            self.e002_target_hold >= self.e002_target_hold_steps
        ) & ~self.e002_success
        self.e002_success |= success_now

        normalized_new = torch.clamp(delta_new / self.e002_delta_ref, 0.0, 1.0)
        # Charging the complement of torque quality makes improving forward
        # motion net-negative whenever pair_gate < 0.5. This penalty concerns
        # missing physical contact; it must not treat unmeasured friction
        # moments as evidence of an invalid drive. Keep the legacy mode for
        # exact reward ablations. Acceptance mode is logged separately.
        invalid_quality = (contact_potential if self.e002_invalid_forward_source == "contact_quality"
                           else pair_gate)
        invalid_forward = normalized_new * (1.0 - invalid_quality)
        # Authorized literature comparison: contact-weighted rotation, adapted
        # to opposing fingers (Tac2Motion Eq. 3; see the literature alignment
        # doc). Use nut-only normal records: bolt contact cannot earn progress.
        # No inferred normal moment is used as a full friction-wrench estimate.
        if self.e002_forward_source == "nut_contact_quality":
            nut_force_mag = torch.norm(self.e002_normal_contact_force, dim=-1)
            nut_force_conf = torch.clamp(
                (nut_force_mag - self.e002_contact_force_off)
                / (self.e002_contact_force_on - self.e002_contact_force_off),
                min=0.0, max=1.0,
            )
            nut_contact_conf = torch.minimum(nut_force_conf, dist_conf)
            forward_quality = torch.minimum(nut_contact_conf[:, 0], nut_contact_conf[:, 1])
        else:
            forward_quality = pair_gate
        self.e002_forward_quality = forward_quality
        forward_reward = 4.0 * forward_quality * normalized_new
        reverse_cost = 4.0 * torch.clamp(delta_reverse / self.e002_delta_ref, 0.0, 1.0)
        reward = (
            forward_reward
            - reverse_cost
            - self.e002_invalid_forward_scale * invalid_forward
            + approach_reward
            + contact_bonus
            + milestone_reward
            + 10.0 * success_now.float()
            - 0.25
            * torch.clamp(
                (torch.abs(directed_step) - self.e002_overspeed_angle)
                / self.e002_delta_ref,
                min=0.0,
                max=1.0,
            )
            - torch.clamp(
                (directed_theta - self.e002_overshoot_angle)
                / self.e002_delta_ref,
                min=0.0,
                max=1.0,
            )
        )

        workspace_escape = (dist > self.e002_workspace_radius).any(dim=-1)
        collision_step = torch.abs(directed_step) > self.e002_hard_step_angle
        acquire_timeout = (
            self.progress_buf >= self.e002_contact_acquire_deadline_steps
        ) & ~self.e002_contact_acquired
        lost_contact = (
            self.e002_contact_acquired
            & (
                self.e002_pair_contact_loss_streak
                >= self.e002_contact_loss_grace_steps
            )
        )
        deadline = (
            self.e002_drive_step >= self.e002_drive_deadline_steps
        ) & ~self.e002_success
        reason = torch.zeros_like(self.e002_termination_reason)
        reason = torch.where(deadline, torch.full_like(reason, 1), reason)
        reason = torch.where(lost_contact, torch.full_like(reason, 2), reason)
        reason = torch.where(workspace_escape, torch.full_like(reason, 3), reason)
        reason = torch.where(collision_step, torch.full_like(reason, 4), reason)
        reason = torch.where(acquire_timeout, torch.full_like(reason, 6), reason)
        reason = torch.where(self.e002_success, torch.full_like(reason, 5), reason)
        self.e002_termination_reason[:] = reason
        self.reset_buf[:] = reason > 0
        failure = (reason > 0) & (reason != 5)
        reward = reward - 2.0 * failure.float() - 8.0 * collision_step.float()
        self.rew_buf[:] = reward

        gross = torch.clamp(self.e002_gross_travel, min=1e-6)
        positive_count = torch.clamp((directed_step > 0.0).float().sum(), min=1.0)
        per_finger_contrib = self.e002_contact_state & (
            signed_torque >= self.e002_torque_min
        )
        both_contrib = per_finger_contrib.all(dim=-1)
        single_contrib = per_finger_contrib.sum(dim=-1) == 1
        self.extras["step_all_reward"] = float(reward.mean().item())
        self.extras["e002/new_high_water"] = float(delta_new.mean().item())
        self.extras["e002/reverse_step"] = float(delta_reverse.mean().item())
        self.extras["e002/net_angle"] = float(directed_theta.mean().item())
        self.extras["e002/high_water_angle"] = float(self.e002_high_water.mean().item())
        self.extras["e002/valid_pair_progress"] = float(
            self.e002_valid_pair_progress.mean().item()
        )
        self.extras["e002/normal_proxy_progress"] = float(
            self.e002_normal_proxy_progress.mean().item()
        )
        self.extras["e002/gross_angle"] = float(self.e002_gross_travel.mean().item())
        self.extras["e002/net_over_gross"] = float(
            (directed_theta / gross).mean().item()
        )
        self.extras["e002/reverse_over_gross"] = float(
            (self.e002_reverse_travel / gross).mean().item()
        )
        self.extras["e002/virtual_advance_mm"] = float(
            (directed_theta * 3.0 / (2.0 * np.pi)).mean().item()
        )
        self.extras["e002/pair_gate"] = float(pair_gate.mean().item())
        self.extras["e002/torque_balance"] = float(torque_balance.mean().item())
        self.extras["e002/contact_acquired"] = float(
            self.e002_contact_acquired.float().mean().item()
        )
        self.extras["e002/contact_streak"] = float(
            self.e002_pair_contact_streak.float().mean().item()
        )
        self.extras["e002/invalid_forward"] = float(invalid_forward.mean().item())
        self.extras["e002/reward/forward"] = float(forward_reward.mean().item())
        self.extras["e002/forward_quality"] = float(forward_quality.mean().item())
        self.extras["e002/reward/reverse_cost"] = float(reverse_cost.mean().item())
        self.extras["e002/reward/invalid_cost"] = float((self.e002_invalid_forward_scale * invalid_forward).mean().item())
        self.extras["e002/pair_drive_ratio"] = float(
            (
                (directed_step.gt(0.0) & both_contrib).float().sum()
                / positive_count
            ).item()
        )
        self.extras["e002/single_finger_drive_ratio"] = float(
            ((directed_step.gt(0.0) & single_contrib).float().sum() / positive_count).item()
        )
        self.extras["e002/contact/index"] = float(
            self.e002_contact_state[:, 0].float().mean().item()
        )
        self.extras["e002/contact/thumb"] = float(
            self.e002_contact_state[:, 1].float().mean().item()
        )
        self.extras["e002/signed_torque/index"] = float(
            signed_torque[:, 0].mean().item()
        )
        self.extras["e002/signed_torque/thumb"] = float(
            signed_torque[:, 1].mean().item()
        )
        for code, name in (
            (1, "deadline"),
            (2, "contact_lost"),
            (3, "workspace"),
            (4, "collision"),
            (5, "success"),
            (6, "contact_acquire_timeout"),
        ):
            self.extras[f"e002/term/{name}"] = float(
                (reason == code).float().mean().item()
            )
        self.prev_hand_actions[:] = actions[:, : self.num_actions]

    def compute_reward(self, actions):
        # Update current angular velocity penalty threshold based on curriculum
        current_angvel_penalty_threshold = self._get_current_angvel_penalty_threshold()

        # pose diff penalty (thumb DOFs masked out)
        pose_diff_penalty = (
            (
                (self.xhand_hand_dof_pos - self.init_pose_buf[..., : self.num_actions])
                ** 2
            )
            * self.pose_diff_penalty_mask
        ).sum(-1)
        # work and torque penalty
        torque_penalty = (self.torques[:, -1, : self.num_actions] ** 2).sum(-1)
        work_penalty = (
            (
                torch.abs(self.torques[:, -1, : self.num_actions])
                * torch.abs(self.dof_vel_finite_diff[:, -1])
            ).sum(-1)
        ) ** 2

        angdiff = self.quat_to_axis_angle(
            quat_mul(self.object_rot, quat_conjugate(self.object_rot_prev))
        )
        object_angvel = angdiff / (self.control_freq_inv * self.dt)

        # Calculate proximity reward
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3]
        thumb_pos, index_pos = (
            self.rigid_body_states[:, self.fingertip_handles[-1], :3],
            self.rigid_body_states[:, self.fingertip_handles[0], :3],
        )
        middle_pos, ring_pos = (
            self.rigid_body_states[:, self.fingertip_handles[1], :3],
            self.rigid_body_states[:, self.fingertip_handles[2], :3],
        )
        thumb_dist, index_dist = torch.norm(thumb_pos - nut_pos, dim=-1), torch.norm(
            index_pos - nut_pos, dim=-1
        )
        middle_dist = torch.norm(middle_pos - nut_pos, dim=-1)
        ring_dist = torch.norm(ring_pos - nut_pos, dim=-1)
        mean_dist = 0.5 * (thumb_dist + index_dist)
        proximity_thres = getattr(
            self, "proximity_dist_threshold", self.reset_dist_threshold
        )
        proximity_thres = max(float(proximity_thres), 1e-6)
        ratio = mean_dist / proximity_thres
        proximity_reward = torch.clamp(1.0 - ratio, min=0.0, max=1.0)

        nut_dof_linvel = (
            (self.nut_dof_pos.squeeze(-1) - self.nut_dof_pos_prev.squeeze(-1))
            / (self.control_freq_inv * self.dt)
        ).squeeze(-1)
        self.nut_dof_vel_cf = nut_dof_linvel.unsqueeze(-1)
        if self.e002_motion_enabled:
            self._compute_e002_a_reward(
                nut_pos=nut_pos,
                nut_dof_vel=nut_dof_linvel,
                actions=actions,
            )
            if self.e002_cycle is not None:
                pair = [self.fingertip_handles[0], self.fingertip_handles[3]]
                radial = self.rigid_body_states[:, pair, :3] - nut_pos[:, None]
                reward, reason = self.e002_cycle.update(
                    self.e002_direction * self.nut_dof_pos[:, 0], self.e002_contact_state,
                    torch.norm(self.e002_normal_contact_force, dim=-1),
                    torch.norm(radial, dim=-1), radial, self.rot_axis_buf,
                    self.e002_forward_quality, self.e002_contact_acquired,
                    self.e002_delta_ref, self.e002_workspace_radius,
                    self.e002_hard_step_angle, self.e002_contact_force_off,
                    normal_force_vectors=self.e002_normal_contact_force,
                    normal_moment_vectors=self.e002_normal_contact_moment,
                    force_cost_signal=(self.e003_peak_contact_force if self.config['env'].get('e003_peak_force_reward', False) else None))
                if self.e002_demo is not None and self.e002_demo.release_only:
                    released = self.e002_cycle.last_event == 2
                    retained = self.e002_cycle.previous >= .5 * self.e002_cycle.target
                    # Code6 is release-subtask success, never full-cycle success5.
                    reason = torch.where(released & retained & (reason == 0), 6, reason)
                    self.e002_cycle.last_reason[:] = reason
                    self.extras['release_learning/success'] = float((reason == 6).float().mean())
                    self.extras['release_learning/seed_step'] = float(self.e002_demo.seed_step.float().mean())
                if self.e002_demo is not None and self.e002_demo.return_only:
                    returned = self.e002_cycle.last_event == 3
                    retained = (self.e002_cycle.previous - self.e002_cycle.stroke_start
                                >= .5 * self.e002_cycle.target)
                    # Code7 is the return subtask, distinct from full-cycle5.
                    reason = torch.where(returned & retained & (reason == 0), 7, reason)
                    self.e002_cycle.last_reason[:] = reason
                    self.extras['return_learning/success'] = float((reason == 7).float().mean())
                self.rew_buf[:] = reward
                self.reset_buf[:] = reason > 0
                self.e002_termination_reason[:] = reason
                self.e002_success[:] = reason == 5
                self.extras['step_all_reward'] = float(reward.mean())
                # The single-stroke reward above only updates contact/diagnostic
                # state here; its unused reward and termination labels are not
                # the cycle's training objective or outcome.
                for key in list(self.extras):
                    if key.startswith('e002/reward/') or key.startswith('e002/term/'):
                        del self.extras[key]
                for name in ('phase', 'cycles', 'drive_events', 'release_events', 'return_events',
                             'return_error', 'contact_progress', 'high', 'reverse'):
                    self.extras['cycle/' + name] = float(getattr(self.e002_cycle, name).float().mean())
                for code in (3, 4, 5, 10, 11, 12, 13, 14):
                    self.extras['cycle/termination_' + str(code)] = float((reason == code).float().mean())
            return
        rotate_reward_raw = torch.clip(
            nut_dof_linvel, max=self.angvel_clip_max, min=self.angvel_clip_min
        )
        rotate_reward = rotate_reward_raw
        r1_phase_reward = torch.zeros_like(rotate_reward)
        r1_stats = None
        if self.r1_motion_enabled:
            rotate_reward, r1_phase_reward, r1_stats = self._compute_r1_motion_reward(
                nut_pos=nut_pos,
                nut_dof_vel=nut_dof_linvel,
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
        if self.r1_motion_enabled:
            self.rew_buf[:] = self.rew_buf + r1_phase_reward

        # Optional smoothness regularization: penalize high-frequency action changes.
        # This helps reduce “抽搐/点戳”式策略（短时接触 + 高频振荡）并鼓励更平滑的持续接触旋转。
        if "action_rate_penalty" in self.reward_scale_dict:
            action_rate_pscale = self._get_reward_scale_by_name("action_rate_penalty")
            if abs(float(action_rate_pscale)) > 0.0:
                hand_actions = actions[:, : self.num_actions]
                action_rate_penalty = ((hand_actions - self.prev_hand_actions) ** 2).sum(
                    -1
                )
                # Avoid cross-episode penalty: `at_reset_buf==1` means the env was just reset last step.
                action_rate_penalty = action_rate_penalty * (
                    1.0 - self.at_reset_buf.float()
                )
                self.rew_buf[:] = self.rew_buf + action_rate_penalty * action_rate_pscale
                self.extras["action_rate_penalty"] = float(
                    action_rate_penalty.mean().item()
                )
                if self.log_debug_metrics and self.num_envs >= 4:
                    self.extras["debug/action_rate_penalty_p95"] = float(
                        torch.quantile(action_rate_penalty, 0.95).item()
                    )

        # Update action history buffer (always), used by optional smoothness terms.
        self.prev_hand_actions[:] = actions[:, : self.num_actions]

        self.reset_buf[:] = self.check_termination(self.object_pos)
        self.extras["step_all_reward"] = float(self.rew_buf.mean().item())
        self.extras["rotation_reward"] = float(rotate_reward.mean().item())
        self.extras["pose_diff_penalty"] = float(pose_diff_penalty.mean().item())
        self.extras["work_done"] = float(work_penalty.mean().item())
        self.extras["torques"] = float(torque_penalty.mean().item())
        self.extras["roll"] = float(torch.abs(object_angvel[:, 0]).mean().item())
        self.extras["pitch"] = float(torch.abs(object_angvel[:, 1]).mean().item())
        self.extras["yaw"] = float(torch.abs(object_angvel[:, 2]).mean().item())
        self.extras["z_dist_penalty"] = float(z_dist_penalty.mean().item())
        self.extras["rotate_penalty"] = float(rotate_penalty.mean().item())

        if self.r1_motion_enabled:
            # These values are required by E001. Contact-force entries are
            # resultant-fingertip proxies, not pairwise contact manifold force.
            phase = r1_stats["phase"]
            contact = r1_stats["contact"]
            signed_torque = r1_stats["signed_torque"]
            positive_rotation = nut_dof_linvel > 0.0
            both_torque = (signed_torque >= self.r1_torque_min).all(dim=-1)
            exactly_one_torque = (signed_torque >= self.r1_torque_min).sum(dim=-1) == 1
            positive_count = torch.clamp(positive_rotation.float().sum(), min=1.0)
            self.extras["r1/rotation_credit_raw"] = float(rotate_reward.mean().item())
            self.extras["r1/phase_reward_raw"] = float(
                r1_stats["phase_reward"].mean().item()
            )
            self.extras["r1/pair_drive_ratio"] = float(
                ((positive_rotation & both_torque).float().sum() / positive_count).item()
            )
            self.extras["r1/single_finger_drive_ratio"] = float(
                ((positive_rotation & exactly_one_torque).float().sum() / positive_count).item()
            )
            self.extras["r1/contact_overlap"] = float(contact.all(dim=-1).float().mean().item())
            self.extras["r1/contact_confidence/index"] = float(
                r1_stats["contact_conf"][:, 0].mean().item()
            )
            self.extras["r1/contact_confidence/thumb"] = float(
                r1_stats["contact_conf"][:, 1].mean().item()
            )
            self.extras["r1/contact_force_proxy/index"] = float(
                r1_stats["force_mag"][:, 0].mean().item()
            )
            self.extras["r1/contact_force_proxy/thumb"] = float(
                r1_stats["force_mag"][:, 1].mean().item()
            )
            self.extras["r1/contact_distance/index"] = float(
                r1_stats["distance"][:, 0].mean().item()
            )
            self.extras["r1/contact_distance/thumb"] = float(
                r1_stats["distance"][:, 1].mean().item()
            )
            self.extras["r1/signed_torque/index"] = float(signed_torque[:, 0].mean().item())
            self.extras["r1/signed_torque/thumb"] = float(signed_torque[:, 1].mean().item())
            self.extras["r1/positive_torque/index"] = float(
                r1_stats["positive_torque"][:, 0].mean().item()
            )
            self.extras["r1/positive_torque/thumb"] = float(
                r1_stats["positive_torque"][:, 1].mean().item()
            )
            self.extras["r1/phase/grasp_ratio"] = float((phase == 0).float().mean().item())
            self.extras["r1/phase/drive_ratio"] = float((phase == 1).float().mean().item())
            self.extras["r1/phase/unload_ratio"] = float((phase == 2).float().mean().item())
            self.extras["r1/phase/return_ratio"] = float((phase == 3).float().mean().item())
            self.extras["r1/transition_count"] = float(
                self.r1_transition_count.float().mean().item()
            )
            self.extras["r1/release_lag_seconds"] = float(
                r1_stats["release_lag"].mean().item()
            )
            self.extras["r1/return_drag_ratio"] = float(
                r1_stats["return_drag"].mean().item()
            )
            self.extras["r1/nut_progress_positive"] = float(
                torch.clamp(nut_dof_linvel, min=0.0).mean().item()
            )
            self.extras["r1/nut_progress_negative"] = float(
                torch.clamp(-nut_dof_linvel, min=0.0).mean().item()
            )
            self.extras["r1/fixed_base_lateral_force_proxy"] = float(
                r1_stats["fixed_base_lateral_proxy"].mean().item()
            )
            self.extras["r1/fixed_base_bending_moment_proxy"] = float(
                r1_stats["fixed_base_moment_proxy"].mean().item()
            )

        # curriculum tracking
        self.extras["curriculum/angvel_penalty_threshold"] = float(
            current_angvel_penalty_threshold
        )

        # screw-specific metrics
        # Prefer finite-difference velocity (more stable/consistent across assets)
        self.extras["screw/angular_velocity"] = float(self.nut_dof_vel_cf.mean().item())
        self.extras["screw/angular_position"] = float(self.nut_dof_pos.mean().item())
        self.extras["screw/positive_vel_ratio"] = float(
            (self.nut_dof_vel_cf > 0).float().mean().item()
        )

        # ---- debug metrics (opt-in) ----
        if self.log_debug_metrics:
            # Disambiguate torque-avoidance vs reset-avoidance (thumb/index)
            self.extras["debug/thumb_dist_mean"] = float(thumb_dist.mean().item())
            self.extras["debug/index_dist_mean"] = float(index_dist.mean().item())
            self.extras["debug/middle_dist_mean"] = float(middle_dist.mean().item())
            self.extras["debug/ring_dist_mean"] = float(ring_dist.mean().item())
            self.extras["debug/thumb_dist_margin_mean"] = float(
                (thumb_dist - self.reset_dist_threshold).mean().item()
            )
            self.extras["debug/index_dist_margin_mean"] = float(
                (index_dist - self.reset_dist_threshold).mean().item()
            )
            self.extras["debug/middle_dist_margin_mean"] = float(
                (middle_dist - self.reset_dist_threshold).mean().item()
            )
            self.extras["debug/ring_dist_margin_mean"] = float(
                (ring_dist - self.reset_dist_threshold).mean().item()
            )

            # Action magnitude per finger (policy output). For Pasini DOF layout: 4 fingers x 4 DOFs.
            # Note: `self.actions` is padded elsewhere to include the nut joint; here we only use the 16 hand actions.
            hand_actions = self.actions[:, : self.num_actions]
            self.extras["debug/action_norm/index_mean"] = float(
                torch.norm(hand_actions[:, 0:4], dim=-1).mean().item()
            )
            self.extras["debug/action_norm/middle_mean"] = float(
                torch.norm(hand_actions[:, 4:8], dim=-1).mean().item()
            )
            self.extras["debug/action_norm/ring_mean"] = float(
                torch.norm(hand_actions[:, 8:12], dim=-1).mean().item()
            )
            self.extras["debug/action_norm/thumb_mean"] = float(
                torch.norm(hand_actions[:, 12:16], dim=-1).mean().item()
            )

            # Torque magnitude per finger (from low-level control torques buffer)
            hand_torques = self.torques[:, -1, : self.num_actions]
            self.extras["debug/torque_norm/index_mean"] = float(
                torch.norm(hand_torques[:, 0:4], dim=-1).mean().item()
            )
            self.extras["debug/torque_norm/middle_mean"] = float(
                torch.norm(hand_torques[:, 4:8], dim=-1).mean().item()
            )
            self.extras["debug/torque_norm/ring_mean"] = float(
                torch.norm(hand_torques[:, 8:12], dim=-1).mean().item()
            )
            self.extras["debug/torque_norm/thumb_mean"] = float(
                torch.norm(hand_torques[:, 12:16], dim=-1).mean().item()
            )

            # Heavy-tail indicators: a few exploding envs can dominate learning dynamics.
            if self.num_envs >= 4:
                self.extras["debug/torque_penalty_p95"] = float(
                    torch.quantile(torque_penalty, 0.95).item()
                )
                self.extras["debug/work_penalty_p95"] = float(
                    torch.quantile(work_penalty, 0.95).item()
                )

            # Termination reason fractions (computed inside check_termination)
            reasons = getattr(self, "_last_termination_reasons", None)
            if isinstance(reasons, dict):
                for k, v in reasons.items():
                    self.extras[f"term/{k}_frac"] = float(v.float().mean().item())
            self.extras["term/any_reset_frac"] = float(
                self.reset_buf.float().mean().item()
            )

        # Hand contact force stats are useful for debugging self-collision; compute only when needed.
        need_hand_cf = self.log_debug_metrics or self.debug_print_top_hand_contacts
        if need_hand_cf:
            hand_cf = torch.norm(
                self.contact_forces[:, : self.num_xhand_hand_bodies, :], dim=-1
            )  # (num_envs, hand_bodies)

            if self.log_debug_metrics:
                self.extras["debug/hand_contact_force_mean"] = float(
                    hand_cf.mean().item()
                )
                if self.num_envs >= 4:
                    self.extras["debug/hand_contact_force_p95"] = float(
                        torch.quantile(hand_cf.mean(dim=1), 0.95).item()
                    )

                def _group_mean(indices):
                    if not indices:
                        return 0.0
                    return float(hand_cf[:, indices].mean().item())

                self.extras["debug/hand_contact_force/thumb_mean"] = _group_mean(
                    getattr(self, "thumb_rb_indices", [])
                )
                self.extras["debug/hand_contact_force/index_mean"] = _group_mean(
                    getattr(self, "index_rb_indices", [])
                )
                self.extras["debug/hand_contact_force/middle_mean"] = _group_mean(
                    getattr(self, "middle_rb_indices", [])
                )
                self.extras["debug/hand_contact_force/ring_mean"] = _group_mean(
                    getattr(self, "ring_rb_indices", [])
                )

            if (
                self.debug_print_top_hand_contacts
                and int(self.progress_buf.max().item()) < 5
                and hasattr(self, "hand_rb_names")
            ):
                mean_cf = hand_cf.mean(dim=0)
                topk = min(8, mean_cf.numel())
                vals, idx = torch.topk(mean_cf, k=topk)
                msg = "top_hand_contact_forces=" + ", ".join(
                    [
                        f"{self.hand_rb_names[int(i)]}:{float(v):.3f}"
                        for v, i in zip(
                            vals.detach().cpu().tolist(),
                            idx.detach().cpu().tolist(),
                        )
                    ]
                )
                tprint(msg)

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
        if self.e002_motion_enabled and not self.config["sim"]["use_gpu_pipeline"]:
            # The CPU tensor path normally fetches at the next substep start.
            # Complete the last substep before matching contacts and body poses.
            self.gym.fetch_results(self.sim, True)
        self._refresh_gym()
        if self.e002_motion_enabled and self.e002_torque_source == "contact_point_normal":
            from dexscrew.tasks.contact_moments import collect_env_contacts, normal_contact_wrench
            force, moment, counts = normal_contact_wrench(
                collect_env_contacts(self.gym, self.envs),
                self.rigid_body_states[:, self.screw_nut_rb_handle, 3:7].numpy(),
                self.screw_nut_rb_handle, [self.fingertip_handles[0], self.fingertip_handles[3]],
                self.num_bodies)
            self.e002_normal_contact_force = torch.as_tensor(force, dtype=torch.float32, device=self.device)
            self.e002_normal_contact_moment = torch.as_tensor(moment, dtype=torch.float32, device=self.device)
            self.e002_normal_contact_count = torch.as_tensor(counts, device=self.device)
        if self.config['env'].get('e003_peak_force_reward', False):
            self._e003_collect_peak_contact_force(False)
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
        targets[:, : self.num_xhand_hand_dofs] = tensor_clamp(
            targets[:, : self.num_xhand_hand_dofs],
            self.xhand_hand_dof_lower_limits,
            self.xhand_hand_dof_upper_limits,
        )
        if self.target_smoothing_alpha > 0.0:
            a = self.target_smoothing_alpha
            targets[:, : self.num_xhand_hand_dofs] = (
                a * self.prev_targets[:, : self.num_xhand_hand_dofs]
                + (1.0 - a) * targets[:, : self.num_xhand_hand_dofs]
            )
        # Optional E003 anti-windup: retain the projected target as the next
        # integrator state, so blocked motion cannot accumulate unreachable
        # targets. Units are radians; this is not a hardware force limit.
        error_limit = self.config['env']['controller'].get('target_error_limit', 0.0)
        if error_limit > 0.0:
            q = self.xhand_hand_dof_pos
            targets[:, : self.num_xhand_hand_dofs] = tensor_clamp(
                targets[:, : self.num_xhand_hand_dofs], q - error_limit, q + error_limit)
            targets[:, : self.num_xhand_hand_dofs] = tensor_clamp(
                targets[:, : self.num_xhand_hand_dofs],
                self.xhand_hand_dof_lower_limits, self.xhand_hand_dof_upper_limits)
        self.cur_targets[:, : self.num_xhand_hand_dofs] = targets[:, : self.num_xhand_hand_dofs]

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
        self._refresh_policy_views()
        return self.obs_dict

    def _refresh_policy_views(self):
        self.obs_dict["priv_info"] = self.priv_info_buf.to(self.rl_device)
        self.obs_dict["proprio_hist"] = self.proprio_hist_buf.to(self.rl_device)
        self.obs_dict["point_cloud_info"] = self.point_cloud_buf.to(self.rl_device)
        self.obs_dict["rot_axis_buf"] = self.rot_axis_buf.to(self.rl_device)

    def step(self, actions, extrin_record: Optional[torch.Tensor] = None):
        # Save extrinsics if evaluating on just one object.
        action_mask = torch.ones_like(actions)
        # Keep Hora semantics but adapt to Pasini's DOF layout (4 fingers x 4 DOFs):
        # Hora masks the pinky always; Pasini has no pinky.
        # For nutbolt-like poses, Hora also masks the ring finger; in Pasini that's actions[8:12].
        if self.apply_action_mask:
            if self.custom_action_mask_indices is not None:
                mask_indices = [int(i) for i in list(self.custom_action_mask_indices)]
                invalid = [i for i in mask_indices if i < 0 or i >= self.num_actions]
                if invalid:
                    raise ValueError(
                        f"env.action_mask_indices contains invalid action ids {invalid}; "
                        f"valid range is [0, {self.num_actions - 1}]"
                    )
                action_mask[:, mask_indices] = 0.0
            elif self.config["env"]["initPose"] != "screwdriver_inclined":
                action_mask[:, 8:12] = 0.0
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
        result = super().step(actions)
        self._refresh_policy_views()
        return result

    def dump_current_pose(self, env_id: int = 0, out_dir: str = "outputs/pose_dumps"):
        """Dump the current hand/object pose to a json file.

        Designed for bootstrapping a better init pose:
        1) Run visualization (headless=False)
        2) Pause at a good grasp (press P)
        3) Dump pose (press O)
        4) Copy `hand_dof_pos` into env.customInitDofPos
        """
        import json
        import os
        import time

        env_id = int(env_id)
        os.makedirs(out_dir, exist_ok=True)

        # Ensure indices are python ints
        hand_actor = int(self.hand_indices[env_id].item())
        obj_actor = int(self.object_indices[env_id].item())

        hand_root = (
            self.root_state_tensor[hand_actor, 0:7].detach().cpu().numpy().tolist()
        )  # [x,y,z,qx,qy,qz,qw]
        obj_root = (
            self.root_state_tensor[obj_actor, 0:7].detach().cpu().numpy().tolist()
        )
        hand_dof_pos = (
            self.xhand_hand_dof_pos[env_id, : self.num_xhand_hand_dofs]
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )

        nut_dof_pos = None
        nut_dof_vel = None
        if hasattr(self, "nut_dof_pos"):
            nut_dof_pos = float(self.nut_dof_pos[env_id].item())
        if hasattr(self, "nut_dof_vel"):
            nut_dof_vel = float(self.nut_dof_vel[env_id].item())

        payload = {
            "timestamp_ms": int(time.time() * 1000),
            "env_id": env_id,
            "hand_dof_names": getattr(self, "hand_dof_names", None),
            "hand_dof_pos": hand_dof_pos,
            "hand_root_state_xyzw": hand_root,
            "object_root_state_xyzw": obj_root,
            "nut_dof_pos": nut_dof_pos,
            "nut_dof_vel": nut_dof_vel,
            "suggested_config_override": {
                "env.customInitDofPos": hand_dof_pos,
            },
        }

        out_path = os.path.join(
            out_dir, f"pasini_pose_env{env_id}_{payload['timestamp_ms']}.json"
        )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[dump_current_pose] wrote: {out_path}")
        task_name = str(self.config.get("name", "XHandPasini<CurrentTask>"))
        print(
            f"[dump_current_pose] copy suggested_config_override into configs/task/{task_name}.yaml -> env.customInitDofPos"
        )
        return out_path
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

    def _e003_collect_peak_contact_force(self, reset):
        if reset:
            self.e003_peak_contact_force = torch.zeros((self.num_envs, 2), device=self.device)
            if not hasattr(self, '_e003_force_groups'):
                self._e003_force_groups = [[i for i,n in enumerate(self.hand_rb_names) if n.startswith('right_'+finger+'_')] for finger in ('index', 'thumb')]
                assert all(self._e003_force_groups)
            return
        self.gym.refresh_net_contact_force_tensor(self.sim)
        # Sum per-body resultant magnitudes across each whole finger. Includes
        # non-nut contacts too; this is not an isolated nut-force bound.
        # Nut-only endpoint forces remain a floor for the excess cost.
        force = torch.stack([self.contact_forces[:,group].norm(dim=-1).sum(-1) for group in self._e003_force_groups], dim=-1)
        self.e003_peak_contact_force[:] = torch.maximum(self.e003_peak_contact_force, force)

    def update_low_level_control(self, step_id):
        if self.config['env'].get('e003_peak_force_reward', False):
            # Before substep i>0 the previous substep has been fetched by
            # VecTask. The final endpoint is captured in post_physics_step.
            self._e003_collect_peak_contact_force(step_id == 0)
        # Refresh DOF state first; on startup the wrapped tensors may contain
        # uninitialized values. Using them before a refresh can inject NaNs into torques.
        self.gym.refresh_dof_state_tensor(self.sim)
        if self.torque_control:
            # Guard: if DOF state is still non-finite (can happen on the first GPU step),
            # do not send NaN torques into the simulator.
            if (not torch.isfinite(self.xhand_hand_dof_pos).all()) or (
                not torch.isfinite(self.xhand_hand_dof_vel).all()
            ):
                if not hasattr(self, "_debug_nonfinite_dof_once"):
                    self._debug_nonfinite_dof_once = True
                    nonfinite_pos = int(
                        (~torch.isfinite(self.xhand_hand_dof_pos)).sum().item()
                    )
                    nonfinite_vel = int(
                        (~torch.isfinite(self.xhand_hand_dof_vel)).sum().item()
                    )
                    print(
                        f"[XHandPasini] Non-finite DOF state at control step {step_id}: pos={nonfinite_pos}, vel={nonfinite_vel}"
                    )
                torques = torch.zeros(
                    (self.num_envs, self.num_dofs),
                    device=self.device,
                    dtype=torch.float,
                )
                self.gym.set_dof_actuation_force_tensor(
                    self.sim, gymtorch.unwrap_tensor(torques)
                )
                return
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

        # finger nut distance check
        nut_states = self.rigid_body_states[:, self.screw_nut_rb_handle]
        nut_pos = nut_states[..., :3]

        thumb_pos = self.rigid_body_states[:, self.fingertip_handles[-1], :3]
        index_pos = self.rigid_body_states[:, self.fingertip_handles[0], :3]
        thumb_dist = torch.norm(thumb_pos - nut_pos, dim=-1)
        index_dist = torch.norm(index_pos - nut_pos, dim=-1)
        finger_dist_reset = torch.logical_or(
            thumb_dist > self.reset_dist_threshold,
            index_dist > self.reset_dist_threshold,
        )
        resets = torch.logical_or(resets, finger_dist_reset)

        # nut constant pos check - terminate if nut position is similar over 10 timesteps
        nut_pos_history_filled = self.progress_buf >= self.nut_termination_history_len
        nut_pos_variance = torch.var(self.nut_dof_pos_history, dim=1).squeeze(
            -1
        )  # (num_envs,)
        nut_pos_stagnant = (
            nut_pos_variance < self.nut_stagnation_eps
        ) & nut_pos_history_filled
        resets = torch.logical_or(resets, nut_pos_stagnant)

        # nut contact check - terminate if nut has 0 contact force over 10 timesteps
        contact_history_filled = self.progress_buf >= self.nut_termination_history_len
        no_contact = torch.all(self.nut_contact_history <= 1e-3, dim=1).squeeze(-1)
        no_contact_reset = no_contact & contact_history_filled
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

            screw_at_limit = current_screw_pos > reset_threshold
            resets = torch.logical_or(resets, screw_at_limit)

        # Store detailed termination reasons for debugging/logging.
        # Keep this opt-in so default runs stay clean.
        if getattr(self, "log_debug_metrics", False):
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
        # Pasini (dexh13_right) uses 4 fingers with 4 DOFs each.
        dof_names = [
            "right_index_joint_0",
            "right_index_joint_1",
            "right_index_joint_2",
            "right_index_joint_3",
            "right_middle_joint_0",
            "right_middle_joint_1",
            "right_middle_joint_2",
            "right_middle_joint_3",
            "right_ring_joint_0",
            "right_ring_joint_1",
            "right_ring_joint_2",
            "right_ring_joint_3",
            "right_thumb_joint_0",
            "right_thumb_joint_1",
            "right_thumb_joint_2",
            "right_thumb_joint_3",
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
            if torch.is_tensor(value):
                nonfinite = (~torch.isfinite(value)).sum().item()
                if nonfinite:
                    if not hasattr(self, "_debug_priv_nonfinite_once"):
                        self._debug_priv_nonfinite_once = set()
                    if name not in self._debug_priv_nonfinite_once:
                        self._debug_priv_nonfinite_once.add(name)
                        print(
                            f"[XHandPasini] Non-finite priv_info '{name}': {int(nonfinite)} elements"
                        )
                    value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
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
            # bulb task
            "bulb": "assets/bulb/0000bulb.urdf",
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

    def _default_object_init_pos(self):
        if self.config["env"]["initPose"] == "nutbolt_inclined":
            return [0.0175, 0.06, 0.0]
        if self.config["env"]["initPose"] in (
            "bulb_inclined",
            "lightbulb_inclined",
            "screwdriver_inclined",
        ):
            return [0.023, 0.03, 0.0]
        raise ValueError(
            f"Unsupported initPose for object initialization: {self.config['env']['initPose']}"
        )

    def _sample_object_init_pos(self):
        object_cfg = self.config["env"]["object"]
        base_pos = np.asarray(
            object_cfg.get("init_pos", self._default_object_init_pos()),
            dtype=np.float32,
        )
        noise_scale = np.asarray(
            object_cfg.get("init_pos_noise", [0.0075, 0.0075, 0.0]),
            dtype=np.float32,
        )
        sampled = base_pos + np.random.uniform(0.0, 1.0, size=3) * noise_scale
        return float(sampled[0]), float(sampled[1]), float(sampled[2])

    def _allocate_task_buffer(self, num_envs):
        # extra buffers for observe randomized params
        self.prop_hist_len = self.config["env"]["hora"]["propHistoryLen"]
        self.priv_info_dim = max([v[1] for k, v in self.priv_info_dict.items()])
        self.point_cloud_sampled_dim = self.config["env"]["hora"][
            "point_cloud_sampled_dim"
        ]
        self.point_cloud_buffer_dim = self.point_cloud_sampled_dim
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
        # For smoothness regularization (optional): penalize action changes across control steps.
        self.prev_hand_actions = torch.zeros(
            (num_envs, self.num_actions), device=self.device, dtype=torch.float
        )
        # R1-motion phase state. Phase ids are: grasp=0, drive=1, unload=2,
        # return=3. These buffers are diagnostic state only and are never
        # appended to the frozen policy observation contract.
        self.r1_phase = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.r1_phase_step = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.r1_phase_nut_pos = torch.zeros(
            num_envs, device=self.device, dtype=torch.float
        )
        self.r1_contact_state = torch.zeros(
            (num_envs, 2), device=self.device, dtype=torch.bool
        )
        self.r1_release_step = torch.full(
            (num_envs, 2), -1, device=self.device, dtype=torch.long
        )
        self.r1_release_lag = torch.zeros(
            num_envs, device=self.device, dtype=torch.float
        )
        self.r1_transition_count = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.r1_last_transition_reason = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        # E002-A high-water progress and contact gate state. The tail command
        # remains deterministic and actor-visible; these values remain reward/
        # diagnostic state and are not privileged actor inputs.
        self.e002_theta_start = torch.zeros(num_envs, device=self.device)
        self.e002_theta_prev = torch.zeros(num_envs, device=self.device)
        self.e002_high_water = torch.zeros(num_envs, device=self.device)
        self.e002_paid_high_water = torch.zeros(num_envs, device=self.device)
        self.e002_gross_travel = torch.zeros(num_envs, device=self.device)
        self.e002_reverse_travel = torch.zeros(num_envs, device=self.device)
        self.e002_contact_state = torch.zeros(
            (num_envs, 2), device=self.device, dtype=torch.bool
        )
        self.e002_pair_contact_streak = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.e002_pair_contact_loss_streak = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.e002_contact_acquired = torch.zeros(
            num_envs, device=self.device, dtype=torch.bool
        )
        self.e002_contact_potential_prev = torch.zeros(
            num_envs, device=self.device
        )
        self.e002_drive_step = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.e002_valid_pair_progress = torch.zeros(
            num_envs, device=self.device
        )
        self.e002_normal_proxy_progress = torch.zeros(num_envs, device=self.device)
        self.e002_target_validated = torch.zeros(
            num_envs, device=self.device, dtype=torch.bool
        )
        self.e002_target_hold = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        self.e002_success = torch.zeros(num_envs, device=self.device, dtype=torch.bool)
        self.e002_milestone_paid = torch.zeros(
            (num_envs, 2), device=self.device, dtype=torch.bool
        )
        self.e002_termination_reason = torch.zeros(
            num_envs, device=self.device, dtype=torch.long
        )
        self.e002_cycle = None
        if self.e002_cycle_cfg.get('enable', False):
            from dexscrew.tasks.e002_cycle import E002Cycle
            self.e002_cycle = E002Cycle(num_envs, self.device, self.e002_cycle_cfg)
        self.e002_demo = None
        demo_cfg = self.config.get('env', {}).get('e002_demo_reset', {})
        if demo_cfg.get('enable', False):
            assert self.e002_cycle is not None
            from dexscrew.tasks.e002_demo_resets import E002DemoResets
            self.e002_demo = E002DemoResets(self, demo_cfg)

    def _setup_r1_motion_config(self, cfg):
        """Parse the bounded R1 two-finger reward/phase configuration.

        Net contact force is intentionally used only as a contact/torque proxy:
        Isaac Gym exposes its resultant at the fingertip body, not a pairwise
        fingertip-to-nut contact manifold. The corresponding extras keep that
        limitation visible for the first instrumentation experiment.
        """

        def as_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, bool):
                return value
            if isinstance(value, (int, float)):
                return bool(value)
            if isinstance(value, str):
                return value.strip().lower() in {"true", "1", "yes", "y", "on"}
            return default

        self.r1_motion_enabled = as_bool(cfg.get("enable", False))
        self.r1_contact_force_on = float(cfg.get("contact_force_on", 0.30))
        self.r1_contact_force_off = float(cfg.get("contact_force_off", 0.15))
        self.r1_contact_dist_on = float(cfg.get("contact_dist_on", 0.045))
        self.r1_contact_dist_off = float(cfg.get("contact_dist_off", 0.050))
        self.r1_torque_min = float(cfg.get("torque_min", 0.01))
        self.r1_torque_clip = float(cfg.get("torque_clip", 0.10))
        self.r1_force_sign = float(cfg.get("force_sign", -1.0))
        self.r1_drive_stroke = float(cfg.get("drive_stroke", 0.12))
        self.r1_drive_max_steps = int(cfg.get("drive_max_steps", 16))
        self.r1_release_window_steps = int(cfg.get("release_window_steps", 5))
        self.r1_return_min_steps = int(cfg.get("return_min_steps", 3))
        self.r1_release_bonus_scale = float(cfg.get("release_bonus_scale", 0.20))
        self.r1_unload_asymmetry_scale = float(
            cfg.get("unload_asymmetry_scale", 0.15)
        )
        self.r1_return_drag_scale = float(cfg.get("return_drag_scale", 0.10))
        self.r1_return_reverse_scale = float(
            cfg.get("return_reverse_scale", 0.50)
        )
        if self.r1_motion_enabled and (
            self.r1_contact_force_on <= self.r1_contact_force_off
            or self.r1_contact_dist_off <= self.r1_contact_dist_on
            or self.r1_torque_clip <= 0.0
            or self.r1_drive_max_steps < 1
            or self.r1_release_window_steps < 1
        ):
            raise ValueError("Invalid env.r1_motion hysteresis or phase configuration")

    def _setup_e002_motion_config(self, cfg):
        """Parse the opt-in E002-A monotonic drive contract."""
        enabled = cfg.get("enable", False)
        if isinstance(enabled, str):
            enabled = enabled.strip().lower() in {"true", "1", "yes", "y", "on"}
        self.e002_motion_enabled = bool(enabled)
        self.e002_cycle_cfg = self.config.get('env', {}).get('e002_cycle', {})
        self.e002_acceptance_mode = str(cfg.get("acceptance_mode", "strict_normal_proxy"))
        if self.e002_acceptance_mode not in ("strict_normal_proxy", "contact_progress"):
            raise ValueError("unknown E002 acceptance mode")
        self.e002_direction = float(cfg.get("direction", 1.0))
        self.e002_target_angle = float(cfg.get("target_angle", np.pi / 6.0))
        self.e002_target_band = float(cfg.get("target_band", np.pi / 36.0))
        self.e002_delta_ref = float(cfg.get("delta_ref", np.pi / 60.0))
        self.e002_torque_min = float(cfg.get("torque_min", 0.01))
        self.e002_torque_saturation = float(cfg.get("torque_saturation", 0.10))
        self.e002_pair_gate_min = float(cfg.get("pair_gate_min", 0.50))
        self.e002_force_sign = float(cfg.get("force_sign", -1.0))
        self.e002_invalid_forward_source = str(cfg.get("invalid_forward_source", "pair_quality"))
        if self.e002_invalid_forward_source not in ("pair_quality", "contact_quality"):
            raise ValueError("unknown E002 invalid-forward source")
        self.e002_torque_source = str(cfg.get("torque_source", "link_origin_proxy"))
        if self.e002_torque_source not in ("link_origin_proxy", "contact_point_normal"):
            raise ValueError("unknown E002 torque source")
        self.e002_forward_source = str(cfg.get("forward_source", "pair_quality"))
        if self.e002_forward_source not in ("pair_quality", "nut_contact_quality"):
            raise ValueError("unknown E002 forward source")
        if self.e002_motion_enabled and self.e002_acceptance_mode == "contact_progress" and (
                self.e002_torque_source != "contact_point_normal"
                or self.e002_forward_source != "nut_contact_quality"
                or self.e002_invalid_forward_source != "contact_quality"):
            raise ValueError("contact_progress requires nut-only contact reward and diagnostic normal moments")
        if (self.e002_motion_enabled and self.e002_forward_source == "nut_contact_quality"
                and self.e002_torque_source != "contact_point_normal"):
            raise ValueError("nut_contact_quality requires nut-specific contact records")
        if self.e002_motion_enabled and self.e002_torque_source == "contact_point_normal" and self.config["sim"]["use_gpu_pipeline"]:
            raise ValueError("contact_point_normal requires CPU tensors; GPU PhysX and CUDA PPO remain supported")
        self.e002_contact_force_on = float(cfg.get("contact_force_on", 0.30))
        self.e002_contact_force_off = float(cfg.get("contact_force_off", 0.15))
        self.e002_contact_dist_on = float(cfg.get("contact_dist_on", 0.045))
        self.e002_contact_dist_off = float(cfg.get("contact_dist_off", 0.050))
        self.e002_drive_deadline_steps = int(cfg.get("drive_deadline_steps", 80))
        # `contact_grace_steps` is accepted as a legacy alias for old launch
        # commands, but no longer conflates missing positive torque with lost
        # physical contact.
        legacy_contact_grace = int(cfg.get("contact_grace_steps", 8))
        self.e002_contact_acquire_deadline_steps = int(
            cfg.get("contact_acquire_deadline_steps", legacy_contact_grace)
        )
        self.e002_contact_stable_steps = int(cfg.get("contact_stable_steps", 3))
        self.e002_contact_loss_grace_steps = int(
            cfg.get("contact_loss_grace_steps", 3)
        )
        self.e002_approach_scale = float(cfg.get("approach_scale", 0.5))
        self.e002_contact_bonus = float(cfg.get("contact_bonus", 1.0))
        self.e002_invalid_forward_scale = float(
            cfg.get("invalid_forward_scale", 4.0)
        )
        self.e002_target_hold_steps = int(cfg.get("target_hold_steps", 4))
        self.e002_workspace_radius = float(cfg.get("workspace_radius", 0.075))
        self.e002_hard_step_angle = float(cfg.get("hard_step_angle", np.pi / 12.0))
        self.e002_overspeed_angle = float(cfg.get("overspeed_angle", np.pi / 36.0))
        self.e002_overshoot_angle = float(cfg.get("overshoot_angle", 7.0 * np.pi / 36.0))
        self.e002_phase_command_dim = int(cfg.get("phase_command_dim", 5))
        if self.e002_motion_enabled and (
            self.e002_direction == 0.0
            or self.e002_target_angle <= 0.0
            or self.e002_delta_ref <= 0.0
            or not (0.0 <= self.e002_pair_gate_min <= 1.0)
            or self.e002_phase_command_dim != 5
            or self.e002_target_hold_steps < 1
            or self.e002_contact_acquire_deadline_steps < 1
            or self.e002_contact_stable_steps < 1
            or self.e002_contact_loss_grace_steps < 1
            or self.e002_approach_scale < 0.0
            or self.e002_contact_bonus < 0.0
            or self.e002_invalid_forward_scale < 0.0
        ):
            raise ValueError("Invalid env.e002_motion configuration")

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
        # Explicitly disable self-collisions to prevent initial explosion
        # hand_asset_options.self_collisions = False
        # Collapse fixed joints to merge tactile sensors into parent links (prevents self-collision between fixed bodies)
        # hand_asset_options.collapse_fixed_joints = False

        if self.torque_control:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_EFFORT)
        else:
            hand_asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        self.hand_asset = self.gym.load_asset(
            self.sim, asset_root, hand_asset_file, hand_asset_options
        )
        # Fingertip rigid bodies for dexh13_right (order matters: index first, thumb last)
        self.fingertip_handles = [
            self.gym.find_asset_rigid_body_index(self.hand_asset, name)
            for name in [
                "right_index_link_3",
                "right_middle_link_3",
                "right_ring_link_3",
                "right_thumb_link_3",
            ]
        ]

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
            if self.config["env"].get("e003_wrench_audit", False):
                # Diagnostic only: solver resultants include joint reactions;
                # they are not assumed to isolate contact or support load.
                for body_name in ("base", "bolt", "nut"):
                    body_index = self.gym.find_asset_rigid_body_index(object_asset, body_name)
                    assert body_index >= 0, body_name
                    props = gymapi.ForceSensorProperties()
                    props.enable_forward_dynamics_forces = False
                    props.enable_constraint_solver_forces = True
                    props.use_world_frame = True
                    assert self.gym.create_asset_force_sensor(object_asset, body_index, gymapi.Transform(), props) >= 0
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

        xhand_dof_lower_limits = [
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
            -0.35,
            0.0,
            0.0,
            0.0,
        ]

        xhand_dof_upper_limits = [
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
            0.35,
            1.57,
            1.57,
            1.57,
        ]

        xhand_effort_limits = [1.0 for _ in xhand_dof_lower_limits]
        # IMPORTANT: The dexh13_right URDF sets DOF velocity limits to 0.
        # A zero velocity limit can lead to non-finite DOF state tensors on GPU pipeline.
        xhand_velocity_limits = [10.0 for _ in xhand_dof_lower_limits]

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
        self.xhand_dof_upper_limits = np.array(xhand_dof_upper_limits)

        for i in range(self.num_xhand_hand_dofs):
            xhand_hand_dof_props["lower"][i] = xhand_dof_lower_limits[i]
            xhand_hand_dof_props["upper"][i] = xhand_dof_upper_limits[i]
            self.xhand_hand_dof_lower_limits.append(xhand_dof_lower_limits[i])
            self.xhand_hand_dof_upper_limits.append(xhand_dof_upper_limits[i])

            xhand_hand_dof_props["effort"][i] = xhand_effort_limits[i]
            xhand_hand_dof_props["velocity"][i] = xhand_velocity_limits[i]

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

            xhand_hand_dof_props["friction"][i] = 0.01
            xhand_hand_dof_props["armature"][i] = 0.001

        self.xhand_hand_dof_lower_limits = to_torch(
            self.xhand_hand_dof_lower_limits, device=self.device
        )
        self.xhand_hand_dof_upper_limits = to_torch(
            self.xhand_hand_dof_upper_limits, device=self.device
        )
        return xhand_hand_dof_props

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

    def _configured_hand_root_quats(self, batch_size):
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
        if hand_root_rpy is None:
            return None
        roll = torch.full(
            (batch_size,), hand_root_rpy[0], device=self.device, dtype=torch.float
        )
        pitch = torch.full(
            (batch_size,), hand_root_rpy[1], device=self.device, dtype=torch.float
        )
        yaw = torch.full(
            (batch_size,), hand_root_rpy[2], device=self.device, dtype=torch.float
        )
        return quat_from_euler_xyz(roll, pitch, yaw)

    def _configured_hand_root_positions(self, batch_size):
        hand_root_pos = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPos"), 3, "env.asset.handRootPos"
        )
        if hand_root_pos is None:
            return None
        pos = torch.tensor(
            hand_root_pos, device=self.device, dtype=torch.float
        ).unsqueeze(0).repeat(batch_size, 1)
        pos_noise = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPosNoise", [0.0, 0.0, 0.0]),
            3,
            "env.asset.handRootPosNoise",
        )
        if any(abs(value) > 0.0 for value in pos_noise):
            pos += torch.rand_like(pos) * torch.tensor(
                pos_noise, device=self.device, dtype=torch.float
            )
        return pos

    def _init_object_pose(self):
        xhand_hand_start_pose = gymapi.Transform()
        hand_root_pos = self._resolve_numeric_vector(
            self.hand_asset_cfg.get("handRootPos"), 3, "env.asset.handRootPos"
        )
        xhand_hand_start_pose.p = gymapi.Vec3(
            *(hand_root_pos if hand_root_pos is not None else [0.0, 0.0, 0.21])
        )

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

        # Object position relative to hand
        pose_dx, pose_dy, pose_dz = 0.00, 0.00, 0.00

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = xhand_hand_start_pose.p.x + pose_dx
        object_start_pose.p.y = xhand_hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = xhand_hand_start_pose.p.z + pose_dz

        return xhand_hand_start_pose, object_start_pose

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
