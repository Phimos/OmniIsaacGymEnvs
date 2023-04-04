import math
from abc import abstractmethod
from pdb import set_trace

import numpy as np
import omni.replicator.isaac as dr
import open3d as o3d
import torch
import torch.nn.functional as F
from omni.isaac.core.prims import RigidContactView, RigidPrim, RigidPrimView, XFormPrim
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.sensor.scripts.camera import Camera
from omni.isaac.utils.scripts.camera_utils import DynamicCamera

from omniisaacgymenvs.robots.articulations.shadow_hand import ShadowHand
from omniisaacgymenvs.robots.articulations.views.shadow_hand_view import ShadowHandView
from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.utils.config_utils.sim_config import SimConfig

YCB_DATASET_DIR = "/Props/YCB/Axis_Aligned"
YCB_DATASET_OBJECTS = [
    "002_master_chef_can",
    "003_cracker_box",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "009_gelatin_box",
    "010_potted_meat_can",
    "011_banana",
    "019_pitcher_base",
    "021_bleach_cleanser",
    "024_bowl",
    "025_mug",
    "035_power_drill",
    "036_wood_block",
    "037_scissors",
    "040_large_marker",
    "051_large_clamp",
    "052_extra_large_clamp",
    "061_foam_brick",
]


class GraspingTask(RLTask):
    def __init__(self, name: str, sim_config: SimConfig, env: VecEnvBase, offset=None) -> None:
        """[summary]"""
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in YCB_DATASET_OBJECTS

        self.obs_type = self._task_cfg["env"]["observationType"]
        if not (self.obs_type in ["openai", "full_no_vel", "full", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]"
            )
        print("Obs type:", self.obs_type)
        self.num_obs_dict = {
            "openai": 42,
            "full_no_vel": 77,
            "full": 157,
            "full_state": 187,
        }

        self.asymmetric_obs = self._task_cfg["env"]["asymmetric_observations"]
        self.use_vel_obs = False

        self.fingertip_obs = True
        self.fingertips = [
            "robot0:ffdistal",
            "robot0:mfdistal",
            "robot0:rfdistal",
            "robot0:lfdistal",
            "robot0:thdistal",
        ]
        self.num_fingertips = len(self.fingertips)

        self.object_scale = torch.tensor([1.0, 1.0, 1.0])
        self.force_torque_obs_scale = 10.0

        num_states = 0
        if self.asymmetric_obs:
            num_states = 187

        self._num_observations = self.num_obs_dict[self.obs_type]
        self._num_actions = 20
        self._num_states = num_states

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self.dist_reward_scale = self._task_cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self._task_cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self._task_cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self._task_cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self._task_cfg["env"]["reachGoalBonus"]
        self.fall_dist = self._task_cfg["env"]["fallDistance"]
        self.fall_penalty = self._task_cfg["env"]["fallPenalty"]
        self.rot_eps = self._task_cfg["env"]["rotEps"]
        self.vel_obs_scale = self._task_cfg["env"]["velObsScale"]

        self.reset_position_noise = self._task_cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self._task_cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self._task_cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self._task_cfg["env"]["resetDofVelRandomInterval"]

        self.hand_dof_speed_scale = self._task_cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self._task_cfg["env"]["useRelativeControl"]
        self.act_moving_average = self._task_cfg["env"]["actionsMovingAverage"]

        self.max_episode_length = self._task_cfg["env"]["episodeLength"]
        self.reset_time = self._task_cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self._task_cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self._task_cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self._task_cfg["env"].get("averFactor", 0.1)

        self.dt = 1.0 / 60
        control_freq_inv = self._task_cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        RLTask.__init__(self, name, env)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.randomization_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.av_factor = torch.tensor(self.av_factor, dtype=torch.float, device=self.device)
        self.total_successes = 0
        self.total_resets = 0

        object_ply_path = "/home/user/Downloads/011_banana.ply"
        num_points = 1024
        pointcloud = o3d.io.read_point_cloud(object_ply_path)
        pointcloud = torch.from_numpy(np.asarray(pointcloud.points)).float()
        pointcloud = pointcloud[torch.randperm(pointcloud.shape[0])[:num_points]]
        self.object_pointcloud = pointcloud.to(self.device)
        self.num_points = num_points

    #####################################################################
    ###==================initialize simulator scenes==================###
    #####################################################################

    def get_hand(self):
        hand_start_translation = torch.tensor([0.0, 0.0, 0.5], device=self.device)
        hand_start_orientation = torch.tensor([0.70711, 0.70711, 0.0, 0.0], device=self.device)

        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=hand_start_translation,
            orientation=hand_start_orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)
        pose_dy, pose_dz = -0.39, 0.10
        return hand_start_translation, pose_dy, pose_dz

    def get_hand_view(self, scene: Scene):
        hand_view = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(hand_view._fingers)
        return hand_view

    def get_object(self, hand_start_translation, pose_dy, pose_dz):
        self.object_start_translation = hand_start_translation.clone()
        self.object_start_translation[1] += pose_dy
        self.object_start_translation[2] += pose_dz
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        if self.object_type == "block":
            self.object_usd_path = f"{self._assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        else:
            # self.object_usd_path = f"{self._assets_root_path}{YCB_DATASET_DIR}/{self.object_type}.usd"
            self.object_usd_path = (
                f"/home/user/Downloads/YCB/Axis_Aligned/{self.object_type}_instanceable_rigid_body.usd"
            )
            assert os.path.exists(self.object_usd_path), "Object usd path does not exist: {}".format(
                self.object_usd_path
            )
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/object")
        obj = XFormPrim(
            prim_path=self.default_zero_env_path + "/object/object",
            name="object",
            translation=self.object_start_translation,
            orientation=self.object_start_orientation,
            scale=self.object_scale,
        )
        self._sim_config.apply_articulation_settings(
            "object", get_prim_at_path(obj.prim_path), self._sim_config.parse_actor_config("object")
        )

    def get_goal(self):
        self.goal_displacement_tensor = torch.tensor([-0.2, -0.06, 0.12], device=self.device)
        self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
        self.goal_start_translation[2] -= 0.04
        self.goal_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        add_reference_to_stage(self.object_usd_path, self.default_zero_env_path + "/goal")
        goal = XFormPrim(
            prim_path=self.default_zero_env_path + "/goal",
            name="goal",
            translation=self.goal_start_translation,
            orientation=self.goal_start_orientation,
            scale=self.object_scale,
        )
        self._sim_config.apply_articulation_settings(
            "goal", get_prim_at_path(goal.prim_path), self._sim_config.parse_actor_config("goal_object")
        )

    def get_cameras(self):
        camera_position = torch.tensor([-0.5, 0.0, 0.5], device=self.device)
        camera_rotation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        camera = Camera(
            prim_path=self.default_zero_env_path + "/camera",
            name="camera",
            resolution=(640, 480),
            translation=camera_position,
            orientation=camera_rotation,
        )
        camera.add_distance_to_camera_to_frame()
        camera.add_pointcloud_to_frame()
        camera.set_dt(self.dt)

        self._camera = camera

        # camera = DynamicCamera(
        #     stage=self._stage, base_path=self.default_zero_env_path + "/camera", camera_name="camera"
        # )

    def set_up_scene(self, scene: Scene) -> None:
        scene.add_default_ground_plane()

        self._stage = get_current_stage()
        self._assets_root_path = get_assets_root_path()
        hand_start_translation, pose_dy, pose_dz = self.get_hand()
        self.get_object(hand_start_translation, pose_dy, pose_dz)
        self.get_goal()

        # self.get_cameras()

        replicate_physics = False if self._dr_randomizer.randomize else True
        super().set_up_scene(scene, replicate_physics)

        self._hands = self.get_hand_view(scene)
        scene.add(self._hands)
        self._objects = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
        )
        scene.add(self._objects)
        self._goals = RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )
        self._goals._non_root_link = True  # hack to ignore kinematics
        scene.add(self._goals)

        # self._camera.initialize()

        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    #####################################################################
    ###=====================compute observations======================###
    #####################################################################

    def compute_fingertip_positions(self):
        positions = self._hands._fingers.get_world_poses(clone=False)[0]
        positions = positions.reshape(self.num_envs, self.num_fingertips, 3)
        positions -= self._env_pos.reshape(self.num_envs, 1, 3)
        return positions

    def compute_fingertip_rotations(self):
        return self._hands._fingers.get_world_poses(clone=False)[1]

    def compute_fingertip_velocities(self):
        return self._hands._fingers.get_velocities(clone=False)

    def compute_fingertip_object_minimum_distances(self):
        object_pointcloud = self.compute_object_pointcloud()  # [num_envs, num_points, 3]
        fingertip_positions = self.compute_fingertip_positions()  # [num_envs, num_fingertips, 3]

        # Compute distances between fingertip positions and object pointcloud
        distances = torch.norm(object_pointcloud[:, None, :, :] - fingertip_positions[:, :, None, :], dim=-1)

        # Find minimum distance for each fingertip and object
        min_distances = distances.min(dim=2)[0]  # [num_envs, num_fingertips]

        return min_distances

    def compute_hand_joint_angles(self):
        return self._hands.get_joint_positions(clone=False)

    def compute_hand_joint_velocities(self):
        return self._hands.get_joint_velocities(clone=False)

    def compute_object_positions(self):
        positions = self._objects.get_world_poses(clone=False)[0]
        positions -= self._env_pos
        return positions

    def compute_object_rotations(self):
        return self._objects.get_world_poses(clone=False)[1]

    def compute_object_velocities(self):
        return self._objects.get_velocities(clone=False)

    def compute_object_linear_velocities(self):
        return self._objects.get_linear_velocities(clone=False)

    def compute_object_angular_velocities(self):
        return self._objects.get_angular_velocities(clone=False)

    def compute_object_pointcloud(self):
        positions, rotations = self._objects.get_world_poses(clone=False)
        pointcloud = self.object_pointcloud.clone()

        pointcloud = pointcloud.reshape(1, -1, 3).repeat(self.num_envs, 1, 1).reshape(-1, 3)
        rotations = rotations.reshape(-1, 1, 4).repeat(1, self.num_points, 1).reshape(-1, 4)

        pointcloud = quat_rotate(rotations, pointcloud).reshape(self.num_envs, -1, 3)
        pointcloud += positions.reshape(self.num_envs, 1, 3)
        pointcloud -= self._env_pos.reshape(self.num_envs, 1, 3)
        return pointcloud

    #####################################################################
    ###=====================RL pipeline functions=====================###
    #####################################################################

    def get_observations(self):
        self.get_object_goal_observations()

        self.fingertip_pos, self.fingertip_rot = self._hands._fingers.get_world_poses(clone=False)
        self.fingertip_pos -= self._env_pos.repeat((1, self.num_fingertips)).reshape(
            self.num_envs * self.num_fingertips, 3
        )
        self.fingertip_velocities = self._hands._fingers.get_velocities(clone=False)

        # fingertip_positions = self.compute_fingertip_positions()
        # print(fingertip_positions)
        # print("fingertip_positions shape: ", fingertip_positions.shape)

        # print(self.fingertip_pos)

        # pointcloud = self.compute_object_pointcloud()
        # print(pointcloud)
        # print(pointcloud.shape)

        # minimum_distance = self.compute_fingertip_object_minimum_distances()
        # print(minimum_distance)
        # print(minimum_distance.shape)

        # print(self._camera.get_current_frame())

        self.hand_dof_pos = self._hands.get_joint_positions(clone=False)
        self.hand_dof_vel = self._hands.get_joint_velocities(clone=False)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.vec_sensor_tensor = self._hands._physics_view.get_force_sensor_forces().reshape(
                self.num_envs, 6 * self.num_fingertips
            )

        if self.obs_type == "openai":
            self.compute_fingertip_observations(True)
        elif self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
            self.compute_full_state(False)
        else:
            print("Unkown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

        observations = {self._hands.name: {"obs_buf": self.obs_buf}}
        return observations

    def compute_fingertip_observations(self, no_vel=False):
        if no_vel:
            # Per https://arxiv.org/pdf/1808.00177.pdf Table 2
            #   Fingertip positions
            #   Object Position, but not orientation
            #   Relative target orientation

            # 3*self.num_fingertips = 15
            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 15)
            self.obs_buf[:, 15:18] = self.object_pos
            self.obs_buf[:, 18:22] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 22:42] = self.actions
        else:
            # 13*self.num_fingertips = 65
            self.obs_buf[:, 0:65] = self.fingertip_state.reshape(self.num_envs, 65)

            self.obs_buf[:, 0:15] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 15:35] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[:, 35:65] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.obs_buf[:, 65:68] = self.object_pos
            self.obs_buf[:, 68:72] = self.object_rot
            self.obs_buf[:, 72:75] = self.object_linvel
            self.obs_buf[:, 75:78] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 78:81] = self.goal_pos
            self.obs_buf[:, 81:85] = self.goal_rot
            self.obs_buf[:, 85:89] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 89:109] = self.actions

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )

            self.obs_buf[:, 24:37] = self.object_pos
            self.obs_buf[:, 27:31] = self.object_rot
            self.obs_buf[:, 31:34] = self.goal_pos
            self.obs_buf[:, 34:38] = self.goal_rot
            self.obs_buf[:, 38:42] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
            self.obs_buf[:, 42:57] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 57:77] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel

            self.obs_buf[:, 48:51] = self.object_pos
            self.obs_buf[:, 51:55] = self.object_rot
            self.obs_buf[:, 55:58] = self.object_linvel
            self.obs_buf[:, 58:61] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 61:64] = self.goal_pos
            self.obs_buf[:, 64:68] = self.goal_rot
            self.obs_buf[:, 68:72] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            # (7+6)*self.num_fingertips = 65
            self.obs_buf[:, 72:87] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[:, 87:107] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[:, 107:137] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.obs_buf[:, 137:157] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.states_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
            # self.states_buf[:, 2*self.num_hand_dofs:3*self.num_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 2 * self.num_hand_dofs  # 48
            self.states_buf[:, obj_obs_start : obj_obs_start + 3] = self.object_pos
            self.states_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.object_rot
            self.states_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start : goal_obs_start + 3] = self.goal_pos
            self.states_buf[:, goal_obs_start + 3 : goal_obs_start + 7] = self.goal_rot
            self.states_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            self.states_buf[
                :, fingertip_obs_start : fingertip_obs_start + 3 * self.num_fingertips
            ] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.states_buf[
                :, fingertip_obs_start + 3 * self.num_fingertips : fingertip_obs_start + 7 * self.num_fingertips
            ] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.states_buf[
                :, fingertip_obs_start + 7 * self.num_fingertips : fingertip_obs_start + 13 * self.num_fingertips
            ] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)

            self.states_buf[
                :, fingertip_obs_start + num_ft_states : fingertip_obs_start + num_ft_states + num_ft_force_torques
            ] = (self.force_torque_obs_scale * self.vec_sensor_tensor)

            # obs_end = 72 + 65 + 30 = 167
            # obs_total = obs_end + num_actions = 187
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end : obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0 : self.num_hand_dofs] = unscale(
                self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits
            )
            self.obs_buf[:, self.num_hand_dofs : 2 * self.num_hand_dofs] = self.vel_obs_scale * self.hand_dof_vel
            self.obs_buf[:, 2 * self.num_hand_dofs : 3 * self.num_hand_dofs] = (
                self.force_torque_obs_scale * self.dof_force_tensor
            )

            obj_obs_start = 3 * self.num_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start : obj_obs_start + 3] = self.object_pos
            self.obs_buf[:, obj_obs_start + 3 : obj_obs_start + 7] = self.object_rot
            self.obs_buf[:, obj_obs_start + 7 : obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10 : obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start : goal_obs_start + 3] = self.goal_pos
            self.obs_buf[:, goal_obs_start + 3 : goal_obs_start + 7] = self.goal_rot
            self.obs_buf[:, goal_obs_start + 7 : goal_obs_start + 11] = quat_mul(
                self.object_rot, quat_conjugate(self.goal_rot)
            )

            # fingertip observations, state(pose and vel) + force-torque sensors
            num_ft_states = 13 * self.num_fingertips  # 65
            num_ft_force_torques = 6 * self.num_fingertips  # 30

            fingertip_obs_start = goal_obs_start + 11  # 72
            self.obs_buf[
                :, fingertip_obs_start : fingertip_obs_start + 3 * self.num_fingertips
            ] = self.fingertip_pos.reshape(self.num_envs, 3 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + 3 * self.num_fingertips : fingertip_obs_start + 7 * self.num_fingertips
            ] = self.fingertip_rot.reshape(self.num_envs, 4 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + 7 * self.num_fingertips : fingertip_obs_start + 13 * self.num_fingertips
            ] = self.fingertip_velocities.reshape(self.num_envs, 6 * self.num_fingertips)
            self.obs_buf[
                :, fingertip_obs_start + num_ft_states : fingertip_obs_start + num_ft_states + num_ft_force_torques
            ] = (self.force_torque_obs_scale * self.vec_sensor_tensor)

            # obs_end = 96 + 65 + 30 = 167
            # obs_total = obs_end + num_actions = 187
            obs_end = fingertip_obs_start + num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end : obs_end + self.num_actions] = self.actions

    def post_reset(self):
        self.num_hand_dofs = self._hands.num_dof
        self.actuated_dof_indices = self._hands.actuated_dof_indices

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        dof_limits = self._hands.get_dof_limits()
        self.hand_dof_lower_limits, self.hand_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        self.hand_dof_default_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        self.hand_dof_default_vel = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)

        self.object_init_pos, self.object_init_rot = self._objects.get_world_poses()
        self.object_init_pos -= self._env_pos
        self.object_init_velocities = torch.zeros_like(
            self._objects.get_velocities(), dtype=torch.float, device=self.device
        )

        self.goal_pos = self.object_init_pos.clone()
        self.goal_pos[:, 2] -= 0.04
        self.goal_rot = self.object_init_rot.clone()

        self.goal_init_pos = self.goal_pos.clone()
        self.goal_init_rot = self.goal_rot.clone()

        # randomize all envs
        indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

    def get_object_goal_observations(self):
        self.object_pos, self.object_rot = self._objects.get_world_poses(clone=False)
        self.object_pos -= self._env_pos
        self.object_velocities = self._objects.get_velocities(clone=False)
        self.object_linvel = self.object_velocities[:, 0:3]
        self.object_angvel = self.object_velocities[:, 3:6]

    def calculate_metrics(self):
        (
            self.rew_buf[:],
            self.reset_buf[:],
            self.reset_goal_buf[:],
            self.progress_buf[:],
            self.successes[:],
            self.consecutive_successes[:],
        ) = compute_hand_reward(
            self.rew_buf,
            self.reset_buf,
            self.reset_goal_buf,
            self.progress_buf,
            self.successes,
            self.consecutive_successes,
            self.max_episode_length,
            self.object_pos,
            self.object_rot,
            self.goal_pos,
            self.goal_rot,
            self.dist_reward_scale,
            self.rot_reward_scale,
            self.rot_eps,
            self.actions,
            self.action_penalty_scale,
            self.success_tolerance,
            self.reach_goal_bonus,
            self.fall_dist,
            self.fall_penalty,
            self.max_consecutive_successes,
            self.av_factor,
        )

        self.reset_buf[:] = 0

        self.extras["consecutive_successes"] = self.consecutive_successes.mean()
        self.randomization_buf += 1

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()
            # The direct average shows the overall result more quickly, but slightly undershoots long term policy performance.
            print(
                "Direct average consecutive successes = {:.1f}".format(
                    direct_average_successes / (self.total_resets + self.num_envs)
                )
            )
            if self.total_resets > 0:
                print(
                    "Post-Reset average consecutive successes = {:.1f}".format(self.total_successes / self.total_resets)
                )

    def pre_physics_step(self, actions):
        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        reset_buf = self.reset_buf.clone()

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids)
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)

        if self.use_relative_control:
            targets = (
                self.prev_targets[:, self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * self.actions
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                targets,
                self.hand_dof_lower_limits[self.actuated_dof_indices],
                self.hand_dof_upper_limits[self.actuated_dof_indices],
            )
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(
                self.actions,
                self.hand_dof_lower_limits[self.actuated_dof_indices],
                self.hand_dof_upper_limits[self.actuated_dof_indices],
            )
            self.cur_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
                + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            )
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(
                self.cur_targets[:, self.actuated_dof_indices],
                self.hand_dof_lower_limits[self.actuated_dof_indices],
                self.hand_dof_upper_limits[self.actuated_dof_indices],
            )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self._hands.set_joint_position_targets(
            self.cur_targets[:, self.actuated_dof_indices], indices=None, joint_indices=self.actuated_dof_indices
        )

        if self._dr_randomizer.randomize:
            rand_envs = torch.where(
                self.randomization_buf >= self._dr_randomizer.min_frequency,
                torch.ones_like(self.randomization_buf),
                torch.zeros_like(self.randomization_buf),
            )
            rand_env_ids = torch.nonzero(torch.logical_and(rand_envs, reset_buf))
            dr.physics_view.step_randomization(rand_env_ids)
            self.randomization_buf[rand_env_ids] = 0

    def is_done(self):
        pass

    def reset_target_pose(self, env_ids):
        # reset goal
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(
            rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        self.goal_pos[env_ids] = self.goal_init_pos[env_ids, 0:3]
        self.goal_rot[env_ids] = new_rot

        goal_pos, goal_rot = self.goal_pos.clone(), self.goal_rot.clone()
        goal_pos[env_ids] = (
            self.goal_pos[env_ids] + self.goal_displacement_tensor + self._env_pos[env_ids]
        )  # add world env pos

        self._goals.set_world_poses(goal_pos[env_ids], goal_rot[env_ids], indices)
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids):
        indices = env_ids.to(dtype=torch.int32)
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_dofs * 2 + 5), device=self.device)

        self.reset_target_pose(env_ids)

        # reset object
        new_object_pos = (
            self.object_init_pos[env_ids] + self.reset_position_noise * rand_floats[:, 0:3] + self._env_pos[env_ids]
        )  # add world env pos

        new_object_rot = randomize_rotation(
            rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        )

        object_velocities = torch.zeros_like(self.object_init_velocities, dtype=torch.float, device=self.device)
        self._objects.set_velocities(object_velocities[env_ids], indices)
        self._objects.set_world_poses(new_object_pos, new_object_rot, indices)

        # reset hand
        delta_max = self.hand_dof_upper_limits - self.hand_dof_default_pos
        delta_min = self.hand_dof_lower_limits - self.hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5 : 5 + self.num_hand_dofs] + 1.0)

        pos = self.hand_dof_default_pos + self.reset_dof_pos_noise * rand_delta
        dof_pos = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        dof_pos[env_ids, :] = pos

        dof_vel = torch.zeros((self.num_envs, self.num_hand_dofs), device=self.device)
        dof_vel[env_ids, :] = (
            self.hand_dof_default_vel
            + self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_hand_dofs : 5 + self.num_hand_dofs * 2]
        )

        self.prev_targets[env_ids, : self.num_hand_dofs] = pos
        self.cur_targets[env_ids, : self.num_hand_dofs] = pos
        self.hand_dof_targets[env_ids, :] = pos

        self._hands.set_joint_position_targets(self.hand_dof_targets[env_ids], indices)
        self._hands.set_joint_positions(dof_pos[env_ids], indices)
        self._hands.set_joint_velocities(dof_vel[env_ids], indices)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def compute_hand_reward(
    rew_buf,
    reset_buf,
    reset_goal_buf,
    progress_buf,
    successes,
    consecutive_successes,
    max_episode_length: float,
    object_pos,
    object_rot,
    target_pos,
    target_rot,
    dist_reward_scale: float,
    rot_reward_scale: float,
    rot_eps: float,
    actions,
    action_penalty_scale: float,
    success_tolerance: float,
    reach_goal_bonus: float,
    fall_dist: float,
    fall_penalty: float,
    max_consecutive_successes: int,
    av_factor: float,
):
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(
        torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0)
    )  # changed quat convention

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0 / (torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions**2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty * action_penalty_scale

    # Find out which envs hit the goal and update successes count
    goal_resets = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reward = torch.where(goal_resets == 1, reward + reach_goal_bonus, reward)

    # Fall penalty: distance to the goal is larger than a threashold
    reward = torch.where(goal_dist >= fall_dist, reward + fall_penalty, reward)

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(
            torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf
        )
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    if max_consecutive_successes > 0:
        reward = torch.where(progress_buf >= max_episode_length - 1, reward + 0.5 * fall_penalty, reward)

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(
        num_resets > 0,
        av_factor * finished_cons_successes / num_resets + (1.0 - av_factor) * consecutive_successes,
        consecutive_successes,
    )

    return reward, resets, goal_resets, progress_buf, successes, cons_successes
