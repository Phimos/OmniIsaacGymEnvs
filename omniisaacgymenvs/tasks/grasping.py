import glob
import math
import re
from pdb import set_trace
from typing import List, Optional, Tuple

import numpy as np
import omni.replicator.isaac as dr
import open3d as o3d
import torch
import torch.nn.functional as F
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
from omni.isaac.core.prims import RigidContactView, RigidPrim, RigidPrimView, XFormPrim
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.scenes.scene import Scene
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.torch import *
from omni.isaac.gym.vec_env import VecEnvBase
from omni.isaac.sensor.scripts.camera import Camera
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive
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


class ShadowHand(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "robot",
        usd_path: Optional[str] = None,
        translation: Optional[torch.Tensor] = None,
        orientation: Optional[torch.Tensor] = None,
    ):
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            self._usd_path = "/home/user/Downloads/assets/robot/urdf/robot.usd"

        self._position = torch.tensor([0.0, 0.0, 0.0]) if translation is None else translation
        self._orientation = torch.tensor([1.0, 0.0, 0.0, 0.0]) if orientation is None else orientation
        add_reference_to_stage(self._usd_path, prim_path)
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=self._position,
            orientation=self._orientation,
            articulation_controller=None,
        )

    def set_shadow_hand_properties(self, stage, shadow_hand_prim):
        for link_prim in shadow_hand_prim.GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                # rb.GetDisableGravityAttr().Set(True)
                rb.GetRetainAccelerationsAttr().Set(True)

    def set_motor_control_mode(self, stage, shadow_hand_path):
        joints_config = {
            "robot0_WRJ1": {"stiffness": 5, "damping": 0.5, "max_force": 4.785},
            "robot0_WRJ0": {"stiffness": 5, "damping": 0.5, "max_force": 2.175},
            "robot0_FFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_FFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_MFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_MFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_RFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_RFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_LFJ4": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ3": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.9},
            "robot0_LFJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.7245},
            "robot0_THJ4": {"stiffness": 1, "damping": 0.1, "max_force": 2.3722},
            "robot0_THJ3": {"stiffness": 1, "damping": 0.1, "max_force": 1.45},
            "robot0_THJ2": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ1": {"stiffness": 1, "damping": 0.1, "max_force": 0.99},
            "robot0_THJ0": {"stiffness": 1, "damping": 0.1, "max_force": 0.81},
        }

        print(f"Setting motor control mode for {self.prim_path}")
        for joint_name, config in joints_config.items():
            set_drive(
                f"{self.prim_path}/robot/shadow_hand/joints/{joint_name}",
                "angular",
                "position",
                0.0,
                config["stiffness"] * np.pi / 180,
                config["damping"] * np.pi / 180,
                config["max_force"],
            )


class ShadowHandView(ArticulationView):
    def __init__(
        self,
        prim_paths_expr: str,
        name: Optional[str] = "RobotView",
    ):
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)

        self._fingertips = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/robot/shadow_hand/robot0.*distal",
            name="FingertipView",
            reset_xform_properties=False,
            track_contact_forces=True,
        )

        self._palms = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/robot/shadow_hand/robot0.*palm",
            name="PalmView",
            reset_xform_properties=False,
        )

        self._thumbs = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/robot/shadow_hand/robot0.*thdistal",
            name="ThumbView",
            reset_xform_properties=False,
            track_contact_forces=True,
        )

        self._meshes = RigidPrimView(
            prim_paths_expr="/World/envs/.*/robot/robot/shadow_hand/robot0_.*",
            name="MeshView",
            reset_xform_properties=False,
        )

    @property
    def arm_dof_names(self) -> List[str]:
        return self._arm_dof_names

    @property
    def arm_dof_indices(self) -> List[int]:
        return self._arm_dof_indices

    @property
    def hand_dof_names(self) -> List[str]:
        return self._hand_dof_names

    @property
    def hand_dof_indices(self) -> List[int]:
        return self._hand_dof_indices

    @property
    def actuated_dof_names(self) -> List[str]:
        return self._actuated_dof_names

    @property
    def actuated_dof_indices(self) -> List[int]:
        return self._actuated_dof_indices

    def initialize(self, physics_sim_view) -> None:
        super().initialize(physics_sim_view)
        self._arm_dof_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ]
        self._arm_dof_indices = [self.get_dof_index(dof_name) for dof_name in self._arm_dof_names]

        self._hand_dof_names = [
            "robot0_WRJ1",
            "robot0_WRJ0",
            "robot0_FFJ3",
            "robot0_FFJ2",
            "robot0_FFJ1",
            "robot0_MFJ3",
            "robot0_MFJ2",
            "robot0_MFJ1",
            "robot0_RFJ3",
            "robot0_RFJ2",
            "robot0_RFJ1",
            "robot0_LFJ4",
            "robot0_LFJ3",
            "robot0_LFJ2",
            "robot0_LFJ1",
            "robot0_THJ4",
            "robot0_THJ3",
            "robot0_THJ2",
            "robot0_THJ1",
            "robot0_THJ0",
        ]
        self._hand_dof_indices = [self.get_dof_index(dof_name) for dof_name in self._hand_dof_names]

        self._actuated_dof_names = self._arm_dof_names + self._hand_dof_names
        self._actuated_dof_indices = self._arm_dof_indices + self._hand_dof_indices

        limit_stiffness = torch.tensor([30.0] * self.num_fixed_tendons, device=self._device)
        damping = torch.tensor([0.1] * self.num_fixed_tendons, device=self._device)
        self.set_fixed_tendon_properties(dampings=damping, limit_stiffnesses=limit_stiffness)


class GraspingTask(RLTask):
    def __init__(self, name: str, sim_config: SimConfig, env: VecEnvBase, offset=None) -> None:
        """[summary]"""
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self.object_type = self._task_cfg["env"]["objectType"]
        assert self.object_type in YCB_DATASET_OBJECTS

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

        # Define observation space
        print("Observation type:", self._task_cfg["env"]["observationType"])
        self.num_points = 1024
        self.num_joints = 30
        self.observation_types = self._task_cfg["env"]["observationType"]
        self.num_observation_per_type = {
            "imagined_pointcloud": self.num_points * 3,
            "imagined_hand_pointcloud": self.num_points * 3,
            "palm_position": 3,
            "palm_rotation": 4,
            "fingertip_position": 3 * self.num_fingertips,
            "fingertip_object_minimum_distance": self.num_fingertips,
            "object_position": 3,
            "object_rotation": 4,
            "goal_position": 3,
            "goal_rotation": 4,
            "joint_position": self.num_joints,
            "joint_velocity": self.num_joints,
        }

        self._num_observations = sum([self.num_observation_per_type[obs_type] for obs_type in self.observation_types])
        self._num_actions = 26
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

        self.table_length = self._task_cfg["env"].get("tableLength", 1.0)
        self.table_width = self._task_cfg["env"].get("tableWidth", 1.2)
        self.table_height = self._task_cfg["env"].get("tableHeight", 0.6)

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

        self.arm_init_joint_positions = torch.tensor(
            [0, -1.25, 2.00, -np.pi / 4, np.pi / 2, -np.pi], device=self.device
        )

        # object_ply_path = "/home/user/Downloads/011_banana.ply"

        # pointcloud = o3d.io.read_point_cloud(object_ply_path)
        # pointcloud = torch.from_numpy(np.asarray(pointcloud.points)).float()
        # pointcloud = pointcloud[torch.randperm(pointcloud.shape[0])[: self.num_points]]
        # self.object_pointcloud = pointcloud.to(self.device)

        self.load_object_pointclouds("/home/user/Downloads/banana.ply")
        self.load_pointclouds()

    #####################################################################
    ###================= Initialize Simulator Scenes =================###
    #####################################################################

    def load_pointclouds(self, pointcloud_dir: os.PathLike = "/home/user/Downloads/assets/robot/another_pointcloud"):
        pointclouds = {}

        for filepath in glob.glob(os.path.join(pointcloud_dir, "*.ply")):
            pointcloud = o3d.io.read_point_cloud(filepath)
            pointcloud = torch.from_numpy(np.asarray(pointcloud.points)).float()
            pointcloud = torch.stack([-pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1]], dim=1)
            name = os.path.splitext(os.path.basename(filepath))[0]
            pointclouds[name] = pointcloud.to(self.device)

        # remove forearm pointcloud and focus on palm & fingers
        pointclouds.pop("forearm")
        self._cached_pointclouds = pointclouds

    def load_object_pointclouds(self, pointcloud_path: os.PathLike):
        pointcloud = o3d.io.read_point_cloud(pointcloud_path)
        pointcloud = torch.from_numpy(np.asarray(pointcloud.points)).float()
        pointcloud = torch.stack([-pointcloud[:, 0], pointcloud[:, 2], pointcloud[:, 1]], dim=1)
        self._object_pointcloud = pointcloud.to(self.device)

    def create_isaac_robot(self):
        robot_usd_path = "/home/user/Downloads/assets/robot/urdf/robot.usd"
        translation = torch.tensor([0.0, 0.0, 0.48], device=self.device)
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        robot = ShadowHand(
            prim_path=self.default_zero_env_path + "/robot",
            name="robot",
            usd_path=robot_usd_path,
            translation=translation,
            orientation=orientation,
        )
        print(robot.prim_path + "/robot/shadow_hand")
        self._sim_config.apply_articulation_settings(
            "robot",
            get_prim_at_path(robot.prim_path + "/robot"),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        robot.set_motor_control_mode(stage=self._stage, shadow_hand_path=robot.prim_path + "/robot/shadow_hand")

    def create_isaac_robot_view(self, scene: Scene):
        robot_view = ShadowHandView(prim_paths_expr="/World/envs/env_.*/robot/robot", name="robot_view")
        scene.add(robot_view._meshes)
        scene.add(robot_view._fingertips)
        scene.add(robot_view._palms)
        scene.add(robot_view._thumbs)

        names = [re.search(r"robot0_(.*)", path).group(1) for path in robot_view._meshes.prim_paths]
        names = names[: len(names) // self.num_envs]
        self._mesh_names = names
        return robot_view

    def create_table(self):
        translation = torch.tensor([1.0, 0.0, self.table_height / 2], device=self.device)
        orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        scale = torch.tensor([self.table_length, self.table_width, self.table_height], device=self.device)
        table = FixedCuboid(
            prim_path=self.default_zero_env_path + "/table",
            name="table",
            translation=translation,
            orientation=orientation,
            scale=scale,
        )

    def create_shadowhand(self, translation: Optional[torch.Tensor] = None, orientation: Optional[torch.Tensor] = None):
        translation = torch.tensor([0.0, 0.0, 0.5], device=self.device) if translation is None else translation
        orientation = (
            torch.tensor([0.70711, 0.70711, 0.0, 0.0], device=self.device) if orientation is None else orientation
        )

        shadow_hand = ShadowHand(
            prim_path=self.default_zero_env_path + "/shadow_hand",
            name="shadow_hand",
            translation=translation,
            orientation=orientation,
        )
        self._sim_config.apply_articulation_settings(
            "shadow_hand",
            get_prim_at_path(shadow_hand.prim_path),
            self._sim_config.parse_actor_config("shadow_hand"),
        )
        shadow_hand.set_shadow_hand_properties(stage=self._stage, shadow_hand_prim=shadow_hand.prim)
        shadow_hand.set_motor_control_mode(stage=self._stage, shadow_hand_path=shadow_hand.prim_path)
        pose_dy, pose_dz = -0.39, 0.10
        return translation, pose_dy, pose_dz

    def create_shadowhand_view(self, scene: Scene):
        hand_view = ShadowHandView(prim_paths_expr="/World/envs/.*/shadow_hand", name="shadow_hand_view")
        scene.add(hand_view._fingers)
        return hand_view

    def get_object_usd_path(self) -> str:
        if self.object_type == "block":
            assets_root_path = get_assets_root_path()
            return f"{assets_root_path}/Isaac/Props/Blocks/block_instanceable.usd"
        else:
            user_assets_root_path = "/home/user/Downloads/YCB/Axis_Aligned"
            return f"{user_assets_root_path}/{self.object_type}_instanceable_rigid_body.usd"

    def create_object(self):
        self.object_start_translation = torch.tensor([1.0, 0.2, self.table_height + 0.1], device=self.device)
        self.object_start_orientation = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)

        self.object_usd_path = self.get_object_usd_path()
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

    def create_object_view(self):
        return RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/object/object",
            name="object_view",
            reset_xform_properties=False,
            masses=torch.tensor([0.07087] * self._num_envs, device=self.device),
        )

    def create_goal(self):
        self.goal_displacement_tensor = torch.tensor([-0.2, -0.06, 0.12], device=self.device)
        self.goal_start_translation = self.object_start_translation + self.goal_displacement_tensor
        self.goal_start_translation[2] += 1.0
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

    def create_goal_view(self):
        return RigidPrimView(
            prim_paths_expr="/World/envs/env_.*/goal/object",
            name="goal_view",
            reset_xform_properties=False,
        )

    def create_camera(self):
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

        self.create_isaac_robot()
        self.create_table()
        self.create_object()
        self.create_goal()

        replicate_physics = False if self._dr_randomizer.randomize else True
        super().set_up_scene(scene, replicate_physics)

        self._robots = self.create_isaac_robot_view(scene)
        self._objects = self.create_object_view()
        self._goals = self.create_goal_view()
        self._goals._non_root_link = True  # hack to ignore kinematics

        scene.add(self._robots)
        scene.add(self._objects)
        scene.add(self._goals)

        if self._dr_randomizer.randomize:
            self._dr_randomizer.apply_on_startup_domain_randomization(self)

    #####################################################################
    ###==================== Compute Observations =====================###
    #####################################################################

    def normalize_joint_positions(self, joint_positions: torch.Tensor) -> torch.Tensor:
        """Normalize joint positions to the range -1 to 1."""
        return (joint_positions - self._dof_lower_limits[self.actuated_dof_indices]) / (
            self._dof_upper_limits[self.actuated_dof_indices] - self._dof_lower_limits[self.actuated_dof_indices]
        ) * 2.0 - 1.0

    def denormalize_joint_positions(self, normalized_joint_positions: torch.Tensor) -> torch.Tensor:
        """Denormalize joint positions from the range -1 to 1."""
        return (normalized_joint_positions + 1.0) / 2.0 * (
            self._dof_upper_limits[self.actuated_dof_indices] - self._dof_lower_limits[self.actuated_dof_indices]
        ) + self._dof_lower_limits[self.actuated_dof_indices]

    def transform_to_palm_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        """Transform points from world coordinates to palm coordinates.

        Args:
            points (torch.Tensor): [num_envs, num_points, 3] or [num_envs, 3]

        Returns:
            torch.Tensor: [num_envs, num_points, 3] or [num_envs, 3] (same shape as input)
        """
        positions = self.compute_palm_positions()
        rotations = self.compute_palm_rotations()
        return transform_points(points, positions, rotations)

    def transform_to_object_coordinates(self, points: torch.Tensor) -> torch.Tensor:
        """Transform points from world coordinates to object coordinates.

        Args:
            points (torch.Tensor): [num_envs, num_points, 3] or [num_envs, 3]

        Returns:
            torch.Tensor: [num_envs, num_points, 3] or [num_envs, 3] (same shape as input)
        """
        positions = self.compute_object_positions()
        rotations = self.compute_object_rotations()
        return transform_points(points, positions, rotations)

    def compute_imagined_hand_pointclouds(self, num_points: Optional[int] = None) -> torch.Tensor:
        """Compute the imagined hand pointclouds.

        Returns:
            torch.Tensor: [num_envs, num_points, 3]
        """
        num_points = num_points or self.num_points
        total_points = sum([pcd.shape[0] for pcd in self._cached_pointclouds.values()])
        num_samples_per_mesh = {
            name: round(num_points * pcd.shape[0] / total_points) for name, pcd in self._cached_pointclouds.items()
        }
        num_samples_per_mesh["palm"] += num_points - sum(num_samples_per_mesh.values())

        positions, rotations = self._robots._meshes.get_world_poses(clone=False)
        positions = positions.reshape(self.num_envs, -1, 3)
        rotations = rotations.reshape(self.num_envs, -1, 4)
        positions -= self._env_pos[:, None, :]

        imagined_pointclouds = []
        for name in self._mesh_names:
            if name not in self._cached_pointclouds:
                continue
            pointcloud = self._cached_pointclouds[name]
            num_samples = num_samples_per_mesh[name]
            sampled_points = pointcloud[torch.randperm(pointcloud.shape[0])[:num_samples]]
            if sampled_points.size(0) == 0:
                continue

            sampled_points = sampled_points.reshape(1, -1, 3).repeat(self.num_envs, 1, 1).reshape(-1, 3)
            rotation = rotations[:, self._mesh_names.index(name)]
            rotation = rotation.reshape(-1, 1, 4).repeat(1, num_samples, 1).reshape(-1, 4)

            sampled_points = quat_rotate(rotation, sampled_points).reshape(self.num_envs, -1, 3)
            sampled_points += positions[:, self._mesh_names.index(name)][:, None, :]

            imagined_pointclouds.append(sampled_points)

        imagined_pointclouds = torch.cat(imagined_pointclouds, dim=1)
        return imagined_pointclouds

    def compute_imagined_pointclouds(
        self, num_points: Optional[int] = None, coordinate_system: str = "world"
    ) -> torch.Tensor:
        """Compute the imagined pointclouds (hand + object).

        Returns:
            torch.Tensor: [num_envs, num_points, 3]
        """
        assert coordinate_system in ["world", "palm"], "Invalid coordinate system"
        num_points = num_points or self.num_points
        hand_pointclouds = self.compute_imagined_hand_pointclouds(num_points // 2)
        object_pointclouds = self.compute_object_pointclouds(num_points // 2)
        pointclouds = torch.cat([hand_pointclouds, object_pointclouds], dim=1)

        if coordinate_system == "palm":
            pointclouds = self.transform_to_palm_coordinates(pointclouds)

        return pointclouds

    def compute_palm_positions(self) -> torch.Tensor:
        """Compute the positions of the palms.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        positions = self._robots._palms.get_world_poses(clone=False)[0]
        positions -= self._env_pos
        return positions

    def compute_palm_rotations(self) -> torch.Tensor:
        """Compute the rotations of the palms. (quaternions - wxyz)

        Returns:
            torch.Tensor: [num_envs, 4]
        """
        return self._robots._palms.get_world_poses(clone=False)[1]

    def compute_thumbtip_positions(self) -> torch.Tensor:
        """Compute the positions of the thumb tips.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        positions = self._robots._thumbs.get_world_poses(clone=False)[0]
        positions = positions.reshape(self.num_envs, 3)
        positions -= self._env_pos
        return positions

    def compute_thumbtip_contact_forces(self) -> torch.Tensor:
        """Compute the contact forces of the thumb tips.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        return self._robots._thumbs.get_net_contact_forces(clone=False)

    def compute_thumbtip_object_minimum_distances(self) -> torch.Tensor:
        """Compute the minimum distance between thumb tip and the object pointcloud.

        Returns:
            torch.Tensor: [num_envs]
        """
        object_pointcloud = self.compute_object_pointclouds()  # [num_envs, num_points, 3]
        fingertip_positions = self.compute_thumbtip_positions()  # [num_envs, 3]

        # Compute distances between fingertip positions and object pointcloud
        distances = torch.norm(object_pointcloud - fingertip_positions[:, None, :], dim=-1)

        # Find minimum distance for each fingertip and object
        min_distances = distances.min(dim=1)[0]  # [num_envs]

        return min_distances

    def compute_fingertip_positions(self) -> torch.Tensor:
        """Compute the positions of the fingertips.

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 3]
        """
        positions = self._robots._fingertips.get_world_poses(clone=False)[0]
        positions = positions.reshape(self.num_envs, self.num_fingertips, 3)
        positions -= self._env_pos.reshape(self.num_envs, 1, 3)
        return positions

    def compute_fingertip_rotations(self) -> torch.Tensor:
        """Compute the rotations of the fingertips.

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 4]
        """
        rotations = self._robots._fingertips.get_world_poses(clone=False)[1]
        rotations = rotations.reshape(self.num_envs, self.num_fingertips, 4)
        return rotations

    def compute_fingertip_velocities(self) -> torch.Tensor:
        """Compute the velocities of the fingertips. (linear and angular)

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 6]
        """
        velocities = self._robots._fingertips.get_velocities(clone=False)
        return velocities.reshape(self.num_envs, self.num_fingertips, 6)

    def compute_fingertip_linear_velocities(self) -> torch.Tensor:
        """Compute the linear velocities of the fingertips.

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 3]
        """
        velocities = self._robots._fingertips.get_linear_velocities(clone=False)
        return velocities.reshape(self.num_envs, self.num_fingertips, 3)

    def compute_fingertip_angular_velocities(self) -> torch.Tensor:
        """Compute the angular velocities of the fingertips.

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 3]
        """
        velocities = self._robots._fingertips.get_angular_velocities(clone=False)
        return velocities.reshape(self.num_envs, self.num_fingertips, 3)

    def compute_fingertip_contact_forces(self) -> torch.Tensor:
        """Compute the contact forces of the fingertips.

        Returns:
            torch.Tensor: [num_envs, num_fingertips, 3]
        """
        forces = self._robots._fingertips.get_net_contact_forces(clone=False)
        return forces.reshape(self.num_envs, self.num_fingertips, 3)

    def compute_fingertip_object_minimum_distances(self) -> torch.Tensor:
        """Compute the minimum distance between each fingertip and the object pointcloud.

        Returns:
            torch.Tensor: [num_envs, num_fingertips]
        """
        object_pointcloud = self.compute_object_pointclouds()  # [num_envs, num_points, 3]
        fingertip_positions = self.compute_fingertip_positions()  # [num_envs, num_fingertips, 3]

        # Compute distances between fingertip positions and object pointcloud
        distances = torch.norm(object_pointcloud[:, None, :, :] - fingertip_positions[:, :, None, :], dim=-1)

        # Find minimum distance for each fingertip and object
        min_distances = distances.min(dim=2)[0]  # [num_envs, num_fingertips]

        return min_distances

    def compute_fingertip_object_distances(self) -> torch.Tensor:
        """Compute the distance between each fingertip and the object.

        Returns:
            torch.Tensor: [num_envs, num_fingertips]
        """
        object_positions = self.compute_object_positions()  # [num_envs, 3]
        fingertip_positions = self.compute_fingertip_positions()  # [num_envs, num_fingertips, 3]

        # Compute distances between fingertip positions and object positions
        distances = torch.norm(object_positions[:, None, :] - fingertip_positions, dim=-1)  # [num_envs, num_fingertips]

        return distances

    def compute_joint_positions(self) -> torch.Tensor:
        """Compute the joint positions of the robot.

        Returns:
            torch.Tensor: [num_envs, num_dofs]
        """
        return self._robots.get_joint_positions(clone=False)

    def compute_joint_velocities(self) -> torch.Tensor:
        """Compute the joint velocities of the robot.

        Returns:
            torch.Tensor: [num_envs, num_dofs]
        """
        return self._robots.get_joint_velocities(clone=False)

    def compute_object_positions(self) -> torch.Tensor:
        """Compute the positions of the object.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        positions = self._objects.get_world_poses(clone=False)[0]
        positions -= self._env_pos
        return positions

    def compute_object_positions_wrt_palm(self) -> torch.Tensor:
        # TODO: check the correctness of this function
        palm_rotation = self.compute_palm_rotations()
        positions = self.compute_object_positions()
        positions = quat_apply(quat_conjugate(palm_rotation), positions)
        return positions

    def compute_object_rotations(self) -> torch.Tensor:
        """Compute the rotations of the object. (quaternions - wxyz)

        Returns:
            torch.Tensor: [num_envs, 4]
        """
        return self._objects.get_world_poses(clone=False)[1]

    def compute_object_velocities(self) -> torch.Tensor:
        """Compute the velocities of the object. (linear and angular)

        Returns:
            torch.Tensor: [num_envs, 6]
        """
        return self._objects.get_velocities(clone=False)

    def compute_object_linear_velocities(self) -> torch.Tensor:
        """Compute the linear velocities of the object.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        return self._objects.get_linear_velocities(clone=False)

    def compute_object_angular_velocities(self) -> torch.Tensor:
        """Compute the angular velocities of the object.

        Returns:
            torch.Tensor: [num_envs, 3]
        """
        return self._objects.get_angular_velocities(clone=False)

    def compute_object_pointclouds(self, num_points: Optional[int] = None) -> torch.Tensor:
        """Compute the pointcloud of the object.

        The object pointcloud should be loaded from the object's mesh file.

        Returns:
            torch.Tensor: [num_envs, num_points, 3]
        """
        num_points = num_points or self.num_points
        positions, rotations = self._objects.get_world_poses(clone=False)
        pointcloud = self._object_pointcloud.clone()
        pointcloud = pointcloud[torch.randperm(pointcloud.shape[0])[:num_points]]

        pointcloud = pointcloud.reshape(1, -1, 3).repeat(self.num_envs, 1, 1).reshape(-1, 3)
        rotations = rotations.reshape(-1, 1, 4).repeat(1, num_points, 1).reshape(-1, 4)

        pointcloud = quat_rotate(rotations, pointcloud).reshape(self.num_envs, -1, 3)
        pointcloud += positions.reshape(self.num_envs, 1, 3)
        pointcloud -= self._env_pos.reshape(self.num_envs, 1, 3)

        return pointcloud

    def compute_camera_pointclouds(self, num_points: Optional[int] = None) -> torch.Tensor:
        """Compute the pointcloud from multiple RGB-D cameras.

        Args:
            num_points (Optional[int], optional): Number of points to sample from the object pointcloud. Defaults to None.

        Returns:
            torch.Tensor: [num_envs, num_points, 3]
        """
        raise NotImplementedError

    def compute_goal_positions(self) -> torch.Tensor:
        positions = self._goals.get_world_poses(clone=False)[0]
        positions -= self._env_pos
        return positions

    def compute_goal_rotations(self) -> torch.Tensor:
        """Compute the rotations of the goal. (quaternions - wxyz)

        Returns:
            torch.Tensor: [num_envs, 4]
        """
        return self._goals.get_world_poses(clone=False)[1]

    #####################################################################
    ###==================== RL Pipeline Functions ====================###
    #####################################################################

    def compute_reach_reward(
        self, epsilon: float = 0.06, clip_min: float = 0.03, clip_max: float = 0.8
    ) -> torch.Tensor:
        distance = self.compute_fingertip_object_distances()
        torch.clamp(distance, clip_min, clip_max, out=distance)
        reward = (1 / (epsilon + distance)).sum(dim=1)
        return reward

    def compute_contact_reward(self, threshold: float = 0.03) -> torch.Tensor:
        # thumbtip_distance = self.compute_thumbtip_object_minimum_distances()
        # fingertip_distance = self.compute_fingertip_object_minimum_distances()

        # thumb_contact = (thumbtip_distance < threshold).to(torch.float32)
        # finger_contact = ((fingertip_distance < threshold).sum(dim=1) >= 2).to(torch.float32)
        # return thumb_contact * finger_contact

        distance = self.compute_fingertip_object_distances()
        thumbtip_contact_force = self.compute_thumbtip_contact_forces()
        fingertip_contact_force = self.compute_fingertip_contact_forces()

        distance = torch.mean(distance, dim=1) < threshold
        thumbtip_contact = torch.norm(thumbtip_contact_force, dim=1) > 0.01
        fingertip_contact = (torch.norm(fingertip_contact_force, dim=2) > 0.01).sum(dim=1) >= 2

        return (distance * thumbtip_contact * fingertip_contact).to(torch.float32)

    def compute_lift_reward(
        self, epsilon: float = 0.04, rotation_weight: float = 0.0, contact_reward: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        object_init_position = self._init_object_positions
        object_position = self.compute_object_positions()
        goal_position = self.compute_goal_positions()
        contact_reward = contact_reward if contact_reward is not None else self.compute_contact_reward()
        reward = 0.0

        lift = torch.clamp(object_position[:, 2] - self.table_height, 0.0, 0.2)
        reward += 10.0 * lift

        mask = lift > 0.02
        if mask.sum() == 0:
            return contact_reward * reward

        reward[mask] += 1.0

        distance = torch.norm(object_position - goal_position, dim=1)
        reward[mask] += 1.0 / (epsilon + distance[mask])

        mask = mask & (distance < 0.1)
        if mask.sum() == 0:
            return contact_reward * reward

        object_quat = self.compute_object_rotations()
        goal_quat = self.compute_goal_rotations()
        theta = torch.arccos(
            torch.clamp(torch.pow(torch.sum(object_quat * goal_quat, dim=1), 2) * 2 - 1, -1 + 1e-8, 1 - 1e-8)
        )
        reward[mask] += 4.0 / (0.4 + theta[mask]) * rotation_weight
        return contact_reward * reward

    def compute_completion_criteria(self, threshold: float = 0.05) -> torch.Tensor:
        object_position = self.compute_object_positions()
        goal_position = self.compute_goal_positions()
        distance = torch.norm(object_position - goal_position, dim=1)
        return distance < threshold

    def compute_dexpoint_reward(
        self,
        epsilon: float = 0.05,
        threshold: float = 0.05,
        weight_reach: float = 0.02,
        weight_contact: float = 1.0,
        weight_lift: float = 1.0,
        weight_penalty: float = 0.01,
        contact_threshold: float = 0.05,
        complete_bonus: float = 250.0,
    ) -> torch.Tensor:
        reach_reward = self.compute_reach_reward()
        contact_reward = self.compute_contact_reward()
        lift_reward = self.compute_lift_reward(contact_reward=contact_reward)
        completion_reward = self.compute_completion_criteria(threshold) * complete_bonus
        joint_velocities = self.compute_joint_velocities()
        action_penalty = torch.sum(torch.pow(torch.clamp(joint_velocities, -1.0, 1.0), 2), dim=1)

        return (
            weight_reach * reach_reward
            + weight_contact * contact_reward
            + weight_lift * lift_reward
            + completion_reward
            - weight_penalty * action_penalty
        )

    def compute_joint_targets(self, actions: torch.Tensor) -> torch.Tensor:
        actions = actions.clone().to(self.device)
        actions = torch.clamp(actions, -1.0, 1.0)
        actions = self.denormalize_joint_positions(actions)

        if self.use_relative_control:
            self._current_targets[:, self.actuated_dof_indices] = (
                self._prev_targets[:, self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * actions
            )

        else:
            self._current_targets[:, self.actuated_dof_indices] = scale(
                actions,
                self._dof_lower_limits[self.actuated_dof_indices],
                self._dof_upper_limits[self.actuated_dof_indices],
            )
            self._current_targets[:, self.actuated_dof_indices] = (
                self.act_moving_average * self._current_targets[:, self.actuated_dof_indices]
                + (1 - self.act_moving_average) * self._prev_targets[:, self.actuated_dof_indices]
            )

        self._current_targets[:, self.actuated_dof_indices] = tensor_clamp(
            self._current_targets[:, self.actuated_dof_indices],
            self._dof_lower_limits[self.actuated_dof_indices],
            self._dof_upper_limits[self.actuated_dof_indices],
        )

        self._prev_targets[:, self.actuated_dof_indices] = self._current_targets[:, self.actuated_dof_indices]

        return self._current_targets

    def post_reset(self):
        self.num_dofs = self._robots.num_dof
        self.actuated_dof_indices = self._robots.actuated_dof_indices

        # Allocate tensors for storing joint targets
        self._init_dof_positions = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        self._init_dof_velocities = torch.zeros(self.num_dofs, dtype=torch.float, device=self.device)
        # TODO: consider init object positions and orientations
        self._init_object_positions = torch.tensor(self.object_start_translation, dtype=torch.float, device=self.device)
        self._init_object_orientations = torch.tensor(
            self.object_start_orientation, dtype=torch.float, device=self.device
        )
        self._prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._current_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._actions = torch.zeros((self.num_envs, self.num_actions), dtype=torch.float, device=self.device)

        self._palm_object_distances = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)

        self._init_dof_positions[self._robots.arm_dof_indices] = self.arm_init_joint_positions

        # Get joint limits
        dof_limits = self._robots.get_dof_limits()
        self._dof_lower_limits, self._dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        print(self._dof_lower_limits, self._dof_upper_limits)
        print(self._dof_lower_limits.shape, self._dof_upper_limits.shape)

        # TODO: reset goal object
        pass

        self.reset_envs_by_indices()

        if self._dr_randomizer.randomize:
            self._dr_randomizer.set_up_domain_randomization(self)

        return
        # self.num_hand_dofs = self._hands.num_dof
        # self.actuated_dof_indices = self._hands.actuated_dof_indices

        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        # self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # dof_limits = self._hands.get_dof_limits()
        # self.hand_dof_lower_limits, self.hand_dof_upper_limits = torch.t(dof_limits[0].to(self.device))

        # self.hand_dof_default_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        # self.hand_dof_default_vel = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)

        # self.object_init_pos, self.object_init_rot = self._objects.get_world_poses()
        # self.object_init_pos -= self._env_pos
        self.object_init_velocities = torch.zeros_like(
            self._objects.get_velocities(), dtype=torch.float, device=self.device
        )

        self.goal_pos = self.object_init_pos.clone()
        self.goal_pos[:, 2] -= 0.04
        self.goal_rot = self.object_init_rot.clone()

        self.goal_init_pos = self.goal_pos.clone()
        self.goal_init_rot = self.goal_rot.clone()

        # randomize all envs
        # indices = torch.arange(self._num_envs, dtype=torch.int64, device=self._device)
        # self.reset_idx(indices)

        # if self._dr_randomizer.randomize:
        #     self._dr_randomizer.set_up_domain_randomization(self)

    def reset_envs_by_indices(self, indices: Optional[torch.Tensor] = None):
        if indices is None:
            indices = torch.arange(self.num_envs)

        indices = indices.to(dtype=torch.long).to(self.device)
        assert indices.min() >= 0 and indices.max() < self.num_envs, "Invalid indices"

        # Reset Object
        # TODO: random object orientation
        noise = torch.rand(indices.shape[0], 3, device=self.device)
        noise = noise * self.reset_position_noise
        object_positions = self._init_object_positions + self._env_pos[indices] + noise
        x_noise = uniform((indices.shape[0], 1), min_val=-0.2, max_val=0.2, device=self.device)
        object_positions[:, 0] += x_noise[:, 0]
        y_noise = uniform((indices.shape[0], 1), min_val=-0.3, max_val=0.3, device=self.device)
        object_positions[:, 1] += y_noise[:, 0]
        object_orientations = self._init_object_orientations[None, :].repeat(indices.shape[0], 1)
        object_velocities = torch.zeros(indices.shape[0], 6, device=self.device)

        self._objects.set_world_poses(object_positions, object_orientations, indices)
        self._objects.set_velocities(object_velocities, indices)

        # Reset Goal
        goal_positions = self._init_object_positions + self._env_pos[indices]
        goal_positions[:, 2] += 0.5
        goal_orientations = self._init_object_orientations[None, :].repeat(indices.shape[0], 1)
        goal_velocities = torch.zeros(indices.shape[0], 6, device=self.device)

        self._goals.set_world_poses(goal_positions, goal_orientations, indices)
        self._goals.set_velocities(goal_velocities, indices)

        # Add randomization to joint positions & velocities
        # TODO: reconsider the randomization & robot init pose
        delta_upper = self._dof_upper_limits - self._init_dof_positions
        delta_lower = self._dof_lower_limits - self._init_dof_positions
        noise = torch.rand(indices.shape[0], self.num_dofs, device=self.device)
        noise = noise * (delta_upper - delta_lower) + delta_lower
        joint_positions = self._init_dof_positions + noise * self.reset_dof_pos_noise / 10.0

        noise = torch.rand(indices.shape[0], self.num_dofs, device=self.device)
        joint_velocities = self._init_dof_velocities + noise * self.reset_dof_vel_noise

        # Set joint positions & velocities
        self._robots.set_joint_position_targets(joint_positions, indices)
        self._robots.set_joint_positions(joint_positions, indices)
        self._robots.set_joint_velocities(joint_velocities, indices)

        # Reset related buffers
        self._prev_targets[indices] = joint_positions
        self._current_targets[indices] = joint_positions

        # Reset other buffers
        distances = torch.norm(self.compute_palm_positions() - self.compute_object_positions(), dim=1)
        self._palm_object_distances[indices] = distances[indices]

        self.progress_buf[indices] = 0
        self.reset_buf[indices] = 0
        self.successes[indices] = 0

    def get_observations(self):
        observations = []

        if "imagined_hand_pointcloud" in self.observation_types:
            observations.append(self.compute_imagined_hand_pointclouds())
        if "imagined_pointcloud" in self.observation_types:
            observations.append(self.compute_imagined_pointclouds())
        if "object_pointcloud" in self.observation_types:
            observations.append(self.compute_object_pointclouds())
        if "camera_pointcloud" in self.observation_types:
            observations.append(self.compute_camera_pointclouds())

        if "fingertip_position" in self.observation_types:
            observations.append(self.compute_fingertip_positions())
        if "palm_position" in self.observation_types:
            observations.append(self.compute_palm_positions())
        if "palm_rotation" in self.observation_types:
            observations.append(self.compute_palm_rotations())
        if "object_position" in self.observation_types:
            observations.append(self.compute_object_positions())
        if "object_rotation" in self.observation_types:
            observations.append(self.compute_object_rotations())
        if "goal_position" in self.observation_types:
            observations.append(self.compute_goal_positions())
        if "goal_rotation" in self.observation_types:
            observations.append(self.compute_goal_rotations())
        if "fingertip_object_minimum_distance" in self.observation_types:
            observations.append(self.compute_fingertip_object_minimum_distances())
        if "joint_position" in self.observation_types:
            observations.append(self.compute_joint_positions())
        if "joint_velocity" in self.observation_types:
            observations.append(self.compute_joint_velocities())

        self.obs_buf[:] = torch.cat([observation.reshape(self.num_envs, -1) for observation in observations], dim=-1)

        return self.obs_buf

    def get_states(self):
        return self.states_buf

    def calculate_metrics(self):
        # distances = torch.norm(self.compute_palm_positions() - self.compute_object_positions(), dim=1)
        # self.rew_buf[:] = self._palm_object_distances - distances
        # self._palm_object_distances[:] = distances

        action_penalty = self.action_penalty_scale * torch.sum(self._actions**2, dim=1)
        distances = torch.mean(self.compute_fingertip_object_distances(), dim=1)
        # self.rew_buf[:] = torch.exp(-10.0 * distances) - action_penalty
        self.rew_buf[:] = self.compute_dexpoint_reward()

        object_positions = self.compute_object_positions()
        fails = torch.where(distances >= 0.8, 1, 0)
        fails = torch.where(object_positions[:, 2] < self.table_height, 1, fails)
        self.rew_buf -= self.fall_penalty * fails

        # successes = torch.where(distances < 0.10, 1, 0)
        # self.rew_buf += self.reach_goal_bonus * successes
        successes = self.compute_completion_criteria().to(torch.long)

        self.reset_buf = torch.where(fails == 1, 1, self.reset_buf)
        self.reset_buf = torch.where(successes == 1, 1, self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length, 1, self.reset_buf)

    def is_done(self):
        pass

    def pre_physics_step(self, actions: torch.Tensor):
        if not self._env._world.is_playing():
            return

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        # goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        reset_buf = self.reset_buf.clone()

        # if only goals need reset, then call set API
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        # self.reset_target_pose(goal_env_ids)
        # elif len(goal_env_ids) > 0:
        # self.reset_target_pose(goal_env_ids)
        if len(env_ids) > 0:
            self.reset_envs_by_indices(env_ids)

        self._actions = actions.clone().detach().to(self.device)

        targets = self.compute_joint_targets(actions)
        self._robots.set_joint_position_targets(
            targets[:, self.actuated_dof_indices],
            indices=None,
            joint_indices=self.actuated_dof_indices,
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


#####################################################################
###======================== JIT Functions ========================###
#####################################################################


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


def uniform(size, *, min_val=-1.0, max_val=1.0, dtype=None, device=None):
    return (max_val - min_val) * torch.rand(size, dtype=dtype, device=device) + min_val


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


def transform_points(points: torch.Tensor, positions: torch.Tensor, rotations: torch.Tensor):
    """Transform points by positions and rotations.

    transform T = [R, t]
    apply p' = T^{-1}  p

    Args:
        points: world points tensor of shape (batch_size, num_points, 3) or (batch_size, 3)
        positions: world2local translation tensor of shape (batch_size, 3)
        rotations: world2local rotation tensor of shape (batch_size, 4)
    """
    assert points.ndim == 3 or points.ndim == 2
    assert points.shape[-1] == 3

    ndim = points.ndim
    points = points.unsqueeze(-2) if ndim == 2 else points
    points = points - positions.unsqueeze(-2)
    num_envs, num_points, _ = points.shape

    points = points.reshape(-1, 3)
    rotations = rotations.reshape(-1, 1, 4).repeat(1, num_points, 1).reshape(-1, 4)
    points = quat_rotate_inverse(rotations, points)
    points = points.reshape(num_envs, num_points, 3)
    print(points[0])
    return points.squeeze(-2) if ndim == 2 else points


def random_points_on_hypersphere(n_points, n_dim, device):
    """Sample points uniformly from the surface of an n-dimensional hypersphere."""
    points = torch.randn(n_points, n_dim, device=device)
    points = points / torch.norm(points, dim=-1, keepdim=True)
    return points
