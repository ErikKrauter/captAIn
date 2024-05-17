import time

import sapien.core as sapien
import trimesh
import trimesh.sample
from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.utils.registration import register_env
from mani_skill2.utils.sapien_utils import set_render_material
from mani_skill2.utils.common import flatten_state_dict
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
from collections import OrderedDict
from mani_skill2.utils.trimesh_utils import get_actor_mesh, visualize_mesh, visualize_facets
from mani_skill2.utils.sapien_utils import vectorize_pose
from mani_skill2.utils.geometry import transform_points
from scipy.spatial.distance import cdist
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_logger_name
import logging
import os
from scipy.spatial.transform import Rotation
import copy
from mani_skill2.utils.common import np_random, random_choice
from gymnasium import spaces


'''
Base class for the TurnFaucet environment used with VAT-Mart, captAIn, and also for data collection

This environment defines how the environment is initialized, and what variables are contained in the info dict.

The environment initialization is a crucial step.
1. The task is sampled, meaning the initial and target angle of the faucet handle are sampled
2. Based on the task a suitable contact point is sampled, as well as the contact normal, the non-contact point. 
The non-contact point is later used for data augmentation
3. Based on the contact point and normal a initial gripper orientation is computed
4. The gripper is moved to that position using inverse kinematics
5. Now that the gripper is correctly positioned and oriented, the actual episode begins, i.e. the agent receives
 observations and starts acting.

The step() function contains all logic of how the agent interacts with the environment and how the environment reacts
'''

@register_env("VAT-TurnFaucet-v0", max_episode_steps=200, override=True)
class VATTurnFaucetEnv(TurnFaucetEnv):

    def __init__(self, mode='rl',
                 min_task_angle_difference=30,
                 max_task_angle_difference=180,
                 restrict_action_space=False,
                 randomize_initial_faucet_pose=False,
                 contact_point_offset=0.05,
                 num_waypoints=8,
                 control_mode='pd_ee_target_delta_pose',
                 **kwargs):

        self.success_threshold = np.deg2rad(5)
        self.min_task_angle_difference = np.deg2rad(min_task_angle_difference)
        self.max_task_angle_difference = np.deg2rad(max_task_angle_difference)
        self.gripper_finger_distance_threshold = 0.3  # I believe this is in meters
        self.rank = get_world_rank()
        self.waypoints = []
        self.mode = mode
        self.restrict_action_space = restrict_action_space
        self.randomize_initial_faucet_pose = randomize_initial_faucet_pose
        self.num_waypoints = num_waypoints  # the time limit is automatically set to be the same value
        self.control_mode_ = control_mode
        self.contact_point = np.zeros(3)
        self.contact_point_local = None
        self.non_contact_point_local = None
        self.non_contact_point = np.zeros(3)
        self.ee_pose_at_base = np.zeros(3)
        self.ee_pose = np.zeros(3)
        self.on_target_link = -1
        self.gripper_x = np.zeros(3)
        self.gripper_y = np.zeros(3)
        self.gripper_z = np.zeros(3)
        self.contact_normal = np.zeros(3)
        self.initial_tcp_pose = np.zeros(3)
        self.pid = os.getpid()
        self.finger_distance_to_contact_point = -1
        self.error = False
        self.contact_point_offset = contact_point_offset

        # a work around to ensure that the action space has the desired dimension
        # we want either a 4 dimensional action space, i.e. control x,y,z, yaw of the end effector
        # or a 6 dimensional action space, i.e. control full pose of the end effector
        # the original action space is 7 dimensional because it also controls the gripper
        # we do not care about the gripper, because we dont perform grasps.

        # during construction of the agent in the run_rl.py script (maniskill2_learn/apis/run_rl.py)
        # the action space is sampled to determine the dimension of the actions. That information is used to
        # construct the neural networks.

        if self.restrict_action_space:
            print('action space is restricted to 4 dim')
            self.action_space = spaces.Box(low=-np.ones(4), high=np.ones(4), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=-np.ones(6), high=np.ones(6), dtype=np.float32)

        super(VATTurnFaucetEnv, self).__init__(control_mode=control_mode, **kwargs)

        if self.restrict_action_space:
            self.action_space = spaces.Box(self.action_space.low[:4], self.action_space.high[:4], dtype=np.float32)
        else:
            self.action_space = spaces.Box(self.action_space.low[:6], self.action_space.high[:6], dtype=np.float32)

    # defines the success criterion
    def evaluate(self, **kwargs):
        angle_dist = self.target_angle - self.current_angle
        # success either if we are close to target angle, or if angle diff has changed signs meaning we overshot
        success = abs(angle_dist) < self.success_threshold or np.sign(angle_dist) != np.sign(self.target_angle_diff)
        out_dict = dict(success=success, angle_dist=angle_dist)
        out_dict.update(self.additional_info)
        self.error = False
        if self.agent.controller.controllers["arm"].get_IK_error():
            self.error = True
        out_dict['error'] = self.error
        return out_dict

    def populate_waypoints(self, action):
        if not self.restrict_action_space:
            # the first three dimenations are the deltax, deltay, deltaz
            # the last three dimenstions form a unit vector showing the rotation axis scaled by the amount to rotate by
            # the delat pos and the rotation axis are represented in end-effector frame / last target frame!
            waypoint = action[:-1]
        else:
            # the first three dimenations are the deltax, deltay, deltaz
            # we restrict the rotation to only the yaw
            pos = action[:3]
            yaw = action[5]
            waypoint = np.hstack([pos, yaw])

        self.waypoints[(self.elapsed_steps - 1)] = waypoint

    def expand_dim(self, action):
        # the RL agent predicts a 4 dim action
        # the low level controller expects a 7 dim action
        if action.shape[0] == 4:
            pos = action[:3]
            yaw = action[-1]
            axis_angle = np.array([0, 0, yaw])
            action = np.hstack([pos, axis_angle])

        gripper = -1
        action = np.hstack([action, gripper])

        return action

    # the step function defines the main interaction logic
    # maniskill and maniskill-learn allow actions to be either arrays or dictionaries
    def step(self, action):
        if isinstance(action, dict):
            # if action is a dictionary the agent is VAT-Mart
            # the action dict, also contains the predicted normal at the contact point
            if action['action'].shape[0] != 7:
                action['action'] = self.expand_dim(action['action'])
        else:
            # RL inference/training
            if action.shape[0] != 7:
                action = self.expand_dim(action)

        # if it is the very first interaction of the episode, the action given by VAT-Mart defines the first gripper
        # pose relative to the contact point. This case needs special treatment
        # first the contact point and normal are extracted from the action, then the gripper pose is constructed
        # then the gripper is position at that pose. It is important to note that the pose is absolute in Base
        # coordinates. All other actions afterward are always relative to the previous pose.
        if self._elapsed_steps == 0 and self.mode == 'vat':

            action = self.extractAxesAndContactPoint(action)

            target_pos, target_rot = action['action'][0:3], action['action'][3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose_base = sapien.Pose(target_pos, target_quat)
            # important to set the control mode. This control mode tells the low-level controller to treat the pose
            # as absolute, instead of relative to the last pose.
            action['control_mode'] = 'pd_base_pose'
            # here the action is passed to the low-level controller
            self.step_action(action)
            counter = 0
            # the action is repeated until the target is reached.
            while np.linalg.norm(self.agent.robot.pose.inv().transform(self.tcp.pose).p - target_pose_base.p) > 0.005:
                if counter > 10:
                    break
                self.step_action(action)
                counter += 1
        else:
            # For all subsequent actions, the actions are relative to the last action
            # no further treatment is needed
            if isinstance(action, dict):
                action['control_mode'] = self.control_mode_
            self.step_action(action)

        # the action must be a numpy array for all subsequent computations
        if isinstance(action, dict):
            action = action["action"]

        self._elapsed_steps += 1

        # save tha action into the array of waypoints
        # during data collection that array contains the actions data by the rl agent
        # later that waypoint array is used to train the Trajectory Generator
        # during VAT-Mart inference, the waypoints array is still populated for visualization purposes
        self.populate_waypoints(action)

        obs = self.get_obs()
        info = self.get_info(obs=obs, action=action)  # this calls evaluate()
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = self.get_done(obs=obs, info=info)
        return obs, reward, terminated, False, info

    # executed right before the action is passed to the robot joints
    def _before_control_step(self):
        self.ee_pose_at_base = vectorize_pose(self.agent.robot.pose.inv().transform(self.tcp.pose))

    # we reset the environment. Important to empty the waypoints list again.
    def reset(self, seed=None, options=None):
        out, _ = super().reset(seed=seed, options=options)
        if not self.restrict_action_space:
            self.waypoints = [[0] * (len(self.agent.action_space.sample())-1) for _ in range(self.num_waypoints)]
        else:
            self.waypoints = [[0] * (len(self.action_space.sample())) for _ in range(self.num_waypoints)]
        return out, {}

    # the additional info will not be part of the observation that the agent receives.
    # It will be returned in the info dictionary.
    # During data collection the info dictionary is filled with information that we need in the dataset for later
    # training of the perception modules.
    # Other than that the info dict contains information used for visualization/analysis purposes
    @property
    def additional_info(self):
        out_dict = dict()
        out_dict['init_contact_point_world'] = self.contact_point
        out_dict['init_contact_point_base'] = self.world_to_base_frame(self.contact_point)
        if self.contact_point_local is not None:
            out_dict['init_contact_point_local'] = self.contact_point_local
        out_dict['non_contact_point_world'] = self.non_contact_point
        out_dict['non_contact_point_base'] = self.world_to_base_frame(self.non_contact_point)
        # the local coordinates of the non-contact point are only needed for visualization of the data set
        # of the affordance predictor. During inference the variable is none and should not be included into the
        # observation, because it leads to issue when saving the trajectory into an h5 file.
        if self.non_contact_point_local is not None:
            out_dict['non_contact_point_local'] = self.non_contact_point_local
        # whether the non-contact point is on the faucet handle or not. Needed for data augmentation and visualization
        out_dict['non_contact_on_movable_part'] = self.on_target_link
        out_dict['contact_normal_world'] = self.contact_normal
        out_dict['gripper_forward_dir'] = self.gripper_x  # world and base coordinates are the same here, because base frame is not rotated
        out_dict['gripper_up_dir'] = self.gripper_z  # world and base coordinates are the same here, because base frame is not rotated
        out_dict['actual_target_motion'] = self.current_angle - self.init_angle  # signed, because we want to also provide direction (negative sign is clockwise)
        out_dict['target_angle'] = self.target_angle
        out_dict['current_angle'] = self.current_angle
        out_dict['init_angle'] = self.init_angle
        out_dict['action_mode'] = 1 if self.action_mode=='pull' else 0  # cannot use strings, the repay buffer complains
        out_dict['ee_pose_at_base'] = self.ee_pose_at_base  # this is tcp pose before action
        out_dict['base_pose'] = vectorize_pose(self.agent.robot.pose)
        out_dict['tcp_pose'] = vectorize_pose(self.tcp.pose) # this is tcp pose after action
        out_dict['target_link_pose'] = vectorize_pose(self.target_link.pose)
        out_dict['waypoints'] = np.hstack(self.waypoints)  # the waypoints are in end effector coordinates space!
        out_dict['model_id'] = int(self.model_id)

        # The pointcloud is not added at this point in time. The pointcloud is added in the VATInfoWrapper
        # First the pointcloud is constructed from the camera images and added to the observations
        # the VATInfoWrapper removes the pointcloud from the observation and appends it to the info dict
        # out_dict['pointcloud']
        return out_dict

    def _compute_distance(self):
        cp = self.current_contactPoint
        if cp is None:
            return 0
        cps = np.tile(cp, (2,1))
        """Compute the distance between the tap and robot fingers."""
        T1 = self.lfinger.pose.to_transformation_matrix()
        T2 = self.rfinger.pose.to_transformation_matrix()
        pcd1 = transform_points(T1, self.lfinger_pcd)
        pcd2 = transform_points(T2, self.rfinger_pcd)
        # trimesh.PointCloud(np.vstack([pcd, pcd1, pcd2])).show()
        distance1 = cdist(cps, pcd1)
        distance2 = cdist(cps, pcd2)
        self.finger_distance_to_contact_point = min(distance1.min(), distance2.min())

        return self.finger_distance_to_contact_point

    def compute_dense_reward(self, info, **kwargs):

        if info["success"]:
            # compute distance just to update the finger_distance_to_contact_point value
            return 10.0

        reward = 0.0
        distance = self._compute_distance()
        reward += 1 - np.tanh(distance * 5.0)

        if distance > self.gripper_finger_distance_threshold:
            # print("Fingers far away")
            reward -= 10

        # is_contacted = any(self.agent.check_contact_fingers(self.target_link))
        # if is_contacted:
        #     reward += 0.25

        angle_diff = abs(self.target_angle - self.current_angle)
        turn_reward_1 = 3 * (1 - np.tanh(angle_diff * 2.0))
        reward += turn_reward_1

        delta_angle = self.last_angle_diff - angle_diff
        turn_reward_2 = np.tanh(delta_angle * 2)

        turn_reward_2 *= 5
        reward += turn_reward_2

        self.last_angle_diff = angle_diff

        # have to update the finger distance and the error flag here instead of in the observation, because else
        # the values are not up to date.

        return reward

    @property
    def current_angle(self):
        return self.faucet.get_qpos()[self.target_joint_idx]

    @property
    def current_contactPoint(self):
        # if the contact point is None than it has not been populated yet
        if self.contact_point_local is None:
            return None

        transform_matrix = self.target_link.pose.to_transformation_matrix()  # local to world
        return transform_matrix[:3, :3] @ self.contact_point_local + transform_matrix[:3, 3]

    def world_to_base_frame(self, point, dir=False):

        if point is None:
            return np.zeros(3)

        to_robot_base_transformation_matrix = self.agent.robot.pose.inv().to_transformation_matrix()
        if dir:
            ret = to_robot_base_transformation_matrix[:3, :3] @ point
        else:
            ret = to_robot_base_transformation_matrix[:3, :3] @ point + to_robot_base_transformation_matrix[:3, 3]
        return ret

    def base_to_world_frame(self, point, dir=False):

        if point is None:
            return np.zeros(3)

        to_world_transformation_matrix = self.agent.robot.pose.to_transformation_matrix()
        if dir:
            ret = to_world_transformation_matrix[:3, :3] @ point
        else:
            ret = to_world_transformation_matrix[:3, :3] @ point + to_world_transformation_matrix[:3, 3]
        return ret

    # This is part of the observation that the agent receives from the environment
    def _get_obs_extra(self) -> OrderedDict:
        # during data collection agent the following values are passed as observations
        if self.mode == 'rl':
            obs = OrderedDict(
                target_link_qpos=self.current_angle,
                dist_to_target=self.target_angle - self.current_angle,
                target_angle_diff=self.target_angle - self.init_angle,
                task=self.target_angle,
                gripper_pos=self.world_to_base_frame(self.initial_tcp_pose.p),
                base_contact_point=self.world_to_base_frame(self.current_contactPoint),
                l_finger_pos=self.world_to_base_frame(self.lfinger.pose.p),
                r_finger_pos=self.world_to_base_frame(self.rfinger.pose.p),
                target_joint_axis=self.world_to_base_frame(self.target_joint_axis, dir=True),
                up=self.world_to_base_frame(self.gripper_z, dir=True),
                forward=self.world_to_base_frame(self.gripper_x, dir=True),
                left=self.world_to_base_frame(self.gripper_y, dir=True),
                joint_origin=self.world_to_base_frame(np.array([self.joint_origin[0], self.joint_origin[1], self.contact_point[2]])),
                # this is needed for the ManiSkill2-Learn Observation wrapper for coordinate transformations into ee frame
                tcp_pose=vectorize_pose(self.tcp.pose),
                # this is needed to tell the robot what faucet to manipulate if we have two handles
                target_link_pos=self.world_to_base_frame(self.target_link_pos),
                # I normalize the model id so that the numbers are smaller!
                # else it leads to really bad training performance
                model_id=(int(self.model_id)-5000)/100,
            )
        # during inference with VAT-Mart we do not use any privileged states in the obseravtions
        elif self.mode == 'vat':
            # ELAPSED_STEPS MUST BE THE LAST ITEM IN THE DICTIONARY!
            # TASK MUST BE THE SECOND LAST ITEM IN THE DICTIONARY!
            obs = OrderedDict(task=self.target_angle - self.init_angle, elapsed_steps=self.elapsed_steps)
        else:
            raise NotImplementedError

        return obs

    # we use Trimesh functionality to accomplish the task of sampling a contact point and its normal
    # some of the faucets handles are hollow, i.e. they have an "inside"
    # to make sure that we only sample points on the outside of the handle we make use of the centroid
    def sampleContactPoint(self):
        # self.custom_print("sampling contact point")
        # Calculate normals for the sampled points

        # for most faucets y is rotation axis, and z shows in handle direction
        normals = self.target_link_mesh.face_normals[self.target_link_mesh.nearest.on_surface(self.target_link_pcd)[2]]
        faucet_x_local = np.array([1, 0, 0])

        mesh_center = self.target_link_mesh.centroid
        max_z = max(self.target_link_pcd[:, 2])
        min_y = min(self.target_link_pcd[:, 1])
        z_threshold = 0.4 * max_z  # the larger the threshold the further the contact points will be to the tip of handle

        # this makes sure that the contact point is far enough from the axis of rotation, to avoid collisions
        # between the gripper and the faucet
        if self.model_id == '5005':
            z_threshold = 0.65 * max_z
        elif self.model_id == '5052':
            z_threshold = 0.3 * max_z
        elif self.model_id == '5023':
            z_threshold = 0.5 * max_z
        elif self.model_id == '5034':
            z_threshold = 0.2 * max_z
        elif self.model_id == '5028':
            z_threshold = 0.6 * max_z
        elif self.model_id == '5018':
            z_threshold = 0.7 * max_z
        elif self.model_id == '5053':
            z_threshold = 0.55 * max_z

        # Filter points based on normals
        # Define the normal threshold to decide front or back
        normal_threshold = 0.85  # the larger this value the more the normal points perpendicular to handle
        front_points = []
        back_points = []
        back_normals = []
        front_normals = []

        unfavorable_contact_points = []

        for point, normal in zip(self.target_link_pcd, normals):

            if point[2] < z_threshold:
                # we only want to add the point if its not at the bottom of the handle (this part is technically 'inside' the faucet)
                if (point[1] - min_y) < 0.02:  # np.dot(normal, np.array([0, -1, 0])) > np.cos(np.deg2rad(45)):
                    pass
                else:
                    unfavorable_contact_points.append(point)
                continue

            # get vector pointing from point to centroid
            center_vector = mesh_center - point
            center_vector /= np.linalg.norm(center_vector)  # Normalize the vector

            # if the x direction of the center vector and the normal vector point in the same direction
            # the point is located inside the handle
            if normal[0] * center_vector[0] > 0:
                # the current point is on the inside of the handle. We need to skip this point
                #print('skipping')
                # instead of skipping we simply invert the normal. This also works.
                normal = -1 * normal

            # which face of the lever is facing the robot depends on the rotation angle of the lever itself
            # if angle negative, that means lever is rotated clockwise
            if self.init_angle < 0:
                # if lever's x axis points in direction of contact point's normal, the cp is on front face
                if np.dot(faucet_x_local, normal) > normal_threshold:  # normal[0] > normal_threshold:
                    front_points.append(point)
                    front_normals.append(normal)
                # if normals points in opposite direction of x axis, then cp is on back face
                elif np.dot(faucet_x_local, normal) < -normal_threshold:  # normal[0] < -normal_threshold:
                    back_points.append(point)
                    back_normals.append(normal)
                else:
                    unfavorable_contact_points.append(point)
            # if the lever is rotated in other direction, the rules reverse
            elif self.init_angle >= 0:
                if np.dot(faucet_x_local, normal) > normal_threshold:
                    back_points.append(point)
                    back_normals.append(normal)
                elif np.dot(faucet_x_local, normal) < -normal_threshold:
                    front_points.append(point)
                    front_normals.append(normal)
                else:
                    unfavorable_contact_points.append(point)

        # Now, front_points and back_points contain points on the front and back of the lever
        self.front_points = np.array(front_points)
        self.back_points = np.array(back_points)
        self.back_normals = np.array(back_normals)
        self.front_normals = np.array(front_normals)

        if min(len(self.front_points), len(self.back_points)) == 0:
            return False

        # Choose a random point from front or back for the contact point
        # if mode is pulling, we need a point from the back side
        # if mode is pushing we need a point from the front side
        contact_idx = self._episode_rng.randint(min(len(self.front_points), len(self.back_points)))
        if self.action_mode == 'pull':
            contact_point = self.back_points[contact_idx]
            contact_normal = self.back_normals[contact_idx]
        elif self.action_mode == 'push':
            contact_point = self.front_points[contact_idx]
            contact_normal = self.front_normals[contact_idx]
        else:
            contact_point = None
            contact_normal = None

        # this is needed to train the affordance predictor on negative samples
        # statid_idx = random_choice(np.arange(len(self.static_link_pcd)), self._episode_rng)
        # non_contact_point = self.static_link_pcd[statid_idx]

        faucet_to_world = self.static_link.pose.to_transformation_matrix()
        target_to_world = self.target_link.pose.to_transformation_matrix()

        non_contacts_static = []
        non_contacts_movable = []

        for p in self.static_link_pcd:
            p_world = faucet_to_world[:3, :3] @ p + faucet_to_world[:3, 3]
            non_contacts_static.append(p_world)

        for p in unfavorable_contact_points:
            p_world = target_to_world[:3, :3] @ p + target_to_world[:3, 3]
            non_contacts_movable.append(p_world)

        non_contacts_static = np.array(non_contacts_static)
        non_contacts_movable = np.array(non_contacts_movable)

        # the contact points and contact normals are in the target links local coordinates
        # we need them to be in global coordinates for all further calculations
        # thus we take the target links transformation matrix to transform the point and normal to global frame
        # the transform_matrix below equates to:
        # R_const = [[0,0,-1],[-1,0,0],[0,1,0]], R_z_phi = [[cos(phi),0,sin(phi)],[0,1,0],[-sin(phi),0,cos(phi)]]
        # transform_matrix[:3,:3] = R_const @ R_z_phi
        self.contact_point_local = contact_point
        self.contact_point = target_to_world[:3, :3] @ contact_point + target_to_world[:3, 3]
        self.contact_normal = target_to_world[:3, :3] @ contact_normal

        if self._episode_rng.uniform() > 0.7:
            # 30% chance that non-contact point is on movable part of the faucet
            if len(non_contacts_movable) == 0:
                return False
            self.on_target_link = 1
            static_idx = random_choice(np.arange(len(non_contacts_movable)), self._episode_rng)
            self.non_contact_point = non_contacts_movable[static_idx]
            self.non_contact_point_local = unfavorable_contact_points[static_idx]
        else:
            # 70% chance that non-contact point is on static part of the faucet
            if len(non_contacts_static) == 0:
                return False
            self.on_target_link = 0
            static_idx = random_choice(np.arange(len(non_contacts_static)), self._episode_rng)
            self.non_contact_point = non_contacts_static[static_idx]
            self.non_contact_point_local = self.static_link_pcd[static_idx]

        return True

        # for debugging / visualization. This is super helpful to understand what the mesh and sampled points looks like
        # visualize_mesh(self.target_link_mesh, self.target_link_pcd)
        # visualize_mesh(self.target_link_mesh, self.front_points, self.front_normals, self.back_points, self.back_normals)
        # visualize_facets(self.target_link_mesh, self.target_link_pcd, self.front_points, self.back_points)

    # first sample a contact point on the faucets handle
    # then construct the grippers pose based on the contact point, the normal of the contact point
    # and the z direction of the gripper
    # note: all vectors and points must be in global frame, i.e. world coordinates
    def constructGripperCoordinateSystem(self, reverse=False):
        # up is x
        # forward is z
        # right is y
        # I want the normal direction of the contact point to align with the x direction of the gripper
        # the z direction of the gripper should point opposite to the global z direction

        # This while loop makes sure that we keep on sampling contact points until we find a suitable one
        # for some faucet models the suitable area of interaction is very small, wherefore several attempts might
        # be needed
        success = False
        counter = 0
        while not success:
            success = self.sampleContactPoint()
            if counter > 10:
                raise Exception(f'sampling a contact point for faucet {self.model_id} failed after 10 attempts')
            if not success:
                # we resample the points on the faucet mesh which are used during contact point sampling
                self.target_link_pcd = trimesh.sample.sample_surface(
                    self.target_link_mesh, 1024, seed=self._episode_seed+counter
                )[0]
                self.static_link_pcd = trimesh.sample.sample_surface(
                    self.static_link_mesh, 256, seed=self._episode_seed+counter
                )[0]

            counter += 1

        # normal is in global coordinates
        self.gripper_x = -self.contact_normal if self.action_mode=='push' else self.contact_normal
        self.gripper_x = self.gripper_x / np.linalg.norm(self.gripper_x)
        z = np.array([0, 0, -1])  # gripper z axis points in negative global z direction
        while abs(np.dot(self.gripper_x, z)) > 0.99:  # just to make sure that z is sufficiently perpendicular
            z = np.random.randn(3).astype(np.float32)
            z = z / np.linalg.norm(z)
        # it must be z cross x, so that y points in correct direction
        self.gripper_y = np.cross(z, self.gripper_x)  # y axis is orthogonal to x and z
        self.gripper_y = self.gripper_y / np.linalg.norm(self.gripper_y)
        # it must be x cross y, so that z points in correct direction
        self.gripper_z = np.cross(self.gripper_x, self.gripper_y)
        self.gripper_z = self.gripper_z / np.linalg.norm(self.gripper_z)

        T = np.eye(4)
        # gripper orientation in global coordinate system
        T[:3, :3] = np.stack([self.gripper_x, self.gripper_y, self.gripper_z], axis=1)
        # contact point is in global coordinates
        T[:3, 3] = self.contact_point + self.contact_point_offset * self.contact_normal

        return T

    def extractAxesAndContactPoint(self, action):

        init_action = copy.deepcopy(action['action'])
        contact_point_base = copy.deepcopy(init_action[:3])
        self.contact_point = self.base_to_world_frame(contact_point_base)

        axis_angle_base = copy.deepcopy(init_action[3:6])  # the axis is in world coordinate system
        r = Rotation.from_rotvec(axis_angle_base).as_matrix()

        self.gripper_x = r[:, 0]
        self.gripper_y = r[:, 1]
        self.gripper_z = r[:, 2]

        self.contact_normal = action['normal'][0]

        init_action[:3] += self.contact_point_offset * self.contact_normal
        action['action'] = init_action
        return action

    # this is done to change the order of initialization
    def _initialize_task(self):
        # print('PASSING INITIALIZATION OF TASK')
        pass

    def _initialize_training(self):
        T = self.constructGripperCoordinateSystem()  # outputs transformation matrix in world coordinates
        gripper_target_pose = sapien.Pose.from_transformation_matrix(T)

        # We need to specify the target pose in the robots base frame, else the IK solution is incorrect
        to_robot_base = self.agent.robot.pose.inv()
        gripper_target_pose = to_robot_base.transform(gripper_target_pose)

        # then we set the arm's joints to the correct values using inverse kinematics
        qpos = self.agent.controller.controllers["arm"].compute_ik(gripper_target_pose)

        if qpos is None:
            self.custom_print("IK RESULTS IN NONE")
            qpos = np.zeros((7,))

        # this directly sets the gripper pose. We do not use an iterative approach here.
        self.agent.reset(np.hstack([qpos, 0, 0]))  # the last two values are for the gripper fingers
        self.initial_tcp_pose = self.tcp.pose

    # here we compute the initial pose of the robot and all of its joints.
    # I need to compute the contact point and use that to construct the coordinate system for the gripper
    # then use that coordinate system to position the gripper and use IK to compute alle joint angles
    def _initialize_agent(self):

        # qpos_neutral = np.array([0, -0.785, 0, -2.356, 0, 1.57, 0.785])
        # FOR SOME REASON IT IS NECESSARY TO FIRST SET THE ROBOT TO THIS NEUTRAL POSE BEFORE DOING INVERSE KINEMATICS
        super()._initialize_agent()
        # we first need to initialize the task
        # this means we sample an initial and a target angle for the faucet
        # based on that we define wheter its a "pulling" or "pushing" task
        super()._initialize_task()
        # based on that we sample a suitable contact point on the faucets handle
        # and construct the gripper coordinate system

        # during training of the data collection agent, the gripper is position at the sampled contact point
        if self.mode == 'rl':
            self._initialize_training()
        # during inference with VAT-Mart the initial gripper pose is predicted by the agent itself, so we do NOT
        # sample a contact point and do NOT position the gripper based on that privileged information
        # instead the agent decides itself what contact point and initial orientation to use
        elif self.mode == 'vat':
            pass
        else:
            raise NotImplementedError

    # this function is part of the _initialize_task() function.
    # it samples an initial and a target angle for the faucet
    # based on that it defines whether the robot shout push or pull the handle
    def _set_init_and_target_angle(self):

        qmin, qmax = self.target_joint.get_limits()[0]
        # self.custom_print(f"qmin: {qmin}, qmax: {qmax}")

        if np.isinf(qmin):
            qmin = -np.pi / 2
        if np.isinf(qmax):
            qmax = np.pi / 2

        mid_point = (qmin + qmax) / 2
        self.init_angle, self.target_angle = 0, 0
        while (self.min_task_angle_difference > abs(self.init_angle - self.target_angle) or
               abs(self.init_angle - self.target_angle) > self.max_task_angle_difference):
            random_init = self._episode_rng.uniform()
            self.init_angle = qmin + (qmax - qmin) * random_init
            random_target = self._episode_rng.uniform()
            self.target_angle = qmin + (qmax - qmin) * random_target

        if self.target_angle < self.init_angle and self.init_angle <= mid_point:  # clockwise
            mode = 'push'
        elif self.init_angle < self.target_angle and self.init_angle > mid_point:  # anti clockwise
            mode = 'push'
        elif self.init_angle < self.target_angle and self.init_angle <= mid_point:  # anti clockwise
            mode = 'pull'
        elif self.target_angle < self.init_angle and self.init_angle > mid_point:  # clockwise
            mode = 'pull'
        self.action_mode = mode
        # self.custom_print(f'initial_angle: {self.init_angle}, target_angle: {self.target_angle}, mode: {mode}')

        # The angle to go
        self.target_angle_diff = self.target_angle - self.init_angle

    # this function is called to init the pose of the faucet.
    # some randomness can be introduced.
    def _initialize_articulations(self):
        p = np.zeros(3)

        if self.randomize_initial_faucet_pose:
            p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
            # p[:2] = self._episode_rng.uniform(-0.1, 0.1, [2])
            p[2] = self.model_offset[2]
            # ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
            ori = self._episode_rng.uniform(-np.deg2rad(15), np.deg2rad(15))
            q = euler2quat(0, 0, ori)
            self.faucet.set_pose(Pose(p, q))
            # print(f'angle: \t {np.rad2deg(ori)} \n offset: \t {np.linalg.norm(p[:2])}')
        else:
            p[2] = self.model_offset[2]
            q = euler2quat(0, 0, 0)
            self.faucet.set_pose(Pose(p, q))



