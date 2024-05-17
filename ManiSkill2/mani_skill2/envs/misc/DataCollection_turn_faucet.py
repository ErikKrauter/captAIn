import sapien.core as sapien
from mani_skill2.envs.misc.VAT_turn_faucet import VATTurnFaucetEnv
from mani_skill2.utils.registration import register_env
import numpy as np
from collections import OrderedDict
from mani_skill2.utils.trimesh_utils import get_actor_mesh, visualize_mesh, visualize_facets
from mani_skill2.utils.sapien_utils import vectorize_pose
from scipy.spatial.transform import Rotation
from mani_skill2.utils.common import np_random, random_choice

'''
This environment is used for Direct Data Collection. Direct Data Collection, means that we do not need to train
a RL agent to learn to solve the task of turning the faucet. Instead a dataset of successful interaction is collected
directly. This class contains all the logic for this direct data collection.

The class inherits from VATTurnFaucetEnv, to access the logic of sampling contact points and positioning the gripper

It is important to note, that despite the fact that we do not actually need an agent to interact with the environment
in order to collect the data set, we still have to define some dummy agent in order to select actions and receive
observations. The entire ManiSkill framework is built around the notion of an agent interacting with the environment.

1. during initialization, an initial faucet angle, target angle and contact point, gripper pose are selected
   this step relies on the logic already implemented in VATTurnFaucetEnv
        
2. The environment receives some random action, We dont care about it. Instead the faucet is rotated 'manually' by delta_phi
   delta_phi being abs(target_angle-init_angle)/num_waypoints

3. compute new gripper pose based on new contact point and contact normal coordinates
4. check if pose is feasible by solving inverse kinematics
5. actually move the tcp to the pose
'''

@register_env("DataCollection-TurnFaucet-v0", max_episode_steps=200, override=True)
class DataCollectionTurnFaucetEnv(VATTurnFaucetEnv):

    def __init__(self, **kwargs):

        super(DataCollectionTurnFaucetEnv, self).__init__(**kwargs)
        self.delta_angle = 0
        self.contact_normal_local = None

    # This is the interaction loop. It contains the entire logic.
    def step(self, action):
        truncated = False

        self.rotateFaucet()
        point, normal = None, None
        # point, normal = self.transformContactPoint()

        # 3. compute new gripper pose based on new contact point and contact normal coordinate
        pose_in_base_coordinates = self.constructGripperPose(point, normal)
        # Rotate the faucet back and see if the gripper can move it instead
        self.rotateFaucet(back=True)

        # construct action from the pose
        action = -1*np.ones(7)  # 7 because low level controller expects 7 dim
        action[:3] = pose_in_base_coordinates.p
        # the orientation of the gripper remains fixed throughout the interaction
        # this results in natural looking trajectories
        quat = self.initial_tcp_pose.q

        action[3:6] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_rotvec()

        # 4. check if pose is feasible by solving inverse kinematics
        if(self.agent.controller.controllers["arm"].compute_ik(pose_in_base_coordinates) is not None):
            # 5. actually move the tcp to the pose
            counter = 0
            tcp_pose_in_base = self.agent.robot.pose.inv().transform(self.tcp.pose)

            # then we set the arm's joints to the correct values using inverse kinematics
            while np.linalg.norm(tcp_pose_in_base.p - pose_in_base_coordinates.p) > 0.005:
                if counter > 10:
                    # we could not reach the position
                    break
                if self.agent.controller.controllers["arm"].get_IK_error():
                    print('IK error')
                    break

                control_mode = 'pd_base_pose'
                self.step_action(dict(action=action, control_mode=control_mode))

                tcp_pose_in_base = self.agent.robot.pose.inv().transform(self.tcp.pose)
                counter += 1
            # 6. add tcp pose to sequence
            #    this is already done in parent class because tcp pose is added to the info dict
        else:
            print('truncating')
            truncated = True

        self._elapsed_steps += 1

        obs = self.get_obs()
        info = self.get_info(obs=obs, action=action)  # this call evaluate() which populates the waypoint key
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = self.get_done(obs=obs, info=info)
        return obs, reward, terminated, truncated, info

    def compute_dense_reward(self, info, **kwargs):

        return 0

    @property
    def current_contactNormal(self):
        if self.contact_normal_local is None:
            return np.zeros(3)
        transform_matrix = self.target_link.pose.to_transformation_matrix()  # local to world
        temp = transform_matrix[:3, :3] @ self.contact_normal_local
        return temp / np.linalg.norm(temp)

    def _get_obs_extra(self) -> OrderedDict:

        obs = OrderedDict(task=self.target_angle - self.init_angle, elapsed_steps=self.elapsed_steps)
        # self.custom_print("outputting obs")
        # self.logger.info(f"{self.rank}: outputting obs")
        return obs

    def _get_obs_agent(self):
        obs = self.agent.get_proprioception()
        obs["base_pose"] = vectorize_pose(self.agent.robot.pose)
        obs.pop('controller', None)
        return obs

    def rotateFaucet(self, back=False):
        qpos = self.faucet.get_qpos()
        if back:
            qpos[self.target_joint_idx] = self.current_angle - self.delta_angle
        else:
            steps_left = max(self.num_waypoints - self._elapsed_steps, 1)
            self.delta_angle = (self.target_angle - self.current_angle) / steps_left
            qpos[self.target_joint_idx] = self.current_angle + self.delta_angle
        self.faucet.set_qpos(qpos)

    def transformContactPoint(self):
        steps_left = max(self.num_waypoints - self._elapsed_steps, 1)
        delta_phi = (self.target_angle - self.current_angle) / steps_left
        phi = self.current_angle + delta_phi  # self.delta_angle
        R_const = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        R_z_phi = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
        R = R_const @ R_z_phi
        contact_point = R @ self.contact_point_local + self.translation_vector
        contact_normal = R @ self.contact_normal_local
        contact_normal /= np.linalg.norm(contact_normal)
        return contact_point, contact_normal

    def constructGripperPose(self, contact_point=None, contact_normal=None):
        normal = self.current_contactNormal if contact_normal is None else contact_normal
        gripper_x = -normal if self.action_mode == 'push' else normal  # normal is in global coordinates
        gripper_x = gripper_x / np.linalg.norm(gripper_x)
        z = np.array([0, 0, -1])  # gripper z axis points in negative global z direction
        # it must be z cross x, so that y points in correct direction
        gripper_y = np.cross(z, gripper_x)  # y axis is orthogonal to x and z
        gripper_y = gripper_y / np.linalg.norm(gripper_y)
        # it must be x cross y, so that z points in correct direction
        gripper_z = np.cross(gripper_x, gripper_y)
        gripper_z = gripper_z / np.linalg.norm(gripper_z)

        T = np.eye(4)
        T[:3, :3] = np.stack([gripper_x, gripper_y, gripper_z], axis=1)  # gripper orientation in global coordinate system
        T[:3, 3] = self.current_contactPoint if contact_point is None else contact_point
        gripper_target_pose = sapien.Pose.from_transformation_matrix(T)
        # We need to specify the target pose in the robots base frame, else the IK solution is incorrect
        to_robot_base = self.agent.robot.pose.inv()
        gripper_target_pose = to_robot_base.transform(gripper_target_pose)
        return gripper_target_pose


    # It would be better to use the function from the parent class here to avoid unnecessary code duplication
    # However, we need the local coordinates of the normal at the contact point
    # this is the only addition/difference to the parent's implementation...
    def sampleContactPoint(self):

        normals = self.target_link_mesh.face_normals[self.target_link_mesh.nearest.on_surface(self.target_link_pcd)[2]]
        faucet_x_local = np.array([1, 0, 0])

        mesh_center = self.target_link_mesh.centroid
        max_z = max(self.target_link_pcd[:, 2])
        min_y = min(self.target_link_pcd[:, 1])
        z_threshold = 0.4 * max_z

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

            # I only want contact points that are at least 1/3 away from the 'stem' of the lever
            # i.e. I prefer points towards the end of the lever
            if point[2] < z_threshold:
                # we only want to add the point if its not at the bottom of the handle (this part is technically 'inside' the faucet)
                if (point[1] - min_y) < 0.02:#np.dot(normal, np.array([0, -1, 0])) > np.cos(np.deg2rad(45)):
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

        contact_idx = self._episode_rng.randint(min(len(self.front_points), len(self.back_points)))
        if self.action_mode == 'pull':
            contact_point = self.back_points[contact_idx]
            contact_normal = self.back_normals[contact_idx]
        elif self.action_mode == 'push':
            contact_point = self.front_points[contact_idx]
            contact_normal = self.front_normals[contact_idx]
        else:
            raise Exception('the current fauect angle configuration is neither a pulling nor a pushing task')

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

        self.contact_point_local = contact_point
        self.contact_normal_local = contact_normal
        self.translation_vector = target_to_world[:3, 3]
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

        # for debugging
        # visualize_mesh(self.target_link_mesh, self.target_link_pcd)
        # visualize_mesh(self.target_link_mesh, self.front_points, self.front_normals, self.back_points, self.back_normals)
        # visualize_mesh(self.target_link_mesh, np.tile(self.contact_point_local, (3, 1)), np.tile(self.contact_normal_local, (3, 1)))
        return True

    def _set_init_and_target_angle(self):
        super()._set_init_and_target_angle()
        self.delta_angle = (self.target_angle - self.init_angle) / self.num_waypoints


