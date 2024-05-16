import sapien.core as sapien
from typing import Dict, Optional, Sequence, Union
from mani_skill2.envs.misc.VAT_turn_faucet import VATTurnFaucetEnv
from mani_skill2.utils.registration import register_env
import numpy as np
from collections import OrderedDict
from scipy.spatial.transform import Rotation
import copy
from mani_skill2.utils.sapien_utils import vectorize_pose


@register_env("ClosedLoop-TurnFaucet-v0", max_episode_steps=200, override=True)
class ClosedLoopTurnFaucetEnv(VATTurnFaucetEnv):

    def __init__(self, trajectory_follow_scale=7.0,
                 trajectory_follow_scale_decay=1.5,
                 trajectory_follow_reward_weight=1.0,
                 use_trajectory_follow_reward=True,
                 use_trajectory_follow_observation=True,
                 use_contact_point_reward=True,
                 use_contact_point_observation=True,
                 use_contact_normal_observation=True,
                 use_contact_point_feature=False,
                 filter_robot_links=False,
                 trajectory_reward_mode='track_direction',
                 penalize_step=0,
                 distance_penalty=10,
                 error_penalty=0,
                 use_relative_trajectory_reward=False,
                 lag_to_contouring_ration=0.8,
                 use_rotational_distance=False,
                 curriculum_half_time=0,
                 **kwargs):

        # Reward
        self.use_trajectory_follow_reward = use_trajectory_follow_reward
        self.use_contact_point_reward = use_contact_point_reward
        self.distance_penalty = distance_penalty
        self.error_penalty = error_penalty
        self.penalize_step = penalize_step
        self.curriculum_factor = 1
        self.curriculum_step = np.power(0.5, 1 / curriculum_half_time) if curriculum_half_time > 0 else 1

        # Trajectory Following
        self.use_relative_trajectory_reward = use_relative_trajectory_reward
        self.trajectory_follow_scale = trajectory_follow_scale
        self.trajectory_follow_scale_decay = trajectory_follow_scale_decay
        self.trajectory_follow_reward_weight = trajectory_follow_reward_weight
        self.trajectory_reward_mode = trajectory_reward_mode
        self.use_rotational_distance = use_rotational_distance
        # Contouring
        self.lag_to_contouring_ration = lag_to_contouring_ration
        self.traj_progress = 0
        self.traj_step = 0
        self.phi = 0
        self.phi_step = 0
        # Helpers for different reward computations
        self.last_tcp_pos = None
        self.current_goal_index = 0
        self.last_distance = None
        self.error_penalty_counter = 0
        self.trajectory_plane = None

        # Observation
        self.use_trajectory_follow_observation = use_trajectory_follow_observation
        self.use_contact_point_observation = use_contact_point_observation
        self.use_contact_normal_observation = use_contact_normal_observation
        self.use_contact_point_feature = use_contact_point_feature
        self.filter_robot_links = filter_robot_links
        # Prefilling some observations
        self.num_environment_steps = kwargs.pop('num_waypoints')
        self.num_reference_waypoints = 8  # kwargs.pop('num_reference_waypoints')  # the time limit is automatically set to be the same value
        self.waypoint_dim = 6  # kwargs.pop('waypoint_dim')
        self.open_loop_trajectory = np.zeros(self.waypoint_dim * self.num_reference_waypoints)
        self.contact_point_feature = np.zeros(128)

        # For logging reward terms to Wandb
        self.traj_follow_reward = 0
        self.turn_reward_1 = 0
        self.turn_reward_2 = 0
        self.contact_point_distance_reward = 0
        super(ClosedLoopTurnFaucetEnv, self).__init__(**kwargs)

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict()

        if self.use_contact_point_observation:
            obs['contact_point'] = self.world_to_base_frame(self.contact_point)
        if self.use_trajectory_follow_observation:
            obs['traj'] = self.open_loop_trajectory
        if self.use_contact_normal_observation:
            obs['normal'] = self.contact_normal

        obs['error'] = np.array(int(self.error))

        if self.use_contact_point_feature:
            obs['contact_point_feature'] = self.contact_point_feature
        if self.trajectory_reward_mode == 'contouring_reward' and self.use_trajectory_follow_observation:
            obs['phase'] = self.fourier_encode(self.phi)

        # ELAPSED_STEPS MUST BE THE LAST ITEM IN THE DICTIONARY!
        # TASK MUST BE THE SECOND LAST ITEM IN THE DICTIONARY!
        obs['task'] = np.array(self.target_angle - self.init_angle)
        obs['elapsed_steps'] = np.array(self.elapsed_steps)

        return obs

    def fourier_encode(self, angle):
        denominators = np.deg2rad(np.array([180, 90, 45, 30, 20, 10]))
        # Convert phase to radians since numpy's trig functions expect radians
        # Generate the encoded features

        encoded_features = []
        for denom in denominators:
            # Add sine and cosine components for each denominator
            encoded_features.append(np.sin(angle / denom))
            encoded_features.append(np.cos(angle / denom))

        return np.array(encoded_features)

    def step(self, input: Union[None, np.ndarray, Dict]):
        #print(f'{os.getpid()}: \t received input: \t {input}')

        expanded_input, action = self.expand_dim(input)

        if self._elapsed_steps == 0 and isinstance(expanded_input, dict):

            # the action is altered in this function
            # it uses the 'normal' to change the position of the first waypoint
            action = self.extract_axes_and_contact_point(expanded_input)

            # this is just needed to double check if the controller has actually reached the target pose
            target_pos, target_rot = action[0:3], action[3:6]
            target_quat = Rotation.from_rotvec(target_rot).as_quat()[[3, 0, 1, 2]]
            target_pose_base = sapien.Pose(target_pos, target_quat)

            control_mode = 'pd_base_pose'
            # we do NOT provide the control_mode as a part of the action dictionary anymore, because strings
            # cause issues in the replay buffer
            self.step_action(dict(action=action, control_mode=control_mode))
            counter = 0
            # iterate till convergence or exceeding counter
            while np.linalg.norm(self.world_to_base_frame(self.tcp.pose.p) - target_pose_base.p) > 0.005:
                if counter > 10:
                    break
                self.step_action(dict(action=action, control_mode=control_mode))
                counter += 1

            initial_waypoint_pos = self.tcp.pose.p
            initial_waypoint_rot = Rotation.from_quat(self.tcp.pose.q[[1, 2, 3, 0]]).as_rotvec()
            initial_waypoint = np.concatenate([initial_waypoint_pos, initial_waypoint_rot], axis=0)
            initial_waypoint = np.expand_dims(initial_waypoint, axis=0)

            # the waypoints are used during reward calculation
            self.waypoints = np.stack(
                [self.open_loop_trajectory[i * self.waypoint_dim:(i + 1) * self.waypoint_dim] for i in
                 range(self.num_reference_waypoints)])

            self.trajectory_plane = self.compute_trajectory_plane()

            self.waypoints = np.concatenate([initial_waypoint, self.waypoints], axis=0)

            step_distances = np.diff(self.waypoints[:, self.trajectory_plane], axis=-2)
            step_lengths = np.linalg.norm(step_distances, axis=-1)
            total_path_length = np.cumsum(step_lengths)[-1]
            task = abs(self.target_angle - self.init_angle)
            num_steps = (task / self.max_task_angle_difference) * (self.num_environment_steps - 1)

            # the phase is what I use to actually compute the contouring reward
            # phi is what I use as the observation for the agent
            # the goal is to traverse the trajectory with a constant speed. The issue is that some trajectories are
            # much shorter than others, because they correspond to turning the faucet by a smaller angle

            # phase goes from 0 to total_path_length within num_steps
            # num_steps is computed taking into account the target angle difference
            self.traj_step = total_path_length / num_steps
            # phi goes from 0 to the target_angle-init_angle within num_steps
            self.phi_step = self.max_task_angle_difference / (self.num_environment_steps - 1)

        else:
            self.step_action(dict(action=action, control_mode=self.control_mode_))

        self._elapsed_steps += 1

        obs = self.get_obs()
        info = self.get_info(obs=obs, action=action)  # this calls evaluate()
        reward = self.get_reward(obs=obs, action=action, info=info)
        terminated = self.get_done(obs=obs, info=info)

        return obs, reward, terminated, False, info

    def expand_dim(self, input):
        # the RL agent predicts a 4 dim action or 6 dim action
        # the low level controller expects a 7 dim action

        # the action can either be a dict or an array. this function takes care of both cases
        def expand(action_):
            if action_.shape[0] == 4:
                pos = action_[:3]
                yaw = action_[-1]
                axis_angle = np.array([0, 0, yaw])
                action_ = np.hstack([pos, axis_angle])

            gripper = -1
            action_ = np.hstack([action_, gripper])
            return action_

        if isinstance(input, dict):
            # if action is a dict. The action is under key 'action'
            action = input['action'].flatten()
            if action.shape[0] == 7:
                # no need to expand dimension
                return
            input['action'] = expand(action)
            return input, input['action']

        elif isinstance(input, np.ndarray):
            action = input
            if input.shape[0] == 7:
                # no need to expand dimension
                return
            action = expand(action)
            return None, action

    def compute_trajectory_plane(self):
        var_x = np.var(self.waypoints[:, 0])
        var_y = np.var(self.waypoints[:, 1])
        var_z = np.var(self.waypoints[:, 2])
        axis = np.argmin([var_x, var_y, var_z])

        # the plane are the indices which I want to take into account for computing the distance
        plane = [i for i in range(3) if i != axis]

        return [0, 1]

    def extract_axes_and_contact_point(self, action):

        init_action = copy.deepcopy(action['action'])
        contact_point_base = copy.deepcopy(init_action[:3])
        self.contact_point = self.base_to_world_frame(contact_point_base)

        world_to_local = self.target_link.pose.inv().to_transformation_matrix()
        self.contact_point_local = world_to_local[:3, :3] @ self.contact_point + world_to_local[:3, 3]

        axis_angle_base = copy.deepcopy(init_action[3:6])  # the axis is in world coordinate system
        r = Rotation.from_rotvec(axis_angle_base).as_matrix()

        self.gripper_x = r[:, 0]
        self.gripper_y = r[:, 1]
        self.gripper_z = r[:, 2]

        self.contact_normal = action['normal'].flatten()

        init_action[:3] += 0.05 * self.contact_normal
        # the open loop trajectory is used in the observation
        self.open_loop_trajectory = action['open_loop_trajectory'].flatten()

        self.contact_point_feature = action['contact_point_feature'].flatten()

        return init_action

    def reset(self, seed=None, options=None):
        self.contact_point = np.zeros(3)
        self.contact_point_local = None  # np.zeros(3)
        self.gripper_x = np.zeros(3)
        self.gripper_y = np.zeros(3)
        self.gripper_z = np.zeros(3)
        self.open_loop_trajectory = np.zeros(self.waypoint_dim * self.num_reference_waypoints)
        self.contact_point_feature = np.zeros(128)
        self.contact_normal = np.zeros(3)

        # stuff concerning reward computation
        self.last_tcp_pos = None
        self.current_goal_index = 0
        self.last_distance = None
        self.error = False  # track if IK solver could not find solution
        self.error_penalty_counter = 0
        self.traj_progress = 0
        self.traj_step = 0
        self.phi = 0
        self.phi_step = 0
        self.trajectory_plane = None

        self.traj_follow_reward = 0
        self.turn_reward_1 = 0
        self.turn_reward_2 = 0
        self.contact_point_distance_reward = 0

        super().reset(seed=seed, options=options)
        # I have to do this after the reset of the parent class, because the partent class sets self.waypoints to the wrong size
        # which leads the replay buffer to complain
        self.waypoints = np.zeros((self.num_reference_waypoints + 1, self.waypoint_dim))
        # in the first obs I do not want to render the robot
        # this obs is used by perception modules to predict contact point and open loop trajectory
        for l in self.agent.robot.get_links():
            for vb in l.get_visual_bodies():
                vb.set_visibility(False)

        out = self.get_obs()

        # in all other steps I will render the robot:
        for l in self.agent.robot.get_links():
            for vb in l.get_visual_bodies():
                visible = not self.filter_robot_links or l.get_name() in ['panda_hand', 'panda_leftfinger', 'panda_rightfinger']
                vb.set_visibility(visible)

        return out, {}

    def _follow_trajectory_reward(self):
        if self.trajectory_reward_mode == 'mean_position':
            return self.mean_position()
        elif self.trajectory_reward_mode == 'mean_position_closest_waypoints':
            return self.mean_position_closest_waypoints()
        elif self.trajectory_reward_mode == 'track_direction':
            return self.track_direction()
        elif self.trajectory_reward_mode == 'track_position':
            return self.track_position()
        elif self.trajectory_reward_mode == 'contouring_reward':
            return self.contouring_reward()
        else:
            raise NotImplementedError

    def mean_position(self):
        # compute the mean positional difference between current tcp pose and all waypoints
        # base reward on this distance
        r_tot = 0
        for i, wp in enumerate(self.waypoints):
            trans_distance = np.linalg.norm(self.tcp.pose.p - wp[:3])
            scale = max(self.trajectory_follow_scale - self.trajectory_follow_scale_decay * i, 1)
            rot_distance = 0
            if self.use_rotational_distance:
                rot = wp[3:6]
                rot_distance = self._radian_distance(rot, self.tcp.pose.q)
            r_trans = 1 - np.tanh(trans_distance * scale)
            r_rot = 1 - np.tanh(rot_distance * scale/2)
            r = 0.7 * r_trans + 0.3 * r_rot if self.use_rotational_distance else r_trans
            r_tot += r
        r_tot /= self.waypoints.shape[0]
        return r_tot

    def mean_position_closest_waypoints(self):
        # compute the mean positional difference between current tcp pose and the two closest waypoints
        # base reward on this distance
        distances = []
        for i, wp in enumerate(self.waypoints):
            distance = np.linalg.norm(self.tcp.pose.p - wp[:3])
            distances.append((distance, i))
        distances.sort()

        trans_distance = 0.5 * distances[0][0] + 0.5 * distances[1][0]

        rot_distance = 0
        if self.use_rotational_distance:
            closest_wp = self.waypoints[distances[0][-1]]
            second_closest_wp = self.waypoints[distances[1][-1]]

            closest_rot = closest_wp[3:6]
            closest_rot_distance = self._radian_distance(closest_rot, self.tcp.pose.q)
            second_closest_rot = second_closest_wp[3:6]
            second_closest_rot_distance = self._radian_distance(second_closest_rot, self.tcp.pose.q)
            rot_distance = 0.5 * closest_rot_distance + 0.5 * second_closest_rot_distance

        r_trans = 1 - np.tanh(trans_distance * self.trajectory_follow_scale)
        r_rot = 1 - np.tanh(rot_distance * self.trajectory_follow_scale/2)
        r_tot = 0.7 * r_trans + 0.3 * r_rot if self.use_rotational_distance else r_trans

        return r_tot

    def track_direction(self):
        # compute the direction vector between first and last waypoint
        # compute projection of last tcp pose to current tcp pose onto that direction vector
        # base reward on that projection

        if self.last_tcp_pos is None:
            self.last_tcp_pos = self.tcp.pose.p
            return 0

        elif sum(self.tcp.pose.p - self.last_tcp_pos) == 0:
            # whe have not moved
            return 0

        # first_pos = self.contact_point + 0.05 * self.contact_normal  # contact point is in world coordinates
        target_direction = self.waypoints[0][:3] - self.waypoints[-1][:3]  # direction from first waypoint to last
        target_direction /= np.linalg.norm(target_direction)

        true_direction = self.tcp.pose.p - self.last_tcp_pos  # direction vector from last position to current position
        true_direction /= np.linalg.norm(true_direction)

        # projection is [-1, 1] and can be used as reward signal directly
        projection = np.dot(true_direction, true_direction)
        self.last_tcp_pos = self.tcp.pose.p

        return projection

    def track_position(self):
        # compute the positional difference between current tcp pose and next waypoint
        # base reward on this distance

        # I should take into account that the trajectory is actually more or less planar

        # to compute the plane in which the trajectory lies, I first compute the variance in the positional componentes
        # along the three axes. The axis with the least variance is the normal of the plane
        #plane = self.compute_trajectory_plane()
        plane = self.trajectory_plane

        current_waypoint = self.waypoints[self.current_goal_index]

        trans_distance = np.linalg.norm(self.tcp.pose.p[plane] - current_waypoint[plane])

        r_rel = 0
        if self.use_relative_trajectory_reward:
            r_rel = self.compute_relative_trajectory_reward(trans_distance)

        rot_distance = 0
        if self.use_rotational_distance:
            rot = current_waypoint[3:6]
            rot_distance = self._radian_distance(rot, self.tcp.pose.q)

        threshold = 15  # milli meters
        if trans_distance <= threshold * 10**(-3):  # if tcp closer to goal than threshold
            self.current_goal_index += 1
            # make sure that index stays in bounds
            self.current_goal_index = np.clip(self.current_goal_index, 0, self.waypoints.shape[0]-1)

        # recompute distance in case the goal has changed
        # distance = np.linalg.norm(self.tcp.pose.p[plane] - self.waypoints[self.current_goal_index][plane])

        r_trans = 1 - np.tanh(trans_distance * self.trajectory_follow_scale)
        r_rot = 1 - np.tanh(rot_distance * self.trajectory_follow_scale/2)
        r_abs = 0.7 * r_trans + 0.3 * r_rot if self.use_rotational_distance else r_trans

        return r_abs + r_rel

    def compute_relative_trajectory_reward(self, distance_):
        # additionally compute if the current tcp pose is better than the las one
        #
        # initialize the last distance if None
        distance_ = abs(distance_)
        self.last_distance = distance_ if self.last_distance is None else self.last_distance

        relative_movement = self.last_distance - distance_
        # is between -1 and 1. Will penalize if last distance is smaller than distance, else it will reward
        r_rel = np.tanh(relative_movement * 20)

        self.last_distance = distance_

        return r_rel

    def contouring_reward(self):

        def get_reference_point():
            step_distances = np.diff(self.waypoints[:, plane], axis=-2)
            step_lengths = np.linalg.norm(step_distances, axis=-1)
            step_directions = step_distances / step_lengths[..., None]

            has_point_been_passed = np.cumsum(step_lengths) < self.traj_progress
            last_point_idx = np.sum(has_point_been_passed)
            next_point_idx = last_point_idx + 1

            last_point = self.waypoints[last_point_idx][plane]
            last_quat = Rotation.from_rotvec(self.waypoints[last_point_idx][3:6]).as_quat()[[3, 0, 1, 2]]
            if next_point_idx != len(self.waypoints):

                next_point = self.waypoints[next_point_idx][plane]

                current_direction = step_directions[last_point_idx]
                progress_along_segment = self.traj_progress - np.cumsum(step_lengths)[last_point_idx - 1] if last_point_idx > 0 else self.traj_progress
                segment_length = step_lengths[last_point_idx]
                segment_progress_ration = progress_along_segment / segment_length

                interpolated_point = segment_progress_ration * next_point + (1-segment_progress_ration) * last_point

                interpolated_quat = None
                if self.use_rotational_distance:
                    next_quat = Rotation.from_rotvec(self.waypoints[next_point_idx][3:6]).as_quat()[[3, 0, 1, 2]]

                    interpolated_quat = segment_progress_ration * next_quat + (1-segment_progress_ration) * last_quat
                    interpolated_quat /= np.linalg.norm(interpolated_quat)

            else:
                # if we reached the end of the path
                interpolated_quat = last_quat if self.use_rotational_distance else None
                interpolated_point = last_point
                current_direction = step_directions[-1]

            return interpolated_point, interpolated_quat, current_direction

        #plane = self.compute_trajectory_plane()
        plane = self.trajectory_plane

        # computes contouring and lag error between current reference point and current tcp pose
        reference_point, reference_quat, step_direction = get_reference_point()
        trans_error = (self.tcp.pose.p[plane] - reference_point[plane]) + 0.00001
        rot_error = 0
        if reference_quat is not None:
            rot_error = self._radian_distance(reference_quat, self.tcp.pose.q)


        cos_angle = np.clip(np.dot(trans_error, step_direction) / (np.linalg.norm(step_direction) * np.linalg.norm(trans_error)), -1.0, 1.0)
        # arccos only define between -1 and 1. It can happen due to floating point arithmetics that cos_angle is
        # slightly above or below that leads to nans
        angle = np.arccos(cos_angle)
        contouring_error = np.abs(np.sin(angle) * np.linalg.norm(trans_error))
        lag_error = np.abs(np.cos(angle) * np.linalg.norm(trans_error))

        #r_contouring = (1 - np.tanh(contouring_error * self.trajectory_follow_scale))
        #r_lag = (1 - np.tanh(lag_error * self.trajectory_follow_scale))
        distance = self.lag_to_contouring_ration * lag_error + (1 - self.lag_to_contouring_ration) * contouring_error
        r_trans = (1 - np.tanh(distance * self.trajectory_follow_scale))

        #r_trans = self.lag_to_contouring_ration * r_lag + (1 - self.lag_to_contouring_ration) * r_contouring
        r_rot = 1 - np.tanh(rot_error * self.trajectory_follow_scale/2)

        r_tot = 0.7 * r_trans + 0.3 * r_rot if self.use_rotational_distance else r_trans

        # update progress along trajectory
        self.traj_progress += self.traj_step
        self.phi = min(self.phi + self.phi_step, abs(self.target_angle - self.init_angle))

        return r_tot

    def compute_dense_reward(self, info, **kwargs):

        self.curriculum_factor *= self.curriculum_step

        if info["success"]:
            info.update(dict(traj_follow_rew=self.traj_follow_reward,
                             turn_faucet_rew=self.turn_reward_1,
                             rel_turn_faucet_rew=self.turn_reward_2,
                             contact_distance_rew=self.contact_point_distance_reward)
                        )
            return 20.0

        reward = 0.0

        # penalize if actions lead to error, meaning IK solver could not find solution
        if self.error:
            self.error_penalty_counter += 1
            reward -= self.error_penalty * self.error_penalty_counter
        else:
            self.error_penalty_counter = 0

        if self.use_contact_point_reward:
            distance = self._compute_distance()
            self.contact_point_distance_reward = 1 - np.tanh(distance * 5.0)
            self.contact_point_distance_reward *= self.curriculum_factor
            reward += self.contact_point_distance_reward

            if distance > self.gripper_finger_distance_threshold:
                # print("Fingers far away")
                self.contact_point_distance_reward -= self.distance_penalty
                reward -= self.distance_penalty
        else:
            self.contact_point_distance_reward = 0

        if self.use_trajectory_follow_reward and sum(self.waypoints.flatten()) != 0:
            self.traj_follow_reward = self._follow_trajectory_reward()
            self.traj_follow_reward *= self.trajectory_follow_reward_weight
            self.traj_follow_reward *= self.curriculum_factor
            reward += self.traj_follow_reward
        else:
            self.traj_follow_reward = 0

        # just penalize each step to motivate the agent to move towards goal
        reward -= self.penalize_step * self.elapsed_steps

        # Reward signal for turning the faucet
        angle_diff = abs(self.target_angle - self.current_angle)
        self.turn_reward_1 = (1 - np.tanh(angle_diff * 2.0))
        self.turn_reward_1 *= 3
        reward += self.turn_reward_1

        delta_angle = np.abs(self.last_angle_diff) - angle_diff
        # this will give 1 if last angle difference was worse than current one
        # and -1 if last angle difference was better than current one
        self.turn_reward_2 = np.tanh(delta_angle * 2)
        self.turn_reward_2 *= 5
        reward += self.turn_reward_2

        self.last_angle_diff = angle_diff

        info.update(dict(traj_follow_rew=self.traj_follow_reward,
                         turn_faucet_rew=self.turn_reward_1,
                         rel_turn_faucet_rew=self.turn_reward_2,
                         contact_distance_rew=self.contact_point_distance_reward)
                    )

        return reward

    def _radian_distance(self, rot1, rot2):

        if len(rot1) == 3:
            # convert to quaternion
            rot1 = Rotation.from_rotvec(rot1).as_quat()[[3, 0, 1, 2]]
        if len(rot2) == 3:
            # convert to quaternion
            rot2 = Rotation.from_rotvec(rot2).as_quat()[[3, 0, 1, 2]]

        rot1 /= np.linalg.norm(rot1)  # just for sanity
        rot2 /= np.linalg.norm(rot2)  # just for sanity

        # clip to avoid errors in acos
        inner_product = np.clip(np.dot(rot1, rot2), -1.0, 1.0)
        rot_distance = 2 * np.arccos(np.abs(inner_product))  # angle between the two quaternions
        return rot_distance

    def _initialize_training(self):
        # overwriting the partent's class method
        pass

