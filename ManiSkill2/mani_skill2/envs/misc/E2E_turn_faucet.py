from mani_skill2.envs.misc.turn_faucet import TurnFaucetEnv
from mani_skill2.utils.registration import register_env
import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat
from collections import OrderedDict
from mani_skill2.utils.sapien_utils import vectorize_pose
@register_env("E2E-TurnFaucet-v0", max_episode_steps=200, override=True)
class E2ETurnFaucetEnv(TurnFaucetEnv):

    def __init__(self,
                 min_task_angle_difference=30,
                 max_task_angle_difference=180,
                 randomize_initial_faucet_pose=False,
                 control_mode='pd_ee_target_delta_pose',
                 **kwargs):

        kwargs.pop('num_waypoints', None)
        self.success_threshold = np.deg2rad(5)
        self.min_task_angle_difference = np.deg2rad(min_task_angle_difference)
        self.max_task_angle_difference = np.deg2rad(max_task_angle_difference)
        self.gripper_finger_distance_threshold = 0.3  # I believe this is in meters
        self.randomize_initial_faucet_pose = randomize_initial_faucet_pose
        self.control_mode_ = control_mode

        super(E2ETurnFaucetEnv, self).__init__(control_mode=control_mode, **kwargs)


    # Here I need to specify how I define success
    # here I can add all info to the dictionary that I want to output.
    def evaluate(self, **kwargs):
        angle_dist = self.target_angle - self.current_angle
        # success either if we are close to target angle, or if angle diff has changed signs meaning we overshot
        success = abs(angle_dist) < self.success_threshold or np.sign(angle_dist) != np.sign(self.target_angle_diff)
        out_dict = dict(success=success, angle_dist=angle_dist)
        return out_dict

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

    def _get_obs_extra(self) -> OrderedDict:
        obs = OrderedDict(task=self.target_angle - self.init_angle,
                          elapsed_steps=self.elapsed_steps,
                          tcp_pose=vectorize_pose(self.tcp.pose),
                          )
        return obs

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

    # this function is called to init the pose of the faucet. They rotate it around up axis a bit and
    # make position a bit random. We do not need that
    # we only need to shift the faucets z direction up by model_offset
    def _initialize_articulations(self):
        p = np.zeros(3)

        if self.randomize_initial_faucet_pose:
            p[:2] = self._episode_rng.uniform(-0.05, 0.05, [2])
            # p[:2] = self._episode_rng.uniform(-0.1, 0.1, [2])
            p[2] = self.model_offset[2]
            ori = self._episode_rng.uniform(-np.pi / 12, np.pi / 12)
            # ori = self._episode_rng.uniform(-np.deg2rad(25), np.deg2rad(25))
            q = euler2quat(0, 0, ori)
            self.faucet.set_pose(Pose(p, q))
        else:
            p[2] = self.model_offset[2]
            q = euler2quat(0, 0, 0)
            self.faucet.set_pose(Pose(p, q))



