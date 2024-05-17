"""
Soft Actor-Critic Algorithms and Applications:
    https://arxiv.org/abs/1812.05905
Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor:
   https://arxiv.org/abs/1801.01290
"""
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from maniskill2_learn.utils.torch.misc import no_grad

from maniskill2_learn.networks import build_model, build_actor_critic
from maniskill2_learn.utils.torch import build_optimizer, load_checkpoint
from maniskill2_learn.utils.torch import BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.meta import get_logger, get_world_rank
from ..builder import MFRL
from maniskill2_learn.utils.data import GDict
from ..perceptionModules import VAT_SAC_Perception

from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion

# this function expects the quaternion to be in WXYZ format
# this format aligns with SAPIEN
# BUT the the scipy Rotation class uses the XYZW convention!
def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
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
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))

'''

This is captAIn. The working title was VAT-SAC...

The main intuition behind captAIn:
The architecture consists of the Perception Module and the Interaction Policy. The Perception Module contains two
sub-modules, namely the Affordance Predictor and the Trajectory Generator. 
The Affordance Predictor takes as input the Pointcloud and the task and predicts an affordance score for every point.
The point with the highest score is picked as the contact point. 
The Trajectory generator takes as input the pointcloud, task, and contact point and samples a single trajectory.
The trajectory consists of the initial orientation of the TCP and a sequence of TCP poses.
The intitial TCP orientation combined with the contact point form the initial TCP pose, in absolute coordinates.
The sequence of TCP poses, however, is in relative coordinates (relative to preceding TCP pose in sequence)

The very first observation from the environment is passed through the perception module. The output is used to drive the
robot to the predicted initial TCP pose. 
All subsequent interactions with the environment are handled by the ineraction policy. The interaction policy is trained
using soft actor-critic.
'''

@MFRL.register_module(name='VAT-SAC')
class VAT_SAC(BaseAgent):
    def __init__(
        self,
        actor_cfg,
        critic_cfg,
        affordance_predictor_cfg,
        trajectory_generator_cfg,
        env_params,
        batch_size=128,
        gamma=0.99,
        update_coeff=0.005,
        alpha=0.2,
        ignore_dones=True,
        target_update_interval=1,
        automatic_alpha_tuning=True,
        target_smooth=0.90,  # For discrete SAC
        alpha_optim_cfg=None,
        target_entropy=None,
        shared_backbone=False,
        detach_actor_feature=False,
        continuous_learning=False,
        perception_batch_size=128,
        perception_update_freq=500,
        perception_update_n=4,
        **kwargs
    ):
        super(VAT_SAC, self).__init__(**kwargs)

        # constructing the perception module
        self.perception = VAT_SAC_Perception(affordance_predictor_cfg,
                                             trajectory_generator_cfg,
                                             batch_size=perception_batch_size,
                                             num_waypoints=8,
                                             waypoint_dim=6,
                                             train=continuous_learning
                                             )

        self.perception_counter = 0

        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        critic_optim_cfg = critic_cfg.pop("optim_cfg")
        action_shape = env_params["action_shape"]
        self.action_shape = action_shape
        self.is_discrete = env_params["is_discrete"]

        self.gamma = gamma
        self.update_coeff = update_coeff
        self.alpha = alpha
        self.ignore_dones = ignore_dones
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.automatic_alpha_tuning = automatic_alpha_tuning

        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        self.control_mode = env_params['control_mode']

        ## everything related to continuous learning
        self.continuous_learning = continuous_learning
        self.perception_batch_size = perception_batch_size
        self.perception_update_freq = perception_update_freq
        self.perception_update_n = perception_update_n

        self.logger = get_logger()

        self.perception.poseTrajectoryGenerator.eval()  # we do not train the generator
        self.perception.affordancePredictor.eval()  # we only train the affordance predictor

        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)
        self.shared_backbone = shared_backbone
        self.detach_actor_feature = detach_actor_feature

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)

        self.target_critic = build_model(critic_cfg)
        hard_update(self.target_critic, self.critic)

        self.log_alpha = nn.Parameter(torch.ones(1, requires_grad=True) * np.log(alpha))
        if target_entropy is None:
            if env_params["is_discrete"]:
                # Use label smoothing to get the target entropy.
                n = np.prod(action_shape)
                explore_rate = (1 - target_smooth) / (n - 1)
                self.target_entropy = -(target_smooth * np.log(target_smooth) + (n - 1) * explore_rate * np.log(explore_rate))
                self.log_alpha = nn.Parameter(torch.tensor(np.log(0.1), requires_grad=True))
                # self.target_entropy = np.log(action_shape) * target_smooth
            else:
                self.target_entropy = -np.prod(action_shape)
        else:
            self.target_entropy = target_entropy
        if self.automatic_alpha_tuning:
            self.alpha = self.log_alpha.exp().item()

        self.alpha_optim = build_optimizer(self.log_alpha, alpha_optim_cfg)

    def filter_out_VAT_samples(self, batch, mask):

        for k in batch.keys():
            if isinstance(batch[k], dict):
                batch[k] = self.filter_out_VAT_samples(batch[k], mask)
            else:
                batch[k] = batch[k][mask]

        return batch

    def padding_batch(self, batch, pad):
        for k in batch.keys():
            if isinstance(batch[k], dict):
                batch[k] = self.padding_batch(batch[k], pad)
            else:
                repeat_size = [pad] + [1] * (len(batch[k].size()) - 1)
                last_row_repeated = batch[k][-1].unsqueeze(0).repeat(*repeat_size)
                # Append the repeated last row to the original tensor
                batch[k] = torch.cat([batch[k], last_row_repeated], dim=0)

        return batch

    def reduce_dim(self, batch):
        # need to reduce to 4 dim:
        action = batch['actions']['action']
        original_left = action[:, :-3]  # position
        original_last_column = action[:, -1:]  # yaw
        reduced_action = torch.cat([original_left, original_last_column], dim=1)
        batch['actions']['action'] = reduced_action
        return batch

    def update_perception_helper(self, recent_trajectory_replay):
        out_dict = self.perception.update_parameters(recent_trajectory_replay, 0)
        return out_dict

    def update_perception(self, recent_trajectory_replay, n_finished_episodes):
        assert self.continuous_learning, 'updating the perception modules is only possible if continuous learning is true'

        self.perception_counter += n_finished_episodes

        if self.perception_counter >= self.perception_update_freq:
            self.perception.poseTrajectoryGenerator.train()
            self.perception.affordancePredictor.train()
            out_dict = dict()
            num_samples = 0
            for _ in range(self.perception_update_n):
                ret_dict = self.update_perception_helper(recent_trajectory_replay)
                for key in ret_dict:
                    out_dict[key] = out_dict.get(key, 0) + ret_dict[key] * self.perception_batch_size
                num_samples += self.perception_batch_size
            self.perception_counter = 0
            self.perception.poseTrajectoryGenerator.eval()
            self.perception.affordancePredictor.eval()
            for key in out_dict:
                out_dict[key] /= num_samples
            return True, out_dict
        else:
            return False, {}

    # during parameter update I must skip all the samples in which elapsed steps is 0!
    def update_parameters(self, memory, updates):
        VAT_action_mask = [self.batch_size]  # just to enter the while loop

        # to avoid filtering out ALL samples from the batch, we must resample
        with torch.no_grad():

            while sum(VAT_action_mask) == self.batch_size:
                sampled_batch = memory.sample(self.batch_size).to_torch(device=self.device, non_blocking=True)
                sampled_batch = self.process_obs(sampled_batch)

                # I will use the 'open loop trajectory' key from the actions to determine whether
                # the sample should be considered for training SAC or not...
                # sampled_batch['actions']['open_loop_trajectory'] of shape B, self.recon_traj_shape
                VAT_action_mask = (sampled_batch['actions']['open_loop_trajectory'] != 0).any(dim=1)
                if sum(VAT_action_mask) == self.batch_size:
                    self.logger.error(f'All samples from batch were filtered out. Will resample!')

            sampled_batch = self.filter_out_VAT_samples(sampled_batch, ~VAT_action_mask)
            
            '''loss_mask = torch.ones(self.batch_size, device=self.device)
            if sum(VAT_action_mask) != 0:
                # if samples from batch were removed, we need to add some padding to recover original batch size
                # self.logger.error(f'Padded the batch by {sum(VAT_action_mask)}!')
                sampled_batch = self.padding_batch(sampled_batch, pad=sum(VAT_action_mask))
                # during loss computation I do not want to take into consideration the padded samples
                # so I set the mask to 0 for the padded samples
                loss_mask[-sum(VAT_action_mask):] = 0'''

            if self.action_shape < 6:
                # this is because in the replay buffer the actions are 6 dimensional
                # because I have expanded the actions during forward call
                sampled_batch = self.reduce_dim(sampled_batch)

        with torch.no_grad():
            next_action, next_log_prob = self.actor(sampled_batch["next_obs"], mode="all")[:2]
            q_next_target = self.target_critic(sampled_batch["next_obs"], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values - self.alpha * next_log_prob
            if self.ignore_dones:
                q_target = sampled_batch["rewards"] + self.gamma * min_q_next_target
            else:
                q_target = sampled_batch["rewards"] + (
                            1 - sampled_batch["dones"].float()) * self.gamma * min_q_next_target
            q_target = q_target.repeat(1, q_next_target.shape[-1])
        q = self.critic(sampled_batch["obs"], sampled_batch["actions"])

        critic_loss = F.mse_loss(q, q_target) * q.shape[-1]
        # critic_loss_unmasked = (q - q_target) ** 2
        # critic_loss_masked = critic_loss_unmasked[loss_mask.bool()]
        # critic_loss = critic_loss_masked.mean()
        with torch.no_grad():
            abs_critic_error = torch.abs(q - q_target).max().item()
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        with torch.no_grad():
            critic_grad = self.critic.grad_norm
        if self.shared_backbone:
            self.critic_optim.zero_grad()

        pi, log_pi = self.actor(sampled_batch["obs"], mode="all", save_feature=self.shared_backbone, detach_visual=self.detach_actor_feature)[:2]
        entropy_term = -log_pi.mean()
        # log_pi_masked = log_pi[loss_mask.bool()]
        # log_pi_masked = log_pi_masked.mean()
        # entropy_term = -log_pi_masked

        visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")
        if visual_feature is not None:
            visual_feature = visual_feature.detach()

        q_pi = self.critic(sampled_batch["obs"], pi, visual_feature=visual_feature)
        q_pi = torch.min(q_pi, dim=-1, keepdim=True).values

        # actor_q_loss_masked = q_pi[loss_mask.bool()]
        # actor_q_loss = actor_q_loss_masked.mean()

        actor_loss = -(q_pi.mean() + self.alpha * entropy_term)
        # actor_loss = -(actor_q_loss + self.alpha * entropy_term)
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()
        with torch.no_grad():
            actor_grad = self.actor.grad_norm

        if self.automatic_alpha_tuning:
            alpha_loss = self.log_alpha.exp() * (entropy_term - self.target_entropy).detach()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            alpha_loss = torch.tensor(0.0).to(self.device)
        if updates % self.target_update_interval == 0:
            soft_update(self.target_critic, self.critic, self.update_coeff)

        ret = {
            "sac/critic_loss": critic_loss.item(),
            "sac/max_critic_abs_err": abs_critic_error,
            "sac/actor_loss": actor_loss.item(),
            "sac/alpha": self.alpha,
            "sac/alpha_loss": alpha_loss.item(),
            "sac/q": torch.min(q, dim=-1).values.mean().item(),
            "sac/q_target": torch.mean(q_target).item(),
            "sac/entropy": entropy_term.item(),
            "sac/target_entropy": self.target_entropy,
            "sac/critic_grad": critic_grad,
            "sac/actor_grad": actor_grad,
        }

        return ret

    def construct_initial_waypoint(self, pointcloud, down, forward, contact_point):

        # construct start pose from gripper directions and contact point
        down /= torch.linalg.norm(down, dim=-1).unsqueeze(-1)
        forward /= torch.linalg.norm(forward, dim=-1).unsqueeze(-1)

        # Computing right as cross product and normalizing
        right = torch.cross(down, forward, dim=-1)
        right = right / torch.linalg.norm(right, dim=-1).unsqueeze(-1)

        # Recomputing down for orthonormal basis
        down = torch.cross(forward, right)
        # down = down / torch.linalg.norm(down, dim=-1).unsqueeze(-1)
        # Constructing gripper orientation
        rot_mat = torch.stack([forward, right, down], dim=-1)  # num_envs, (num_traj), 3, 3
        axis_angle = matrix_to_axis_angle(rot_mat)

        normal = self.construct_normal(pointcloud, contact_point, forward)

        recon_rot_mat = axis_angle_to_matrix(axis_angle)

        # Comparisons (might need to be adapted based on actual use-case, e.g., using torch.allclose for tensors)
        down_bool = torch.allclose(down, recon_rot_mat[:, 2])
        right_bool = torch.allclose(right, recon_rot_mat[:, 1])
        forward_bool = torch.allclose(forward, recon_rot_mat[:, 0])

        # Concatenating tensors
        wp = torch.cat([contact_point, axis_angle], dim=-1)

        return wp, normal

    def construct_normal(self, pointcloud, contact_point, forward):
        # we need the normal at the contact point in order to compute the first pose of the robot
        # the normal is defined as pointing away from the surface the contact point is located on
        # we know that the forward direction predicted by the model is collinear with the normal

        original_mean_distance = self.calculate_mean_distance(contact_point, pointcloud)

        shifted_point_plus = contact_point + forward * 0.01
        mean_distance_plus = self.calculate_mean_distance(shifted_point_plus, pointcloud)

        shifted_point_minus = contact_point - forward * 0.01
        mean_distance_minus = self.calculate_mean_distance(shifted_point_minus, pointcloud)

        mean_distance_plus_tensor = torch.stack(mean_distance_plus)  # Stack list of tensors into a single tensor
        mean_distance_minus_tensor = torch.stack(mean_distance_minus)
        original_mean_distance_tensor = torch.stack(original_mean_distance)

        condition = (mean_distance_plus_tensor > original_mean_distance_tensor) & (
                    mean_distance_plus_tensor > mean_distance_minus_tensor)

        normal = torch.empty_like(forward)

        # Apply condition to select between forward and -forward
        normal[condition] = forward[condition]
        normal[~condition] = -forward[~condition]

        return normal

    def calculate_mean_distance(self, point, point_cloud):
        # poincloud of shape (environment_num, num_traj, 650, 3), point is of shape (environment_num, num_traj, 3)
        distances = torch.sqrt(torch.sum((point_cloud - point.unsqueeze(1)) ** 2, dim=-1))
        distance_list = []
        mask = distances <= 0.05
        for i in range(distances.shape[0]):
            d = distances[i, mask[i]]
            distance_list.append(torch.mean(d))

        return distance_list


    def expand_dim(self, action):

        # Split the tensor into two parts: before the insertion point (everything except the last column) and the last column

        environment_num = action.shape[0]
        zeros_to_insert = torch.zeros(environment_num, 2, device=action.device)  # Shape (N, 2) for two columns of zeros

        left = action[:, :-1]  # All columns except the last
        last_column = action[:, -1:]  # The last column, kept as a 2D tensor for correct concatenation

        # Concatenate the parts together: before the insertion point, zeros, and the last column
        return torch.cat([left, zeros_to_insert, last_column], dim=1)

    # The forward function contains the logic for environment interaction
    # ManiSkill uses vectorized environments. This means the agent interacts with multiple environments at the same time
    # The environment observations are concatenated in the batch dimension. The agent predicts a seperate action for
    # every environment.
    # This is done just to accelerate the training process.
    # Unfortunately, I could not make it work properly. For some reason processing a batch of observations did not lead
    # to the same results as processing each observation individually. There is some underlying issue, I could not find
    # So instead, the forward function loops over every environment observation and invokes the forward_single function
    # The actions are then organized into a single dictionary.
    @no_grad
    def forward(self, obs, **kwargs):
        n_envs = obs['state'].shape[0]

        if n_envs == 1:
            obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
            return self.forward_single(obs, 0, **kwargs)

        action = None
        for i in range(n_envs):
            single_obs = obs.slice(i).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
            for k in single_obs.keys():
                single_obs[k] = single_obs[k].unsqueeze(0)
            action_single = self.forward_single(single_obs, i, **kwargs)

            # merging the dictionaries
            if action is None:
                action = action_single
            else:
                for k in action_single.keys():
                    action[k] = np.concatenate([action[k], action_single[k]])

        return action


    # This function performs the actual forward pass through the networks
    def forward_single(self, obs, index, **kwargs):

        # getting all observations from environments that were just reset, i.e. elapsed step == 0
        # for those we need special treatment
        # the very first step in the actions needs to be done by perception module

        elapsed_steps = obs['state'][:, -1]
        # if there are no environments going through their first step, we dont need to compute the perception approach
        if elapsed_steps == 0:
            pointcloud = obs['xyz']  # num_env, 1200, 3
            task = obs['state'][:, -2].view(1, -1)

            # Compute affordances and select contact points
            affordances, pn2_features = self.perception.affordancePredictor.inference(pointcloud, torch.sign(task))  # num_env, 650, 1
            aff_max_idx = torch.argmax(affordances, dim=1)  # num_env, 1
            num_envs_first_step = aff_max_idx.shape[0]
            environment_vector = torch.arange(0, num_envs_first_step).to(self.device)  # num_envs
            contact_point = pointcloud[environment_vector, aff_max_idx.squeeze(-1)].clone()

            contact_point_feature = pn2_features[environment_vector, aff_max_idx.squeeze(-1)].clone()

            # Sample trajectory
            trajectories = self.perception.poseTrajectoryGenerator.sample_trajectories(pointcloud, task, contact_point, num_trajectories=1)  # num_env, num_trajectories, 38
            recon_traj = trajectories[:, 0, :].clone()  # num_env, 38

            # from the reconstructed trajectory we need to extract the very first waypoint and also the normal
            forward = recon_traj[..., :3]  # num_env, 3
            down = recon_traj[..., 3:6]  # num_env, 3

            initial_waypoint, normal = self.construct_initial_waypoint(pointcloud, down, forward, contact_point)

            # we need to pass the normal and the reconstructed trajectory to every environment
            # for the environments that go through their first step the normal is set to the computed normal above (same for the trajectory)
            # for the other environment I need to set some dummy data
            self.normal_shape = normal.shape[1]
            normal_full = normal.clone()

            # we do not need the initial waypoint in the open loop trajectory
            recon_traj_full = recon_traj[..., 6:].clone()
            self.recon_traj_shape = recon_traj_full.shape[1]

            self.contact_point_feature_shape = contact_point_feature.shape[1]
            cp_feature_full = contact_point_feature

            # fuse together the actions from SAC and perception module
            # for all environments that go through their first step, we take the actions from perception module,
            # for the rest we keep the actions from SAC
            action = initial_waypoint.clone()
        else:
            # if its not the first step, then SAC is going to compute the action
            action = super().forward(obs, **kwargs)
            if self.action_shape < 6:
                # I have to expand the dimenions to 6 in order to match the action dimenions from the peception module
                # the very first action given bey the perception modules will always be 6 dimensional
                # beccause we need all 6 degrees of freedom to position the gripper at the contact point in the corret
                # orientation
                action = self.expand_dim(action)
            # the dummy data we need so that the replay buffer doesnt complain about inhomogeneous actions
            dummy_value = 0
            recon_traj_full = torch.full((1, self.recon_traj_shape), dummy_value, dtype=torch.float32, device=action.device)
            normal_full = torch.full((1, self.normal_shape), dummy_value, dtype=torch.float32, device=action.device)
            cp_feature_full = torch.full((1, self.contact_point_feature_shape), dummy_value, dtype=torch.float32, device=action.device)
            affordances = torch.full((1, 1200, 1), dummy_value, dtype=torch.float32, device=action.device)

        # we must pass the action as a dictionary in order to be able to give the environment the reconstructed
        # trajectory and the normal
        # Both those entities will be added to the observations, in the case that the environment is going through its first step
        # Additionally the reconstructed trajectory will be used in the reward computation to guide the agent

        # if the environment is NOT going through its first step, it will ignore the normal and the trajectory, so its
        # fine to simply pass dummy data for them.
        action_out = action.detach().cpu().numpy()
        traj_out = recon_traj_full.detach().cpu().numpy()
        normal_out = normal_full.cpu().numpy()
        cp_feature_out = cp_feature_full.cpu().numpy()
        # output predicted affordance map too, if you want to visualize the affordance later
        # the affordance map will automatically be inside of the replay buffer if its part of the action
        affordances_out = affordances.cpu().numpy()

        return dict(action=action_out, open_loop_trajectory=traj_out, normal=normal_out, contact_point_feature=cp_feature_out) #, affordance=affordances_out)

