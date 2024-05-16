import json
import logging
from maniskill2_learn.networks import build_model
from maniskill2_learn.utils.torch import build_optimizer, get_mean_lr, BasePerception, load_checkpoint, freeze_params
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_logger_name
from copy import deepcopy
from ..builder import MBRL
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from maniskill2_learn.utils.data import GDict
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_quaternion
from maniskill2_learn.utils.torch.misc import no_grad


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


@MBRL.register_module(name='VAT-Mart')
class VATPerception(BasePerception):
    def __init__(self, affordance_predictor_cfg,
                 trajectory_scorer_cfg,
                 trajectory_generator_cfg,
                 pose_trajectory_generator_cfg=None,
                 batch_size=1024,
                 share_backbone=False,
                 detach_features=False,
                 env_params=None,
                 num_traj=100,
                 num_waypoints=4,  # will be overwritten in run_rl
                 waypoint_dim=6,  # will be overwritten in run_rl
                 feat_dim=128,
                 mode="trajectoryScorer",
                 control_mode='pd_ee_delta_pose',
                 use_dataset=False,
                 trajectory_scorer_checkpoint_path=None,
                 trajectory_generator_checkpoint_path=None,
                 affordance_predictor_checkpoint_path=None,
                 pose_trajectory_generator_checkpoint_path=None,
                 ):
        super(VATPerception, self).__init__()
        log_level = logging.ERROR
        env_suffix = f""
        self.logger = get_logger("Agent-" + env_suffix, log_level=log_level)
        self.world_rank = get_world_rank()
        self.batch_size = batch_size
        self.use_dataset = use_dataset

        if env_params is not None:
            # for inference, the action shape is given by environment
            # it must align with the environment config that the RL agent was trained on
            self.waypoint_dim = env_params['action_shape']
            self.waypoint_dim = self.waypoint_dim-1 if self.waypoint_dim == 7 else self.waypoint_dim
            self.control_mode = env_params['control_mode']
            self.logger.info(f'VAT USES CONTROL MODE {self.control_mode}')
            self.num_steps = env_params['num_waypoints']
        else:
            # for training we populate waypoint_dim by sampling a data point from the dataset and
            # computing the waypoint dim and number of waypoint, i.e traj len from the data sample
            self.waypoint_dim = waypoint_dim
            self.num_steps = num_waypoints

        self.num_trajectories_to_generate = num_traj
        self.mode = mode
        self.use_pose_trajectory = pose_trajectory_generator_checkpoint_path is not None

        # need to propagate num_traj, num_waypoints, waypoint_dim, feat_dim to the other configs
        # because the influence the construction of MLPs

        trajectory_scorer_cfg = deepcopy(trajectory_scorer_cfg)
        trajectory_generator_cfg = deepcopy(trajectory_generator_cfg)
        affordance_predictor_cfg = deepcopy(affordance_predictor_cfg)

        trajectory_scorer_optim_cfg = trajectory_scorer_cfg.pop('optim_cfg')
        trajectory_generator_optim_cfg = trajectory_generator_cfg.pop('optim_cfg')
        affordance_predictor_optim_cfg = affordance_predictor_cfg.pop('optim_cfg')

        trajectory_scorer_lr_scheduler_cfg = trajectory_scorer_cfg.pop('lr_scheduler_cfg', None)
        trajectory_generator_lr_scheduler_cfg = trajectory_generator_cfg.pop('lr_scheduler_cfg', None)
        affordance_predictor_lr_scheduler_cfg = affordance_predictor_cfg.pop('lr_scheduler_cfg', None)

        if mode == "trajectoryScorer":
            self.trajectoryScorer = build_model(trajectory_scorer_cfg)
            self.trajectoryScorer_optim = build_optimizer(self.trajectoryScorer, trajectory_scorer_optim_cfg)
            self.trajectoryScorer_lr_scheduler = None
            if trajectory_scorer_lr_scheduler_cfg:
                trajectory_scorer_lr_scheduler_cfg['optimizer'] = self.trajectoryScorer_optim
                self.trajectoryScorer_lr_scheduler = build_lr_scheduler(trajectory_scorer_lr_scheduler_cfg)

        elif mode == "trajectoryGenerator":
            self.trajectoryGenerator = build_model(trajectory_generator_cfg)
            self.trajectoryGenerator_optim = build_optimizer(self.trajectoryGenerator, trajectory_generator_optim_cfg)
            self.trajectoryGenerator_lr_scheduler = None
            if trajectory_generator_lr_scheduler_cfg:
                trajectory_generator_lr_scheduler_cfg['optimizer'] = self.trajectoryGenerator_optim
                self.trajectoryGenerator_lr_scheduler = build_lr_scheduler(trajectory_generator_lr_scheduler_cfg)

        elif mode == "affordancePredictor":
            self.affordancePredictor = build_model(affordance_predictor_cfg)

            self.affordancePredictor_optim = build_optimizer(self.affordancePredictor, affordance_predictor_optim_cfg)
            self.affordancePredictor_lr_scheduler = None

            if affordance_predictor_lr_scheduler_cfg:
                affordance_predictor_lr_scheduler_cfg['optimizer'] = self.affordancePredictor_optim
                self.affordancePredictor_lr_scheduler = build_lr_scheduler(affordance_predictor_lr_scheduler_cfg)

            if not self.use_dataset:
                assert trajectory_scorer_cfg is not None and trajectory_generator_cfg is not None,\
                    f'Need to provide trajectory_scorer_cfg and trajectory_generator_cfg when use_dataset is False for training affordace Predictor'

                # load and freeze parameters
                self.trajectoryScorer = build_model(trajectory_scorer_cfg)
                mapping = {r'^trajectoryScorer\.': ''}  # remove trajectoryScorer prefix when loading
                load_checkpoint(self.trajectoryScorer, trajectory_scorer_checkpoint_path, 'cuda', logger=get_logger(), keys_map=mapping)
                self.trajectoryScorer.eval()
                # freeze_params(self.trajectoryScorer)

                self.trajectoryGenerator = build_model(trajectory_generator_cfg)
                mapping = {r'^trajectoryGenerator\.': ''}
                load_checkpoint(self.trajectoryGenerator, trajectory_generator_checkpoint_path, 'cuda', logger=get_logger(), keys_map=mapping)
                self.trajectoryGenerator.eval()
                # freeze_params(self.trajectoryGenerator)

            else:
                self.logger.error('USING DATASET FOR TRAINING AFFORDANCE PREDICTOR')

        elif mode == 'inference':
            self.logger.info('VAT in inference mode!')
            if trajectory_generator_checkpoint_path and trajectory_scorer_checkpoint_path and affordance_predictor_checkpoint_path:
                self.logger.info('loading all models individually')
                # in this case all three networks will be loaded individually
                # the affordance prediction module was trained on the dataset directly, thus we have three seperate
                # checkpoints, one for the affordace predictor, one for th generator and one for the scorer
                self.trajectoryScorer = build_model(trajectory_scorer_cfg)
                mapping = {r'^trajectoryScorer\.': ''}  # remove trajectoryScorer prefix when loading
                load_checkpoint(self.trajectoryScorer, trajectory_scorer_checkpoint_path, 'cuda', logger=get_logger(),
                                keys_map=mapping)
                self.trajectoryScorer.eval()
                # freeze_params(self.trajectoryScorer)

                self.trajectoryGenerator = build_model(trajectory_generator_cfg)
                mapping = {r'^trajectoryGenerator\.': ''}
                load_checkpoint(self.trajectoryGenerator, trajectory_generator_checkpoint_path, 'cuda',
                                logger=get_logger(), keys_map=mapping)
                self.trajectoryGenerator.eval()
                # freeze_params(self.trajectoryGenerator)

                self.affordancePredictor = build_model(affordance_predictor_cfg)
                mapping = {r'^affordancePredictor\.': ''}  # remove trajectoryScorer prefix when loading
                load_checkpoint(self.affordancePredictor, affordance_predictor_checkpoint_path, 'cuda', logger=get_logger(),
                                keys_map=mapping)
                self.affordancePredictor.eval()
                # freeze_params(self.affordancePredictor)

                if self.use_pose_trajectory:
                    _ = pose_trajectory_generator_cfg.pop('optim_cfg')
                    _ = pose_trajectory_generator_cfg.pop('lr_scheduler_cfg', None)
                    self.poseTrajectoryGenerator = build_model(pose_trajectory_generator_cfg)
                    mapping = {r'^trajectoryGenerator\.': ''}
                    load_checkpoint(self.poseTrajectoryGenerator, pose_trajectory_generator_checkpoint_path, 'cuda',
                                    logger=get_logger(), keys_map=mapping)
                    self.poseTrajectoryGenerator.eval()
                    # freeze_params(self.trajectoryGenerator)
            else:
                # in this case the affordance predictor module was trained using the other two networks,
                # thus we can resume from the last affordance predictor checkpoint directly
                self.logger.info("resuming from affordance predictor's checkpoint")
                self.trajectoryScorer = build_model(trajectory_scorer_cfg)
                self.trajectoryGenerator = build_model(trajectory_generator_cfg)
                self.affordancePredictor = build_model(affordance_predictor_cfg)
                # freeze_params(self.trajectoryScorer)
                # freeze_params(self.trajectoryGenerator)
                # freeze_params(self.affordancePredictor)

            self.waypoints = None

        else:
            raise NotImplementedError

        # used in TrajectoryScorer and AffordancePredictor
        self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.BCELoss = torch.nn.BCELoss(reduction='none')
        self.L1Loss = torch.nn.L1Loss(reduction='none')


    def trajectory_to_waypoints(self, pointcloud, trajectory, contact_point, expand_dim=False, waypoint_dim=None):
        # up, forward, waypoints
        num_envs = trajectory.shape[0]
        wp_dim = self.waypoint_dim if waypoint_dim is None else waypoint_dim

        forward = trajectory[..., :3]  # num_envs, (num_traj),  3
        down = trajectory[..., 3:6]  # num_envs, (num_traj), 3
        waypoints_flat = trajectory[..., 6:]  # num_envs, (num_traj), 32

        initial_waypoint, normal = self.construct_initial_waypoint(pointcloud, down, forward, contact_point)

        num_waypoints = waypoints_flat.shape[1] // wp_dim
        waypoints = waypoints_flat.view(num_envs, num_waypoints, wp_dim)  # num_envs, num_waypoints, waypoint_dim

        if wp_dim == 4 and expand_dim:
            # we MUST expand the 4 dimensional waypoints to 6
            # because the initial waypoint is 6 dimensional
            pos = waypoints[..., :3]  # Extract position from waypoints  num_envs, num_waypoints, 3
            yaw = waypoints[..., -1]  # Extract yaw from waypoints  num_envs, num_waypoints, 1
            axis_angle = torch.zeros(num_envs, num_waypoints, 3, device=trajectory.device)
            axis_angle[..., -1] = yaw  # Set yaw as the z-component of axis_angle
            # Concatenate pos and axis_angle to get the expanded waypoints
            waypoints = torch.cat([pos, axis_angle], dim=-1)  # num_envs, num_waypoints, 6

        # Prepend initial_waypoint to the waypoints tensor
        initial_waypoint_expanded = initial_waypoint.unsqueeze(1)  # Add an extra dimension for concatenation
        waypoints = torch.cat([initial_waypoint_expanded, waypoints], dim=1)

        return waypoints, normal

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

    def prefill_action(self, obs):
        N =obs['xyz'].shape[0] # number of environments
        num_points = obs['xyz'].shape[1]  # number points in pointcloud
        dummy_value = 0

        action_ = torch.full((N, 6), dummy_value, dtype=torch.float32, device=self.device)
        affordance_ = torch.full((N, num_points, 1), dummy_value, dtype=torch.float32, device=self.device)
        normal_ = torch.full((N, 3), dummy_value, dtype=torch.float32, device=self.device)
        trajectories_ = torch.full((N, self.num_trajectories_to_generate, self.num_steps + 1, 6), dummy_value, dtype=torch.float32, device=self.device)
        pose_trajectory_ = torch.full((N, self.num_steps + 1, 6), dummy_value, dtype=torch.float32, device=self.device)

        return dict(action=action_, affordance=affordance_, normal=normal_, trajectories=trajectories_, pose_trajectory=pose_trajectory_)

    @no_grad
    def forward(self, obs, **kwargs):
        n_envs = obs['state'].shape[0]

        action = []
        if self.waypoints is None:
            # pre_fill waypoints with dummy data
            dummy_value = 0
            self.waypoints = torch.full((n_envs, self.num_steps + 1, 6), dummy_value, dtype=torch.float32,
                                        device=self.device)

        if n_envs == 1:
            obs = GDict(obs).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
            return self.forward_single(obs, 0)

        for i in range(n_envs):
            single_obs = obs.slice(i).to_torch(dtype="float32", device=self.device, non_blocking=True, wrapper=False)
            for k in single_obs.keys():
                single_obs[k] = single_obs[k].unsqueeze(0)
            action_single = self.forward_single(single_obs, i)
            action.append(action_single)

        return action

    def forward_single(self, obs, index):
        assert self.mode == 'inference', 'forward pass with VAT is only possible in inference mode'

        #action = self.prefill_action(obs)
        action = dict()

        elapsed_steps = obs['state'][:, -1]

        # recompute waypoints if environment has reset
        if elapsed_steps == 0:

            pointcloud = obs['xyz']  # num_env, 1200, 3
            task = obs['state'][:, -2].view(1, -1)

            # Compute affordances and select contact points
            affordances, _ = self.affordancePredictor.inference(pointcloud, torch.sign(task))  # num_env, 650, 1
            aff_max_idx = torch.argmax(affordances, dim=1)  # num_env, 1
            num_envs_first_step = aff_max_idx.shape[0]
            environment_vector = torch.arange(0, num_envs_first_step).to(self.device)  # num_envs
            contact_point = pointcloud[environment_vector, aff_max_idx.squeeze(-1)].clone()

            # Sample, score, and select trajectories
            trajectories = self.trajectoryGenerator.sample_trajectories(pointcloud, task, contact_point, num_trajectories=self.num_trajectories_to_generate)  # num_env, num_trajectories, 38
            traj_scores = self.trajectoryScorer.score_trajectories(pointcloud, task, trajectories,
                                                                   contact_point)  # num_env, num_trajectories, 1
            top_score, top_idx = traj_scores.max(dim=1)  # num_envs, 1 both
            recon_traj = trajectories[environment_vector, top_idx.squeeze(-1)].clone()  # num_env, 38

            # Compute waypoints
            waypoints, normal = self.trajectory_to_waypoints(pointcloud, recon_traj, contact_point, expand_dim=True)

            # Now update the waypoints for the environments that go through first step
            self.waypoints[index] = waypoints.clone()  # num_envs, num_waypoints+1, 6

            action['normal'] = normal.clone()

            action['affordance'] = affordances.clone()

            num_envs_first_step, num_trajectories, traj_len = trajectories.shape
            # too lazy to vectorize the computation over number of trajectories
            waypoints = torch.zeros(size=(num_envs_first_step, num_trajectories, self.waypoints.shape[1], self.waypoints.shape[2]), device=self.device)
            for i in range(num_trajectories):
                traj = trajectories[:, i, :].squeeze(1)
                waypoints[:, i, :, :], normal = self.trajectory_to_waypoints(pointcloud, traj, contact_point, expand_dim=True)
                # Adjust first waypoints based on normal, broadcast normal across waypoints
                waypoints[:, i, 0, :3] += 0.05 * normal.clone()

            action['trajectories'] = waypoints.clone()

            if self.use_pose_trajectory:
                pose_trajectory = self.poseTrajectoryGenerator.sample_trajectories(pointcloud, task, contact_point,
                                                                               num_trajectories=1)  # num_env, 1, 54
                pose_trajectory = pose_trajectory.squeeze(1)
                pose_waypoints, normal = self.trajectory_to_waypoints(pointcloud, pose_trajectory, contact_point,
                                                                      expand_dim=False, waypoint_dim=6)
                # Adjust first waypoints based on normal, broadcast normal across waypoints
                pose_waypoints[:, 0, :3] += 0.05 * normal.clone()
                action['pose_trajectory'] = pose_waypoints.clone()

        # set the action to be the waypoint at the current time step
        waypoint = self.waypoints[index, elapsed_steps.long()]
        action['action'] = waypoint.clone()

        for k in action.keys():
            if isinstance(action[k], torch.Tensor):
                action[k] = action[k].cpu().numpy()

        return action

    def construct_trajectory(self, down, forward, waypoints):
        # the trajectory must be of shape B x 30 ---> B x (3 X 3 X num_waypoints X 6)

        axis = torch.cat([forward, down], dim=1)
        trask_traj = torch.cat([axis, waypoints], dim=1)

        return trask_traj

    def extract_info_from_batch(self, sampled_batch):
        # infos: ['actual_target_motion', 'angle_dist', 'current_angle', 'elapsed_steps', 'gripper_forward_dir', 'gripper_up_dir', 'init_angle', 'init_contact_point_world', 'pointcloud', 'reward', 'success', 'target_angle', 'waypoints']

        # the trajectories we sample, have already been padded to all contain 4 transitions. The padding
        # is done during sampling from the replay buffer, see sampling class
        # padding is done by appending last transition to the trajectory until full length is reached.

        contact_point = sampled_batch['infos']['init_contact_point_base']  # batch x num_waypoints x 3
        contact_point = contact_point[:, 0, :]  # does not matter from what step to take the contact point --> batch x 3

        pointcloud = sampled_batch['infos']['pointcloud']  # batch x num_waypoints x n_points x 3
        pointcloud = pointcloud[:, 0, :, :]  # must use point cloud from very first step  --> batch x n_points x 3

        waypoints = sampled_batch['infos']['waypoints']  # batch x num_waypoints x (num_waypoints x action_size)
        waypoints = waypoints[:, -1, :]  # must take waypoint list from very last step, its already padded --> batch x (num_waypoints x action_size)

        # the gripper direction are in world frame which is identical to the base frame, because they are only translated
        # and not rotated
        down = sampled_batch['infos']['gripper_up_dir']  # batch x num_waypoints x 3
        down = down[:, 0, :]  # does not matter what step to take from

        forward = sampled_batch['infos']['gripper_forward_dir']  # batch x num_waypoints x 3
        forward = forward[:, 0, :]  # does not matter what step to take from

        trajectory = self.construct_trajectory(down, forward, waypoints)

        # target = sampled_batch['infos']['target_angle']  # batch x num_waypoints x 1
        init = sampled_batch['infos']['init_angle']  # batch x num_waypoints x 1
        current = sampled_batch['infos']['current_angle']

        # now I assume that the network will learn that it is a pulling or pushing task, based on the
        # initial position of the faucet handle given the pointcloud and contact point
        # the it need to learn that it has to move the handle by task amount starting from that initial position
        # task = target - init  # batch x num_waypoints x 1
        # task = task[:, 0, :]  # does not matter what step to take from  --> # batch x 1

        # the task motion is artificially set during data collection to either result in a positive or negative
        # sample, i.e. for positive samples task_motion = current_angle - init_angle --> success flag set to True
        # for negative samples task_motion = current_angle + offset - init_angle --> success flag set to False
        task_motion = sampled_batch['infos']['task_motion']
        task_motion = task_motion[:, -1, :]  # import to take last element here

        actual_motion = current - init  # same as task_motion for trajectories with success==True
        actual_motion = actual_motion[:, -1, :]  # import to take last element here

        success = sampled_batch['infos']['success']  # batch x num_waypoints x 1
        success = success[:, -1, :]  # need success flag from last step

        # flag whether the contact_point is a real contact point or was substituted with a point from static part of the object
        is_contact_point = sampled_batch['infos']['is_contact_point']
        is_contact_point = is_contact_point[:, -1, :]  # does not matter
        # success = success.astype(int)  # batch x 1
        # success_rate = torch.sum(success)/(success.shape[0]*success.shape[1])
        # self.logger.error(f'Success rate: {success_rate}')

        return pointcloud, task_motion, actual_motion, trajectory, contact_point, success, is_contact_point

    def compute_test_loss(self, memory):
        logger = get_logger()
        logger.info(f"Begin to compute test loss with batch size {self.batch_size}!")
        out_dict = dict()
        ret_dict = dict()
        num_samples = 0

        from maniskill2_learn.utils.meta import TqdmToLogger
        from tqdm import tqdm

        tqdm_obj = tqdm(total=memory.data_size, file=TqdmToLogger(), mininterval=20)

        for sampled_batch in memory.mini_batch_sampler(self.batch_size, drop_last=True, max_num_batches=3, auto_restart=True, traj_len=self.num_steps):
            sampled_batch = sampled_batch.to_torch(device="cuda", dtype="float32", non_blocking=True)

            pointcloud, task, actual_motion, trajectory, contact_point, success, is_contact_point = self.extract_info_from_batch(sampled_batch)

            if self.mode == 'trajectoryScorer':
                ret_dict = self.eval_trajectoryScorer(pointcloud, task, trajectory, contact_point, success, is_contact_point)
            elif self.mode == 'trajectoryGenerator':
                ret_dict = self.eval_trajectoryGenerator(pointcloud, actual_motion, trajectory, contact_point, is_contact_point)
            elif self.mode == 'affordancePredictor':
                if self.use_dataset:
                    ret_dict = self.eval_affordancePredictor_from_dataset(pointcloud, torch.sign(task), contact_point, success)
                else:
                    ret_dict = self.eval_affordancePredictor(pointcloud, actual_motion, contact_point, is_contact_point)
            else:
                raise NotImplementedError

            for key in ret_dict:
                out_dict[key] = out_dict.get(key, 0) + ret_dict[key] * len(sampled_batch)
            num_samples += len(sampled_batch)
            tqdm_obj.update(len(sampled_batch))

        logger.info(f"We compute the test loss over {num_samples} samples!")

        for key in out_dict:
            out_dict[key] /= num_samples

        return out_dict

    def update_parameters(self, memory, updates):
        batch_size = self.batch_size
        sampled_batch = memory.sample(batch_size, traj_len=self.num_steps).to_torch(device=self.device, non_blocking=True, dtype='float32')

        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype='float32', non_blocking=True)
        # self.logger.error(f'rank {self.world_rank}: sampled a batch')

        pointcloud, task, actual_motion, trajectory, contact_point, success, is_contact_point = self.extract_info_from_batch(sampled_batch)

        ret_dict = dict()

        if self.mode == 'trajectoryScorer':
            # trajectory scorer is trained to tell apart successful trajectories from unsuccessful ones
            # the key is that the majority of trajectories used, are indeed successful, but we augment the dataset
            # by setting the task to something different in hindsight and seeting success=False
            ret_dict = self.update_parameters_trajectoryScorer(pointcloud, task, trajectory, contact_point, success, is_contact_point)
        elif self.mode == 'trajectoryGenerator':
            # trajectory generator is trained mostly on successful trajectories. As the task we use the motion that the
            # faucet handle actually achieved. Thus by definition all trajectories achieve the task (not all were successful
            # on the real task).
            # the trajectory generator learns to generate trajectories that reach the motion provided as the task.
            ret_dict = self.update_parameters_trajectoryGenerator(pointcloud, actual_motion, trajectory, contact_point, is_contact_point)
        elif self.mode == 'affordancePredictor':
            if self.use_dataset:
                ret_dict = self.update_parameters_affordacePredictor_from_dataset(pointcloud, torch.sign(task), contact_point, success)
            else:
                # the affordance predictor is given the actual motion and shall predict if the contact point
                # is suitable to achieve that motion
                # the affordance predictor is always only given real contact points for training. Thus the predictor
                # never actually learns to handle points that are not on the movable object...
                # the affordance predictor simply learns whether the given contact point is suitable for that motion
                # it does not learn whether the point is a suitable contact point in the first place.
                # in the baseline the mask out all points that are not on the movable part... so they cheat
                ret_dict = self.update_parameters_affordancePredictor(pointcloud, actual_motion, contact_point, is_contact_point)
        else:
            raise NotImplementedError

        return ret_dict

    def eval_trajectoryScorer(self, pointcloud, task, trajectory, contact_point, success, is_contact_point):
        ret_dict = defaultdict(list)

        valid_mask = is_contact_point.squeeze(-1).bool()  # Ensure is_contact_point is of shape (batch_size, 1)
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]
        valid_success = success[valid_mask]

        with self.trajectoryScorer.no_sync():
            if valid_success.nelement() > 0:
                pred_result_logits, _ = self.trajectoryScorer(valid_pointcloud, valid_task, valid_trajectory, valid_contact_point)

                logit_threshold = np.log(0.5 / (1 - 0.5))  # convert from probability threshold to logit threshold

                def torch_accuracy_score(y_true, y_pred):
                    correct = torch.eq(y_true, y_pred).sum().float()
                    accuracy = correct / y_true.shape[0]
                    return accuracy

                def torch_precision_recall_f1_score(y_true, y_pred):
                    true_positives = torch.logical_and(y_true == 1, y_pred == 1).sum().float()
                    predicted_positives = y_pred.sum().float()
                    actual_positives = y_true.sum().float()

                    precision = true_positives / (predicted_positives + 1e-10)
                    recall = true_positives / (actual_positives + 1e-10)
                    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

                    return precision, recall, f1

                # Convert probabilities to binary predictions using 0.5 as the threshold
                binary_predictions = (pred_result_logits > logit_threshold).int()

                # Compute the metrics
                accuracy = torch_accuracy_score(valid_success, binary_predictions)
                precision, recall, f1 = torch_precision_recall_f1_score(valid_success, binary_predictions)

                # auc_roc = roc_auc_score(success, predictions)  # Use the predicted probabilities for AUC-ROC

                loss = self.BCEWithLogitsLoss(pred_result_logits, valid_success)
                loss = loss.mean()
                ret_dict["trajectoryScorer/test/bce_loss"] = loss.item()
                ret_dict["trajectoryScorer/test/accuracy"] = accuracy.item()
                ret_dict["trajectoryScorer/test/precision"] = precision.item()
                ret_dict["trajectoryScorer/test/recall"] = recall.item()
                ret_dict["trajectoryScorer/test/f1_score"] = f1.item()

            ret_dict = dict(ret_dict)

        return ret_dict

    def eval_trajectoryGenerator(self, pointcloud, task, trajectory, contact_point, is_contact_point):
        ret_dict = defaultdict(list)

        valid_mask = is_contact_point.squeeze(-1).bool()  # Ensure is_contact_point is of shape (batch_size, 1)
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]

        with self.trajectoryGenerator.no_sync():
            if valid_contact_point.nelement() > 0:
                reconstructed_traj, mu, logvar = self.trajectoryGenerator(valid_pointcloud, valid_task, valid_trajectory, valid_contact_point)

                wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory,
                                                                                           reconstructed_traj)

                kl_loss = self.KL_loss(mu, logvar)

                ret_dict["trajectoryGenerator/test/kl_loss"] = kl_loss.item()
                ret_dict["trajectoryGenerator/test/wp_pos_loss"] = wp_pos_loss.item()
                ret_dict["trajectoryGenerator/test/init_dir_loss"] = init_dir_loss.item()
                ret_dict["trajectoryGenerator/test/wp_dir_loss"] = wp_dir_loss.item()

                loss = kl_loss * self.trajectoryGenerator.lbd_kl + wp_pos_loss * self.trajectoryGenerator.lbd_recon_pos + wp_dir_loss * self.trajectoryGenerator.lbd_recon_dir + init_dir_loss * self.trajectoryGenerator.lbd_init_dir
                ret_dict["trajectoryGenerator/test/total_loss"] = loss.item()

            ret_dict = dict(ret_dict)

        return ret_dict

    def eval_affordancePredictor(self, pointcloud, task, contact_point, is_contact_point):

        ret_dict = defaultdict(list)
        bs = pointcloud.shape[0]

        trajs = self.trajectoryGenerator.sample_trajectories(pointcloud, task, contact_point, num_trajectories=self.num_trajectories_to_generate)
        gt_scores = self.trajectoryScorer.score_trajectories(pointcloud, task, trajs, contact_point)  # applies sigmoid internally
        topk = self.affordancePredictor.topk
        gt_scores = gt_scores.view(bs, self.num_trajectories_to_generate, 1).topk(k=topk, dim=1)[
            0].mean(dim=1).view(-1)
        contact_point_mask = is_contact_point.squeeze(-1)
        # Use the mask to set scores to 0 for non-contact points
        gt_scores = gt_scores * contact_point_mask + (1 - contact_point_mask) * 0
        predicted_score = self.affordancePredictor(pointcloud, task, contact_point)  # applies sigmoid internally # Rank 0 is stuck here
        loss = self.L1Loss(predicted_score, gt_scores).mean()
        ret_dict["affordancePredictor/test/loss"] = loss.item()
        ret_dict = dict(ret_dict)

        return ret_dict

    def eval_affordancePredictor_from_dataset(self, pointcloud, task, contact_point, success):

        ret_dict = defaultdict(list)

        with self.affordancePredictor.no_sync():
            predicted_score = self.affordancePredictor(pointcloud, task, contact_point)

            def torch_accuracy_score(y_true, y_pred):
                correct = torch.eq(y_true, y_pred).sum().float()
                accuracy = correct / y_true.shape[0]
                return accuracy

            def torch_precision_recall_f1_score(y_true, y_pred):
                true_positives = torch.logical_and(y_true == 1, y_pred == 1).sum().float()
                predicted_positives = y_pred.sum().float()
                actual_positives = y_true.sum().float()

                precision = true_positives / (predicted_positives + 1e-10)
                recall = true_positives / (actual_positives + 1e-10)
                f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

                return precision, recall, f1

            binary_scores = (predicted_score > 0.5).int()

            accuracy = torch_accuracy_score(success, binary_scores)
            precision, recall, f1 = torch_precision_recall_f1_score(success, binary_scores)

            loss = self.BCELoss(predicted_score, success)
            loss = loss.mean()
            ret_dict["affordancePredictor/test/loss"] = loss.item()
            ret_dict["affordancePredictor/test/accuracy"] = accuracy.item()
            ret_dict["affordancePredictor/test/precision"] = precision.item()
            ret_dict["affordancePredictor/test/recall"] = recall.item()
            ret_dict["affordancePredictor/test/f1_score"] = f1.item()
            ret_dict = dict(ret_dict)

        return ret_dict

    def update_parameters_affordacePredictor_from_dataset(self, pointcloud, task, contact_point, success):
        ret_dict = defaultdict(list)

        if self.affordancePredictor_lr_scheduler is not None:
            self.affordancePredictor_lr_scheduler.step()
            # self.logger.error(f'rank {self.world_rank}: stepped lr scheduler')

        self.affordancePredictor_optim.zero_grad()

        predicted_score = self.affordancePredictor(pointcloud, task, contact_point)  # applies sigmoid internally
        loss = self.BCELoss(predicted_score, success)
        loss = loss.mean()

        ret_dict["affordancePredictor/loss"] = loss.item()

        loss.backward()

        self.affordancePredictor_optim.step()
        # self.logger.error(f'rank {self.world_rank}: stepped optimizer')
        ret_dict["affordancePredictor/grad_norm"] = self.grad_norm

        if self.affordancePredictor_lr_scheduler is not None:
            ret_dict["affordancePredictor/lr"] = get_mean_lr(self.affordancePredictor_optim)

        ret_dict = dict(ret_dict)

        return ret_dict

    def update_parameters_affordancePredictor(self, pointcloud, task, contact_point, is_contact_point):
        ret_dict = defaultdict(list)
        bs = pointcloud.shape[0]
        # self.logger.error(f'rank {self.world_rank}: entered update parameters of affordance Predictor')

        if self.affordancePredictor_lr_scheduler is not None:
            self.affordancePredictor_lr_scheduler.step()
            # self.logger.error(f'rank {self.world_rank}: stepped lr scheduler')

        self.affordancePredictor_optim.zero_grad()
        # self.logger.error(f'rank {self.world_rank}: did zero grad on optimizer')

        with torch.no_grad():
            self.trajectoryGenerator.eval()
            trajs = self.trajectoryGenerator.sample_trajectories(pointcloud, task, contact_point, num_trajectories=self.num_trajectories_to_generate) # Rank 1 is stuck here
            # self.logger.error(f'rank {self.world_rank}: forward through Generator in update')
        with torch.no_grad():
            self.trajectoryScorer.eval()
            gt_scores = self.trajectoryScorer.score_trajectories(pointcloud, task, trajs, contact_point)  # applies sigmoid internally
            # self.logger.error(f'rank {self.world_rank}: forward through Scorer in update')
            topk = self.affordancePredictor.topk
            gt_scores = gt_scores.view(bs, self.num_trajectories_to_generate, 1).topk(k=topk, dim=1)[0].mean(dim=1).view(-1)
            # Adjust gt_scores based on is_contact_point
            # Convert is_contact_point to a mask with the same device as gt_scores
            contact_point_mask = is_contact_point.squeeze(-1)
            # Use the mask to set scores to 0 for non-contact points
            gt_scores = gt_scores * contact_point_mask + (1 - contact_point_mask) * 0


        predicted_score = self.affordancePredictor(pointcloud, task, contact_point) # applies sigmoid internally
        # self.logger.error(f'rank {self.world_rank}: forward through Affordance Predictor in update')
        loss = self.L1Loss(predicted_score, gt_scores).mean()

        if torch.isnan(loss).any():
            self.logger.error("NAN in loss")
            exit(0)

        ret_dict["affordancePredictor/loss"] = loss.item()

        loss.backward()
        # self.logger.error(f'rank {self.world_rank}: backwarded loss')
        self.affordancePredictor_optim.step()
        # self.logger.error(f'rank {self.world_rank}: stepped optimizer')

        ret_dict["affordancePredictor/grad_norm"] = self.grad_norm

        if self.affordancePredictor_lr_scheduler is not None:
            ret_dict["affordancePredictor/lr"] = get_mean_lr(self.affordancePredictor_optim)

        ret_dict = dict(ret_dict)

        return ret_dict

    def update_parameters_trajectoryGenerator(self, pointcloud, task, trajectory, contact_point, is_contact_point):
        ret_dict = defaultdict(list)

        # Filter samples where is_contact_point is True
        valid_mask = is_contact_point.squeeze(-1).bool()  # Ensure is_contact_point is of shape (batch_size, 1)
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]

        if self.trajectoryGenerator_lr_scheduler is not None:
            self.trajectoryGenerator_lr_scheduler.step()

        self.trajectoryGenerator_optim.zero_grad()
        if valid_pointcloud.nelement() > 0:
            reconstructed_traj, mu, logvar = self.trajectoryGenerator(valid_pointcloud, valid_task, valid_trajectory, valid_contact_point)

            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory,
                                                                                       reconstructed_traj)

            kl_loss = self.KL_loss(mu, logvar)

            ret_dict["trajectoryGenerator/kl_loss"] = kl_loss.item()
            ret_dict["trajectoryGenerator/wp_pos_loss"] = wp_pos_loss.item()
            ret_dict["trajectoryGenerator/init_dir_loss"] = init_dir_loss.item()
            ret_dict["trajectoryGenerator/wp_dir_loss"] = wp_dir_loss.item()

            loss = kl_loss * self.trajectoryGenerator.lbd_kl + wp_pos_loss * self.trajectoryGenerator.lbd_recon_pos + wp_dir_loss * self.trajectoryGenerator.lbd_recon_dir + init_dir_loss * self.trajectoryGenerator.lbd_init_dir
            ret_dict["trajectoryGenerator/total_loss"] = loss.item()

            if torch.isnan(loss).any():
                self.logger.error("NAN in loss")
                exit(0)

            loss.backward()

            # grad_norms = self.collect_gradient_norms(self.trajectoryGenerator)

            # ret_dict.update(grad_norms)

            self.trajectoryGenerator_optim.step()

        ret_dict["trajectoryGenerator/grad_norm"] = self.grad_norm

        if self.trajectoryGenerator_lr_scheduler is not None:
            ret_dict["trajectoryGenerator/lr"] = get_mean_lr(self.trajectoryGenerator_optim)

        ret_dict = dict(ret_dict)
        return ret_dict

    def update_parameters_trajectoryScorer(self, pointcloud, task, trajectory, contact_point, success, is_contact_point):

        ret_dict = defaultdict(list)

        # Filter samples where is_contact_point is True
        valid_mask = is_contact_point.squeeze(-1).bool()
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]
        valid_success = success[valid_mask]

        if self.trajectoryScorer_lr_scheduler is not None:
            self.trajectoryScorer_lr_scheduler.step()

        self.trajectoryScorer_optim.zero_grad()
        if valid_success.nelement() > 0:
            pred_result_logits, pred_whole_feats = self.trajectoryScorer(valid_pointcloud, valid_task, valid_trajectory, valid_contact_point)

            loss = self.BCEWithLogitsLoss(pred_result_logits, valid_success)

            loss = loss.mean()
            ret_dict["trajectoryScorer/bce_loss"] = loss.item()

            if torch.isnan(loss).any():
                self.logger.error("NAN in loss")
                exit(0)

            loss.backward()

            # grad_norms = self.collect_gradient_norms(self.trajectoryScorer)

            # ret_dict.update(grad_norms)

            self.trajectoryScorer_optim.step()

        ret_dict["trajectoryScorer/grad_norm"] = self.grad_norm

        if self.trajectoryScorer_lr_scheduler is not None:
            ret_dict["trajectoryScorer/lr"] = get_mean_lr(self.trajectoryScorer_optim)

        ret_dict = dict(ret_dict)

        return ret_dict


    def KL_loss(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        #if torch.isnan(loss).any():
            #self.logger.error("NAN in KL divergence term")
            # exit(0)
        loss = torch.mean(loss)
        return loss

    def bgs(self, d6s, **kwargs):
        bsz = d6s.shape[0]
        b1 = torch.nn.functional.normalize(d6s[:, :, 0], p=2, dim=1)  # forward
        a2 = d6s[:, :, 1]  # down
        b2 = torch.nn.functional.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b3 = torch.cross(b2, b1, dim=1)
        # THE NETWORK NEVER LEARNS TO PREDICT PERPENDICULAR AXES!! BECAUSE WE FORCE THEM TO BE PERPENDICULAR HERE!
        return torch.stack([b1, b3, b2], dim=1).permute(0, 2, 1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    def compute_reconstruction_loss(self, trajectory, reconstructed_traj):
        if not self.waypoint_dim == 4:
            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss_6dim(trajectory,
                                                                                            reconstructed_traj)
        else:
            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss_4dim(trajectory,
                                                                                            reconstructed_traj)
        return wp_pos_loss, wp_dir_loss, init_dir_loss

    def compute_reconstruction_loss_6dim(self, trajectory, reconstructed_traj):
        # self.num_steps+1 because we have num_steps waypoints and the direction vector
        bs = reconstructed_traj.shape[0]
        reconstructed_traj = reconstructed_traj.view((bs, self.num_steps+1, -1))
        trajectory = trajectory.view(reconstructed_traj.shape)
        recon_dir = reconstructed_traj[:, 0, :]
        recon_wps = reconstructed_traj[:, 1:, :]
        input_dir = trajectory[:, 0, :]
        input_wps = trajectory[:, 1:, :]

        recon_wps_pos = recon_wps[:, :, :3]
        recon_wps_dir = recon_wps[:, :, 3:]

        input_wps_pos = input_wps[:, :, :3]
        input_wps_dir = input_wps[:, :, 3:]

        wp_pos_loss = self.L1Loss(recon_wps_pos.reshape(bs, -1), input_wps_pos.reshape(bs, -1))
        wp_pos_loss = wp_pos_loss.mean()

        pred_Rs = self.bgs(recon_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='pred')  # 12, 3, 3  # creates rotation matrix from the two directions
        gt_Rs = self.bgs(input_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='gt')  # 12, 3, 3

        recon_wps_mat = axis_angle_to_matrix(recon_wps_dir)  # 12, 4, 3, 3
        input_wps_mat = axis_angle_to_matrix(input_wps_dir)  # 12, 4, 3, 3

        input_wps_mat = input_wps_mat.view(-1, 3, 3)
        recon_wps_mat =recon_wps_mat.view(-1, 3, 3)

        dir_loss = self.bgdR(gt_Rs, pred_Rs)

        wp_dir_loss = self.bgdR(input_wps_mat, recon_wps_mat)
        wp_dir_loss = wp_dir_loss.mean()
        init_dir_loss = dir_loss.mean()

        return wp_pos_loss, wp_dir_loss, init_dir_loss

    def compute_reconstruction_loss_4dim(self, trajectory, reconstructed_traj):
        # self.num_steps+1 because we have num_steps waypoints and the direction vector
        bs = reconstructed_traj.shape[0]

        recon_dir = reconstructed_traj[:, :6].view((bs, 1, -1))
        recon_wps = reconstructed_traj[:, 6:].view((bs, self.num_steps, -1))
        input_dir = trajectory[:, :6].view((bs, 1, -1))
        input_wps = trajectory[:, 6:].view((bs, self.num_steps, -1))

        recon_wps_pos = recon_wps[:, :, :3]
        recon_wps_dir = recon_wps[:, :, 3].view((bs, self.num_steps, -1))

        input_wps_pos = input_wps[:, :, :3]
        input_wps_dir = input_wps[:, :, 3].view((bs, self.num_steps, -1))

        wp_pos_loss = self.L1Loss(recon_wps_pos.reshape(bs, -1), input_wps_pos.reshape(bs, -1))
        wp_pos_loss = wp_pos_loss.mean()

        pred_Rs = self.bgs(recon_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='pred')  # 12, 3, 3  # creates rotation matrix from the two directions
        gt_Rs = self.bgs(input_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='gt')  # 12, 3, 3
        dir_loss = self.bgdR(gt_Rs, pred_Rs)
        init_dir_loss = dir_loss.mean()

        wp_dir_loss = self.L1Loss(recon_wps_dir.reshape(bs, -1),
                                  input_wps_dir.reshape(bs, -1))
        wp_dir_loss = wp_dir_loss.mean()

        return wp_pos_loss, wp_dir_loss, init_dir_loss
