import json
import logging
from maniskill2_learn.networks import build_model
from maniskill2_learn.utils.torch import build_optimizer, get_mean_lr, BaseAgent, BasePerception, load_checkpoint, freeze_params
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.meta import get_logger, get_world_rank, get_logger_name
from copy import deepcopy
from ..builder import MBRL
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
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


@MBRL.register_module()
class VAT_SAC_Perception(BasePerception):
    def __init__(self, affordance_predictor_cfg,
                 trajectory_generator_cfg,
                 batch_size=64,
                 env_params=None,
                 num_waypoints=4,  # will be overwritten in run_rl
                 waypoint_dim=6,  # will be overwritten in run_rl
                 train=False
                 ):

        super(VAT_SAC_Perception, self).__init__()
        log_level = logging.ERROR
        env_suffix = f""
        self.logger = get_logger("Agent-" + env_suffix, log_level=log_level)
        self.world_rank = get_world_rank()
        self.batch_size = batch_size

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

        # Trajectory Generator
        trajectory_generator_cfg = deepcopy(trajectory_generator_cfg)
        trajectory_generator_optim_cfg = trajectory_generator_cfg.pop('optim_cfg', None)
        trajectory_generator_lr_scheduler_cfg = trajectory_generator_cfg.pop('lr_scheduler_cfg', None)
        trajectory_generator_checkpoint_path = trajectory_generator_cfg.pop('trajectory_generator_checkpoint_path')
        self.poseTrajectoryGenerator = build_model(trajectory_generator_cfg)
        # we never train the trajectory generator
        if train and 1==2:
            assert trajectory_generator_optim_cfg is not None
            self.trajectoryGenerator_optim = build_optimizer(self.poseTrajectoryGenerator, trajectory_generator_optim_cfg)
            self.trajectoryGenerator_lr_scheduler = None
            if trajectory_generator_lr_scheduler_cfg:
                trajectory_generator_lr_scheduler_cfg['optimizer'] = self.trajectoryGenerator_optim
                self.trajectoryGenerator_lr_scheduler = build_lr_scheduler(trajectory_generator_lr_scheduler_cfg)

        if trajectory_generator_checkpoint_path != '':
            mapping = {r'^trajectoryGenerator\.': ''}
            load_checkpoint(self.poseTrajectoryGenerator, trajectory_generator_checkpoint_path, 'cuda',
                            logger=get_logger(), keys_map=mapping)

        # Affordance Predictor
        affordance_predictor_cfg = deepcopy(affordance_predictor_cfg)
        affordance_predictor_optim_cfg = affordance_predictor_cfg.pop('optim_cfg', None)
        affordance_predictor_lr_scheduler_cfg = affordance_predictor_cfg.pop('lr_scheduler_cfg', None)
        affordance_predictor_checkpoint_path = affordance_predictor_cfg.pop('affordance_predictor_checkpoint_path')
        self.affordancePredictor = build_model(affordance_predictor_cfg)
        if train:
            assert affordance_predictor_optim_cfg is not None
            self.affordancePredictor_optim = build_optimizer(self.affordancePredictor, affordance_predictor_optim_cfg)
            self.affordancePredictor_lr_scheduler = None
            if affordance_predictor_lr_scheduler_cfg:
                affordance_predictor_lr_scheduler_cfg['optimizer'] = self.affordancePredictor_optim
                self.affordancePredictor_lr_scheduler = build_lr_scheduler(affordance_predictor_lr_scheduler_cfg)

        if affordance_predictor_checkpoint_path != '':
            mapping = {r'^affordancePredictor\.': ''}
            load_checkpoint(self.affordancePredictor, affordance_predictor_checkpoint_path, 'cuda', logger=get_logger(),
                            keys_map=mapping)

        self.BCELoss = torch.nn.BCELoss(reduction='none')
        self.L1Loss = torch.nn.L1Loss(reduction='none')

    def construct_trajectory(self, down, forward, tcp_pose):
        # the trajectory must be of shape B x (3 + 3 + num_waypoints X 6)
        positions = tcp_pose[..., :3]
        quaternions = tcp_pose[..., 3:]  # wxyz convention
        rotations = quaternion_to_axis_angle(quaternions)
        # Concatenate positions and rotations
        concatenated = torch.cat((positions, rotations), dim=-1)  # Shape: [batch, waypoints, 6]
        # Flatten the last two dimensions
        flattened = concatenated.view(concatenated.shape[0], -1)  # Shape: [batch, waypoints*6]
        axis = torch.cat([forward, down], dim=1)
        trask_traj = torch.cat([axis, flattened], dim=1) # Shape: [batch, (waypoints+1)*6]

        return trask_traj

    def sub_sample(self, traj):
        # traj of shape B, traj_len
        if traj.shape[1] == (3+3+self.num_steps*self.waypoint_dim):
            return traj

        if traj.shape[1] < (3+3+self.num_steps*self.waypoint_dim):
            return F.pad(traj, (0, 30 - traj.shape[1]))

        else:
            B, L = traj.shape
            # need to sub sample
            # first 6 elements need to be preserved
            actual_num_waypoints = int(L / self.waypoint_dim)
            desired_num_waypoints = self.num_steps + 1

            if actual_num_waypoints % desired_num_waypoints == 0:
                stride = int(actual_num_waypoints / desired_num_waypoints)
                waypoint_indices = torch.arange(0, desired_num_waypoints * stride, stride) * self.waypoint_dim
                """tensor([ 0, 12, 24, 36, 48, 60, 72, 84, 96])"""

                # base_indices look like this: tensor([ 0, 12, 24, 36, 48, 60, 72, 84, 96])
                # need them to be like this: [0,1,2,3,4,5, 12,13,14,15,16,17, etc.]
                offset = torch.arange(self.waypoint_dim).repeat(desired_num_waypoints, 1)
                """tensor([[0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5]])"""

                indices = (waypoint_indices.unsqueeze(1) + offset).view(
                    -1)  ## tensor([  0,   1,   2,   3,   4,   5,  12,  13,  14,  15,  16,  17,  24,  25,26,  27,  28,  29,  36,  37,  38,  39,  40,  41,  48,  49,  50,  51, 52,  53,  60,  61,  62,  63,  64,  65,  72,  73,  74,  75,  76,  77, 84, 85, 86, 87, 88,  89,  96,  97,  98,  99, 100, 101])
                """tensor([  0,   1,   2,   3,   4,   5,  12,  13,  14,  15,  16,  17,  24,  25,
                 26,  27,  28,  29,  36,  37,  38,  39,  40,  41,  48,  49,  50,  51,
                 52,  53,  60,  61,  62,  63,  64,  65,  72,  73,  74,  75,  76,  77,
                 84,  85,  86,  87,  88,  89,  96,  97,  98,  99, 100, 101])"""

                sub_sampled_traj = traj[:, indices].reshape(B, -1)

            else:
                # Calculate floating-point indices to represent the ideal distribution of waypoints
                floating_indices = torch.linspace(0, actual_num_waypoints - 1, steps=desired_num_waypoints)
                """tensor([ 0.0000,  4.1250,  8.2500, 12.3750, 16.5000, 20.6250, 24.7500, 28.8750,
                33.0000])"""
                # Round these indices to the nearest integer to use as valid indices
                waypoint_indices = torch.round(floating_indices).long()
                """tensor([ 0,  4,  8, 12, 16, 21, 25, 29, 33])"""

                # Get indices of the start of each waypoint
                waypoint_starts = torch.arange(0, actual_num_waypoints) * self.waypoint_dim
                """tensor([  0,   6,  12,  18,  24,  30,  36,  42,  48,  54,  60,  66,  72,  78,
                 84,  90,  96, 102, 108, 114, 120, 126, 132, 138, 144, 150, 156, 162,
                168, 174, 180, 186, 192, 198])"""

                # Get the actual starting indices for these waypoints
                indices = waypoint_starts[waypoint_indices]
                """tensor([  0,  24,  48,  72,  96, 126, 150, 174, 198])"""
                # Adjust indices to account for each waypoint having multiple elements
                # We replicate the rounded indices for each element in the waypoint
                offset = torch.arange(self.waypoint_dim).repeat(desired_num_waypoints, 1)
                """tensor([[0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5],
                [0, 1, 2, 3, 4, 5]])"""

                indices = (indices.unsqueeze(1) + offset).view(-1)
                """tensor([  0,   1,   2,   3,   4,   5,  24,  25,  26,  27,  28,  29,  48,  49,
                 50,  51,  52,  53,  72,  73,  74,  75,  76,  77,  96,  97,  98,  99,
                100, 101, 126, 127, 128, 129, 130, 131, 150, 151, 152, 153, 154, 155,
                174, 175, 176, 177, 178, 179, 198, 199, 200, 201, 202, 203])"""

                # Clip indices to ensure they do not go out of the bounds of the original tensor's length
                indices = torch.clamp(indices, 0, L - 1)

                # Use the calculated indices to select elements from the trajectory
                sub_sampled_traj = traj[:, indices].view(B, -1)
            return sub_sampled_traj

    def extract_info_from_batch(self, sampled_batch):

        contact_point = sampled_batch['infos']['init_contact_point_base']  # batch x num_waypoints x 3
        contact_point = contact_point[:, 0, :]  # does not matter from what step to take the contact point --> batch x 3

        pointcloud = sampled_batch['obs']['xyz']  # batch x num_waypoints x n_points x 3
        pointcloud = pointcloud[:, 0, :, :]  # must use point cloud from very first step  --> batch x n_points x 3

        tcp_pose = sampled_batch['infos']['tcp_pose']  # batch x num_waypoints x 7 (3 pos dim, 4 quaternion)

        down = sampled_batch['infos']['gripper_up_dir']  # batch x num_waypoints x 3
        down = down[:, 0, :]  # does not matter what step to take from

        forward = sampled_batch['infos']['gripper_forward_dir']  # batch x num_waypoints x 3
        forward = forward[:, 0, :]  # does not matter what step to take from

        trajectory = self.construct_trajectory(down, forward, tcp_pose)

        trajectory = self.sub_sample(trajectory)

        init = sampled_batch['infos']['init_angle']  # batch x num_waypoints x 1
        current = sampled_batch['infos']['current_angle']
        target = sampled_batch['infos']['target_angle']

        task_motion = target - init
        task_motion = task_motion[:, 0, :]

        actual_motion = current - init  # same as task_motion for trajectories with success==True
        actual_motion = actual_motion[:, -1, :]  # import to take last element here

        success = sampled_batch['infos']['success']  # batch x num_waypoints x 1
        success = success[:, -1, :]  # need success flag from last step

        # is_contact_point = sampled_batch['infos']['is_contact_point']
        # is_contact_point = is_contact_point[:, -1, :]  # does not matter

        return pointcloud, task_motion, actual_motion, trajectory, contact_point, success #, is_contact_point

    def compute_test_loss(self, memory):
        logger = get_logger()
        logger.info(f"Begin to compute test loss with batch size {self.batch_size}!")
        out_dict = dict()
        num_samples = 0

        from maniskill2_learn.utils.meta import TqdmToLogger
        from tqdm import tqdm

        tqdm_obj = tqdm(total=memory.data_size, file=TqdmToLogger(), mininterval=20)

        for sampled_batch in memory.mini_batch_sampler(self.batch_size, drop_last=True, max_num_batches=3, auto_restart=True, traj_len=self.num_steps):
            sampled_batch = sampled_batch.to_torch(device="cuda", dtype="float32", non_blocking=True)

            pointcloud, task, actual_motion, trajectory, contact_point, success= self.extract_info_from_batch(sampled_batch)

            valid_mask = success.squeeze(-1).bool()
            valid_pointcloud = pointcloud[valid_mask]
            valid_task = task[valid_mask]
            valid_actual_motion = actual_motion[valid_mask]
            valid_trajectory = trajectory[valid_mask]
            valid_contact_point = contact_point[valid_mask]

            # ret_dict = self.eval_trajectoryGenerator(valid_pointcloud, valid_actual_motion, valid_trajectory, valid_contact_point)

            ret_dict = self.eval_affordancePredictor_from_dataset(pointcloud, torch.sign(task), contact_point, success)

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
        try:
            sampled_batch = memory.sample(batch_size, traj_len=80).to_torch(device=self.device, non_blocking=True, dtype='float32')
        except Exception as e:
            print('stop')
        # sampled_batch = sampled_batch.to_torch(device=self.device, dtype='float32', non_blocking=True)
        # self.logger.error(f'rank {self.world_rank}: sampled a batch')

        pointcloud, task, actual_motion, trajectory, contact_point, success = self.extract_info_from_batch(sampled_batch)

        valid_mask = success.squeeze(-1).bool()
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_actual_motion = actual_motion[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]

        #ret_dict = self.update_parameters_trajectoryGenerator(valid_pointcloud, valid_actual_motion, valid_trajectory, valid_contact_point, valid_is_contact_point)
        ret_dict = dict()
        ret_dict = self.update_parameters_affordancePredictor_from_dataset(pointcloud, torch.sign(task), contact_point, success)

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

    def update_parameters_affordancePredictor_from_dataset(self, pointcloud, task, contact_point, success):
        ret_dict = defaultdict(list)

        if self.affordancePredictor_lr_scheduler is not None:
            self.affordancePredictor_lr_scheduler.step()
            # self.logger.error(f'rank {self.world_rank}: stepped lr scheduler')

        self.affordancePredictor_optim.zero_grad()

        '''predicted_score = self.affordancePredictor(pointcloud, task, contact_point)  # applies sigmoid internally
        loss = self.BCELoss(predicted_score, success)
        loss = loss.mean()

        ret_dict["affordancePredictor/loss"] = loss.item()'''
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
        ret_dict["affordancePredictor/loss"] = loss.item()
        ret_dict["affordancePredictor/accuracy"] = accuracy.item()
        ret_dict["affordancePredictor/precision"] = precision.item()
        ret_dict["affordancePredictor/recall"] = recall.item()
        ret_dict["affordancePredictor/f1_score"] = f1.item()

        loss.backward()

        self.affordancePredictor_optim.step()
        # self.logger.error(f'rank {self.world_rank}: stepped optimizer')
        ret_dict["affordancePredictor/grad_norm"] = self.grad_norm

        if self.affordancePredictor_lr_scheduler is not None:
            ret_dict["affordancePredictor/lr"] = get_mean_lr(self.affordancePredictor_optim)

        ret_dict = dict(ret_dict)

        return ret_dict

    def eval_trajectoryGenerator(self, pointcloud, task, trajectory, contact_point, is_contact_point):
        ret_dict = defaultdict(list)

        valid_mask = is_contact_point.squeeze(-1).bool()  # Ensure is_contact_point is of shape (batch_size, 1)
        valid_pointcloud = pointcloud[valid_mask]
        valid_task = task[valid_mask]
        valid_trajectory = trajectory[valid_mask]
        valid_contact_point = contact_point[valid_mask]

        with self.poseTrajectoryGenerator.no_sync():
            if valid_contact_point.nelement() > 0:
                reconstructed_traj, mu, logvar = self.poseTrajectoryGenerator(valid_pointcloud, valid_task,
                                                                          valid_trajectory, valid_contact_point)

                wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory,
                                                                                           reconstructed_traj)

                kl_loss = self.KL_loss(mu, logvar)

                ret_dict["trajectoryGenerator/test/kl_loss"] = kl_loss.item()
                ret_dict["trajectoryGenerator/test/wp_pos_loss"] = wp_pos_loss.item()
                ret_dict["trajectoryGenerator/test/init_dir_loss"] = init_dir_loss.item()
                ret_dict["trajectoryGenerator/test/wp_dir_loss"] = wp_dir_loss.item()

                loss = kl_loss * self.poseTrajectoryGenerator.lbd_kl + wp_pos_loss * self.poseTrajectoryGenerator.lbd_recon_pos + wp_dir_loss * self.poseTrajectoryGenerator.lbd_recon_dir + init_dir_loss * self.poseTrajectoryGenerator.lbd_init_dir
                ret_dict["trajectoryGenerator/test/total_loss"] = loss.item()

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
            reconstructed_traj, mu, logvar = self.poseTrajectoryGenerator(valid_pointcloud, valid_task, valid_trajectory,
                                                                      valid_contact_point)

            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory,
                                                                                       reconstructed_traj)

            kl_loss = self.KL_loss(mu, logvar)

            ret_dict["trajectoryGenerator/kl_loss"] = kl_loss.item()
            ret_dict["trajectoryGenerator/wp_pos_loss"] = wp_pos_loss.item()
            ret_dict["trajectoryGenerator/init_dir_loss"] = init_dir_loss.item()
            ret_dict["trajectoryGenerator/wp_dir_loss"] = wp_dir_loss.item()

            loss = kl_loss * self.poseTrajectoryGenerator.lbd_kl + wp_pos_loss * self.poseTrajectoryGenerator.lbd_recon_pos + wp_dir_loss * self.poseTrajectoryGenerator.lbd_recon_dir + init_dir_loss * self.poseTrajectoryGenerator.lbd_init_dir
            ret_dict["trajectoryGenerator/total_loss"] = loss.item()

            if torch.isnan(loss).any():
                self.logger.error("NAN in loss")
                exit(0)

            loss.backward()

            self.trajectoryGenerator_optim.step()

        ret_dict["trajectoryGenerator/grad_norm"] = self.grad_norm

        if self.trajectoryGenerator_lr_scheduler is not None:
            ret_dict["trajectoryGenerator/lr"] = get_mean_lr(self.trajectoryGenerator_optim)

        ret_dict = dict(ret_dict)
        return ret_dict

    def KL_loss(self, mu, logvar):
        mu = mu.view(mu.shape[0], -1)
        logvar = logvar.view(logvar.shape[0], -1)
        loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
        # if torch.isnan(loss).any():
        # self.logger.error("NAN in KL divergence term")
        # exit(0)
        loss = torch.mean(loss)
        return loss

    def bgs(self, d6s, **kwargs):
        bsz = d6s.shape[0]
        b1 = torch.nn.functional.normalize(d6s[:, :, 0], p=2, dim=1)  # forward
        a2 = d6s[:, :, 1]  # down
        b2 = torch.nn.functional.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1,
                                           p=2, dim=1)
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
        reconstructed_traj = reconstructed_traj.view((bs, self.num_steps + 1, -1))
        trajectory = trajectory.view(reconstructed_traj.shape)
        recon_dir = reconstructed_traj[:, 0, :]
        recon_wps = reconstructed_traj[:, 1:, :]
        input_dir = trajectory[:, 0, :]
        input_wps = trajectory[:, 1:, :]

        recon_wps_pos = recon_wps[:, :, :3]
        recon_wps_dir = recon_wps[:, :, 3:]

        input_wps_pos = input_wps[:, :, :3]
        input_wps_dir = input_wps[:, :, 3:]

        wp_pos_loss = self.L1Loss(recon_wps_pos.reshape(bs, self.num_steps * 3),
                                  input_wps_pos.reshape(bs, self.num_steps * 3))
        wp_pos_loss = wp_pos_loss.mean()

        pred_Rs = self.bgs(recon_dir.reshape(-1, 2, 3).permute(0, 2, 1),
                           case='pred')  # 12, 3, 3  # creates rotation matrix from the two directions
        gt_Rs = self.bgs(input_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='gt')  # 12, 3, 3

        recon_wps_mat = axis_angle_to_matrix(recon_wps_dir)  # 12, 4, 3, 3
        input_wps_mat = axis_angle_to_matrix(input_wps_dir)  # 12, 4, 3, 3

        input_wps_mat = input_wps_mat.view(-1, 3, 3)
        recon_wps_mat = recon_wps_mat.view(-1, 3, 3)

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

        pred_Rs = self.bgs(recon_dir.reshape(-1, 2, 3).permute(0, 2, 1),
                           case='pred')  # 12, 3, 3  # creates rotation matrix from the two directions
        gt_Rs = self.bgs(input_dir.reshape(-1, 2, 3).permute(0, 2, 1), case='gt')  # 12, 3, 3
        dir_loss = self.bgdR(gt_Rs, pred_Rs)
        init_dir_loss = dir_loss.mean()

        wp_dir_loss = self.L1Loss(recon_wps_dir.reshape(bs, -1),
                                  input_wps_dir.reshape(bs, -1))
        wp_dir_loss = wp_dir_loss.mean()

        return wp_pos_loss, wp_dir_loss, init_dir_loss
