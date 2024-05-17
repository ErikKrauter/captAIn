from maniskill2_learn.networks import build_model
from maniskill2_learn.utils.torch import build_optimizer, get_mean_lr, BasePerception, load_checkpoint, freeze_params
from maniskill2_learn.schedulers import build_lr_scheduler
from maniskill2_learn.utils.meta import get_logger, get_world_rank
from copy import deepcopy
from ..builder import MBRL
from collections import defaultdict
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from maniskill2_learn.utils.data import GDict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pytorch3d.transforms import axis_angle_to_matrix


'''
This class is basically the same as the regular Trajectory Generator, just the this time the objective is 
to reconstruct the TCP poses instead of the action sequence. Ths Pose Trajectory Generator is used in captAIn later.
'''

# this function expects the quaternion to be in WXYZ format
# this format aligns with SAPIEN
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

@MBRL.register_module()
class PoseTrajectoryGenerator(BasePerception):
    def __init__(self, trajectory_generator_cfg,
                 batch_size=1024,
                 env_params=None,
                 num_traj=100,
                 num_waypoints=4,  # will be overwritten in run_rl
                 waypoint_dim=6,  # will be overwritten in run_rl
                 mode="trajectoryGenerator",
                 **kwargs
                 ):
        super(PoseTrajectoryGenerator, self).__init__()
        self.logger = get_logger()
        self.world_rank = get_world_rank()
        self.batch_size = batch_size

        if env_params is not None:
            # for inference, the action shape is given by environment
            # it must align with the environment config that the RL agent was trained on
            self.waypoint_dim = env_params['action_shape']
            # self.waypoint_dim = self.waypoint_dim-1 if self.waypoint_dim == 7 else self.waypoint_dim
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
        self.L1Loss = torch.nn.L1Loss(reduction='none')

        # need to propagate num_traj, num_waypoints, waypoint_dim, feat_dim to the other configs
        # because the influence the construction of MLPs

        trajectory_generator_cfg = deepcopy(trajectory_generator_cfg)
        trajectory_generator_optim_cfg = trajectory_generator_cfg.pop('optim_cfg')
        trajectory_generator_lr_scheduler_cfg = trajectory_generator_cfg.pop('lr_scheduler_cfg', None)

        self.trajectoryGenerator = build_model(trajectory_generator_cfg)
        self.trajectoryGenerator_optim = build_optimizer(self.trajectoryGenerator, trajectory_generator_optim_cfg)
        self.trajectoryGenerator_lr_scheduler = None
        if trajectory_generator_lr_scheduler_cfg:
            trajectory_generator_lr_scheduler_cfg['optimizer'] = self.trajectoryGenerator_optim
            self.trajectoryGenerator_lr_scheduler = build_lr_scheduler(trajectory_generator_lr_scheduler_cfg)


    def extract_info_from_batch(self, sampled_batch):

        contact_point = sampled_batch['infos']['init_contact_point_base']  # batch x num_waypoints x 3
        contact_point = contact_point[:, 0, :]  # does not matter from what step to take the contact point --> batch x 3

        pointcloud = sampled_batch['infos']['pointcloud']  # batch x num_waypoints x n_points x 3
        pointcloud = pointcloud[:, 0, :, :]  # must use point cloud from very first step  --> batch x n_points x 3

        tcp_pose = sampled_batch['infos']['tcp_pose']  # batch x num_waypoints x 7 (3 pos dim, 4 quaternion)

        down = sampled_batch['infos']['gripper_up_dir']  # batch x num_waypoints x 3
        down = down[:, 0, :]  # does not matter what step to take from

        forward = sampled_batch['infos']['gripper_forward_dir']  # batch x num_waypoints x 3
        forward = forward[:, 0, :]  # does not matter what step to take from

        trajectory = self.construct_trajectory(down, forward, tcp_pose)

        init = sampled_batch['infos']['init_angle']  # batch x num_waypoints x 1
        current = sampled_batch['infos']['current_angle']

        task_motion = sampled_batch['infos']['task_motion']
        task_motion = task_motion[:, -1, :]  # import to take last element here

        actual_motion = current - init  # same as task_motion for trajectories with success==True
        actual_motion = actual_motion[:, -1, :]  # import to take last element here

        success = sampled_batch['infos']['success']  # batch x num_waypoints x 1
        success = success[:, -1, :]  # need success flag from last step

        is_contact_point = sampled_batch['infos']['is_contact_point']
        is_contact_point = is_contact_point[:, -1, :]  # does not matter

        return pointcloud, task_motion, actual_motion, trajectory, contact_point, success, is_contact_point


    def construct_trajectory(self, down, forward, tcp_pose):
        # the trajectory must be of shape B x 30 ---> B x (3 X 3 X num_waypoints X 6)
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

            pointcloud, task, actual_motion, trajectory, contact_point, success, is_contact_point = self.extract_info_from_batch(sampled_batch)

            ret_dict = self.eval_trajectoryGenerator(pointcloud, actual_motion, trajectory, contact_point, is_contact_point)

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

        pointcloud, task, actual_motion, trajectory, contact_point, success, is_contact_point = self.extract_info_from_batch(sampled_batch)

        ret_dict = self.update_parameters_trajectoryGenerator(pointcloud, actual_motion, trajectory, contact_point, is_contact_point)

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

                wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory, reconstructed_traj)

                kl_loss = self.KL_loss(mu, logvar)

                ret_dict["trajectoryGenerator/test/kl_loss"] = kl_loss.item()
                ret_dict["trajectoryGenerator/test/wp_pos_loss"] = wp_pos_loss.item()
                ret_dict["trajectoryGenerator/test/init_dir_loss"] = init_dir_loss.item()
                ret_dict["trajectoryGenerator/test/wp_dir_loss"] = wp_dir_loss.item()

                loss = kl_loss * self.trajectoryGenerator.lbd_kl + wp_pos_loss * self.trajectoryGenerator.lbd_recon_pos + wp_dir_loss * self.trajectoryGenerator.lbd_recon_dir + init_dir_loss * self.trajectoryGenerator.lbd_init_dir
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
            reconstructed_traj, mu, logvar = self.trajectoryGenerator(valid_pointcloud, valid_task, valid_trajectory, valid_contact_point)

            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss(valid_trajectory, reconstructed_traj)

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
            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss_6dim(trajectory, reconstructed_traj)
        else:
            wp_pos_loss, wp_dir_loss, init_dir_loss = self.compute_reconstruction_loss_4dim(trajectory, reconstructed_traj)
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

        wp_pos_loss = self.L1Loss(recon_wps_pos.reshape(bs, self.num_steps * 3),
                              input_wps_pos.reshape(bs, self.num_steps * 3))
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
