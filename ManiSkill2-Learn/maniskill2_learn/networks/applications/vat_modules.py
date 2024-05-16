from gymnasium.spaces import Discrete, Box
from maniskill2_learn.utils.data import to_torch, GDict, DictArray, recover_with_mask
from maniskill2_learn.utils.torch import ExtendedModule, avg_grad
from ..builder import VATMARTNETWORKS, build_backbone, build_reg_head
import torch.nn.functional as F
from maniskill2_learn.utils.meta import get_logger
from ..utils import replace_placeholder_with_args, get_kwargs_from_shape, combine_obs_with_action
import torch, torch.nn as nn


class VATModelBase(ExtendedModule):
    def __init__(self, backbone_cfg=None, head_cfg=None, mlp_cp_cfg=None, mlp_traj_cfg=None, mlp_task_cfg=None):
        super(VATModelBase, self).__init__()
        self.backbone = build_backbone(backbone_cfg)  # pointnet2
        self.mlp_cp = build_backbone(mlp_cp_cfg)  # LinearMLP 1 layer
        self.mlp_traj = build_backbone(mlp_traj_cfg)  # LinearMLP 1 layer
        self.mlp_task = build_backbone(mlp_task_cfg)  # LinearMLP 1 layer
        self.head = build_backbone(head_cfg)  # LinearMLP

    def forward(self, point_cloud, task, traj, contact_point, n=1):

        batch_size = point_cloud.shape[0]
        features = []

        # dim contact point batch x 3
        # dim of point cloud batch x n_points x 3
        point_cloud[:, 0, :] = contact_point  # I think this is done to make sure that the contact point is in the pc
        # in VAT they only have xyz features in point cloud so the repeat them to have 6D points to work with pn2
        # point_cloud = point_cloud.repeat(1, 1, 2)
        #  during training of affordance module we need to forward multiple trajectories at once thru the
        #  trajectory scoring module. We take as input n many trajectories
        #  thus we need to expand the dimension of the task and contact point to accomodate for it
        #  we will not do it for pointcloud because it will be too large for cuda memory
        #  instead we expand the features of the pointcloud oce it was passed thru
        if n > 1:
            task = task.unsqueeze(dim=1).repeat(1, n, 1).view(batch_size * n, -1)
            contact_point = contact_point.unsqueeze(dim=1).repeat(1, n, 1).view(batch_size * n, -1)

        whole_pointcloud_features = self.backbone(point_cloud)  # B, 128, 1200
        '''get_logger().info(
            f'mean(whole_pointcloud_features): {torch.mean(torch.flatten(whole_pointcloud_features))}')
        get_logger().info(
            f'std(whole_pointcloud_features): {torch.std(torch.flatten(whole_pointcloud_features))}')
        get_logger().info(
            f'min(whole_pointcloud_features): {min(torch.flatten(whole_pointcloud_features))}')
        get_logger().info(
            f'max(whole_pointcloud_features): {max(torch.flatten(whole_pointcloud_features))}')'''

        pn2_contact_point_features = whole_pointcloud_features[:, :, 0]  # B x 128
        '''get_logger().info(
            f'mean(pn2_contact_point_features): {torch.mean(torch.flatten(pn2_contact_point_features))}')
        get_logger().info(
            f'std(pn2_contact_point_features): {torch.std(torch.flatten(pn2_contact_point_features))}')
        get_logger().info(
            f'min(pn2_contact_point_features): {min(torch.flatten(pn2_contact_point_features))}')
        get_logger().info(
            f'max(pn2_contact_point_features): {max(torch.flatten(pn2_contact_point_features))}')'''
        if n > 1:
            pn2_contact_point_features = pn2_contact_point_features.unsqueeze(dim=1).repeat(1, n, 1).view(batch_size * n, -1)
        features.append(pn2_contact_point_features)

        if self.mlp_task is not None:
            task_features = self.mlp_task(task)  # B, 32
            features.append(task_features)
            '''get_logger().info(
                f'mean(task_features): {torch.mean(torch.flatten(task_features))}')
            get_logger().info(
                f'std(task_features): {torch.std(torch.flatten(task_features))}')
            get_logger().info(
                f'min(task_features): {min(torch.flatten(task_features))}')
            get_logger().info(
                f'max(task_features): {max(torch.flatten(task_features))}')'''
        if self.mlp_traj is not None:
            if traj.ndim == 3:
                traj = traj.view(batch_size * n, -1)
            traj_features = self.mlp_traj(traj)  # B, 256
            features.append(traj_features)
            '''get_logger().info(
                f'mean(traj_features): {torch.mean(torch.flatten(traj_features))}')
            get_logger().info(
                f'std(traj_features): {torch.std(torch.flatten(traj_features))}')
            get_logger().info(
                f'min(traj_features): {min(torch.flatten(traj_features))}')
            get_logger().info(
                f'max(traj_features): {max(torch.flatten(traj_features))}')'''

        if self.mlp_cp is not None:
            cp_features = self.mlp_cp(contact_point)  # B, 32
            features.append(cp_features)
            '''get_logger().info(
                f'mean(cp_features): {torch.mean(torch.flatten(cp_features))}')
            get_logger().info(
                f'std(cp_features): {torch.std(torch.flatten(cp_features))}')
            get_logger().info(
                f'min(cp_features): {min(torch.flatten(cp_features))}')
            get_logger().info(
                f'max(cp_features): {max(torch.flatten(cp_features))}')'''


        # [pn2_contact_point_features, task_features, traj_features, cp_features]
        # all_features = torch.cat(features, dim=-1)
        if self.head is not None:
            # in this case the head is a cVAE, meaning its the trajectory generation module
            if hasattr(self.head, 'encoder'):
                all_features = torch.cat(features, dim=-1)
                # the decoder only get the condition as input, which shall not contain trajectory features
                condition = torch.cat([features[0], features[1], features[3]], dim=-1)
                reconstructed_waypoints, mu, logvar  = self.head(all_features, condition)
                return reconstructed_waypoints, mu, logvar
            else:
                all_features = torch.cat(features, dim=-1)
                pred_result_logits = self.head(all_features)
                '''get_logger().info(f'mean(all_features): {torch.mean(torch.flatten(all_features))}')
                get_logger().info(f'std(all_features): {torch.std(torch.flatten(all_features))}')
                get_logger().info(f'min(all_features): {min(torch.flatten(all_features))}')
                get_logger().info(f'max(all_features): {max(torch.flatten(all_features))}')'''
        else:
            pred_result_logits = None

        return pred_result_logits, whole_pointcloud_features

@VATMARTNETWORKS.register_module()
class TrajectoryScorer(VATModelBase):
    def __init__(self, backbone_cfg=None, head_cfg=None, mlp_cp_cfg=None, mlp_traj_cfg=None, mlp_task_cfg=None):
        super(TrajectoryScorer, self).__init__(backbone_cfg=backbone_cfg, head_cfg=head_cfg, mlp_cp_cfg=mlp_cp_cfg,
                                               mlp_task_cfg=mlp_task_cfg, mlp_traj_cfg=mlp_traj_cfg)

    def score_trajectories(self, pointcloud, task, traj, contact_point):
        num_traj = traj.shape[1]
        score, _ = self.forward(pointcloud, task, traj, contact_point, n=num_traj)
        score = score.view(-1, num_traj, 1)
        return torch.sigmoid(score)




@VATMARTNETWORKS.register_module()
class AffordancePredictor(VATModelBase):
    def __init__(self, backbone_cfg=None, head_cfg=None, mlp_cp_cfg=None, mlp_traj_cfg=None, mlp_task_cfg=None, topk=5):
        super(AffordancePredictor, self).__init__(backbone_cfg=backbone_cfg, head_cfg=head_cfg, mlp_cp_cfg=mlp_cp_cfg,
                                                  mlp_task_cfg=mlp_task_cfg, mlp_traj_cfg=mlp_traj_cfg)
        self.topk = topk

    def forward(self, pointcloud, task, contact_point):

        pred_logits, _ = super().forward(pointcloud, task, None, contact_point)
        return torch.sigmoid(pred_logits)

    def inference(self, pointcloud, task):
        batch_size = pointcloud.shape[0]
        num_points = pointcloud.shape[1]
        query_points = pointcloud.view(batch_size * num_points, -1)  # check
        cp_feats = self.mlp_cp(query_points)  # check
        # pcs = pointcloud.repeat(1, 1, 2)
        whole_feats = self.backbone(pointcloud)  # check
        pn2_cp_feats = whole_feats.permute(0, 2, 1).reshape(batch_size * num_points, -1)  # check
        task = task.view(-1, 1)
        task_feats = self.mlp_task(task)
        task_feats = task_feats.unsqueeze(dim=1).repeat(1, num_points, 1).view(batch_size * num_points, -1)
        all_features = torch.cat([pn2_cp_feats, task_feats, cp_feats], dim=-1)
        logits = self.head(all_features)
        scores = torch.sigmoid(logits.view(batch_size, num_points, -1))
        return scores, pn2_cp_feats.view(batch_size, num_points, -1)


@VATMARTNETWORKS.register_module()
class TrajectoryGenerator(VATModelBase):
    def __init__(self, backbone_cfg=None, vae_cfg=None, mlp_cp_cfg=None, mlp_traj_cfg=None, mlp_task_cfg=None):
        self.lbd_kl = vae_cfg.pop("lbd_kl", None)
        self.lbd_recon_pos = vae_cfg.pop("lbd_recon_pos", None)
        self.lbd_init_dir = vae_cfg.pop("lbd_init_dir", None)
        self.lbd_recon_dir = vae_cfg.pop("lbd_recon_dir", None)

        super(TrajectoryGenerator, self).__init__(backbone_cfg=backbone_cfg, head_cfg=vae_cfg, mlp_cp_cfg=mlp_cp_cfg,
                                                  mlp_task_cfg=mlp_task_cfg, mlp_traj_cfg=mlp_traj_cfg)


    def sample_trajectories(self, pointcloud, task, contact_point, num_trajectories=100):
        batch_size = pointcloud.shape[0]

        # f_task B, 32
        f_task = self.mlp_task(task)
        # after unsqueze B,1,32 after repeat B,100,32 after view B*100,32
        f_task = f_task.unsqueeze(dim=1).repeat(1, num_trajectories, 1).view(batch_size * num_trajectories, -1)

        f_cp = self.mlp_cp(contact_point)
        f_cp = f_cp.unsqueeze(dim=1).repeat(1, num_trajectories, 1).view(batch_size * num_trajectories, -1)

        pointcloud[:, 0, :] = contact_point
        # pointcloud = pointcloud.repeat(1, 1, 2)
        whole_pointcloud_features = self.backbone(pointcloud)

        pn2_contact_point_features = whole_pointcloud_features[:, :, 0]
        pn2_contact_point_features = pn2_contact_point_features.unsqueeze(dim=1).repeat(1, num_trajectories, 1).view(batch_size * num_trajectories, -1)

        condition = torch.cat([pn2_contact_point_features, f_task, f_cp], dim=-1)
        trajs = self.head.decode(condition, z=None, maximum_a_posteriori=num_trajectories==1)
        trajs = trajs.view(batch_size, num_trajectories, -1)
        return trajs
