"""
PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    https://arxiv.org/abs/1612.00593
Reference Code:
    https://github.com/fxia22/pointnet.pytorch.git
"""

import numpy as np
from copy import deepcopy
import torch, torch.nn as nn, torch.nn.functional as F

from ..modules.attention import MultiHeadAttention
from .mlp import ConvMLP, LinearMLP
from ..builder import BACKBONES, build_backbone
from maniskill2_learn.utils.data import dict_to_seq, split_dim, GDict, repeat
from maniskill2_learn.utils.torch import masked_average, masked_max, ExtendedModule
# from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from maniskill2_learn.networks.backbones.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
from maniskill2_learn.utils.meta import get_logger, get_world_rank

'''

This is the PointNet++ backbone used in all the perception modules.

'''

@BACKBONES.register_module(name='PointNet2')
class PointNet2(ExtendedModule):

    def __init__(self, hparams, n_points, global_feature=False):
        super().__init__()
        print("CONSTRUCTING POINTNET++")
        print(hparams)
        self.hparams = hparams
        self.repeat_pc = False
        self.n_points = n_points
        self.global_feature = global_feature
        self.diagnostics = False

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        '''self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,  # 1024
                radius=0.1,
                nsample=32,
                mlp=[3, 32, 32, 64],
                use_xyz=True,
                number=10
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
                number=11
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                number=12
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
                number=13
            )
        )'''

        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(self.n_points*0.75),  # 1024
                radius=0.015,
                nsample=32,
                mlp=[3, 64, 64, 128],
                use_xyz=True,
                number=10,
                diagnostic=self.diagnostics
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(self.n_points*0.75*0.5),
                radius=0.04,
                nsample=32,
                mlp=[128+3, 128, 128, 256],
                use_xyz=True,
                number=11,
                diagnostic=self.diagnostics
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=int(self.n_points*0.75*0.5*0.5),
                radius=0.16,
                nsample=32,
                mlp=[256+3, 256, 256, 512],
                use_xyz=True,
                number=12,
                diagnostic=self.diagnostics
            )
        )

        if not self.global_feature:
            self.FP_modules = nn.ModuleList()
            '''self.FP_modules.append(PointnetFPModule(mlp=[128+3, 128, 128, 128], number=3))
            self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128], number=2))
            self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256], number=1))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], number=0))'''
            self.FP_modules.append(PointnetFPModule(mlp=[128, 128, 128], number=2, diagnostic=self.diagnostics))
            self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 128], number=1, diagnostic=self.diagnostics))
            self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256], number=0, diagnostic=self.diagnostics))

            self.fc_layer = nn.Sequential(
                nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.hparams['feat_dim']),
                nn.ReLU(True),
            )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

    def forward(self, pointcloud, concat_state=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        logger = get_logger()
        rank = get_world_rank()

        if isinstance(pointcloud, dict):
            xyz = pointcloud["xyz"]
            xyz = xyz[..., 0:3].contiguous()
            with torch.no_grad():
                features = []
                if "rgb" in pointcloud:
                    features.append(pointcloud["rgb"])
                if "seg" in pointcloud:
                    features.append(pointcloud["seg"])
                if concat_state is not None:  # [B, C]
                    features.append(concat_state[:, None, :].expand(-1, xyz.shape[1], -1))
                if len(features) == 0:
                    features = None
                else:
                    features = torch.cat(features, dim=-1)
                    features = features.permute(0, 2, 1).contiguous()
        else:
            if pointcloud.shape[-1] == 3 and self.repeat_pc:
                pointcloud = pointcloud.repeat(1, 1, 2)
            xyz, features = self._break_up_pc(pointcloud)

        # xyz (B, N, 3), features (B, 3, N)
        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            if torch.any(torch.isnan(li_xyz)):
                logger.error(f"NANS in li_xyz of rank {rank} iteration {i} of SA_modules")
                # exit(0)
            if torch.any(torch.isnan(li_features)):
                logger.error(f"NANS in li_features of rank {rank} iteration {i} of SA_modules")
                # exit(0)
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        if self.global_feature:
            global_feature = l_features[-1].max(-1)[0]
            return global_feature

        # we reduce number of points per hierarchy but aggregate more features per point
        # l_xyz[0] (B, 1200, 3), l_xyz[1] (B, 1024, 3), l_xyz[2] (B, 256, 3), l_xyz[3] (B, 64, 3), l_xyz[4] (B, 16, 3)
        # l_features[0] (B, 3, 10k), l_features[1] (B, 64, 1024), l_features[2] (B, 128, 256), l_features[3] (B, 256, 64), l_features[4] (B, 512, 16)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )
            if torch.any(torch.isnan(l_features[i - 1])):
                logger.error(f"NANS in l_features rank {rank} iteration {i} of FP_modules")
                # exit(0)
                # logger.error(f"li_xyz {l_features[i - 1]}")
        # l_features[0] (B, 128, N), l_features[1] (B, 128, 1024), l_features[2] (B, 256, 256), l_features[3] (B, 256, 64), l_features[4] (B, 512, 16)
        out = self.fc_layer(l_features[0])  # B, 128, N where 128 = feat_dim


        if torch.any(torch.isnan(out)):
            logger.error(f"NANS in output of pointnet2 of rank {rank}")
            # exit()

        return out



    """
    self.SAModules = list(PointnetSAModules) * 4
    PointnetSAModule(
      (groupers): ModuleList(
      # this first select centroid in pc thru fps, then ball query around centroids to select local point cloud regions,
       then groups features of those regions together into tensor of shape (B, num features, num centroids, num points in ball)
       # where num centroids = npoint, num points in ball = nsample see above in definition of SAModule
        (0): QueryAndGroup()  
      )
      # this is a 1x1 convolution over each point in each local region.
      # a point can be part of multiple local regions.
      # However, because we substract the xyz valueas of the centroid from each xyz value of the point in the region
      # the features of the very same point are different depending on in which region he is.
      # thus the convolution with the same kernel will lead to different results depending on the region the point is in
      # I think the rgb features are independant from the local region, but i am not sure.
      # But because we concanetate the xyz features with rgb features and process them with a 1x1x6 kernel the overall
      # resulting features after the 1d conv will differe for the same point in different regions
      (mlps): ModuleList(
        (0): Sequential(
          (0): Conv2d(3, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (1): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (2): ReLU(inplace=True)
          (3): Conv2d(32, 32, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (4): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (5): ReLU(inplace=True)
          (6): Conv2d(32, 64, kernel_size=(1, 1), stride=(1, 1), bias=False)
          (7): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          (8): ReLU(inplace=True)
        )
      )
    )
    
    # we run through several of those SA modules. the input pointcloud to the consecutive SA module are the centroids
    # sampled in then previous model. The feature tensor given to the next SA module are the global feature vectors of 
    # each centroid's region computed in previous SA module.
    # each consecutive SA module samples fewer centroids, with larger ball radius and further accumulates the features
    # this way we increase the receptive field. We store all features and centoids from every SA model in a list
    
    # then the FP modules are used to go back to the full pointcloud, essentially traversing the hierarchy in the 
    # oposite direction. The goal is to use the most global features from the last SA moudules and interpolate them 
    # with the features from the SA module one hierarchy before.
    # this is very similar to the U-net architecture but for point clouds
    
    self.FPModules = list(PointnetFPModule) * 4
    PointnetFPModule(
  (mlp): Sequential(
    (0): Conv2d(768, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (2): ReLU(inplace=True)
    (3): Conv2d(256, 256, kernel_size=(1, 1), stride=(1, 1), bias=False)
    (4): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (5): ReLU(inplace=True)
  )
)   
    # uc = centroids of hierarchy above, lc = centroids of current layer (one hierarchy below)
    # first find the three nearest neighbors of lc in uc
        # there are more lc than uc, thus uc will be neighbors multiple times
    # then computed weighted sum of the features of those three nearest neighbors antiproportional to distance
        # this accumulates features from global (uc's features)
    # concatenate ls's feature vector with the accumulated uc features
    # pass that feature vector through 1D convolutions, decreasing number of channler, i.e. feature vector
        # that combines local features with global features and accumulates them
    # this newly computed feature vector is the new global feature vector (uc's features) for next iteration
    # the lc centroids become the new uc centroids in next iteration
    # this way we propagate back until we reach full pointcloud size. 
    # this tripples down global features from all hierarchies back to the original points in pointcloud  
    
    fc layer:
    Sequential(
  (0): Conv1d(128, 128, kernel_size=(1,), stride=(1,), bias=False)
  (1): BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (2): ReLU(inplace=True)
)
    # finally the aggregated feature vector at each point in pointcloud is passed throgh another 1D convlution
    # results in 128 features vector per point in pointcloud.
    # the features were aggregated from all hierarchies of the pointcloud and thus containt local and global information
    
    # in VAT we use the feature vector at the contact point as input to the other perception networks
    """