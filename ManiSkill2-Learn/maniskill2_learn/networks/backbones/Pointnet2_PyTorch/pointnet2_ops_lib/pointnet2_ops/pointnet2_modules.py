from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
# from pointnet2_ops import pointnet2_utils
from maniskill2_learn.networks.backbones.Pointnet2_PyTorch.pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
from maniskill2_learn.utils.meta import get_logger, get_world_rank
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class DiagnosticLayer(nn.Module):
    def __init__(self, layer, name=''):
        super(DiagnosticLayer, self).__init__()
        self.layer = layer
        self.name = name
        self.counter = 0
        self.tensor_cache = []
        self.cache_size = 1

    def forward(self, x):
        # Check input
        err = self.check_for_invalid_values(x, f'Input to {self.name}')
        # self.cache_tensor(x, 'input')
        if self.counter % 200 == 0:
            self.print_stats(x, f'Input to {self.name}')

        # Forward pass through the actual layer
        x = self.layer(x)

        # Check output
        err |= self.check_for_invalid_values(x, f'Output of {self.name}')
        #self.cache_tensor(x, 'output')
        if self.counter % 200 == 0:
            self.print_stats(x, f'Output of {self.name}')

        self.counter += 1

        #if err and get_world_rank() == 0:
            #self.save_cached_tensors()

        return x

    def cache_tensor(self, tensor, stage):
        if len(self.tensor_cache) >= self.cache_size:
            self.tensor_cache.pop(0)
        self.tensor_cache.append((self.name, stage, tensor.clone().detach(), self.counter))

    def save_cached_tensors(self):
        file_name = f'nan_tensors_{self.name}_{self.counter}.pt'
        torch.save(self.tensor_cache, file_name)
        get_logger().error(f'NaN detected in {self.name}. Tensors saved to {file_name}')

    def get_gradient_norm(self):
        grad_norms = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2)
                grad_norms[f'{self.name}.{name}'] = grad_norm.item()
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    get_logger().error(
                        f'Gradient problem (NaN or Inf) in {self.name}, parameter: {name}, grad norm: {grad_norm}')
        return grad_norms

    def check_for_invalid_values(self, tensor, name):
        err = False
        if torch.isnan(tensor).any():
            get_logger().error(f'NaNs detected in {name}')
            err = True
        if torch.isinf(tensor).any():
            get_logger().error(f'Infs detected in {name}')
            err = True
        if torch.isinf(tensor).any() and not torch.isposinf(tensor).all():
            get_logger().error(f'-Infs detected in {name}')
            err = True
        if err:
            get_logger().error(f'{name} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, '
                              f'Min: {tensor.min().item()}, Max: {tensor.max().item()}')
        return err

    def print_stats(self, tensor, name):
        get_logger().info(f'{name} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, '
              f'Min: {tensor.min().item()}, Max: {tensor.max().item()}')


def build_shared_mlp(mlp_spec: List[int], bn: bool = True, diagnostic: bool = False, number=99):
    layers = []
    for i in range(1, len(mlp_spec)):
        conv = nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        layers.append(DiagnosticLayer(conv, f'{number}_Conv2d_{i}') if diagnostic else conv)

        if bn:
            bn_layer = nn.BatchNorm2d(mlp_spec[i])
            layers.append(DiagnosticLayer(bn_layer, f'{number}_BatchNorm2d_{i}',) if diagnostic else bn_layer)

        relu = nn.ReLU(True)
        layers.append(DiagnosticLayer(relu, f'{number}_ReLU_{i}') if diagnostic else relu)

    return nn.Sequential(*layers)


'''def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)'''


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None


    def check_for_invalid_values(self, tensor, name):
        err = False
        if torch.isnan(tensor).any():
            get_logger().error(f'NaNs detected in {name}')
            err = True
        if torch.isinf(tensor).any():
            get_logger().error(f'Infs detected in {name}')
            err = True
        if torch.isinf(tensor).any() and not torch.isposinf(tensor).all():
            get_logger().error(f'-Infs detected in {name}')
            err = True
        if err:
            get_logger().error(f'{name} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, '
                              f'Min: {tensor.min().item()}, Max: {tensor.max().item()}')

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )
        # self.check_for_invalid_values(new_xyz, f'new_xyz after fps and gather in _PointnetSAModuleBase')

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)


            # self.check_for_invalid_values(new_features, f'new_features after self.groupers in _PointnetSAModuleBase')

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            # self.check_for_invalid_values(new_features, f'new_features after self.mlps in _PointnetSAModuleBase')

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)

            # self.check_for_invalid_values(new_features, f'new_features after max_pool2d in _PointnetSAModuleBase')

            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True, number=100, diagnostic=False):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, int(nsample), use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            #if use_xyz:
                #mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn=bn, diagnostic=diagnostic, number=number))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True, number=100, diagnostic=False
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
            number=number,
            diagnostic=diagnostic
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True, flag=False, number=99, diagnostic=False):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.batchnorm_running_stats_backup = None
        self.mlps = build_shared_mlp(mlp, bn=bn, diagnostic=diagnostic, number=number)
        self.flag = flag
        self.number = number

    def check_for_invalid_values(self, tensor, name):
        err = False
        if torch.isnan(tensor).any():
            get_logger().error(f'NaNs detected in {name} number {self.number}')
            err = True
        if torch.isinf(tensor).any():
            get_logger().error(f'Infs detected in {name} number {self.number}')
            err = True
        if torch.isinf(tensor).any() and not torch.isposinf(tensor).all():
            get_logger().error(f'-Infs detected in {name} number {self.number}')
            err = True
        if err:
            get_logger().error(f'{name} number {self.number} - Mean: {tensor.mean().item():.4f}, Std: {tensor.std().item():.4f}, '
                               f'Min: {tensor.min().item()}, Max: {tensor.max().item()}')

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)

            # self.check_for_invalid_values(dist, f'dist after three_nn in PointnetFPModule')

            dist_recip = 1.0 / (dist + 1e-8)

            # self.check_for_invalid_values(dist_recip, f'dist_recip in PointnetFPModule')

            norm = torch.sum(dist_recip, dim=2, keepdim=True)

            # self.check_for_invalid_values(norm, f'norm in PointnetFPModule')

            weight = dist_recip / norm

            # self.check_for_invalid_values(weight, f'weight in PointnetFPModule')

            #if torch.any(torch.isclose(dist, torch.zeros_like(dist), atol=0.001, rtol=0.001)):
                #get_logger().error(
                    #f"distance close to zero in dist in PointnetFPModule number {self.number}")
                #get_logger().error(
                    #f"dist: \t {dist} \n dist_recipt: \t {dist_recip} \n norm: \t {norm} \n weight: \t {weight}")

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )

            # self.check_for_invalid_values(interpolated_feats, f'interpolated_feats in PointnetFPModule')

        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)

            # self.check_for_invalid_values(new_features, f'new_features after torch.cat in PointnetFPModule')

        else:
            new_features = interpolated_feats

        # self.backup_running_stats(self.mlps)

        if self.flag:
            if torch.randn(1) > 0.8:
                nan_mask = torch.rand_like(new_features) < 0.1  # 10% chance of being NaN
                new_features[nan_mask] = float('nan')
                if torch.isnan(new_features).any():
                    print("NaNs in input:", torch.isnan(new_features).any())

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlps(new_features)  # B, 768, 64, 1

        if torch.any(torch.isnan(new_features)):
            get_logger().error(
                f"NANS in new_features after self.mlps in PointnetFPModule with size "
                f"{self.mlps[0].in_channels}")
            # self.restore_running_stats(self.mlps)

            # new_features = torch.rand_like(new_features)

        return new_features.squeeze(-1)

    def backup_running_stats(self, model):
        backup = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                backup[name] = {
                    "running_mean": module.running_mean.clone(),
                    "running_var": module.running_var.clone()
                }
        self.batchnorm_running_stats_backup = copy.deepcopy(backup)

    def restore_running_stats(self, model):
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)) and name in self.batchnorm_running_stats_backup:
                module.running_mean = self.batchnorm_running_stats_backup[name]["running_mean"]
                module.running_var = self.batchnorm_running_stats_backup[name]["running_var"]
