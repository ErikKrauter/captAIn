import torch.nn as nn, torch, numpy as np
from torch.nn import Parameter
from ..builder import REGHEADS
from maniskill2_learn.utils.torch import ExtendedModule
from maniskill2_learn.utils.data import is_num, is_not_null, to_np


@REGHEADS.register_module()
class SimpleDeterministicHead(ExtendedModule):
    def __init__(self, scale_prior=1, bias_prior=0, noise_std=0.1, bound=None, dim_output=None, num_heads=1):
        # The noise is the Gaussian noise for exploration.
        super(SimpleDeterministicHead, self).__init__()
        self.bound = bound
        if bound is not None:
            low = bound[0]
            high = bound[1]
            scale_prior = (high - low) / 2
            bias_prior = (high + low) / 2
            self.lb, self.ub = [Parameter(torch.tensor(bound[i]), requires_grad=False) for i in [0, 1]]

        self.scale_prior = Parameter(torch.tensor(scale_prior, dtype=torch.float32), requires_grad=False)
        self.bias_prior = Parameter(torch.tensor(bias_prior, dtype=torch.float32), requires_grad=False)
        self.noise_std = noise_std

    def forward(self, feature, mode="explore",  num_actions=1, **kwargs):
        """
        Forward will return action with exploration, log p, mean action, log std, std
        """
        assert num_actions == 1
        mean = torch.tanh(feature)
        noise = mean.clone().normal_(0., std=self.noise_std)
        action = (mean + noise).clamp(self.lb, self.ub) * self.scale_prior + self.bias_prior
        mean = mean * self.scale_prior + self.bias_prior

        # print(2, time.time() - st)
        if mode == "mean" or mode == "eval":
            return mean
        elif mode == "explore" or mode == "sample":
            sample = self.clamp_action(action)
            return sample
        elif mode == "all":
            # sample, log_p, mean, log_std, std
            return action, torch.ones_like(action) * -np.inf, mean, torch.ones_like(action) * -np.inf, torch.zeros_like(action)
        else:
            raise ValueError(f"Unsupported mode {mode}!!")

    def clamp_action(self, action):
        action = (action - self.bias_prior) / self.scale_prior
        action = action.clamp(self.lb, self.ub)
        return action * self.scale_prior + self.bias_prior