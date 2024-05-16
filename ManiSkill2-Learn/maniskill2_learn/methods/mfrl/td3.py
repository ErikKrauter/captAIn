from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from maniskill2_learn.networks import build_model, build_actor_critic
from maniskill2_learn.utils.torch import build_optimizer
from maniskill2_learn.utils.data import to_torch
from maniskill2_learn.utils.torch import BaseAgent, hard_update, soft_update
from maniskill2_learn.utils.meta import get_logger
from ..builder import MFRL


@MFRL.register_module()
class TD3(BaseAgent):
    def __init__(self, actor_cfg, critic_cfg, env_params, batch_size=128, gamma=0.99, update_coeff=0.005,
                 action_noise=0.2, noise_clip=0.5, actor_update_interval=2, detach_actor_feature=False,
                 shared_backbone=False, ignore_dones=True):
        super(TD3, self).__init__()
        actor_cfg = deepcopy(actor_cfg)
        critic_cfg = deepcopy(critic_cfg)
        target_critic_cfg = deepcopy(critic_cfg)

        actor_optim_cfg = actor_cfg.pop("optim_cfg")
        critic_optim_cfg = critic_cfg.pop("optim_cfg")

        self.gamma = gamma
        self.batch_size = batch_size
        self.update_coeff = update_coeff
        self.action_noise = action_noise
        self.noise_clip = noise_clip
        self.actor_update_interval = actor_update_interval
        self.detach_actor_feature = detach_actor_feature
        self.ignore_dones = ignore_dones
        self.shared_backbone = shared_backbone
        # self.action_space = env_params["action_space"]

        actor_cfg.update(env_params)
        critic_cfg.update(env_params)

        # this creates two critics and one actor
        # if shared_backbone, then actor and critic will share pointnet weights
        # note that both critics also share pointnet weights
        # Note that critic_cfg is passed by reference and inside build_actor_critic the critic_cfg is changed
        # thus when calling build_model(critic_cfg) later the visual backbone will point to the exact same instance
        # as the backbone used in the actor and critic
        self.actor, self.critic = build_actor_critic(actor_cfg, critic_cfg, shared_backbone)

        self.actor_optim = build_optimizer(self.actor, actor_optim_cfg)
        self.critic_optim = build_optimizer(self.critic, critic_optim_cfg)

        # this creates two target critics
        # create from copy of critic to make sure that we do not share visual backbone with actor and critic
        self.target_actor, self.target_critic = build_actor_critic(actor_cfg, target_critic_cfg, shared_backbone)

        hard_update(self.target_critic, self.critic)
        hard_update(self.target_actor, self.actor)

        self.logger = get_logger()


    def update_parameters(self, memory, updates):
        sampled_batch = memory.sample(self.batch_size).to_torch(device=self.device, non_blocking=True)
        sampled_batch = self.process_obs(sampled_batch)
        # print(f"sampled td3 batch: {sampled_batch}")
        # print(f"length of batch {len(sampled_batch)}")

        '''for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]'''
        with torch.no_grad():
            _, _, next_action_prob, _, _ = self.target_actor(sampled_batch['next_obs'], mode='all')  # next_action_prob has no noise added
            # for some reason we add noise outside the policy head.
            noise = (torch.randn_like(next_action_prob) * self.action_noise).clamp(-self.noise_clip, self.noise_clip)
            # next_action = (next_action_prob + noise).clamp(self.target_actor.head.bound[0][0], self.target_actor.head.bound[1][0])
            next_action = self.target_actor.head.clamp_action(next_action_prob + noise)
            q_next_target = self.target_critic(sampled_batch['next_obs'], next_action)
            min_q_next_target = torch.min(q_next_target, dim=-1, keepdim=True).values
            if self.ignore_dones:
                q_target = sampled_batch['rewards'] + self.gamma * min_q_next_target
            else:
                q_target = sampled_batch['rewards'] + (
                            1 - sampled_batch['dones'].int()) * self.gamma * min_q_next_target
            q_target.repeat(1, q_next_target.shape[-1])

        q = self.critic(sampled_batch['obs'], sampled_batch['actions'])
        critic_loss = F.mse_loss(q, q_target) * q.shape[-1]

        with torch.no_grad():
            abs_critic_error = torch.abs(q - q_target).max().item()

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        with torch.no_grad():
            critic_grad = self.critic.grad_norm

        if self.shared_backbone:
            self.critic_optim.zero_grad()

        actor_grad = 0
        if updates % self.actor_update_interval == 0:
            # self.logger.info(f"rank {rank} entering actor update at update iteration: {updates}")
            # no noise added to action, due to mode=eval
            action = self.actor(sampled_batch['obs'], mode='eval', save_feature=self.shared_backbone, detach_visual=self.detach_actor_feature)
            visual_feature = self.actor.backbone.pop_attr("saved_visual_feature")
            if visual_feature is not None:
                visual_feature = visual_feature.detach()
            policy_loss = -self.critic(sampled_batch['obs'], action, visual_feature=visual_feature).mean()
            # self.logger.info(f"rank {rank} finished policy loss calculation")
            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()
            with torch.no_grad():
                actor_grad = self.actor.grad_norm

            soft_update(self.target_critic, self.critic, self.update_coeff)
            soft_update(self.target_actor, self.actor, self.update_coeff)
            # in stable baselines 3 they also do a seperate polyak update for running stats
        else:
            policy_loss = torch.zeros(1)

        ret = {
            "td3/critic_loss": critic_loss.item(),
            "td3/max_critic_abs_err": abs_critic_error,
            "td3/actor_loss": policy_loss.item(),
            "td3/q": torch.min(q, dim=-1).values.mean().item(),
            "td3/q_target": torch.mean(q_target).item(),
            "td3/critic_grad": critic_grad,
        }
        if actor_grad:
            ret["td3/actor_grad"] = actor_grad
        return ret
