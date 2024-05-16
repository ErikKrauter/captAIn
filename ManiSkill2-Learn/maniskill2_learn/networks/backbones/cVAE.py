import torch
import torch.nn as nn

from ..utils import combine_obs_with_action
from maniskill2_learn.utils.torch import get_one_device
from ..builder import BACKBONES, build_backbone


@BACKBONES.register_module()
class CVAE(nn.Module):
    def __init__(self, encoder_cfg, decoder_cfg, latent_dim, log_sig_min=-4, log_sig_max=15):
        super(CVAE, self).__init__()
        # assert encoder_cfg.mlp_spec[-1] // 2 + cond_dim == decoder_cfg.mlp_spec[0]
        self.encoder = build_backbone(encoder_cfg)
        self.decoder = build_backbone(decoder_cfg)
        self.get_mu = nn.Linear(encoder_cfg.mlp_spec[-1], latent_dim)
        self.get_logvar = nn.Linear(encoder_cfg.mlp_spec[-1], latent_dim)
        self.latent_dim = latent_dim
        self.log_sig_min, self.log_sig_max = log_sig_min, log_sig_max

    def forward(self, feats, cond):
        encoded = self.encoder(feats)
        mu = self.get_mu(encoded)
        logvar = self.get_logvar(encoded)  # .clamp(self.log_sig_min, self.log_sig_max)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return self.decode(cond, z), mu, logvar

    def decode(self, cond, z=None, maximum_a_posteriori=False):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            batch_size = cond.shape[0]
            device = get_one_device(cond)
            z = torch.randn(batch_size, self.latent_dim, device=device).clamp(-0.5, 0.5)
            if maximum_a_posteriori:
                z = torch.full((batch_size, self.latent_dim), 0.5, dtype=torch.float32, device=device)
        inputs = combine_obs_with_action(cond, z)
        return self.decoder(inputs)
