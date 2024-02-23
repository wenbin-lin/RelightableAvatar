import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer
from . import crit


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)
        self.img2mse = lambda x, y: torch.mean((x - y)**2)

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        if 'resd' in ret:
            offset_loss = torch.norm(ret['resd'], dim=2).mean()
            scalar_stats.update({'offset_loss': offset_loss})
            loss += 0.02 * offset_loss

        if 'gradients' in ret:
            gradients = ret['gradients']
            grad_loss = (torch.norm(gradients, dim=2) - 1.0)**2
            grad_loss = grad_loss.mean()
            scalar_stats.update({'grad_loss': grad_loss})
            loss += 0.01 * grad_loss

        if 'observed_gradients' in ret:
            ogradients = ret['observed_gradients']
            ograd_loss = (torch.norm(ogradients, dim=2) - 1.0)**2
            ograd_loss = ograd_loss.mean()
            scalar_stats.update({'ograd_loss': ograd_loss})
            loss += 0.01 * ograd_loss

        if 'msk_sdf' in ret:
            mask_loss = crit.sdf_mask_crit(ret, batch)
            scalar_stats.update({'mask_loss': mask_loss})
            loss += mask_loss

        mask = batch['mask_at_box']
        img_loss = self.img2mse(ret['rgb_map'][mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
