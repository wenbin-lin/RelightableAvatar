import torch.nn as nn
from lib.config import cfg
import torch
from lib.networks.renderer import make_renderer


def get_latent_smooth_loss(model_outputs):
    d_diff = model_outputs['diffuse_albedo_values']
    d_rough = model_outputs['roughness_values']
    d_xi_diff = model_outputs['random_xi_diffuse_albedo']
    d_xi_rough = model_outputs['random_xi_roughness']
    loss = nn.L1Loss()(d_diff, d_xi_diff) + nn.L1Loss()(d_rough, d_xi_rough) 
    return loss 

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

def get_kl_loss(latent_values):
    latent_values = latent_values.view(-1, 32)
    loss = kl_divergence(0.05, latent_values)
    return loss

def get_albedo_smooth_loss(albedo_values, albedo_nn_values):
    pt_num = albedo_values.shape[-2]
    nn_num = albedo_nn_values.shape[-2] // pt_num
    albedo_values = albedo_values.view(-1, pt_num, 1, 3)
    albedo_nn_values = albedo_nn_values.view(-1, pt_num, nn_num, 3)
    albedo_diff = albedo_values - albedo_nn_values

    scale = torch.mean(albedo_nn_values, dim=-2, keepdim=True) + 1e-6
    loss = torch.mean(torch.abs(albedo_diff) / scale)

    return loss

def get_roughness_smooth_loss(roughness_values, roughness_nn_values):
    pt_num = roughness_values.shape[-2]
    nn_num = roughness_nn_values.shape[-2] // pt_num
    ch_num = roughness_values.shape[-1]
    roughness_values = roughness_values.view(-1, pt_num, 1, ch_num)
    roughness_nn_values = roughness_nn_values.view(-1, pt_num, nn_num, ch_num)
    diff = roughness_values - roughness_nn_values

    scale = torch.sum(roughness_nn_values, dim=-2, keepdim=True) + 1e-6
    loss = torch.mean(torch.abs(diff) / scale)

    return loss


class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net
        self.renderer = make_renderer(cfg, self.net)

        self.img2mse = lambda x, y: torch.mean((x - y)**2)
        self.img2l1 = lambda x, y: torch.mean(torch.abs(x - y))

    def forward(self, batch):
        ret = self.renderer.render(batch)
        scalar_stats = {}
        loss = 0

        mask = batch['mask_at_box']
        # l1 loss after gamma correction
        img_loss = self.img2l1(torch.pow(ret['rgb_map'] + 1e-6, 1 / 2.2)[mask], batch['rgb'][mask])
        scalar_stats.update({'img_loss': img_loss})
        loss += img_loss

        if 'diffuse_albedo_values' in ret:
            latent_smooth_loss = get_latent_smooth_loss(ret)
            scalar_stats.update({'latent_smooth_loss': latent_smooth_loss})
            loss += 0.01 * latent_smooth_loss

        if 'brdf_latent' in ret:
            kl_loss = get_kl_loss(ret['brdf_latent'])
            scalar_stats.update({'kl_loss': kl_loss})
            loss += 0.001 * kl_loss

        if 'diffuse_albedo_nn_values' in ret:
            albedo_reg_loss = get_albedo_smooth_loss(ret['diffuse_albedo_values'], ret['diffuse_albedo_nn_values'])
            scalar_stats.update({'albedo_reg_loss': albedo_reg_loss})
            loss += 0.0005 * albedo_reg_loss

        if 'roughness_nn_values' in ret:
            roughness_reg_loss = get_roughness_smooth_loss(ret['roughness_values'], ret['roughness_nn_values'])
            scalar_stats.update({'rougheness_reg_loss': roughness_reg_loss})
            loss += 0.0005 * roughness_reg_loss

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return ret, loss, scalar_stats, image_stats
