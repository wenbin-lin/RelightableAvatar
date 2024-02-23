import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from . import embedder
from .models import *
from lib.utils import net_utils
import os
from lib.utils import sample_utils
from lib.utils.body_model import lbs
from lib.utils.body_model.body_model import BodyModel

TINY_NUMBER = 1e-6

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.actvn = nn.ReLU()

        self.deform_net = DeformNetwork(d_feature=72, d_in=3, d_out_1=1, d_out_2=3, 
                                        n_blocks=3, d_hidden=256, n_layers_a=2, n_layers_b=2, 
                                        skip_in=[], multires=10, weight_norm=True)

        self.body_model = BodyModel()

        train_frame_index = np.arange(cfg.begin_ith_frame, cfg.begin_ith_frame + cfg.num_train_frame * cfg.frame_interval, step=cfg.frame_interval)
        param_dict = {}
        param_dict.update({'root_orient_' + str(fr): nn.Parameter(torch.zeros(3, dtype=torch.float32)) for idx, fr in enumerate(train_frame_index)})
        param_dict.update({'pose_body_' + str(fr): nn.Parameter(torch.zeros(72, dtype=torch.float32)) for idx, fr in enumerate(train_frame_index)})
        param_dict.update({'trans_' + str(fr): nn.Parameter(torch.zeros(3, dtype=torch.float32)) for idx, fr in enumerate(train_frame_index)})
        param_dict.update({'betas': nn.Parameter(torch.zeros((1, 10), dtype=torch.float32))})
        self.body_poses = nn.ParameterDict(param_dict)

    def init_smpl_param(self, frame_id_all, Rh_all, Th_all, poses_all, beta):
        param_dict = {}
        param_dict.update({'root_orient_' + str(fr): nn.Parameter(torch.tensor(Rh_all[idx], dtype=torch.float32)) for idx, fr in enumerate(frame_id_all)})
        param_dict.update({'pose_body_' + str(fr): nn.Parameter(torch.tensor(poses_all[idx], dtype=torch.float32)) for idx, fr in enumerate(frame_id_all)})
        param_dict.update({'trans_' + str(fr): nn.Parameter(torch.tensor(Th_all[idx], dtype=torch.float32)) for idx, fr in enumerate(frame_id_all)})
        param_dict.update({'betas': nn.Parameter(torch.tensor(beta, dtype=torch.float32))})
        self.body_poses = nn.ParameterDict(param_dict)

    def freeze_smpl(self):
        for k, v in self.body_poses.items():
            self.body_poses[k].requires_grad_(False)

    def recompute_smpl(self, batch):
        f_idx = batch['file_index'][0].item()
    
        root_orient = self.body_poses['root_orient_' + str(f_idx)].unsqueeze(0)
        pose_body = self.body_poses['pose_body_' + str(f_idx)].unsqueeze(0)
        trans = self.body_poses['trans_' + str(f_idx)].unsqueeze(0)
        betas = self.body_poses['betas']

        full_pose = torch.cat((torch.zeros(1, 3, device=pose_body.device), pose_body[:, 3:]), dim=-1)
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, minimal_shape = self.body_model(betas, full_pose)
        
        rot = lbs.batch_rodrigues(root_orient)

        joints_transform = bone_transforms.clone()
        joints_transform[0, :, :3, -1] = Jtrs_posed
        joints_transform[0, 0, :3, :3] = rot

        batch['A'] = bone_transforms
        batch['poses'] = full_pose
        batch['pvertices'] = verts_posed
        batch['Th'][0] = trans
        batch['R'] = rot
        batch['joints_transform'] = joints_transform                                                                   

    def calculate_residual_deformation(self, tpose, batch):
        latent = batch['poses'].detach()
        tpose_input = tpose.detach()
        input_bw = self.get_tpose_skinning(tpose_input, batch)
        tpose_deformed = self.deform_net(latent[0].T, tpose[0], input_bw[0])
        resd = tpose_deformed[None] - tpose
        return resd
        
    def calculate_residual_deformation_can2deformed(self, tpose, batch):
        latent = batch['poses'].detach()
        tpose_input = tpose.detach()
        input_bw = self.get_tpose_skinning(tpose_input, batch)
        tpose_deformed = self.deform_net.inverse(latent[0].T, tpose[0], input_bw[0])
        return tpose_deformed[None]
    
    def get_tpose_skinning(self, pts, batch):
        pbw, _ = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        return pbw

    def pose_points_to_tpose_points(self, pose_pts, pose_dirs, batch):
        """
        pose_pts: n_batch, n_point, 3
        """

        # initial blend weights of points at i
        pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'], K=5)
        pbw = pbw.permute(0, 2, 1)

        # transform points from i to i_0
        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])
        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd, pbw
    
    def pose_points_to_tpose_points_mesh_vertices(self, pose_pts, pose_dirs, mesh_vertices, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        pbw = sample_utils.sample_blend_closest_points_single(pose_pts, mesh_vertices, batch['mesh_tbw'])

        pbw = pbw.permute(0, 2, 1)

        init_tpose = pose_points_to_tpose_points(pose_pts, pbw,
                                                 batch['A'])
        init_bigpose = tpose_points_to_pose_points(init_tpose, pbw,
                                                   batch['big_A'])

        resd = self.calculate_residual_deformation(init_bigpose, batch)
        tpose = init_bigpose + resd

        if cfg.tpose_viewdir and pose_dirs is not None:
            init_tdirs = pose_dirs_to_tpose_dirs(pose_dirs, pbw,
                                                 batch['A'])
            tpose_dirs = tpose_dirs_to_pose_dirs(init_tdirs, pbw,
                                                 batch['big_A'])
        else:
            tpose_dirs = None

        return tpose, tpose_dirs, init_bigpose, resd, pbw

    def forward(self, wpts, viewdir, dists, n_pixel, n_sample, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])
        viewdir = viewdir[None]
        pose_dirs = world_dirs_to_pose_dirs(viewdir, batch['R'])

        if 'mesh_vertices' in batch.keys():
            mesh_vertices = world_points_to_pose_points(batch['mesh_vertices'], batch['R'], batch['Th'])

        with torch.no_grad():
            pbw, pnorm = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
            pnorm = pnorm[..., 0]
            norm_th = 0.10
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind][None]
            pose_dirs = pose_dirs[pind][None]

        # transform points from the pose space to the tpose space
        if 'mesh_vertices' in batch.keys():
            tpose, tpose_dirs, init_bigpose, resd, pbw = self.pose_points_to_tpose_points_mesh_vertices(
                pose_pts, pose_dirs, mesh_vertices, batch)
        else:
            tpose, tpose_dirs, init_bigpose, resd, pbw = self.pose_points_to_tpose_points(
                pose_pts, pose_dirs, batch)
        tpose = tpose[0]

        viewdir = -viewdir[0]

        # use alpha to filer out point far from surface
        alpha = self.tpose_human.get_alpha(tpose, dists, batch)

        n_batch, n_point = wpts.shape[:2]
        alpha_all = torch.zeros([n_batch * n_point]).to(wpts)
        alpha_all[pind[0]] = alpha

        alpha_all = alpha_all.view(n_pixel, n_sample)

        weights = alpha_all * torch.cumprod(
        torch.cat(
            [torch.ones((alpha_all.shape[0], 1)).to(alpha_all), 1. - alpha_all + 1e-10],
            -1), -1)[:, :-1]

        weights = weights.view(-1)

        near_surface_pind = weights > 1e-3
        near_surface_pind = near_surface_pind[pind[0]]

        if torch.sum(near_surface_pind) == 0:
            near_surface_pind[0] = True

        pind[pind.clone()] = near_surface_pind
        pose_pts = pose_pts[:, near_surface_pind, :]
        viewdir = viewdir[near_surface_pind, :]
        pose_dirs = pose_dirs[:, near_surface_pind, :]
        tpose = tpose[near_surface_pind, :]
        tpose_dirs = tpose_dirs[:, near_surface_pind, :]
        init_bigpose = init_bigpose[:, near_surface_pind, :]
        resd = resd[:, near_surface_pind, :]
        pbw = pbw[:, :, near_surface_pind]

        ret = self.tpose_human(tpose, pose_pts, viewdir, dists, batch, pbw)

        ind = ret['sdf'][:, 0].detach().abs() < 0.02
        init_bigpose = init_bigpose[0][ind][None].detach().clone()

        tbounds = batch['tbounds'][0]
        tbounds[0] -= 0.05
        tbounds[1] += 0.05
        inside = tpose > tbounds[:1]
        inside = inside * (tpose < tbounds[1:])
        outside = torch.sum(inside, dim=1) != 3
        ret['raw'][outside] = 0

        n_batch, n_point = wpts.shape[:2]
        raw = torch.zeros([n_batch, n_point, 4]).to(wpts)
        raw[pind] = ret['raw']
        sdf = 10 * torch.ones([n_batch, n_point, 1]).to(wpts)
        sdf[pind] = ret['sdf']
        ret.update({'raw': raw, 'sdf': sdf, 'resd': resd})

        ret.update({'gradients': ret['gradients'][None]})

        # sg items
        if self.training:
            sg_diffuse_rgb_values = ret['sg_diffuse_rgb_values'].view(n_batch, -1, 3)
            sg_specular_rgb_values = ret['sg_specular_rgb_values'].view(n_batch, -1, 3)
            normal_values = ret['normal_values'].view(n_batch, -1, 3)
            diffuse_albedo_values = ret['diffuse_albedo_values'].view(n_batch, -1, 3)
            roughness_values = ret['roughness_values'].expand(-1, 9).view(n_batch, -1, 9)

            diffuse_albedo_nn_values = ret['diffuse_albedo_nn_values'].view(n_batch, -1, 3)
            roughness_nn_values = ret['roughness_nn_values'].expand(-1, 9).view(n_batch, -1, 9)

            vis_shadow = ret['vis_shadow'].view(n_batch, -1, 3)
            random_xi_diffuse_albedo = ret['random_xi_diffuse_albedo'].view(n_batch, -1, 3)
            random_xi_roughness = ret['random_xi_roughness'].expand(-1, 9).view(n_batch, -1, 9)
        
        # visualize 3 channals(1, 4, 7) of roughness
        else:
            sg_diffuse_rgb_values = torch.zeros([n_batch, n_point, 3]).to(wpts)
            sg_diffuse_rgb_values[pind] = ret['sg_diffuse_rgb_values']
            sg_specular_rgb_values = torch.zeros([n_batch, n_point, 3]).to(wpts)
            sg_specular_rgb_values[pind] = ret['sg_specular_rgb_values']

            normal_values = torch.zeros([n_batch, n_point, 3]).to(wpts)
            normal_values[pind] = ret['normal_values']
            diffuse_albedo_values = torch.zeros([n_batch, n_point, 3]).to(wpts)
            diffuse_albedo_values[pind] = ret['diffuse_albedo_values']
            roughness_values = torch.zeros([n_batch, n_point, 3]).to(wpts)
            roughness_values[pind] = ret['roughness_values'][..., [1, 4, 7]]

            diffuse_albedo_nn_values = ret['diffuse_albedo_nn_values'].view(n_batch, -1, 3)
            roughness_nn_values = ret['roughness_nn_values'][..., [1, 4, 7]].view(n_batch, -1, 3)

            vis_shadow = torch.zeros([n_batch, n_point, 3]).to(wpts)
            vis_shadow[pind] = ret['vis_shadow']
            random_xi_diffuse_albedo = torch.zeros([n_batch, n_point, 3]).to(wpts)
            random_xi_diffuse_albedo[pind] = ret['random_xi_diffuse_albedo']
            random_xi_roughness = torch.zeros([n_batch, n_point, 3]).to(wpts)
            random_xi_roughness[pind] = ret['random_xi_roughness'][..., [1, 4, 7]]

        brdf_latent = ret['brdf_latent'].view(n_batch, -1, 32)

        ret.update({'sg_diffuse_rgb_values': sg_diffuse_rgb_values, 
                    'sg_specular_rgb_values': sg_specular_rgb_values, 'normal_values': normal_values, 
                    'diffuse_albedo_values': diffuse_albedo_values, 'roughness_values': roughness_values, 
                    'diffuse_albedo_nn_values': diffuse_albedo_nn_values, 'roughness_nn_values': roughness_nn_values, 
                    'vis_shadow': vis_shadow, 'random_xi_diffuse_albedo': random_xi_diffuse_albedo, 
                    'random_xi_roughness': random_xi_roughness, 'brdf_latent': brdf_latent})

        return ret


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.sdf_network = SDFNetwork()
        self.beta_network = BetaNetwork()
        self.color_sg_network = ColorSGNetwork()

    def sdf_to_alpha(self, sdf, beta):
        x = -sdf

        # select points whose x is smaller than 0: 1 / beta * 0.5 * exp(x/beta)
        ind0 = x <= 0
        val0 = 1 / beta * (0.5 * torch.exp(x[ind0] / beta))

        # select points whose x is bigger than 0: 1 / beta * (1 - 0.5 * exp(-x/beta))
        ind1 = x > 0
        val1 = 1 / beta * (1 - 0.5 * torch.exp(-x[ind1] / beta))

        val = torch.zeros_like(sdf)
        val[ind0] = val0
        val[ind1] = val1

        return val

    def get_alpha(self, wpts, dists, batch):
        # calculate alpha
        wpts = wpts.detach()
        sdf_nn_output = self.sdf_network(wpts, batch)
        sdf = sdf_nn_output[:, :1]
        beta = self.beta_network(wpts).clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(sdf, beta)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        return alpha

    def forward(self, wpts, posed_pts, viewdir, dists, batch, pbw):
        # calculate sdf
        wpts.requires_grad_()
        with torch.enable_grad():
            sdf_nn_output = self.sdf_network(wpts, batch)
            sdf = sdf_nn_output[:, :1]

        # calculate normal
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        gradients = pose_dirs_to_tpose_dirs(gradients[None], pbw, batch['big_A'])
        gradients = tpose_dirs_to_pose_dirs(gradients, pbw, batch['A'])
        gradients = torch.matmul(gradients, batch['R'].transpose(1, 2))
        gradients = gradients[0]

        # calculate alpha
        wpts = wpts.detach()
        beta = self.beta_network(wpts).clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(sdf, beta)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        rot_w2t = pose_dirs_to_tpose_dirs_Rot(pbw, batch['A'])
        rot_t2big = tpose_dirs_to_pose_dirs_Rot(pbw, batch['big_A'])
        rot_w2big = torch.einsum('ijk,ikl->ijl', rot_t2big, rot_w2t)

        j_transform = batch['joints_transform']
        poses = batch['poses']
        color_sg_ret = self.color_sg_network(wpts, gradients, viewdir, posed_pts, j_transform, poses, rot_w2big)

        rgb = color_sg_ret['sg_rgb']

        sg_diffuse_rgb_values = color_sg_ret['sg_diffuse_rgb']
        sg_specular_rgb_values = color_sg_ret['sg_specular_rgb']

        normal_values = color_sg_ret['normals']
        diffuse_albedo_values = color_sg_ret['diffuse_albedo']
        roughness_values = color_sg_ret['roughness']

        diffuse_albedo_nn_values = color_sg_ret['diffuse_albedo_nn']
        roughness_nn_values = color_sg_ret['roughness_nn']

        vis_shadow = color_sg_ret['vis_shadow']

        random_xi_diffuse_albedo = color_sg_ret['random_xi_diffuse_albedo']
        random_xi_roughness = color_sg_ret['random_xi_roughness']

        brdf_latent = color_sg_ret['brdf_latent']

        raw = torch.cat((rgb, alpha[:, None]), dim=1)

        ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients, 
                'sg_diffuse_rgb_values': sg_diffuse_rgb_values,
                'sg_specular_rgb_values': sg_specular_rgb_values, 
                'normal_values': normal_values,
                'diffuse_albedo_values': diffuse_albedo_values, 
                'roughness_values': roughness_values, 
                'diffuse_albedo_nn_values': diffuse_albedo_nn_values, 
                'roughness_nn_values': roughness_nn_values, 
                'vis_shadow': vis_shadow, 
                'random_xi_diffuse_albedo': random_xi_diffuse_albedo, 
                'random_xi_roughness': random_xi_roughness, 
                'brdf_latent': brdf_latent}

        return ret


def hemisphere_int(lambda_val, cos_beta):
    lambda_val = lambda_val + TINY_NUMBER
    
    inv_lambda_val = 1. / lambda_val
    t = torch.sqrt(lambda_val) * (1.6988 + 10.8438 * inv_lambda_val) / (
                1. + 6.2201 * inv_lambda_val + 10.2415 * inv_lambda_val * inv_lambda_val)

    ### note: for numeric stability
    inv_a = torch.exp(-t)
    mask = (cos_beta >= 0).float()
    inv_b = torch.exp(-t * torch.clamp(cos_beta, min=0.))
    s1 = (1. - inv_a * inv_b) / (1. - inv_a + inv_b - inv_a * inv_b)
    b = torch.exp(t * torch.clamp(cos_beta, max=0.))
    s2 = (b - inv_a) / ((1. - inv_a) * (b + 1.))
    s = mask * s1 + (1. - mask) * s2

    A_b = 2. * np.pi / lambda_val * (torch.exp(-lambda_val) - torch.exp(-2. * lambda_val))
    A_u = 2. * np.pi / lambda_val * (1. - torch.exp(-lambda_val))

    return A_b * (1. - s) + A_u * s


def lambda_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    # assume lambda1 << lambda2
    ratio = lambda1 / lambda2

    # for insurance
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)
    dot = torch.sum(lobe1 * lobe2, dim=-1, keepdim=True)
    tmp = torch.sqrt(ratio * ratio + 1. + 2. * ratio * dot)
    tmp = torch.min(tmp, ratio + 1.)

    lambda3 = lambda2 * tmp
    lambda1_over_lambda3 = ratio / tmp
    lambda2_over_lambda3 = 1. / tmp
    diff = lambda2 * (tmp - ratio - 1.)

    final_lobes = lambda1_over_lambda3 * lobe1 + lambda2_over_lambda3 * lobe2
    final_lambdas = lambda3
    final_mus = mu1 * mu2 * torch.exp(diff)

    return final_lobes, final_lambdas, final_mus


def lambda_no_trick(lobe1, lambda1, mu1, lobe2, lambda2, mu2):
    lobe1 = norm_axis(lobe1)
    lobe2 = norm_axis(lobe2)

    final_lobes_temp = (lambda1 * lobe1 + lambda2 * lobe2) / (lambda1 + lambda2)
    final_lobes_norm = torch.norm(final_lobes_temp, dim=-1, keepdim=True)

    lambda_temp = lambda1 + lambda2

    final_lobes = final_lobes_temp / final_lobes_norm
    final_lambdas = lambda_temp * final_lobes_norm
    final_mus = mu1 * mu2 * torch.exp(lambda_temp * (final_lobes_norm - 1.0))

    return final_lobes, final_lambdas, final_mus


def norm_axis(x):
    return x / (torch.norm(x, dim=-1, keepdim=True) + TINY_NUMBER)


def get_diffuse_visibility(points, normals, j_transform, root_rot, poses, rot_w2big, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=8, min_vis=False):
    ########################################
    # sample dirs according to the light SG
    ########################################

    n_lobe = lgtSGLobes.shape[0]
    n_points = points.shape[0]
    light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    # add samples from SG lobes
    z_axis = torch.zeros_like(light_dirs).cuda()
    z_axis[:, :, 2] = 1

    light_dirs = norm_axis(light_dirs) #[num_lobes, 1, 3]
    U = norm_axis(torch.cross(z_axis, light_dirs))
    V = norm_axis(torch.cross(light_dirs, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    r_phi_range = torch.clip(torch.arccos(1 - 1.0 / sharpness), max=np.pi / 3)

    r_theta_random = torch.rand(n_points, n_lobe, nsamp) / nsamp
    r_theta_uniform = torch.linspace(0, (nsamp - 1) / nsamp, nsamp).expand(n_points, n_lobe, nsamp)
    r_theta = (r_theta_random + r_theta_uniform).cuda() * 2 * np.pi
    r_phi = torch.ones(n_points, n_lobe, nsamp).cuda() * r_phi_range

    U = U.unsqueeze(0).expand(n_points, -1, nsamp, -1)
    V = V.unsqueeze(0).expand(n_points, -1, nsamp, -1)
    light_dirs = light_dirs.unsqueeze(0).expand(n_points, -1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, -1, 3)

    # [n_points, n_lobes, n_samples, 3]
    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + light_dirs * torch.cos(r_phi)

    ########################################
    # visibility
    ########################################
    sample_dir = sample_dir.reshape(-1, n_lobe * nsamp, 3)
    input_p = points.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    normals = normals.unsqueeze(1).expand(-1, n_lobe * nsamp, 3)
    # if cos < 0, it is invisible
    cos_term = torch.sum(normals * sample_dir, dim=-1) > TINY_NUMBER
    
    input_dir = torch.matmul(-sample_dir.reshape(-1, 3), root_rot)
    input_dir = input_dir.view(n_points, n_lobe * nsamp, 3)

    if not cfg.train.use_part_wise:
        input_dir = torch.einsum('ijk,ilk->ilj', rot_w2big, input_dir)

    batch_size = 400000
    n_mask_dir = input_p[cos_term].shape[0]
    pred_vis = torch.zeros(n_mask_dir).cuda()

    # print(input_p.shape)
    # print(n_mask_dir)

    with torch.no_grad():
        if n_mask_dir > 0:
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                if cfg.train.use_vis:
                    pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx], j_transform, poses)
                else:
                    pred_vis[indx] = 1.0

    vis = torch.zeros(n_points, n_lobe * nsamp).cuda()
    vis[cos_term] = pred_vis.float()

    # avoid zero visibility
    if min_vis:
        vis = vis.reshape(n_points, n_lobe * nsamp)
        vis_sum = torch.sum(vis, dim=-1)
        min_vis_scale = n_lobe * nsamp / 10.0
        low_vis_mask = vis_sum < min_vis_scale
        vis_sum_mask = vis_sum[low_vis_mask].unsqueeze(-1)
        if vis_sum_mask.shape[0] > 0:
            vis[low_vis_mask, :] = vis[low_vis_mask, :] * min_vis_scale / (vis_sum_mask + 1e-6)

    vis = vis.reshape(n_points, n_lobe, nsamp)
    
    sample_dir = sample_dir.reshape(n_points, n_lobe, nsamp, 3)
    lgtSGLambdas = lgtSGLambdas.reshape(1, n_lobe, 1)

    weight_vis = torch.exp(lgtSGLambdas * (torch.sum(sample_dir * light_dirs, dim=-1, keepdim=False) - 1.))
    vis = torch.sum(vis * weight_vis, dim=-1) / (torch.sum(weight_vis, dim=-1) + TINY_NUMBER)
    return vis


def get_specular_visibility(points, normals, viewdirs, j_transform, root_rot, poses, rot_w2big, VisModel, lgtSGLobes, lgtSGLambdas, nsamp=24):
    ########################################
    # sample dirs according to the BRDF SG
    ########################################

    light_dirs = lgtSGLobes.clone().detach().unsqueeze(-2)
    lgtSGLambdas = lgtSGLambdas.clone().detach().unsqueeze(-2)

    n_dot_v = torch.sum(normals * viewdirs, dim=-1, keepdim=True)
    n_dot_v = torch.clamp(n_dot_v, min=0.)
    ref_dir = -viewdirs + 2 * n_dot_v * normals
    ref_dir = ref_dir.unsqueeze(1)
    
    # add samples from BRDF SG lobes
    z_axis = torch.zeros_like(ref_dir).cuda()
    z_axis[:, :, 2] = 1

    U = norm_axis(torch.cross(z_axis, ref_dir))
    V = norm_axis(torch.cross(ref_dir, U))
    # r_phi depends on the sg sharpness
    sharpness = lgtSGLambdas[:, :, 0]
    sharpness = torch.clip(sharpness, min=1.0, max=1e8)
    r_phi_range = torch.arccos(1 - 1.0 / sharpness)
    r_phi_range = torch.clip(torch.arccos(1 - 1.0 / sharpness), max=np.pi / 3)

    n_lobe = ref_dir.shape[0]
    r_theta_random = torch.rand(n_lobe, nsamp) / nsamp
    r_theta_uniform = torch.linspace(0, (nsamp - 1) / nsamp, nsamp).expand(n_lobe, nsamp)
    r_theta = (r_theta_random + r_theta_uniform).cuda() * 2 * np.pi
    r_phi = torch.ones(n_lobe, nsamp).cuda() * r_phi_range

    U = U.expand(-1, nsamp, -1)
    V = V.expand(-1, nsamp, -1)
    r_theta = r_theta.unsqueeze(-1).expand(-1, -1, 3)
    r_phi = r_phi.unsqueeze(-1).expand(-1, -1, 3)

    sample_dir = U * torch.cos(r_theta) * torch.sin(r_phi) \
                + V * torch.sin(r_theta) * torch.sin(r_phi) \
                + ref_dir * torch.cos(r_phi)

    batch_size = 400000
    input_p = points.unsqueeze(1).expand(-1, nsamp, 3)
    input_dir = sample_dir
    normals = normals.unsqueeze(1).expand(-1, nsamp, 3)
    # if cos < 0, it is invisible
    cos_term = torch.sum(normals * input_dir, dim=-1) > TINY_NUMBER
    n_mask_dir = input_p[cos_term].shape[0]
    pred_vis = torch.zeros(n_mask_dir).cuda()

    input_dir = torch.matmul(-input_dir, root_rot)
    if not cfg.train.use_part_wise:
        pt_num_ori = rot_w2big.shape[0]
        input_dir = torch.einsum('ijk,limk->limj', rot_w2big, input_dir.view((-1, pt_num_ori, nsamp, 3)))
        input_dir = input_dir.reshape(-1, nsamp, 3)

    with torch.no_grad():
        if n_mask_dir > 0:
            for i, indx in enumerate(torch.split(torch.arange(n_mask_dir).cuda(), batch_size, dim=0)):
                if cfg.train.use_vis:
                    pred_vis[indx] = VisModel(input_p[cos_term][indx], input_dir[cos_term][indx], j_transform, poses)
                else:
                    pred_vis[indx] = 1.0

    vis = torch.zeros(points.shape[0], nsamp).cuda()
    vis[cos_term] = pred_vis.float()

    weight_vis = torch.exp(sharpness * (torch.sum(sample_dir * light_dirs, dim=-1) - 1.))
    inf_idx = torch.isinf(torch.sum(weight_vis, dim=-1))
    inf_sample = weight_vis[inf_idx]

    reset_inf = inf_sample.clone()
    reset_inf[torch.isinf(inf_sample)] = 1.0
    reset_inf[~torch.isinf(inf_sample)] = 0.0
    weight_vis[inf_idx] = reset_inf

    vis = torch.sum(vis * weight_vis, dim=-1) / (torch.sum(weight_vis, dim=-1) + TINY_NUMBER)

    return vis


def render_with_all_sg(points, normal, viewdirs, posed_pts, j_transform, poses, rot_w2big, lgtSGs, 
                       specular_reflectance, roughness, diffuse_albedo, roughness_basis, VisModel):

    M = lgtSGs.shape[0]
    dots_shape = list(normal.shape[:-1])

    # env light
    lgtSGs = lgtSGs.unsqueeze(0).expand(dots_shape + [M, 7])  # [dots_shape, M, 7]
    
    ret = render_with_sg(points, normal, viewdirs, posed_pts, j_transform, poses, rot_w2big, lgtSGs, 
                         specular_reflectance, roughness, diffuse_albedo, roughness_basis, VisModel)

    return ret

    
#######################################################################################################
# below is the SG renderer
#######################################################################################################
def render_with_sg(points, normal, viewdirs, posed_pts, j_transform, poses, rot_w2big, 
                   lgtSGs, specular_reflectance, roughness, diffuse_albedo, roughness_basis, VisModel):
    '''
    :param points: [batch_size, 3]
    :param normal: [batch_size, 3]; ----> camera; must have unit norm
    :param viewdirs: [batch_size, 3]; ----> camera; must have unit norm
    :param lgtSGs: [batch_size, M, 7]
    :param specular_reflectance: [1, 1]; 
    :param roughness: [batch_size, 1]; values must be positive
    :param diffuse_albedo: [batch_size, 3]; values must lie in [0,1]
    '''

    if cfg.train.use_part_wise:
        pts = posed_pts
    else:
        pts = points[None]

    M = lgtSGs.shape[1]
    dots_shape = list(normal.shape[:-1])

    ########################################
    # light
    ########################################

    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True) + TINY_NUMBER)
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4]) # sharpness
    origin_lgtSGMus = torch.abs(lgtSGs[..., -3:])  # positive values
    
    ########################################
    # specular color
    ########################################
    normal = normal.unsqueeze(-2).expand(dots_shape + [M, 3])  # [dots_shape, M, 3]
    viewdirs = viewdirs.unsqueeze(-2).expand(dots_shape + [M, 3]).detach()  # [dots_shape, M, 3]
    
    # limiting the intensity of specular reflection
    roughness_scale = roughness
    roughness_scale_norm = torch.sum(roughness_scale, dim=-1, keepdim=True)
    roughness_scale_norm = torch.maximum(roughness_scale_norm * 40.0, torch.ones_like(roughness_scale_norm))
    roughness_scale = roughness_scale / roughness_scale_norm
    roughness_scale = roughness_scale.transpose(0, 1).reshape((-1, 1))

    n_basis = roughness_basis.shape[0]
    roughness_values = roughness_basis.expand(dots_shape + [n_basis])
    roughness_values = roughness_values.transpose(0, 1).reshape((-1 ,1))

    dots_shape_basis = list(np.array(dots_shape) * n_basis)

    # expand the size
    normal_basis = normal[:, 0].expand([n_basis] + dots_shape + [3])
    normal_basis = normal_basis.reshape((-1, 3))

    viewdirs_basis = viewdirs[:, 0].expand([n_basis] + dots_shape + [3])
    viewdirs_basis = viewdirs_basis.reshape((-1, 3))

    pts_basis = pts.expand([n_basis] + dots_shape + [3])
    pts_basis = pts_basis.reshape((-1, 3))

    origin_lgtSGMus_basis = origin_lgtSGMus[:, 0].unsqueeze(0).expand([n_basis] + dots_shape + [3])
    origin_lgtSGMus_basis = origin_lgtSGMus_basis.reshape([-1, 3])

    lgtSGLobes_basis = lgtSGLobes[:, 0].unsqueeze(0).expand([n_basis] + dots_shape + [3])
    lgtSGLobes_basis = lgtSGLobes_basis.reshape([-1, 3])
    lgtSGLambdas_basis = lgtSGLambdas[:, 0].unsqueeze(0).expand([n_basis] + dots_shape + [1])
    lgtSGLambdas_basis = lgtSGLambdas_basis.reshape([-1, 1])

    # NDF
    brdfSGLobes = normal_basis  # use normal as the brdf SG lobes
    inv_roughness_pow4 = 2. / (roughness_values * roughness_values * roughness_values * roughness_values)

    brdfSGLambdas = inv_roughness_pow4
    mu_val = (inv_roughness_pow4 / np.pi)
    brdfSGMus = mu_val

    # perform spherical warping
    v_dot_lobe = torch.sum(brdfSGLobes * viewdirs_basis, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_lobe = torch.clamp(v_dot_lobe, min=0.)
    warpBrdfSGLobes = 2 * v_dot_lobe * brdfSGLobes - viewdirs_basis # reflection output dir
    warpBrdfSGLobes = warpBrdfSGLobes / (torch.norm(warpBrdfSGLobes, dim=-1, keepdim=True) + TINY_NUMBER)
    warpBrdfSGLambdas = brdfSGLambdas / (4 * v_dot_lobe + TINY_NUMBER)
    warpBrdfSGMus = brdfSGMus  # [..., M, 3]

    new_half = warpBrdfSGLobes + viewdirs_basis
    new_half = new_half / (torch.norm(new_half, dim=-1, keepdim=True) + TINY_NUMBER)
    v_dot_h = torch.sum(viewdirs_basis * new_half, dim=-1, keepdim=True)
    ### note: for numeric stability
    v_dot_h = torch.clamp(v_dot_h, min=0.)

    specular_reflectance = specular_reflectance.expand(dots_shape_basis + [3])
    F = specular_reflectance + (1. - specular_reflectance) * torch.pow(2.0, -(5.55473 * v_dot_h + 6.8316) * v_dot_h)

    dot1 = torch.sum(warpBrdfSGLobes * normal_basis, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot1 = torch.clamp(dot1, min=0.)
    dot2 = torch.sum(viewdirs_basis * normal_basis, dim=-1, keepdim=True)  # equals <o, n>
    ### note: for numeric stability
    dot2 = torch.clamp(dot2, min=0.)
    k = (roughness_values + 1.) * (roughness_values + 1.) / 8.
    G1 = dot1 / (dot1 * (1 - k) + k + TINY_NUMBER)  # k<1 implies roughness < 1.828
    G2 = dot2 / (dot2 * (1 - k) + k + TINY_NUMBER)
    G = G1 * G2

    Moi = F * G / (4 * dot1 * dot2 + TINY_NUMBER)
    warpBrdfSGMus = warpBrdfSGMus * Moi

    # light occlcusion
    root_rot = torch.clone(j_transform[0, 0, :3, :3])
    j_transform_input = torch.clone(j_transform[0])
    j_transform_input[0, :3, :3] = torch.eye(3, device=j_transform.device)

    # light SG visibility
    min_vis = diffuse_albedo.requires_grad
    light_vis = get_diffuse_visibility(pts[0], normal[:, 0, :], j_transform_input, 
                                        root_rot, poses[0], rot_w2big, VisModel,
                                        lgtSGLobes[0], lgtSGLambdas[0], nsamp=4, min_vis=min_vis)
    light_vis = light_vis.unsqueeze(-1).expand(dots_shape + [M, 3])

    # BRDF SG visibility
    brdf_vis = get_specular_visibility(pts_basis, normal_basis, viewdirs_basis, 
                                    j_transform_input, root_rot, poses[0], rot_w2big, 
                                    VisModel, warpBrdfSGLobes, warpBrdfSGLambdas, nsamp=16)
    
    brdf_vis = brdf_vis.unsqueeze(-1).unsqueeze(-1).expand(dots_shape_basis + [M, 3])
    light_vis_basis = light_vis.unsqueeze(0).expand([n_basis] + dots_shape + [M, 3])
    light_vis_basis = light_vis_basis.reshape([-1, M, 3])

    lgtSGMus = origin_lgtSGMus_basis[:, None, :] * brdf_vis * light_vis_basis

    vis_shadow = torch.mean(light_vis * origin_lgtSGMus, axis=1).squeeze()

    # multiply with light sg
    lgtSGLobes_basis = lgtSGLobes_basis.unsqueeze(1).expand(dots_shape_basis + [M, 3])
    lgtSGLambdas_basis = lgtSGLambdas_basis.unsqueeze(1).expand(dots_shape_basis + [M, 1])
    warpBrdfSGLobes = warpBrdfSGLobes.unsqueeze(1).expand(dots_shape_basis + [M, 3])
    warpBrdfSGLambdas = warpBrdfSGLambdas.unsqueeze(1).expand(dots_shape_basis + [M, 1])
    warpBrdfSGMus = warpBrdfSGMus.unsqueeze(1).expand(dots_shape_basis + [M, 3])

    normal_basis = normal_basis.unsqueeze(1).expand(dots_shape_basis + [M, 3])

    final_lobes, final_lambdas, final_mus = lambda_no_trick(lgtSGLobes_basis, lgtSGLambdas_basis, lgtSGMus,
                                                            warpBrdfSGLobes, warpBrdfSGLambdas, warpBrdfSGMus)

    # now multiply with clamped cosine, and perform hemisphere integral
    mu_cos = 32.7080
    lambda_cos = 0.0315
    alpha_cos = 31.7003
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal_basis, lambda_cos, mu_cos,
                                                    final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal_basis, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal_basis, dim=-1, keepdim=True)

    specular_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    specular_rgb = specular_rgb.sum(dim=-2) * roughness_scale
    specular_rgb = specular_rgb.reshape(([n_basis] + dots_shape + [3]))
    specular_rgb = specular_rgb.sum(dim=0, keepdim=False)
    specular_rgb = torch.clamp(specular_rgb, min=0.)

    ########################################
    # per-point hemisphere integral of envmap
    ########################################
    # diffuse visibility
    lgtSGMus = origin_lgtSGMus * light_vis
    
    # multiply with light sg
    final_lobes = lgtSGLobes
    final_lambdas = lgtSGLambdas
    final_mus = lgtSGMus / np.pi

    # now multiply with clamped cosine, and perform hemisphere integral
    lobe_prime, lambda_prime, mu_prime = lambda_trick(normal, lambda_cos, mu_cos,
                                                      final_lobes, final_lambdas, final_mus)

    dot1 = torch.sum(lobe_prime * normal, dim=-1, keepdim=True)
    dot2 = torch.sum(final_lobes * normal, dim=-1, keepdim=True)
    diffuse_rgb = mu_prime * hemisphere_int(lambda_prime, dot1) - \
                    final_mus * alpha_cos * hemisphere_int(final_lambdas, dot2)
    diffuse_rgb = diffuse_rgb.sum(dim=-2)
    diffuse_rgb = torch.clamp(diffuse_rgb, min=0.)

    # assume predicted albedo in sRGB space, and transform it to the linear space
    diffuse_rgb = diffuse_rgb * torch.pow(diffuse_albedo, 2.2)

    # combine diffue and specular rgb
    rgb = specular_rgb + diffuse_rgb
    ret = {'sg_rgb': rgb,
           'sg_specular_rgb': specular_rgb,
           'sg_diffuse_rgb': diffuse_rgb,
           'vis_shadow': vis_shadow}

    return ret


class ColorSGNetwork(nn.Module):
    def __init__(self):
        super(ColorSGNetwork, self).__init__()

        # roughness basis
        self.num_roughness_basis = 9

        self.envmap_material_network = EnvmapMaterialNetwork(multires=10,
                                                            brdf_encoder_dims=[ 512, 512, 512, 512 ], 
                                                            brdf_decoder_dims=[ 128, 128 ], 
                                                            num_lgt_sgs=128, 
                                                            upper_hemi=False, 
                                                            specular_albedo=0.02, 
                                                            latent_dim=32, 
                                                            num_roughness_basis=self.num_roughness_basis)
        
        self.roughness_basis = nn.Parameter(torch.from_numpy(np.linspace(0.20, 0.99, self.num_roughness_basis, dtype=np.float32)))
        self.roughness_basis.requires_grad_(False)

        if cfg.train.use_part_wise:
            self.visibility_network = VisNetwork()
        else:
            self.visibility_network = TotalVisNetwork()

    def get_sg_render(self, points, view_dirs, normals, posed_pts, j_transform, poses, rot_w2big):
        view_dirs = view_dirs / (torch.norm(view_dirs, dim=-1, keepdim=True) + 1e-6)
        normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
        ret = { 'normals': normals, }

        # sg renderer
        sg_envmap_material = self.envmap_material_network(points)

        sg_ret = render_with_all_sg(points=points.detach(),
                                    normal=normals.detach(), 
                                    viewdirs=view_dirs, 
                                    posed_pts=posed_pts.detach(), 
                                    j_transform=j_transform.detach(), 
                                    poses=poses.detach(), 
                                    rot_w2big=rot_w2big.detach(), 
                                    lgtSGs=sg_envmap_material['sg_lgtSGs'],
                                    specular_reflectance=sg_envmap_material['sg_specular_reflectance'],
                                    roughness=sg_envmap_material['sg_roughness'],
                                    diffuse_albedo=sg_envmap_material['sg_diffuse_albedo'],
                                    roughness_basis=self.roughness_basis, 
                                    VisModel=self.visibility_network)
        ret.update(sg_ret)
        ret.update({'diffuse_albedo': sg_envmap_material['sg_diffuse_albedo'],
                    'roughness': sg_envmap_material['sg_roughness'],
                    'diffuse_albedo_nn': sg_envmap_material['sg_diffuse_albedo_nn'],
                    'roughness_nn': sg_envmap_material['sg_roughness_nn'],
                    'random_xi_roughness': sg_envmap_material['random_xi_roughness'],
                    'random_xi_diffuse_albedo': sg_envmap_material['random_xi_diffuse_albedo']})
        ret.update({'brdf_latent': sg_envmap_material['brdf_latent']})

        return ret

    def forward(self, points, normals, view_dirs, posed_pts, j_transform, poses, rot_w2big):

        ret = self.get_sg_render(points, 
                                view_dirs,
                                normals,  
                                posed_pts, 
                                j_transform, 
                                poses, 
                                rot_w2big)

        return ret
