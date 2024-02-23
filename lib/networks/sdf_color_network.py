import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from lib.config import cfg
from lib.utils.blend_utils import *
from . import embedder
from .models import *
from lib.utils import net_utils
import os
from lib.utils import sample_utils
from lib.utils.body_model import lbs
from lib.utils.body_model.body_model import BodyModel


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
        if self.training and cfg.train.warm_up:
            iter_step = batch['iter_step']
            if iter_step < cfg.train.warm_up_iters:
                resd = torch.zeros_like(tpose)
                return resd

        latent = batch['poses'].detach()
        tpose_input = tpose.detach()
        if self.training:
            latent = latent + torch.randn(size=latent.shape, device=latent.device) * 0.01
            tpose_input = tpose_input + torch.randn(size=tpose_input.shape, device=tpose_input.device) * 0.02
        input_bw = self.get_tpose_skinning(tpose_input, batch)

        tpose_deformed = self.deform_net(latent[0].T, tpose[0], input_bw[0])

        resd = tpose_deformed[None] - tpose

        return resd
        
    def calculate_residual_deformation_can2deformed(self, tpose, batch):
        if self.training and cfg.train.warm_up:
            iter_step = batch['iter_step']
            if iter_step < cfg.train.warm_up_iters:
                return tpose

        latent = batch['poses'].detach()
        tpose_input = tpose.detach()
        if self.training:
            latent = latent + torch.randn(size=latent.shape, device=latent.device) * 0.01
            tpose_input = tpose_input + torch.randn(size=tpose_input.shape, device=tpose_input.device) * 0.02
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
        if self.training and cfg.train.warm_up:
            iter_step = batch['iter_step']
            if iter_step > cfg.train.warm_up_iters:
                pbw = sample_utils.sample_blend_closest_points_single(pose_pts, mesh_vertices, batch['mesh_tbw'])
            else:
                pbw, _ = sample_utils.sample_blend_closest_points(pose_pts, batch['pvertices'], batch['weights'])
        else:
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

    def gradient_of_deformed_sdf(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            resd = self.calculate_residual_deformation(x, batch)
            tpose = x + resd
            tpose = tpose[0]
            y = self.tpose_human.sdf_network(tpose, batch)[:, :1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients, y[None]

    def forward(self, wpts, viewdir, dists, batch):
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
            norm_th = 0.1
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
        
        if cfg.tpose_viewdir:
            viewdir = tpose_dirs[0]
        else:
            viewdir = viewdir[0]

        ret = self.tpose_human(tpose, viewdir, dists, batch, pbw)

        ind = ret['sdf'][:, 0].detach().abs() < 0.02
        init_bigpose = init_bigpose[0][ind][None].detach().clone()

        if ret['raw'].requires_grad and ind.sum() != 0:
            observed_gradients, _ = self.gradient_of_deformed_sdf(
                init_bigpose, batch)
            ret.update({'observed_gradients': observed_gradients})

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

        return ret


class TPoseHuman(nn.Module):
    def __init__(self):
        super(TPoseHuman, self).__init__()

        self.sdf_network = SDFNetwork()
        self.beta_network = BetaNetwork()
        self.color_network = ColorNetwork()

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

    def forward(self, wpts, viewdir, dists, batch, pbw):
        # calculate sdf
        wpts.requires_grad_()
        with torch.enable_grad():
            sdf_nn_output = self.sdf_network(wpts, batch)
            sdf = sdf_nn_output[:, :1]

        feature_vector = sdf_nn_output[:, 1:]

        # calculate normal
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(outputs=sdf,
                                        inputs=wpts,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]

        # calculate alpha
        wpts = wpts.detach()
        beta = self.beta_network(wpts).clamp(1e-9, 1e6)
        alpha = self.sdf_to_alpha(sdf, beta)
        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * 0.005)
        alpha = raw2alpha(alpha[:, 0], dists)

        # calculate color
        ind = batch['latent_index']
        rgb = self.color_network(wpts, gradients, viewdir, feature_vector, ind)

        raw = torch.cat((rgb, alpha[:, None]), dim=1)
        ret = {'raw': raw, 'sdf': sdf, 'gradients': gradients}

        return ret


