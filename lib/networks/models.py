import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from lib.config import cfg
import pickle
from . import embedder


class SDFNetwork(nn.Module):
    def __init__(self):
        super(SDFNetwork, self).__init__()

        d_in = 3
        d_out = 257
        d_hidden = 256
        n_layers = 8

        dims = [d_in] + [d_hidden for _ in range(n_layers)] + [d_out]

        self.embed_fn_fine = None

        multires = 6
        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder(multires,
                                                       input_dims=d_in)
            self.embed_fn_fine = embed_fn
            dims[0] = input_ch

        skip_in = [4]
        bias = 0.5
        scale = 1
        geometric_init = True
        weight_norm = True
        activation = 'softplus'

        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.scale = scale

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight,
                                          mean=np.sqrt(np.pi) /
                                          np.sqrt(dims[l]),
                                          std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):],
                                            0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0,
                                          np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        if activation == 'softplus':
            self.activation = nn.Softplus(beta=100)
        else:
            assert activation == 'relu'
            self.activation = nn.ReLU()

    def forward(self, inputs, batch):
        inputs = inputs * self.scale
        if self.embed_fn_fine is not None:
            inputs = self.embed_fn_fine(inputs)

        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.activation(x)
        return torch.cat([x[:, :1] / self.scale, x[:, 1:]], dim=-1)

    def sdf(self, x, batch):
        return self.forward(x, batch)[:, :1]

    def gradient(self, x, batch):
        x.requires_grad_(True)
        with torch.enable_grad():
            y = self.sdf(x, batch)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(outputs=y,
                                        inputs=x,
                                        grad_outputs=d_output,
                                        create_graph=True,
                                        retain_graph=True,
                                        only_inputs=True)[0]
        return gradients.unsqueeze(1)


class BetaNetwork(nn.Module):
    def __init__(self):
        super(BetaNetwork, self).__init__()
        init_val = 0.1
        self.register_parameter('beta', nn.Parameter(torch.tensor(init_val)))

    def set_bata(self, val):
        self.beta = val

    def forward(self, x):
        beta = self.beta
        return beta


class ColorNetwork(nn.Module):
    def __init__(self):
        super(ColorNetwork, self).__init__()

        self.color_latent = nn.Embedding(cfg.num_latent_code, 128)

        d_feature = 256
        d_in = 9
        d_out = 3
        d_hidden = 256
        n_layers = 4
        squeeze_out = True

        self.squeeze_out = squeeze_out
        dims = [d_in + d_feature] + [d_hidden
                                     for _ in range(n_layers)] + [d_out]

        self.embedview_fn = None
        multires_view = 4
        if multires_view > 0:
            embedview_fn, input_ch = embedder.get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        self.lin0 = nn.Linear(dims[0], d_hidden)
        self.lin1 = nn.Linear(d_hidden, d_hidden)
        self.lin2 = nn.Linear(d_hidden, d_hidden)
        self.lin3 = nn.Linear(d_hidden + 128, d_hidden)
        self.lin4 = nn.Linear(d_hidden, d_out)

        weight_norm = True
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

        self.relu = nn.ReLU()

    def forward(self, points, normals, view_dirs, feature_vectors,
                latent_index):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        rendering_input = torch.cat(
            [points, view_dirs, normals, feature_vectors], dim=-1)

        x = rendering_input

        net = self.relu(self.lin0(x))
        net = self.relu(self.lin1(net))
        net = self.relu(self.lin2(net))

        latent = self.color_latent(latent_index)
        latent = latent.expand(net.size(0), latent.size(1))
        features = torch.cat((net, latent), dim=1)

        net = self.relu(self.lin3(features))
        x = self.lin4(net)

        if self.squeeze_out:
            x = torch.sigmoid(x)

        return x


# Bidirectional Deformation
class DeformNetwork(nn.Module):
    def __init__(self,
                 d_feature,
                 d_in,
                 d_out_1,
                 d_out_2,
                 n_blocks,
                 d_hidden,
                 n_layers_a,
                 n_layers_b,
                 skip_in=(4,),
                 multires=0,
                 weight_norm=True):
        super(DeformNetwork, self).__init__()
        
        self.n_blocks = n_blocks
        self.skip_in = skip_in

        bw_dim = 24

        # part a
        # xy -> z
        ori_in = d_in - 1
        dims_in = ori_in
        dims = [dims_in + bw_dim + d_feature] + [d_hidden for _ in range(n_layers_a)] + [d_out_1]

        self.embed_fn_1 = None

        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder_anneal(multires, input_dims=dims_in)
            self.embed_fn_1 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature + bw_dim

        self.num_layers_a = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_a - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_a - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_a - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_a_"+str(l), lin)

        # part b
        # z -> xy
        ori_in = 1
        dims_in = ori_in
        dims = [dims_in + bw_dim + d_feature] + [d_hidden for _ in range(n_layers_b)] + [d_out_2]

        self.embed_fn_2 = None

        if multires > 0:
            embed_fn, input_ch = embedder.get_embedder_anneal(multires, input_dims=dims_in)
            self.embed_fn_2 = embed_fn
            dims_in = input_ch
            dims[0] = input_ch + d_feature + bw_dim

        self.num_layers_b = len(dims)
        for i_b in range(self.n_blocks):
            for l in range(0, self.num_layers_b - 1):
                if l + 1 in self.skip_in:
                    out_dim = dims[l + 1] - dims_in
                else:
                    out_dim = dims[l + 1]

                lin = nn.Linear(dims[l], out_dim)

                if l == self.num_layers_b - 2:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight, 0.0)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :ori_in], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, ori_in:], 0.0)
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight[:, :-(dims_in - ori_in)], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims_in - ori_in):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

                if weight_norm and l < self.num_layers_b - 2:
                    lin = nn.utils.weight_norm(lin)

                setattr(self, "lin"+str(i_b)+"_b_"+str(l), lin)

        # latent code
        for i_b in range(self.n_blocks):
            lin = nn.Linear(d_feature, d_feature)
            torch.nn.init.constant_(lin.bias, 0.0)
            torch.nn.init.constant_(lin.weight, 0.0)
            setattr(self, "lin"+str(i_b)+"_c", lin)

        self.activation = nn.Softplus(beta=100)


    def forward(self, deformation_code, input_pts, input_bw, alpha_ratio=1.0):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part a
            if form == 0:
                # zyx
                if mode == 0:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
            else:
                # xyz
                if mode == 0:
                    x_focus = x[:, [0]]
                    x_other = x[:, [1,2]]
                elif mode == 1:
                    x_focus = x[:, [1]]
                    x_other = x[:, [0,2]]
                else:
                    x_focus = x[:, [2]]
                    x_other = x[:, [0,1]]
            x_ori = x_other # xy
            if self.embed_fn_1 is not None:
                x_other = self.embed_fn_1(x_other, alpha_ratio)
            x_other = torch.cat([x_other, input_bw, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_focus = x_focus - 0.05 * torch.tanh(x)

            # part b
            x_focus_ori = x_focus # z'
            if self.embed_fn_2 is not None:
                x_focus = self.embed_fn_2(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, input_bw, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            trans_2d = 0.05 * torch.tanh(x[:, 1:])
            x_other = x_ori - trans_2d

            if form == 0:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_other[:,[0]], x_focus_ori, x_other[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)

        return x


    def inverse(self, deformation_code, input_pts, input_bw, alpha_ratio=1.0):
        batch_size = input_pts.shape[0]
        x = input_pts
        for i_b in range(self.n_blocks):
            i_b = self.n_blocks - 1 - i_b # inverse
            form = (i_b // 3) % 2
            mode = i_b % 3

            lin = getattr(self, "lin"+str(i_b)+"_c")
            deform_code_ib = lin(deformation_code) + deformation_code
            deform_code_ib = deform_code_ib.repeat(batch_size, 1)
            # part b
            if form == 0:
                # axis: z -> y -> x
                if mode == 0:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
            else:
                # axis: x -> y -> z
                if mode == 0:
                    x_focus = x[:, [1,2]]
                    x_other = x[:, [0]]
                elif mode == 1:
                    x_focus = x[:, [0,2]]
                    x_other = x[:, [1]]
                else:
                    x_focus = x[:, [0,1]]
                    x_other = x[:, [2]]
            x_ori = x_other # z'
            if self.embed_fn_2 is not None:
                x_other = self.embed_fn_2(x_other, alpha_ratio)
            x_other = torch.cat([x_other, input_bw, deform_code_ib], dim=-1)
            x = x_other
            for l in range(0, self.num_layers_b - 1):
                lin = getattr(self, "lin"+str(i_b)+"_b_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_other], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_b - 2:
                    x = self.activation(x)

            trans_2d = 0.05 * torch.tanh(x[:, 1:])
            x_focus = x_focus + trans_2d

            # part a
            x_focus_ori = x_focus # xy
            if self.embed_fn_1 is not None:
                x_focus = self.embed_fn_1(x_focus, alpha_ratio)
            x_focus = torch.cat([x_focus, input_bw, deform_code_ib], dim=-1)
            x = x_focus
            for l in range(0, self.num_layers_a - 1):
                lin = getattr(self, "lin"+str(i_b)+"_a_"+str(l))
                if l in self.skip_in:
                    x = torch.cat([x, x_focus], 1) / np.sqrt(2)
                x = lin(x)
                if l < self.num_layers_a - 2:
                    x = self.activation(x)

            x_other = x_ori + 0.05 * torch.tanh(x)
            if form == 0:
                if mode == 0:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
            else:
                if mode == 0:
                    x = torch.cat([x_other, x_focus_ori], dim=-1)
                elif mode == 1:
                    x = torch.cat([x_focus_ori[:,[0]], x_other, x_focus_ori[:,[1]]], dim=-1)
                else:
                    x = torch.cat([x_focus_ori, x_other], dim=-1)

        return x
    


def compute_energy(lgtSGs):
    lgtLambda = torch.abs(lgtSGs[:, 3:4]) 
    lgtMu = torch.abs(lgtSGs[:, 4:]) 
    energy = lgtMu * 2.0 * np.pi / lgtLambda * (1.0 - torch.exp(-2.0 * lgtLambda))
    return energy


def fibonacci_sphere(samples=1):
    '''
    uniformly distribute points on a sphere
    reference: https://github.com/Kai-46/PhySG/blob/master/code/model/sg_envmap_material.py
    '''
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])
    points = np.array(points)
    return points


class EnvmapMaterialNetwork(nn.Module):
    def __init__(self, multires=0, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 num_lgt_sgs=32,
                 upper_hemi=False,
                 specular_albedo=0.02,
                 latent_dim=32, 
                 num_roughness_basis=9):
        super().__init__()

        self.n_basis = num_roughness_basis

        input_dim = 3
        self.embed_fn = None
        if multires > 0:
            self.brdf_embed_fn, brdf_input_dim = embedder.get_embedder(multires, input_dim)

        self.numLgtSGs = num_lgt_sgs
        self.envmap = None

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)
        ############## spatially-varying BRDF ############
        
        # print('BRDF encoder network size: ', brdf_encoder_dims)
        # print('BRDF decoder network size: ', brdf_decoder_dims)

        brdf_encoder_layer = []
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, 3 + self.n_basis))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)

        ############## fresnel ############
        spec = torch.zeros([1, 1])
        spec[:] = specular_albedo
        self.specular_reflectance = nn.Parameter(spec, requires_grad=False)
        
        ################### light SGs ####################
        # print('Number of Light SG: ', self.numLgtSGs)

        # by using normal distribution, the lobes are uniformly distributed on a sphere at initialization
        self.lgtSGs = nn.Parameter(torch.randn(self.numLgtSGs, 7), requires_grad=True)   # [M, 7]; lobe + lambda + mu
        self.lgtSGs.data[:, -2:] = self.lgtSGs.data[:, -3:-2].expand((-1, 2))

        # make sure lambda is not too close to zero
        self.lgtSGs.data[:, 3:4] = 10. + torch.abs(self.lgtSGs.data[:, 3:4] * 20.)
        # init envmap energy
        energy = compute_energy(self.lgtSGs.data)
        self.lgtSGs.data[:, 4:] = torch.abs(self.lgtSGs.data[:, 4:]) / torch.sum(energy, dim=0, keepdim=True) * 2. * np.pi * 0.8
        energy = compute_energy(self.lgtSGs.data)
        # print('init envmap energy: ', torch.sum(energy, dim=0).clone().cpu().numpy())

        # deterministicly initialize lobes
        lobes = fibonacci_sphere(self.numLgtSGs//2).astype(np.float32)
        self.lgtSGs.data[:self.numLgtSGs//2, :3] = torch.from_numpy(lobes)
        self.lgtSGs.data[self.numLgtSGs//2:, :3] = torch.from_numpy(lobes)
        
        # check if lobes are in upper hemisphere
        self.upper_hemi = upper_hemi
        if self.upper_hemi:
            print('Restricting lobes to upper hemisphere!')
            self.restrict_lobes_upper = lambda lgtSGs: torch.cat((lgtSGs[..., :1], torch.abs(lgtSGs[..., 1:2]), lgtSGs[..., 2:]), dim=-1)
            # limit lobes to upper hemisphere
            self.lgtSGs.data = self.restrict_lobes_upper(self.lgtSGs.data)
        
        # for material smooth loss
        self.nn_displacment = 0.01
        self.nn_num = 1
        self.gaussian_noise = 0.01

    def forward(self, points):
        points_nn = torch.cat([points + d for d in torch.randn((self.nn_num, 3)).cuda() * self.nn_displacment], dim=0)

        if self.brdf_embed_fn is not None:
            points = self.brdf_embed_fn(points)

        if self.brdf_embed_fn is not None:
            points_nn = self.brdf_embed_fn(points_nn)

        brdf_latent = self.brdf_encoder_layer(points)
        brdf_lc = torch.sigmoid(brdf_latent)
        brdf = self.brdf_decoder_layer(brdf_lc)
        roughness = torch.sigmoid(brdf[..., 3:])
        diffuse_albedo = torch.sigmoid(brdf[..., :3])

        brdf_latent_nn = self.brdf_encoder_layer(points_nn)
        brdf_lc_nn = torch.sigmoid(brdf_latent_nn)
        brdf_nn = self.brdf_decoder_layer(brdf_lc_nn)
        roughness_nn = torch.sigmoid(brdf_nn[..., 3:])
        diffuse_albedo_nn = torch.sigmoid(brdf_nn[..., :3])

        rand_lc = brdf_lc + torch.randn(brdf_lc.shape).cuda() * self.gaussian_noise
        random_xi_brdf = self.brdf_decoder_layer(rand_lc)
        random_xi_roughness = torch.sigmoid(random_xi_brdf[..., 3:])
        random_xi_diffuse = torch.sigmoid(random_xi_brdf[..., :3])

        lgtSGs = self.lgtSGs
        if self.upper_hemi:
            # limit lobes to upper hemisphere
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        specular_reflectance = self.specular_reflectance
        self.specular_reflectance.requires_grad = False

        ret = dict([
            ('sg_lgtSGs', lgtSGs),
            ('sg_specular_reflectance', specular_reflectance),
            ('sg_roughness', roughness),
            ('sg_diffuse_albedo', diffuse_albedo),
            ('sg_roughness_nn', roughness_nn),
            ('sg_diffuse_albedo_nn', diffuse_albedo_nn),
            ('random_xi_roughness', random_xi_roughness),
            ('random_xi_diffuse_albedo', random_xi_diffuse),
            ('brdf_latent', brdf_latent),
        ])
        return ret

    def get_light(self):
        lgtSGs = self.lgtSGs.clone().detach()
        # limit lobes to upper hemisphere
        if self.upper_hemi:
            lgtSGs = self.restrict_lobes_upper(lgtSGs)

        return lgtSGs


## VisModel for single body part
class PartVisNetwork(nn.Module):
    def __init__(self, points_multires=10, dirs_multires=4, dims=[128, 128, 128], pose_input_dim=3):
        super().__init__()

        p_input_dim = 3
        self.p_embed_fn = None
        if points_multires > 0:
            self.p_embed_fn, p_input_dim = embedder.get_embedder(multires=points_multires, input_dims=p_input_dim)

        dir_input_dim = 3
        self.dir_embed_fn = None
        if dirs_multires > 0:
            self.dir_embed_fn, dir_input_dim = embedder.get_embedder(multires=dirs_multires, input_dims=dir_input_dim)

        self.actv_fn = nn.ReLU()

        vis_layer = []
        dim = p_input_dim + dir_input_dim
        dim += pose_input_dim

        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, 1))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, points, view_dirs, pose_feature):
        if self.p_embed_fn is not None:
            points = self.p_embed_fn(points)
        if self.dir_embed_fn is not None:
            view_dirs = self.dir_embed_fn(view_dirs)
        vis = self.vis_layer(torch.cat([points, view_dirs, pose_feature], -1))
        return vis


## VisModel with part-wise design
class VisNetwork(nn.Module):
    def __init__(self, part_file=cfg.body_part_path, 
                hidden_size=128, num_layers=3):
        super(VisNetwork, self).__init__()
        with open(part_file, 'rb') as f:
            part = pickle.load(f)
            part_root = []
            for part_id in range(len(part)):
                part_root.append(part[part_id][0])
            
        part_num = len(part)

        self.part_root = part_root
        self.part_num = part_num
        self.part = []
        self.part_all = []
        for i in range(self.part_num):
            self.part.append(torch.from_numpy(np.array([part_root[i]]).astype(np.int64)))
            self.part_all.append(torch.from_numpy(np.array(part[i]).astype(np.int64)))

        self.part_net_list = nn.ModuleList()
        for i in range(self.part_num):
            self.part_net_list.append(PartVisNetwork(dims=[hidden_size] * num_layers, pose_input_dim=3 * len(part[i])))

    def forward(self, vert_pos, view_dir, j_transform, poses, return_all_part=False):
        vert_num = vert_pos.shape[0]

        # transform vert_pos and view_dir to the canonical space of each part
        joint_pos = j_transform[:, :3, -1]
        part_transform_R = j_transform[self.part_root, :3, :3]
        part_transform_RT = torch.transpose(part_transform_R, 1, 2)
        view_dir_part = torch.einsum('ijk,lk->lij', part_transform_RT, view_dir)
        pose_feature_all = torch.tile(poses.reshape(24, 3), (vert_num, 1, 1))

        pred_part = []
        for part_id in range(self.part_num):
            part_joint_idx = self.part[part_id]
            part_joint_nn_idx = self.part_all[part_id]

            vert_pos_relative = vert_pos[:, None, :] - torch.tile(joint_pos[part_joint_idx], (vert_num, 1, 1))
            vert_pos_relative = torch.einsum('ij,klj->kli', part_transform_RT[part_id], vert_pos_relative)

            pos_feature = vert_pos_relative.reshape(vert_num, -1)
            view_feature = view_dir_part[:, part_id, :]
            pose_feature = pose_feature_all[:, part_joint_nn_idx, :].reshape(vert_num, -1)

            pred_part.append(self.part_net_list[part_id](pos_feature, view_feature, pose_feature))

        pred = torch.cat(pred_part, dim=-1)
        pred = torch.sigmoid(pred)
        pred_min = torch.prod(pred, dim=-1, keepdim=False)

        if return_all_part:
            return torch.cat((pred, pred_min[:, None]), dim=-1)
        else:
            return pred_min


## naive VisNetwork without part-wise design
class TotalVisNetwork(nn.Module):
    def __init__(self):
        super(TotalVisNetwork, self).__init__()

        dims=[256, 256, 256, 256]
        points_multires = 10
        dirs_multires = 4
        pose_feat_dim = 72

        self.p_embed_fn, p_input_dim = embedder.get_embedder(multires=points_multires, input_dims=3)
        self.dir_embed_fn, dir_input_dim = embedder.get_embedder(multires=dirs_multires, input_dims=3)
        self.actv_fn = nn.ReLU()

        vis_layer = []
        dim = p_input_dim + dir_input_dim + pose_feat_dim

        for i in range(len(dims)):
            vis_layer.append(nn.Linear(dim, dims[i]))
            vis_layer.append(self.actv_fn)
            dim = dims[i]
        vis_layer.append(nn.Linear(dim, 1))
        self.vis_layer = nn.Sequential(*vis_layer)

    def forward(self, vert_pos, view_dir, j_transform, poses):
        vert_num = vert_pos.shape[0]
        pose_feature = torch.tile(poses, (vert_num, 1))
        pos_feature = self.p_embed_fn(vert_pos)
        view_feature = self.dir_embed_fn(view_dir)

        vis = self.vis_layer(torch.cat([pos_feature, view_feature, pose_feature], -1))[:, 0]
        vis = torch.sigmoid(vis)
        return vis
