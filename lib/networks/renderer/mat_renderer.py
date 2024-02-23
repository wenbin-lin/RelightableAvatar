import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *

import mcubes
import trimesh
import open3d as o3d
from lib.utils.blend_utils import *
from lib.utils import sample_utils

class Renderer:
    def __init__(self, net):
        self.net = net
        self.grid_pts = None
        self.mesh_vertices = None
        self.mesh_faces = None
        self.mesh_smp = False

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        ret = raw_decoder(wpts, viewdir, dists, n_pixel, n_sample)

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d

        raw_decoder = lambda wpts_val, viewdir_val, dists_val, n_pixel, n_sample: self.net(
            wpts_val, viewdir_val, dists_val, n_pixel, n_sample, batch)

        # compute the color and density
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret.update({
            'rgb_map': rgb_map,
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4)
        })

        # for saving images
        if not self.net.training:
            sg_values = {}
            sg_item_names = ['sg_diffuse_rgb_values', 'sg_specular_rgb_values', 'normal_values', 
                        'diffuse_albedo_values', 'roughness_values', 'vis_shadow']
            sg_map_names = ['sg_diffuse_rgb_map', 'sg_specular_rgb_map', 'normal_map', 
                        'diffuse_albedo_map', 'roughness_map', 'vis_shadow_map']
            for item_id in range(len(sg_item_names)):
                sg_values.update({sg_item_names[item_id]: ret[sg_item_names[item_id]].reshape(-1, n_sample, 3)})
            
            sg_data_map = rawsg2outputs(
                raw, z_vals, sg_values, sg_item_names, sg_map_names, cfg.white_bkgd)

            for item_id in range(len(sg_map_names)):
                ret.update({sg_map_names[item_id]: sg_data_map[sg_map_names[item_id]].view(n_batch, n_pixel, -1)})

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def compute_grid_pts(self, batch):
        tbounds = batch['tbounds'][0].cpu().numpy()

        voxel_size = cfg.voxel_size
        x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)
        self.grid_pts = torch.from_numpy(pts).to(batch['tbounds'])
    
    def batchify_rays(self, wpts, sdf_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = sdf_decoder(wpts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret
    
    def extract_mesh(self, batch):
        device = batch['tvertices'].device
        if self.grid_pts is None:
            self.compute_grid_pts(batch)
            self.grid_pts = self.grid_pts.to(device)
        sh = self.grid_pts.shape
        bs = batch['tvertices'].shape[0]
        pts = self.grid_pts
        pts = pts.view(bs, -1, 3)
        tbw, tnorm = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        tnorm = tnorm[..., 0]
        norm_th = 0.1
        inside = tnorm < norm_th 

        pts = pts[inside]
        sdf_decoder = lambda x: self.net.tpose_human.sdf_network(x, batch)

        sdf = self.batchify_rays(pts, sdf_decoder, self.net, 2048 * 64, batch)

        inside = inside.detach().cpu().numpy()
        full_sdf = 10 * np.ones(inside.shape)
        full_sdf[inside] = sdf[:, 0]
        sdf = -full_sdf

        cube = sdf.reshape(*sh[:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)
        vertices, triangles = mcubes.marching_cubes(cube, 0)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['tbounds'][0, 0].detach().cpu().numpy()

        mesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        triangles = triangles[labels == 0]
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_o3d.remove_unreferenced_vertices()

        if self.mesh_smp:
            mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=len(mesh_o3d.triangles) // 8)

        vertices = np.array(mesh_o3d.vertices).astype(np.float32)
        triangles = np.array(mesh_o3d.triangles).astype(np.int64)

        self.mesh_vertices = torch.from_numpy(vertices).to(device)
        self.mesh_faces = torch.from_numpy(triangles).to(device)

    def get_mesh_v_tbw(self, batch):
        if self.mesh_vertices is None:
            self.extract_mesh(batch)

        mesh_vertices_deformed = self.net.calculate_residual_deformation_can2deformed(self.mesh_vertices[None], batch)
        batch['mesh_vertices_can'] = mesh_vertices_deformed

        tbw = self.net.get_tpose_skinning(mesh_vertices_deformed, batch)
        tbw = tbw.permute(0, 2, 1)

        tpose_pts = pose_points_to_tpose_points(mesh_vertices_deformed, tbw, batch['big_A'])
        pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
        pose_pts = pose_points_to_world_points(pose_pts, batch['R'], batch['Th'])

        batch['mesh_vertices'] = pose_pts
        batch['mesh_tbw'] = tbw.permute(0, 2, 1)

    def render(self, batch):
        if not cfg.novel_pose:
            self.net.recompute_smpl(batch)

        if self.grid_pts is None:
            self.extract_mesh(batch)

        self.get_mesh_v_tbw(batch)

        ray_o = batch['ray_o']
        ray_d = batch['ray_d']
        near = batch['near']
        far = batch['far']
        sh = ray_o.shape

        # volume rendering for each pixel
        n_batch, n_pixel = ray_o.shape[:2]
        chunk = 2048
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk, batch)
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
