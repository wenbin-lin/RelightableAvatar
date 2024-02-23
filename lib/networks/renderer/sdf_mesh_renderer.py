import torch
from lib.config import cfg
from .nerf_net_utils import *
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *
from lib.utils import sample_utils
import open3d as o3d


class Renderer:
    def __init__(self, net):
        self.net = net

    def batchify_sdf(self, wpts, sdf_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            resd = net.calculate_residual_deformation(wpts[i:i + chunk][None], batch)
            if cfg.mesh_cfg.non_rigid_deform:
                wpts[i:i + chunk] = wpts[i:i + chunk] + resd[0]
            ret = sdf_decoder(wpts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        if not cfg.novel_pose:
            self.net.recompute_smpl(batch)

        pts = batch['pts']
        sh = pts.shape

        pts = pts.view(sh[0], -1, 3)

        tbw, tnorm = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        tnorm = tnorm[..., 0]
        norm_th = 0.15 # 0.1
        inside = tnorm < norm_th 

        pts = pts[inside]

        sdf_decoder = lambda x: self.net.tpose_human.sdf_network(x, batch)

        sdf = self.batchify_sdf(pts, sdf_decoder, self.net, 2048 * 64, batch)

        inside = inside.detach().cpu().numpy()
        full_sdf = 10 * np.ones(inside.shape)
        full_sdf[inside] = sdf[:, 0]
        sdf = -full_sdf

        # marching cubes
        cube = sdf.reshape(*sh[1:-1])
        cube = np.pad(cube, 10, mode='constant', constant_values=-10)
        vertices, triangles = mcubes.marching_cubes(cube, 0)
        mesh = trimesh.Trimesh(vertices, triangles)
        vertices = (vertices - 10) * cfg.voxel_size[0]
        vertices = vertices + batch['tbounds'][0, 0].detach().cpu().numpy()

        # get the largest connected part
        labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
        triangles = triangles[labels == 0]
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(triangles)
        mesh_o3d.remove_unreferenced_vertices()
        vertices = np.array(mesh_o3d.vertices)
        triangles = np.array(mesh_o3d.triangles)

        # transform vertices to the world space
        pts = torch.from_numpy(vertices).to(pts)[None]
        tbw = self.net.get_tpose_skinning(pts, batch)
        tbw = tbw.permute(0, 2, 1)

        deformed_pts = pts

        tpose_pts = pose_points_to_tpose_points(deformed_pts, tbw,
                                                batch['big_A'])
        pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
        world_pts = pose_points_to_world_points(pose_pts, batch['R'],
                                               batch['Th'])
        
        deformed_vertices = deformed_pts[0].detach().cpu().numpy()
        posed_vertices = pose_pts[0].detach().cpu().numpy()
        world_vertices = world_pts[0].detach().cpu().numpy()

        ret = {
            'deformed_vertices': deformed_vertices,
            'posed_vertices': posed_vertices, 
            'world_vertices': world_vertices, 
            'triangle': triangles
        }

        return ret
