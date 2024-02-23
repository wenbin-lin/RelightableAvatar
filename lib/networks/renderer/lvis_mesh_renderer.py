import torch
from lib.config import cfg
from .nerf_net_utils import *
import numpy as np
import mcubes
import trimesh
from lib.utils.blend_utils import *
from lib.utils import sample_utils
import open3d as o3d
import os
import pymeshfix as mf
from scipy.spatial import KDTree


def mesh_smplify(mesh_vertices, mesh_faces, scale=8):
    mesh_o3d = o3d.geometry.TriangleMesh()
    mesh_o3d.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh_faces)

    mesh_smp = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=mesh_faces.shape[0] // scale)
    return np.asarray(mesh_smp.vertices), np.asarray(mesh_smp.triangles)


def body_devide(mesh_v, mesh_f, mesh_bw, out_path):
    mesh_bw_verts = mesh_bw
    mesh_bw_faces = mesh_bw_verts[mesh_f]
    mesh_bw_faces = np.sum(mesh_bw_faces, axis=1)

    v_joints_idx = np.argmax(mesh_bw_verts, axis=-1)
    f_joints_idx = np.argmax(mesh_bw_faces, axis=-1)

    joint2part = np.load(cfg.joint2part_path)
    joint_num = joint2part.shape[0]
    part_num = cfg.part_num

    v_part_id = np.zeros(mesh_v.shape[0], dtype=np.int32)
    for i in range(joint_num):
        v_part_id[v_joints_idx == i] = joint2part[i]

    f_part_id = np.zeros(mesh_f.shape[0], dtype=np.int32)
    for i in range(joint_num):
        f_part_id[f_joints_idx == i] = joint2part[i]

    for i in range(part_num):
        f_part = mesh_f[f_part_id == i]
        part_mesh = trimesh.Trimesh(vertices=mesh_v, faces=f_part)

        # fill the hole in part mesh
        meshfix = mf.MeshFix(np.array(part_mesh.vertices), np.array(part_mesh.faces))
        meshfix.repair(verbose=False)
        f = meshfix.faces()
        v = meshfix.points()

        part_mesh_fixed = trimesh.Trimesh(vertices=v, faces=f, process=False)
        part_mesh_fixed.export(out_path + '/part_mesh/{:04d}.ply'.format(i))

        # mark in-body faces (add during hole filling)
        tree = KDTree(mesh_v)
        distance, _ = tree.query(v)
        # distance = np.fabs(trimesh.proximity.signed_distance(part_mesh, v))
        distance_face = np.mean(distance[f], axis=-1)
        inbody_faces_idx = np.where(distance_face > 1e-5)[0]
        np.save(out_path + '/part_mesh/inbody_faces_idx_{:04d}.npy'.format(i), inbody_faces_idx)
        # m = trimesh.Trimesh(vertices=v, faces=f[inbody_faces_idx], process=False)
        # m.export(out_path + '/part_mesh/inbody_faces_{:04d}.ply'.format(i))

class Renderer:
    def __init__(self, net):
        self.net = net
    
    def batchify_sdf_wo_deformation(self, wpts, sdf_decoder, net, chunk, batch):
        """Render rays in smaller minibatches to avoid OOM.
        """
        n_point = wpts.shape[0]
        all_ret = []
        for i in range(0, n_point, chunk):
            ret = sdf_decoder(wpts[i:i + chunk])[:, :1]
            all_ret.append(ret.detach().cpu().numpy())
        all_ret = np.concatenate(all_ret, 0)
        return all_ret

    def render(self, batch):
        pts = batch['pts']
        sh = pts.shape

        pts = pts.view(sh[0], -1, 3)
        device = pts.device

        frame_index = batch['frame_index'][0].item()

        tbw, tnorm = sample_utils.sample_blend_closest_points(pts, batch['tvertices'], batch['weights'])
        tnorm = tnorm[..., 0]
        norm_th = 0.15 # 0.1
        inside = tnorm < norm_th 

        pts = pts[inside]

        out_path = './data/lvis_dataset/{}/'.format(cfg.exp_name)
        os.makedirs(out_path, exist_ok=True)
        if frame_index == 0:
            sdf_decoder = lambda x: self.net.tpose_human.sdf_network(x, batch)
            sdf = self.batchify_sdf_wo_deformation(pts, sdf_decoder, self.net, 2048 * 64, batch)

            inside = inside.detach().cpu().numpy()
            full_sdf = 10 * np.ones(inside.shape)
            full_sdf[inside] = sdf[:, 0]
            sdf = -full_sdf

            # marching cubes
            cube = sdf.reshape(*sh[1:-1])
            cube = np.pad(cube, 10, mode='constant', constant_values=-10)
            vertices, part_mesh_faces = mcubes.marching_cubes(cube, 0)
            mesh = trimesh.Trimesh(vertices, part_mesh_faces)
            vertices = (vertices - 10) * cfg.voxel_size[0]
            vertices = vertices + batch['tbounds'][0, 0].detach().cpu().numpy()

            # get the largest connected part
            labels = trimesh.graph.connected_component_labels(mesh.face_adjacency)
            part_mesh_faces = part_mesh_faces[labels == 0]
            
            mesh_o3d = o3d.geometry.TriangleMesh()
            mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
            mesh_o3d.triangles = o3d.utility.Vector3iVector(part_mesh_faces)
            mesh_o3d.remove_unreferenced_vertices()
            vertices = np.array(mesh_o3d.vertices)
            part_mesh_faces = np.array(mesh_o3d.triangles)

            can_mesh = trimesh.Trimesh(vertices=vertices, faces=part_mesh_faces, process=False)
            can_mesh.export(out_path + '/can_mesh.ply')

            vertices_smp, faces_smp = mesh_smplify(vertices, part_mesh_faces)
            can_mesh_smp = trimesh.Trimesh(vertices=vertices_smp, faces=faces_smp, process=False)
            can_mesh_smp.export(out_path + '/can_mesh_smp.ply')

        os.makedirs(out_path + '/part_mesh/', exist_ok=True)
        if frame_index == 0:
            can_mesh = trimesh.load(out_path + '/can_mesh_smp.ply', process=False)
            mesh_v = np.array(can_mesh.vertices)
            mesh_f = np.array(can_mesh.faces)

            mesh_v_torch = torch.from_numpy(mesh_v).type(torch.float32).to(device)[None]
            mesh_bw = self.net.get_tpose_skinning(mesh_v_torch, batch)
            mesh_bw = mesh_bw.detach().cpu().numpy()[0]

            body_devide(mesh_v, mesh_f, mesh_bw, out_path)

        full_mesh = trimesh.load(out_path + '/can_mesh_smp.ply', process=False)
        can_mesh_v = torch.from_numpy(full_mesh.vertices).type(torch.float32).to(device)[None]
        defomred_v = self.net.calculate_residual_deformation_can2deformed(can_mesh_v, batch)
        tbw = self.net.get_tpose_skinning(defomred_v, batch)
        tbw = tbw.permute(0, 2, 1)
        tpose_pts = pose_points_to_tpose_points(defomred_v, tbw,
                                                batch['big_A'])
        pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
        full_mesh_vertices = pose_pts[0].detach().cpu().numpy()

        part_mesh_vertices = []
        part_mesh_faces = []
        part_num = cfg.part_num
        for i in range(part_num):
            part_mesh = trimesh.load(out_path + '/part_mesh/{:04d}.ply'.format(i), process=False)
            part_mesh_faces.append(np.array(part_mesh.faces))
        
            can_mesh_v = torch.from_numpy(part_mesh.vertices).type(torch.float32).to(device)[None]
            defomred_v = self.net.calculate_residual_deformation_can2deformed(can_mesh_v, batch)
            tbw = self.net.get_tpose_skinning(defomred_v, batch)
            tbw = tbw.permute(0, 2, 1)

            tpose_pts = pose_points_to_tpose_points(defomred_v, tbw,
                                                    batch['big_A'])
            pose_pts = tpose_points_to_pose_points(tpose_pts, tbw, batch['A'])
            part_mesh_vertices.append(pose_pts[0].detach().cpu().numpy())

        ret = {
            'full_mesh_vertices': full_mesh_vertices,
            'part_mesh_vertices': part_mesh_vertices,
            'part_mesh_faces': part_mesh_faces
        }

        return ret
