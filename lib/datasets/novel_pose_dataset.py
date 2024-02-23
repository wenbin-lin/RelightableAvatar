import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import torch
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils import data_utils
import copy
import math
from scipy.spatial.transform import Rotation as Rotation
from lib.utils.body_model.body_model import BodyModel


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        self.num_cams = 1

        self.novel_poses_data = np.load(cfg.novel_poses_path)
        self.novel_poses_all = self.novel_poses_data['poses_all']
        self.Rh_all = self.novel_poses_data['Rh_all']
        self.Th_all = self.novel_poses_data['Th_all']

        self.frame_id_all = np.arange(self.novel_poses_all.shape[0])

        self.num_frames = self.frame_id_all.shape[0]

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()

        self.nrays = cfg.N_rand

        self.novel_view = False
        self.novel_view_num = 50

        self.body_model = BodyModel()

    def set_beta(self, beta):
        self.beta = beta

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()
        angle = 30
        big_poses[5] = np.deg2rad(angle)
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)
        big_A = data_utils.get_rigid_transformation(
            big_poses, self.joints, self.parents)
        big_A = big_A.astype(np.float32)
        return big_A
    
    def prepare_input(self, i):
        Rh = self.Rh_all[i]
        Th = self.Th_all[i]
        poses = self.novel_poses_all[i]

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        poses[0, :] = 0.0

        betas_torch = torch.from_numpy(self.beta)
        poses_torch = torch.from_numpy(poses.reshape(1, 72))
        verts_posed, _, _, _, _ = self.body_model(betas_torch, poses_torch)
        pxyz = verts_posed.numpy()[0].astype(np.float32)
        wxyz = (np.dot(pxyz, R.T) + Th).astype(np.float32)

        joints = self.joints
        parents = self.parents
        A, canonical_joints, joints_transform = data_utils.get_rigid_transformation(
            poses, joints, parents, return_joints=True, return_joints_transform=True)

        joints_transform[0, :3, :3] = R
        poses = poses.ravel().astype(np.float32) 

        return wxyz, pxyz, A, joints_transform, Rh, Th, poses

    def __getitem__(self, index):
        W = int(cfg.img_w * cfg.ratio)
        H = int(cfg.img_h * cfg.ratio)

        frame_idx = index

        # set camera
        fov = 40
        fx = (W / 2.0) / math.tan(math.radians(fov / 2))
        fy = (H / 2.0) / math.tan(math.radians(fov / 2))
        K = np.array([[fx, 0.0, W / 2], 
                      [0.0, fy, H / 2], 
                      [0.0, 0.0, 1.0]])

        rot_theta = 0.0
        if self.novel_view:
            rot_theta = ((index % self.novel_view_num) / self.novel_view_num) * 2 * np.pi
        R = np.array([
            [math.cos(rot_theta), -math.sin(rot_theta), 0], 
            [0, 0, -1], 
            [math.sin(rot_theta), math.cos(rot_theta), 0]
        ])
        R_base = Rotation.from_rotvec(np.array([np.pi / 10, 0.0, 0.0]))
        R = (Rotation.from_matrix(R) * R_base).as_matrix()
        T = np.array([[0], [0.9], [4.0]])

        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tvertices = np.load(vertices_path).astype(np.float32)
        tbounds = data_utils.get_bounds(tvertices)

        wpts, pvertices, A, joints_transform, Rh, Th, poses = self.prepare_input(frame_idx)

        pbounds = data_utils.get_bounds(pvertices)
        wbounds = data_utils.get_bounds(wpts)

        ray_o, ray_d, near, far, mask_at_box = data_utils.get_rays_within_bounds(
            H, W, K, R, T, wbounds)

        # nerf
        ret = {
            'ray_o': ray_o,
            'ray_d': ray_d,
            'near': near,
            'far': far,
            'mask_at_box': mask_at_box, 
        }

        # blend weight
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'joints_transform': joints_transform.astype(np.float32), 
            'weights': self.weights,
            'tvertices': tvertices,
            'pvertices': pvertices,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)

        # transformation
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)

        latent_index = index // self.num_cams
        meta = {
            'latent_index': latent_index,
            'frame_index': frame_idx,
            'file_index': frame_idx,
            'cam_ind': 0, 
            'Ks': np.array(K), 
            'Rs': np.array(T), 
            'Ts': np.array(T), 
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_frames
