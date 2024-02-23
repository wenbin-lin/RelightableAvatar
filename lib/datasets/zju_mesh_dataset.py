import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import json
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils import data_utils
from glob import glob


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split, **kwargs):
        self.data_root = data_root
        self.human = human
        self.split = split

        self.num_cams = 1

        begin = cfg.mesh_cfg.begin_ith_frame
        end = cfg.mesh_cfg.end_ith_frame
        interval = cfg.mesh_cfg.frame_interval

        self.frame_id_all = np.arange(begin, end, interval)

        self.num_frames = self.frame_id_all.shape[0]

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()

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
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A = data_utils.get_rigid_transformation(poses, joints, parents)

        poses = poses.ravel().astype(np.float32)

        return A, Rh, Th, poses
    

    def __getitem__(self, index):
        frame_index = self.frame_id_all[index]
        file_index = self.frame_id_all[index]

        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tpose = np.load(vertices_path).astype(np.float32)
        tbounds = data_utils.get_bounds(tpose)

        A, Rh, Th, poses = self.prepare_input(frame_index)

        tbounds[0, :] = tbounds[0, :] - 0.10
        tbounds[1, :] = tbounds[1, :] + 0.10

        voxel_size = cfg.voxel_size
        x = np.arange(tbounds[0, 0], tbounds[1, 0] + voxel_size[0],
                      voxel_size[0])
        y = np.arange(tbounds[0, 1], tbounds[1, 1] + voxel_size[1],
                      voxel_size[1])
        z = np.arange(tbounds[0, 2], tbounds[1, 2] + voxel_size[2],
                      voxel_size[2])
        pts = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1)
        pts = pts.astype(np.float32)

        ret = {'pts': pts}
        meta = {
            'A': A,
            'big_A': self.big_A,
            'poses': poses,
            'weights': self.weights,
            'tvertices': tpose,
            'tbounds': tbounds,
        }
        ret.update(meta)

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        meta = {
            'R': R,
            'Th': Th,
            'frame_index': frame_index, 
            'file_index': file_index,
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_frames
