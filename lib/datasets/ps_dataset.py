import torch.utils.data as data
from lib.utils import base_utils
from PIL import Image
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils import data_utils
import pickle
from scipy.spatial.transform import Rotation as Rotation


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_camera(camera_path):
    camera = read_pickle(camera_path)
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    camera = {'K': K, 'R': R, 'T': T, 'D': D}
    return camera


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        camera_path = os.path.join(self.data_root, 'camera.pkl')
        self.cam = get_camera(camera_path)
        self.num_train_frame = cfg.num_train_frame
        self.frame_num = self.num_train_frame

        self.ims = [None] * self.num_train_frame
        self.num_cams = 1

        params_path = ann_file
        self.params = np.load(params_path, allow_pickle=True).item()

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()
        self.nrays = cfg.N_rand

        R_system = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.rot_system = Rotation.from_matrix(R_system)
        self.cam['R'] = self.cam['R'] @ R_system.T

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
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                    '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path)

        wxyz = self.rot_system.apply(wxyz).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        Rh = self.params['pose'][i][:3].copy()
        Th = self.params['trans'][i]

        root_j = self.joints[0]
        root_trans = Rotation.from_rotvec(Rh).apply(root_j) - root_j
        Th = Th - root_trans

        Rh = (self.rot_system * Rotation.from_rotvec(Rh)).as_rotvec().astype(np.float32)
        Th = self.rot_system.apply(Th).astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # calculate the skeleton transformation
        poses = self.params['pose'][i].reshape(-1, 3).copy()
        poses[0, :] = 0.0
        joints = self.joints
        parents = self.parents
        A, canonical_joints, joints_transform = data_utils.get_rigid_transformation(
            poses, joints, parents, return_joints=True, return_joints_transform=True)

        # posed_joints = np.dot(canonical_joints, R.T) + Th

        joints_transform[0, :3, :3] = R
        poses = poses.ravel().astype(np.float32)
        return wxyz, pxyz, A, joints_transform, Rh, Th, poses
    
    def get_all_smpl_params(self):
        frame_id_list = []
        Rh_list = []
        Th_list = []
        poses_list = []
        for frame_id in range(self.frame_num):
            Rh = self.params['pose'][frame_id][:3].copy()
            Th = self.params['trans'][frame_id].astype(np.float32)

            root_j = self.joints[0]
            root_trans = Rotation.from_rotvec(Rh).apply(root_j) - root_j
            Th = Th - root_trans

            Rh = (self.rot_system * Rotation.from_rotvec(Rh)).as_rotvec().astype(np.float32)
            Th = self.rot_system.apply(Th).astype(np.float32)

            poses = self.params['pose'][frame_id].copy()
            poses = poses.reshape(-1, 3)
            poses[0, :] = 0.0

            beta = self.params['beta']

            Rh = Rh.astype(np.float32).reshape(-1, )
            Th = Th.astype(np.float32).reshape(-1, )
            poses = poses.astype(np.float32).reshape(-1, )
            beta = beta.astype(np.float32).reshape((1, 10))

            frame_id_list.append(frame_id)
            Rh_list.append(Rh)
            Th_list.append(Th)
            poses_list.append(poses)

        frame_id_all = np.array(frame_id_list)
        Rh_all = np.array(Rh_list)
        Th_all = np.array(Th_list)
        poses_all = np.array(poses_list)
        return frame_id_all, Rh_all, Th_all, poses_all, beta


    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, 'image',
                                '{}.jpg'.format(index))
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk_path = os.path.join(self.data_root, 'mask', '{}.png'.format(index))
        msk = imageio.imread(msk_path)
        orig_msk = msk.copy()

        frame_index = index
        latent_index = index

        K = self.cam['K']
        D = self.cam['D']
        img = cv2.undistort(img, K, D)
        msk = cv2.undistort(msk, K, D)

        R = self.cam['R']
        T = self.cam['T'][:, None]
        RT = np.concatenate([R, T], axis=1).astype(np.float32)
        
        # mask before resize
        if cfg.mask_bkgd:
            img[msk == 0] = 0
            if cfg.white_bkgd:
                img[msk == 0] = 1
        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H), interpolation=cv2.INTER_NEAREST)
        K = K.copy().astype(np.float32)
        K[:2] = K[:2] * cfg.ratio

        # read v_shaped
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tvertices = np.load(vertices_path).astype(np.float32)
        tbounds = data_utils.get_bounds(tvertices)

        wpts, pvertices, A, joints_transform, Rh, Th, poses = self.prepare_input(frame_index)

        pbounds = data_utils.get_bounds(pvertices)
        wbounds = data_utils.get_bounds(wpts)

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = data_utils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)

        if cfg.erode_edge:
            orig_msk = data_utils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]
        H, W = img.shape[:2]

        # nerf
        ret = {
            'rgb': rgb,
            'occupancy': occupancy,
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
            'frame_index': frame_index,
            'file_index': frame_index,
            'cam_ind': 0, 
            'Ks': np.array([self.cam['K']]), 
            'Rs': np.array([self.cam['R']]), 
            'Ts': np.array([self.cam['T']]), 
            'coord': coord
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return self.num_train_frame
