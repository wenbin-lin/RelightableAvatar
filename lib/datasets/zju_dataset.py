import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import imageio
import cv2
from lib.config import cfg
from lib.utils import data_utils
import copy
from scipy.spatial.transform import Rotation as Rotation


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.human = human
        self.split = split

        annots = np.load(ann_file, allow_pickle=True).item()
        self.cams = annots['cams']

        num_cams = len(self.cams['K'])

        self.cam_Ks = copy.deepcopy(self.cams['K'])
        self.cam_Ks = np.array(self.cam_Ks)
        for i in range(num_cams):
            self.cam_Ks[i][:2] = self.cam_Ks[i][:2] * cfg.ratio

        if len(cfg.test_view) == 0:
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]
            if len(test_view) == 0:
                test_view = [0]
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view

        i = cfg.begin_ith_frame
        i_intv = cfg.frame_interval
        ni = cfg.num_train_frame

        self.frame_num = ni
        self.frame_index = np.arange(i, i + ni * i_intv, step=i_intv)

        self.ims = np.array([
            np.array(ims_data['ims'])[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.cam_inds = np.array([
            np.arange(len(ims_data['ims']))[view]
            for ims_data in annots['ims'][i:i + ni * i_intv][::i_intv]
        ]).ravel()
        self.num_cams = len(view)
        self.view = view

        self.lbs_root = os.path.join(self.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))
        weights = np.load(os.path.join(self.lbs_root, 'weights.npy'))
        self.weights = weights.astype(np.float32)
        self.big_A = self.load_bigpose()

        # add rotation to the system (cameras and smpl params) to make the z-axis go up
        self.R_system = np.array([[0, 1, 0], 
                                  [1, 0, 0], 
                                  [0, 0, -1]])
        
        for i in range(self.num_cams):        
            Rs = self.cams['R'][i]
            Rs = Rs @ self.R_system
            self.cams['R'][i] = Rs

        self.nrays = cfg.N_rand

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

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, 'mask',
                                    self.ims[index])[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.jpg'
        msk_cihp = imageio.imread(msk_path)
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root or 'mixamo' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)
            msk_erode = cv2.erode(msk.copy(), kernel)
            msk_dilate = cv2.dilate(msk.copy(), kernel)
            msk[(msk_dilate - msk_erode) == 1] = 100

        return msk, orig_msk

    def prepare_input(self, i):
        # read xyz in the world coordinate system
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        if not os.path.exists(vertices_path):
            vertices_path = os.path.join(self.data_root, cfg.vertices,
                    '{:06d}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)

        # transform smpl from the world coordinate to the smpl coordinate
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))
        if not os.path.exists(params_path):
            params_path = os.path.join(self.data_root, cfg.params,
                    '{:06d}.npy'.format(i))
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)
        Th = params['Th'].astype(np.float32)

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)

        # recompute wxyz to make the z-axis go up
        rot_new = Rotation.from_matrix(self.R_system) * Rotation.from_rotvec(Rh)
        rot_new_vec = rot_new.as_rotvec()
        Rh_new = np.array([[rot_new_vec[0, 0], rot_new_vec[0, 1], rot_new_vec[0, 2]]])
        Rh = Rh_new

        Th_new = np.array([[Th[0, 1], Th[0, 0], -Th[0, 2]]])
        Th = Th_new

        R = cv2.Rodrigues(Rh)[0].astype(np.float32)
        wxyz = (np.dot(pxyz, R.T) + Th).astype(np.float32)

        # calculate the skeleton transformation
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints
        parents = self.parents
        A, canonical_joints, joints_transform = data_utils.get_rigid_transformation(
            poses, joints, parents, return_joints=True, return_joints_transform=True)

        # posed_joints = np.dot(canonical_joints, R.T) + Th

        joints_transform[0, :3, :3] = R

        poses = poses.ravel().astype(np.float32)

        return wxyz, pxyz, A, joints_transform, Rh, Th, poses
    
    def get_all_smpl_params(self):
        view_num = len(self.view)
        frame_id_list = []
        Rh_list = []
        Th_list = []
        poses_list = []
        for frame_id in range(self.frame_num):
            img_path = os.path.join(self.data_root, self.ims[frame_id * view_num])
            i = int(os.path.basename(img_path)[:-4])

            params_path = os.path.join(self.data_root, cfg.params,
                                    '{}.npy'.format(i))
            if not os.path.exists(params_path):
                params_path = os.path.join(self.data_root, cfg.params,
                        '{:06d}.npy'.format(i))
            params = np.load(params_path, allow_pickle=True).item()
            Rh = params['Rh'].astype(np.float32).reshape(-1, )
            Th = params['Th'].astype(np.float32).reshape(-1, )
            poses = params['poses'].reshape(-1, )
            beta = params['shapes']

            frame_id_list.append(i)
            Rh_list.append(Rh)
            Th_list.append(Th)
            poses_list.append(poses)

        frame_id_all = np.array(frame_id_list)
        Rh_all = np.array(Rh_list)
        Th_all = np.array(Th_list)
        poses_all = np.array(poses_list)
        return frame_id_all, Rh_all, Th_all, poses_all, beta


    def __getitem__(self, index):      
        img_path = os.path.join(self.data_root, self.ims[index])
        img = imageio.imread(img_path).astype(np.float32) / 255.
        msk, orig_msk = self.get_mask(index)

        H, W = img.shape[:2]
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)

        cam_ind = self.cam_inds[index]
        K = np.array(self.cams['K'][cam_ind])
        D = np.array(self.cams['D'][cam_ind])
        img = cv2.undistort(img, K, D)

        msk = cv2.undistort(msk, K, D)
        orig_msk = cv2.undistort(orig_msk, K, D)

        R = np.array(self.cams['R'][cam_ind])
        T = np.array(self.cams['T'][cam_ind]) / 1000.

        # reduce the image resolution by ratio
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)
        if cfg.mask_bkgd:
            img[msk == 0] = 0
        K[:2] = K[:2] * cfg.ratio

        if self.human in ['CoreView_313', 'CoreView_315']:
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        elif 'deepcap' in self.data_root:
            i = int(os.path.basename(img_path).split('_')[-1][:-4])
            frame_index = i
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # read v_shaped
        if cfg.get('use_bigpose', False):
            vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        else:
            vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')
        tvertices = np.load(vertices_path).astype(np.float32)
        tbounds = data_utils.get_bounds(tvertices)

        wpts, pvertices, A, joints_transform, Rh, Th, poses = self.prepare_input(i)

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
            'file_index': i,
            'cam_ind': cam_ind, 
            'Ks': np.array(self.cam_Ks), 
            'Rs': np.array(self.cams['R']), 
            'Ts': np.array(self.cams['T']) / 1000., 
            'coord': coord
        }
        ret.update(meta)

        return ret

    def __len__(self):
        return len(self.ims)
