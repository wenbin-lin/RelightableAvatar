import pickle
import os
import h5py
import sys
import numpy as np
import torch
import open3d as o3d
from lib.utils.body_model.body_model import BodyModel
import cv2
import tqdm


def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()


def get_KRTD(camera):
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1
    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    return K, R, T, D


def extract_image(data_path):
    data_root = os.path.dirname(data_path)
    img_dir = os.path.join(data_root, 'image')
    os.makedirs(img_dir, exist_ok=True)

    if len(os.listdir(img_dir)) >= 200:
        return

    cap = cv2.VideoCapture(data_path)

    ret, frame = cap.read()
    i = 0

    while ret:
        cv2.imwrite(os.path.join(img_dir, '{}.jpg'.format(i)), frame)
        ret, frame = cap.read()
        i = i + 1

    cap.release()


def extract_mask(masks, mask_dir):
    if len(os.listdir(mask_dir)) >= len(masks):
        return

    for i in tqdm.tqdm(range(len(masks))):
        mask = masks[i].astype(np.uint8)

        # erode the mask
        border = 4
        kernel = np.ones((border, border), np.uint8)
        mask = cv2.erode(mask.copy(), kernel)

        cv2.imwrite(os.path.join(mask_dir, '{}.png'.format(i)), mask)


data_root = 'data/people_snapshot'
# videos = ['male-3-casual']
videos = os.listdir(data_root)

model_paths = [
    './data/smplx/smpl/basicmodel_f_lbs_10_207_0_v1.1.0.pkl',
    './data/smplx/smpl/basicmodel_m_lbs_10_207_0_v1.1.0.pkl'
]

for video in videos:
    camera_path = os.path.join(data_root, video, 'camera.pkl')
    camera = read_pickle(camera_path)
    K, R, T, D = get_KRTD(camera)

    # process video
    video_path = os.path.join(data_root, video, video + '.mp4')
    extract_image(video_path)

    # process mask
    mask_path = os.path.join(data_root, video, 'masks.hdf5')
    masks = h5py.File(mask_path)['masks']
    mask_dir = os.path.join(data_root, video, 'mask')
    os.makedirs(mask_dir, exist_ok=True)
    extract_mask(masks, mask_dir)

    smpl_path = os.path.join(data_root, video, 'reconstructed_poses.hdf5')
    smpl = h5py.File(smpl_path)
    betas = smpl['betas']
    pose = smpl['pose']
    trans = smpl['trans']

    pose = pose[len(pose) - len(masks):]
    trans = trans[len(trans) - len(masks):]

    # process smpl parameters
    params = {'beta': np.array(betas), 'pose': pose, 'trans': trans}
    params_path = os.path.join(data_root, video, 'params.npy')
    np.save(params_path, params)

    if 'female' in video:
        model_path = model_paths[0]
    else:
        model_path = model_paths[1]
    body_model = BodyModel(model_path=model_path)

    img_dir = os.path.join(data_root, video, 'image')
    num_img = len(os.listdir(img_dir))

    vertices_dir = os.path.join(data_root, video, 'vertices')
    os.makedirs(vertices_dir, exist_ok=True)

    if len(os.listdir(vertices_dir)) < num_img:
        beta_torch = torch.from_numpy(np.array(betas)[None])
        for i in tqdm.tqdm(range(num_img)):
            pose_torch = torch.from_numpy(np.array(pose[i])[None])
            vertices, _, _, _, _ = body_model(beta_torch, pose_torch)
            vertices = vertices.numpy()[0]
            vertices = vertices + trans[i]
            np.save(os.path.join(vertices_dir, '{}.npy'.format(i)), vertices)

    big_pose = np.zeros((72, )).astype(np.float32)
    angle = 30
    big_pose[5] = np.deg2rad(angle)
    big_pose[8] = np.deg2rad(-angle)
    beta_torch = torch.from_numpy(np.array(betas)[None])
    pose_torch = torch.from_numpy(big_pose[None])
    vertices, _, joints, _, _ = body_model(beta_torch, pose_torch)
    vertices = vertices.numpy()[0]
    joints = joints.numpy()[0]

    lbs_dir = os.path.join(data_root, video, 'lbs')
    os.makedirs(lbs_dir, exist_ok=True)

    lbs_weights = body_model.lbs_weights.numpy()
    parents = body_model.kintree_table.numpy()[0]

    np.save(os.path.join(lbs_dir, 'bigpose_vertices.npy'), vertices)
    np.save(os.path.join(lbs_dir, 'joints.npy'), joints)
    np.save(os.path.join(lbs_dir, 'weights.npy'), lbs_weights)
    np.save(os.path.join(lbs_dir, 'parents.npy'), parents)
