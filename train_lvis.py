from lib.config import cfg
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from lib.networks.models import VisNetwork
from lib.utils import data_utils
import os


class Visibility_Dataset():
    def __init__(self, split='train', dataset_path=None):
        self.sample_points_num = 1024 * 16
        self.view_num = 4
        mesh_trans = [0, 0.2, 0]

        self.sampled_points_path = os.path.join(dataset_path, 'sampled_points')
        self.part_visibility_path = os.path.join(dataset_path, 'part_visibility')
        self.poses_all = np.load(cfg.novel_poses_path)
        self.poses_all[:, :3] = 0.0
        self.light_pos_all = np.load(cfg.light_pos_path)[:self.view_num * self.poses_all.shape[0]]

        self.lbs_root = os.path.join(cfg.train_dataset.data_root, 'lbs')
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))
        self.joints = joints.astype(np.float32)
        self.joints = joints + mesh_trans
        self.parents = np.load(os.path.join(self.lbs_root, 'parents.npy'))

    def __getitem__(self, index):
        pose_id = index // self.view_num
        light_id = index % self.view_num

        pose = self.poses_all[pose_id]
        light_pos = self.light_pos_all[index]

        part_vis_file = os.path.join(self.part_visibility_path, '{:04d}_{:04d}.npy'.format(pose_id, light_id))
        part_visibility = np.load(part_vis_file).reshape((self.sample_points_num, 15))
        is_visible_all = np.all(part_visibility, axis=-1)
        all_visibility = np.concatenate((part_visibility, is_visible_all.reshape((-1, 1)).astype(np.float32)), axis=-1)

        vert_pose_file = os.path.join(self.sampled_points_path, '{:04d}_{:04d}.npy'.format(pose_id, light_id))
        vert_pos = np.load(vert_pose_file).reshape((self.sample_points_num, 3))

        light_dir = vert_pos - light_pos[None]
        light_dir = light_dir / np.linalg.norm(light_dir, axis=-1)[:, None]

        _, _, j_transform = data_utils.get_rigid_transformation(pose.reshape((-1, 3)), self.joints, self.parents,
                                                                return_joints=True,
                                                                return_joints_transform=True)

        vert_pos = vert_pos.astype(np.float32)
        light_dir = light_dir.astype(np.float32)
        j_transform = j_transform.astype(np.float32)
        pose = pose.astype(np.float32)
        all_visibility = all_visibility.astype(np.float32)

        return vert_pos, light_dir, j_transform, pose, all_visibility

    def __len__(self):
        return self.view_num * self.poses_all.shape[0]


def train(data_loader):
    total_loss = 0.0
    total_acc = np.zeros(shape=(cfg.part_num + 1,))
    counter = 0
    model.train()
    for vert_pos, light_dir, j_transform, poses, label in tqdm(data_loader):
        optimizer.zero_grad()
        vert_pos = vert_pos[0].to(device)
        light_dir = light_dir[0].to(device)
        j_transform = j_transform[0].to(device)
        poses = poses[0].to(device)
        label = label[0].to(device)
        outputs = model(vert_pos, light_dir, j_transform, poses, return_all_part=True)
        loss = loss_function(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()
        acc = torch.logical_xor(label > 0.5, outputs > 0.5).float()
        acc = torch.mean(acc, dim=0)
        total_acc = total_acc + acc.detach().cpu().numpy()
        counter += 1

    return total_loss / counter, total_acc / counter


def train_model(ckpt_path, epoch_num):
    for epoch in range(epoch_num):
        train_loss, train_acc = train(train_loader)
        print('epoch: {}, loss: {}, error: {:.2f}%'.format(epoch, train_loss, train_acc[-1] * 100.0))
        for i in range(train_acc.shape[0] - 1):
            print('epoch: {}, part: {}, error: {:.2f}%'.format(epoch, i, train_acc[i] * 100.0))
            
        torch.save({'epoch': epoch, 'train_loss': train_loss,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, ckpt_path)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    model = VisNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCELoss()

    dataset_path = os.path.join('./data/lvis_dataset', cfg.exp_name_geo)
    train_dataset = Visibility_Dataset('train', dataset_path)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    ckpt_path = cfg.trained_model_dir
    os.makedirs(ckpt_path, exist_ok=True)
    ckpt_path = os.path.join(ckpt_path, 'latest.pth')

    train_model(ckpt_path, epoch_num=32)
