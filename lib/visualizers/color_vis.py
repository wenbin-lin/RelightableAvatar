import matplotlib.pyplot as plt
import numpy as np
from lib.config import cfg
import os
import cv2
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = cfg.result_dir
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize_image(self, output, batch):
        rgb_pred = output['rgb_map'][0].detach().cpu().numpy()
        rgb_gt = batch['rgb'][0].detach().cpu().numpy()
        print('mse: {}'.format(np.mean((rgb_pred - rgb_gt)**2)))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        img_gt = np.zeros((H, W, 3))
        img_gt[mask_at_box] = rgb_gt

        result_dir = os.path.join(cfg.result_dir, 'comparison')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index, view_index),
            (img_gt[..., [2, 1, 0]] * 255))

    def visualize_acc(self, output, batch):
        acc_pred = output['acc_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        acc = np.zeros((H, W))
        acc[mask_at_box] = acc_pred

        result_dir = os.path.join(cfg.result_dir, 'acc')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index),
            (acc * 100))

    def visualize_depth(self, output, batch):
        depth_pred = output['depth_map'][0].detach().cpu().numpy()

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        depth = np.zeros((H, W))
        depth[mask_at_box] = depth_pred

        result_dir = os.path.join(cfg.result_dir, 'depth')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index),
            (depth * 50))

    def visualize(self, output, batch):
        self.visualize_image(output, batch)
        self.visualize_acc(output, batch)
        self.visualize_depth(output, batch)
