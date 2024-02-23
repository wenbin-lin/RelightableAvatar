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
        rgb_pred = np.power(rgb_pred + 1e-6, 1 / 2.2)

        if 'rgb' in batch.keys():
            rgb_gt = batch['rgb'][0].detach().cpu().numpy()
            print('mse: {}'.format(np.mean((rgb_pred - rgb_gt)**2)))

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        img_pred = np.zeros((H, W, 3))
        img_pred[mask_at_box] = rgb_pred

        if 'rgb' in batch.keys():
            img_gt = np.zeros((H, W, 3))
            img_gt[mask_at_box] = rgb_gt

        result_dir = os.path.join(cfg.result_dir, 'relighting')
        os.makedirs(result_dir, exist_ok=True)
        frame_index = batch['frame_index'].item()
        view_index = batch['cam_ind'].item()
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index),
            (img_pred[..., [2, 1, 0]] * 255))
        
        if 'rgb' in batch.keys():
            result_dir = os.path.join(cfg.result_dir, 'input')
            os.makedirs(result_dir, exist_ok=True)
            cv2.imwrite(
                '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index, view_index),
                (img_gt[..., [2, 1, 0]] * 255))

    def visualize_sg(self, output, batch):
        sg_map_names = ['sg_diffuse_rgb_map', 'sg_specular_rgb_map', 'normal_map', 
                        'diffuse_albedo_map', 'roughness_map', 'vis_shadow_map']
        sg_out_dir = ['diffuse_rgb', 'specular_rgb', 'normal', 
                        'albedo', 'roughness', 'vis_shadow']

        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        H, W = batch['H'].item(), batch['W'].item()
        mask_at_box = mask_at_box.reshape(H, W)

        for i in range(len(sg_map_names)):
            item_pred = output[sg_map_names[i]][0].detach().cpu().numpy()
            item_map = np.zeros((H, W, 3))
            item_map[mask_at_box] = item_pred

            if sg_map_names[i] == 'normal_map':
                item_map = (item_map[..., [2, 1, 0]] + 1) / 2
            elif 'albedo' in sg_map_names[i] or 'vis_shadow' in sg_map_names[i]:
                item_map = item_map[..., [2, 1, 0]]
            elif 'rgb' in sg_map_names[i]:
                item_map = np.power(item_map[..., [2, 1, 0]] + 1e-6, 1 / 2.2)
            elif 'roughness' in sg_map_names[i]:
                item_map = item_map # / 4

            result_dir = os.path.join(cfg.result_dir, sg_out_dir[i])
            os.makedirs(result_dir, exist_ok=True)
            frame_index = batch['frame_index'].item()
            view_index = batch['cam_ind'].item()
            cv2.imwrite(
                '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                    view_index),
                (item_map * 255))

    def visualize(self, output, batch):
        self.visualize_image(output, batch)
        self.visualize_sg(output, batch)

