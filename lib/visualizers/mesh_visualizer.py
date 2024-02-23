from lib.config import cfg
import os
import trimesh
from termcolor import colored


class Visualizer:
    def __init__(self):
        result_dir = 'data/animation/{}'.format(cfg.exp_name)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

    def visualize(self, output, batch):
        if cfg.mesh_cfg.out_type == 'deformed':
            vertices = output['deformed_vertices']
        elif cfg.mesh_cfg.out_type == 'posed':
            vertices = output['posed_vertices']
        else:
            vertices = output['world_vertices']

        faces = output['triangle']

        mesh = trimesh.Trimesh(vertices, faces, process=False)
        result_dir = 'data/animation/{}'.format(cfg.exp_name)
        os.makedirs(result_dir, exist_ok=True)

        result_dir = os.path.join(result_dir, cfg.mesh_cfg.out_type)
        os.makedirs(result_dir, exist_ok=True)

        frame_index = batch['frame_index'][0].item()
        mesh_path = os.path.join(result_dir, '{:04d}.ply'.format(frame_index))

        mesh.export(mesh_path)
