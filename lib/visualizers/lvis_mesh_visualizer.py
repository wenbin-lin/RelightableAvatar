from lib.config import cfg
import os
import trimesh
from termcolor import colored
import numpy as np


# compute intersect of posed mesh and sampled rays
def compute_intersect(mesh_full_v, mesh_part_v, mesh_part_f, light_pos,
                      sample_point_num=1024, noise_scale=0.05):
    vert_offset = [0, 0.2, 0]
    part_num = cfg.part_num

    v_id = np.random.choice(mesh_full_v.shape[0], sample_point_num, replace=False)
    sample_noise = np.random.normal(loc=0.0, scale=1.0, size=(sample_point_num, 3)) * noise_scale
    vert_pos = mesh_full_v[v_id] + vert_offset + sample_noise

    r_ori = light_pos.reshape((1, 3))
    r_dir = vert_pos - r_ori
    r_ori = np.tile(r_ori, (sample_point_num, 1))

    intersect_info = []
    for part_id in range(part_num):
        part_mesh = trimesh.Trimesh(vertices=mesh_part_v[part_id], faces=mesh_part_f[part_id], process=False)
        part_mesh.vertices = part_mesh.vertices + vert_offset

        locations, index_ray, index_tri = part_mesh.ray.intersects_location(ray_origins=r_ori, ray_directions=r_dir)
        intersect_info.append({'locations': locations, 'index_ray': index_ray, 'index_tri': index_tri})

    return intersect_info, vert_pos


# get point visibility data from the computed intersection
def parse_intersect(intersect_info, vert_pos, light_pos, inbody_faces_idx, sample_point_per_ray=16):
    ray_num = vert_pos.shape[0]
    part_num = cfg.part_num
    end_ratio = 1 / 4
    noise_scale = 0.2

    light_pos_curr = light_pos.reshape((1, 3))
    vert_cam_dist = np.sqrt(np.sum(np.square(vert_pos - light_pos_curr), axis=-1))

    ray_dir = vert_pos - light_pos_curr
    ray_dir = ray_dir / np.linalg.norm(ray_dir, axis=-1)[:, None]

    sample_noise = np.random.normal(loc=0.0, scale=1.0, size=(ray_num, sample_point_per_ray)) * noise_scale
    sample_noise_on_ray = np.einsum('ij,ik->ijk', sample_noise, ray_dir)

    locations_list = []
    index_ray_list = []
    intersect_part_id_list = []
    new_faces_mask_list = []

    for part_id in range(part_num):
        locations = intersect_info[part_id]['locations']
        index_ray = intersect_info[part_id]['index_ray'].reshape((-1, 1))
        intersect_part_id = np.ones(shape=(index_ray.shape[0], 1), dtype=np.int32) * part_id
        index_tri = intersect_info[part_id]['index_tri']

        inbody_f_idx = inbody_faces_idx[part_id]
        on_inbody_faces = np.array([f in inbody_f_idx for f in index_tri]).reshape((-1, 1))

        if locations.shape[0] > 0:
            locations_list.append(locations)
            index_ray_list.append(index_ray)
            intersect_part_id_list.append(intersect_part_id)
            new_faces_mask_list.append(on_inbody_faces)

    locations = np.vstack(locations_list)
    index_ray = np.vstack(index_ray_list).reshape((-1,))
    intersect_part_id = np.vstack(intersect_part_id_list).reshape((-1,))
    on_inbody_faces = np.vstack(new_faces_mask_list).reshape((-1,))

    # sort by ray index
    sort_idx = index_ray.argsort()
    locations = locations[sort_idx]
    index_ray = index_ray[sort_idx]
    intersect_part_id = intersect_part_id[sort_idx]
    on_inbody_faces = on_inbody_faces[sort_idx]

    # compute num of intersect points
    num_intersect = np.zeros(shape=(ray_num, ), dtype=np.int32)
    ray_id = 0
    for i in range(index_ray.shape[0]):
        if index_ray[i] > ray_id:
            ray_id = index_ray[i]
        num_intersect[ray_id] = num_intersect[ray_id] + 1
    num_intersect_prefix_sum = np.cumsum(num_intersect)

    locations_light_dist = np.sqrt(np.sum(np.square(locations - light_pos_curr), axis=-1))

    dist_near = np.ones(ray_num) * 1e6
    dist_far = np.zeros(ray_num)
    intersect_near = np.zeros(shape=(ray_num, 3))
    intersect_far = np.zeros(shape=(ray_num, 3))

    # sort by distance from light
    for i in range(ray_num):
        start = 0 if i == 0 else num_intersect_prefix_sum[i - 1]
        end = num_intersect_prefix_sum[i]

        if start == end:
            dist_near[i] = vert_cam_dist[i]
            dist_far[i] = vert_cam_dist[i]
            intersect_near[i] = vert_pos[i]
            intersect_far[i] = vert_pos[i]
            continue

        sort_idx = locations_light_dist[start: end].argsort()
        sort_idx = sort_idx + start
        locations[start: end] = locations[sort_idx]
        intersect_part_id[start: end] = intersect_part_id[sort_idx]
        on_inbody_faces[start: end] = on_inbody_faces[sort_idx]
        locations_light_dist[start: end] = locations_light_dist[sort_idx]

        dist_near[i] = locations_light_dist[start]
        dist_far[i] = locations_light_dist[end - 1]
        intersect_near[i] = locations[start]
        intersect_far[i] = locations[end - 1]

    # sample training points around the ray-mesh intersection range
    near_far_dist = dist_far - dist_near
    intersect_far = intersect_far + 0.5 * np.exp(-near_far_dist.reshape(-1, 1)) * ray_dir

    near_weights = np.linspace(start=-end_ratio, stop=1 + end_ratio, num=sample_point_per_ray)
    far_weights = 1.0 - near_weights

    sampled_points = np.einsum('ij,k->ikj', intersect_near, near_weights) + np.einsum('ij,k->ikj',
                                                                                      intersect_far,
                                                                                      far_weights)

    sampled_points = sampled_points + sample_noise_on_ray
    sampled_points_dist = np.sqrt(np.sum(np.square(sampled_points - light_pos_curr), axis=-1))

    mid_points_list = []

    # use midpoint of intersection as threshold of visibility (skip in-body points)
    ray_part_thresh = np.ones(shape=(ray_num, part_num)) * 1e6
    for i in range(ray_num):
        start = 0 if i == 0 else num_intersect_prefix_sum[i - 1]
        end = num_intersect_prefix_sum[i]
        if start == end:
            pass
        # the part mesh is watertight
        part_first_intersect = {}
        part_second_intersect = {}
        for j in range(start, end):
            is_invalid = on_inbody_faces[j]
            if intersect_part_id[j] not in part_first_intersect.keys():
                if is_invalid:
                    is_found = 0
                    for k in np.arange(start=j - 1, stop=start - 1, step=-1):
                        if not on_inbody_faces[k]:
                            part_first_intersect[intersect_part_id[j]] = max(locations_light_dist[k],
                                                                             locations_light_dist[j] - 0.1)
                            is_found = 1
                            break
                    if is_found == 0:
                        part_first_intersect[intersect_part_id[j]] = locations_light_dist[j]
                else:
                    part_first_intersect[intersect_part_id[j]] = locations_light_dist[j]

            elif intersect_part_id[j] not in part_second_intersect.keys():
                if is_invalid:
                    is_found = 0
                    for k in np.arange(start=j + 1, stop=end, step=1):
                        if not on_inbody_faces[k]:
                            # part_second_intersect[intersect_part_id[j]] = locations_light_dist[k]
                            part_second_intersect[intersect_part_id[j]] = min(locations_light_dist[k],
                                                                              locations_light_dist[j] + 0.1)
                            is_found = 1
                            break
                    if is_found == 0:
                        part_second_intersect[intersect_part_id[j]] = locations_light_dist[j]
                else:
                    part_second_intersect[intersect_part_id[j]] = locations_light_dist[j]

        if len(part_first_intersect) != len(part_second_intersect):
            for k in part_first_intersect.keys():
                if k not in part_second_intersect.keys():
                    part_second_intersect[k] = part_first_intersect[k]
                    # print(part_first_intersect)
                    # print(part_second_intersect)
                    # print(k)

        for k in part_first_intersect.keys():
            mid_points = (part_first_intersect[k] + part_second_intersect[
                k]) / 2.0  # part_second_intersect[k]
            ray_part_thresh[i, k] = mid_points
            mid_points_list.append(light_pos_curr + ray_dir[i] * mid_points)

    part_visibility = (np.tile(sampled_points_dist[:, :, None], (1, 1, part_num)) <
                       np.tile(ray_part_thresh[:, None, :], (1, sample_point_per_ray, 1)))

    return sampled_points.astype(np.float32), part_visibility


class Visualizer:
    def __init__(self):
        result_dir = 'data/lvis_dataset/{}'.format(cfg.exp_name)
        print(
            colored('the results are saved at {}'.format(result_dir),
                    'yellow'))

        self.light_pos = np.load(cfg.light_pos_path)
        self.sampled_view_num = 4

    def visualize(self, output, batch):
        full_mesh_v = output['full_mesh_vertices']
        part_mesh_v = output['part_mesh_vertices']
        part_mesh_f = output['part_mesh_faces']
        frame_index = batch['frame_index'][0].item()
        out_path = './data/lvis_dataset/{}/'.format(cfg.exp_name)

        if frame_index == 0:
            self.inbody_facec_idx = []
            for i in range(cfg.part_num):
                self.inbody_facec_idx.append(np.load(out_path + '/part_mesh/inbody_faces_idx_{:04d}.npy'.format(i)))
            os.makedirs(out_path + 'sampled_points/', exist_ok=True)
            os.makedirs(out_path + 'part_visibility/', exist_ok=True)

        for i in range(self.sampled_view_num):
            light_pos = self.light_pos[frame_index * self.sampled_view_num + i]
            intersect_info, vert_pos = compute_intersect(full_mesh_v, part_mesh_v, part_mesh_f, light_pos)
            sampled_points, part_visibility = parse_intersect(intersect_info, vert_pos, light_pos, self.inbody_facec_idx)

            np.save(out_path + 'sampled_points/{:04d}_{:04d}.npy'.format(frame_index, i), sampled_points)
            np.save(out_path + 'part_visibility/{:04d}_{:04d}.npy'.format(frame_index, i), part_visibility)
