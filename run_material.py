import os
from lib.config import cfg, args
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import imageio


if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0
    cfg.eval = True

    network = make_network(cfg).cuda()
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch,
                 strict=False)

    network.eval()

    if cfg.novel_light:
        load_envmap(network, cfg.novel_light_path)

    save_envmap(network, cfg)

    data_loader = make_data_loader(cfg, is_train=False)

    if cfg.vis_pose_sequence:
        data_loader.dataset.set_beta(network.body_poses['betas'].detach().cpu().numpy())

    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)


def load_envmap(network, path):
    sg_path = os.path.join(path, 'sg_scaled_128.npy')
    sg_np = np.load(sg_path)

    sg_np = sg_np.astype(np.float32)
    lgtSGs = network.tpose_human.color_sg_network.envmap_material_network.lgtSGs
    load_sgs = torch.from_numpy(sg_np).to(lgtSGs.data.device)
    lgtSGs.data = load_sgs


def save_envmap(network, cfg):
    sg = network.tpose_human.color_sg_network.envmap_material_network.lgtSGs.data.clone().detach().cpu()
    sg[:, 3:] = torch.abs(sg[:, 3:])
    env_map = compute_envmap(sg, 256, 512).numpy()
    result_dir = os.path.join(cfg.result_dir, 'envmap')
    os.makedirs(result_dir, exist_ok=True)
    cv2.imwrite('{}/envmap.png'.format(result_dir), (env_map[..., [2, 1, 0]] / 1.0 * 255))
    np.save('{}/sg_128.npy'.format(result_dir), sg)
    imageio.imwrite('{}/envmap.exr'.format(result_dir), env_map, flags=0x0001)
    env_map = np.clip(np.power(env_map, 1./2.2), 0., 1.)
    cv2.imwrite('{}/envmap_gamma.png'.format(result_dir), (env_map[..., [2, 1, 0]] / 1.0 * 255))


def compute_envmap(lgtSGs, H, W, upper_hemi=False):
    # same convetion as blender    
    if upper_hemi:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi/2., H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    else:
        phi, theta = torch.meshgrid([torch.linspace(0., np.pi, H), 
                                     torch.linspace(1.0 * np.pi, -1.0 * np.pi, W)])
    viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), 
                            torch.sin(theta) * torch.sin(phi), 
                            torch.cos(phi)], dim=-1)    # [H, W, 3]
                            
    rgb = render_envmap_sg(lgtSGs, viewdirs)
    envmap = rgb.reshape((H, W, 3))
    return envmap


def render_envmap_sg(lgtSGs, viewdirs):
    viewdirs = viewdirs.to(lgtSGs.device)
    viewdirs = viewdirs.unsqueeze(-2)  # [..., 1, 3]

    # [M, 7] ---> [..., M, 7]
    dots_sh = list(viewdirs.shape[:-2])
    M = lgtSGs.shape[0]
    lgtSGs = lgtSGs.view([1,] * len(dots_sh) + [M, 7]).expand(dots_sh + [M, 7])
    
    lgtSGLobes = lgtSGs[..., :3] / (torch.norm(lgtSGs[..., :3], dim=-1, keepdim=True))
    lgtSGLambdas = torch.abs(lgtSGs[..., 3:4])
    lgtSGMus = torch.abs(lgtSGs[..., -3:]) 
    # [..., M, 3]
    rgb = lgtSGMus * torch.exp(lgtSGLambdas * \
        (torch.sum(viewdirs * lgtSGLobes, dim=-1, keepdim=True) - 1.))
    rgb = torch.sum(rgb, dim=-2)  # [..., 3]
    return rgb


if __name__ == '__main__':
    globals()['run_' + args.type]()
