import torch
from lib.utils.optimizer.radam import RAdam


_optimizer_factory = {
    'adam': torch.optim.Adam,
    'radam': RAdam,
    'sgd': torch.optim.SGD
}


def make_optimizer(cfg, net, lr=None, weight_decay=None, freeze_geo=True, freeze_smpl=True):
    params = []
    lr = cfg.train.lr if lr is None else lr
    weight_decay = cfg.train.weight_decay if weight_decay is None else weight_decay

    for key, value in net.named_parameters():
        if not value.requires_grad:
            continue
        # freeze geo model
        if freeze_geo:
            if key.startswith('deform_net'):
                continue
            if key.startswith('tpose_human.sdf_network'):
                continue
            if key.startswith('tpose_human.beta_network'):
                continue
        # freeze smpl paramaters
        if freeze_smpl:
            if key.startswith('body_poses'):
                continue
            if key.startswith('betas'):
                continue
        # do not optimize body shape
        if key.startswith('body_poses.betas'):
            continue
        # freeze vis model for the last stage
        if key.startswith('tpose_human.color_sg_network.visibility_network'):
            continue
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    if 'adam' in cfg.train.optim:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, weight_decay=weight_decay)
    else:
        optimizer = _optimizer_factory[cfg.train.optim](params, lr, momentum=0.9)

    return optimizer
