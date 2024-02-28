import numpy as np
import pickle

import torch
import torch.nn as nn

from lib.config import cfg
from . import lbs

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()
    

class BodyModel(nn.Module):

    def __init__(self, model_path=None):
        super(BodyModel, self).__init__()
        self.load_smpl_model(model_path)

    def load_smpl_model(self, model_path=None):
        if model_path is None:
            smpl = read_pickle(cfg.smpl_model_path)
        else:
            smpl = read_pickle(model_path)

        J_regressor = torch.from_numpy(smpl['J_regressor'].toarray().astype(np.float32))
        kintree_table = torch.from_numpy(np.array(smpl['kintree_table']).astype(np.int64))
        v_template = torch.from_numpy(np.array(smpl['v_template']).astype(np.float32))
        lbs_weights = torch.from_numpy(np.array(smpl['weights']).astype(np.float32))
        posedirs = torch.from_numpy(np.array(smpl['posedirs']).astype(np.float32))
        shapedirs = torch.from_numpy(np.array(smpl['shapedirs'])[:, :, :10].astype(np.float32))

        smpl_v_num = v_template.shape[0]
        posedirs = posedirs.reshape(smpl_v_num * 3, -1).T
        v_template = v_template[None]

        self.register_buffer('v_template', v_template)
        self.register_buffer('posedirs', posedirs)
        self.register_buffer('shapedirs', shapedirs)
        self.register_buffer('J_regressor', J_regressor)
        self.register_buffer('lbs_weights', lbs_weights)
        self.register_buffer('kintree_table', kintree_table)

    def forward(self, betas, full_pose):
        verts_posed, Jtrs_posed, Jtrs, bone_transforms, _, minimal_shape, = lbs.lbs(betas=betas,
                                                                                    pose=full_pose,
                                                                                    v_template=self.v_template.clone(),
                                                                                    clothed_v_template=None,
                                                                                    shapedirs=self.shapedirs.clone(),
                                                                                    posedirs=self.posedirs.clone(),
                                                                                    J_regressor=self.J_regressor.clone(),
                                                                                    parents=self.kintree_table[0].long(),
                                                                                    lbs_weights=self.lbs_weights.clone(),
                                                                                    dtype=torch.float32)
        return verts_posed, Jtrs_posed, Jtrs, bone_transforms, minimal_shape


