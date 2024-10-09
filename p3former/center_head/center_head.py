import torch
import gc
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from p3former.task_modules.convolutions.spconv import conv3x3

@MODELS.register_module()
class _OffsetPredictor(nn.Module):
    """OffsetPredictor."""
    def __init__(self, init_size, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pt_fea_dim = 256

        self.conv1 = conv3x3(self.pt_fea_dim, self.pt_fea_dim, indice_key='offset_head_conv1')
        self.bn1 = nn.BatchNorm1d(self.pt_fea_dim)
        self.act1 = nn.LeakyReLU()
        self.conv2 = conv3x3(self.pt_fea_dim, 2 * init_size, indice_key='offset_head_conv2')
        self.bn2 = nn.BatchNorm1d(2 * init_size)
        self.act2 = nn.LeakyReLU()
        self.conv3 = conv3x3(2 * init_size, init_size, indice_key='offset_head_conv3')
        self.bn3 = nn.BatchNorm1d(init_size)
        self.act3 = nn.LeakyReLU()

        self.offset = nn.Sequential(
            nn.Linear(init_size+3, init_size, bias=True),
            nn.BatchNorm1d(init_size),
            nn.ReLU()
        )

        self.offset_linear = nn.Linear(init_size, embedding_dim, bias=True)

    def forward(self, fea, batch_inputs_dict) -> list:
        fea = self.conv1(fea)
        fea.features = self.act1(self.bn1(fea.features))
        fea = self.conv2(fea)
        fea.features = self.act2(self.bn2(fea.features))
        fea = self.conv3(fea)
        fea.features = self.act3(self.bn3(fea.features))

        grid_ind = batch_inputs_dict['voxels']['grid']
        xyz = batch_inputs_dict['points']
        fea = fea.dense()
        fea = fea.permute(0, 2, 3, 4, 1)
        pt_ins_fea_list = []
        for batch_i, grid_ind_i in enumerate(grid_ind):

            pt_ins_fea_list.append(fea[batch_i, 
                                       np.array(grid_ind[batch_i][:,0]), 
                                       np.array(grid_ind[batch_i][:,1]), 
                                       np.array(grid_ind[batch_i][:,2])])

        pt_pred_offsets_list = []
        for batch_i, pt_ins_fea in enumerate(pt_ins_fea_list):
            pt_pred_offsets_list.append(self.offset_linear(self.offset(torch.cat([pt_ins_fea, xyz[batch_i][:,:3].cuda()],dim=1))))
        
        return pt_pred_offsets_list
    