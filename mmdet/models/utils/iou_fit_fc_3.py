import torch.nn as nn
# from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
#                       xavier_init)
import torch
#from torch.nn.modules import loss
# from torch.nn.modules.linear import _LinearWithBias
# from torch.nn.init import xavier_uniform_
# from torch.nn.init import constant_
# from torch.nn.init import xavier_normal_
# from torch.nn import functional as F
# import numpy as np
# from box_iou_rotated import obb_overlaps
#from transform_rbbox import (polygonToRotRectangle_batch, RotBox2Polys_torch, RotBox2Polys, obb2poly)


class IOUfitFC3(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(IOUfitFC3, self).__init__()
        self.fc1 = FC(in_channels=in_features, out_channels=512, act_func='ReLU', with_act=True)
        self.bottle2 = Bottleneck(in_channels=512)
        self.bottle3 = Bottleneck(in_channels=512)
        self.bottle4 = Bottleneck(in_channels=512)
        self.fc5 = FC(in_channels=512, out_channels=1, with_act=False)

        self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss_weight = 10.0

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1) #N,16
        out = self.fc1(input) #N,512
        out = self.bottle2(out)
        out = self.bottle3(out)
        out = self.bottle4(out)
        out = self.fc5(out)
        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels):
        super(Bottleneck, self).__init__()
        self.fc1 = nn.Linear(in_features=in_channels, out_features=in_channels // 4)
        self.fc2 = nn.Linear(in_features=in_channels // 4, out_features=in_channels)
        self.relu = nn.ReLU()

    def forward(self, input):#  N,C
        out = self.relu(self.fc1(input))#N,C/4
        out = self.relu(self.fc2(out))  # N,C
        return out

class Conv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 with_act=True,
                 with_bn=False,
                 act_func='ReLU',
                 add_residual=False):
        super(Conv, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        #self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                stride=stride, padding=padding))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            if act_func=='ReLU':
                self.activate = nn.ReLU()
                layers.append(self.activate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out

class FC(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_act=True,
                 with_bn=False,
                 act_func='ReLU',
                 add_residual=False):
        super(FC, self).__init__()
        # self.in_channels = in_channels
        # self.feedforward_channels = feedforward_channels
        # self.act_cfg = act_cfg
        self.add_residual = add_residual
        layers = nn.ModuleList()
        #self.relu = build_activation_layer(dict(type='ReLU'))
        layers.append(nn.Linear(in_channels, out_channels))
        if with_bn == True:
            layers.append(nn.BatchNorm1d(out_channels))
        if with_act == True:
            if act_func=='ReLU':
                self.activate = nn.ReLU()
                layers.append(self.activate)

        self.layers = nn.Sequential(*layers)

    def forward(self, x, residual=None):
        """Forward function for `FFN`."""
        out = self.layers(x)
        if not self.add_residual:
            return out
        if residual is None:
            residual = x
        return residual + out


def get_model(in_features, hidden_features=None):

    model = IOUfitFC3(in_features, hidden_features)  # 101
    return model