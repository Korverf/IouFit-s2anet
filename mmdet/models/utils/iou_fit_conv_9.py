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


class IOUfitConv9(nn.Module):
    def __init__(self, in_features, hidden_features=None):
        super(IOUfitConv9, self).__init__()
        self.fc1 = FC(in_channels=in_features, out_channels=512,
                      act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        # self.fc2 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
        #               act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        # self.fc3 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
        #               act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        # self.fc4 = FC(in_channels=512, feedforward_channels=512, out_channels=512,
        #               act_func='ReLU', add_residual=False, with_act=True, with_bn=False)
        # self.fc5 = FC(in_channels=512, feedforward_channels=512, out_channels=1,
        #               add_residual=False, with_act=False, with_bn=False)
        self.conv2 = Conv(in_channels=1, out_channels=2, kernel_size=5, stride=2, padding=2) #256
        self.conv3 = Conv(in_channels=2, out_channels=4, kernel_size=5, stride=2, padding=2) #128
        self.conv4 = Conv(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=2) #64
        self.conv5 = Conv(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2) #32
        self.conv6 = Conv(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2) #16
        self.fc7 = FC(in_channels=16, out_channels=1, add_residual=False, with_act=False, with_bn=False)
        self.fc8 = FC(in_channels=32, out_channels=1, add_residual=False, with_act=False, with_bn=False)

        # self.out_fc = Linear(hidden_features,1)
        #self.sigmoid = nn.Sigmoid()
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.loss_weight = 10.0
        #self.scale_factor = 1

    def forward(self, pred, gt):
        input = torch.cat([pred, gt], dim=-1) #N,16
        out = self.fc1(input) #N,512
        out = torch.unsqueeze(out, dim=1) #N,1,512
        out = self.conv2(out) #N,2,256
        out = self.conv3(out) #N,4,128
        out = self.conv4(out) #N,8,64
        out = self.conv5(out) #N,16,32
        out = self.conv6(out) #N,32,16
        #out = out.view(out.size(0), -1)  #N,32*16
        out = self.fc7(out) #N,32,1
        out = out.squeeze(-1)#N,32
        out = self.fc8(out)  # N,1
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
                 with_bn=True,
                 act_func='ReLU',
                 add_residual=True):
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

    model = IOUfitConv9(in_features, hidden_features)  # 101
    return model