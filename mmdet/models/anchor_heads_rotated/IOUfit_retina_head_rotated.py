import numpy as np
import torch.nn as nn
from mmcv.cnn import normal_init

from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob
from .anchor_head_rotated import AnchorHeadRotated
from mmdet.core import (build_bbox_coder, rotated_box_to_poly)
from mmdet.models.utils import IOUfitFC3 as IOUfitModule#IOUfitModule
from mmdet.ops import obb_overlaps
import torch

@HEADS.register_module
class IOUfitRetinaHeadRotated(AnchorHeadRotated):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 octave_base_scale=4,
                 scales_per_octave=3,
                 anchor_angles=[0.,],
                 conv_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.octave_base_scale = octave_base_scale
        self.scales_per_octave = scales_per_octave
        self.anchor_angles = anchor_angles
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        octave_scales = np.array(
            [2**(i / scales_per_octave) for i in range(scales_per_octave)])
        anchor_scales = octave_scales * octave_base_scale
        super(IOUfitRetinaHeadRotated, self).__init__(
            num_classes, in_channels, anchor_scales=anchor_scales, anchor_angles=anchor_angles, **kwargs)
        self.IOUfit = IOUfitModule(in_features=16, hidden_features=16)

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        self.retina_cls = nn.Conv2d(
            self.feat_channels,
            self.num_anchors * self.cls_out_channels,
            3,
            padding=1)
        self.retina_reg = nn.Conv2d(
            self.feat_channels, self.num_anchors * 5, 3, padding=1)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.retina_cls, std=0.01, bias=bias_cls)
        normal_init(self.retina_reg, std=0.01)

    def forward_single(self, x):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.retina_cls(cls_feat)
        bbox_pred = self.retina_reg(reg_feat)
        return cls_score, bbox_pred
    
    def loss_single(self, cls_score, bbox_pred, anchors, labels, label_weights,
                    bbox_targets, bbox_weights, num_total_samples, cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        assert reg_decoded_bbox==True
        loss_bbox = torch.zeros(1).cuda(bbox_pred.get_device())
        pos_inds = labels > 0
        if pos_inds.any():
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_bbox_coder(bbox_coder_cfg)
            anchors = anchors.reshape(-1, 5)
            bbox_pred = bbox_coder.decode(anchors, bbox_pred)
            pos_bbox_target = bbox_targets[pos_inds.type(torch.bool)]  #
            pos_bbox_pred = bbox_pred.view(
                    bbox_pred.size(0), 5)[pos_inds]
            pos_poly_target, pos_poly_pred = rotated_box_to_poly(pos_bbox_target), rotated_box_to_poly(pos_bbox_pred)

            #归一化
            x = torch.cat([pos_poly_pred[:, 0::2], pos_poly_target[:, 0::2]], dim=1) #N,8
            xmin, _ = torch.min(x, dim=1)
            xmax, _ = torch.max(x, dim=1)
            y = torch.cat([pos_poly_pred[:, 1::2], pos_poly_target[:, 1::2]], dim=1)  #N,8
            ymin, _ = torch.min(y, dim=1) #N
            ymax, _ = torch.max(y, dim=1)
            w = torch.sub(xmax, xmin).unsqueeze(-1)
            h = torch.sub(ymax, ymin).unsqueeze(-1)
            long_side, _ = torch.max(torch.cat([w, h], dim=1), dim=1) #N

            pos_poly_pred[:, 0::2] = pos_poly_pred[:, 0::2] - xmin.unsqueeze(-1)
            pos_poly_pred[:, 1::2] = pos_poly_pred[:, 1::2] - ymin.unsqueeze(-1)
            pos_poly_pred = pos_poly_pred / long_side.unsqueeze(-1)

            pos_poly_target[:, 0::2] = torch.sub(pos_poly_target[:, 0::2], xmin.unsqueeze(-1))
            pos_poly_target[:, 1::2] = torch.sub(pos_poly_target[:, 1::2], ymin.unsqueeze(-1))
            pos_poly_target = pos_poly_target / long_side.unsqueeze(-1)

            # caculate IOU as gt
            IoU_targets = obb_overlaps(pos_bbox_pred, pos_bbox_target.detach(), is_aligned=True).squeeze(1)\
               .clamp(min=1e-6, max=1) #未归一化的5参数框
            # use MLP to fit the rotate IOU
            iou_fit_value = self.IOUfit(pos_poly_pred, pos_poly_target) #0~1
            iou_fit_value = iou_fit_value[:, 0].clamp(min=1e-6, max=1)

            loss_bbox = self.loss_bbox(
                iou_fit_value,
                weight=bbox_weights[pos_inds],
                avg_factor=num_total_samples,
                reduce=True
                )
        return loss_cls, loss_bbox
