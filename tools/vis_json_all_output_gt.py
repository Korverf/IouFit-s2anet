# coding=utf-8
import copy
import json
import numpy as np
from shapely.geometry import Polygon
import os
import mmcv
import cv2

'''
    把各种错误预测结果在原图基础上画出，不分类别，预测框和真值框用两种颜色，把类别显示出来。
'''

def draw_poly_detections(img, detections, labels, class_names, putText=True, showStart=False, colormap=None):
    """

    :param img:
    :param detections:
    :param class_names:
    :param scale:
    :param cfg:
    :param threshold:
    :return:
    """
    import pdb
    import cv2
    import random
    assert isinstance(class_names, (tuple, list))

    img = mmcv.imread(img)
    color_white = (255, 255, 255)

    for j, name in enumerate(class_names):
        if colormap is None:
            color_gt = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            color_pred = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
        else:
            color_gt = colormap[0]
            color_pred = colormap[1]
        try:
            dets = detections[name]
            gts = labels[name]
        except:
            pdb.set_trace()
        # 绘制预测框
        if len(dets) > 0:
            for det in dets:
                bbox = det[:8]
                score = det[-1]
                bbox = list(map(int, bbox))
                # print('===================')
                # print(bbox)
                # print(showStart,putText)
                if showStart:# 在第一个坐标绘制一个小圆点
                    cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
                #绘制边框
                for i in range(3):
                    cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i+1) * 2], bbox[(i+1) * 2 + 1]),
                             color=color_pred, thickness=2,lineType=cv2.LINE_AA)
                cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color_pred, thickness=2,lineType=cv2.LINE_AA)
                #在边框周围放置类别及置信度信息
                if putText:
                    cv2.putText(img, '%s %.3f' % (class_names[j], score), (bbox[0], bbox[1] + 10),
                                color=color_white, fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
        # 绘制真值框
        if len(gts) > 0:
            for gt in gts:
                bbox = gt[:8]
                bbox = list(map(int, bbox))
                # print('===================')
                # print(bbox)
                # print(showStart, putText)
                if showStart:  # 在第一个坐标绘制一个小圆点
                    cv2.circle(img, (bbox[0], bbox[1]), 3, (0, 0, 255), -1)
                # 绘制边框
                for i in range(3):
                    cv2.line(img, (bbox[i * 2], bbox[i * 2 + 1]), (bbox[(i + 1) * 2], bbox[(i + 1) * 2 + 1]),
                             color=color_gt, thickness=2, lineType=cv2.LINE_AA)
                cv2.line(img, (bbox[6], bbox[7]), (bbox[0], bbox[1]), color=color_gt, thickness=2, lineType=cv2.LINE_AA)
                # 在边框周围放置类别及置信度信息
                if putText:
                    cv2.putText(img, '%s' % (class_names[j]), (bbox[0], bbox[1] + 10), color=color_white,
                                fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5)
    return img


def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (x, y, w, h)
    :param rec2: (x, y, w, h)
    :return: scala value of IoU
    """
    g = np.asarray(rec1)
    p = np.asarray(rec2)

    g = Polygon(g[:8].reshape((4, 2)))
    p = Polygon(p[:8].reshape((4, 2)))
    if not g.is_valid or not p.is_valid:
        return 0
    inter = Polygon(g).intersection(Polygon(p)).area
    union = g.area + p.area - inter
    if union == 0:
        return 0
    else:
        return inter / union


img_root = '/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/Dataset/HRSC2016/Test/images'
vis_root = '/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/Dataset/HRSC2016/vis_with_gt/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016_1'
if not os.path.exists(vis_root):
    os.mkdir(vis_root)
classes = ['ship']

#gt_list = json.load(open('/mnt/Dataset/rsaicp/data/val_nms/val_F1_nms.json'))
gt_list = json.load(open('/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/Dataset/HRSC2016/Test/HRSC_L1_test_F1.json'))
pred_list = json.load(open('/media/yyw/607ca214-10cb-4af6-af97-eac656e135d1/yyw/yyf/projects/s2anet/work_dirs/'
                           'IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016_1/results_txt/IOUfit_retinanet_obb_r50_fpn_6x_hrsc2016_1.json'))
threshold = 0.0  # 0.95

colormap = [
    (0, 255, 0),
    (0, 0, 255)]

names = []
img_list = dict()
for gt in gt_list:
    gt_name = gt['image_name']
    labels_list = gt['labels']
    labels_pred_list = None
    for pred in pred_list:
        if pred['image_name'] == gt_name:
            labels_pred_list = pred['labels']
    img_path = os.path.join(img_root, gt_name)
    vis_path = os.path.join(vis_root, gt_name)

    detections = {classes[i]: np.zeros((0, 9)) for i in range(1)}
    labels = {classes[i]: np.zeros((0, 8)) for i in range(1)}
    for label in labels_list:
        class_gt = label['category_id']
        loc1, loc2, loc3, loc4 = label['points']
        p00, p01 = float(loc1[0]), float(loc1[1])
        p10, p11 = float(loc2[0]), float(loc2[1])
        p20, p21 = float(loc3[0]), float(loc3[1])
        p30, p31 = float(loc4[0]), float(loc4[1])
        gt_loc = (p00, p01, p10, p11, p20, p21, p30, p31)
        labels[class_gt] = np.concatenate((labels[class_gt], np.asarray([gt_loc])))

    if labels_pred_list is not None:
        for label_pred in labels_pred_list:  # 该图的预测结果按类别放在pred_dict中
            class_pred = label_pred['category_id']
            conf = label_pred['confidence']
            # if conf < threshold[class_pred]:
            if conf < threshold:
                continue
            loc1, loc2, loc3, loc4 = label_pred['points']
            p00, p01 = float(loc1[0]), float(loc1[1])
            p10, p11 = float(loc2[0]), float(loc2[1])
            p20, p21 = float(loc3[0]), float(loc3[1])
            p30, p31 = float(loc4[0]), float(loc4[1])
            detections[class_pred] = np.concatenate((detections[class_pred], np.asarray([(p00, p01, p10, p11, p20, p21, p30, p31, conf)])))

    img = draw_poly_detections(img_path, detections, labels, classes, colormap=colormap, showStart=True, putText=True)
    cv2.imwrite(vis_path, img)








