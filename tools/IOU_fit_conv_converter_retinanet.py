import torch
from collections import OrderedDict

saved_pth = torch.load('../work_dirs/iou_fit_conv_9/iter_7690000.pth')
saved_state_dict = saved_pth['state_dict']
new_state=OrderedDict()
for k, v in saved_state_dict.items():
    if k=='fc1.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.weight':v})
    elif k=='fc1.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.bias': v})
    elif k=='conv2.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.conv2.layers.0.weight': v})
    elif k=='conv2.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.conv2.layers.0.bias': v})
    elif k == 'conv3.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.conv3.layers.0.weight': v})
    elif k == 'conv3.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.conv3.layers.0.bias': v})
    elif k=='conv4.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.conv4.layers.0.weight': v})
    elif k=='conv4.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.conv4.layers.0.bias': v})
    elif k=='conv5.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.conv5.layers.0.weight': v})
    elif k=='conv5.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.conv5.layers.0.bias': v})
    elif k == 'conv6.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.conv6.layers.0.weight': v})
    elif k == 'conv6.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.conv6.layers.0.bias': v})
    elif k == 'fc7.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc7.layers.0.weight': v})
    elif k == 'fc7.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc7.layers.0.bias': v})
    elif k == 'fc8.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc8.layers.0.weight': v})
    elif k == 'fc8.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc8.layers.0.bias': v})


#torch.save(new_state, '../work_dirs/iou_fit_conv_7/iou_fit_conv_retinanet.pth')

# # for old pytorch version to load:
torch.save(new_state, '../work_dirs/iou_fit_conv_9/iou_fit_conv_retinanet.pth', _use_new_zipfile_serialization=False)







