import torch
from collections import OrderedDict

saved_pth = torch.load('../work_dirs/iou_fit_fc_3_3/iter_1900000.pth')
saved_state_dict = saved_pth['state_dict']
new_state=OrderedDict()
for k, v in saved_state_dict.items():
    if k=='fc1.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.weight':v})
    elif k=='fc1.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc1.layers.0.bias': v})
    elif k=='bottle2.fc1.weight':
        new_state.update({'bbox_head.IOUfit.bottle2.fc1.weight': v})
    elif k=='bottle2.fc1.bias':
        new_state.update({'bbox_head.IOUfit.bottle2.fc1.bias': v})
    elif k == 'bottle2.fc2.weight':
        new_state.update({'bbox_head.IOUfit.bottle2.fc2.weight': v})
    elif k == 'bottle2.fc2.bias':
        new_state.update({'bbox_head.IOUfit.bottle2.fc2.bias': v})
    elif k=='bottle3.fc1.weight':
        new_state.update({'bbox_head.IOUfit.bottle3.fc1.weight': v})
    elif k=='bottle3.fc1.bias':
        new_state.update({'bbox_head.IOUfit.bottle3.fc1.bias': v})
    elif k=='bottle3.fc2.weight':
        new_state.update({'bbox_head.IOUfit.bottle3.fc2.weight': v})
    elif k=='bottle3.fc2.bias':
        new_state.update({'bbox_head.IOUfit.bottle3.fc2.bias': v})
    elif k == 'bottle4.fc1.weight':
        new_state.update({'bbox_head.IOUfit.bottle4.fc1.weight': v})
    elif k == 'bottle4.fc1.bias':
        new_state.update({'bbox_head.IOUfit.bottle4.fc1.bias': v})
    elif k == 'bottle4.fc2.weight':
        new_state.update({'bbox_head.IOUfit.bottle4.fc2.weight': v})
    elif k == 'bottle4.fc2.bias':
        new_state.update({'bbox_head.IOUfit.bottle4.fc2.bias': v})
    elif k == 'fc5.layers.0.weight':
        new_state.update({'bbox_head.IOUfit.fc5.layers.0.weight': v})
    elif k == 'fc5.layers.0.bias':
        new_state.update({'bbox_head.IOUfit.fc5.layers.0.bias': v})


#torch.save(new_state, '../work_dirs/iou_fit_conv_7/iou_fit_conv_retinanet.pth')

# # for old pytorch version to load:
torch.save(new_state, '../work_dirs/iou_fit_fc_3_3/iou_fit_fc_retinanet.pth', _use_new_zipfile_serialization=False)







