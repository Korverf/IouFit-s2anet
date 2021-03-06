PI = 3.141592653
# model settings
model = dict(
    type='RetinaNet',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5),
    bbox_head=dict(
        type='PIOURetinaHeadRotated',
        num_classes=2,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        octave_base_scale=4,
        scales_per_octave=3,
        anchor_ratios=[0.5, 1.0, 2.0],
        anchor_angles=[0., PI/3, PI/6, PI/2], # more angles lead to a mAP improvement.
        anchor_strides=[8, 16, 32, 64, 128],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='PIoULoss', loss_weight=1.0, img_size=800)))
# training and testing settings
train_cfg = dict(
    assigner=dict(
        type='MaxIoUAssigner',
        pos_iou_thr=0.5,
        neg_iou_thr=0.4,
        min_pos_iou=0,
        ignore_iof_thr=-1,
        iou_calculator=dict(type='BboxOverlaps2D_rotated')),
    bbox_coder=dict(type='DeltaXYWHABBoxCoder',
                    target_means=(0., 0., 0., 0., 0.),
                    target_stds=(1., 1., 1., 1., 1.),
                    clip_border=True),
    reg_decoded_bbox=True, # Set True to use IoULoss
    allowed_border=-1,
    pos_weight=-1,
    debug=False)
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),  # 15fps
    max_per_img=2000)
# dataset settings
dataset_type = 'HRSC2016Dataset'
data_root = '/home/yyw/yyf/Dataset/HRSC2016/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RotatedResize', img_scale=(800, 512), keep_ratio=True),
    dict(type='RotatedRandomFlip', flip_ratio=0.5),
    dict(type='RandomRotate', rate=0.5, angles=[30, 60, 90, 120, 150], auto_bound=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 512),
        flip=False,
        transforms=[
            dict(type='RotatedResize', keep_ratio=True),
            dict(type='RotatedRandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'Train/train.txt',
        img_prefix=data_root + 'Train/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/test.txt',
        img_prefix=data_root + 'Test/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Test/test.txt',
        img_prefix=data_root + 'Test/',
        pipeline=test_pipeline))
evaluation = dict(
    gt_dir='/home/yyw/yyf/Dataset/HRSC2016/Test/Annotations/',
    imagesetfile='/home/yyw/yyf/Dataset/HRSC2016/Test/test.txt')
# optimizer
optimizer = dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[48, 66])
checkpoint_config = dict(interval=12)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
# runtime settings
total_epochs = 72
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/retinanet_obb_r50_fpn_6x_hrsc2016_piou_rerun'
load_iou_fit_module = None
load_from = None
resume_from = None
workflow = [('train', 1)]


#AP50: 83.28     AP75: 47.35      mAP: 46.12
#APs:  [83.28039928464615, 82.3049667965044, 73.39244114445582, 70.73300385672808,
# 60.942021680316806, 47.3471998621582, 28.167644218735315, 11.769563008130081,
# 3.225806451612903, 0.07215007215007214]
