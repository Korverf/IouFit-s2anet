# model settings
model = dict(
    type='S2ANetDetector',
    pretrained='torchvision://resnet101',
    backbone=dict(
        type='ResNet',
        depth=101,
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
        type='S2ANetHead',
        num_classes=2,
        in_channels=256,
        feat_channels=256,
        stacked_convs=2,
        with_orconv=True,
        anchor_ratios=[1.0],
        anchor_strides=[8, 16, 32, 64, 128],
        anchor_scales=[4],
        target_means=[.0, .0, .0, .0, .0],
        target_stds=[1.0, 1.0, 1.0, 1.0, 1.0],
        loss_fam_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_fam_bbox=dict(type='RotatedIoULoss', loss_weight=1.0),
        loss_odm_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_odm_bbox=dict(type='RotatedIoULoss', loss_weight=1.0)))
# training and testing settings
train_cfg = dict(
    fam_cfg=dict(
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
        debug=False),
    odm_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.6,
            neg_iou_thr=0.5,
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
        debug=False))
test_cfg = dict(
    nms_pre=2000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms_rotated', iou_thr=0.1),
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
    imgs_per_gpu=1,
    workers_per_gpu=1,
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
optimizer = dict(type='SGD', lr=0.00125, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[194, 260])
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
total_epochs = 324
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/s2anet_r101_fpn_27x_hrsc2016_iouloss'
load_iou_fit_module = None
load_from = None
resume_from = None
workflow = [('train', 1)]
