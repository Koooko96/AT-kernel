# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='ResNet_ATkernel',
        layers=[3, 4, 6, 3], 
        replace_conv2_with_ATkernel=[0, 0, 0, 1],
        stride2or1block=[1, 1], 
        ATkernel_size=3,
        group_channels=4,
        resolution=224),
    neck=dict(type='GlobalAveragePooling'),
    head=dict(
        type='LinearClsHead',
        num_classes=1000,
        in_channels=2048,
        loss=dict(
            type='LabelSmoothLoss',
            loss_weight=1.0,
            label_smooth_val=0.1,
            num_classes=1000),
        topk=(1, 5),
    ))


# dataset settings
dataset_type = 'ImageNet'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', size=(256, -1)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]
data = dict(
    samples_per_gpu=128,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_prefix='To Be Fixed (.../imagenet/train)',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_prefix='To Be Fixed (.../imagenet/val)',
        ann_file='To Be Fixed (.../imagenet/meta/val.txt)',
        pipeline=test_pipeline),
    test=dict(
        # replace `data/val` with `data/test` for standard test
        type=dataset_type,
        data_prefix='To Be Fixed (.../imagenet/val)',
        ann_file='To Be Fixed (.../imagenet/meta/val.txt)',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='accuracy')


# optimizer
# change the lr of (batch size / 256) * 0.1
# batch size is indicated in the name of this config file (512 = 128 per GPU x 4 GPU)
optimizer = dict(
    type='SGD', lr=0.2, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    min_lr=0,
    warmup='linear',
    warmup_iters=5,
    warmup_ratio=0.01,
    warmup_by_epoch=True)
runner = dict(type='EpochBasedRunner', max_epochs=100)


# checkpoint saving
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]