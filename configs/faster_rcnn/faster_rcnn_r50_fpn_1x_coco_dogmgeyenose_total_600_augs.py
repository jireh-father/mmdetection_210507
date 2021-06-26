_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_class2.py',
    # '../_base_/datasets/coco_detection_custom.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'DogMergedEyesNoseDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)


albu_train_transforms = [
    dict(
        type='RandomRotate90',
        p=0.5),
    dict(type='JpegCompression', quality_lower=70, quality_upper=100, p=0.3),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.3),
]


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(500, 500), (700, 700)],
        multiscale_mode='range',
        keep_ratio=True),
dict(type='PhotoMetricDistortion'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),

    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'],
        meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg',
                   'pad_shape', 'scale_factor'))
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(600, 600)],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
classes = ('dog_eye', 'dog_nose')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'merged_dog_eyes_nose/train.json',
        img_prefix=data_root + 'merged_dog_eyes_nose/train_images',
        classes=classes, pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'merged_dog_eyes_nose/val.json',
        img_prefix=data_root + 'merged_dog_eyes_nose/val_images',
        classes=classes, pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'merged_dog_eyes_nose/val.json',
        img_prefix=data_root + 'merged_dog_eyes_nose/val_images',
        classes=classes, pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
