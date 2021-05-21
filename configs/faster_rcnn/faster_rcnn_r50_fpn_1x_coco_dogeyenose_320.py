_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn_class3.py',
    # '../_base_/datasets/coco_detection_custom.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

# dataset settings
dataset_type = 'DogEyesNoseDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(320, 320), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=[(320, 320)],
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
classes = ('dog_left_eye', 'dog_right_eye', 'dog_nose')

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'dog_eyes_nose/train.json',
        img_prefix=data_root + 'dog_eyes_nose/images',
        classes=classes, pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'dog_eyes_nose/val.json',
        img_prefix=data_root + 'dog_eyes_nose/images',
        classes=classes, pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'dog_eyes_nose/val.json',
        img_prefix=data_root + 'dog_eyes_nose/images',
        classes=classes, pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
