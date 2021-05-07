_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn_custom.py',
    '../_base_/datasets/coco_detection_custom.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
work_dir = './work_dirs/plant_mrcnn'