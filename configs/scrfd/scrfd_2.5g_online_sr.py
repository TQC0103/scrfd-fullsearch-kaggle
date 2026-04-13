_base_ = ['./scrfd_2.5g.py']

sr_state_file = 'sr_scheduler_state.json'
sr_crop_choices = [0.3, 0.45, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_keypoints=True),
    dict(
        type='RandomSquareCrop',
        crop_choice=sr_crop_choices,
        crop_choice_weights=[1.0] * len(sr_crop_choices),
        scheduler_state_file=sr_state_file,
        scheduler_reload_interval=16,
        bbox_clip_border=False),
    dict(
        type='Resize',
        img_scale=(640, 640),
        keep_ratio=False,
        bbox_clip_border=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='PhotoMetricDistortion',
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18),
    dict(
        type='Normalize',
        mean=[127.5, 127.5, 127.5],
        std=[128.0, 128.0, 128.0],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore',
            'gt_keypointss'
        ])
]

data = dict(train=dict(pipeline=train_pipeline))

custom_hooks = [
    dict(
        type='OnlineSRSchedulerHook',
        state_file=sr_state_file,
        target_strides=(8, 16, 32),
        target_positive_ratios=(0.5, 0.3, 0.2),
        loss_weight=0.65,
        deficit_weight=0.35,
        update_momentum=0.6,
        temperature=0.8,
        min_crop_prob=0.03,
        priority='LOW')
]
