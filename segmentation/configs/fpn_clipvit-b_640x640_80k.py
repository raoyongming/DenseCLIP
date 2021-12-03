_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k_clip_640.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrained/ViT-B-16.pt',
    backbone=dict(
        type='CLIPVisionTransformer',
        patch_size=16,
        width=768,
        layers=12,
        out_indices=[3, 5, 7, 11],
        input_resolution=640,
        style='pytorch'),
    neck=dict(in_channels=[768, 768, 768, 768]),
    decode_head=dict(channels=256, num_classes=150),
    test_cfg=dict(mode='slide', crop_size=(640, 640), stride=(426, 426)), 
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4)