_base_ = [
    '_base_/models/fpn_r50.py', '_base_/datasets/ade20k_clip.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    pretrained='pretrained/RN50.pt',
    backbone=dict(
        type='CLIPResNet',
        layers=[3, 4, 23, 3],
        style='pytorch'),
    decode_head=dict(channels=256, num_classes=150),
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)

optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4)