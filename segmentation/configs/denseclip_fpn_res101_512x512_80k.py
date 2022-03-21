
_base_ = [
    '_base_/models/denseclip_r50.py', '_base_/datasets/ade20k_clip.py',
    '_base_/default_runtime.py', '_base_/schedules/schedule_80k.py'
]

model = dict(
    type='DenseCLIP',
    pretrained='pretrained/RN101.pt',
    context_length=5,
    text_head=False,
    text_dim=512,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 23, 3],
        output_dim=512,
        input_resolution=512,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=512,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=512,
        dropout=0.1,
        outdim=512,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048+150],
        out_channels=256,
        num_outs=4),
    decode_head=dict(
        type='FPNHead',
        num_classes=150,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

lr_config = dict(policy='poly', power=0.9, min_lr=1e-6, by_epoch=False,
                warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6)


optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001, 
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))

data = dict(samples_per_gpu=4)

