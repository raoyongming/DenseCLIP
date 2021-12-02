_base_ = [
    '_base_/models/retinanet_r50_fpn.py',
    '_base_/datasets/coco_detection_clip.py',
    '_base_/default_runtime.py'
]

model = dict(
    type='DenseCLIP_RetinaNet',
    pretrained='pretrained/RN50.pt',
    context_length=5,
    clip_head=False,
    seg_loss=True,
    backbone=dict(
        type='CLIPResNetWithAttention',
        layers=[3, 4, 6, 3],
        output_dim=1024,
        input_resolution=1344,
        style='pytorch'),
    text_encoder=dict(
        type='CLIPTextContextEncoder',
        context_length=13,
        embed_dim=1024,
        transformer_width=512,
        transformer_heads=8,
        transformer_layers=12,
        style='pytorch'),
    context_decoder=dict(
        type='ContextDecoder',
        transformer_width=256,
        transformer_heads=4,
        transformer_layers=3,
        visual_dim=1024,
        dropout=0.1,
        outdim=1024,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048 + 80],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_input',
        num_outs=5))
# optimizer
optimizer = dict(type='AdamW', lr=0.0001, weight_decay=0.0001,
        paramwise_cfg=dict(custom_keys={'backbone': dict(lr_mult=0.1),
                                        'text_encoder': dict(lr_mult=0.0),
                                        'norm': dict(decay_mult=0.)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
total_epochs = 12
