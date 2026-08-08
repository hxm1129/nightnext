_base_ = [
    '../../_base_/models/mscan.py',
    '../../_base_/datasets/acdcnight5121024.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]
norm_cfg = dict(type='BN', requires_grad=True)
ham_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
find_unused_parameters = True
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        embed_dims=[64, 128, 320, 512],
        depths=[2, 2, 4, 2],
        init_cfg=None),
    decode_head=dict(
        type='LightHamHeadFreqAwareboundryDPCF222',
        in_channels=[64, 128, 320, 512],
        in_index=[0, 1, 2, 3],
        channels=256,
        ham_channels=256,
        ham_kwargs=dict(MD_R=16),
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=ham_norm_cfg,
        align_corners=False,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0),
            dict(
                type='LovaszLoss',
                loss_type='multi_class',
                classes='present',
                per_image=True,
                reduction='mean',
                loss_weight=0.5)
        ],
         boundary_loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_name='loss_boundary',
            loss_weight=0.0),
        boundary_guidance=False,
        dpcf_use_boundary=False,
        boundary_threshold=0.1),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=(1024, 1024), stride=(768, 768)))


load_from = 'pretrained/segnext_small_1024x1024_city_160k.pth'

data = dict(samples_per_gpu=4)
evaluation = dict(interval=8000, metric='mIoU')
checkpoint_config = dict(by_epoch=False, interval=8000)
runner = dict(type='IterBasedRunner', max_iters=80000)

optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
