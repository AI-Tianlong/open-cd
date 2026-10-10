from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck
# Decoder_Head
from mmseg.models.decode_heads.uper_head import UPerHead
from opencd.models.decode_heads.changerstar_head import ChangeStarHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
# Optimizer
from mmengine.optim.optimizer import OptimWrapper
from mmengine.optim.scheduler.lr_scheduler import LinearLR, PolyLR
# Evaluation
from mmseg.evaluation import IoUMetric

# HOOK
from mmengine.runner.loops import IterBasedTrainLoop, ValLoop, TestLoop
from mmengine.hooks.iter_timer_hook import IterTimerHook
from mmengine.hooks.logger_hook import LoggerHook
from mmengine.hooks.param_scheduler_hook import ParamSchedulerHook
from mmengine.hooks.checkpoint_hook import CheckpointHook
from mmengine.hooks.sampler_seed_hook import DistSamplerSeedHook
from opencd.engine.hooks.visualization_hook import CDVisualizationHook
from mmseg.engine.hooks.visualization_hook import SegVisualizationHook

from mmengine.config import read_base

with read_base():
    from .._base_.datasets.coast_cd import *
    from .._base_.default_runtime import * # 这里会影响是iter输出，还是epoch输出


num_classes = 3 # unchanged water_to_land  land_to_water

# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b1_20220624-02e5a6a1.pth'  # noqa
checkpoint = 'checkpoints/resnetv1c/4chan/resnet18_v1c-4chan.pth'

crop_size = (512,512)
norm_cfg = dict(type=SyncBN, requires_grad=True)
data_preprocessor = dict(
    type=DualInputSegDataPreProcessor,
    mean=[0.0, 0.0, 0.0, 0.0] * 2,
    std=[10000.0, 10000.0, 10000.0, 10000.0] * 2,
    size = crop_size,
    # size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    # test_cfg=dict(size_divisor=32)
    )

model = dict(
    type=SiamEncoderDecoder,
    data_preprocessor=data_preprocessor,
    # pretrained=checkpoint,
    backbone=dict(
        type=ResNetV1c,
        in_channels=4,
        depth=18,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
    ),

    neck=dict(type=FeatureFusionNeck, policy='concat'),
    
    decode_head=dict(
        type=ChangeStarHead,
        inference_mode='t1t2',
        in_channels=[1, 1, 1, 1], # useless, placeholder
        in_index=[0, 1, 2, 3],
        channels=96, # same with inner_channels in changemixin_cfg
        num_classes=num_classes,
        out_channels=num_classes,
        threshold=0.5,
        seg_head_cfg=dict(
            type=UPerHead,
            in_channels=[64, 128, 256, 512],
            in_index=[0, 1, 2, 3],
            pool_scales=(1, 2, 3, 6),
            channels=128,
            norm_cfg=norm_cfg,
            align_corners=False),
        changemixin_cfg=dict(
            in_channels=128 * 2,
            inner_channels=96, # d_c
            num_convs=1, # N
            ),
        loss_decode=dict(
            # type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)),
            type=CrossEntropyLoss, use_sigmoid=False, loss_weight=1.0)), 
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer=dict(
    type=AdamW, 
    lr=0.001,
    betas=(0.9, 0.999), 
    weight_decay=0.05)

optim_wrapper = dict(type=OptimWrapper, optimizer=optimizer)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=1000),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1000,
        end=40000,
        eta_min=0.0,
        by_epoch=False,
    )
]

# training schedule for 40k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=4000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, save_best='mIoU', max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
    # visualization=dict(type=CDVisualizationHook, interval=1, 
    #                    img_shape=(1024, 1024, 3)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])
