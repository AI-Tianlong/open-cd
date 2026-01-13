
# checkpoint = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/segformer/mit_b0_20220624-7e0fe6dd.pth'  # noqa

from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.mit import MixVisionTransformer
# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck
# Decoder_Head
from mmseg.models.decode_heads.segformer_head import SegformerHead
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

    from .._base_.datasets.levir_cd import *
    # from ..common.standard_256x256_40k_levircd import *
    # from .._base_.schedules.schedule_40k import *
    # from .._base_.models.changeformer_mit_b0 import *

checkpoint = 'checkpoints/segformer/mit_b0_20220624-7e0fe6dd.pth'

norm_cfg = dict(type=SyncBN, requires_grad=True)
data_preprocessor = dict(
    type=DualInputSegDataPreProcessor,
    mean=[123.675, 116.28, 103.53] * 2,
    std=[58.395, 57.12, 57.375] * 2,
    bgr_to_rgb=True,
    size_divisor=32,
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32))

model = dict(
    type=SiamEncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=MixVisionTransformer,
        in_channels=3,
        embed_dims=32,
        num_stages=4,
        num_layers=[2, 2, 2, 2],
        num_heads=[1, 2, 5, 8],
        patch_sizes=[7, 3, 3, 3],
        sr_ratios=[8, 4, 2, 1],
        out_indices=(0, 1, 2, 3),
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1),
    neck=dict(type=FeatureFusionNeck, policy='concat'),
    
    decode_head=dict(
        type=SegformerHead,
        in_channels=[v * 2 for v in [32, 64, 160, 256]],
        in_index=[0, 1, 2, 3],
        channels=256,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
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

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer,
    paramwise_cfg=dict(
        custom_keys={
            'pos_block': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'head': dict(lr_mult=10.)
        }))

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
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=4000,
                    save_best='mIoU'),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
    # visualization=dict(type=CDVisualizationHook, interval=1, 
    #                    img_shape=(1024, 1024, 3)))


val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])
