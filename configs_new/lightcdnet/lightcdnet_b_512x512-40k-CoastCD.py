
from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import LayerNorm as LN
from torch.optim import AdamW

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.models.change_detectors.dual_input_encoder_decoder import DIEncoderDecoder
# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
from opencd.models.backbones.lightcdnet import LightCDNet
# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck
from opencd.models.necks.tiny_fpn import TinyFPN
# Decoder_Head
from opencd.models.decode_heads.bit_head import BITHead
from opencd.models.decode_heads.ds_fpn_head import DS_FPNHead
from mmseg.models.decode_heads.fcn_head import FCNHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.dice_loss import DiceLoss
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

# # checkpoint = 'open-mmlab://resnet18_v1c'  # noqa
# checkpoint = 'checkpoints/resnetv1c/4chan/resnet18_v1c-4chan.pth'

crop_size = (512,512)
norm_cfg = dict(type=SyncBN, requires_grad=True)
data_preprocessor = dict(
    type=DualInputSegDataPreProcessor,
    mean=[0.0, 0.0, 0.0, 0.0] * 2,
    std=[10000.0, 10000.0, 10000.0, 10000.0] * 2,
    size = crop_size,
    pad_val=0,
    seg_pad_val=255,
    )

model = dict(
    type=DIEncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=LightCDNet,
        in_channels=4,
        stage_repeat_num=[4, 8, 4],
        net_type="base"),

    neck=dict(
        type=TinyFPN,
        exist_early_x=True,
        early_x_for_fpn=True,
        custom_block='conv',
        in_channels=[24, 116, 232, 464],
        out_channels=48,
        num_outs=4),

    decode_head=dict(
        type=DS_FPNHead,
        num_classes=num_classes,
        in_channels=[48, 48, 48, 48],
        in_index=[0, 1, 2, 3],
        channels=48,
        dropout_ratio=0.,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type=CrossEntropyLoss, 
                         use_sigmoid=False, 
                         loss_weight=1.0, 
                         class_weight=[0.0845, 1.0000, 1.9861]),
    ),

    auxiliary_head=dict(
        type=FCNHead,
        in_channels=24,
        in_index=0,
        channels=24,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.,
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type=CrossEntropyLoss, 
                         use_sigmoid=False, 
                         loss_weight=0.4,
                         class_weight=[0.0845, 1.0000, 1.9861]),
    ),

    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

# optimizer
optimizer=dict(
    type=AdamW, 
    lr=0.003,
    betas=(0.9, 0.999), 
    weight_decay=0.05)

optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=optimizer)

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
train_cfg = dict(type=IterBasedTrainLoop, max_iters=40000, val_interval=2000)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=1000, save_best='mIoU', max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
    # visualization=dict(type=CDVisualizationHook, interval=1, 
    #                    img_shape=(1024, 1024, 3)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])

