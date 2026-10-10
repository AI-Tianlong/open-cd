from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import LayerNorm as LN
from torch.optim import AdamW

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.models.change_detectors.changedino_siamencoder_decoder import ChangeDINO_SiamEncoderDecoder
from opencd.models.change_detectors.atl_coastcdnet_siamencoder_decoder import CoastCDNet_SiamEncoderDecoder
from mmseg.models.backbones.mobilenet_v2 import MobileNetV2


# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
from opencd.models.backbones.changedino_encoder import ChangeDINO_Encoder
# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck

# Decoder_Head
from opencd.models.decode_heads.bit_head import BITHead
from opencd.models.decode_heads.coastcd_head_edge_fre_ndwi import CoastCD_Head
from mmseg.models.decode_heads.fcn_head import FCNHead
from opencd.models.decode_heads.changedino_head import ChangeDINO_Decoder
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.dice_loss import DiceLoss
from mmseg.models.losses.focal_loss import FocalLoss
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
    from .._base_.datasets.coast_cd_train_all import *
    from .._base_.default_runtime import * # 这里会影响是iter输出，还是epoch输出

find_unused_parameters =  True

bit_norm_cfg = dict(type=LN, requires_grad=True)

num_classes = 3 # unchanged water_to_land  land_to_water

# checkpoint = 'open-mmlab://resnet18_v1c'  # noqa
dino_weight = 'checkpoints/changedino/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'
mobilenetv2_checkpoint = 'checkpoints/changedino/4chan/mobilenet_v2-b0353104-4chan.pth'
resnet18_checkpoint = 'checkpoints/resnetv1c/4chan/resnet18_v1c-4chan.pth'

crop_size = (512, 512)
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
    type=ChangeDINO_SiamEncoderDecoder,
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type=ChangeDINO_Encoder,
        in_channels=4,
        backbone="mobilenetv2",
        backbone_pretrained = mobilenetv2_checkpoint,
        channels = [16, 24, 32, 96, 320],
        fpn_channels=128,
        deform_groups=4,
        gamma_mode="SE",
        beta_mode="contextgatedconv",
        
        dino_weight=dino_weight,
        extract_ids=[5, 11, 17, 23],
        ),

    neck=dict(  
        type=FeatureFusionNeck, 
        policy='abs_diff', # 绝对值差后，作为decoder的输入
        ),  # 直接拼接
    
    decode_head=dict(
        type=ChangeDINO_Decoder,
        num_classes=num_classes,
        # in_channels=[128, 128, 128, 128],
        in_channels=128,
        channels = 128, # fpn_channels
        n_layers=[1, 1, 1, 1],
        loss_decode=dict(
            type=CrossEntropyLoss, 
            loss_name='loss_ce_weights',
            use_sigmoid=False,
            loss_weight=1.0,
            class_weight=[0.0658, 1.0000, 1.0000])),
    #     loss_decode=[
    #                 dict(type=FocalLoss, 
    #                      use_sigmoid=False, 
    #                      gamma=2.0, # 2.0
    #                      alpha=0.55, # 0.5
    #                      loss_weight=1.0, 
    #                      class_weight=[0.0658, 1.0000, 1.0000] 
    #                 ),
    #                 dict(type=DiceLoss, 
    #                      use_sigmoid=True, 
    #                      loss_weight=0.5,
    #                      loss_name='loss_dice'
    #                 )
    #                 ],
    # ),
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
    optimizer=optimizer)

# learning policy
param_scheduler = [
    dict(
        type=LinearLR, start_factor=1e-6, by_epoch=False, begin=0, end=200),
    dict(
        type=PolyLR,
        power=1.0,
        begin=1000,
        end=80000,
        eta_min=0.0,
        by_epoch=False,
    )
]


# training schedule for 40k
train_cfg = dict(type=IterBasedTrainLoop, max_iters=80000, val_interval=8000)
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