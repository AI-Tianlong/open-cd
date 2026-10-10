from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.optim import AdamW
from torch.nn.modules.activation import ReLU

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.models.change_detectors.atl_A2Net_encoder_decoder import A2Net_EncoderDecoder

# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.mit import MixVisionTransformer
from opencd.models.backbones.atl_lwganet import LWGANet, LWGANet_L2_1242_e96_k11_RELU

# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck
# Decoder_Head
from mmseg.models.decode_heads.segformer_head import SegformerHead
from opencd.models.decode_heads.atl_A2Net import A2Net_Head
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

checkpoint = 'checkpoints/LWGANet/4chan/lwganet_l2_e296.pth'

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
    type=A2Net_EncoderDecoder,
    data_preprocessor=data_preprocessor,
    # pretrained=checkpoint,
    backbone=dict(
        type=LWGANet, # LWGANet_L2_1242_e96_k11_RELU
        in_chans=4,
        num_classes=num_classes,
        stem_dim=96,
        depths=(1, 4, 4, 2),
        att_kernel=(11, 11, 11, 11),
        norm_layer=norm_cfg,
        act_layer=ReLU,
        drop_path_rate=0.1,
        fork_feat=True,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        ),

    decode_head=dict(
        type=A2Net_Head,
        # NeighborFeatureAggregation([in_channels], channels)
        # TemporalFusionModule(channels, channels)
        in_channels = [96, 96, 192, 384, 768], # 传给 
        in_index=[0, 1, 2, 3, 4],
        channels = 32 * 2,  
        num_classes = num_classes,
        loss_decode=[
                    dict(type=CrossEntropyLoss, 
                         use_sigmoid=False, 
                         loss_weight=1.0, 
                         # 直接使用脚本计算出的精确值，这是最科学的
                         class_weight=[0.0845, 1.0000, 1.9861] 
                    ),
                    dict(type=DiceLoss, 
                         use_sigmoid=True, 
                         # Dice 本身对不平衡有抗性，所以给它更高的 Loss 权重
                         loss_weight=3.0,
                         loss_name='loss_dice'
                    )
                    ],
        ),

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
   )

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
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=2000, save_best='mIoU', max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
    # visualization=dict(type=CDVisualizationHook, interval=1, 
    #                    img_shape=(1024, 1024, 3)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])
