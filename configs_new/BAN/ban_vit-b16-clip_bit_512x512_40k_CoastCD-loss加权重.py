from torch.nn.modules.batchnorm import SyncBatchNorm as SyncBN
from torch.nn.modules.normalization import LayerNorm as LN
from torch.optim import AdamW

# Encoder_Decoder
from opencd.models.change_detectors.siamencoder_decoder import SiamEncoderDecoder
from opencd.models.change_detectors.ban import BAN
# DataPreProcessor
from opencd.models.data_preprocessor import DualInputSegDataPreProcessor
# Backbone
from mmseg.models.backbones.resnet import ResNetV1c
from mmseg.models.backbones.vit import VisionTransformer
from mmseg.models.text_encoder.clip_text_encoder import QuickGELU

# Neck
from opencd.models.necks.feature_fusion import FeatureFusionNeck
# Decoder_Head
from opencd.models.decode_heads.bit_head import BITHead
from opencd.models.decode_heads.ban_head import BitemporalAdapterHead
from opencd.models.decode_heads.ban_utils import BAN_BITHead
# Loss
from mmseg.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmseg.models.losses.dice_loss import DiceLoss
# Optimizer
from mmengine.optim.optimizer import AmpOptimWrapper
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
crop_size = (512,512)
norm_cfg = dict(type=SyncBN, requires_grad=True)

data_preprocessor = dict(
    type=DualInputSegDataPreProcessor,
    mean=[0.0, 0.0, 0.0, 0.0] * 2,
    std=[10000.0, 10000.0, 10000.0, 10000.0] * 2,
    # size = crop_size,
    size_divisor=32, 
    pad_val=0,
    seg_pad_val=255,
    test_cfg=dict(size_divisor=32)
    )

checkpoint_r18 = 'checkpoints/resnetv1c/4chan/resnet18_v1c-4chan.pth'
clip_vit = 'checkpoints/BAN/clip_vit-base-patch16-224_3rdparty-d08f8887.pth'

model = dict(
    type=BAN,
        data_preprocessor=data_preprocessor,
        pretrained=clip_vit,
        asymetric_input=True,

    encoder_resolution=dict(
        size=(224, 224),
        mode='bilinear'),

    image_encoder=dict(
        type=VisionTransformer,
        img_size=(224, 224),
        patch_size=16,
        patch_pad=0,
        in_channels=4,
        embed_dims=768,
        num_layers=9,
        num_heads=12,
        mlp_ratio=4,
        out_origin=False,
        out_indices=(2, 5, 8),
        qkv_bias=True,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        with_cls_token=True,
        output_cls_token=True,
        patch_bias=False,
        pre_norm=True,
        norm_cfg=dict(type=LN, eps=1e-5),
        act_cfg=dict(type=QuickGELU),
        norm_eval=False,
        interpolate_mode='bicubic',
        # frozen_exclude=['pos_embed'],
        frozen_exclude=[]
        ),

    decode_head=dict(
        type=BitemporalAdapterHead,
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

        ban_cfg=dict(
            clip_channels=768,
            fusion_index=[0, 1, 2],
            side_enc_cfg=dict(
                type='mmseg.ResNetV1c',
                init_cfg=dict(type='Pretrained', checkpoint=checkpoint_r18),
                in_channels=4,
                depth=18,
                num_stages=3,
                out_indices=(2,),
                dilations=(1, 1, 1),
                strides=(1, 2, 1),
                norm_cfg=dict(type=SyncBN, requires_grad=True),
                norm_eval=False,
                style='pytorch',
                contract_dilation=True)),
        ban_dec_cfg=dict(
            type=BAN_BITHead,
            in_channels=256,
            channels=32,
            num_classes=num_classes)),
            
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)))

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=AdamW, lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys={
            'img_encoder': dict(lr_mult=0.1, decay_mult=1.0),
            'norm': dict(decay_mult=0.),
            'mask_decoder': dict(lr_mult=10.)
        }),
    loss_scale='dynamic',
    clip_grad=dict(max_norm=0.01, norm_type=2))


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
    checkpoint=dict(type=CheckpointHook, by_epoch=False, interval=1000, save_best='mIoU', max_keep_ckpts=4),
    sampler_seed=dict(type=DistSamplerSeedHook),
    visualization=dict(type=SegVisualizationHook))
    # visualization=dict(type=CDVisualizationHook, interval=1, 
    #                    img_shape=(1024, 1024, 3)))

val_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])  # 'mDice', 'mFscore'
test_evaluator = dict(
    type=IoUMetric, iou_metrics=['mIoU', 'mFscore'])
