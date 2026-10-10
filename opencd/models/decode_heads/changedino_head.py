import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from mmseg.models.decode_heads.decode_head import BaseDecodeHead

from opencd.models.backbones.changedino_modules.blocks.fpn import FPN, DsBnRelu
from opencd.models.backbones.changedino_modules.blocks.cbam import CBAM
from opencd.models.backbones.changedino_modules.blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from opencd.models.backbones.changedino_modules.blocks.diffatts import TransformerBlock
from opencd.models.backbones.changedino_modules.blocks.refine import LearnableSoftMorph
from opencd.models.backbones.changedino_modules.backbone.mobilenetv2 import mobilenet_v2

# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList

from mmseg.models.losses import accuracy
from mmseg.models.utils import resize



class FuseGated(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(2 * dim, dim, 1, bias=True), nn.Sigmoid())
        self.mix = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU(inplace=True),
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        g = self.gate(torch.cat([x1, x2], dim=1))
        fused = x2 + g * x1
        return self.mix(fused)



class ChangeDINO_Decoder(BaseDecodeHead):
# class ChangeDINO_Decoder(nn.Module):
    def __init__(self,
                #  in_channels=[128,128,128,128],
                #  channels=128,
                 fpn_channels=128,

                 n_layers=[1, 1, 1, 1],
                 **kwargs,
    ):
        
        super().__init__(**kwargs)
        # self.num_classes = num_classes

        self.refiner = LearnableSoftMorph(3, 5)

        self.p5_to_p4 = FuseGated(fpn_channels)
        self.p4_to_p3 = FuseGated(fpn_channels)
        self.p3_to_p2 = FuseGated(fpn_channels)

        self.tb5 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[0])
            ]
        )
        self.tb4 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="CDA",
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=3,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[1])
            ]
        )
        self.tb3 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=2,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[2])
            ]
        )
        self.tb2 = nn.Sequential(
            *[
                TransformerBlock(
                    dim=fpn_channels,
                    spatial_attn_type="OCDA",
                    window_size=8,
                    overlap_ratio=0.5,
                    num_channel_heads=8,
                    num_spatial_heads=4,
                    depth=1,
                    ffn_expansion_factor=2,
                    bias=False,
                    LayerNorm_type="BiasFree",
                )
                for _ in range(n_layers[3])
            ]
        )
        self.p5_head = nn.Conv2d(fpn_channels, self.num_classes, 1)
        self.p4_head = nn.Conv2d(fpn_channels, self.num_classes, 1)
        self.p3_head = nn.Conv2d(fpn_channels, self.num_classes, 1)
        self.p2_head = nn.Conv2d(fpn_channels, self.num_classes, 1)
    
    # size，变成256？ 我觉得 应该512啊？
    # def forward(self, x1s, x2s, size=(256, 256)):
    def forward(self, inputs):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        # inputs = self._transform_inputs(inputs)
        ### Extract backbone features
        # size = (128,128)

        # t1_p2, t1_p3, t1_p4, t1_p5 = x1s
        # t2_p2, t2_p3, t2_p4, t2_p5 = x2s

        # diff_p2 = torch.abs(t1_p2 - t2_p2) # [4,128,128,128]
        # diff_p3 = torch.abs(t1_p3 - t2_p3) # [4,128,64,64]
        # diff_p4 = torch.abs(t1_p4 - t2_p4) # [4,128,32,32]
        # diff_p5 = torch.abs(t1_p5 - t2_p5) # [4,128,16,16]

        diff_p2, diff_p3, diff_p4, diff_p5 = inputs

        fea_p5 = self.tb5(diff_p5)     # [4,128,16,16]
        pred_p5 = self.p5_head(fea_p5) # [4,2,16,16]
        
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4) 
        fea_p4 = self.tb4(fea_p4)      # [4,128,32,32]
        pred_p4 = self.p4_head(fea_p4) # [4,2,32,32]
        
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        fea_p3 = self.tb3(fea_p3)      # [4,128,64,64]
        pred_p3 = self.p3_head(fea_p3) # [4,2,64,64]
        
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        fea_p2 = self.tb2(fea_p2)      # [4,128,128,128]
        pred_p2 = self.p2_head(fea_p2) # [4,2,128,128] # 不对，应该是3

        # pred_p2 = F.interpolate(pred_p2, size=size, mode="bilinear", align_corners=False)
        # pred_p3 = F.interpolate(pred_p3, size=size, mode="bilinear", align_corners=False)
        # pred_p4 = F.interpolate(pred_p4, size=size, mode="bilinear", align_corners=False)
        # pred_p5 = F.interpolate(pred_p5, size=size, mode="bilinear", align_corners=False)

        final_pred = self.refiner(pred_p2)

        return final_pred, (pred_p2, pred_p3, pred_p4, pred_p5)

    def loss(self, inputs: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Forward function for training.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_data_samples (list[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `img_metas` or `gt_semantic_seg`.
            train_cfg (dict): The training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        # forward returns (final_pred, aux_preds)
        seg_logits, aux_preds = self.forward(inputs)

        losses = self.loss_by_feat(seg_logits, aux_preds, batch_data_samples)
        return losses

    def loss_by_feat(self, seg_logits: Tensor,
                     aux_preds: Tuple[Tensor],
                     batch_data_samples: SampleList) -> dict:
        """Compute segmentation loss.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_data_samples (List[:obj:`SegDataSample`]): The seg
                data samples. It usually includes information such
                as `metainfo` and `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        seg_label = self._stack_batch_gt(batch_data_samples)
        loss = dict()
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None
        seg_label = seg_label.squeeze(1)

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode
        
        # 1. Loss for the final prediction
        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                loss[loss_decode.loss_name] = loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
            else:
                loss[loss_decode.loss_name] += loss_decode(
                    seg_logits,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
        
        # 2. Loss for auxiliary predictions (Deep Supervision)
        for i, aux_p in enumerate(aux_preds):
            # resize aux prediction to match label size
            aux_p = resize(
                input=aux_p,
                size=seg_label.shape[1:], # [H, W]
                mode='bilinear',
                align_corners=self.align_corners)
            
            for loss_decode in losses_decode:
                # Use a specific name for each aux loss or just aggregate
                # Here we aggregate into the same loss keys but scale by 0.5 as in original code
                aux_loss_value = 0.5 * loss_decode(
                    aux_p,
                    seg_label,
                    weight=seg_weight,
                    ignore_index=self.ignore_index)
                
                loss[loss_decode.loss_name] += aux_loss_value

        loss['acc_seg'] = accuracy(
            seg_logits, seg_label, ignore_index=self.ignore_index)
        return loss

    def predict_by_feat(self, seg_logits_tuple: Tuple[Tensor],
                                batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits_tuple (Tuple[Tensor]): The output from decode head forward function.
                It is a tuple (final_pred, aux_preds), we only need final_pred.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # Unpack the tuple to get the main prediction
        seg_logits = seg_logits_tuple[0]

        if isinstance(batch_img_metas[0]['img_shape'], torch.Size):
            # slide inference
            size = batch_img_metas[0]['img_shape']
        elif 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape'][:2]
        else:
            size = batch_img_metas[0]['img_shape']

        seg_logits = resize(
            input=seg_logits,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners)
        return seg_logits
