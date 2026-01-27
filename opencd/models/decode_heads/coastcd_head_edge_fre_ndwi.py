# Copyright (c) Open-CD. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_activation_layer, build_norm_layer
from mmengine.model import ModuleList, Sequential

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.models.utils import Upsample
from opencd.registry import MODELS
from mmengine.model import BaseModule

import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from mmseg.registry import MODELS
from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from mmseg.models.utils import resize
from mmseg.models.losses import accuracy
from torch import Tensor



class FourierUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels * 2)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # import pdb; pdb.set_trace()
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm='ortho')     # complex
        Xr, Xi = X.real, X.imag
        Freq = torch.cat([Xr, Xi], dim=1)        # [B,2C,H,Wf]
        Freq = self.act(self.bn(self.conv(Freq)))
        Xr2, Xi2 = torch.chunk(Freq, 2, dim=1)
        X2 = torch.complex(Xr2, Xi2)
        y = torch.fft.irfft2(X2, s=(H, W), norm='ortho')
        return y

class FrequencyBranch(nn.Module):
    def __init__(self, in_ch, mid_ch=64, local_kernel=3, use_fourier=True):
        super().__init__()
        self.use_fourier = use_fourier
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        if use_fourier:
            self.fu = FourierUnit(mid_ch)
        else:
            self.fu = nn.Sequential(
                nn.Conv2d(mid_ch, mid_ch, 7, padding=3, groups=mid_ch, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.ReLU(inplace=True),
            )
        self.local = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, local_kernel, padding=local_kernel//2, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        # import pdb; pdb.set_trace() #[]
        z = self.proj(x)  # [2,32,128,128] --> [2,16,128,128]
        z = z + self.fu(z) # [2,16,128,128] --> [2,16,128,128]
        z = self.local(z)  # [2,16,128,128] --> [2,16,128,128]
        return z

class SobelGrad(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        return torch.sqrt(gx*gx + gy*gy + 1e-6)

class EdgeBranch(nn.Module):
    def __init__(self, in_ch, mid_ch=64, use_sobel=True):
        super().__init__()
        self.use_sobel = use_sobel
        self.proj = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.reduce = nn.Conv2d(mid_ch, 1, 1)
        self.sobel = SobelGrad()
        self.edge_feat = nn.Sequential(
            nn.Conv2d(1, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.edge_pred = nn.Conv2d(mid_ch, 1, 1)

    def forward(self, x):
        z = self.proj(x)
        s = self.reduce(z)                       # [B,1,H,W]
        g = self.sobel(s) if self.use_sobel else torch.abs(s)
        f = self.edge_feat(g)                    # [B,mid,H,W]
        edge_logit = self.edge_pred(f)           # [B,1,H,W]
        return f, edge_logit  # 可以用来去学loss

class CrossAttention(nn.Module):
    def __init__(self,
                 in_dims,
                 embed_dims,
                 num_heads,
                 dropout_rate=0.,
                 apply_softmax=True):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.scale = in_dims ** -0.5

        self.apply_softmax = apply_softmax

        self.to_q = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_k = nn.Linear(in_dims, embed_dims, bias=False)
        self.to_v = nn.Linear(in_dims, embed_dims, bias=False)

        self.fc_out = nn.Sequential(
            nn.Linear(embed_dims, in_dims),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x, ref):
        b, n = x.shape[:2]
        h = self.num_heads

        q = self.to_q(x)
        k = self.to_k(ref)
        v = self.to_v(ref)

        q = q.reshape((b, n, h, -1)).permute((0, 2, 1, 3))
        k = k.reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))
        v = v.reshape((b, ref.shape[1], h, -1)).permute((0, 2, 1, 3))

        mult = torch.matmul(q, k.transpose(-1,-2)) * self.scale

        if self.apply_softmax:
            mult = F.softmax(mult, dim=-1)

        out = torch.matmul(mult, v)
        out = out.permute((0,2,1,3)).flatten(2)
        return self.fc_out(out)


class FeedForward(nn.Sequential):
    def __init__(self, dim, hidden_dim, dropout_rate=0.):
        super().__init__(
            # TODO:to be more mmlab
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout_rate)
        )


class TransformerEncoder(nn.Module):
    def __init__(self,
                 in_dims,
                 embed_dims,
                 num_heads,
                 drop_rate,
                 norm_cfg,
                 apply_softmax=True):
        super(TransformerEncoder, self).__init__()
        self.attn = CrossAttention(
            in_dims,
            embed_dims,
            num_heads,
            dropout_rate=drop_rate,
            apply_softmax=apply_softmax)
        self.ff = FeedForward(
            in_dims,
            embed_dims,
            drop_rate
        )
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]
    def forward(self, x):
        x_ = self.attn(self.norm1(x),self.norm1(x)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


class TransformerDecoder(nn.Module):
    def __init__(
            self,
            in_dims,
            embed_dims,
            num_heads,
            drop_rate,
            norm_cfg,
            apply_softmax=True
    ):
        super(TransformerDecoder, self).__init__()
        self.attn = CrossAttention(
            in_dims,
            embed_dims,
            num_heads,
            dropout_rate=drop_rate,
            apply_softmax=apply_softmax)
        self.ff = FeedForward(
            in_dims,
            embed_dims,
            drop_rate
        )
        self.norm1 = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm1_ = build_norm_layer(norm_cfg, in_dims)[1]
        self.norm2 = build_norm_layer(norm_cfg, in_dims)[1]

    def forward(self, x, ref):
        x_ = self.attn(self.norm1(x),self.norm1_(ref)) + x
        y = self.ff(self.norm2(x_)) + x_
        return y


@MODELS.register_module()
class CoastCD_Head(BaseDecodeHead):
# class CoastCD_Head(BaseModule):

    """BIT Head + CoastCD_Net

    This head is the improved implementation of'Remote Sensing Image
    Change Detection With Transformers<https://github.com/justchenhao/BIT_CD>'

    Args:
        in_channels (int): Number of input feature channels (from backbone). Default:  512
        channels (int): Number of output channels of pre_process. Default:  32.
        embed_dims (int): Number of expanded channels of Attention block. Default:  64.
        enc_depth (int): Depth of block of transformer encoder. Default:  1.
        enc_with_pos (bool): Using position embedding in transformer encoder.
            Default:  True
        dec_depth (int): Depth of block of transformer decoder. Default:  8.
        num_heads (int): Number of Multi-Head Cross-Attention Head of transformer encoder.
            Default:  8.
        use_tokenizer (bool),Using semantic token. Default:  True
        token_len (int): Number of dims of token. Default:  4.
        pre_upsample (int): Scale factor of upsample of pre_process.
            (default upsample to 64x64)
            Default: 2.
    """

    def __init__(self,
                 in_channels=256,
                 channels=32,
                 embed_dims=64,
                 enc_depth=1,
                 enc_with_pos=True,
                 dec_depth=8,
                 num_heads=8,
                 drop_rate=0.,
                 pool_size=2,
                 pool_mode='max',
                 use_tokenizer=True,
                 token_len=4,
                 pre_upsample=2,
                 upsample_size=4,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True),
                 binary_change_head=dict(),
                 semantic_change_head=dict(),
                 Frequency_Branch=True,
                 Edge_Branch=True,
                 **kwargs):
        super().__init__(in_channels, channels, **kwargs)

        self.binary_change_head = MODELS.build(binary_change_head)
        self.semantic_chage_head = MODELS.build(semantic_change_head)
        
        # import pdb;pdb.set_trace()
        self.binary_ndwi_mask_embedding = nn.Embedding(2, 32)
        self.semantic_ndwi_mask_embedding = nn.Embedding(3, 32)

        self.if_Frequency_Branch = Frequency_Branch
        self.if_Edge_Branch = Edge_Branch

        if self.if_Frequency_Branch:
            self.Frequency_Branch = FrequencyBranch(
                        in_ch=self.channels,   # 128
                        mid_ch=16,
                        use_fourier=True,      # 不想用FFT就改 False
                        local_kernel=3
                    )
        
        if self.if_Edge_Branch:
            self.Edge_Branch = EdgeBranch(
                in_ch=self.channels,
                mid_ch=16,
                use_sobel=True,        # Sobel更稳定
                # with_pred=True         # 输出 edge_logit 用于 loss
            )

        fuse_in = self.channels
        if self.if_Frequency_Branch:
            fuse_in += 16
        if self.if_Edge_Branch:
            fuse_in += 16

        self.Refine_Fuse = nn.Sequential(
            nn.Conv2d(fuse_in, self.channels, 1, bias=False), # 从128+64+64 --> 128
            nn.BatchNorm2d(self.channels),
            nn.ReLU(inplace=True),
        )

        # self.conv_seg = None # 这个也可以去计算一个 loss呀？用来优化模型！！！

        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.embed_dims=embed_dims
        self.use_tokenizer = use_tokenizer
        self.num_heads=num_heads
        if not use_tokenizer:
            # If a tokenzier is not to be used，then downsample the feature maps
            self.pool_size = pool_size
            self.pool_mode = pool_mode
            self.token_len = pool_size * pool_size
        else:
            self.token_len = token_len
            self.conv_att = ConvModule(
                self.channels,
                self.token_len,
                1,
                conv_cfg=self.conv_cfg,
            )

        self.enc_with_pos = enc_with_pos
        if enc_with_pos:
            self.enc_pos_embedding = nn.Parameter(torch.randn(1, self.token_len * 2, self.channels))

        # pre_process to backbone feature
        self.pre_process = Sequential(
            Upsample(scale_factor=pre_upsample, mode='bilinear', align_corners=self.align_corners),
            ConvModule(
                self.in_channels, # [256]
                self.channels,    # [32]
                3,
                padding=1,
                conv_cfg=self.conv_cfg
            )
        )

        # Transformer Encoder
        self.encoder = ModuleList()
        for _ in range(enc_depth):
            block = TransformerEncoder(
                self.channels,
                self.embed_dims,
                self.num_heads,
                drop_rate=drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.encoder.append(block)

        # Transformer Decoder
        self.decoder = ModuleList()
        for _ in range(dec_depth):
            block = TransformerDecoder(
                self.channels,
                self.embed_dims,
                self.num_heads,
                drop_rate=drop_rate,
                norm_cfg=self.norm_cfg,
            )
            self.decoder.append(block)

        self.upsample = Upsample(scale_factor=upsample_size,mode='bilinear',align_corners=self.align_corners)

    # Token
    def _forward_semantic_tokens(self, x): # [8,32,128,128]
        b, c = x.shape[:2]  # 8, 32
        att_map = self.conv_att(x) # [8,4,128,128]
        att_map = att_map.reshape((b, self.token_len, 1, -1)) #[8,4,1,16384]
        att_map = F.softmax(att_map, dim=-1)  # [8,4,1,16384]
        x = x.reshape((b, 1, c, -1))          # [8,1,32,16384]
        tokens = (x * att_map).sum(-1)   
        return tokens

    def _forward_reshaped_tokens(self, x):
        if self.pool_mode == 'max':
            x = F.adaptive_max_pool2d(x, (self.pool_size, self.pool_size))
        elif self.pool_mode == 'avg':
            x = F.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        else:
            x = x
        tokens = x.permute((0, 2, 3, 1)).flatten(1, 2)
        return tokens


    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        inputs = self._transform_inputs(inputs)

        x1, x2 = torch.chunk(inputs, 2, dim=1) # [B,512,64,64]-->[B,256,64,64][B,256,64,64]
        x1 = self.pre_process(x1)              # [B,256,64,64]-->[B,32,128,128]
        x2 = self.pre_process(x2)
        # Tokenization
        if self.use_tokenizer:
            token1 = self._forward_semantic_tokens(x1) # [8,4,32]
            token2 = self._forward_semantic_tokens(x2) # [8,4,32]
        else:
            token1 = self._forward_reshaped_tokens(x1)
            token2 = self._forward_reshaped_tokens(x2)

        # Transformer encoder forward
        token = torch.cat([token1, token2], dim=1) # [8,8,32]
        if self.enc_with_pos:
            token += self.enc_pos_embedding
        for i, _encoder in enumerate(self.encoder):
            token = _encoder(token)
        token1, token2 = torch.chunk(token, 2, dim=1)

        # Transformer decoder forward
        for _decoder in self.decoder:
            b, c, h, w = x1.shape
            x1 = x1.permute((0, 2, 3, 1)).flatten(1, 2)
            x2 = x2.permute((0, 2, 3, 1)).flatten(1, 2)
            
            x1 = _decoder(x1, token1)
            x2 = _decoder(x2, token2)
            
            x1 = x1.transpose(1, 2).reshape((b, c, h, w))
            x2 = x2.transpose(1, 2).reshape((b, c, h, w))

        # # Feature differencing
        # y = torch.abs(x1 - x2)  # 
        # y = self.upsample(y)

        # import pdb; pdb.set_trace()
        # return y

        return x1, x2

    def forward(self, inputs):
        """Forward function."""
        # output = self._forward_feature(inputs)
        # [x, NDWI_tuple] NDWI_tuple = (NDWI_change_mask_2, NDWI_change_mask_3, img_from_NDWI, img_to_NDWI)
        

        if isinstance(inputs, list) and len(inputs)==2 and len(inputs[1])==4:
            inputs_x = inputs[0]
            NDWI_tuple = inputs[1]

            NDWI_change_mask_2, NDWI_change_mask_3, img_from_NDWI, img_to_NDWI = NDWI_tuple
            # [B,1,512,512] 0/1  [B,1,512,512] 0/1/2  [B,1,512,512] [B,1,512,512]
        else:
            raise TypeError(f"请检查inputs！")

        x1_feat, x2_feat = self._forward_feature(inputs_x) # [8,128,128,128][8,128,128,128]

        y = torch.abs(x1_feat - x2_feat) # [8,128,128,128]

        to_fuse = [y]

        if self.if_Frequency_Branch:
            y_fft = self.Frequency_Branch(y) #[2,32,128,128]
            to_fuse.append(y_fft) # [4,64,128,128]
        
        if self.if_Edge_Branch:
            f, edge_logit  = self.Edge_Branch(y)
            to_fuse.append(f)

        y = self.Refine_Fuse(torch.cat(to_fuse, dim=1))  # back to [B,128,h,w]


        # binary_change_head
        binary_change_embedding = self.binary_ndwi_mask_embedding(NDWI_change_mask_2.squeeze(1)).permute(0, 3, 1, 2)     #  [8,1,512,512]->[8,512,512]->[8,512,512,32]->[8,32,512,512]
        binary_change_embedding = F.interpolate(binary_change_embedding, size=y.shape[2:], mode='nearest')
        input_to_binary_head = torch.cat([y, binary_change_embedding], dim=1) # y[8,128,128,128] | [8,160,128,128]
        binary_change_seglogits = self.binary_change_head([input_to_binary_head]) # [B,2,H,W]
        

        # semantic_change_head
        NDWI_change_mask_3 = (NDWI_change_mask_3+ 1).long().clamp(0, 2) # [-1,0,1]->[0,1,2]

        semantic_change_embedding = self.semantic_ndwi_mask_embedding(NDWI_change_mask_3.squeeze(1)).permute(0, 3, 1, 2) #  [8,1,512,512]->[8,512,512]->[8,512,512,32]->[8,32,512,512]
        semantic_change_embedding = F.interpolate(semantic_change_embedding, size=y.shape[2:], mode='nearest')
        
        # 先和 embedding 拼接再gate，还是gate完拼接？,
        input_to_semanitc_head = torch.cat([y, semantic_change_embedding], dim=1)
        # binary gate # 我怎么感觉。这里需要传入一个sigmoid? 

        binary_probs = torch.softmax(binary_change_seglogits, dim=1)
        binary_change_gate = binary_probs[:,1:2,:,:]  # 存疑
        input_to_semanitc_head = input_to_semanitc_head * (1 + binary_change_gate) # 软门控
        

        semantic_change_seglogits = self.semantic_chage_head([input_to_semanitc_head])

        # output = self.cls_seg(output) # 直接变成 [B,3,H,W], 这个也能去参与个loss计算呀！，先不用
        
        # import pdb;pdb.set_trace()
        return [binary_change_seglogits, semantic_change_seglogits, edge_logit]  # [8,2,128,128][8,3,128,128][8,1,128,128]

    def loss(self, inputs: Tuple[Tensor], 
             batch_data_samples: SampleList,
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

        # import pdb; pdb.set_trace()
        # inputs: x:[B,512,64,64] | NDWI_tuple = (NDWI_change_mask_2, NDWI_change_mask_3, img_from_NDWI, img_to_NDWI) 
        seg_logits = self.forward(inputs) # (binary_change_seglogits, semantic_change_seglogits)
        # [8,512,64,64][8,1,512,512]
        losses = self.loss_by_feat(seg_logits, batch_data_samples)
        return losses

    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

        seg_logits = self.forward(inputs) 
        seg_logits = seg_logits[1] # binary semantic

        return self.predict_by_feat(seg_logits, batch_img_metas)
    

    def loss_by_feat(self, 
                     seg_logits: List[Tensor],
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

        seg_label = self._stack_batch_gt(batch_data_samples) # [8,1,512,512] [0,1,2]
        seg_label_binary = (seg_label!=0).long()             # [8,1,512,512] [0,1]

        # import pdb;pdb.set_trace()
        # seg_label_binary = seg_label

        loss = dict()

        # seg_logits (binary_change_seglogits, semantic_change_seglogits)

        for index in range(len(seg_logits)): # [8,2,128,128][8,3,128,128]  edge:[8,1,128,128]

            seg_logits[index] = resize(
                input=seg_logits[index],
                size=seg_label.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)
        
        if self.sampler is not None:
            seg_weight = self.sampler.sample(seg_logits, seg_label)
        else:
            seg_weight = None

        seg_label = seg_label.squeeze(1)
        seg_label_binary = seg_label_binary.squeeze(1)

        

        def extract_edge_from_mask(mask):
            """
            从分割掩膜中提取边缘 GT
            mask: [B, H, W] 或 [B, 1, H, W] (0, 1, 2...)
            return: [B, 1, H, W] 二值边缘图 (0.0 或 1.0)
            """

            threshold = 0.1
        
            # 定义 Sobel 卷积核 (固定权重，不可学习)
            # X 方向提取垂直边缘
            sobel_x = torch.tensor([[-1, 0, 1], 
                                        [-2, 0, 2], 
                                        [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            # Y 方向提取水平边缘
            sobel_y = torch.tensor([[-1, -2, -1], 
                                        [0, 0, 0], 
                                        [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

            # 1. 预处理：转为 float 并增加通道维
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) # [B, 1, H, W]
            mask = mask.float()
            
            # 2. 把 mask 移到和卷积核一样的设备上
            sobel_x = sobel_x.to(mask.device)
            sobel_y = sobel_y.to(mask.device)

            # 3. 卷积操作 (Padding=1 保持尺寸不变)
            edge_x = F.conv2d(mask, sobel_x, padding=1)
            edge_y = F.conv2d(mask, sobel_y, padding=1)

            # 4. 计算梯度幅值: sqrt(x^2 + y^2)
            edge_magnitude = torch.sqrt(edge_x ** 2 + edge_y ** 2)

            # 5. 二值化：只要有梯度就是边缘
            # 这里会生成有点粗的边缘，这对训练由于有容错性，其实是好事
            edge_binary = (edge_magnitude > threshold).float()
            
            return edge_binary 

        # import pdb; pdb.set_trace()
        seg_label_edge = extract_edge_from_mask(seg_label) 
        seg_label_edge = seg_label_edge.squeeze(1)

        # # ================== 插入开始: 保存 TIF 调试 ==================
        # # 1. 取 Batch 中的第一张图 (假设 [B, 1, H, W])
        # # .detach(): 从计算图中分离
        # # .cpu(): 放到内存
        # # .squeeze(): 变成 [H, W] 二维矩阵
        # import cv2
        # import numpy as np
        # edge_vis = seg_label_edge[0].detach().cpu().squeeze().numpy()
        
        # # 2. 映射到 0-255 (否则 0和1 肉眼看不出区别，都是黑的)
        # # 如果是 float (0.0~1.0) -> * 255
        # # 如果是 int (0, 1) -> * 255
        # edge_vis = (edge_vis * 255).astype(np.uint8)
        
        # # 3. 保存文件
        # # 建议加个随机数或者迭代次数，防止瞬间被覆盖，这里为了简单直接覆盖
        # cv2.imwrite('/data/AI-Tianlong/2-part2-coastCD/open-cd/tools/atl_test/debug_edge_check.tif', edge_vis)
        
        # print(f"DEBUG: 边缘图已保存到 debug_edge_check.tif，当前最大值: {edge_vis.max()}")
        # ================== 插入结束 ==================

        # 这里应该用 binary的losscode 和 semantic的loss_decode
        # 
        # if not isinstance(self.loss_decode, nn.ModuleList):
        #     losses_decode = [self.loss_decode]
        # else:
        #     losses_decode = self.loss_decode

        losses_decode = [self.binary_change_head.loss_decode, 
                         self.semantic_chage_head.loss_decode,
                         self.loss_decode] # 约束edge

        for loss_decode in losses_decode:
            if loss_decode.loss_name not in loss:
                if 'binary_head' in loss_decode.loss_name:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits[0],
                        seg_label_binary,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                elif 'semantic_head' in loss_decode.loss_name:
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits[1],
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                
                elif 'edge' in loss_decode.loss_name:
                    # import pdb; pdb.set_trace()
                    loss[loss_decode.loss_name] = loss_decode(
                        seg_logits[2],
                        seg_label_edge,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)

            else:
                if 'binary_head' in loss_decode.loss_name:
                    loss[loss_decode.loss_name] += loss_decode(
                        seg_logits[0],
                        seg_label_binary,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                elif 'semantic_head' in loss_decode.loss_name:
                        loss[loss_decode.loss_name] += loss_decode(
                        seg_logits[1],
                        seg_label,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)
                elif 'edge' in loss_decode.loss_name:
                        loss[loss_decode.loss_name] += loss_decode(
                        seg_logits[2],
                        seg_label_edge,
                        weight=seg_weight,
                        ignore_index=self.ignore_index)


        loss['acc_seg_binary'] = accuracy(seg_logits[0], seg_label_binary, ignore_index=self.ignore_index)

        loss['acc_seg'] = accuracy(seg_logits[1], seg_label, ignore_index=self.ignore_index)
        

        # import pdb; pdb.set_trace()
        return loss

    def predict_by_feat(self, seg_logits: Tensor,
                        batch_img_metas: List[dict]) -> Tensor:
        """Transform a batch of output seg_logits to the input shape.

        Args:
            seg_logits (Tensor): The output from decode head forward function.
            batch_img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            Tensor: Outputs segmentation logits map.
        """

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
