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

from mmseg.models.decode_heads.fcn_head import FCNHead


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
# class CoastCD_Head(BaseDecodeHead):
class CoastCD_Head(BaseModule):

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
                 semantic_chage_head=dict(),
                 **kwargs):
        super().__init__(in_channels, channels, **kwargs)

        self.binary_change_head = MODELS.build(binary_change_head)
        self.semantic_chage_head = MODELS.build(semantic_chage_head)

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

        import pdb;pdb.set_trace()

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


        x1_feat, x2_feat = self._forward_feature(inputs)

        



        output = self.cls_seg(output) # 直接变成 [B,3,H,W]
        return output
