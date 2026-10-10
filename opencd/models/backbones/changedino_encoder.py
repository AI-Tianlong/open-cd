import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .changedino_modules.blocks.fpn import FPN, DsBnRelu
from .changedino_modules.blocks.cbam import CBAM
from .changedino_modules.blocks.adapter import DINOV3Wrapper, DenseAdapterLite
from .changedino_modules.blocks.diffatts import TransformerBlock
from .changedino_modules.blocks.refine import LearnableSoftMorph
from .changedino_modules.backbone.mobilenetv2 import mobilenet_v2

from mmengine.model import BaseModule, ModuleList, Sequential

# 这里写成传参的。然后通过MODELS.build来构建。
def get_backbone(backbone_name, 
                 in_channels=4,
                 backbone_pretrained=None
                 ):
    if backbone_name == "mobilenetv2":
        backbone = mobilenet_v2(in_channels=in_channels,
                                checkpoints=backbone_pretrained,
                                pretrained=True, 
                                progress=True)
        backbone.channels = [16, 24, 32, 96, 320]

    elif backbone_name == "resnet18d":
        backbone = timm.create_model("resnet18d", pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


class PyramidFeatureFusion(nn.Module):
    def __init__(
        self,
        in_dims=[128, 128, 128, 128],
        dense_dim=1024,
        patch_size=16,
        hidden_dim=256,
    ):
        super().__init__()
        self.in_dims = in_dims
        self.dense_dim = dense_dim
        self.hidden_dim = hidden_dim
        self.patch_size = patch_size

        self.c4 = nn.Sequential(
            DsBnRelu(in_dims[3] + hidden_dim, in_dims[3]), 
            CBAM(in_dims[3], 8)
        )
        self.c3 = nn.Sequential(
            DsBnRelu(in_dims[2] + hidden_dim, in_dims[2]), 
            CBAM(in_dims[2], 8)
        )
        self.c2 = nn.Sequential(
            DsBnRelu(in_dims[1] + hidden_dim, in_dims[1]),
            CBAM(in_dims[1], 8)
        )
        self.c1 = nn.Sequential(
            DsBnRelu(in_dims[0] + hidden_dim, in_dims[0]), 
            CBAM(in_dims[0], 8)
        )

    def forward(self, feas, ds_feas):
        # process backbone (CNN) features
        x1, x2, x3, x4 = (
            feas  # [B, 128, 64, 64], [B, 128, 32, 32], [B, 128, 16, 16], [B, 128, 8, 8]
        )
        a1, a2, a3, a4 = (
            ds_feas  # [B, 256, 64, 64], [B, 256, 32, 32], [B, 256, 16, 16], [B, 256, 8, 8]
        )

        x4 = torch.cat([x4, a4], 1)
        x4 = self.c4(x4)

        x3 = torch.cat([x3, a3], 1)
        x3 = self.c3(x3)

        x2 = torch.cat([x2, a2], 1)
        x2 = self.c2(x2)

        x1 = torch.cat([x1, a1], 1)
        x1 = self.c1(x1)

        return x1, x2, x3, x4

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

# class ChangeDINO_Encoder(nn.Module):
class ChangeDINO_Encoder(BaseModule):
    def __init__(self,
                 in_channels=4,
                 backbone="mobilenetv2",
                 backbone_pretrained=None,

                 fpn_channels=128,
                 deform_groups=4,
                 gamma_mode="SE",
                 beta_mode="contextgatedconv",

                 dino_weight=None,
                 device="cuda",
                 extract_ids=[5, 11, 17, 23],
                 **kwargs):
        super().__init__()
        
        self.in_channels = in_channels
        self.backbone_name = backbone
        self.backbone = get_backbone(backbone_name=backbone, 
                                     in_channels=4,
                                     backbone_pretrained=backbone_pretrained)
        

        self.fpn = FPN(
            in_channels=self.backbone.channels[-4:],
            out_channels=fpn_channels,
            deform_groups=deform_groups,
            gamma_mode=gamma_mode,
            beta_mode=beta_mode,
        )
        dense_out_dim = fpn_channels * 2
        
        self.dino = DINOV3Wrapper(  # DINOv3-这里，只用RGB作为输入。
            weights_path=dino_weight, 
            device=device, 
            extract_ids=extract_ids
        )

        self.dense_adp = DenseAdapterLite(
            in_dim=1024, 
            out_dim=dense_out_dim, 
            bottleneck=fpn_channels // 2
        )
        self.pff = PyramidFeatureFusion( # DFFM
            in_dims=[fpn_channels] * 4,
            dense_dim=1024,
            patch_size=self.dino.patch_size,
            hidden_dim=dense_out_dim,
        )


    def forward(self, x):
        """
        x1: [B, 3, H, W]
        x2: [B, 3, H, W]
        return: [B, 1, H, W]
        """
        # [B,4,512,512]-->[B,16,256,256][4,24,128,128][4,32,64,64][4,96,32,32]
        fea = self.backbone.forward(x) # mobilenet的输出: 
        
        # [B,128,128,128]][B,128,64,64][B,128,32,32][B,128,16,16]
        fea = self.fpn(fea[-4:])  # t1_p1, t1_p2, t1_p3, t1_p4
        
        # 只取RGB三通道
        x_dino = x[:, :3, :, :] if self.in_channels > 3 else x
        x_dino = x_dino.to(x.device)
        ds_fea = self.dino(x_dino) # [B,3,512,512]-->[B,1024,64,64][B,1024,64,64][B,1024,64,64][B,1024,64,64]

        # process dense features
        # [B,256,128,128]][B,256,64,64][B,256,32,32][B,256,16,16]
        ds_fea = self.dense_adp(ds_fea)

        # [B,128,128,128]][B,128,64,64][B,128,32,32][B,128,16,16]
        fea = self.pff(fea, ds_fea)    # 过DFFM

        return fea