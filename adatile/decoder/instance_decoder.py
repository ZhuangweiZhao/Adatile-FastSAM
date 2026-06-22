'''
Instance Decoder — Light U-Net for Few-Shot Instance Segmentation
==================================================================

Lightweight U-Net style decoder for binary instance mask prediction.
Takes FastSAM P4 features, outputs full-resolution binary mask.

架构 | Architecture (3 upsampling blocks, ~1.0M params):
    P4 [B, 1280, H/16, W/16]
        |
    Proj: 1x1 Conv 1280->256 + BN + ReLU
        |
    Up1:  3x3 Conv 256->128 + BN + ReLU + Upsample(x4)  -> H/4
        |
    Up2:  3x3 Conv 128->64  + BN + ReLU + Upsample(x2)  -> H/2
        |
    Up3:  3x3 Conv 64->32   + BN + ReLU + Upsample(x2)  -> H
        |
    Head: 1x1 Conv 32->1  -> Mask

参数 | Params:
    Proj:  1280*256 + 256*2 = 328K
    Up1:   256*128*9 + 128*3 = 295K
    Up2:   128*64*9 + 64*3 = 74K
    Up3:   64*32*9 + 32*3 = 19K
    Head:  32*1 + 1 = 0.03K
    Total: ~716K

特点 | Features:
    - 1x1 投影层降低通道数，节省参数 (1280->256)
    - 3x3 卷积提供空间推理能力
    - 无上采样转置卷积，全用双线性插值 (稳定、无棋盘效应)
    - BN + ReLU 在每层，训练稳定

对比 LightDecoder | vs LightDecoder:
    - LightDecoder: 1280->64->64->32->32->1 (binary, ~716K)
    - InstanceDecoder: 1280->256->128->64->32->1 (~716K)
    - InstanceDecoder 的早期通道更宽 (256 vs 64)，更适合实例分割
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceDecoder(nn.Module):
    '''
    Light U-Net decoder for few-shot instance segmentation.

    Input:  P4 feature [B, 1280, H/16, W/16]
    Output: binary mask [B, 1, H, W]

    Usage:
        decoder = InstanceDecoder()
        mask = decoder(p4_features)  # [B, 1, H, W]

    For few-shot: freeze FastSAM, train decoder on support set.
    '''

    def __init__(self, in_channels: int = 1280):
        super().__init__()

        # Stage 1: Project from 1280 to 256 (1x1 to save params)
        # 第一步：1x1 投影降维
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # Stage 2: 256 -> 128, Upsample x4 (H/16 -> H/4)
        # 第二层：通道减半，4倍上采样
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Stage 3: 128 -> 64, Upsample x2 (H/4 -> H/2)
        # 第三层：通道减半，2倍上采样
        self.up2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 4: 64 -> 32, Upsample x2 (H/2 -> H)
        # 第四层：通道减半，2倍上采样
        self.up3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Mask head: 32 -> 1
        # 输出头：32通道 -> 1通道二值mask
        self.mask_head = nn.Conv2d(32, 1, kernel_size=1, bias=True)

        self._log_params()

    def _log_params(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'[InstanceDecoder] Total params: {total/1e6:.2f}M | Trainable: {trainable/1e6:.2f}M')

    def forward(
        self,
        p4: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Forward pass.

        Args:
            p4: [B, 1280, H/16, W/16] P4 features from FastSAM

        Returns:
            [B, 1, H, W] binary mask logits (sigmoid for probability)
        '''
        # Project
        x = self.proj(p4)  # [B, 256, H/16, W/16]

        # Up1: conv + upscale x4
        x = self.up1(x)  # [B, 128, H/16, W/16]
        x = F.interpolate(x, scale_factor=4, mode='bilinear',
                         align_corners=False)  # [B, 128, H/4, W/4]

        # Up2: conv + upscale x2
        x = self.up2(x)  # [B, 64, H/4, W/4]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                         align_corners=False)  # [B, 64, H/2, W/2]

        # Up3: conv + upscale x2
        x = self.up3(x)  # [B, 32, H/2, W/2]
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                         align_corners=False)  # [B, 32, H, W]

        # Mask head
        mask = self.mask_head(x)  # [B, 1, H, W]

        return mask

    def predict(self, p4: torch.Tensor) -> torch.Tensor:
        '''
        Predict binary mask (with sigmoid).

        Returns:
            [B, 1, H, W] float mask in [0, 1]
        '''
        logits = self.forward(p4)
        return torch.sigmoid(logits)


def build_instance_decoder(
    in_channels: int = 1280,
    pretrained: str | None = None,
) -> InstanceDecoder:
    '''
    Factory function for InstanceDecoder.

    Args:
        in_channels: number of input channels (1280 for P4)
        pretrained: path to pretrained weights

    Returns:
        InstanceDecoder instance
    '''
    model = InstanceDecoder(in_channels=in_channels)
    if pretrained is not None:
        state = torch.load(pretrained, map_location='cpu', weights_only=True)
        model.load_state_dict(state)
        print(f'[InstanceDecoder] Loaded pretrained weights from {pretrained}')
    return model


if __name__ == '__main__':
    # Quick test | 快速测试
    decoder = InstanceDecoder()
    x = torch.randn(2, 1280, 64, 64)  # P4 from 1024x1024 image
    out = decoder(x)
    print(f'Input:  {x.shape}')
    print(f'Output: {out.shape}')
    pred = decoder.predict(x)
    print(f'Pred range: [{pred.min():.4f}, {pred.max():.4f}]')
