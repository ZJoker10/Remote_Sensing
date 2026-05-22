# phase1_baseline/resnet_unet.py
# Standard U-Net with a ResNet34 encoder (ImageNet pretrained via timm).
# This is your BENCHMARK — all later phases must beat this IoU.
#
# Architecture:
#   Encoder: ResNet34 → 5 feature maps at strides [1, 2, 4, 8, 16]
#   Bottleneck: 512-channel feature map
#   Decoder: 4× bilinear up-sampling + skip connections (classic U-Net)
#   Head: 1×1 conv → single binary logit map

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────────────────────
# Decoder building blocks
# ─────────────────────────────────────────────────────────────────────────────

class DoubleConv(nn.Module):
    """Two consecutive 3×3 conv layers with BN and ReLU (U-Net's basic block)."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    """
    One decoder step:
        1. Bilinear upsample × 2
        2. Concatenate with the skip connection from the encoder
        3. DoubleConv
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)

        # Handle odd spatial dimensions (pad if needed)
        if x.shape != skip.shape:
            x = F.pad(x, [0, skip.shape[3] - x.shape[3],
                          0, skip.shape[2] - x.shape[2]])

        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─────────────────────────────────────────────────────────────────────────────
# Full ResNet U-Net
# ─────────────────────────────────────────────────────────────────────────────

class ResNetUNet(nn.Module):
    """
    U-Net with a ResNet34 backbone encoder.

    ResNet34 feature map channels at each scale:
        layer0 (stride 2):  64   ← initial conv + bn + relu
        layer1 (stride 4):  64   ← after maxpool
        layer2 (stride 8):  128
        layer3 (stride 16): 256
        layer4 (stride 32): 512  ← bottleneck

    Decoder channels: 256 → 128 → 64 → 32 → final
    """

    # ResNet34 skip-connection channel sizes (in reverse order, encoder → decoder)
    SKIP_CHANNELS = [256, 128, 64, 64]   # layer3, layer2, layer1, layer0
    BOTTLENECK_CH = 512

    def __init__(self, num_classes: int = 1, pretrained: bool = True):
        super().__init__()

        # ── Encoder: ResNet34 via timm features_only mode ─────────────────────
        # With features_only=True, timm returns a FeatureListNet.
        # We call it as backbone(x) and it returns a list of 5 feature maps.
        # We store the backbone itself, not its sub-layers.
        self.backbone = timm.create_model(
            "resnet34", pretrained=pretrained, features_only=True,
            out_indices=(0, 1, 2, 3, 4),   # all 5 scales
        )
        # Channel sizes at each scale for ResNet34:
        # idx 0 → 64  (stride 2,  H/2)
        # idx 1 → 64  (stride 4,  H/4)
        # idx 2 → 128 (stride 8,  H/8)
        # idx 3 → 256 (stride 16, H/16)
        # idx 4 → 512 (stride 32, H/32) ← bottleneck

        # ── Bottleneck DoubleConv ─────────────────────────────────────────────
        self.bottleneck = DoubleConv(self.BOTTLENECK_CH, self.BOTTLENECK_CH)

        # ── Decoder (4 up-blocks) ─────────────────────────────────────────────
        # up4: bottleneck (512) + skip from enc3 (256) → 256
        self.up4 = UpBlock(self.BOTTLENECK_CH, self.SKIP_CHANNELS[0], 256)
        # up3: 256 + skip from enc2 (128) → 128
        self.up3 = UpBlock(256, self.SKIP_CHANNELS[1], 128)
        # up2: 128 + skip from enc1 (64) → 64
        self.up2 = UpBlock(128, self.SKIP_CHANNELS[2], 64)
        # up1: 64 + skip from enc0 (64) → 32
        self.up1 = UpBlock(64, self.SKIP_CHANNELS[3], 32)

        # Final upsample back to full resolution + 1×1 classification head
        self.final_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        # ── Encoding path ────────────────────────────────────────────────────
        # backbone returns a list of 5 feature maps in one call
        s0, s1, s2, s3, s4 = self.backbone(x)
        # s0: (B, 64,  H/2,  W/2)
        # s1: (B, 64,  H/4,  W/4)
        # s2: (B, 128, H/8,  W/8)
        # s3: (B, 256, H/16, W/16)
        # s4: (B, 512, H/32, W/32)

        # ── Bottleneck ────────────────────────────────────────────────────────
        b = self.bottleneck(s4)

        # ── Decoding path (with skip connections) ────────────────────────────
        d4 = self.up4(b,  s3)   # (B, 256, H/16, W/16)
        d3 = self.up3(d4, s2)   # (B, 128, H/8,  W/8)
        d2 = self.up2(d3, s1)   # (B, 64,  H/4,  W/4)
        d1 = self.up1(d2, s0)   # (B, 32,  H/2,  W/2)

        # Back to full input resolution
        out = self.final_up(d1)   # (B, 32, H, W)
        out = self.final_conv(out)  # (B, 1,  H, W) — raw logits

        return out


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = ResNetUNet(num_classes=1, pretrained=False)
    model.eval()

    dummy = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")   # Expected: (2, 1, 512, 512)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Params: {total_params:.1f}M")
