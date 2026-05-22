# phase3_swin_unet/swin_unet_decoder.py
# Swin Transformer U-Net decoder piped to the SatMAE++ encoder.
#
# Architecture overview:
#   SatMAE++ encoder  →  4 feature maps (pyramid from ViT blocks)
#   FeaturePyramidAdapter  →  channel-normalize them
#   Swin decoder stages  →  hierarchical self-attention via shifted windows
#   4× bilinear upsamplings  →  back to full input resolution
#   1×1 conv head  →  binary logit mask
#
# Why Swin in the decoder?
#   Local shifted-window attention lets the decoder reason about building
#   shape coherence (straight edges, rectangular outlines) at each scale,
#   which global attention or plain convolutions handle less efficiently.


import sys
import os
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Add project root to sys.path FIRST
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


# 2. THEN import your custom local modules

from timm.models.swin_transformer import SwinTransformerBlock
from phase2_satmae.satmae_encoder import SatMAEPlusEncoder, FeaturePyramidAdapter
# ─────────────────────────────────────────────────────────────────────────────
# Swin decoder stage
# ─────────────────────────────────────────────────────────────────────────────

class SwinDecoderStage(nn.Module):
    """
    One decoder stage:
        1. Bilinear ×2 upsample of the incoming feature map.
        2. Concatenate with skip connection (from the encoder pyramid).
        3. 1×1 conv to fuse channels.
        4. N × SwinTransformerBlock for shifted-window self-attention.

    Args:
        in_ch:       Channels coming from the previous decoder stage.
        skip_ch:     Channels from the encoder skip connection.
        out_ch:      Output channels for this stage.
        input_resolution: (H, W) of the feature map after upsampling.
        num_heads:   Attention heads for Swin blocks.
        window_size: Swin window size (default 7, or 8 if resolution % 7 != 0).
        depth:       How many Swin blocks to apply per stage.
    """

    def __init__(
        self,
        in_ch: int,
        skip_ch: int,
        out_ch: int,
        input_resolution: tuple,
        num_heads: int,
        window_size: int = 7,
        depth: int = 2,
    ) -> None:
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=False)

        # Fuse upsampled features + skip via 1×1 conv
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

        # Swin Transformer blocks operating on the fused feature map.
        # timm SwinTransformerBlock requires window_size to divide H and W evenly.
        # We use window_size=8 (divides 64/128/256/512 cleanly) and pass
        # input_resolution so timm can pre-compute the attention mask.
        self.swin_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=out_ch,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                dynamic_mask=True,   # recompute mask if resolution changes
            )
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Align spatial dims if encoder crop differs slightly
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)

        x = torch.cat([x, skip], dim=1)     # (B, in_ch+skip_ch, H, W)
        x = self.fuse(x)                     # (B, out_ch, H, W)

        # timm SwinTransformerBlock expects (B, H, W, C) — channels last
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)   # (B, H, W, C)

        for blk in self.swin_blocks:
            x = blk(x)               # in: (B,H,W,C)  out: (B,H,W,C)
        x = self.norm(x)

        # Back to (B, C, H, W) for the rest of the decoder
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Full SatMAE++ + Swin-Unet segmentation model
# ─────────────────────────────────────────────────────────────────────────────

class SatMAESwinUNet(nn.Module):
    """
    Full segmentation model:
        Encoder: SatMAE++ (frozen ViT-Large, multi-scale extraction)
        Adapter: FeaturePyramidAdapter (1024 → [128, 256, 512, 1024])
        Decoder: 4 × SwinDecoderStage (hierarchical shifted-window attention)
        Head:    1×1 conv → binary logit

    For a 512×512 input with patch_size=16:
        ViT grid = 32×32 (all 4 encoder scales are at this resolution)
        Decoder progressively up-samples: 32 → 64 → 128 → 256 → 512
    """

    # Encoder output channels after the pyramid adapter
    # shallow scale → deep scale
    ENC_CHANNELS: list[int] = [128, 256, 512, 1024]

    # Decoder stage output channels (deep → shallow, mirrors U-Net)
    DEC_CHANNELS: list[int] = [512, 256, 128, 64]

    def __init__(
        self,
        img_size: int = 512,
        patch_size: int = 16,
        num_classes: int = 1,
        window_size: int = 7,
        swin_depths: list = [2, 2, 2, 2],   # blocks per decoder stage
        swin_heads: list = [16, 8, 4, 2],   # heads per decoder stage
        satmae_ckpt: str = None,
        freeze_encoder: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size: int = patch_size
        self.grid: int = img_size // patch_size   # 32 for 512px / patch=16

        # ── SatMAE++ encoder ──────────────────────────────────────────────────
        self.encoder = SatMAEPlusEncoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            extract_layers=[5, 11, 17, 23],
            freeze=freeze_encoder,
        )
        if satmae_ckpt:
            self.encoder.load_satmae_checkpoint(satmae_ckpt)

        # ── Pyramid adapter ───────────────────────────────────────────────────
        self.adapter = FeaturePyramidAdapter(1024, self.ENC_CHANNELS)

        # ── Decoder stages (4 stages, going deep→shallow) ─────────────────────
        # Stage 0: bottleneck (deepest enc scale) → first decoder feature map
        # The "in_ch" at stage 0 = deepest encoder scale, no skip yet.
        # We treat the deepest encoder feature as the bottleneck.

        # Stage 1: DEC[0] output + skip from enc scale 2 (ENC[2]=512)
        # Stage 2: DEC[1] output + skip from enc scale 1 (ENC[1]=256)
        # Stage 3: DEC[2] output + skip from enc scale 0 (ENC[0]=128)

        # Resolutions at each decoder stage (starting from grid=32)
        resolutions: list[tuple[int, int]] = [
            (self.grid * 2, self.grid * 2),   # 64×64
            (self.grid * 4, self.grid * 4),   # 128×128
            (self.grid * 8, self.grid * 8),   # 256×256
            (self.grid * 16, self.grid * 16),  # 512×512  ← last stage
        ]

        # window_size=8 divides all decoder resolutions (64,128,256,512) evenly
        _wsize = 8
        self.dec_stages = nn.ModuleList([
            SwinDecoderStage(
                in_ch=self.ENC_CHANNELS[3] if i == 0 else self.DEC_CHANNELS[i - 1],
                skip_ch=self.ENC_CHANNELS[2 - i] if i < 3 else 0,
                out_ch=self.DEC_CHANNELS[i],
                input_resolution=resolutions[i],
                num_heads=swin_heads[i],
                window_size=_wsize,
                depth=swin_depths[i],
            )
            for i in range(4)
        ])

        # ── Segmentation head ─────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(self.DEC_CHANNELS[-1], 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1),  # raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            logits: (B, 1, H, W)
        """
        B, _, H, W = x.shape

        # ── Encode ────────────────────────────────────────────────────────────
        # feats[0] = shallowest, feats[3] = deepest
        feats = self.encoder(x)          # 4 × (B, 1024, grid, grid)
        # 4 × (B, ENC_CHANNELS[i], grid, grid)
        feats = self.adapter(feats)

        # ── Decode ────────────────────────────────────────────────────────────
        # Start from the deepest feature, fusing progressively shallower skips
        d = feats[3]   # bottleneck: (B, 1024, 32, 32)

        for i, stage in enumerate(self.dec_stages):
            skip_idx: int = 2 - i   # 2, 1, 0, then no skip
            skip: Any | torch.Tensor = feats[skip_idx] if skip_idx >= 0 else torch.zeros_like(d)
            # Replace last stage skip (skip_idx = -1) with zeros if no skip
            if skip_idx < 0:
                skip: torch.Tensor = torch.zeros(B, 0, d.shape[2] * 2, d.shape[3] * 2,
                                   device=d.device)
            d = stage(d, skip)

        # ── Final up-sample to input resolution if needed ─────────────────────
        if d.shape[2:] != (H, W):
            d = F.interpolate(d, size=(H, W), mode="bilinear",
                              align_corners=False)

        logits = self.head(d)   # (B, 1, H, W)
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# Differential learning rates: encoder vs decoder
# ─────────────────────────────────────────────────────────────────────────────

def get_param_groups(model: SatMAESwinUNet, base_lr: float, encoder_lr_scale: float):
    """
    If fine-tuning the encoder, use a much smaller LR for encoder parameters
    to avoid catastrophic forgetting of pre-trained representations.

    Returns param groups compatible with torch.optim optimizers.
    """
    encoder_params: list[nn.Parameter] = list(model.encoder.parameters())
    encoder_ids: set[int] = set(id(p) for p in encoder_params)

    other_params: list[nn.Parameter] = [p for p in model.parameters() if id(p) not in encoder_ids]

    return [
        {"params": other_params,   "lr": base_lr},
        {"params": encoder_params, "lr": base_lr * encoder_lr_scale},
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = SatMAESwinUNet(
        img_size=512, patch_size=16,
        freeze_encoder=True,
        satmae_ckpt=None,   # Skip loading for the test
    )
    model.eval()

    dummy: torch.Tensor = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        out = model(dummy)

    print(f"Input:  {dummy.shape}")
    print(f"Output: {out.shape}")    # Expected: (1, 1, 512, 512)

    total: float = sum(p.numel() for p in model.parameters()) / 1e6
    trainable: float = sum(p.numel()
                    for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Total params:     {total:.1f}M")
    print(f"Trainable params: {trainable:.1f}M")
