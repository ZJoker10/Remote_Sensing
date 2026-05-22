# phase2_satmae/satmae_encoder.py
# SatMAE++ as a frozen (or low-LR) multi-scale feature extractor.
#
# SatMAE++ is a ViT-Large masked autoencoder pre-trained on multi-resolution
# satellite imagery. We discard the MAE decoder and keep only the encoder.
#
# Key challenge: ViT produces flat patch tokens, not spatial feature maps.
# We must un-flatten them back into (B, C, H/patch, W/patch) feature maps
# to be compatible with a CNN-style decoder.
#
# Multi-scale output strategy:
#   We extract intermediate ViT block outputs at several depths to build a
#   feature pyramid (similar to how ViTDet / DINOv2 do it).

import math
import torch
import torch.nn as nn
from functools import partial


# ─────────────────────────────────────────────────────────────────────────────
# Minimal ViT implementation compatible with SatMAE++ checkpoint format
# ─────────────────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Split image into non-overlapping patches and project them."""

    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=1024):
        super().__init__()
        self.img_size   = img_size
        self.patch_size = patch_size
        self.n_patches  = (img_size // patch_size) ** 2
        # Linear projection via a strided convolution
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, 3, H, W)
        x = self.proj(x)             # (B, embed_dim, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N_patches, embed_dim)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=16, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, drop=0.):
        super().__init__()
        hidden = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = Attention(dim, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp   = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
# SatMAE++ Encoder Wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SatMAEPlusEncoder(nn.Module):
    """
    SatMAE++ encoder used as a multi-scale feature extractor.

    Outputs a list of spatial feature maps at 4 different depths,
    forming a coarse feature pyramid:
        scale_0 (deepest  ): stride 16 — from block depth[3] (e.g. block 23)
        scale_1             : stride 16 — from block depth[2] (e.g. block 17)
        scale_2             : stride 16 — from block depth[1] (e.g. block 11)
        scale_3 (shallowest): stride 16 — from block depth[0] (e.g. block 5)

    Note: ViT doesn't have multiple spatial resolutions like a CNN. All scales
    share the same spatial grid (H/patch × W/patch). The variation is in
    *semantic depth*, not spatial resolution. The decoder handles up-sampling.

    Args:
        img_size:      Input image resolution (must be divisible by patch_size).
        patch_size:    Patch size used in pre-training (16 for SatMAE++).
        embed_dim:     Token dimension (1024 for ViT-Large).
        depth:         Total number of transformer blocks (24 for ViT-Large).
        num_heads:     Number of attention heads (16 for ViT-Large).
        extract_layers: Block indices from which to extract feature maps.
        freeze:        If True, disables all gradients (fully frozen encoder).
    """

    def __init__(
        self,
        img_size: int    = 512,
        patch_size: int  = 16,
        embed_dim: int   = 1024,
        depth: int       = 24,
        num_heads: int   = 16,
        extract_layers: list = [5, 11, 17, 23],  # shallow → deep
        freeze: bool     = True,
    ):
        super().__init__()
        self.patch_size     = patch_size
        self.embed_dim      = embed_dim
        self.extract_layers = extract_layers

        # ── Patch embedding ──────────────────────────────────────────────────
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        n_patches = self.patch_embed.n_patches

        # ── Positional embedding (learned, same shape as SatMAE++) ───────────
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))

        # ── Transformer blocks ───────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio=4.0)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # ── Freeze if requested ──────────────────────────────────────────────
        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    @property
    def grid_size(self):
        """Spatial grid size (tokens per side) given the current patch_size."""
        # This is set dynamically after load based on pos_embed shape
        n = self.pos_embed.shape[1] - 1   # subtract CLS token
        return int(math.sqrt(n))

    def load_satmae_checkpoint(self, ckpt_path: str):
        """
        Load official SatMAE++ weights.
        The checkpoint has keys like:
            model.patch_embed.proj.weight
            model.cls_token
            model.pos_embed
            model.blocks.N.norm1.weight  ...

        We strip the 'model.' prefix and discard the decoder weights.
        """
        print(f"[SatMAE++] Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # SatMAE++ checkpoints are typically stored under 'model' key
        state_dict = ckpt.get("model", ckpt)

        # Strip 'model.' prefix if present
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                k = k[len("model."):]
            # Skip decoder weights (not needed for feature extraction)
            if k.startswith("decoder"):
                continue
            cleaned[k] = v
# Check for pos_embed size mismatch and remove it if necessary
        if 'pos_embed' in cleaned:
            ckpt_shape = cleaned['pos_embed'].shape
            model_shape = self.state_dict()['pos_embed'].shape
            
            if ckpt_shape != model_shape:
                print(f"Skipping pos_embed: Checkpoint shape {ckpt_shape} does not match model shape {model_shape}.")
                del cleaned['pos_embed']


        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        print(f"[SatMAE++] Missing keys:    {len(missing)}")
        print(f"[SatMAE++] Unexpected keys: {len(unexpected)}")
        print("[SatMAE++] Encoder loaded successfully.")

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, 3, H, W) input satellite image

        Returns:
            features: list of 4 tensors, each (B, embed_dim, grid, grid)
                      ordered shallow → deep (smallest semantic level first)
        """
        B = x.shape[0]
        grid = self.grid_size

        # ── Tokenize ─────────────────────────────────────────────────────────
        tokens = self.patch_embed(x)                     # (B, N, D)
        cls = self.cls_token.expand(B, -1, -1)           # (B, 1, D)
        tokens = torch.cat([cls, tokens], dim=1)         # (B, N+1, D)
        tokens = tokens + self.pos_embed                  # add positional info

        # ── Run blocks, extracting at specified layers ────────────────────────
        features = []
        for i, block in enumerate(self.blocks):
            tokens = block(tokens)
            if i in self.extract_layers:
                # Drop CLS token and reshape flat tokens → 2D spatial map
                patch_tokens = tokens[:, 1:, :]          # (B, N, D)
                spatial = patch_tokens.transpose(1, 2)   # (B, D, N)
                spatial = spatial.reshape(B, self.embed_dim, grid, grid)
                features.append(spatial)

        # features[0] = shallowest block extraction (first in extract_layers)
        # features[-1] = deepest block extraction
        return features  # list of 4 × (B, 1024, grid, grid)


# ─────────────────────────────────────────────────────────────────────────────
# Channel projection: adapt SatMAE's 1024-d tokens to decoder channel sizes
# ─────────────────────────────────────────────────────────────────────────────

class FeaturePyramidAdapter(nn.Module):
    """
    Project SatMAE++ features (all D=1024) to the channel widths expected
    by the decoder [256, 512, 256, 128] (or any other set).

    Also applies a 3×3 conv to smooth/localize the features after
    reshaping from token space.
    """

    def __init__(self, embed_dim: int = 1024, out_channels: list = [128, 256, 512, 1024]):
        super().__init__()
        self.adapters = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(embed_dim, ch, 1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
            )
            for ch in out_channels
        ])

    def forward(self, features: list):
        return [adapter(feat)
                for adapter, feat in zip(self.adapters, features)]


# ─────────────────────────────────────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    encoder = SatMAEPlusEncoder(
        img_size=512, patch_size=16, embed_dim=1024,
        depth=24, num_heads=16,
        extract_layers=[5, 11, 17, 23],
        freeze=True,   # No gradients during backbone forward
    )
    encoder.eval()

    dummy = torch.randn(2, 3, 512, 512)
    with torch.no_grad():
        feats = encoder(dummy)

    print("SatMAE++ multi-scale feature shapes:")
    for i, f in enumerate(feats):
        print(f"  Scale {i}: {f.shape}")   # All (2, 1024, 32, 32) for 512px input

    # Test the adapter
    adapter = FeaturePyramidAdapter(1024, [128, 256, 512, 1024])
    adapted = adapter(feats)
    print("\nAfter FeaturePyramidAdapter:")
    for i, f in enumerate(adapted):
        print(f"  Scale {i}: {f.shape}")
