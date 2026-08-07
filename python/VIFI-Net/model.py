"""VIFI-Net: cross-modal fusion network for 3D EIT reconstruction.

Pipeline (matches the paper):
  1. Voltage branch: the Dual-M SMP voltage differences Y (2D dv matrix) are
     encoded by ``VoltageEncoder`` into (i) FiLM conditioning parameters and
     (ii) a compact set of voltage tokens.
  2. Image branch: the UREIT initial image X_init is encoded by the 3D
     encoder with Sensitivity-Aware Spatial Attention (SASA) into a three-
     scale feature pyramid.
  3. Decoder: at each scale, Cross-Domain Attention lets the image features
     query the voltage tokens (residual fusion), followed by FiLM modulation
     and the standard up-sample + skip-connection path.
  4. The decoder outputs a residual update, and the final reconstruction is
     ``X_init + delta_X``.

Architectural note (intentional deviation from the paper):
  The paper describes three down-sampling stages (64 -> 128 -> 256 -> 512);
  this implementation uses two (64 -> 128 -> 256), i.e. the deepest feature
  stays at 256 channels and one up/down stage is removed. This is an
  intentional simplification that reduces parameters and memory; extend
  ``ImageEncoder3D``/``ImageDecoder3D`` with one more stage to match the
  paper exactly.

Shape convention:
  X_init and sensitivity_map must have even spatial dimensions (D, H, W),
  because every down-sampling conv has stride 2 with padding 1 and the
  decoder doubles the size with ``Upsample(scale_factor=2)``; odd sizes
  would break the skip-connection concatenation by one voxel.

All layer structures and forward computations follow the original
implementation; only code hygiene and correctness fixes (e.g. registering
the sensitivity map as a non-persistent buffer) were applied.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "VIFINet",
    "VoltageEncoder",
    "ImageEncoder3D",
    "ImageDecoder3D",
    "CrossDomainAttention",
    "SensitivityAwareSpatialAttention3D",
]


def _divisible_groups(num_channels: int, default_groups: int = 32) -> int:
    """Largest divisor of ``num_channels`` that does not exceed ``default_groups``.

    GroupNorm requires ``num_channels % num_groups == 0``; for channels below
    the default, this walks downward until a valid group count is found
    (always terminates at 1).
    """
    groups = min(default_groups, num_channels)
    while num_channels % groups != 0:
        groups -= 1
    return groups


# ============================================================================
# Part 1: 3D basic building blocks
# ============================================================================


class ConvBlock3D(nn.Module):
    """3D convolution block: Conv3d -> GroupNorm -> SiLU (Swish)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 32,
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, bias=False,
        )
        self.gn = nn.GroupNorm(
            num_groups=_divisible_groups(out_channels, groups),
            num_channels=out_channels, affine=True,
        )
        self.swish = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.swish(self.gn(self.conv(x)))


class ConvBlock(nn.Module):
    """2D convolution block: Conv2d -> GroupNorm -> SiLU.

    Used by the voltage branch, which processes 2D dv matrices.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False,
        )
        self.gn = nn.GroupNorm(
            num_groups=_divisible_groups(out_channels),
            num_channels=out_channels, affine=True,
        )
        self.silu = nn.SiLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.silu(self.gn(self.conv(x)))


class ResBlock3D(nn.Module):
    """3D residual block: x + ConvBlock(x) with constant channel count."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3D(channels, channels),
            ConvBlock3D(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class SpatialAttention2D(nn.Module):
    """2D spatial attention (channel-wise max/avg pooling -> 7x7 conv).

    Input:  (B, C, H, W)
    Output: (B, C, H, W), channel attention gates multiplied elementwise.
    """

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        avg_pool = torch.mean(x, dim=1, keepdim=True)          # (B, 1, H, W)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)        # (B, 1, H, W)
        concat = torch.cat([avg_pool, max_pool], dim=1)        # (B, 2, H, W)
        attn = self.sigmoid(self.conv(concat))
        return x * attn


class SensitivityAwareSpatialAttention3D(nn.Module):
    """Sensitivity-aware spatial attention (SASA).

    Fusion of MaxPool, AvgPool and a prior sensitivity map through a small
    MLP (pointwise conv + ReLU + pointwise conv + Tanh), applied as a
    residual bidirectional modulation:

        out = x + x * Tanh(MLP(MaxPool(x), AvgPool(x), S))

    Args:
        sensitivity_map: (D, H, W) prior sensitivity volume, registered as a
            non-persistent buffer so it follows ``.to(device)`` without
            entering the state dict.
        hidden_channels: intermediate channels of the MLP.
        use_depthwise_spatial: if True, the concatenated 3-channel input is
            first processed by a 3x3x3 depthwise conv (local context); if
            False, only pointwise convolutions are used. Note: the 1x1x1
            ``spatial_conv`` created in the False branch is kept for
            interface compatibility but is not used in ``forward``.
    """

    def __init__(
        self,
        sensitivity_map: Tensor,
        hidden_channels: int = 32,
        use_depthwise_spatial: bool = False,
    ):
        super().__init__()
        assert sensitivity_map.dim() == 3, \
            "sensitivity_map must have shape (D, H, W)"
        self.register_buffer("sensitivity_map", sensitivity_map, persistent=False)
        self.use_depthwise_spatial = use_depthwise_spatial

        if use_depthwise_spatial:
            # Depthwise 3x3x3 conv: local spatial continuity, few parameters.
            self.spatial_conv = nn.Conv3d(
                in_channels=3, out_channels=3, kernel_size=3,
                padding=1, groups=3, bias=False,
            )
        else:
            # Pointwise-only path (recommended). Kept for compatibility;
            # see the class docstring.
            self.spatial_conv = nn.Conv3d(
                in_channels=3, out_channels=3, kernel_size=1, bias=False,
            )

        # MLP: channel expansion -> ReLU -> channel compression -> Tanh.
        self.conv_expand = nn.Conv3d(3, hidden_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_compress = nn.Conv3d(hidden_channels, 1, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x: Tensor) -> Tensor:
        """Apply SASA modulation. ``x``: (B, C, D, H, W)."""
        B, C, D, H, W = x.shape

        max_pool = torch.max(x, dim=1, keepdim=True)[0]        # (B, 1, D, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)          # (B, 1, D, H, W)

        s_map = self.sensitivity_map.view(1, 1, *self.sensitivity_map.shape)
        s_map = s_map.expand(B, 1, D, H, W)

        concat_feat = torch.cat([max_pool, avg_pool, s_map], dim=1)  # (B, 3, D, H, W)

        if self.use_depthwise_spatial:
            concat_feat = self.spatial_conv(concat_feat)

        mask = self.conv_expand(concat_feat)
        mask = self.relu(mask)
        mask = self.conv_compress(mask)                        # (B, 1, D, H, W)
        mask = self.tanh(mask)                                 # range (-1, 1)

        # Residual bidirectional modulation; mask broadcasts over channels.
        return x + x * mask


# ============================================================================
# Part 2: Voltage encoder (2D, the input is a dv matrix)
# ============================================================================


class CrossDomainAttention(nn.Module):
    """Cross-domain attention: image features query voltage tokens.

    ``query``:     (B, seq_q, query_dim)  -- flattened image feature maps
    ``key_value``: (B, seq_kv, token_dim) -- voltage tokens

    The attention output is added residually to the image features by the
    caller.
    """

    def __init__(self, query_dim: int, token_dim: int, num_heads: int = 4):
        super().__init__()
        assert query_dim % num_heads == 0, "query_dim must be divisible by num_heads"
        self.query_dim = query_dim
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.w_q = nn.Linear(query_dim, query_dim)   # image features
        self.w_k = nn.Linear(token_dim, query_dim)   # voltage tokens
        self.w_v = nn.Linear(token_dim, query_dim)
        self.w_o = nn.Linear(query_dim, query_dim)

    def forward(self, query: Tensor, key_value: Tensor) -> Tensor:
        B, seq_q, _ = query.shape
        _, seq_kv, _ = key_value.shape

        Q = self.w_q(query)                            # (B, seq_q, query_dim)
        K = self.w_k(key_value)                        # (B, seq_kv, query_dim)
        V = self.w_v(key_value)                        # (B, seq_kv, query_dim)

        # Split into heads: (B, heads, seq, head_dim).
        Q = Q.view(B, seq_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_kv, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Merge heads and project.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(B, seq_q, self.query_dim)
        return self.w_o(attn_output)


class VoltageEncoder(nn.Module):
    """Voltage-domain encoder: maps the dv matrix to FiLM params + tokens.

    Args:
        token_dim: token width used by the cross-domain attention.
        volt_dim:  number of voltage channels (Dual-M SMP independent
            channels; e.g. 464 for a 32-electrode system).
    """

    def __init__(self, token_dim: int = 128, volt_dim: int = 464):
        super().__init__()
        self.token_dim = token_dim

        # Two-stage strided convs enlarge the receptive field for larger
        # electrode counts; input Y is (B, 1, H, W).
        self.conv_blocks = nn.Sequential(
            ConvBlock(1, 64, stride=2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, self.token_dim, stride=2),
        )

        # Attention pooling to a fixed (4, 4) token grid.
        self.sa_block = SpatialAttention2D()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # FiLM conditioner on the voltage vector y.
        self.mlp_film = nn.Sequential(
            nn.Linear(volt_dim, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
        )
        self.film_64 = nn.Linear(512, 2 * 64)
        self.film_128 = nn.Linear(512, 2 * 128)
        self.film_256 = nn.Linear(512, 2 * 256)

    def forward(self, Y: Tensor, y: Tensor) -> tuple[dict[str, tuple[Tensor, Tensor]], Tensor]:
        """Encode voltage data.

        Args:
            Y: (B, 1, H, W) voltage-difference matrix (Dual-M SMP channels
                reshaped onto the electrode layout).
            y: (B, volt_dim) raw voltage vector for FiLM conditioning.

        Returns:
            film_params: dict {'64','128','256'} -> (gamma, beta) with
                gamma = 1 + tanh(...) and beta from the corresponding head.
            tokens: (B, seq_kv, token_dim) voltage tokens for cross-domain
                attention.
        """
        latent_film = self.mlp_film(y)                 # (B, 512)

        feat = self.conv_blocks(Y)
        feat = self.sa_block(feat)
        feat = self.pool(feat)                         # (B, token_dim, 4, 4)
        tokens = feat.flatten(start_dim=-2)            # (B, token_dim, 16)
        tokens = tokens.transpose(1, 2)                # (B, 16, token_dim)

        def get_film(proj: nn.Linear) -> tuple[Tensor, Tensor]:
            gamma, beta = proj(latent_film).chunk(2, dim=-1)
            return 1.0 + torch.tanh(gamma), beta

        film_params = {
            "64": get_film(self.film_64),
            "128": get_film(self.film_128),
            "256": get_film(self.film_256),
        }
        return film_params, tokens


# ============================================================================
# Part 3: Image-domain 3D architecture
# ============================================================================


class FiLMLayer3D(nn.Module):
    """3D FiLM modulation: feat * gamma + beta with broadcast over (D, H, W)."""

    def forward(self, feat: Tensor, gamma: Tensor, beta: Tensor) -> Tensor:
        # gamma/beta: (B, C) -> (B, C, 1, 1, 1)
        return feat * gamma.view(*gamma.shape, 1, 1, 1) + beta.view(*beta.shape, 1, 1, 1)


class ImageEncoder3D(nn.Module):
    """3D image encoder with a three-scale feature pyramid.

    NOTE: the paper uses three down-sampling stages (64 -> 128 -> 256 ->
    512); this implementation keeps two (64 -> 128 -> 256), i.e. one level
    shallower. See the module docstring.
    """

    def __init__(self, sensitivity_map: Tensor):
        super().__init__()
        self.stem = ConvBlock3D(1, 64)

        self.e1_res = ResBlock3D(64)
        self.e1_sasa = SensitivityAwareSpatialAttention3D(sensitivity_map)

        self.down1 = ConvBlock3D(64, 128, stride=2)   # 1st down-sampling
        self.e2_res = ResBlock3D(128)

        self.down2 = ConvBlock3D(128, 256, stride=2)  # 2nd down-sampling
        self.e3_res = ResBlock3D(256)
        # (3rd down-sampling stage of the paper is intentionally omitted.)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        e1 = self.e1_sasa(self.e1_res(self.stem(x)))   # (B, 64, ...)
        e2 = self.e2_res(self.down1(e1))               # (B, 128, ...)
        e3 = self.e3_res(self.down2(e2))               # (B, 256, ...)
        return e1, e2, e3


class ImageDecoder3D(nn.Module):
    """3D decoder: cross-domain attention + FiLM + up-sampling/skip path.

    At each of the three scales the image feature queries the voltage tokens
    (residual attention), is FiLM-conditioned, up-sampled and fused with the
    corresponding encoder feature via a skip connection. The head predicts a
    residual image ``delta_X``.
    """

    def __init__(self, token_dim: int = 128, use_film: bool = True):
        super().__init__()
        self.use_film = use_film

        # FiLM layers per scale.
        self.film256 = FiLMLayer3D()
        self.film128 = FiLMLayer3D()
        self.film64 = FiLMLayer3D()

        # Decoder stages (up-sample, 1x1 conv, skip fusion, residual block).
        self.up1 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv_up1 = ConvBlock3D(256, 128, kernel_size=1, padding=0)
        self.cat1 = ConvBlock3D(256, 128, kernel_size=1, padding=0)   # 128 + 128
        self.res1 = ResBlock3D(128)

        self.up2 = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.conv_up2 = ConvBlock3D(128, 64, kernel_size=1, padding=0)
        self.cat2 = ConvBlock3D(128, 64, kernel_size=1, padding=0)    # 64 + 64
        self.res2 = ResBlock3D(64)

        # Cross-domain attention at the three scales.
        self.cross_attn_d1 = CrossDomainAttention(query_dim=256, token_dim=token_dim)
        self.cross_attn_d2 = CrossDomainAttention(query_dim=128, token_dim=token_dim)
        self.cross_attn_d3 = CrossDomainAttention(query_dim=64, token_dim=token_dim)

        self.head = nn.Conv3d(64, 1, kernel_size=1)

    def forward(
        self,
        e3: Tensor,
        e2: Tensor,
        e1: Tensor,
        v_tokens: Tensor,
        film: dict[str, tuple[Tensor, Tensor]],
    ) -> Tensor:
        """Decode to a residual image.

        Args:
            e3, e2, e1: encoder features at the three scales.
            v_tokens: voltage tokens from ``VoltageEncoder``.
            film: FiLM parameters {'256','128','64'} -> (gamma, beta).
        """
        # ---- scale 1: deepest feature (256 channels) ---------------------
        B, C, D, H, W = e3.shape
        e3_flat = e3.view(B, C, -1).transpose(1, 2)
        d1 = e3 + self.cross_attn_d1(e3_flat, v_tokens).transpose(1, 2).view(B, C, D, H, W)
        feat = self.film256(d1, *film["256"]) if self.use_film else d1

        feat = self.conv_up1(self.up1(feat))
        feat = self.res1(self.cat1(torch.cat([feat, e2], dim=1)))

        # ---- scale 2 (128 channels) --------------------------------------
        B, C, D, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).transpose(1, 2)
        d2 = feat + self.cross_attn_d2(feat_flat, v_tokens).transpose(1, 2).view(B, C, D, H, W)
        feat = self.film128(d2, *film["128"]) if self.use_film else d2

        feat = self.conv_up2(self.up2(feat))
        feat = self.res2(self.cat2(torch.cat([feat, e1], dim=1)))

        # ---- scale 3 (64 channels) ---------------------------------------
        B, C, D, H, W = feat.shape
        feat_flat = feat.view(B, C, -1).transpose(1, 2)
        d3 = feat + self.cross_attn_d3(feat_flat, v_tokens).transpose(1, 2).view(B, C, D, H, W)
        feat = self.film64(d3, *film["64"]) if self.use_film else d3

        return self.head(feat)


class VIFINet(nn.Module):
    """VIFI-Net: voltage-guided 3D EIT reconstruction network.

    Args:
        sensitivity_map: (D, H, W) prior sensitivity volume of the forward
            model (e.g. from UREIT or the FEM mesh).
        token_dim: voltage-token width (see ``VoltageEncoder``).
        use_film: enable FiLM conditioning in the decoder.
    """

    def __init__(self, sensitivity_map: Tensor, token_dim: int = 128, use_film: bool = True):
        super().__init__()
        self.v_enc = VoltageEncoder(token_dim=token_dim)
        self.i_enc = ImageEncoder3D(sensitivity_map)
        self.decoder = ImageDecoder3D(token_dim=token_dim, use_film=use_film)

    def forward(self, Y: Tensor, y: Tensor, X_init: Tensor) -> Tensor:
        """Full reconstruction.

        Args:
            Y: (B, 1, H, W) voltage-difference matrix.
            y: (B, volt_dim) voltage vector.
            X_init: (B, 1, D, H, W) initial image from UREIT.

        Returns:
            (B, 1, D, H, W) refined reconstruction (residual learning).
        """
        film_params, v_tokens = self.v_enc(Y, y)
        e1, e2, e3 = self.i_enc(X_init)
        delta_x = self.decoder(e3, e2, e1, v_tokens, film_params)
        return X_init + delta_x
