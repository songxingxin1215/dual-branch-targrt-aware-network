from __future__ import annotations

import torch
import torch.nn as nn


class DepthwiseSeparableConv(nn.Module):
    """A depthwise spatial convolution followed by a pointwise channel mixing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
    ):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(inputs))


class ContextBlock(nn.Module):
    """A residual block with dilated depthwise-separable spatial context."""

    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            DepthwiseSeparableConv(channels, channels, dilation=dilation),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            DepthwiseSeparableConv(channels, channels, dilation=dilation),
            nn.BatchNorm2d(channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(inputs + self.block(inputs))


class TargetAwareHSIDetector(nn.Module):
    """Dual-branch, center-aware binary detector for HSI target detection.

    The input contains a raw spectral patch followed by its full-image PCA
    representation. Both spectral branches are fused at every spatial location;
    a learned gate suppresses background-like responses before global pooling
    and the explicit center-pixel feature are combined for classification.

    Input shape is ``(batch, 1, raw_bands + pca_channels, height, width)``.
    """

    def __init__(
        self,
        num_bands: int,
        pca_channels: int = 32,
        patch_size: int = 9,
        hidden_channels: int = 64,
    ):
        super().__init__()
        if num_bands < 1 or pca_channels < 1:
            raise ValueError("num_bands and pca_channels must be positive")
        if patch_size < 3 or patch_size % 2 == 0:
            raise ValueError("patch_size must be an odd integer of at least 3")
        if hidden_channels < 1:
            raise ValueError("hidden_channels must be positive")

        self.num_bands = num_bands
        self.pca_channels = pca_channels
        self.patch_size = patch_size

        self.raw_branch = nn.Sequential(
            nn.Conv2d(num_bands, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            ContextBlock(hidden_channels),
            ContextBlock(hidden_channels, dilation=2),
            nn.Conv2d(hidden_channels, pca_channels, 1, bias=False),
            nn.BatchNorm2d(pca_channels),
            nn.SiLU(inplace=True),
        )

        self.pca_normalize = nn.Sequential(
            nn.Conv2d(pca_channels, pca_channels, 1, bias=False),
            nn.BatchNorm2d(pca_channels),
            nn.SiLU(inplace=True),
        )
        self.pca_branch = nn.Sequential(
            ContextBlock(pca_channels),
            ContextBlock(pca_channels, dilation=3),
        )

        fused_input_channels = pca_channels * 3
        self.target_gate = nn.Sequential(
            nn.Conv2d(fused_input_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True),
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),
        )

        self.local_context = DepthwiseSeparableConv(
            fused_input_channels, pca_channels, dilation=1
        )
        self.mid_context = DepthwiseSeparableConv(
            fused_input_channels, pca_channels, dilation=2
        )
        self.long_context = DepthwiseSeparableConv(
            fused_input_channels, pca_channels, dilation=4
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(pca_channels * 6, pca_channels, 1, bias=False),
            nn.BatchNorm2d(pca_channels),
            nn.SiLU(inplace=True),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.LayerNorm(pca_channels * 2),
            nn.Linear(pca_channels * 2, pca_channels),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(pca_channels, 2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 5 or inputs.shape[1] != 1:
            raise ValueError(
                f"expected input shape (B, 1, C, H, W), got {tuple(inputs.shape)}"
            )
        expected_channels = self.num_bands + self.pca_channels
        if inputs.shape[2] != expected_channels:
            raise ValueError(
                f"expected {expected_channels} spectral channels, got {inputs.shape[2]}"
            )
        if inputs.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"expected spatial size ({self.patch_size}, {self.patch_size}), "
                f"got {tuple(inputs.shape[-2:])}"
            )

        patches = inputs.squeeze(1)
        raw = patches[:, : self.num_bands]
        pca = patches[:, self.num_bands :]

        raw_feature = self.raw_branch(raw)
        pca_feature = self.pca_branch(self.pca_normalize(pca))
        fused_input = torch.cat((raw_feature, pca_feature, pca), dim=1)
        gate = self.target_gate(fused_input)

        contexts = torch.cat(
            (
                self.local_context(fused_input),
                self.mid_context(fused_input),
                self.long_context(fused_input),
                raw_feature,
                pca_feature,
                pca,
            ),
            dim=1,
        )
        fused = self.fusion(contexts) * gate

        pooled = self.pool(fused).flatten(1)
        center = fused[:, :, self.patch_size // 2, self.patch_size // 2]
        return self.head(torch.cat((pooled, center), dim=1))
