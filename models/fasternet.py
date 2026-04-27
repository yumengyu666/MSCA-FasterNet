"""FasterNet backbone implementation.

Adapted from the official FasterNet implementation:
https://github.com/JierunChen/FasterNet

Reference:
    Chen et al., "Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks",
    CVPR 2023.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple

# DropPath (Stochastic Depth) implementation
# Essential for modern lightweight networks to improve generalization
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample.

    When applied in main path of residual blocks, this layer randomly drops
    entire samples (not individual activations) during training with probability
    drop_prob. This is equivalent to training an ensemble of sub-networks.

    Reference:
        Deep Networks with Stochastic Depth (Huang et al., ECCV 2016)
    """

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with batches of different sizes: generate shape [B, 1, 1, ...]
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class PartialConv3x3(nn.Module):
    """Partial Convolution (PConv) - the core building block of FasterNet.

    Only convolves on a subset of input channels (1/n_div), leaving the rest
    unchanged. This reduces redundant computation while maintaining expressiveness.
    """

    def __init__(self, dim: int, n_div: int = 4, forward: str = "split_cat"):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(
            self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False
        )

        if forward == "slicing":
            self.forward = self._forward_slicing
        elif forward == "split_cat":
            self.forward = self._forward_split_cat
        else:
            raise NotImplementedError(f"forward method '{forward}' is not supported")

    def _forward_slicing(self, x: torch.Tensor) -> torch.Tensor:
        # Only used for inference
        x = x.clone()
        x[:, : self.dim_conv3, :, :] = self.partial_conv3(x[:, : self.dim_conv3, :, :])
        return x

    def _forward_split_cat(self, x: torch.Tensor) -> torch.Tensor:
        # Used for training
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)
        return x


class FasterNetBlock(nn.Module):
    """FasterNet Block: PConv -> PWConv1 (expand) -> PWConv2 (compress).

    Optionally inserts an MSCA module after PConv.
    Includes DropPath (Stochastic Depth) for improved generalization.
    """

    def __init__(
        self,
        dim: int,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        pconv_fw: str = "split_cat",
        msca_module: Optional[nn.Module] = None,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.mlp_ratio = mlp_ratio

        self.pconv = PartialConv3x3(dim, n_div, pconv_fw)
        self.msca = msca_module  # MSCA module (optional)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.pwconv1 = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, 1, 0, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
        )
        self.pwconv2 = nn.Conv2d(mlp_hidden_dim, dim, 1, 1, 0, bias=False)

        # DropPath (Stochastic Depth) for regularization
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        # PConv: partial spatial feature extraction
        x = self.pconv(x)

        # MSCA: multi-scale channel attention (if present)
        if self.msca is not None:
            x = self.msca(x)

        # PWConv: channel mixing
        x = self.pwconv1(x)
        x = self.pwconv2(x)

        # Residual connection with DropPath
        x = residual + self.drop_path(x)
        return x


class FasterNetStage(nn.Module):
    """A stage of FasterNet: multiple consecutive FasterNetBlocks."""

    def __init__(
        self,
        dim: int,
        depth: int,
        n_div: int = 4,
        mlp_ratio: float = 2.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        pconv_fw: str = "split_cat",
        msca_indices: Optional[List[int]] = None,
        msca_module_factory=None,
        drop_path_rate: float = 0.0,
        drop_path_rates: Optional[List[float]] = None,
    ):
        super().__init__()
        self.depth = depth

        if msca_indices is None:
            msca_indices = []

        # Support both legacy (uniform rate) and new (per-block rates)
        if drop_path_rates is not None:
            assert len(drop_path_rates) == depth, \
                f"drop_path_rates length {len(drop_path_rates)} != depth {depth}"
        else:
            drop_path_rates = [drop_path_rate] * depth

        blocks = []
        for i in range(depth):
            # Insert MSCA module at specified block indices
            msca = None
            if i in msca_indices and msca_module_factory is not None:
                msca = msca_module_factory(dim)

            # Stochastic depth: linearly increasing rate per block
            blocks.append(
                FasterNetBlock(
                    dim=dim,
                    n_div=n_div,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pconv_fw=pconv_fw,
                    msca_module=msca,
                    drop_path=drop_path_rates[i],
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.blocks(x)
        return x


class FasterNetEmbedding(nn.Module):
    """Patch embedding: non-overlapping 4x4 convolution with stride 4."""

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 40,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, 4, 4, 0, bias=False),
            norm_layer(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x)


class FasterNetMerging(nn.Module):
    """Spatial downsampling and channel expansion: 2x2 conv with stride 2."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
    ):
        super().__init__()
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 2, 2, 0, bias=False),
            norm_layer(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.merge(x)


class FasterNet(nn.Module):
    """FasterNet backbone.

    Generates multi-stage feature maps suitable for downstream tasks.
    Supports optional MSCA insertion and intermediate feature output for fusion.
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dim: int = 40,
        depths: List[int] = [1, 2, 8, 2],
        mlp_ratio: float = 2.0,
        n_div: int = 4,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.BatchNorm2d,
        pconv_fw: str = "split_cat",
        msca_config: Optional[dict] = None,
        out_indices: Tuple[int, ...] = (1, 2, 3),
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            in_channels: Input image channels (3 for RGB).
            embed_dim: Embedding dimension.
            depths: Number of blocks in each stage.
            mlp_ratio: MLP expansion ratio.
            n_div: PConv channel division ratio.
            act_layer: Activation layer.
            norm_layer: Normalization layer.
            pconv_fw: PConv forward method.
            msca_config: Dict specifying MSCA insertion, e.g.
                {"stage": 2, "indices": [6, 7], "factory": <callable>}.
                Stage index is 0-based (0=Stage1, 1=Stage2, ...).
            out_indices: Stages whose outputs are returned (0-based).
            drop_path_rate: Maximum DropPath rate (stochastic depth).
                Rates are linearly decayed from 0 to this value across all blocks.
        """
        super().__init__()
        self.out_indices = out_indices
        self.embed_dim = embed_dim
        self.depths = depths

        # Channel dimensions for each stage
        dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.dims = dims  # e.g., [40, 80, 160, 320]

        # Embedding
        self.embedding = FasterNetEmbedding(in_channels, embed_dim, act_layer, norm_layer)

        # Stages and merging layers
        self.stages = nn.ModuleList()
        self.mergings = nn.ModuleList()

        # Parse MSCA config
        msca_stage = -1
        msca_indices = []
        msca_factory = None
        if msca_config is not None:
            msca_stage = msca_config.get("stage", -1)
            msca_indices = msca_config.get("indices", [])
            msca_factory = msca_config.get("factory", None)

        # Compute per-block drop path rates with linear decay
        # Standard stochastic depth: rate linearly increases from 0 to drop_path_rate
        # Reference: Huang et al., "Deep Networks with Stochastic Depth", ECCV 2016
        total_blocks = sum(depths)
        block_idx = 0
        for i in range(len(depths)):
            # Determine MSCA insertion for this stage
            stage_msca_indices = []
            stage_msca_factory = None
            if i == msca_stage and msca_factory is not None:
                stage_msca_indices = msca_indices
                stage_msca_factory = msca_factory

            # Linear decay: first block gets ~0, last block gets drop_path_rate
            # Compute per-block rates for this stage
            dpr_list = [
                drop_path_rate * block_idx / max(total_blocks - 1, 1)
                for block_idx in range(block_idx, block_idx + depths[i])
            ]

            # Stage (pass list of per-block dpr)
            self.stages.append(
                FasterNetStage(
                    dim=dims[i],
                    depth=depths[i],
                    n_div=n_div,
                    mlp_ratio=mlp_ratio,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    pconv_fw=pconv_fw,
                    msca_indices=stage_msca_indices,
                    msca_module_factory=stage_msca_factory,
                    drop_path_rates=dpr_list,
                )
            )
            block_idx += depths[i]

            # Merging (except after the last stage)
            if i < len(depths) - 1:
                self.mergings.append(
                    FasterNetMerging(dims[i], dims[i + 1], act_layer, norm_layer)
                )

    def forward(self, x: torch.Tensor) -> dict:
        """Forward pass returning intermediate features.

        Returns:
            Dictionary mapping stage index (0-based) to feature tensor.
            e.g., {1: stage2_feat, 2: stage3_feat, 3: stage4_feat}
        """
        features = {}

        x = self.embedding(x)

        for i in range(len(self.depths)):
            x = self.stages[i](x)
            if i in self.out_indices:
                features[i] = x
            if i < len(self.depths) - 1:
                x = self.mergings[i](x)

        return features


def FasterNetT0(**kwargs) -> FasterNet:
    """Constructs FasterNet-T0 model (smallest variant).

    Architecture:
        embed_dim=40, depths=[1,2,8,2], n_div=4, mlp_ratio=2.0
        Parameters: ~3.9M (without classification head)
        FLOPs: ~0.34G (224x224 input)
    """
    default_kwargs = {
        "in_channels": 3,
        "embed_dim": 40,
        "depths": [1, 2, 8, 2],
        "mlp_ratio": 2.0,
        "n_div": 4,
        "act_layer": nn.GELU,
        "norm_layer": nn.BatchNorm2d,
        "pconv_fw": "split_cat",
        "out_indices": (1, 2, 3),
    }
    default_kwargs.update(kwargs)
    return FasterNet(**default_kwargs)
