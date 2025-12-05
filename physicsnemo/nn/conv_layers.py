# SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import contextlib
from typing import Any, Dict, List, Set

import nvtx
import torch
from torch import nn
from torch.nn.functional import silu

from physicsnemo.nn import Attention, Linear, get_group_norm
from physicsnemo.nn.utils.utils import _validate_amp
from physicsnemo.nn.utils.weight_init import _weight_init


class CubeEmbedding(nn.Module):
    """
    3D Image Cube Embedding
    Args:
        img_size (tuple[int]): Image size [T, Lat, Lon].
        patch_size (tuple[int]): Patch token size [T, Lat, Lon].
        in_chans (int): Number of input image channels.
        embed_dim (int): Number of projection output channels.
        norm_layer (nn.Module, optional): Normalization layer. Default: torch.nn.LayerNorm
    """

    def __init__(
        self, img_size, patch_size, in_chans, embed_dim, norm_layer=nn.LayerNorm
    ):
        super().__init__()
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
            img_size[2] // patch_size[2],
        ]

        self.img_size = img_size
        self.patches_resolution = patches_resolution
        self.embed_dim = embed_dim
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x: torch.Tensor):
        B, C, T, Lat, Lon = x.shape
        x = self.proj(x).reshape(B, self.embed_dim, -1).transpose(1, 2)  # B T*Lat*Lon C
        if self.norm is not None:
            x = self.norm(x)
        x = x.transpose(1, 2).reshape(B, self.embed_dim, *self.patches_resolution)
        return x


class ConvBlock(nn.Module):
    """
    Conv2d block
    Args:
        in_chans (int): Number of input channels.
        out_chans (int): Number of output channels.
        num_groups (int): Number of groups to separate the channels into for group normalization.
        num_residuals (int, optinal): Number of Conv2d operator. Default: 2
        upsample (int, optinal): 1: Upsample, 0: Conv, -1: Downsample. Default: 0
    """

    def __init__(self, in_chans, out_chans, num_groups, num_residuals=2, upsample=0):
        super().__init__()
        if upsample == 1:
            self.conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2)
        elif upsample == -1:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=2, padding=1
            )
        elif upsample == 0:
            self.conv = nn.Conv2d(
                in_chans, out_chans, kernel_size=(3, 3), stride=1, padding=1
            )

        blk = []
        for i in range(num_residuals):
            blk.append(
                nn.Conv2d(out_chans, out_chans, kernel_size=3, stride=1, padding=1)
            )
            blk.append(nn.GroupNorm(num_groups, out_chans))
            blk.append(nn.SiLU())

        self.b = nn.Sequential(*blk)

    def forward(self, x):
        x = self.conv(x)
        x_skip = x
        x = self.b(x)
        return x + x_skip


class Conv2d(torch.nn.Module):
    """
    A custom 2D convolutional layer implementation with support for up-sampling,
    down-sampling, and custom weight and bias initializations. The layer's weights
    and biases canbe initialized using custom initialization strategies like
    "kaiming_normal", and can be further scaled by factors `init_weight` and
    `init_bias`.

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel : int
        Size of the convolving kernel.
    bias : bool, optional
        The biases of the layer. If set to `None`, the layer will not learn an
        additive bias. By default True.
    up : bool, optional
        Whether to perform up-sampling. By default False.
    down : bool, optional
        Whether to perform down-sampling. By default False.
    resample_filter : List[int], optional
        Filter to be used for resampling. By default [1, 1].
    fused_resample : bool, optional
        If True, performs fused up-sampling and convolution or fused down-sampling
        and convolution. By default False.
    init_mode : str, optional (default="kaiming_normal")
        init_mode : str, optional (default="kaiming_normal")
        The mode/type of initialization to use for weights and biases. Supported modes
        are:
        - "xavier_uniform": Xavier (Glorot) uniform initialization.
        - "xavier_normal": Xavier (Glorot) normal initialization.
        - "kaiming_uniform": Kaiming (He) uniform initialization.
        - "kaiming_normal": Kaiming (He) normal initialization.
        By default "kaiming_normal".
    init_weight : float, optional
        A scaling factor to multiply with the initialized weights. By default 1.0.
    init_bias : float, optional
        A scaling factor to multiply with the initialized biases. By default 0.0.
    fused_conv_bias: bool, optional
        A boolean flag indicating whether bias will be passed as a parameter of conv2d. By default False.
    amp_mode : bool, optional
        A boolean flag indicating whether mixed-precision (AMP) training is enabled. Defaults to False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel: int,
        bias: bool = True,
        up: bool = False,
        down: bool = False,
        resample_filter: List[int] = [1, 1],
        fused_resample: bool = False,
        init_mode: str = "kaiming_normal",
        init_weight: float = 1.0,
        init_bias: float = 0.0,
        fused_conv_bias: bool = False,
        amp_mode: bool = False,
    ):
        if up and down:
            raise ValueError("Both 'up' and 'down' cannot be true at the same time.")
        if not kernel and fused_conv_bias:
            print(
                "Warning: Kernel is required when fused_conv_bias is enabled. Setting fused_conv_bias to False."
            )
            fused_conv_bias = False

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample
        self.fused_conv_bias = fused_conv_bias
        self.amp_mode = amp_mode
        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_channels * kernel * kernel,
            fan_out=out_channels * kernel * kernel,
        )
        self.weight = (
            torch.nn.Parameter(
                _weight_init([out_channels, in_channels, kernel, kernel], **init_kwargs)
                * init_weight
            )
            if kernel
            else None
        )
        self.bias = (
            torch.nn.Parameter(_weight_init([out_channels], **init_kwargs) * init_bias)
            if kernel and bias
            else None
        )
        f = torch.as_tensor(resample_filter, dtype=torch.float32)
        f = f.ger(f).unsqueeze(0).unsqueeze(1) / f.sum().square()
        self.register_buffer("resample_filter", f if up or down else None)

    def forward(self, x):
        weight, bias, resample_filter = self.weight, self.bias, self.resample_filter
        _validate_amp(self.amp_mode)
        if not self.amp_mode:
            if self.weight is not None and self.weight.dtype != x.dtype:
                weight = self.weight.to(x.dtype)
            if self.bias is not None and self.bias.dtype != x.dtype:
                bias = self.bias.to(x.dtype)
            if (
                self.resample_filter is not None
                and self.resample_filter.dtype != x.dtype
            ):
                resample_filter = self.resample_filter.to(x.dtype)

        w = weight if weight is not None else None
        b = bias if bias is not None else None
        f = resample_filter if resample_filter is not None else None
        w_pad = w.shape[-1] // 2 if w is not None else 0
        f_pad = (f.shape[-1] - 1) // 2 if f is not None else 0

        if self.fused_resample and self.up and w is not None:
            x = torch.nn.functional.conv_transpose2d(
                x,
                f.mul(4).tile([self.in_channels, 1, 1, 1]),
                groups=self.in_channels,
                stride=2,
                padding=max(f_pad - w_pad, 0),
            )
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x, w, padding=max(w_pad - f_pad, 0), bias=b
                )
            else:
                x = torch.nn.functional.conv2d(x, w, padding=max(w_pad - f_pad, 0))
        elif self.fused_resample and self.down and w is not None:
            x = torch.nn.functional.conv2d(x, w, padding=w_pad + f_pad)
            if self.fused_conv_bias:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                    bias=b,
                )
            else:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.out_channels, 1, 1, 1]),
                    groups=self.out_channels,
                    stride=2,
                )
        else:
            if self.up:
                x = torch.nn.functional.conv_transpose2d(
                    x,
                    f.mul(4).tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if self.down:
                x = torch.nn.functional.conv2d(
                    x,
                    f.tile([self.in_channels, 1, 1, 1]),
                    groups=self.in_channels,
                    stride=2,
                    padding=f_pad,
                )
            if w is not None:  # ask in corrdiff channel whether w will ever be none
                if self.fused_conv_bias:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad, bias=b)
                else:
                    x = torch.nn.functional.conv2d(x, w, padding=w_pad)
        if b is not None and not self.fused_conv_bias:
            x = x.add_(b.reshape(1, -1, 1, 1))
        return x


class UNetBlock(torch.nn.Module):
    """
    Unified U-Net block with optional up/downsampling and self-attention. Represents
    the union of all features employed by the DDPM++, NCSN++, and ADM architectures.

    Parameters:
    -----------
    in_channels : int
        Number of input channels :math:`C_{in}`.
    out_channels : int
        Number of output channels :math:`C_{out}`.
    emb_channels : int
        Number of embedding channels :math:`C_{emb}`.
    up : bool, optional, default=False
        If True, applies upsampling in the forward pass.
    down : bool, optional, default=False
        If True, applies downsampling in the forward pass.
    attention : bool, optional, default=False
        If True, enables the self-attention mechanism in the block.
    num_heads : int, optional, default=None
        Number of attention heads. If None, defaults to :math:`C_{out} / 64`.
    channels_per_head : int, optional, default=64
        Number of channels per attention head.
    dropout : float, optional, default=0.0
        Dropout probability.
    skip_scale : float, optional, default=1.0
        Scale factor applied to skip connections.
    eps : float, optional, default=1e-5
        Epsilon value used for normalization layers.
    resample_filter : List[int], optional, default=``[1, 1]``
        Filter for resampling layers.
    resample_proj : bool, optional, default=False
        If True, resampling projection is enabled.
    adaptive_scale : bool, optional, default=True
        If True, uses adaptive scaling in the forward pass.
    init : dict, optional, default=``{}``
        Initialization parameters for convolutional and linear layers.
    init_zero : dict, optional, default=``{'init_weight': 0}``
        Initialization parameters with zero weights for certain layers.
    init_attn : dict, optional, default=``None``
        Initialization parameters specific to attention mechanism layers.
        Defaults to ``init`` if not provided.
    use_apex_gn : bool, optional, default=False
        A boolean flag indicating whether we want to use Apex GroupNorm for NHWC layout.
        Need to set this as False on cpu.
    act : str, optional, default=None
        The activation function to use when fusing activation with GroupNorm.
    fused_conv_bias: bool, optional, default=False
        A boolean flag indicating whether bias will be passed as a parameter of conv2d.
    profile_mode: bool, optional, default=False
        A boolean flag indicating whether to enable all nvtx annotations during profiling.
    amp_mode : bool, optional, default=False
        A boolean flag indicating whether mixed-precision (AMP) training is
        enabled.

    Forward
    -------
    x : torch.Tensor
        Input tensor of shape :math:`(B, C_{in}, H, W)`, where :math:`B` is batch
        size, :math:`C_{in}` is ``in_channels``, and :math:`H, W` are spatial
        dimensions.
    emb : torch.Tensor
        Embedding tensor of shape :math:`(B, C_{emb})`, where :math:`B` is batch
        size, and :math:`C_{emb}` is ``emb_channels``.

    Outputs
    -------
    torch.Tensor
        Output tensor of shape :math:`(B, C_{out}, H, W)`, where :math:`B` is batch
        size, :math:`C_{out}` is ``out_channels``, and :math:`H, W` are spatial
        dimensions.
    """

    # NOTE: these attributes have specific usage in old checkpoints, do not
    # reuse them!
    _reserved_attributes: Set[str] = set(["norm2", "qkv", "proj"])

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        emb_channels: int,
        up: bool = False,
        down: bool = False,
        attention: bool = False,
        num_heads: int | None = None,
        channels_per_head: int = 64,
        dropout: float = 0.0,
        skip_scale: float = 1.0,
        eps: float = 1e-5,
        resample_filter: List[int] = [1, 1],
        resample_proj: bool = False,
        adaptive_scale: bool = True,
        init: Dict[str, Any] = dict(),
        init_zero: Dict[str, Any] = dict(init_weight=0),
        init_attn: Any = None,
        use_apex_gn: bool = False,
        act: str = "silu",
        fused_conv_bias: bool = False,
        profile_mode: bool = False,
        amp_mode: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.num_heads = (
            0
            if not attention
            else (
                num_heads
                if num_heads is not None
                else out_channels // channels_per_head
            )
        )
        self.attention = attention
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale
        self.profile_mode = profile_mode
        self.amp_mode = amp_mode
        self.norm0 = get_group_norm(
            num_channels=in_channels,
            eps=eps,
            use_apex_gn=use_apex_gn,
            act=act,
            amp_mode=amp_mode,
        )
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init,
        )
        self.affine = Linear(
            in_features=emb_channels,
            out_features=out_channels * (2 if adaptive_scale else 1),
            amp_mode=amp_mode,
            **init,
        )
        if self.adaptive_scale:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
            )
        else:
            self.norm1 = get_group_norm(
                num_channels=out_channels,
                eps=eps,
                use_apex_gn=use_apex_gn,
                act=act,
                amp_mode=amp_mode,
            )
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            fused_conv_bias=fused_conv_bias,
            amp_mode=amp_mode,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            fused_conv_bias = fused_conv_bias if kernel != 0 else False
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                fused_conv_bias=fused_conv_bias,
                amp_mode=amp_mode,
                **init,
            )

        if self.attention:
            self.attn = Attention(
                out_channels=out_channels,
                num_heads=self.num_heads,
                eps=eps,
                init_zero=init_zero,
                init_attn=init_attn,
                init=init,
                use_apex_gn=use_apex_gn,
                amp_mode=amp_mode,
                fused_conv_bias=fused_conv_bias,
            )
        else:
            self.attn = None
        # A hook to migrate legacy attention module
        self.register_load_state_dict_pre_hook(self._migrate_attention_module)

    def forward(self, x, emb):
        with (
            nvtx.annotate(message="UNetBlock", color="purple")
            if self.profile_mode
            else contextlib.nullcontext()
        ):
            orig = x
            x = self.conv0(self.norm0(x))
            params = self.affine(emb).unsqueeze(2).unsqueeze(3)
            _validate_amp(self.amp_mode)
            if not self.amp_mode:
                if params.dtype != x.dtype:
                    params = params.to(x.dtype)  # type: ignore

            if self.adaptive_scale:
                scale, shift = params.chunk(chunks=2, dim=1)
                x = silu(torch.addcmul(shift, self.norm1(x), scale + 1))
            else:
                x = self.norm1(x.add_(params))

            x = self.conv1(
                torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            )
            x = x.add_(self.skip(orig) if self.skip is not None else orig)
            x = x * self.skip_scale

            if self.attn:
                x = self.attn(x)
                x = x * self.skip_scale
            return x

    def __setattr__(self, name, value):
        """Prevent setting attributes with reserved names.

        Parameters
        ----------
        name : str
            Attribute name.
        value : Any
            Attribute value.
        """
        if name in getattr(self.__class__, "_reserved_attributes", set()):
            raise AttributeError(f"Attribute '{name}' is reserved and cannot be set.")
        super().__setattr__(name, value)

    @staticmethod
    def _migrate_attention_module(
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """``load_state_dict`` pre-hook that handles legacy checkpoints that
        stored attention layers at root.

        The earliest versions of ``UNetBlock`` stored the attention-layer
        parameters directly on the block using attribute names contained in
        ``_reserved_attributes``.  These have since been moved under the
        dedicated ``attn`` sub-module.  This helper migrates the parameter
        names so that older checkpoints can still be loaded.
        """

        _mapping = {
            f"{prefix}norm2.weight": f"{prefix}attn.norm.weight",
            f"{prefix}norm2.bias": f"{prefix}attn.norm.bias",
            f"{prefix}qkv.weight": f"{prefix}attn.qkv.weight",
            f"{prefix}qkv.bias": f"{prefix}attn.qkv.bias",
            f"{prefix}proj.weight": f"{prefix}attn.proj.weight",
            f"{prefix}proj.bias": f"{prefix}attn.proj.bias",
        }

        for old_key, new_key in _mapping.items():
            if old_key in state_dict:
                # NOTE: Only migrate if destination key not already present to
                # avoid accidental overwriting when both are present.
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
                else:
                    raise ValueError(
                        f"Checkpoint contains both legacy and new keys for {old_key}"
                    )
