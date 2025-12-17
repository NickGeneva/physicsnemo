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

from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

from physicsnemo import Module
from physicsnemo.core import ModelMetaData

from physicsnemo.diffusion.multi_diffusion import RandomPatching2D


# TODO: use version from diffusion package once refactor is complete
class EDMPreconditioner(Module):
    """
    Diffusion preconditioner wrapper based on EDM formulation.
    """

    def __init__(self, model: Module, sigma_data: float):
        super().__init__(meta=ModelMetaData())
        self.model = model
        self.sigma_data = sigma_data

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        condition: Dict[str, Tensor],
        **model_kwargs: Any,
    ):
        if not torch.compiler.is_compiling():
            B = x.shape[0]
            if t.shape != (B,):
                raise ValueError(
                    f"Expected t to be of shape (B,), but got {t.shape} instead."
                    f"Expected tensor of shape (B, D) but got tensor of shape {x.shape}"
                )
            if not all(v.shape[0] == B for v in condition.values()):
                raise ValueError(
                    f"Expected all condition values to be of shape (B, *), "
                    f"but got {condition.values()} instead."
                )

        sigma = t

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.log() / 4

        return c_skip * x + c_out * self.model(
            c_in * x, c_noise, condition, **model_kwargs
        )


# TODO: use version from diffusion package once refactor is complete
class DiffusionAdapter(Module):
    """
    Adapter to make a model callable with the correct
    signature ``forward(x, t, condition, **model_kwargs) -> torch.Tensor``.
    """

    def __init__(
        self,
        model: Module,
    ):
        super().__init__(meta=ModelMetaData())
        self.model = model

    def forward(self, x, t, condition, **model_kwargs):
        # NOTE: this is hardcoded for the SongUNetPosEmbd model to make it
        # simpler here.
        return self.model(x, t, None, **model_kwargs)


# TODO: use version from diffusion package once refactor is complete
class EDMLoss:
    """
    EDM loss function for conditional and unconditional multi-diffusion.

    Parameters
    ----------
    model : Callable[[Tensor, Tensor, Dict[str, Tensor], *Any], Tensor]
        The model to compute the loss for.
    P_mean : float, optional
        Mean value for the noise level.
    P_std : float, optional
        Standard deviation for the noise level.
    sigma_data : float, optional
        Standard deviation for the data.
    """

    def __init__(
        self,
        model: Callable[[Tensor, Tensor, Dict[str, Tensor], *Any], Tensor],
        P_mean: float = 0.0,
        P_std: float = 1.2,
        sigma_data: float = 0.5,
        patching: Optional[RandomPatching2D] = None,
    ):
        self.model = model
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.patching = patching

    def get_noise_params(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Method to compute the noise parameters for the EDM loss.

        Parameters
        ----------
        x : torch.Tensor
            Latent state of shape :math:`(B, *)`.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            - Noise ``n`` of shape :math:`(B, 1, ..., 1)` to be added to the latent state.
            - Noise level ``sigma`` of shape :math:`(B,)`.
            - Weight ``weight`` of shape :math:`(B, 1, ..., 1)` to multiply the loss.
        """
        # Sample noise level
        rnd_normal = torch.randn([x.shape[0]], device=x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        # Loss weight
        weight = (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2
        # Sample noise
        n = torch.randn_like(x) * sigma
        return (
            n.view(x.shape[0], *((1,) * (x.ndim - 1))),
            sigma.view(x.shape[0], *((1,) * (x.ndim - 1))),
            weight.view(x.shape[0], *((1,) * (x.ndim - 1))),
        )

    def __call__(
        self,
        x: Tensor,  # (B, C, H, W)
        condition: Dict[str, Tensor],
        **model_kwargs: Any,
    ) -> Tensor:
        """
        Calculate and return the loss corresponding to the EDM formulation.
        """

        if self.patching:
            # (P * B, C, H_p, W_p)
            x = self.patching.apply(x)
            # TODO: adapt for conditional multi-diffusion
            # Patched conditioning on y_lr and interp(img_lr)
            # (batch_size * patch_num, 2*c_in, patch_shape_y, patch_shape_x)
            y_lr_patched = self.patching.apply(input=y_lr, additional_input=img_lr)

        # Compute noise parameters
        n, sigma, weight = self.get_noise_params(x)

        x_0 = self.model(
            x + n,
            sigma,
            condition,
            global_index=(
                self.patching.global_index(x.shape[0], x.device)
                if self.patching is not None
                else None
            ),
            **model_kwargs,
        )
        return weight * ((x_0 - x) ** 2)
