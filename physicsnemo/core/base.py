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

"""Minimal base to break import cycles between registry and module.

This defines a lightweight base class that can be used by the registry for
issubclass checks without importing the full `Module` implementation.
"""

from __future__ import annotations

from abc import ABC

import torch


class RegisterableModule(torch.nn.Module, ABC):
    """Marker base class for models that can be registered.

    The concrete `physicsnemo.core.Module` should subclass this type.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if self.__class__ is RegisterableModule:
            raise TypeError(
                "RegisterableModule is an internal base and cannot be instantiated directly."
            )


__all__ = ["RegisterableModule"]
