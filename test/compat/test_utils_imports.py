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

"""
Compatibility layer tests for physicsnemo v1 import paths.

The compat layer allows:
>>> import physicsnemo.compat as physicsnemo
>>> # Old paths like `physicsnemo.utils.filesystem` resolve to `physicsnemo.core.filesystem`

NOTE

These test should expire with the compat layer and be removed at the same time.

"""

import importlib
import sys

import pytest

migrations = {
    "physicsnemo.utils.filesystem": "physicsnemo.core.filesystem",
    "physicsnemo.utils.version_check": "physicsnemo.core.version_check",
    "physicsnemo.models.meta": "physicsnemo.core.meta",
    "physicsnemo.models.module": "physicsnemo.core.module",
    "physicsnemo.utils.neighbors": "physicsnemo.nn.neighbors",
    "physicsnemo.utils.sdf": "physicsnemo.nn.sdf",
}


def _clear_physicsnemo_modules():
    """
    Remove relevant modules from sys.modules so each test can import fresh.
    """
    for name in list(sys.modules.keys()):
        if name == "physicsnemo" or name.startswith("physicsnemo."):
            sys.modules.pop(name, None)


@pytest.mark.parametrize("old_name", migrations.keys())
def test_old_utils_import_fails_without_compat(old_name, monkeypatch):
    # Ensure compat is not enabled via env var
    monkeypatch.delenv("PHYSICSNEMO_ENABLE_COMPAT", raising=False)
    _clear_physicsnemo_modules()

    # Import base package without compat side effects
    importlib.import_module("physicsnemo")

    # Old path should fail without compat
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(old_name)


@pytest.mark.parametrize("old_name, new_name", migrations.items())
def test_old_utils_import_works_with_env_compat(old_name, new_name, monkeypatch):
    # Enable via env var before first import (compat installs at import-time)
    monkeypatch.setenv("PHYSICSNEMO_ENABLE_COMPAT", "1")
    _clear_physicsnemo_modules()

    # Import emits a deprecation warning when installing aliases
    with pytest.warns(DeprecationWarning):
        importlib.import_module("physicsnemo")

    fs_old = importlib.import_module(old_name)
    fs_new = importlib.import_module(new_name)
    assert fs_old is fs_new
