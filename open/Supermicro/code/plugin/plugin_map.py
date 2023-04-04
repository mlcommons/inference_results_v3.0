# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import ctypes

from code.common.constants import *
from code.common.systems.system_list import SystemClassifications
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import Callable, List


@dataclass
class LoadablePlugin:
    """
    Dataclass which describes a loadable TensorRT plugin, with constraints
    """

    path: str
    """str: Path to the TRT plugin library"""

    constraints: List[Callable[[], bool]] = field(default_factory=lambda: list())
    """List[Callable[[], bool]: list of constraints that describes whether the plugin can be loaded """

    def get_full_path(self):
        return os.path.join("build", "plugins", self.path)

    def load(self, args: dict):
        if self.can_load(args):
            print(f"Loading TensorRT plugin from {self.get_full_path()}")
            ctypes.CDLL(self.get_full_path())

    def can_load(self, args: dict):
        for constraint in self.constraints:
            if not constraint(args):
                return False
        return True


class LoadablePlugins(Enum):
    PixelShuffle3DPlugin = LoadablePlugin("pixelShuffle3DPlugin/libpixelshuffle3dplugin.so")
    Conv3d1x1x1k4Plugin = LoadablePlugin("conv3D1X1X1K4Plugin/libconv3D1X1X1K4Plugin.so")
    Conv3d3x3x3c1k32Plugin = LoadablePlugin("conv3D3X3X3C1K32Plugin/libconv3D3X3X3C1K32Plugin.so")
    # TODO: Need other specifier than Hopper
    Bertfp8Plugin = LoadablePlugin("../FasterTransformer/build/lib/libbert_fp8_plugin.so", [lambda x: SystemClassifications.is_hopper() or SystemClassifications.is_ada(), lambda args: args.get('use_fp8', False)])
    DLRMInteractionsPlugin = LoadablePlugin("DLRMInteractionsPlugin/libdlrminteractionsplugin.so")
    NMSOptPlugin = LoadablePlugin("NMSOptPlugin/libnmsoptplugin.so")
    RetinaNetConcatOutputPlugin = LoadablePlugin("retinanetConcatPlugin/libretinanetconcatplugin.so")
    RNNTOptPlugin = LoadablePlugin("RNNTOptPlugin/librnntoptplugin.so")


base_plugin_map = {
    Benchmark.UNET3D: [LoadablePlugins.PixelShuffle3DPlugin, LoadablePlugins.Conv3d1x1x1k4Plugin, LoadablePlugins.Conv3d3x3x3c1k32Plugin],
    Benchmark.BERT: [LoadablePlugins.Bertfp8Plugin],
    Benchmark.DLRM: [LoadablePlugins.DLRMInteractionsPlugin],
    Benchmark.RNNT: [LoadablePlugins.RNNTOptPlugin],
    Benchmark.ResNet50: [],
    Benchmark.Retinanet: [LoadablePlugins.NMSOptPlugin, LoadablePlugins.RetinaNetConcatOutputPlugin],
}
