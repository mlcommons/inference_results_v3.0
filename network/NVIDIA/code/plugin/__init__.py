# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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
import os

from code.common.constants import *
from code.plugin.plugin_map import base_plugin_map


def load_trt_plugin_by_network(network_name: str) -> None:
    """
    Load all loadable plugins for a given network from plugin map
    """
    benchmark = Benchmark.get_match(network_name)
    for plugin in base_plugin_map[benchmark]:
        plugin.value.load()


def get_trt_plugin_paths_by_network(network_name: str):
    """
    Return a list of loadable plugin paths for the given network
    """
    benchmark = Benchmark.get_match(network_name)
    plugins = []
    for plugin in base_plugin_map[benchmark]:
        if plugin.value.can_load():
            plugins.append(plugin.value.get_full_path())

    return plugins
