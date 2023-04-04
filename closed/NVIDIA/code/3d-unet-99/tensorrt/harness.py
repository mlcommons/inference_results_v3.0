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

import os
from typing import Dict

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import pycuda
    import pycuda.autoprimaryctx

import code.common.arguments as common_args
from code.common import logging, dict_get, run_command, args_to_string
from code.common.harness import BaseBenchmarkHarness
from code.common.constants import Benchmark


class UNet3DKiTS19Harness(BaseBenchmarkHarness):
    """UNet3DKiTS19 harness."""

    def __init__(self, args: Dict, benchmark: Benchmark) -> None:
        super().__init__(args, benchmark)
        custom_flags = [
            "unet3d_sw_gaussian_patch_path",
        ]
        self.use_jemalloc = False
        self.flag_builder_custom_args = common_args.LOADGEN_ARGS +\
            common_args.LWIS_ARGS +\
            common_args.SHARED_ARGS +\
            custom_flags

    def _get_harness_executable(self) -> str:
        """Return path to UNet3DKiTS19 harness binary."""
        return "./build/bin/harness_3dunet"

    def _build_custom_flags(self, flag_dict: Dict) -> str:
        if self.has_dla:
            flag_dict["dla_engines"] = self.dla_engine

        if self.has_gpu and self.has_dla:
            pass
        elif self.has_gpu:
            flag_dict["max_dlas"] = 0
        elif self.has_dla:
            flag_dict["max_dlas"] = 1
        else:
            raise ValueError("Cannot specify --no_gpu and --gpu_only at the same time")

        # Deliver path to preconditioned Gaussian patch numpy file
        gaussian_patch_path_str = dict_get(self.args,
                                           "unet3d_sw_gaussian_patch_path",
                                           default="./build/preprocessed_data/KiTS19/etc/gaussian_patches.npy")
        patch_kernel_impl = dict_get(self.args, "slice_overlap_patch_kernel_cg_impl", False)

        flag_dict.update({
            "unet3d_sw_gaussian_patch_path": gaussian_patch_path_str,
            "slice_overlap_patch_kernel_cg_impl": patch_kernel_impl,
        })

        argstr = args_to_string(flag_dict) + " --scenario " + self.scenario.valstr() + " --model " + self.name

        return argstr
