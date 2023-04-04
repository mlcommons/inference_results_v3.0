#!/usr/bin/env python3
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

import argparse
import ctypes
import os
import re
import onnx

# The plugin .so file has to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("retinanet")

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import pycuda.driver as cuda
    import pycuda.autoprimaryctx
    import tensorrt as trt

from code.common import logging, dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.utils import get_dyn_ranges
from code.common.systems.system_list import SystemClassifications, DETECTED_SYSTEM
from importlib import import_module

RetinanetEntropyCalibrator = import_module("code.retinanet.tensorrt.calibrator").RetinanetEntropyCalibrator
RetinanetGraphSurgeon = import_module("code.retinanet.tensorrt.retinanet_graphsurgeon").RetinanetGraphSurgeon

INPUT_SHAPE = (3, 800, 800)


class Retinanet(TensorRTEngineBuilder):
    def __init__(self, args):
        # Retinanet need a bigger workspace
        workspace_size = dict_get(args, "workspace_size", default=(8 << 30))
        logging.info(f"Using workspace size: {workspace_size}")
        super().__init__(args, Benchmark.Retinanet, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/retinanet-resnext50-32x4d/retinanet-fpn.onnx")
        self.cache_file = None
        self.use_nmsopt = dict_get(args, "use_nmsopt", default=True)

        if self.precision == "int8":
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=10)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=50)
            cache_file = dict_get(self.args, "cache_file", default="code/retinanet/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/open-images-v6-mlperf/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "open-images-v6-mlperf/calibration/Retinanet/fp32")

            self.calibrator = RetinanetEntropyCalibrator(calib_image_dir, cache_file, calib_batch_size,
                                                         calib_max_batches, force_calibration, calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)

    def initialize(self):
        # Create network.
        network_creation_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        self.network = self.builder.create_network(network_creation_flag)

        parser = trt.OnnxParser(self.network, self.logger)
        self.compute_sm = DETECTED_SYSTEM.get_compute_sm()
        retinanet_gs = RetinanetGraphSurgeon(
            onnx_path=self.model_path,
            compute_sm=self.compute_sm,
            precision=self.precision,
            device_type=self.device_type,
            cache_file=self.cache_file,
            need_calibration=self.need_calibration,
            nms_type='nmsopt' if self.use_nmsopt else 'efficientnms',
            retinanet_args=self.args,
            subnetwork_gs=None  # TODO: to enable the subnetwork for DLA
        )
        model = retinanet_gs.process_onnx()
        success = parser.parse(onnx._serialize(model))
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"Retinanet onnx model processing failed! Error: {err_desc}")

        # Set input dtype and format
        input_tensor = self.network.get_input(0)
        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            dynamic_range_dict = dict()
            if os.path.exists(self.cache_file):
                dynamic_range_dict = get_dyn_ranges(self.cache_file)
                input_dr = dynamic_range_dict.get("images", -1)
                if input_dr == -1:
                    raise RuntimeError(f"Cannot find 'images' in the calibration cache. Exiting...")
                input_tensor.set_dynamic_range(-input_dr, input_dr)
            else:
                print("WARNING: Calibration cache file not found! Calibration is required")
        if self.input_format == "linear":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
        elif self.input_format == "chw4":
            input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

        self.initialized = True
