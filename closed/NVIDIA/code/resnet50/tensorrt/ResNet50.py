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

import tensorrt as trt
import os
import platform
import onnx

from collections import namedtuple
from importlib import import_module
from code.common import logging, dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.systems.system_list import DETECTED_SYSTEM

RN50Calibrator = import_module("code.resnet50.tensorrt.calibrator").RN50Calibrator
RN50GraphSurgeon = import_module("code.resnet50.tensorrt.rn50_graphsurgeon").RN50GraphSurgeon


class ResNet50(TensorRTEngineBuilder):
    """Resnet50 engine builder."""
    # subnetwork tuple including name, batch_size
    subnetwork = namedtuple('subnetwork', ['name', 'batch_size'])
    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(1 << 30))
        logging.info(f"Using workspace size: {workspace_size}")

        super().__init__(args, Benchmark.ResNet50, workspace_size=workspace_size)

        # Model path
        self.model_path = dict_get(args, "model_path", default="build/models/ResNet50/resnet50_v1.onnx")

        self.cache_file = None
        self.need_calibration = False
        self.subnetworks = None

        if self.precision == "int8":
            # Get calibrator variables
            calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=500)
            force_calibration = dict_get(self.args, "force_calibration", default=False)
            cache_file = dict_get(self.args, "cache_file", default="code/resnet50/tensorrt/calibrator.cache")
            preprocessed_data_dir = dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data")
            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/imagenet/cal_map.txt")
            calib_image_dir = os.path.join(preprocessed_data_dir, "imagenet/ResNet50/fp32")

            # Set up calibrator
            self.calibrator = RN50Calibrator(calib_batch_size=calib_batch_size, calib_max_batches=calib_max_batches,
                                             force_calibration=force_calibration, cache_file=cache_file,
                                             image_dir=calib_image_dir, calib_data_map=calib_data_map)
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file
            self.need_calibration = force_calibration or not os.path.exists(cache_file)

    def _parse_subnetwork_loop(self):
        if self.device_type == "gpu":
            # Parse res3 subnetwork loop
            self.gpu_res2res3_loop_count = self.args.get("gpu_res2res3_loop_count", 1)
            if self.batch_size % self.gpu_res2res3_loop_count != 0:
                raise ValueError(f"batch_size must be divisible by gpu_res2res3_loop_count")
            if self.gpu_res2res3_loop_count < 1:
                raise ValueError(f"gpu_res2res3_loop_count must be at least 1")
            if self.gpu_res2res3_loop_count > 1:
                logging.info(f"Using batch splitting for res2_3 subnetwork: {self.name}-{self.scenario}")
                logging.info(f"Looping over res2_3 subnetwork for {self.gpu_res2res3_loop_count} times to process {self.gpu_res2res3_loop_count}x{self.batch_size // self.gpu_res2res3_loop_count}={self.batch_size} batches")
                self.subnetworks = [
                    self.subnetwork('preres2', self.batch_size),
                    self.subnetwork('res2_3', self.batch_size // self.gpu_res2res3_loop_count),
                    self.subnetwork('postres3', self.batch_size)]

        if self.device_type == "dla":
            # Parse DLA subnetwork loop
            self.dla_loop_count = self.args.get("dla_loop_count", 1)
            if self.dla_loop_count < 1:
                raise ValueError(f"dla_loop_count must be at least 1")
            if self.dla_loop_count > 1:
                logging.info(f"Using DLA loop method for: {self.name}-{self.scenario}")
                logging.info(f"Looping over DLA subnetwork for {self.dla_loop_count} times to process {self.dla_loop_count}x{self.batch_size}={self.dla_loop_count * self.batch_size} batches")
                self.subnetworks = [
                    self.subnetwork('dla', self.batch_size),
                    self.subnetwork('topk', self.batch_size * self.dla_loop_count)]

    def _discard_topk_output_value(self):
        """
        Unmark topk_layer_output_value, just leaving topk_layer_output_index
        """

        assert self.network.num_outputs == 2, "Two outputs expected"
        assert self.network.get_output(0).name == "topk_layer_output_value",\
            f"unexpected tensor: {self.network.get_output(0).name}"
        assert self.network.get_output(1).name == "topk_layer_output_index",\
            f"unexpected tensor: {self.network.get_output(1).name}"
        logging.info(f"Unmarking output: {self.network.get_output(0).name}")
        self.network.unmark_output(self.network.get_output(0))

    def _set_input_tensor_type(self, input_tensor_name, use_dla=None, tensor_format=None, dynamic_range=None):
        """
        Set input tensor dtype and format
        """

        input_tensor = self.network.get_input(0)
        assert input_tensor.name == input_tensor_name,\
            f"input_tensor_name: {input_tensor_name} does not match network input {input_tensor.name}"

        if self.input_dtype == "int8":
            input_tensor.dtype = trt.int8
            if dynamic_range is not None:
                input_tensor.dynamic_range = dynamic_range
        # Set tensor format
        if tensor_format is not None:
            input_tensor.allowed_formats = 1 << int(tensor_format)
        # Set the same format as the input data if not specified
        else:
            if self.input_format == "linear":
                input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            elif self.input_format == "chw4":
                # WAR for DLA reformat bug in https://nvbugs/3713387
                # For resnet50, inputs dims are [3, 224, 224]
                # For those particular dims, CHW4 == DLA_HWC4
                # so can use same CHW4 data for both GPU and DLA engine
                # By lying to TRT and saying input is DLA_HWC4,
                # we elide the pre-DLA reformat layer
                if use_dla is not None:
                    input_tensor.allowed_formats = 1 << int(trt.TensorFormat.DLA_HWC4)
                else:
                    input_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

    def _set_output_tensor_type(self, output_tensor_name, tensor_format=None, dynamic_range=None):
        """
        Set output tensors dtype and format
        For DLA looping method, the input format of the TopK engine has to match against the output format of the DLA engine
        """

        output_tensor = self.network.get_output(0)
        assert output_tensor.name == output_tensor_name,\
            f"output_tensor_name: {output_tensor_name} does not match network output {output_tensor.name}"

        if self.input_dtype == "int8":
            output_tensor.dtype = trt.int8
            if dynamic_range is not None:
                output_tensor.dynamic_range = dynamic_range
        # Set tensor format
        if tensor_format is not None:
            output_tensor.allowed_formats = 1 << int(tensor_format)
        # Set the same format as the input data if not specified
        else:
            if self.input_format == "linear":
                output_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)
            elif self.input_format == "chw4":
                output_tensor.allowed_formats = 1 << int(trt.TensorFormat.CHW4)

    def initialize(self, subnetwork_gs=None):
        """
        Parse input ONNX file to a TRT network. Apply layer optimizations and fusion plugins on network.
        """

        # Query system id for architecture
        self.compute_sm = DETECTED_SYSTEM.get_compute_sm()

        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Parse from onnx file.
        parser = trt.OnnxParser(self.network, self.logger)

        rn50_gs = RN50GraphSurgeon(self.model_path,
                                   self.compute_sm, self.device_type,
                                   self.precision,
                                   self.cache_file, self.need_calibration,
                                   self.args, subnetwork_gs)
        model = rn50_gs.process_onnx()
        success = parser.parse(onnx._serialize(model))
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"ResNet50 onnx model processing failed! Error: {err_desc}")

        # Setup the entire network for gpu engine and non-loop DLA engine
        if subnetwork_gs is None:
            self._discard_topk_output_value()
            self._set_input_tensor_type("input_tensor_0", self.dla_core)
        # Setup subnetwork running on DLA for DLA looping
        elif subnetwork_gs == "dla":
            self._set_input_tensor_type("input_tensor_0", self.dla_core, dynamic_range=(-128, 127))
            self._set_output_tensor_type("fc_replaced_out_0", dynamic_range=(-128, 127))
        # Setup topK subnetwork
        elif subnetwork_gs == "topk":
            self._discard_topk_output_value()
            self._set_input_tensor_type("fc_replaced_out_0", dynamic_range=(-128, 127))
        # Setup PreRes2 subnetwork
        elif subnetwork_gs == "preres2":
            self._set_input_tensor_type("input_tensor_0", dynamic_range=(-128, 127))
            self._set_output_tensor_type("pool1_out_0", tensor_format=trt.TensorFormat.CHW32)
        # Setup PreRes3 subnetwork
        elif subnetwork_gs == "preres3":
            self._set_input_tensor_type("input_tensor_0", dynamic_range=(-128, 127))
            self._set_output_tensor_type("res2c_relu_out_0", tensor_format=trt.TensorFormat.CHW32)
        # Setup Res2_3 subnetwork
        elif subnetwork_gs == "res2_3":
            self._set_input_tensor_type("pool1_out_0", tensor_format=trt.TensorFormat.CHW32)
            self._set_output_tensor_type("res3d_relu_out_0", tensor_format=trt.TensorFormat.CHW32)
        # Setup Res3 subnetwork
        elif subnetwork_gs == "res3":
            self._set_input_tensor_type("res2c_relu_out_0", tensor_format=trt.TensorFormat.CHW32)
            self._set_output_tensor_type("res3d_relu_out_0", tensor_format=trt.TensorFormat.CHW32)
        # Setup PostRes3 subnetwork
        elif subnetwork_gs == "postres3":
            self._discard_topk_output_value()
            self._set_input_tensor_type("res3d_relu_out_0", tensor_format=trt.TensorFormat.CHW32)
        else:
            raise RuntimeError(f"ResNet50 initialize failed! Invalid subnetwork: {subnetwork_gs}")

        self.initialized = True

    def build_engines(self):
        self._parse_subnetwork_loop()
        # For non-loop GPU engine and non-loop DLA engine, use the base function
        if self.subnetworks is None:
            super().build_engines()
        else:
            for subnetwork in self.subnetworks:
                self.initialize(subnetwork_gs=subnetwork.name)
                self.batch_size = subnetwork.batch_size
                engine_name = self._get_engine_fpath(f"{self.device_type}-{subnetwork.name}", self.batch_size)
                super().build_engines(engine_name=engine_name)
