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

##
# Plugin .so files have to be loaded at global scope and before `import torch` to avoid cuda version mismatch.
##
from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("3d-unet-kits")

import ctypes
import os
import struct
from importlib import import_module
from typing import Dict

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import tensorrt as trt
    import onnx
    import onnx_graphsurgeon as gs

from code.common import dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.utils import get_dyn_ranges

UNet3DKiTS19MinMaxCalibrator = import_module("code.3d-unet.tensorrt.calibrator").UNet3DKiTS19MinMaxCalibrator


class UnetBuilder(TensorRTEngineBuilder):
    """Build and calibrate UNet network."""

    def __init__(self, args: Dict) -> None:
        workspace_size = dict_get(args, "workspace_size", default=(8 << 30))

        super().__init__(args, Benchmark.UNET3D, workspace_size=workspace_size)

        # input channel
        self.num_input_channel = 1

        # input volume dimension == KiTS19 ROI size, determined from training
        self.input_volume_dim = [128, 128, 128]

        self.use_plugin_optimizations = self.precision in ["int8", "fp32"]

        # use InstNorm3D plugin
        self.use_instnorm3d_plugin = True
        # use pixelShuffle plugin
        self.enable_pixelshuffle3d_plugin = self.use_plugin_optimizations
        self.enable_pixelshuffle3d_plugin_concat_fuse = self.use_plugin_optimizations
        # Deconv->Conv conversion
        self.use_conv_for_deconv = self.use_plugin_optimizations
        self.pixel_shuffle_cdwh = True  # If false, do dhwc
        # use last layer plugin
        self.use_conv3d1x1x1k4_plugin = self.use_plugin_optimizations
        # use first layer plugin
        self.use_conv3d3x3x3c1k32_plugin = self.use_plugin_optimizations

        # Model is imported from ONNX
        self.model_path = dict_get(args, "model_path", default="build/models/3d-unet-kits19/3dUNetKiTS19.onnx")
        self.export_graphsurgeoned_model = False

        force_calibration = dict_get(self.args, "force_calibration", default=False)

        # Calibrator
        if self.precision == "int8" or force_calibration:
            self.apply_flag(trt.BuilderFlag.INT8)
            preprocessed_data_dir = os.path.join(dict_get(self.args, "preprocessed_data_dir", default="build/preprocessed_data"),
                                                 "KiTS19", "calibration")

            calib_batch_size = dict_get(self.args, "calib_batch_size", default=1)
            calib_max_batches = dict_get(self.args, "calib_max_batches", default=20)

            calib_data_map = dict_get(self.args, "calib_data_map", default="data_maps/kits19/cal_map.txt")
            calib_volume_dir = os.path.join(preprocessed_data_dir, "fp32")

            cache_file = dict_get(self.args, "cache_file", default="code/3d-unet/tensorrt/calibrator.cache")

            self.calibrator = UNet3DKiTS19MinMaxCalibrator(calib_volume_dir, cache_file,
                                                           calib_batch_size, calib_max_batches,
                                                           force_calibration, calib_data_map)
            assert self.calibrator, "Calibrator is not init'ed"
            assert self.calibrator.get_algorithm() == trt.CalibrationAlgoType.MINMAX_CALIBRATION, "Calibrator type is not MinMax"
            self.builder_config.int8_calibrator = self.calibrator
            self.cache_file = cache_file

        # TRT builder flag
        if self.precision == "fp16":
            self.apply_flag(trt.BuilderFlag.FP16)
        elif self.precision == "int8":
            self.apply_flag(trt.BuilderFlag.FP16)
            self.apply_flag(trt.BuilderFlag.INT8)

    def preprocess_onnx(self, model: str) -> gs.Graph:
        """
        Manipulate original ONNX file with graphSurgeon: insert InstanceNormalization
        3D and PixelShuffle plugin, and export the new ONNX graph.
        """
        graph = gs.import_onnx(model)
        if self.use_instnorm3d_plugin:
            for node in graph.nodes:
                # Replace InstanceNormalization with InstanceNormalization_TRT plugin node
                if node.op == "InstanceNormalization":
                    node.op = "InstanceNormalization_TRT"
                    node.attrs["scales"] = node.inputs[1]
                    node.attrs["bias"] = node.inputs[2]
                    node.attrs["plugin_version"] = "2"
                    node.attrs["plugin_namespace"] = ""
                    node.attrs["relu"] = 0
                    node.attrs["alpha"] = 0.0
                    scales = node.attrs["scales"].values
                    biases = node.attrs["bias"].values
                    assert len(scales) == len(biases), "Scales and biases do not have the same length!"
                    del node.inputs[2]
                    del node.inputs[1]

            # Set relu node attributes to INSTNORM3D plugin and remove relu nodes.
            nodes = [node for node in graph.nodes if node.op == "InstanceNormalization_TRT"]
            for node in nodes:
                relu_node = node.o()
                node.attrs["relu"] = 1
                node.attrs["alpha"] = 0.0
                node.outputs = relu_node.outputs
                relu_node.outputs.clear()

        if self.use_conv3d1x1x1k4_plugin:
            nodes = [node for node in graph.nodes if node.op == "InstanceNormalization_TRT"]
            last_layer_node = nodes[-1].o()
            last_layer_node.op = "CONV3D1X1X1K4_TRT"
            assert len(last_layer_node.inputs) == 3, "Weight and Bias needed"
            weights = last_layer_node.inputs[1]
            weights_shape = weights.values.shape
            weights_c = weights_shape[1]
            weights_k = weights_shape[0]
            assert weights_shape == (3, 32, 1, 1, 1), "Expecting c == 32 and k == 3"
            bias = last_layer_node.inputs[2]
            bias_shape = bias.values.shape
            assert bias_shape[0] == weights_k, "Unexpected Bias shape"
            last_layer_node.attrs["inputChannels"] = weights_c
            last_layer_node.attrs["weights"] = weights
            last_layer_node.attrs["bias"] = bias
            last_layer_node.attrs["plugin_version"] = "1"
            last_layer_node.attrs["plugin_namespace"] = ""
            del last_layer_node.inputs[2]
            del last_layer_node.inputs[1]

            # add the identity layer, since the last layer is quantized
            identity_out = gs.Variable("output", dtype=np.float32)
            identity = gs.Node(op="Identity", inputs=last_layer_node.outputs, outputs=[identity_out])
            graph.nodes.append(identity)
            graph.outputs.append(identity_out)
            last_layer_node.outputs[0].name = "conv3d1x1x1k4_out"

        if self.use_conv3d3x3x3c1k32_plugin:
            nodes = [node for node in graph.nodes if node.op == "Conv"]
            first_layer_node = nodes[0]
            first_layer_node.op = "CONV3D3X3X3C1K32_TRT"
            assert len(first_layer_node.inputs) == 2, "Weight needed"
            weights = first_layer_node.inputs[1]
            weights_shape = weights.values.shape
            weights_c = weights_shape[1]
            weights_k = weights_shape[0]
            assert weights_shape == (32, 1, 3, 3, 3), "Expecting c == 1 and k == 32"
            first_layer_node.attrs["inputChannels"] = weights_c
            first_layer_node.attrs["weights"] = weights
            first_layer_node.attrs["plugin_version"] = "1"
            first_layer_node.attrs["plugin_namespace"] = ""
            del first_layer_node.inputs[1]

        # Convert Deconv to Conv + PixelShuffle
        if self.use_conv_for_deconv:
            added_nodes = []
            input_d = graph.inputs[0].shape[2]
            input_h = graph.inputs[0].shape[3]
            input_w = graph.inputs[0].shape[4]

            # We start the conversion from the lowest dimension
            current_d = input_d // 32
            current_h = input_h // 32
            current_w = input_w // 32

            for (node_idx, node) in enumerate(graph.nodes):
                if node.op == "ConvTranspose":
                    name = node.name
                    node.op = "Conv"
                    assert node.attrs["kernel_shape"] == [2, 2, 2], "The conversion only makes sense for 2x2x2 deconv"
                    node.attrs["kernel_shape"] = [1, 1, 1]
                    assert node.attrs["strides"] == [2, 2, 2], "The conversion only makes sense for stride=2x2x2 deconv"
                    node.attrs["strides"] = [1, 1, 1]

                    # Transpose weights from cktrs to (ktrs)c111 or (trsk)c111
                    assert len(node.inputs) >= 2, "Weights needed"
                    weights = node.inputs[1]
                    weights_shape = weights.values.shape
                    weights_c = weights_shape[0]
                    weights_k = weights_shape[1]
                    assert weights_shape[2:] == (2, 2, 2), "The conversion only makes sense for 2x2x2 deconv"
                    weights_transpose_axes = (1, 2, 3, 4, 0) if self.pixel_shuffle_cdwh else (2, 3, 4, 1, 0)
                    weights.values = weights.values.transpose(weights_transpose_axes).reshape(weights_k * 8, weights_c, 1, 1, 1)

                    # Check bias sanity
                    assert len(node.inputs) == 3, "Bias needed"
                    bias = node.inputs[2]
                    bias_shape = bias.values.shape
                    assert bias_shape[0] == weights_k, "Unexpected Bias shape"
                    bias.values = bias.values.repeat(8).reshape([1, weights_k, 2, 2, 2]).transpose(weights_transpose_axes).reshape(weights_k * 8)

                    deconv_output = node.outputs[0]
                    concat_node = graph.nodes[node_idx + 1]
                    assert concat_node.op == "Concat", "Cannot find the right Concat node"
                    if self.enable_pixelshuffle3d_plugin:
                        # Insert PixelShuffle
                        pixel_shuffle_output = gs.Variable(name + "_pixelshuffle_plugin_out")
                        pixel_shuffle_node = gs.Node(
                            "PIXELSHUFFLE3D_TRT", name + "_pixelshuffle_plugin",
                            {}, [deconv_output], [pixel_shuffle_output])
                        pixel_shuffle_node.op = "PIXELSHUFFLE3D_TRT"
                        pixel_shuffle_node.attrs["R"] = 2
                        pixel_shuffle_node.attrs["S"] = 2
                        pixel_shuffle_node.attrs["T"] = 2
                        pixel_shuffle_node.attrs["plugin_version"] = "1"
                        pixel_shuffle_node.attrs["plugin_namespace"] = ""
                        assert concat_node.inputs[0] is deconv_output, "Wrong concat order"
                        if self.enable_pixelshuffle3d_plugin_concat_fuse:
                            pixel_shuffle_node.outputs = concat_node.outputs
                            pixel_shuffle_node.inputs.append(concat_node.inputs[1])
                            concat_node.outputs.clear()
                        else:
                            concat_node.inputs[0] = pixel_shuffle_output
                        added_nodes.extend([pixel_shuffle_node])
                    else:
                        reshape1_shape = [0, weights_k, 2, 2, 2, current_d, current_h, current_w] if self.pixel_shuffle_cdwh else\
                                         [0, 2, 2, 2, weights_k, current_d, current_h, current_w]
                        shuffle_axes = [0, 1, 5, 2, 6, 3, 7, 4] if self.pixel_shuffle_cdwh else [0, 4, 5, 1, 6, 2, 7, 3]
                        current_d *= 2
                        current_h *= 2
                        current_w *= 2
                        reshape2_shape = [0, weights_k, current_d, current_h, current_w]
                        reshape1_shape_const = gs.Constant(name + "_pixelshuffle_reshape1_shape", np.array(reshape1_shape, dtype=np.int32))
                        reshape2_shape_const = gs.Constant(name + "_pixelshuffle_reshape2_shape", np.array(reshape2_shape, dtype=np.int32))
                        reshape1_output = gs.Variable(name + "_pixelshuffle_reshape1_out")
                        shuffle_output = gs.Variable(name + "_pixelshuffle_shuffle_out")
                        reshape2_output = gs.Variable(name + "_pixelshuffle_reshape2_out")
                        reshape1_node = gs.Node(
                            "Reshape", name + "_pixelshuffle_reshape1",
                            {}, [deconv_output, reshape1_shape_const], [reshape1_output])
                        shuffle_node = gs.Node(
                            "Transpose", name + "_pixelshuffle_transpose",
                            {"perm": shuffle_axes}, [reshape1_output], [shuffle_output])
                        reshape2_node = gs.Node(
                            "Reshape", name + "_pixelshuffle_reshape2",
                            {}, [shuffle_output, reshape2_shape_const], [reshape2_output])
                        assert concat_node.inputs[0] is deconv_output, "Wrong concat order"
                        concat_node.inputs[0] = reshape2_output
                        added_nodes.extend([reshape1_node, shuffle_node, reshape2_node])
            graph.nodes.extend(added_nodes)

        # Remove the four unnecessary outputs.
        graph.outputs = [output for output in graph.outputs if output.name == "output"]

        # Remove dead nodes.
        graph.cleanup().toposort()

        # Add names to the layer after the graph is topsorted.
        uniq_num = 0
        for node in graph.nodes:
            if not node.name or node.name.isdigit():
                op_name = str(node.op)
                node.name = f'gs_{op_name}_{uniq_num}'
                node.attrs['name'] = node.name
                uniq_num += 1
            for out_idx, out_tensor in enumerate(node.outputs):
                postfix = "_" + out_idx if len(node.outputs) > 1 else ""
                if not out_tensor.name or out_tensor.name.isdigit():
                    out_tensor.name = node.name + "__output" + postfix

        surgeoned_graph = gs.export_onnx(graph)

        # If needed, save the graph surgoned onnx
        if self.export_graphsurgeoned_model:
            onnx.save(surgeoned_graph, "GraphSurgeoned_3dunet_kits19.onnx")

        # return the surgeoned graph
        return surgeoned_graph

    def initialize(self) -> None:
        """
        Parse the processed model to create the network.
        """
        # Create network.
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # if calibration cache is not found
        need_calibration = False

        # Input shape
        channel_idx = 1
        input_tensor_dim = [-1] + self.input_volume_dim
        input_tensor_dim.insert(channel_idx, self.num_input_channel)

        # Parse from onnx file.
        parser = trt.OnnxParser(self.network, self.logger)
        model = self.preprocess_onnx(onnx.load(self.model_path))
        success = parser.parse(onnx._serialize(model))
        if not success:
            err_desc = parser.get_error(0).desc()
            raise RuntimeError(f"3D-UNet onnx model parsing failed! Error: {err_desc}")

        # Set input/output tensor dtype and formats
        input_tensor = self.network.get_input(0)
        output_tensor = self.network.get_output(0)

        input_tensor.shape = input_tensor_dim

        # Use Linear input if INT8 / FP32 always Linear
        assert self.input_dtype == "int8" and self.input_format == "linear",\
            "3D-UNet has to be built with INT8 LINEAR input"

        input_tensor.dtype = trt.int8
        dynamic_range_dict = dict()
        if os.path.exists(self.cache_file):
            dynamic_range_dict = get_dyn_ranges(self.cache_file)
            input_dr = dynamic_range_dict.get("input", 0)
            input_tensor.set_dynamic_range(-input_dr, input_dr)
        else:
            print("WARNING: Calibration cache file not found! Calibration is required.")
            need_calibration = True

        input_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

        # Always use FP16 LINEAR output
        # workaround for calibration not working with the identity layer properly, with last layer plugin
        force_calibration = dict_get(self.args, "force_calibration", default=need_calibration)
        output_tensor.dtype = trt.float32 if force_calibration else trt.float16
        output_tensor.allowed_formats = 1 << int(trt.TensorFormat.LINEAR)

        self.initialized = True
