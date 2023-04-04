#! /usr/bin/env python3

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
import os
import argparse
import tensorrt as trt
from collections import namedtuple
from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np

import onnx
import onnx_graphsurgeon as gs

from code.common import logging, dict_get
from code.common.fields import Fields
from code.common.utils import get_dyn_ranges


@gs.Graph.register()
def AveragePool(self, name, in_t, attrs):
    """
    Create and add AveragePool op to Graph
    """
    out_t = self.layer(op="AveragePool", name=name,
                       inputs=[in_t], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def Reshape(self, name, in_t, shape, attrs=dict()):
    """
    Create and add Reshape op to Graph
    """
    out_t = self.layer(op="Reshape", name=name,
                       inputs=[in_t, shape], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def MatMul(self, name, A, B, attrs=dict()):
    """
    Create and add MatMul op to Graph
    """
    out_t = self.layer(op="MatMul", name=name,
                       inputs=[A, B], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def Conv(self, name, A, K, attrs):
    """
    Create and add Conv op to Graph
    """
    out_t = self.layer(op="Conv", name=name,
                       inputs=[A, K], outputs=[name],
                       attrs=attrs)[0]
    out_t.name = "{}_out_0".format(name)
    return out_t


@gs.Graph.register()
def TopK(self, name, in_t, attrs):
    """
    Create and add TopK op to Graph
    """
    out_t = self.layer(op="TopK", name=name,
                       inputs=[in_t], outputs=["{}_value".format(name),
                                               "{}_index".format(name)],
                       attrs=attrs)
    out_t[0].name = "{}_output_value".format(name)
    out_t[1].name = "{}_output_index".format(name)
    out_t[0].dtype = np.float32
    out_t[1].dtype = np.int32
    return out_t


@gs.Graph.register()
def RES2PLUGIN(self, plugin_op, plugin_name, plugin_in, plugin_out, attrs):
    """
    Create PLUGIN made for RES2; set plugin type by providing plugin_op
    """
    # NOTE: do NOT clear input tensor's output
    for _o in plugin_out:
        _o.inputs.clear()
    return self.layer(op=plugin_op, name=plugin_name,
                      inputs=plugin_in, outputs=plugin_out,
                      attrs=attrs)[0]


@gs.Graph.register()
def BETA1SmallKPlugin(self, plugin_op, plugin_name, plugin_in, plugin_out, attrs):
    """
    Create a plugin layer for beta=1 smallk.
    """
    for _o in plugin_out:
        _o.inputs.clear()
    return self.layer(op=plugin_op, name=plugin_name,
                      inputs=plugin_in, outputs=plugin_out,
                      attrs=attrs)[0]


class RN50GraphSurgeon(object):
    """
    Using ONNX Graph Surgeon, this class updates the ResNet50 ONNX graph for:
    1. Op and Tensor names
    2. Endpoint of RN50 to be more lightweight
    3. Fuse ops
    4. Set dynamic range of tensors if with quantization from calibration results
    """

    # put the name (str) of the method performing fusion in order to list
    # ex) "Beta1Smallk": ["fuse_beta1_conv"],
    #     this defines fusion whose policy: Beta1Smallk will happen by
    #     calling self.fuse_beta1_conv()
    # Expand the different fusion by defining the fusion method, then specify here
    # NOTE: order is important
    fusion_map = {
        "Res2Mega": ["fuse_res2_mega"],
        "Beta1Smallk": ["fuse_beta1_conv"],
        "NO_FUSE": [],
        "Unknown": [],
    }

    # put the name (str) of the class method performing op touchup in order to list
    op_touchup_map = {
        "generic": ["add_squeeze",
                    "add_fc",
                    "add_topk",
                    "remove_obsolete"],
        "ConvForFC": ["add_squeeze",
                      "add_conv",
                      "add_topk",
                      "remove_obsolete"],
        "Unknown": [],
    }

    # Map that maps original ONNX op name to conventional ResNet name
    op_name_map = {
        "resnet_model/conv2d/Conv2D": "conv1",
        "resnet_model/batch_normalization/FusedBatchNorm": "scale_conv1",
        "resnet_model/Relu": "conv1_relu",
        "resnet_model/max_pooling2d/MaxPool": "pool1",

        "Conv__128": "res2a_branch2a",
        "resnet_model/Relu_1": "res2a_branch2a_relu",
        "Conv__129": "res2a_branch2b",
        "resnet_model/Relu_2": "res2a_branch2b_relu",
        "Conv__130": "res2a_branch2c",
        "Conv__123": "res2a_branch1",
        "resnet_model/add": "res2a",
        "resnet_model/Relu_3": "res2a_relu",

        "Conv__131": "res2b_branch2a",
        "resnet_model/Relu_4": "res2b_branch2a_relu",
        "Conv__132": "res2b_branch2b",
        "resnet_model/Relu_5": "res2b_branch2b_relu",
        "Conv__133": "res2b_branch2c",
        "resnet_model/add_1": "res2b",
        "resnet_model/Relu_6": "res2b_relu",

        "Conv__138": "res2c_branch2a",
        "resnet_model/Relu_7": "res2c_branch2a_relu",
        "Conv__139": "res2c_branch2b",
        "resnet_model/Relu_8": "res2c_branch2b_relu",
        "Conv__140": "res2c_branch2c",
        "resnet_model/add_2": "res2c",
        "resnet_model/Relu_9": "res2c_relu",

        "Conv__145": "res3a_branch2a",
        "resnet_model/Relu_10": "res3a_branch2a_relu",
        "Conv__146": "res3a_branch2b",
        "resnet_model/Relu_11": "res3a_branch2b_relu",
        "Conv__147": "res3a_branch2c",
        "Conv__152": "res3a_branch1",
        "resnet_model/add_3": "res3a",
        "resnet_model/Relu_12": "res3a_relu",

        "Conv__153": "res3b_branch2a",
        "resnet_model/Relu_13": "res3b_branch2a_relu",
        "Conv__154": "res3b_branch2b",
        "resnet_model/Relu_14": "res3b_branch2b_relu",
        "Conv__155": "res3b_branch2c",
        "resnet_model/add_4": "res3b",
        "resnet_model/Relu_15": "res3b_relu",

        "Conv__160": "res3c_branch2a",
        "resnet_model/Relu_16": "res3c_branch2a_relu",
        "Conv__161": "res3c_branch2b",
        "resnet_model/Relu_17": "res3c_branch2b_relu",
        "Conv__162": "res3c_branch2c",
        "resnet_model/add_5": "res3c",
        "resnet_model/Relu_18": "res3c_relu",

        "Conv__167": "res3d_branch2a",
        "resnet_model/Relu_19": "res3d_branch2a_relu",
        "Conv__168": "res3d_branch2b",
        "resnet_model/Relu_20": "res3d_branch2b_relu",
        "Conv__169": "res3d_branch2c",
        "resnet_model/add_6": "res3d",
        "resnet_model/Relu_21": "res3d_relu",

        "Conv__174": "res4a_branch2a",
        "resnet_model/Relu_22": "res4a_branch2a_relu",
        "Conv__175": "res4a_branch2b",
        "resnet_model/Relu_23": "res4a_branch2b_relu",
        "Conv__176": "res4a_branch2c",
        "Conv__181": "res4a_branch1",
        "resnet_model/add_7": "res4a",
        "resnet_model/Relu_24": "res4a_relu",

        "Conv__182": "res4b_branch2a",
        "resnet_model/Relu_25": "res4b_branch2a_relu",
        "Conv__183": "res4b_branch2b",
        "resnet_model/Relu_26": "res4b_branch2b_relu",
        "Conv__184": "res4b_branch2c",
        "resnet_model/add_8": "res4b",
        "resnet_model/Relu_27": "res4b_relu",

        "Conv__189": "res4c_branch2a",
        "resnet_model/Relu_28": "res4c_branch2a_relu",
        "Conv__190": "res4c_branch2b",
        "resnet_model/Relu_29": "res4c_branch2b_relu",
        "Conv__191": "res4c_branch2c",
        "resnet_model/add_9": "res4c",
        "resnet_model/Relu_30": "res4c_relu",

        "Conv__196": "res4d_branch2a",
        "resnet_model/Relu_31": "res4d_branch2a_relu",
        "Conv__197": "res4d_branch2b",
        "resnet_model/Relu_32": "res4d_branch2b_relu",
        "Conv__198": "res4d_branch2c",
        "resnet_model/add_10": "res4d",
        "resnet_model/Relu_33": "res4d_relu",

        "Conv__203": "res4e_branch2a",
        "resnet_model/Relu_34": "res4e_branch2a_relu",
        "Conv__204": "res4e_branch2b",
        "resnet_model/Relu_35": "res4e_branch2b_relu",
        "Conv__205": "res4e_branch2c",
        "resnet_model/add_11": "res4e",
        "resnet_model/Relu_36": "res4e_relu",

        "Conv__210": "res4f_branch2a",
        "resnet_model/Relu_37": "res4f_branch2a_relu",
        "Conv__211": "res4f_branch2b",
        "resnet_model/Relu_38": "res4f_branch2b_relu",
        "Conv__212": "res4f_branch2c",
        "resnet_model/add_12": "res4f",
        "resnet_model/Relu_39": "res4f_relu",

        "Conv__217": "res5a_branch1",
        "Conv__222": "res5a_branch2a",
        "resnet_model/Relu_40": "res5a_branch2a_relu",
        "Conv__223": "res5a_branch2b",
        "resnet_model/Relu_41": "res5a_branch2b_relu",
        "Conv__224": "res5a_branch2c",
        "resnet_model/add_13": "res5a",
        "resnet_model/Relu_42": "res5a_relu",

        "Conv__225": "res5b_branch2a",
        "resnet_model/Relu_43": "res5b_branch2a_relu",
        "Conv__226": "res5b_branch2b",
        "resnet_model/Relu_44": "res5b_branch2b_relu",
        "Conv__227": "res5b_branch2c",
        "resnet_model/add_14": "res5b",
        "resnet_model/Relu_45": "res5b_relu",

        "Conv__232": "res5c_branch2a",
        "resnet_model/Relu_46": "res5c_branch2a_relu",
        "Conv__233": "res5c_branch2b",
        "resnet_model/Relu_47": "res5c_branch2b_relu",
        "Conv__234": "res5c_branch2c",
        "resnet_model/add_15": "res5c",
        "resnet_model/Relu_48": "res5c_relu",

        "resnet_model/Mean": "pool5",
        "reshape__269": "reshape",
        "resnet_model/Squeeze": "squeeze",
        "resnet_model/dense/MatMul": "fc1000",
        "resnet_model/dense/BiasAdd": "bias_add",
        "resnet_model/final_dense": "final_dense",
        "graph_outputs_Identity__6": "prob",
        "graph_outputs_Identity__4": "topk",
    }

    # Define the which at which tensor to divide the subnetwork
    subnetwork_map = {
        "dla": [
            {
                "tensor_name": "fc_replaced_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 1000, 1, 1),
                "tensor_io_type": "output",
            },
        ],
        "topk": [
            {
                "tensor_name": "fc_replaced_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 1000, 1, 1),
                "tensor_io_type": "input",
            },
        ],
        "preres2": [
            {
                "tensor_name": "pool1_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 64, 56, 56),
                "tensor_io_type": "output",
            },
        ],
        "preres3": [
            {
                "tensor_name": "res2c_relu_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 256, 56, 56),
                "tensor_io_type": "output",
            },
        ],
        "res2_3": [
            {
                "tensor_name": "pool1_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 64, 56, 56),
                "tensor_io_type": "input",
            },
            {
                "tensor_name": "res3d_relu_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 512, 28, 28),
                "tensor_io_type": "output",
            },
        ],
        "res3": [
            {
                "tensor_name": "res2c_relu_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 256, 56, 56),
                "tensor_io_type": "input",
            },
            {
                "tensor_name": "res3d_relu_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 512, 28, 28),
                "tensor_io_type": "output",
            },
        ],
        "postres3": [
            {
                "tensor_name": "res3d_relu_out_0",
                "tensor_shape": (gs.Tensor.DYNAMIC, 512, 28, 28),
                "tensor_io_type": "input",
            },
        ],
        "res3end": [
            {
                "tensor_name" : "res2c_relu_out_0",
                "tensor_shape" : (gs.Tensor.DYNAMIC, 256, 56, 56),
                "tensor_io_type" : "input",
            },
        ],
    }

    def __init__(self, onnx_path, compute_sm, device_type, precision,
                 cache_file, need_calibration, rn50_args, subnetwork_gs=None):
        """
        Initialize the class, with:
           onnx file path, compute SMs, compute precision, calibration cache file

        Or, anything that is required to determine how to update graph
        """
        self.onnx_path = onnx_path
        self.compute_sm = compute_sm
        self.device_type = device_type
        self.precision = precision
        self.cache_file = cache_file
        self.need_calibration = need_calibration
        self.rn50_args = rn50_args
        self.subnetwork_gs = subnetwork_gs
        self.output_names = list()
        self.model = None
        self.dyn_range_map = {}
        if os.path.exists(self.cache_file):
            self.dyn_range_map = get_dyn_ranges(self.cache_file)
        else:
            print("WARNING: No calibration cache available for parsing.")

    def process_onnx(self):
        """
        One stop shopping; just call this!

        It does below in order:
        1. import ONNX graph from self.onnx_path
        2. rename all the ops and tensors as in self.op_name_map
        3. replace/add/prune ops from graph self.op_touchup_map
        4. fuse ops as in self.fusion_map
        5. export ONNX graph for TRT, save it under self.model
        """

        self.import_onnx()
        self.rename_nodes()
        self.touchup_ops()
        self.fuse_ops()
        # if seperate_subnetwork
        if self.subnetwork_gs is not None:
            self.extract_subnetwork(self.subnetwork_gs)
        self.export_onnx()

        return self.model

    def import_onnx(self):
        """
        Import onnx graph for surgery
        """
        model = onnx.load(self.onnx_path)
        self.graph = gs.import_onnx(model)

    def export_onnx(self):
        """
        Return graph back to TRT
        """
        self.model = gs.export_onnx(self.graph)

    def fuse_ops(self):
        """
        Handle fusion of ops
        NOTE: per GPU arch, different fusion may be done

        Here, a chain of ops is mapped to a fused op

        NOTE: User may specify a policy that leads to multiple fusion
              where fusion_map needs to specify multiple such methods in order
        """
        policy_selector = {
            'Unknown': ['Unknown'],
            '75': ['Res2Mega'],
            '80': ['Res2Mega', 'Beta1Smallk'],
            '86': ['Res2Mega', 'Beta1Smallk'],
            '87': ['Res2Mega', 'Beta1Smallk'],
            '89': ['Res2Mega', 'Beta1Smallk'],
            '90': ['Res2Mega', 'Beta1Smallk'],
        }
        policy = 'Unknown'
        if self.device_type != 'gpu' or self.need_calibration:
            policies = ['NO_FUSE']
        elif self.precision in ('int8',):
            policies = policy_selector.get(str(self.compute_sm), 'Unknown')

        # Filter out the beta=1 small-k optimization if turned off by configs
        disable_beta1_smallk = dict_get(self.rn50_args, Fields.disable_beta1_smallk.name, default=False)
        if disable_beta1_smallk:
            policies.remove('Beta1Smallk')

        for policy in policies:
            for _f in self.fusion_map.get(policy):
                self.runme(_f)

    def touchup_ops(self):
        """
        Handle op touchup
        NOTE: different precision may lead to different op mix

        Some notes for what this function does, as of 09/25/2020:

        Original ONNX graph has a tail with Squeeze-and-Excitation block,
        followed by SoftMax/ArgMax for classification.

        Suspecting the performance of kernel as a reason, we are changing
        this sub-graph (after ReLU*), with more lightweight Pooling-FC-TopK
        sub-graph. As per kernel performance, w/ INT8, Conv1x1x1 replaces FC

            [BEFORE]                           [AFTER]
              Add                                Add
               |                                  |
              ReLU*                              ReLU*
               |                                  |
            ReduceMean                          AvgPool
               |                ==>               |
             Reshape                            FC/Conv
               |                                  |
             Squeeze                             TopK
               |                                  |
             MatMul                          +----+-----+
               |                             |          |
              Add                          Value       Index
               |
            Identity
               |
          +----+----+
          |         |
        SoftMax   ArgMax
          |         |
        Identity  Identity
          |         |
         Prob      Index

        In order to realize this in a modular way, entry in the op_touchup_map
        is selected, based on the condition (i.e. if precision is INT8 or not)
        and the series of calls are made to add new sub-graph after ReLU*.
        After adding the proper combination of ops (add_* calls), original sub-graph
        is removed by de-registering outputs from graph and clean up the graph.

        NOTE: the above is not necessarily what this touchup_ops is limited to achieve;
              if the graph is in need to be manipulated in a different way, it can be
              mapped from here to op_touchup_map, which defines needed calls in order.
        """
        policy_selector = {
            'int8': 'ConvForFC',
            'Unknown': 'Unknown',
        }
        policy = policy_selector.get(self.precision, 'generic')
        for _f in self.op_touchup_map.get(policy):
            self.runme(_f)

    def extract_subnetwork(self, subnetwork):
        """
        Extract subnetwork

        RN50 subnetwork definition:

        DLA - topK partition
                       __
           conv1         |
             |           |
            ...          |  -- DLA subnetwork (can fully run on DLA)
             |           |
        fc_replaced    __|
             |           |
         topk_layer    __|  -- topK subnetwork

        ==============================================================
        PreRes3 - Res3 - PostRes3 Partition
                                     __
           conv1                       |
             |                         |
            ...                        |  -- PreRes3 subnetwork
             |                         |
        RES2_FULL_FUSION             __|
             |                         |
            ...                        |
             |                         |  -- Res3 subnetwork
        SmallTileGEMM_TRT_res3d_       |
        branch2c_conv_residual_relu  __|
             |                         |
            ...                        |  -- PostRes3 subnetwork
             |                         |
         topk_layer                  __|

        ==============================================================
        PreRes2 - Res2_3 - PostRes3 Partition
                                     __
           conv1                       |
             |                         |
            ...                        |  -- PreRes2 subnetwork
             |                         |
           pool1                     __|
             |                         |
            ...                        |
             |                         |  -- Res2_3 subnetwork
        SmallTileGEMM_TRT_res3d_       |
        branch2c_conv_residual_relu  __|
             |                         |
            ...                        |  -- PostRes3 subnetwork
             |                         |
         topk_layer                  __|

        """

        logging.info(f"Extracting subnetwork: {subnetwork}")
        tensors = self.graph.tensors()
        if subnetwork in self.subnetwork_map:
            for tensor in self.subnetwork_map[subnetwork]:
                tensor_name = tensor["tensor_name"]
                tensor_shape = tensor["tensor_shape"]
                tensor_io_type = tensor["tensor_io_type"]
                if tensor_io_type == "input":
                    self.graph.inputs = [tensors[tensor_name].to_variable(dtype=np.float32, shape=tensor_shape)]
                else:
                    self.graph.outputs = [tensors[tensor_name].to_variable(dtype=np.float32, shape=tensor_shape)]
        else:
            logging.warning(f"Unsupported subnetwork option: {subnetwork}. Do nothing.")

        self.cleanup_graph()

    def cleanup_graph(self):
        """
        Cleanup graph
        """
        self.graph.cleanup().toposort()

    def rename_nodes(self):
        """
        Rename op and tensors in the graph
        """
        self.rename_ops()
        self.rename_tensors()

    def rename_ops(self):
        """
        Rename op names as in self.op_name_map
        """
        logging.info("Renaming layers")
        for node in self.graph.nodes:
            if node.name in self.op_name_map:
                new_name = self.op_name_map[node.name]
                # logging.info("Renaming layer: {} -> {}".format(node.name, new_name))
                node.name = new_name

    def rename_tensors(self):
        """
        Update tensor name to be consistent to it's producer op
        """
        logging.info("Renaming tensors")
        for node in self.graph.nodes:
            for t_idx, out_tensor in enumerate(node.outputs):
                if not out_tensor.name or node.name not in out_tensor.name:
                    # logging.info("Naming tensor: {} -- {}_out_{}".format(node.name, node.name, t_idx))
                    out_tensor.name = "{}_out_{}".format(node.name, t_idx)
        assert len(self.graph.inputs) == 1, "only one input is expected: {}".format(self.graph.inputs)
        graph_input = self.graph.inputs[0]
        graph_input.name = graph_input.name.replace(":", "_")

    def fuse_beta1_conv(self):
        """
        Fuse all conv+scale+bias+beta+relu layers after res2
        """
        logging.info("Replacing all branch2c beta=1 conv with smallk kernel.")
        plugin_op_name = "SmallTileGEMM_TRT"

        BetaConvTuple = namedtuple("BetaConvTuple", "residual conv_in relu_out")
        op_names_list = [BetaConvTuple("res3a", "res3a_branch2c", "res3a_relu"),
                         BetaConvTuple("res3b", "res3b_branch2c", "res3b_relu"),
                         BetaConvTuple("res3c", "res3c_branch2c", "res3c_relu"),
                         BetaConvTuple("res3d", "res3d_branch2c", "res3d_relu"),
                         BetaConvTuple("res4a", "res4a_branch2c", "res4a_relu"),
                         BetaConvTuple("res4b", "res4b_branch2c", "res4b_relu"),
                         BetaConvTuple("res4c", "res4c_branch2c", "res4c_relu"),
                         BetaConvTuple("res4d", "res4d_branch2c", "res4d_relu"),
                         BetaConvTuple("res4e", "res4e_branch2c", "res4e_relu"),
                         BetaConvTuple("res4f", "res4f_branch2c", "res4f_relu"),
                         BetaConvTuple("res5a", "res5a_branch2c", "res5a_relu"),
                         BetaConvTuple("res5b", "res5b_branch2c", "res5b_relu"),
                         BetaConvTuple("res5c", "res5c_branch2c", "res5c_relu"),
                         ]

        for op_names_tuple in op_names_list:
            plugin_layer_name = f"{plugin_op_name}_{op_names_tuple.conv_in}_conv_residual_relu"
            op_dict = dict()
            [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if _n.name in op_names_tuple]
            op_list = [op_dict[_n] for _n in op_names_tuple]
            assert len(op_names_tuple) == len(op_list), "Need to capture all op objects in op_names_tuple"

            plugin_inp = [op_dict[op_names_tuple.conv_in].inputs[0], op_dict[op_names_tuple.residual].inputs[1]]
            plugin_out = [op_dict[op_names_tuple.relu_out].outputs[0]]

            # Create dummy input for scale and rescale (all ones)
            K = op_dict[op_names_tuple.conv_in].inputs[1].shape[0]
            C = op_dict[op_names_tuple.conv_in].inputs[1].shape[1]
            scale = gs.Constant("scale", values=np.ones((K), dtype=np.float32))
            rescale = gs.Constant("rescale", values=np.ones((K), dtype=np.float32))

            # Dynamic ranges for input/output/residual (Ti/To/Tr)
            dyn_list = [
                self.dyn_range_map[op_dict[op_names_tuple.conv_in].inputs[0].name],
                self.dyn_range_map[op_dict[op_names_tuple.relu_out].outputs[0].name],
                self.dyn_range_map[op_dict[op_names_tuple.residual].inputs[1].name],
            ]

            dynamic_ranges = np.array(dyn_list, dtype=np.float32)
            dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_layer_name),
                                    values=dynamic_ranges)

            plugin_field_dict = {
                "inputChannels": gs.Constant("C", values=np.array([C], dtype=np.int32)),
                "filterDimR": gs.Constant("R", values=np.array([1], dtype=np.int32)),
                "filterDimS": gs.Constant("S", values=np.array([1], dtype=np.int32)),
                "weight": op_dict[op_names_tuple.conv_in].inputs[1],
                "bias": op_dict[op_names_tuple.conv_in].inputs[2],
                "scale": scale,
                "rescale": rescale,
                "dynamicRanges": dyn_const,
                "epilogueScaleBiasBetaRelu": gs.Constant("epilogue_sbbr", values=np.array([1], dtype=np.int32)),
                # Dummy fields to supress warnings
                "fairShareCacheSize": gs.Constant("fairShareCacheSize", values=np.array([0], dtype=np.int32)),
            }

            attrs = {
                "plugin_version": "1",
                "plugin_namespace": "",
            }
            attrs.update(plugin_field_dict)

            # replace ops with plugin
            logging.info(f"Fusing {plugin_layer_name} with smallk...")
            self.graph.BETA1SmallKPlugin(plugin_op_name, plugin_layer_name, plugin_inp, plugin_out, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} fused successful for res3/4/5 branch2c".format(plugin_op_name))

    def fuse_res2_mega(self):
        """
        Search and replace all the res2 layers with the res2 megakernel plugin.
        This fusion is for mega fusion of entire res2a_*
        """
        logging.info("Fusing ops in res2_mega")
        op_names_list = ["res2a_branch1",
                         "res2a_branch2a", "res2a_branch2a_relu",
                         "res2a_branch2b", "res2a_branch2b_relu",
                         "res2a_branch2c", "res2a", "res2a_relu",
                         "res2b_branch2a", "res2b_branch2a_relu",
                         "res2b_branch2b", "res2b_branch2b_relu",
                         "res2b_branch2c", "res2b", "res2b_relu",
                         "res2c_branch2a", "res2c_branch2a_relu",
                         "res2c_branch2b", "res2c_branch2b_relu",
                         "res2c_branch2c", "res2c", "res2c_relu",
                         ]

        # setup plugin info
        plugin_name = "RES2_FULL_FUSION"

        # prep fusion: constants and attributes
        op_dict = dict()
        [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if _n.name in op_names_list]
        op_list = [op_dict[_n] for _n in op_names_list]
        assert len(op_names_list) == len(op_list), "Need to capture all op objects in op_names_list"

        plugin_inp = [op_list[0].inputs[0]]
        plugin_out = [op_list[-1].outputs[0]]

        scale64 = gs.Constant("scale64", values=np.ones((64), dtype=np.float32))
        scale256 = gs.Constant("scale256", values=np.ones((256), dtype=np.float32))
        rescale = gs.Constant("rescale", values=np.ones((256), dtype=np.float32))

        # build array with dynamic ranges required for the fusion plugin
        # NOTE: order matters
        dyn_list = [
            self.dyn_range_map[plugin_inp[0].name],

            self.dyn_range_map[op_list[0].outputs[0].name],

            self.dyn_range_map[op_list[2].outputs[0].name],
            self.dyn_range_map[op_list[4].outputs[0].name],
            self.dyn_range_map[op_list[5].outputs[0].name],
            self.dyn_range_map[op_list[7].outputs[0].name],

            self.dyn_range_map[op_list[9].outputs[0].name],
            self.dyn_range_map[op_list[11].outputs[0].name],
            self.dyn_range_map[op_list[12].outputs[0].name],
            self.dyn_range_map[op_list[14].outputs[0].name],

            self.dyn_range_map[op_list[16].outputs[0].name],
            self.dyn_range_map[op_list[18].outputs[0].name],
            self.dyn_range_map[op_list[19].outputs[0].name],
            self.dyn_range_map[op_list[21].outputs[0].name],
        ]

        dynamic_ranges = np.array(dyn_list, dtype=np.float32)
        dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_name),
                                values=dynamic_ranges)

        # this becomes attributes to ONNX node that fusion plugin uses
        # NOTE: order does not matter
        plugin_field_dict = {
            "c_res2a_br1_w": op_list[0].inputs[1],
            "s_res2a_br1_s": scale256,
            "s_res2a_br1_b": op_list[0].inputs[2],

            "c_res2a_br2a_w": op_list[1].inputs[1],
            "s_res2a_br2a_s": scale64,
            "s_res2a_br2a_b": op_list[1].inputs[2],
            "c_res2a_br2b_w": op_list[3].inputs[1],
            "s_res2a_br2b_s": scale64,
            "s_res2a_br2b_b": op_list[3].inputs[2],
            "c_res2a_br2c_w": op_list[5].inputs[1],
            "s_res2a_br2c_s": scale256,
            "s_res2a_br2c_b": op_list[5].inputs[2],

            "c_res2b_br2a_w": op_list[8].inputs[1],
            "s_res2b_br2a_s": scale64,
            "s_res2b_br2a_b": op_list[8].inputs[2],
            "c_res2b_br2b_w": op_list[10].inputs[1],
            "s_res2b_br2b_s": scale64,
            "s_res2b_br2b_b": op_list[10].inputs[2],
            "c_res2b_br2c_w": op_list[12].inputs[1],
            "s_res2b_br2c_s": scale256,
            "s_res2b_br2c_b": op_list[12].inputs[2],

            "c_res2c_br2a_w": op_list[15].inputs[1],
            "s_res2c_br2a_s": scale64,
            "s_res2c_br2a_b": op_list[15].inputs[2],
            "c_res2c_br2b_w": op_list[17].inputs[1],
            "s_res2c_br2b_s": scale64,
            "s_res2c_br2b_b": op_list[17].inputs[2],
            "c_res2c_br2c_w": op_list[19].inputs[1],
            "s_res2c_br2c_s": scale256,
            "s_res2c_br2c_b": op_list[19].inputs[2],

            "r_res2a_br2c_r": rescale,
            "r_res2b_br2c_r": rescale,
            "r_res2c_br2c_r": rescale,

            "dynamic_ranges": dyn_const,
        }

        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        attrs.update(plugin_field_dict)

        # replace ops with plugin
        self.graph.RES2PLUGIN("RnRes2FullFusion_TRT", plugin_name, plugin_inp, plugin_out, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} successful".format(plugin_name))

    def add_squeeze(self):
        """
        add new squeeze layer
        """
        logging.info("Adding Squeeze")
        # find input to squeeze to be added
        last_relu_op = [_n for _n in self.graph.nodes if _n.name == "res5c_relu"][0]
        # add AveragePool
        attrs = {
            "kernel_shape": [7, 7]
        }
        squeeze_replaced_out = self.graph.AveragePool("squeeze_replaced", last_relu_op.outputs[0], attrs)

    def add_fc(self):
        """
        add FC layer
        """
        logging.info("Adding FC layer")
        # fetch some attrs from old fc1000; note MatMul doesn't have bias
        old_fc_op = [_n for _n in self.graph.nodes if _n.name == "fc1000"][0]
        old_fc_kernel = old_fc_op.inputs[1]
        fc_kernel_weights = old_fc_kernel.values[:, 1:]
        # instantiate fc weight
        # NOTE: expects KM weight, if transpose is not set (default not set)
        fc_weight = gs.Constant("fc_replaced_weight", values=fc_kernel_weights)
        # find input to fc to be added
        squeeze_replaced_op = [_n for _n in self.graph.nodes if _n.name == "squeeze_replaced"][0]
        squeeze_replaced_out = squeeze_replaced_op.outputs[0]
        # reshape input
        reshape_shape = np.array([-1, fc_kernel_weights.shape[0]], dtype=np.int64)
        fc_reshape_shape = gs.Constant("fc_reshape_shape", values=reshape_shape)
        # add FC: Reshape=>MatMul
        fc_reshape_out = self.graph.Reshape("fc_reshape_input", squeeze_replaced_out, fc_reshape_shape)
        fc_out = self.graph.MatMul("fc_replaced", fc_reshape_out, fc_weight)

    def add_conv(self):
        """
        add Conv layer
        """
        logging.info("Adding Conv layer, instead of FC")
        # fetch some attrs from old fc1000; note MatMul doesn't have bias
        old_fc_op = [_n for _n in self.graph.nodes if _n.name == "fc1000"][0]
        old_fc_kernel = old_fc_op.inputs[1]
        # instantiate fc weight and attrs
        # NOTE: ONNX uses MCkHkW format
        fc_kernel_weights = old_fc_kernel.values.transpose()[1:, :].reshape(1000, 2048, 1, 1)
        fc_weight = gs.Constant("fc_replaced_weight", values=fc_kernel_weights)
        attrs = {
            "kernel_shape": [1, 1]
        }
        # find input to fc to be added
        squeeze_replaced_op = [_n for _n in self.graph.nodes if _n.name == "squeeze_replaced"][0]
        squeeze_replaced_out = squeeze_replaced_op.outputs[0]
        # add FC: Conv
        fc_out = self.graph.Conv("fc_replaced", squeeze_replaced_out, fc_weight, attrs)

    def add_topk(self):
        """
        add topk layer
        """
        logging.info("Adding TopK layer")
        # find input to topk to be added
        fc_op = [_n for _n in self.graph.nodes if _n.name == "fc_replaced"][0]
        fc_op_out = fc_op.outputs[0]
        # set attrs
        attrs = {
            "axis": 1,
            "k": 1,
            "largest": 1,
        }
        # add TopK
        topk_out_list = self.graph.TopK("topk_layer", fc_op_out, attrs)

    def remove_obsolete(self):
        """
        Remove obsolete layers
        """
        logging.info("Removing obsolete layers")
        topk_op = [_n for _n in self.graph.nodes if _n.name == "topk_layer"][0]
        self.graph.outputs = topk_op.outputs
        self.cleanup_graph()

    def runme(self, x):
        """
        Run class method from method name string
        """
        def selfer(x): return "self.{}()".format(x)
        return eval(selfer(x))


def parse_args():
    """
    Arguments that can be used for standalone run
    """
    parser = argparse.ArgumentParser(description=RN50GraphSurgeon.__doc__)
    parser.add_argument('--onnx-fpath',
                        dest='onnx_path',
                        type=str,
                        default='build/models/ResNet50/resnet50_v1.onnx',
                        help='Input ONNX file for ResNet50')
    parser.add_argument('--output-onnx-fname',
                        dest='out_onnx',
                        type=str,
                        default='rn50_discharged.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--calibration-cache-fpath',
                        dest='calibcache',
                        type=str,
                        default='code/resnet50/tensorrt/calibrator.cache',
                        help='Calibration cache file')
    parser.add_argument('--compute-sm',
                        dest='compute_sm',
                        type=str,
                        default='75',
                        choices={'Unknown',
                                 '75',  # Turing
                                 '80',  # Ampere (A100, A30)
                                 '86',  # Ampere (A40, A10, A16, A2)
                                 '87',  # Orin AGX
                                 '90',  # Hopper
                                 },
                        help='GPU Architecture of choice')
    parser.add_argument('--precision',
                        dest='precision',
                        type=str,
                        default='int8',
                        choices={'int8', 'fp16', 'fp32'},
                        help='Compute precision')
    parser.add_argument('--non-gpu',
                        dest='nongpu',
                        default=False,
                        action='store_true',
                        help='Device is not GPU, i.e. DLA')
    parser.add_argument('--need-calibration',
                        dest='need_cal',
                        default=False,
                        action='store_true',
                        help='In case calibration is required; do not fuse for example')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.info("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    Standalone run manipulates input ONNX graph and returns updated ONNX graph
    How to run:
        in container: python3 -m code.resnet50.tensorrt.rn50_graphsurgeon --help
    """
    rn50gs = RN50GraphSurgeon(args.onnx_path,
                              args.compute_sm,
                              ('dla' if args.nongpu else 'gpu'),
                              args.precision,
                              args.calibcache,
                              args.need_cal,
                              args.subnetwork)
    model = rn50gs.process_onnx()
    onnx.save(rn50gs.model, args.out_onnx)


if __name__ == '__main__':
    args = parse_args()
    main(args)
