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


__doc__ = """Scripts for modifying Retinanet onnx graphs
"""

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
from code.retinanet.tensorrt.onnx_generator.anchor_generator import AnchorGenerator


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


class RetinanetGraphSurgeon(object):
    """
    The class will take the output of the torch2onnx graph from MLPerf inference repo,
    and apply the changes listed in process_onnx().
    """
    fusion_map = {
        "Beta1Smallk": ["fuse_beta1_conv"],
    }

    # Constants for generating default anchor boxes for retinanet
    anchor_sizes = tuple((x, int(x * 2 ** (1.0 / 3)), int(x * 2 ** (2.0 / 3))) for x in [32, 64, 128, 256, 512])
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)

    def __init__(self, onnx_path, compute_sm, precision, device_type,
                 cache_file, need_calibration, nms_type,
                 retinanet_args=None, subnetwork_gs=None):
        self.onnx_path = onnx_path
        self.compute_sm = compute_sm
        self.precision = precision
        self.device_type = device_type
        self.cache_file = cache_file
        self.need_calibration = need_calibration
        self.nms_type = nms_type
        self.retinanet_args = retinanet_args
        self.subnetwork_gs = subnetwork_gs
        self.output_names = list()
        self.model = None
        self.dyn_range_map = {}
        if os.path.exists(self.cache_file):
            self.dyn_range_map = get_dyn_ranges(self.cache_file)
        else:
            print("WARNING: No calibration cache available for parsing.")

        # Create the anchor generator instance for later use
        self.anchor_generator = AnchorGenerator(
            self.anchor_sizes, self.aspect_ratios
        )

    def process_onnx(self):
        """
        This function will do the following:
        1. Import ONNX graph
        2. Rename ops and tensors
        3. Add post-processing NMS layers
        4. Add fusion ops
        5. (TODO optional) Extract subnetwork for DLA
        6. Clean-up graphs
        7. Export ONNX graph for TRT and save it.
        """
        self.import_onnx()
        self.rename_nodes()
        self.rename_tensors()
        self.add_nms()
        self.fuse_ops()
        if self.subnetwork_gs is not None:
            self.extract_subnetwork(self.subnetwork_gs)
        self.cleanup_graph()
        self.export_onnx()

        return self.model

    def import_onnx(self):
        """
        Import onnx graph using onnx and graphsurgeon
        """
        logging.info(f"Loading graph from onnx file {self.onnx_path}")
        model = onnx.load(self.onnx_path)
        self.graph = gs.import_onnx(model)

    def export_onnx(self):
        """
        Export the graph which can be saved later
        """
        self.model = gs.export_onnx(self.graph)

    def save_onnx(self, path):
        """
        Save post-processed graph to an onnx file
        """
        logging.info(f"Saving the graph model to onnx file {path}")
        onnx.save(self.model, path)

    def rename_nodes(self):
        """
        The backbone has 5 resNeXt blocks.
        For blocks starting res2a, the number corresponds to the starting layer
        for branch2a, branch2b, branch2c, (optional) branch1, residual.
        - branch2a/b/c has conv+mul+add+relu layers
        - branch1 has conv+mul+add layers
        - the residual has add+relu layers
        """
        logging.info("Renaming layers...")

        backbone_rename_map = {
            # Res1
            "Conv_0": "res1_conv",
            "Mul_1": "res1_scale",
            "Add_2": "res1_bias",
            "Relu_3": "res1_relu",
            "MaxPool_4": "res1_maxpool"
        }
        resblock_layer_map = {
            'res2a': [5, 9, 13, 16, 19],
            'res2b': [21, 25, 29, 32],
            'res2c': [34, 38, 42, 45],

            'res3a': [47, 51, 55, 58, 61],
            'res3b': [63, 67, 71, 74],
            'res3c': [76, 80, 84, 87],
            'res3d': [89, 93, 97, 100],

            'res4a': [102, 106, 110, 113, 116],
            'res4b': [118, 122, 126, 129],
            'res4c': [131, 135, 139, 142],
            'res4d': [144, 148, 152, 155],
            'res4e': [157, 161, 165, 168],
            'res4f': [170, 174, 178, 181],

            'res5a': [183, 187, 191, 194, 197],
            'res5b': [199, 203, 207, 210],
            'res5c': [212, 216, 220, 223],
        }

        # Construct the backbone_rename_map from the layer map
        for block, layers in resblock_layer_map.items():
            if len(layers) >= 4:
                dic = {
                    # 2a
                    f"Conv_{layers[0]}": f"{block}_branch2a_conv",
                    f"Mul_{layers[0]+1}": f"{block}_branch2a_scale",
                    f"Add_{layers[0]+2}": f"{block}_branch2a_bias",
                    f"Relu_{layers[0]+3}": f"{block}_branch2a_relu",

                    # 2b
                    f"Conv_{layers[1]}": f"{block}_branch2b_conv",
                    f"Mul_{layers[1]+1}": f"{block}_branch2b_scale",
                    f"Add_{layers[1]+2}": f"{block}_branch2b_bias",
                    f"Relu_{layers[1]+3}": f"{block}_branch2b_relu",

                    # 2c
                    f"Conv_{layers[2]}": f"{block}_branch2c_conv",
                    f"Mul_{layers[2]+1}": f"{block}_branch2c_scale",
                    f"Add_{layers[2]+2}": f"{block}_branch2c_bias",

                    # Residual connection
                    f"Add_{layers[-1]}": f"{block}_residual_add",
                    f"Relu_{layers[-1] + 1}": f"{block}_relu",
                }

                if len(layers) == 5:
                    # res-a layers, which has the branch1
                    br1 = {
                        # 1
                        f"Conv_{layers[3]}": f"{block}_branch1_conv",
                        f"Mul_{layers[3]+1}": f"{block}_branch1_scale",
                        f"Add_{layers[3]+2}": f"{block}_branch1_bias",
                    }
                    dic.update(br1)

                backbone_rename_map.update(dic)

            else:
                raise RuntimeError(f"ResNeXt block {block} has illegal numbers of layers. Should be 4 or 5")

        count = 0
        for node in self.graph.nodes:
            if node.name in backbone_rename_map:
                new_name = backbone_rename_map[node.name]
                logging.debug("Renaming layer: {} -> {}".format(node.name, new_name))
                node.name = new_name
                count += 1
        logging.info(f"Renamed {count} layers.")

    def rename_tensors(self):
        """
        Update tensor name to be consistent to its producer op
        TODO: for now, only backbone resnext layers are renamed
        """
        logging.info("Renaming tensors to match layer names")
        for node in self.graph.nodes:
            if 'res' in node.name:
                for t_idx, out_tensor in enumerate(node.outputs):
                    if not out_tensor.name or node.name not in out_tensor.name:
                        logging.debug("Naming tensor: {} -- {}_out_{}".format(node.name, node.name, t_idx))
                        out_tensor.name = "{}_out_{}".format(node.name, t_idx)
                # Rename the constant scale/bias tensor.
                if 'scale' in node.name or 'bias' in node.name:
                    for input_tensor in node.inputs:
                        if 'out' not in input_tensor.name:
                            tensor_name = f"{node.name}_value"
                            input_tensor.name = tensor_name

        assert len(self.graph.inputs) == 1, "only one input is expected: {}".format(self.graph.inputs)
        graph_input = self.graph.inputs[0]

    def add_nms(self):
        """
        Append the Non-Maximum Suppression (NMS) layer to the conv heads
        """
        logging.info(f"Adding NMS layer {self.nms_type} to the graph...")
        if self.nms_type == 'none':
            return
        elif self.nms_type == 'efficientnms':
            self.add_efficientnms()
        elif self.nms_type == 'nmsopt':
            self.add_nmsopt()
        else:
            raise NotImplementedError(f"No such nms {self.nms_type}, exiting...")

    def add_efficientnms(self):
        """
        Add the open-sourced efficientNMS as documented in
        https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin

        Note the efficientNMS needs to be followed by a RetinaNet Concat Plugin, which
        arranges the output in the right order and concat into 1 tensor.
        """
        tensors = self.graph.tensors()
        # EfficientNMS use xywh-format anchor boxes, scaled down to [0,1]
        np_anchor_xywh_scaled = self.anchor_generator(scale_retinanet=True, order="xywh")[0].detach().cpu().numpy()
        anchor = np.expand_dims(np_anchor_xywh_scaled, axis=0)
        anchor_tensor = gs.Constant(name="anchor", values=anchor)

        op = 'EfficientNMS_TRT'
        node_name = 'efficientNMS'

        # Populate the plugin fields
        node_attrs = {
            "background_class": -1,
            "score_threshold": 0.05,
            "iou_threshold": 0.5,
            "max_output_boxes": 1000,
            "score_activation": True,
            "box_coding": 1,
        }
        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        attrs.update(node_attrs)

        # Create ouptut tensors for the EfficientNMS
        num_detections = gs.Variable(name="num_detections",
                                     dtype=np.int32,
                                     shape=["batch_size", 1])
        detection_boxes = gs.Variable(name="detection_boxes",
                                      dtype=np.float32,
                                      shape=["batch_size", 1000, 4])
        detection_scores = gs.Variable(name="detection_scores",
                                       dtype=np.float32,
                                       shape=["batch_size", 1000])
        detection_classes = gs.Variable(name="detection_classes",
                                        dtype=np.int32,
                                        shape=["batch_size", 1000])

        nms_inputs = [tensors["bbox_regression"], tensors["cls_logits"], anchor_tensor]
        nms_outputs = [num_detections, detection_boxes, detection_scores, detection_classes]

        self.graph.layer(op="EfficientNMS_TRT",
                         name="EfficientNMS",
                         inputs=nms_inputs,
                         outputs=nms_outputs,
                         attrs=attrs)

        # Add Retinanet concat plugin
        concat_final_output = gs.Variable(name="concat_final_output",
                                          dtype=np.float32,
                                          shape=["batch_size", 7001])
        attrs = {
            "plugin_version": "1",
            "plugin_namespace": "",
        }
        self.graph.layer(op="RetinanetConcatNmsOutputsPlugin",
                         name="RetinanetConcatNmsOutputsPlugin",
                         inputs=[num_detections, detection_boxes, detection_scores, detection_classes],
                         outputs=[concat_final_output],
                         attrs=attrs)
        self.graph.outputs = [concat_final_output]

        self.cleanup_graph()

    def add_nmsopt(self):
        """
        Add the optimized NMS implementation from the MLPerf repo.
        """
        tensors = self.graph.tensors()
        # NMSOPT uses ltrb-format anchor boxes, scaled down to [0,1]
        np_anchor_ltrb_scaled = self.anchor_generator(scale_retinanet=True, order="ltrb")[0].detach().cpu().numpy()
        anchor = np.expand_dims(np_anchor_ltrb_scaled, axis=0)
        anchor = np.reshape(anchor, (1, -1, 1))
        anchor_tensor = gs.Constant(name="anchor", values=anchor)

        op = 'NMS_OPT_TRT'
        node_name = 'nmsopt'

        node_attrs = {
            "shareLocation": True,
            "varianceEncodedInTarget": True,  # RetinaNet variance is all 1
            "backgroundLabelId": -1,
            "numClasses": 264,
            "topK": 1000,
            "keepTopK": 1000,
            "confidenceThreshold": 0.05,
            "nmsThreshold": 0.5,
            "inputOrder": [0, 5, 10],
            "confSigmoid": True,
            "confSoftmax": False,
            "isNormalized": True,
            "codeType": 1,
            "numLayers": 5,
        }
        attrs = {
            "plugin_version": "2",
            "plugin_namespace": "",
        }
        attrs.update(node_attrs)

        # Add nmsopt layer
        nms_output = gs.Variable(name="nmsopt_output",
                                 dtype=np.float32,
                                 shape=["batch_size", 7001])

        # Feature size from small to large
        feature_map_sizes = [100, 50, 25, 13, 7]
        conv_loc_outputs = [tensors[f"regression_head_{size}x{size}"]
                            for size in feature_map_sizes]
        conv_conf_outputs = [tensors[f"classification_head_{size}x{size}"]
                             for size in feature_map_sizes]

        nms_inputs = conv_loc_outputs + conv_conf_outputs + [anchor_tensor]
        nms_outputs = [nms_output]

        self.graph.layer(op="NMS_OPT_TRT",
                         name="nmsopt",
                         inputs=nms_inputs,
                         outputs=nms_outputs,
                         attrs=attrs)

        self.graph.outputs = [nms_output]
        self.graph.cleanup().toposort()

    def fuse_beta1_conv(self):
        """
        Fuse all conv+scale+bias+beta+relu layers for backbone layers branch2c
        """
        logging.info("Replacing all branch2c beta=1 conv with smallk kernel.")
        plugin_op_name = "SmallTileGEMM_TRT"

        # Check the dynamic range map exists
        assert self.dyn_range_map != {}, "The calibration cache has to exist. exiting..."

        beta1_op_list = [
            'res2a', 'res2b', 'res2c',
            'res3a', 'res3b', 'res3c', 'res3d',
            'res4a', 'res4b', 'res4c', 'res4d', 'res4e', 'res4f',
            # 'res5a', 'res5b', 'res5c', # C = 1024 is not supported for beta=1
        ]

        for op in beta1_op_list:
            plugin_layer_name = f"{op}_conv_residual_relu_smallk"
            op_dict = dict()
            [op_dict.update({_n.name: _n}) for _n in self.graph.nodes if op in _n.name]

            conv_op_name = f"{op}_branch2c_conv"
            scale_op_name = f"{op}_branch2c_scale"
            bias_op_name = f"{op}_branch2c_bias"
            residual_add_op_name = f"{op}_residual_add"
            final_relu_op_name = f"{op}_relu"

            plugin_input = [op_dict[conv_op_name].inputs[0],
                            op_dict[residual_add_op_name].inputs[1]]
            plugin_output = [op_dict[final_relu_op_name].outputs[0]]

            # Get kernel parameters and weights/scale/bias
            K = op_dict[conv_op_name].inputs[1].shape[0]
            C = op_dict[conv_op_name].inputs[1].shape[1]
            weight = op_dict[conv_op_name].inputs[1]
            scale = op_dict[scale_op_name].inputs[1]
            bias = op_dict[bias_op_name].inputs[1]
            rescale = gs.Constant("rescale", values=np.ones((K), dtype=np.float32))

            # Dynamic ranges for input/output/residual (Ti/To/Tr)
            dyn_list = [
                self.dyn_range_map[op_dict[conv_op_name].inputs[0].name],
                self.dyn_range_map[op_dict[final_relu_op_name].outputs[0].name],
                self.dyn_range_map[op_dict[residual_add_op_name].inputs[1].name],
            ]
            dynamic_ranges = np.array(dyn_list, dtype=np.float32)
            dyn_const = gs.Constant("{}_dynamic_ranges".format(plugin_layer_name),
                                    values=dynamic_ranges)

            plugin_field_dict = {
                "inputChannels": gs.Constant("C", values=np.array([C], dtype=np.int32)),
                "filterDimR": gs.Constant("R", values=np.array([1], dtype=np.int32)),
                "filterDimS": gs.Constant("S", values=np.array([1], dtype=np.int32)),
                "weight": weight,
                "bias": bias,
                "scale": scale,
                "rescale": rescale,
                "dynamicRanges": dyn_const,
                "epilogueScaleBiasBetaRelu": gs.Constant("epilogue_sbbr", values=np.array([1], dtype=np.int32)),
            }

            attrs = {
                "plugin_version": "1",
                "plugin_namespace": "",
            }
            attrs.update(plugin_field_dict)

            # replace ops with plugin
            logging.info(f"Fusing {plugin_layer_name} with smallk...")
            self.graph.BETA1SmallKPlugin(plugin_op_name, plugin_layer_name, plugin_input,
                                         plugin_output, attrs)

        # graph cleanup
        self.cleanup_graph()

        # done
        logging.info("Plugin {} created successful for res2/3/4/5 branch2c".format(plugin_op_name))

    def fuse_ops(self):
        policy_selector = {
            # beta=1 smallk is officially supported through Cask since rel/5.3
            # TODO: [CFK-9235] to pick up sm90 beta=1 smallk
            '80': [],
            '87': ['Beta1Smallk'],
            '89': [],
            '90': [],
        }
        # Select the fusion policy based on the SM
        # Don't fuse the layers if we are calibrating - we need the original scales.
        if self.device_type != 'gpu' or self.need_calibration:
            policies = ['NO_FUSE']
        elif self.precision in ('int8',):
            policies = policy_selector.get(str(self.compute_sm), 'Unknown')

        # Filter out the beta=1 small-k optimization if turned off by configs
        disable_beta1_smallk = dict_get(self.retinanet_args, Fields.disable_beta1_smallk.name, default=False)
        if disable_beta1_smallk:
            if 'Beta1Smallk' in policies:
                policies.remove('Beta1Smallk')
            else:
                print('WARNING: Beta1Smallk is not enabled in the config, please remove the disable smallk flag.')

        for policy in policies:
            for _f in self.fusion_map.get(policy, []):
                self.runme(_f)

    def extract_subnetwork(self, subnetwork_gs):
        """
        TODO: add subnetwork support.
        """
        pass

    def cleanup_graph(self):
        """
        Cleanup graph and re-sort it.
        """
        self.graph.cleanup().toposort()

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--onnx-fpath',
                        type=str,
                        default='build/models/retinanet-resnext50-32x4d/submission/retinanet_resnext50_32x4d_efficientNMS.800x800.onnx',
                        help='Input ONNX file for ResNet50')
    parser.add_argument('--output-onnx-fpath',
                        type=str,
                        default='/tmp/retinanet_graphsurgeon.onnx',
                        help='Output ONNX filename')
    parser.add_argument('--calibration-cache-fpath',
                        type=str,
                        default='code/retinanet/tensorrt/calibrator.cache',
                        help='Calibration cache file')
    parser.add_argument('--compute-sm',
                        type=str,
                        required=True,
                        help='GPU Architecture of choice',
                        choices={'80', '86', '87', '89', '90'})
    parser.add_argument('--precision',
                        type=str,
                        default='int8',
                        choices={'int8', 'fp16', 'fp32'},
                        help='Compute precision')
    parser.add_argument('--is-dla',
                        default=False,
                        action='store_true',
                        help='Device is DLA')
    parser.add_argument('--nms-type',
                        default='efficientnms',
                        choices={'efficientnms', 'nmsopt', 'none'},
                        help='which type of nms to use.')
    parser.add_argument('--need-calibration',
                        default=False,
                        action='store_true',
                        help='In case calibration is required; do not fuse for example')

    args = parser.parse_args()
    for key, value in vars(args).items():
        if value is not None:
            logging.debug("Parsed args -- {}: {}".format(key, value))

    return args


def main(args):
    """
    commandline entrance of the graphsurgeon. Example commands:
        python3 -m code.retinanet.tensorrt.retinanet_graphsurgeon --compute-sm=90 --output-onnx-fpath=/home/scratch.zhihanj_sw/temp/models/retinanet_graphsurgeon_beta1.onnx --nms-type=efficientnms
    """
    device_type = 'dla' if args.is_dla else 'gpu'
    retinanet_gs = RetinanetGraphSurgeon(args.onnx_fpath, args.compute_sm, args.precision, device_type,
                                         args.calibration_cache_fpath, args.need_calibration, args.nms_type,
                                         {}, None)
    model = retinanet_gs.process_onnx()
    retinanet_gs.save_onnx(args.output_onnx_fpath)


if __name__ == '__main__':
    args = parse_args()
    main(args)
