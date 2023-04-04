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

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
import json
import onnx
import os
import tensorrt as trt

from code.common.constants import G_BUILD_DIR
from code.common.systems.system_list import DETECTED_SYSTEM, SystemClassifications
from code.bert.tensorrt.builder_utils import mark


def bert_squad_fp8_fastertransfomer(network, weights_dict, cfg, max_seq_len):
    if not SystemClassifications.is_hopper():
        raise RuntimeError("fp8 only supported on Hopper")
    plg_registry = trt.get_plugin_registry()

    pc_ft = plg_registry.get_plugin_creator("BertFp8Plugin", "1", "")

    fields = []
    fields.append(trt.PluginField("num_heads", np.array([cfg.N], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("size_per_head", np.array([cfg.H], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("num_layers", np.array([cfg.L], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("max_seq_len", np.array([max_seq_len], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("vocab_size", np.array([cfg.vocab_size], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("max_position_embeddings", np.array([cfg.max_position_embeddings], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("token_type_vocab_size", np.array([cfg.type_vocab_size], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("remove_padding", np.array([1], dtype=np.int32), trt.PluginFieldType.INT32))
    fields.append(trt.PluginField("fp8_mode", np.array([2], dtype=np.int32), trt.PluginFieldType.INT32))
    weightDirPath = os.path.join(G_BUILD_DIR, "models/bert/fp8/faster-transformer-bert-fp8-weights-scales/")
    fields.append(trt.PluginField("weightDirPath", np.array(list(weightDirPath.encode()), dtype=np.int8), trt.PluginFieldType.CHAR))
    pfc = trt.PluginFieldCollection(fields)
    ft_fp8_plugin = pc_ft.create_plugin("ft_plugin", pfc)

    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=(-1, -1))
    token_type_ids = network.add_input(name="token_type_ids", dtype=trt.int32, shape=(-1, -1))
    sequence_lengths = network.add_input(name="sequence_lengths", dtype=trt.int32, shape=(-1,))

    inputs = [input_ids, token_type_ids, sequence_lengths]
    ft_bert_layer = network.add_plugin_v2(inputs, ft_fp8_plugin)
    ft_bert_layer.name = 'ft_bert'

    # (bs, 384, 1024)
    last_embeddings = ft_bert_layer.get_output(0)
    last_embeddings.name = 'last_embeddings'

    Wsquad = weights_dict['cls_squad_output_weights']
    Bsquad = weights_dict['cls_squad_output_bias']

    #squad_output = network.add_fully_connected(embeddings, 2, Wsquad, Bsquad)
    last_embeddings_packed = network.add_shuffle(last_embeddings)
    last_embeddings_packed.reshape_dims = (-1, 1024, 1, 1)

    squad_output = network.add_convolution(last_embeddings_packed.get_output(0), 2, (1, 1), Wsquad, Bsquad)
    squad_output.name = 'squad_FC_Layer'
    logits = squad_output.get_output(0)
    logits.name = 'squad_logits'

    # output shape will be [bs * 384, 2, 1, 1]
    mark(network, logits, trt.float16)
