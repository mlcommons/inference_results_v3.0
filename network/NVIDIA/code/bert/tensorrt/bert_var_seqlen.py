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

# TODO for now, this is loads the precompiled dev versions of the BERT TRT plugins from yko's TRT fork

import os
from importlib import import_module
from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import pycuda
    import pycuda.autoinit
    import tensorrt as trt

from code.common import logging, dict_get
from code.common.builder import TensorRTEngineBuilder
from code.common.constants import Benchmark
from code.common.systems.system_list import DETECTED_SYSTEM, SystemClassifications
from code.bert.tensorrt.builder_utils import BertConfig, get_onnx_fake_quant_weights
from code.bert.tensorrt.int8_builder_var_seqlen import bert_squad_int8_var_seqlen
from code.bert.tensorrt.int8_builder_vs_il import bert_squad_int8_vs_il
from code.bert.tensorrt.fp16_builder_var_seqlen import bert_squad_fp16_var_seqlen

# to run with a different seq_len, we need to run preprocessing again and point to the resulting folder
# by setting the variable:
# PREPROCESSED_DATA_DIR=/data/projects/bert/squad/v1.1/s128_q64_d128/


class SquadLogitsTacticSelector(trt.IAlgorithmSelector):
    def select_algorithms(self, ctx, choices):
        if "squad_logits" in ctx.name:  # Apply to all profiles
            # MLPINF-596. For now, there's only one troublesome tactic:
            # turing_fp16_i8816cudnn_int8_256x64_ldg16_relu_singleBuffer_small_nt_v1
            # [TensorRT] VERBOSE: Tactic: 3524082626922414020 time 0.033088
            forbidden_set = {3524082626922414020}
            filtered_idxs = [idx for idx, choice in enumerate(choices) if choice.algorithm_variant.tactic not in forbidden_set]
            to_ret = filtered_idxs
        else:
            # By default, say that all tactics are acceptable:
            to_ret = [idx for idx, alg in enumerate(choices)]
        return to_ret

    def report_algorithms(self, ctx, choices):

        pass


class BERTBuilder(TensorRTEngineBuilder):
    """To build engines in lwis mode, we expect a single sequence length and a single batch size."""

    def __init__(self, args):
        workspace_size = dict_get(args, "workspace_size", default=(5 << 30))
        logging.info(f"Using workspace size: {workspace_size}")
        super().__init__(args, Benchmark.BERT, workspace_size=workspace_size)
        self.bert_config_path = "code/bert/tensorrt/bert_config.json"

        self.seq_len = 384  # default sequence length

        # The opt_shape provided to TRT builder in the optimization profile
        self.bert_opt_seqlen = dict_get(args, "bert_opt_seqlen", default=self.seq_len)

        self.batch_size = dict_get(args, "batch_size", default=1)

        self.num_profiles = 1
        if 'gpu_inference_streams' in args:
            # use gpu_inference_streams to determine the number of duplicated profiles
            # in the engine when not using lwis mode
            self.num_profiles = args['gpu_inference_streams']

        self.is_int8 = args['precision'] == 'int8'

        if self.is_int8:
            self.model_path = dict_get(args, "model_path", default="build/models/bert/bert_large_v1_1_fake_quant.onnx")
        else:
            self.model_path = dict_get(args, "model_path", default="build/models/bert/bert_large_v1_1.onnx")

        self.bert_config = BertConfig(self.bert_config_path)

        self.enable_interleaved = False
        if self.is_int8 and 'enable_interleaved' in args:
            self.enable_interleaved = args['enable_interleaved']

        # Small-Tile GEMM Plugin
        # Since it doesn't support interleaved format, two options are mutually exclusive
        self.use_small_tile_gemm_plugin = self.args.get("use_small_tile_gemm_plugin", False)
        if self.enable_interleaved and self.use_small_tile_gemm_plugin:
            assert False, "Small-Tile GEMM Plugin doesn't support interleaved format."

        if self.batch_size > 512:
            # tactics selection is limited at very large batch sizes
            self.builder_config.max_workspace_size = 7 << 30

    def initialize(self):
        self.initialized = True

    def _get_engine_fpath(self, device_type, batch_size):
        """Get engine file path given the config for this model."""
        if device_type is None:
            device_type = self.device_type

        base_name = f"{self.name}-{self.scenario.valstr()}-{device_type}-{self.precision}"
        bert_metadata = f"S_{self.seq_len}_B_{self.batch_size}_P_{self.num_profiles}_vs"
        if self.enable_interleaved:
            bert_metadata += "_il"
        return f"{self.engine_dir}/{base_name}_{bert_metadata}.{self.config_ver}.plan"

    def build_engines(self):
        """
        Calls self.initialize() if it has not been called yet.
        Creates optimization profiles for multiple SeqLen and BatchSize combinations
        Builds and saves the engine.
        TODO do we also need multiple profiles per setting?
        """

        # Load weights
        weights_dict = get_onnx_fake_quant_weights(self.model_path)

        if not self.initialized:
            self.initialize()

        # Create output directory
        os.makedirs(self.engine_dir, exist_ok=True)

        input_shape = (-1, )
        cu_seqlen_shape = (-1,)

        self.profiles = []

        with self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network:

            # Looks like the tactics available with even large WS are not competitive anyway.
            # Might be able to reduce this also

            self.builder_config.set_flag(trt.BuilderFlag.FP16)
            if self.is_int8:
                self.builder_config.set_flag(trt.BuilderFlag.INT8)
                if self.enable_interleaved:
                    bert_squad_int8_vs_il(network, weights_dict, self.bert_config, input_shape, cu_seqlen_shape)
                else:
                    bert_squad_int8_var_seqlen(network, weights_dict, self.bert_config,
                                               input_shape, cu_seqlen_shape, self.use_small_tile_gemm_plugin)
            else:
                bert_squad_fp16_var_seqlen(network, weights_dict, self.bert_config, input_shape, cu_seqlen_shape)

            engine_name = self._get_engine_fpath(self.device_type, None)
            logging.info(f"Building {engine_name}")

            # The harness expects i -> S -> B. This should be fine, since now there is only one S per engine
            for i in range(self.num_profiles):
                profile = self.builder.create_optimization_profile()
                assert network.num_inputs == 4, "Unexpected number of inputs"
                assert network.get_input(0).name == 'input_ids'
                assert network.get_input(1).name == 'segment_ids'
                assert network.get_input(2).name == 'cu_seqlens'
                assert network.get_input(3).name == 'max_seqlen'

                B = self.batch_size
                S = self.seq_len
                # TODO Like this, we can only control granularity using multiples of max_seqlen (B*S)
                # Investigate if this can be improved otherwise
                min_shape = (1,)  # TODO is it an issue to cover such a wide range?
                max_shape = (B * S,)
                opt_shape = (B * self.bert_opt_seqlen,)
                profile.set_shape('input_ids', min_shape, opt_shape, max_shape)
                profile.set_shape('segment_ids', min_shape, opt_shape, max_shape)
                profile.set_shape('cu_seqlens', (1 + 1,), (B + 1,), (B + 1,))
                profile.set_shape('max_seqlen', (1,), (S,), (S,))
                if not profile:
                    raise RuntimeError("Invalid optimization profile!")
                self.builder_config.add_optimization_profile(profile)
                self.profiles.append(profile)

            # Apply tactic selector/filter for Turing GPUs:
            if SystemClassifications.is_turing():
                tactic_selector = SquadLogitsTacticSelector()
                self.builder_config.algorithm_selector = tactic_selector

            # Build engines
            engine = self.builder.build_engine(network, self.builder_config)
            assert engine is not None, "Engine Build Failed!"
            buf = engine.serialize()
            with open(engine_name, 'wb') as f:
                f.write(buf)

    # BERT does not need calibration.
    def calibrate(self):
        logging.info("BERT does not need calibration.")
