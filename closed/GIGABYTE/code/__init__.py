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

from collections import namedtuple
from importlib import import_module

from code.common import logging
from code.common.constants import Benchmark
from code.common.arguments import apply_overrides


# Instead of storing the objects themselves in maps, we store object locations, as we do not want to import redundant
# modules on every run. Some modules include CDLLs and TensorRT plugins, or have large imports that impact runtime.
# Dynamic imports are also preferred, as some modules (due to their legacy model / directory names) include dashes.

ModuleLocation = namedtuple("ModuleLocation", ("module_path", "cls_name"))
G_BENCHMARK_CLASS_MAP = {
    Benchmark.ResNet50: ModuleLocation("code.resnet50.tensorrt.ResNet50", "ResNet50"),
    Benchmark.Retinanet: ModuleLocation("code.retinanet.tensorrt.Retinanet", "Retinanet"),
    Benchmark.BERT: ModuleLocation("code.bert.tensorrt.bert_var_seqlen", "BERTBuilder"),
    Benchmark.RNNT: ModuleLocation("code.rnnt.tensorrt.rnn-t_builder", "RNNTBuilder"),
    Benchmark.DLRM: ModuleLocation("code.dlrm.tensorrt.dlrm", "DLRMBuilder"),
    Benchmark.UNET3D: ModuleLocation("code.3d-unet.tensorrt.3d-unet", "UnetBuilder"),
}
G_HARNESS_CLASS_MAP = {
    "triton_cpu_harness": ModuleLocation("code.common.server_harness_cpu", "TritonHarnessCPU"),
    "triton_inferentia_harness": ModuleLocation("code.common.server_harness_inferentia", "TritonHarnessInferentia"),
    "triton_harness": ModuleLocation("code.common.server_harness", "TritonHarness"),
    "triton_unified_harness": ModuleLocation("code.common.triton_unified_harness", "TritonUnifiedHarness"),
    "bert_harness": ModuleLocation("code.bert.tensorrt.harness", "BertHarness"),
    "dlrm_harness": ModuleLocation("code.dlrm.tensorrt.harness", "DLRMHarness"),
    "rnnt_harness": ModuleLocation("code.rnnt.tensorrt.harness", "RNNTHarness"),
    "unet3d_harness": ModuleLocation("code.3d-unet.tensorrt.harness", "UNet3DKiTS19Harness"),
    "lwis_harness": ModuleLocation("code.common.lwis_harness", "LWISHarness"),
    "profiler_harness": ModuleLocation("code.internal.profiler", "ProfilerHarness"),
}


def get_cls(module_loc: ModuleLocation) -> type:
    """
    Returns the specified class denoted by a ModuleLocation.

    Args:
        module_loc (ModuleLocation):
            The ModuleLocation to specify the import path of the class

    Returns:
        type: The imported class located at module_loc
    """
    return getattr(import_module(module_loc.module_path), module_loc.cls_name)


def get_benchmark(conf):
    """Return module of benchmark initialized with config."""

    benchmark = conf["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning(f"'benchmark: {benchmark}' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(conf["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    if benchmark not in G_BENCHMARK_CLASS_MAP:
        raise ValueError(f"Unknown benchmark: {benchmark}")

    cls = get_cls(G_BENCHMARK_CLASS_MAP[benchmark])
    if Benchmark.BERT == benchmark:
        # Currently only BERT uses gpu_inference_streams to generate engines
        conf = apply_overrides(conf, ['gpu_inference_streams'])
    return cls(conf)


def get_harness(config, profile):
    """Refactors harness generation for use by functions other than handle_run_harness."""
    benchmark = config["benchmark"]
    if not isinstance(benchmark, Benchmark):
        logging.warning("'benchmark' in config is not Benchmark Enum member. This behavior is deprecated.")
        benchmark = Benchmark.get_match(benchmark)
        if benchmark is None:
            ttype = type(config["benchmark"])
            raise ValueError(f"'benchmark' in config is not of supported type '{ttype}'")

    if "triton_unified" in config.get("config_ver"):
        k = "triton_unified_harness"
        config["inference_server"] = "triton"
    elif config.get("use_cpu"):
        k = "triton_cpu_harness"
        config["inference_server"] = "triton"
    elif config.get("use_inferentia"):
        k = "triton_inferentia_harness"
        config["inference_server"] = "triton"
    elif config.get("use_triton"):
        k = "triton_harness"
        config["inference_server"] = "triton"
    elif Benchmark.BERT == benchmark:
        k = "bert_harness"
        config["inference_server"] = "custom"
    elif Benchmark.DLRM == benchmark:
        k = "dlrm_harness"
        config["inference_server"] = "custom"
    elif Benchmark.RNNT == benchmark:
        k = "rnnt_harness"
        config["inference_server"] = "custom"
    elif Benchmark.UNET3D == benchmark:
        k = "unet3d_harness"
        config["inference_server"] = "custom"
    else:
        k = "lwis_harness"
        config["inference_server"] = "lwis"

    harness = get_cls(G_HARNESS_CLASS_MAP[k])(config, benchmark)

    # Attempt to run profiler. Note that this is only available internally.
    if profile is not None:
        try:
            harness = get_cls(G_HARNESS_CLASS_MAP["profiler_harness"])(harness, profile)
        except BaseException:
            logging.info("Could not load profiler: Are you an internal user?")

    return harness, config
