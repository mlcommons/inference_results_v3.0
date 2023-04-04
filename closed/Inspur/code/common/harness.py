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

import re
import os
import sys
from importlib import import_module
from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
    import tensorrt as trt

import code.common.arguments as common_args
from collections import namedtuple
from code.common import logging, dict_get, run_command, args_to_string
from code.common.constants import Benchmark, G_HIGH_ACC_ENABLED_BENCHMARKS, G_MLCOMMONS_INF_REPO_PATH, VERSION
from code.common.log_parser import from_loadgen_by_keys, scenario_loadgen_log_keys
from code.common.submission import generate_measurements_entry
from code.common.systems.base import obj_to_codestr
from code.common.systems.system_list import DETECTED_SYSTEM, SystemClassifications
from code.plugin import get_trt_plugin_paths_by_network


_new_path = [os.path.join(G_MLCOMMONS_INF_REPO_PATH, "tools", "submission")] + sys.path
with ScopedRestrictedImport(_new_path):
    submission_checker = import_module("submission_checker")
    _version_str = VERSION if VERSION in submission_checker.MODEL_CONFIG else "v2.1"
    benchmark_qsl_size_map = submission_checker.MODEL_CONFIG[_version_str]["performance-sample-count"].copy()
    # TODO: Remove this once this is merged upstream into MLCommons
    benchmark_qsl_size_map["3d-unet-99"] = 43
    benchmark_qsl_size_map["3d-unet-99.9"] = 43
    # TODO: After Slack discussion, still unknown why this is set to 2048 in our code
    benchmark_qsl_size_map["resnet"] = 2048
    benchmark_qsl_size_map["resnet50"] = benchmark_qsl_size_map["resnet"]  # submission-checker uses 'resnet' instead of 'resnet50'

    # Check for query constraints documented in https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#scenarios
    _min_queries = submission_checker.MODEL_CONFIG[_version_str]["min-queries"].copy()
    # Offline uses min. samples/query since min query count is always 1. For other scenarios, these values are the same
    # across benchmarks.
    QUERY_METRIC_CONSTRAINTS = {
        "Offline": ("effective_samples_per_query", submission_checker.OFFLINE_MIN_SPQ),
        "Server": ("effective_min_query_count", _min_queries["resnet"]["Server"]),
        "MultiStream": ("effective_min_query_count", _min_queries["resnet"]["MultiStream"]),
        "SingleStream": ("effective_min_query_count", _min_queries["resnet"]["SingleStream"]),
    }


class BaseBenchmarkHarness:
    """Base class for benchmark harnesses."""

    # subnetwork tuple including engine prefix, batch_size
    subnetwork = namedtuple('subnetwork', ['prefix', 'batch_size'])
    def __init__(self, args, benchmark):
        self.args = args
        self.benchmark = benchmark
        self.name = self.benchmark.valstr()
        self.verbose = dict_get(args, "verbose", default=None)
        self.verbose_nvtx = dict_get(args, "verbose_nvtx", default=None)
        if self.verbose:
            logging.info(f"===== Harness arguments for {self.name} =====")
            for key in args:
                logging.info("{:}={:}".format(key, args[key]))

        self.system_id = args["system_id"]
        self.scenario = args["scenario"]
        self.config_ver = args["config_ver"]
        self.engine_dir = "./build/engines/{:}/{:}/{:}".format(self.system_id, self.name, self.scenario.valstr())
        self.precision = args["precision"]
        self.has_gpu = dict_get(args, "gpu_batch_size", default=None) is not None
        self.has_dla = dict_get(args, "dla_batch_size", default=None) is not None
        self.gpu_res2res3_loop_count = dict_get(args, "gpu_res2res3_loop_count", default=None)
        self.dla_loop_count = dict_get(args, "dla_loop_count", default=None)
        self.gpu_subnetworks = None
        self.dla_subnetworks = None
        if self.gpu_res2res3_loop_count is not None and self.gpu_res2res3_loop_count < 1:
            logging.info("Overriding gpu_res2res3_loop_count to 1 when gpu_res2res3_loop_count value is invalid")
            self.args["gpu_res2res3_loop_count"] = 1
        if self.dla_loop_count is not None and self.dla_loop_count < 1:
            logging.info("Overriding dla_loop_count to 1 when dla_loop_count value is invalid")
            self.args["dla_loop_count"] = 1

        self._parse_subnetwork_loop()

        # Enumerate engine files
        # Engine not needed if we are only generating measurements/ entries
        self.skip_file_checks = dict_get(self.args, "skip_file_checks", False)
        self.gpu_engine = None
        self.dla_engine = None
        self.enumerate_engines()

        # Enumerate harness executable
        self.executable = self._get_harness_executable()
        self.check_file_exists(self.executable)

        self.use_jemalloc = False

        self.env_vars = os.environ.copy()
        self.flag_builder_custom_args = []

    def _get_harness_executable(self):
        raise NotImplementedError("BaseBenchmarkHarness cannot be called directly")

    def _build_custom_flags(self, flag_dict):
        """
        Handles any custom flags to insert into flag_dict. Can return either a flag_dict, or a converted arg string.
        """
        return flag_dict

    def _get_engine_fpath(self, device_type, batch_size):
        return "{:}/{:}-{:}-{:}-b{:}-{:}.{:}.plan".format(self.engine_dir, self.name, self.scenario.valstr(),
                                                          device_type, batch_size, self.precision, self.config_ver)

    def _append_config_ver_name(self, system_name):
        if "maxq" in self.config_ver.lower():
            system_name += "_MaxQ"
        if "hetero" in self.config_ver.lower():
            system_name += "_HeteroMultiUse"
        return system_name

    def _parse_subnetwork_loop(self):
        # Parse res3 subnetwork loop
        if self.has_gpu and self.gpu_res2res3_loop_count is not None and self.gpu_res2res3_loop_count > 1:
            # Only RN50 supports Res3 batch looping
            if self.benchmark == Benchmark.ResNet50:
                self.gpu_subnetworks = [
                    self.subnetwork('gpu-preres2', self.args["gpu_batch_size"]),
                    self.subnetwork('gpu-res2_3', self.args["gpu_batch_size"] // self.gpu_res2res3_loop_count),
                    self.subnetwork('gpu-postres3', self.args["gpu_batch_size"])]
            else:
                logging.warning(f"Only ResNet50 supports Res3 looping, using default gpu_res2res3_loop_count = 1")

        # Parse DLA subnetwork loop
        if self.has_dla and self.dla_loop_count is not None and self.dla_loop_count > 1:
            # Only RN50 supports DLA batch looping
            if self.benchmark == Benchmark.ResNet50:
                self.dla_subnetworks = [
                    self.subnetwork('dla-dla', self.args["dla_batch_size"]),
                    self.subnetwork('dla-topk', self.args["dla_batch_size"] * self.dla_loop_count)]
            else:
                logging.warning(f"Only ResNet50 supports DLA looping, using default dla_loop_count = 1")

    def get_system_name(self, add_trt=True):
        override_system_name = dict_get(self.args, "system_name", default=None)
        if override_system_name not in {None, ""}:
            return override_system_name

        system_name = self.system_id

        if add_trt:
            system_name = f"{system_name}_TRT"

        return self._append_config_ver_name(system_name)

    def _get_submission_benchmark_name(self):
        full_benchmark_name = self.name
        if dict_get(self.args, "accuracy_level", "99%") == "99.9%":
            full_benchmark_name += "-99.9"
        elif self.name in G_HIGH_ACC_ENABLED_BENCHMARKS:
            full_benchmark_name += "-99"
        return full_benchmark_name

    def get_full_log_dir(self):
        return os.path.join(self.args["log_dir"], self.get_system_name(), self._get_submission_benchmark_name(),
                            self.scenario.valstr())

    def enumerate_engines(self):
        if self.has_gpu:
            if self.gpu_subnetworks is not None and self.gpu_res2res3_loop_count > 1:
                # Append subnework GPU engines and concat them with separator ','
                gpu_engine_list = []
                for subnetwork in self.gpu_subnetworks:
                    engine_path = self._get_engine_fpath(subnetwork.prefix, subnetwork.batch_size)
                    self.check_file_exists(engine_path)
                    gpu_engine_list.append(engine_path)
                self.gpu_engine = ','.join(gpu_engine_list)
            else:
                self.gpu_engine = self._get_engine_fpath("gpu", self.args["gpu_batch_size"])
                self.check_file_exists(self.gpu_engine)

        if self.has_dla:
            if self.dla_subnetworks is not None and self.dla_loop_count > 1:
                # Append subnework DlA engines and concat them with separator ','
                dla_engine_list = []
                for subnetwork in self.dla_subnetworks:
                    engine_path = self._get_engine_fpath(subnetwork.prefix, subnetwork.batch_size)
                    self.check_file_exists(engine_path)
                    dla_engine_list.append(engine_path)
                self.dla_engine = ','.join(dla_engine_list)
            else:
                self.dla_engine = self._get_engine_fpath("dla", self.args["dla_batch_size"])
                self.check_file_exists(self.dla_engine)

    def check_file_exists(self, f):
        """Check if file exists. Complain if configured to do so."""

        if not os.path.isfile(f):
            if self.skip_file_checks:
                print("Note: File {} does not exist. Attempting to continue regardless, as hard file checks are disabled.".format(f))
                return False
            else:
                raise RuntimeError("File {:} does not exist.".format(f))
        return True

    def build_default_flags(self):
        flag_dict = {}
        flag_dict["verbose"] = self.verbose
        flag_dict["verbose_nvtx"] = self.verbose_nvtx

        # Handle plugins
        plugins = get_trt_plugin_paths_by_network(self.name, self.args)
        if len(plugins) > 0:
            logging.info(f"The harness will load {len(plugins)} plugins: {plugins}")
            flag_dict["plugins"] = ",".join(plugins)

        # Generate flags for logfile names.
        log_dir = self.get_full_log_dir()
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        flag_dict["logfile_outdir"] = log_dir
        flag_dict["logfile_prefix"] = "mlperf_log_"

        # Handle performance sample count
        perf_sample_count = dict_get(self.args, "performance_sample_count", None)
        perf_sample_count_override = dict_get(self.args, "performance_sample_count_override", None)
        if perf_sample_count_override is not None:
            flag_dict["performance_sample_count"] = perf_sample_count_override
        elif perf_sample_count is not None:
            flag_dict["performance_sample_count"] = perf_sample_count
        elif benchmark_qsl_size_map[self._get_submission_benchmark_name()] > 0:
            flag_dict["performance_sample_count"] = benchmark_qsl_size_map[self._get_submission_benchmark_name()]
        else:
            flag_dict["performance_sample_count"] = self.args["gpu_batch_size"]

        # Handle custom arguments
        for arg in self.flag_builder_custom_args:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_scenario_specific_flags(self):
        """Return flags specific to current scenario."""

        flag_dict = {}

        scenario_keys = common_args.getScenarioMetricArgs(self.scenario)

        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is None:
                raise ValueError("Missing required key {:}".format(arg))
            flag_dict[arg] = val

        # Handle RUN_ARGS
        for arg in scenario_keys:
            val = dict_get(self.args, arg, None)
            if val is not None:
                flag_dict[arg] = val

        return flag_dict

    def build_non_custom_flags(self):
        """Returns the flag_dict for all flags excluding custom ones.
        """
        flag_dict = self.build_default_flags()
        flag_dict.update(self.build_scenario_specific_flags())

        # Handle engines
        if self.has_gpu:
            flag_dict["gpu_engines"] = self.gpu_engine

        # MLPINF-853: Special handing of --fast. Use min_duration=60000 and min_query_count=1.
        if flag_dict.get("fast", False):
            if "min_duration" not in flag_dict:
                flag_dict["min_duration"] = 60000
            if "min_query_count" not in flag_dict:
                flag_dict["min_query_count"] = 1
            flag_dict["fast"] = None
        return flag_dict

    def prepend_ld_preload(self, so_path):
        if "LD_PRELOAD" in self.env_vars:
            self.env_vars["LD_PRELOAD"] = ":".join([so_path, self.env_vars["LD_PRELOAD"]])
        else:
            self.env_vars["LD_PRELOAD"] = so_path

        logging.info("Updated LD_PRELOAD: " + self.env_vars["LD_PRELOAD"])

    def run_harness(self, flag_dict=None, skip_generate_measurements=False):
        if flag_dict is None:
            flag_dict = self.build_non_custom_flags()

        if not skip_generate_measurements:
            # Generates the entries in the `measurements/` directory, and updates flag_dict accordingly
            generate_measurements_entry(
                self.get_system_name(),
                self.name,
                self._get_submission_benchmark_name(),
                self.scenario,
                self.args["input_dtype"],
                self.args["precision"],
                flag_dict)

        argstr = self._build_custom_flags(flag_dict)
        if type(argstr) is dict:
            argstr = args_to_string(flag_dict)

        # Handle environment variables
        if self.use_jemalloc:
            import platform
            self.prepend_ld_preload(f"/usr/lib/{platform.processor()}-linux-gnu/libjemalloc.so.2")

        cmd = "{:} {:}".format(self.executable, argstr)
        output = run_command(cmd, get_output=True, custom_env=self.env_vars)

        # Return harness result.
        scenario_key = scenario_loadgen_log_keys[self.scenario]
        results = from_loadgen_by_keys(os.path.join(self.args["log_dir"],
                                                    self.get_system_name(),
                                                    self._get_submission_benchmark_name(),
                                                    self.scenario.valstr()),
                                       ["result_validity", scenario_key, "early_stopping_met"])
        test_mode = flag_dict.get("test_mode", "PerformanceOnly")
        satisfies_query_constraint = float(results.get(scenario_key, "0.0")) >= QUERY_METRIC_CONSTRAINTS[self.scenario.valstr()][1]
        results.update({"system_name": self.get_system_name(),
                        "benchmark_short": self.benchmark.valstr(),
                        "benchmark_full": self._get_submission_benchmark_name(),
                        "scenario": self.scenario.valstr(),
                        "test_mode": test_mode,
                        "tensorrt_version": trt.__version__,
                        "detected_system": obj_to_codestr(DETECTED_SYSTEM),
                        "scenario_key": scenario_key,
                        "satisfies_query_constraint": satisfies_query_constraint})

        # Special DLRM thing
        if self.benchmark == Benchmark.DLRM and test_mode == "PerformanceOnly":
            partitions = np.load(os.path.expandvars(self.args["sample_partition_path"]))
            partition_mean_size = np.mean(partitions[1:] - partitions[:-1])
            results["dlrm_partition_mean_size"] = partition_mean_size
            results["dlrm_pairs_per_second"] = results[scenario_key] * partition_mean_size
        return results
