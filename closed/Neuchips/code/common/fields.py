# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

from __future__ import annotations
from argparse import ArgumentParser
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any, Dict, Tuple, Optional

import os
import sys
sys.path.insert(0, os.getcwd())

from code.common.constants import *
from code.common.systems.system_list import KnownSystem, SystemClassifications


_LWIS_BENCHMARKS = (Benchmark.ResNet50,
                    Benchmark.SSDMobileNet,
                    Benchmark.SSDResNet34,
                    Benchmark.UNET3D)
_CV_BENCHMARKS = (Benchmark.ResNet50,
                  Benchmark.SSDMobileNet,
                  Benchmark.SSDResNet34)
_HARNESS_ACTIONS = (Action.RunHarness,
                    Action.RunAuditHarness)


@dataclass(frozen=True)
class Field:
    """
    Represents a configuration parameter used for any benchmark workload.
    """

    name: str
    """str: Identifier for this field"""

    description: str
    """str: Description of this field's meaning and usage"""

    value_type: type
    """type: The expected type of the value of the field"""

    default: Any = None
    """Any: The default value of this Field"""

    supported_actions: Tuple[Action, ...] = tuple(Action)
    """Tuple[Action, ...]: Actions this Field is used in"""

    supported_benchmarks: Tuple[Benchmark, ...] = tuple(Benchmark)
    """Tuple[Benchmark, ...]: Benchmarks this Field is used in"""

    supported_scenarios: Tuple[Scenario, ...] = tuple(Scenario)
    """Tuple[Scenario, ...]: Scenarios this Field is used in"""

    supported_systems: Tuple[Callable[KnownSystem, bool], ...] = (lambda s: True,)
    """Tuple[System, ...]: Systems this Field is used in"""

    supported_harnesses: Tuple[HarnessType, ...] = tuple(HarnessType)
    """Tuple[HarnessType, ...]: Harnesses this Field can be used in"""

    supported_power_settings: Tuple[PowerSetting, ...] = tuple(PowerSetting)
    """Tuple[HarnessType, ...]: Harnesses this Field can be used in"""

    required: bool = False
    """bool: Whether or not this Field is required to run the workload"""

    no_argparse: bool = False
    """bool: If True, this Field should not be added to argparsers"""

    argparse_opts: Optional[Dict[str, Any]] = None
    """Dict[str, Any]: Optional kwargs to use for Argparse.Argument"""

    def add_to_argparser(self, argparser: ArgumentParser, allow_argparse_default=False):
        """
        Adds this field to an argparser as a command line argument to be parsed. If this Field has a value_type of bool,
        it is treated as a store_true argument (i.e. --verbose).

        If no_argparse is True, this method returns immediately without adding to the argparser.

        Args:
            argparser (argparse.ArgumentParser):
                The argparser to add this field to as an argument.
            allow_argparse_default (bool):
                If False, disallows using 'default' in argparse_opts. This is useful to prevent bugs where a default
                argparse value will override any changes made in the BenchmarkConfiguration, as CLI overrides take the
                highest precedence for runtime arguments. Default: False.
        """
        if self.no_argparse:
            return

        kwargs = dict() if self.argparse_opts is None else self.argparse_opts
        if not allow_argparse_default and "default" in kwargs:
            raise RuntimeError(" ".join([
                f"Field({self.name}) contains 'default' in argparse_opts, which is disallowed.",
                "If you wish to specify a default value, use the Field.default instead."
            ]))

        if self.value_type is bool:
            argparser.add_argument(f"--{self.name}", help=self.description, action="store_true", **kwargs)
        else:
            argparser.add_argument(f"--{self.name}", help=self.description, type=self.value_type, **kwargs)

    def supports(self, action: Optional[Action], benchmark: Benchmark, scenario: Scenario,
                 system: KnownSystem, workload_setting: WorkloadSetting) -> bool:
        """
        Whether or not this Field supports the given workload.

        Args:
            action (Optional[Action]):
                The Action of the workload. If None, unused during the check.
            benchmark (Benchmark):
                The Benchmark of the workload
            scenario (Scenario):
                The Scenario of the workload
            system (KnownSystem):
                The System running the workload
            workload_setting (WorkloadSetting):
                The WorkloadSetting field of the config to get fields for

        Returns:
            bool: True if all of the parameters of the workload are supported by this Field. False otherwise.
            harness_type is only checked if `action` is Action.RunHarness, Action.RunAuditHarness, or None.
        """
        return (action is None or action in self.supported_actions) and \
            benchmark in self.supported_benchmarks and \
            scenario in self.supported_scenarios and \
            all(f(system) for f in self.supported_systems) and \
            (action not in (list(_HARNESS_ACTIONS) + [None]) or
                workload_setting.harness_type in self.supported_harnesses) and \
            workload_setting.power_setting in self.supported_power_settings


@unique
class Fields(Enum):
    """
    Defines a set of known Fields for BenchmarkConfigurations.
    """

    gpu_batch_size: Field = Field(
        "gpu_batch_size",
        "Batch size to use for the GPU.",
        supported_systems=(SystemClassifications.gpu_based,),
        value_type=int,
        required=True)

    #dla_batch_size: Field = Field(
    #    "dla_batch_size",
    #    "Batch size to use for the DLA. Xavier-only argument.",
    #    value_type=int,
    #    supported_benchmarks=_CV_BENCHMARKS,
    #    supported_scenarios=(Scenario.Offline,),
    #    supported_systems=(SystemClassifications.is_soc,),
    #    required=True)

    verbose: Field = Field(
        "verbose",
        "Whether to use verbose output.",
        value_type=bool)

    verbose_nvtx: Field = Field(
        "verbose_nvtx",
        "Turn ProfilingVerbosity to kVERBOSE so NVTX prints layer detail. Used only when profiling.",
        value_type=bool)

    workspace_size: Field = Field(
        "workspace_size",
        "The maximum size (in bytes) of temporary workspace that any layer in the network can use in TRT engine builder.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,))

    power_limit: Field = Field(
        "power_limit",
        "Set power upper limit to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,))

    cpu_freq: Field = Field(
        "cpu_freq",
        "Set cpu frequency to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,))

    '''
    soc_gpu_freq: Field = Field(
        "soc_gpu_freq",
        "Set power upper limit to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_dla_freq: Field = Field(
        "soc_dla_freq",
        "Set dla clock to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_cpu_freq: Field = Field(
        "soc_cpu_freq",
        "Set cpu clock to the specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    soc_emc_freq: Field = Field(
        "soc_emc_freq",
        "Set emc clock to specified value.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))

    orin_num_cores: Field = Field(
        "orin_num_cores",
        "Enable orin_num_cores CPU cores on Orin, turn off others.",
        value_type=int,
        supported_power_settings=(PowerSetting.MaxQ,),
        supported_systems=(SystemClassifications.is_soc,))
    '''

    data_dir: Field = Field(
        "data_dir",
        "Directory containing raw, unprocessed dataset.",
        value_type=str)

    preprocessed_data_dir: Field = Field(
        "preprocessed_data_dir",
        "Directory containing preprocessed dataset.",
        value_type=str)

    precision: Field = Field(
        "precision",
        "Precision to use for the network",
        value_type=str,
        required=True,
        argparse_opts={"choices": Precision.as_strings()})

    input_dtype: Field = Field(
        "input_dtype",
        "Precision of the input",
        value_type=str,
        required=True,
        argparse_opts={"choices": Precision.as_strings()})

    input_format: Field = Field(
        "input_format",
        "Format/layout of the input",
        value_type=str,
        required=True,
        supported_systems=(SystemClassifications.gpu_based,),
        argparse_opts={"choices": InputFormats.as_strings()})

    audio_fp16_input: Field = Field(
        "audio_fp16_input",
        "Is input format for raw audio in fp16?",
        value_type=bool,
        supported_benchmarks=(Benchmark.RNNT,))

    force_calibration: Field = Field(
        "force_calibration",
        "Run quantization calibration, even if the cache exists.",
        value_type=bool,
        supported_actions=(Action.GenerateEngines, Action.Calibrate))

    calib_batch_size: Field = Field(
        "calib_batch_size",
        "Batch size to use when calibrating",
        value_type=int,
        supported_actions=(Action.Calibrate,))

    calib_max_batches: Field = Field(
        "calib_max_batches",
        "Number of batches to run for calibration.",
        value_type=int,
        supported_actions=(Action.Calibrate,))

    cache_file: Field = Field(
        "cache_file",
        "Path to calibration cache.",
        value_type=str,
        supported_actions=(Action.GenerateEngines, Action.Calibrate))

    calib_data_map: Field = Field(
        "calib_data_map",
        "Path to the data map of the calibration set.",
        value_type=str,
        supported_actions=(Action.Calibrate,))

    benchmark: Field = Field(
        "benchmark",
        "Name of the benchmark.",
        value_type=str,
        argparse_opts={"choices": Benchmark.as_strings()},
        required=True)

    scenario: Field = Field(
        "scenario",
        "Name of the scenario.",
        value_type=str,
        argparse_opts={"choices": Scenario.as_strings()},
        required=True)

    system: Field = Field(
        "system",
        "System object. Note this cannot be passed in via CLI, but is used to validate BenchmarkConfiguration objects.",
        value_type=str,
        no_argparse=True,
        required=True)

    #dla_core: Field = Field(
    #    "dla_core",
    #    "DLA Core ID to use.",
    #    value_type=int,
    #    supported_actions=(Action.GenerateEngines, Action.RunHarness, Action.RunAuditHarness),
    #    supported_benchmarks=_CV_BENCHMARKS,
    #    supported_scenarios=(Scenario.Offline,),
    #    supported_systems=(SystemClassifications.is_soc,))

    model_path: Field = Field(
        "model_path",
        "Path to the model weights.",
        value_type=str)

    active_sms: Field = Field(
        "active_sms",
        "Percentage of active SMs while generating engines.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,))

    log_dir: Field = Field(
        "log_dir",
        "Directory for all output logs.",
        value_type=str)

    use_graphs: Field = Field(
        "use_graphs",
        "Enable CUDA graphs.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.gpu_based,))

    nopipelined_execution: Field = Field(
        "nopipelined_execution",
        """Disable pipelined execution. RNNT Only. Pipelined Execution should be enabled for Offline/Server, disabled for
        SingleStream""",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    nobatch_sorting: Field = Field(
        "nobatch_sorting",
        "Disable batch sorting by sequence length",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    noenable_audio_processing: Field = Field(
        "noenable_audio_processing",
        "Disable DALI preprocessing and fallback to preprocessed npy files",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    nouse_copy_kernel: Field = Field(
        "nouse_copy_kernel",
        "Disable using DALI's scatter gather kernel instead of using cudamemcpyAsync",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    num_warmups: Field = Field(
        "num_warmups",
        """Number of samples to warmup on. A value of -1 runs two full batches for each stream
        (2*batch_size*streams_per_gpu*NUM_GPUS), 0 turns off warmups. (Default: -1)""",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    max_seq_length: Field = Field(
        "max_seq_length",
        "Max sequence length for audio. (Default: 128)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    audio_batch_size: Field = Field(
        "audio_batch_size",
        "Batch size for DALI's processing. (Default: 256)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    audio_buffer_num_lines: Field = Field(
        "audio_buffer_num_lines",
        "Number of audio samples in flight for DALI's processing. (Default: 4096)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    dali_batches_issue_ahead: Field = Field(
        "dali_batches_issue_ahead",
        "Number of batches for which cudamemcpy is issued ahead of DALI compute. (Default: 4)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    dali_pipeline_depth: Field = Field(
        "dali_pipeline_depth",
        "Depth of sub-batch processing in DALI pipeline. (Default: 4)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    disable_encoder_plugin: Field = Field(
        "disable_encoder_plugin",
        "Disable the INT8 Encoder TRT plugin and use the fallback TRT API for Encoder",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.RNNT,))

    use_small_tile_gemm_plugin: Field = Field(
        "use_small_tile_gemm_plugin",
        "Enable the Small Time GEMM Plugin for TensorRT",
        value_type=bool,
        supported_benchmarks=(Benchmark.BERT, Benchmark.DLRM),
        supported_systems=(SystemClassifications.gpu_based,))

    gemm_plugin_fairshare_cache_size: Field = Field(
        "gemm_plugin_fairshare_cache_size",
        "Cache size (i.e. number of cache entries) for the FairShare cache in the GEMM Plugin",
        value_type=int,
        supported_benchmarks=(Benchmark.BERT, Benchmark.DLRM),
        supported_systems=(SystemClassifications.gpu_based,))

    enable_interleaved: Field = Field(
        "enable_interleaved",
        """Use interleaved data format in the TensorRT engine:
        Run the inference in channel-leading format, i.e. [1, C, Sum(seq_lens), 1], instead of the normal sequence-leading format, i.e. [Sum(seq_lens), C, 1, 1].""",
        value_type=bool,
        supported_benchmarks=(Benchmark.BERT,),
        supported_systems=(SystemClassifications.gpu_based,))

    devices: Field = Field(
        "devices",
        "Comma-separated list of numbered devices.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.ResNet50, Benchmark.SSDMobileNet, Benchmark.SSDResNet34, Benchmark.UNET3D,
                              Benchmark.BERT, Benchmark.RNNT))  # For some reason only harness_dlrm does not support this.

    map_path: Field = Field(
        "map_path",
        "Path to map file for samples. Not used if coalesced_tensor is True (BERT, DLRM).",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.ResNet50, Benchmark.SSDMobileNet, Benchmark.SSDResNet34, Benchmark.UNET3D,
                              Benchmark.RNNT),
        required=True)

    tensor_path: Field = Field(
        "tensor_path",
        "Path to preprocessed samples in .npy format",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        required=True)

    performance_sample_count: Field = Field(
        "performance_sample_count",
        "Number of samples to load in performance set. 0=use default",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    performance_sample_count_override: Field = Field(
        "performance_sample_count_override",
        "Number of samples to load in performance set; overriding performance_sample_count. 0=don't override",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    gpu_copy_streams: Field = Field(
        "gpu_copy_streams",
        """Number of copy streams to use for GPU. Used in generate_engines as well to determine number of TensorRT
        optimization profiles""",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,))

    gpu_inference_streams: Field = Field(
        "gpu_inference_streams",
        """Number of inference streams to use for GPU. Used in generate_engines as well to determine number of TensorRT
        optimization profiles""",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS + (Action.GenerateEngines,),
        supported_systems=(SystemClassifications.gpu_based,))

    #dla_copy_streams: Field = Field(
    #    "dla_copy_streams",
    #    "Number of copy streams to use for DLA",
    #    value_type=int,
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_benchmarks=_CV_BENCHMARKS,
    #    supported_scenarios=(Scenario.Offline,),
    #    supported_systems=(SystemClassifications.is_soc,))

    #dla_inference_streams: Field = Field(
    #    "dla_inference_streams",
    #    "Number of inference streams to use for DLA",
    #    value_type=int,
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_benchmarks=_CV_BENCHMARKS,
    #    supported_scenarios=(Scenario.Offline,),
    #    supported_systems=(SystemClassifications.is_soc,))

    run_infer_on_copy_streams: Field = Field(
        "run_infer_on_copy_streams",
        "Run inference on copy streams",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    warmup_duration: Field = Field(
        "warmup_duration",
        "Minimum duration to perform warmup for",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.DLRM,))

    #use_direct_host_access: Field = Field(
    #    "use_direct_host_access",
    #    "Use direct access to host memory for all devices. (Xavier Unified Memory)",
    #    value_type=bool,
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_benchmarks=_CV_BENCHMARKS + (Benchmark.UNET3D,),
    #    supported_systems=(SystemClassifications.is_soc,))

    use_deque_limit: Field = Field(
        "use_deque_limit",
        "Use a max number of elements dequed from work queue (LWIS only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS)

    deque_timeout_usec: Field = Field(
        "deque_timeout_usec",
        "Timeout in us for deque from work queue (LWIS only)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.BERT, Benchmark.DLRM))

    use_batcher_thread_per_device: Field = Field(
        "use_batcher_thread_per_device",
        "Enable a separate batcher thread per device",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS)

    use_cuda_thread_per_device: Field = Field(
        "use_cuda_thread_per_device",
        "Enable a separate cuda thread per device",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS,
        supported_systems=(SystemClassifications.gpu_based,))

    start_from_device: Field = Field(
        "start_from_device",
        "Assuming that inputs start from device memory in QSL",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.start_from_device_enabled,))

    end_on_device: Field = Field(
        "end_on_device",
        "Allows output to remain device memory for QuerySampleComplete. As of v1.1, only supported on 3D-UNET.",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.UNET3D,),
        supported_systems=(SystemClassifications.end_on_device_enabled,))

    #max_dlas: Field = Field(
    #    "max_dlas",
    #    "Max number of DLAs to use per device",
    #    value_type=int,
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_benchmarks=_CV_BENCHMARKS,
    #    supported_systems=(SystemClassifications.is_soc,))

    coalesced_tensor: Field = Field(
        "coalesced_tensor",
        "Turn on if all the samples are coalesced into one single npy file (LWIS Only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.BERT, Benchmark.DLRM))

    assume_contiguous: Field = Field(
        "assume_contiguous",
        "Assume that the data in a query is already contiguous (LWIS Only)",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=tuple(set(_LWIS_BENCHMARKS) - set((Benchmark.UNET3D,))))

    complete_threads: Field = Field(
        "complete_threads",
        "Number of threads per device for sending responses",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS + (Benchmark.DLRM,))

    use_same_context: Field = Field(
        "use_same_context",
        "Use the same TRT context for all copy streams (shape must be static and gpu_inference_streams must be 1).",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=_LWIS_BENCHMARKS)

    # TODO: From here down, I stopped checking for supported_* fields to save time. Add these after.
    unet3d_sw_gaussian_patch_path: Field = Field(
        "unet3d_sw_gaussian_patch_path",
        "Path to the numpy file holding preconditioned Gaussian patches used in 3D-UNet-KiTS19.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.UNET3D,))

    mlperf_conf_path: Field = Field(
        "mlperf_conf_path",
        "Path to mlperf.conf",
        value_type=str)

    user_conf_path: Field = Field(
        "user_conf_path",
        "Path to user.conf",
        value_type=str)

    test_mode: Field = Field(
        "test_mode",
        "Testing mode for Loadgen",
        value_type=str,
        argparse_opts={"choices": ["SubmissionRun", "AccuracyOnly", "PerformanceOnly", "FindPeakPerformance"]})

    min_duration: Field = Field(
        "min_duration",
        "Minimum test duration",
        value_type=int)

    max_duration: Field = Field(
        "max_duration",
        "Maximum test duration",
        value_type=int)

    min_query_count: Field = Field(
        "min_query_count",
        "Minimum number of queries in test",
        value_type=int)

    max_query_count: Field = Field(
        "max_query_count",
        "Maximum number of queries in test",
        value_type=int)

    qsl_rng_seed: Field = Field(
        "qsl_rng_seed",
        "Seed for RNG that specifies which QSL samples are chosen for performance set and the order in which samples are processed in AccuracyOnly mode",
        value_type=int)

    sample_index_rng_seed: Field = Field(
        "sample_index_rng_seed",
        "Seed for RNG that specifies order in which samples from performance set are included in queries",
        value_type=int)

    use_spin_wait: Field = Field(
        "use_spin_wait",
        "Use spin waiting for LWIS. Recommended for single stream",
        value_type=bool)

    # TODO: I think 'fast' needs to be handled specially, but not sure.
    fast: Field = Field(
        "fast",
        "If set, will set min_duration to 1 minute (60000ms). For Offline and Server, min_query_count is set to 1.",
        value_type=bool)

    logfile_suffix: Field = Field(
        "logfile_suffix",
        "Specify the filename suffix for the LoadGen log files",
        value_type=str)

    logfile_prefix_with_datetime: Field = Field(
        "logfile_prefix_with_datetime",
        "Prefix filenames for LoadGen log files",
        value_type=bool)

    log_copy_detail_to_stdout: Field = Field(
        "log_copy_detail_to_stdout",
        "Copy LoadGen detailed logging to stdout",
        value_type=bool)

    disable_log_copy_summary_to_stdout: Field = Field(
        "disable_log_copy_summary_to_stdout",
        "Disable copy LoadGen summary logging to stdout",
        value_type=bool)

    log_mode: Field = Field(
        "log_mode",
        "Logging mode for Loadgen",
        value_type=str,
        argparse_opts={"choices": ["AsyncPoll", "EndOfTestOnly", "Synchronous"]})

    log_mode_async_poll_interval_ms: Field = Field(
        "log_mode_async_poll_interval_ms",
        "Specify the poll interval for asynchrounous logging",
        value_type=int)

    log_enable_trace: Field = Field(
        "log_enable_trace",
        "Enable trace logging",
        value_type=bool)

    use_triton: Field = Field(
        "use_triton",
        "Use Triton harness",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        required=True)

    preferred_batch_size: Field = Field(
        "preferred_batch_size",
        "Preferred batch sizes",
        value_type=str)

    max_queue_delay_usec: Field = Field(
        "max_queue_delay_usec",
        "Set max queuing delay for Triton in usec.",
        value_type=int)

    instance_group_count: Field = Field(
        "instance_group_count",
        "Set number of instance groups on each GPU.",
        value_type=int)

    request_timeout_usec: Field = Field(
        "request_timeout_usec",
        "Set the timeout for every request in usec.",
        value_type=int)

    buffer_manager_thread_count: Field = Field(
        "buffer_manager_thread_count",
        "The number of threads used to accelerate copies and other operations required to manage input and output tensor contents.",
        value_type=int,
        default=0)

    gather_kernel_buffer_threshold: Field = Field(
        "gather_kernel_buffer_threshold",
        "Set the threshold number of buffers for triton to use gather kernel to gather input data. 0 disables the gather kernel",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        default=0)

    batch_triton_requests: Field = Field(
        "batch_triton_requests",
        "Send a batch of query samples to triton instead of single query at a time",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        value_type=bool)

    output_pinned_memory: Field = Field(
        "output_pinned_memory",
        "Use pinned memory when data transfer for output is between device mem and non-pinned sys mem",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        value_type=bool)

    use_concurrent_harness: Field = Field(
        "use_concurrent_harness",
        "Use multiple threads for batching and triton issue while using the triton harness",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        value_type=bool)

    num_concurrent_batchers: Field = Field(
        "num_concurrent_batchers",
        "Number of threads that will batch samples to form triton requests. Only used when the concurrent triton harness is used",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        value_type=int,
        default=1)

    num_concurrent_issuers: Field = Field(
        "num_concurrent_issuers",
        "Number of threads that will issue requests to triton. Only used when the concurrent triton harness is used.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        value_type=int,
        default=1
    )

    #dla_num_batchers: Field = Field(
    #    "dla_num_batchers",
    #    "Number of threads that will batch samples to form triton requests. Only used when the concurrent DLA triton harness is used",
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_harnesses=(HarnessType.Triton,),
    #    supported_systems=(SystemClassifications.is_soc,),
    #    value_type=int,
    #)

    #dla_num_issuers: Field = Field(
    #    "dla_num_issuers",
    #    "Number of threads that will issue requests to triton. Only used when the concurrent DLA triton harness is used.",
    #    supported_actions=_HARNESS_ACTIONS,
    #    supported_harnesses=(HarnessType.Triton,),
    #    supported_systems=(SystemClassifications.is_soc,),
    #    value_type=int,
    #)

    '''
    inferentia_compiled_model_batch_size: Field = Field(
        "inferentia_compiled_model_batch_size",
        "Batch size of the compiled inferentia model",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        supported_systems=(SystemClassifications.inferentia_based,),
        value_type=int,
    )

    inferentia_compiled_model_framework: Field = Field(
        "inferentia_compiled_model_framework",
        "The framework the inferentia model was compiled in. Pytorch or TF.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        supported_systems=(SystemClassifications.inferentia_based,),
        value_type=str,
    )

    inferentia_threads_per_core: Field = Field(
        "inferentia_threads_per_core",
        "Number of threads executing model on each neuron core.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        supported_systems=(SystemClassifications.inferentia_based,),
        value_type=int,
    )

    inferentia_neuron_core_count: Field = Field(
        "inferentia_neuron_core_count",
        "Number of neuron cores in the system. Depends on instance type.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        supported_systems=(SystemClassifications.inferentia_based,),
        value_type=int,
    )

    inferentia_request_batch_size: Field = Field(
        "inferentia_request_batch_size",
        "Size of the triton harness batch request.",
        supported_actions=_HARNESS_ACTIONS,
        supported_harnesses=(HarnessType.Triton,),
        supported_systems=(SystemClassifications.inferentia_based,),
        value_type=int,
    )
    '''

    # Many of these metrics have the scenario explicitly in the name because these are the names that Loadgen expects.
    server_target_qps: Field = Field(
        "server_target_qps",
        "Target QPS used for server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_target_latency_ns: Field = Field(
        "server_target_latency_ns",
        "Desired latency constraint for server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_target_latency_percentile: Field = Field(
        "server_target_latency_percentile",
        "Desired latency percentile constraint for server scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    server_num_issue_query_threads: Field = Field(
        "server_num_issue_query_threads",
        "Number of IssueQuery threads to use for Loadgen in Server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    schedule_rng_seed: Field = Field(
        "schedule_rng_seed",
        "Seed for RNG that affects the poisson arrival process in server scenario",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Server,))

    accuracy_log_rng_seed: Field = Field(
        "accuracy_log_rng_seed",
        "Affects which samples have their query returns logged to the accuracy log in performance mode.",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS)

    single_stream_expected_latency_ns: Field = Field(
        "single_stream_expected_latency_ns",
        "Inverse of desired target QPS",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.SingleStream,))

    single_stream_target_latency_percentile: Field = Field(
        "single_stream_target_latency_percentile",
        "Desired latency percentile constraint for single stream scenario",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.SingleStream,))

    multi_stream_expected_latency_ns: Field = Field(
        "multi_stream_expected_latency_ns",
        "Expected latency to process a query with multiple Samples, in nanoseconds",
        value_type=int,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    multi_stream_target_latency_percentile: Field = Field(
        "multi_stream_target_latency_percentile",
        "Desired latency percentile to report as a performance metric, for multi stream scenario",
        value_type=float,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    multi_stream_samples_per_query: Field = Field(
        "multi_stream_samples_per_query",
        "Number of samples bundled together as a single query (default: 8)",
        value_type=int,
        default=8,
        required=True,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.MultiStream,))

    offline_expected_qps: Field = Field(
        "offline_expected_qps",
        "Target samples per second rate for the SUT (Offline mode)",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_scenarios=(Scenario.Offline,))

    sample_partition_path: Field = Field(
        "sample_partition_path",
        "Path to sample partition file in npy format.",
        value_type=str,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    enable_interleaved_top_mlp: Field = Field(
        "enable_interleaved_top_mlp",
        """Use interleaved data format in the TensorRT engine:
        Run DLRM top MLP in interleaved format, i.e. in [N/2, C, 2, 1] instead of [N, C, 1, 1].""",
        value_type=bool,
        supported_benchmarks=(Benchmark.DLRM,),
        supported_systems=(SystemClassifications.gpu_based,))

    output_padding_granularity: Field = Field(
        "output_padding_granularity",
        """The granularity to pad the DLRM interaction output to. For example, if output_padding_granularity=32, then
        the DLRM interaction output will be padded to become a multiple of 32.""",
        value_type=int,
        supported_benchmarks=(Benchmark.DLRM,),
        supported_systems=(SystemClassifications.gpu_based,))

    embedding_weights_on_gpu_part: Field = Field(
        "embedding_weights_on_gpu_part",
        """Percentage of the embedding weights to store on GPU. Lower this if your GPU has lower VRAM. Default: 1.0
        (100% of embedding weights on GPU)""",
        value_type=float,
        supported_benchmarks=(Benchmark.DLRM,),
        supported_systems=(SystemClassifications.gpu_based,))

    num_staging_threads: Field = Field(
        "num_staging_threads",
        "Number of staging threads in DLRM BatchMaker",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    num_staging_batches: Field = Field(
        "num_staging_batches",
        "Number of staging batches in DLRM BatchMaker",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    max_pairs_per_staging_thread: Field = Field(
        "max_pairs_per_staging_thread",
        "Maximum pairs to copy in one BatchMaker staging thread",
        value_type=int,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    split_threshold: Field = Field(
        "split_threshold",
        "Sample size threshold to start round-robin on DLRM BatchMakers, if NUMA config is set up (default: 500)",
        value_type=int,
        default=500,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    gpu_num_bundles: Field = Field(
        "gpu_num_bundles",
        "Number of event+buffer bundles per GPU for DLRM (default: 2)",
        value_type=int,
        default=2,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,),
        supported_systems=(SystemClassifications.gpu_based,))

    check_contiguity: Field = Field(
        "check_contiguity",
        "Check if inputs are already contiguous in QSL to avoid copying",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.DLRM,))

    use_jemalloc: Field = Field(
        "use_jemalloc",
        "Use libjemalloc.so.2 as the malloc(3) implementation",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS)

    bert_opt_seqlen: Field = Field(
        "bert_opt_seqlen",
        "The opt_shape for BERT provided to TRT builder in the optimization profile.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    graphs_max_seqlen: Field = Field(
        "graphs_max_seqlen",
        "The max sequence length for BERT when capturing CUDA Graphs.",
        value_type=int,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    graph_specs: Field = Field(
        "graph_specs",
        """Specify a comma separated list of (maxSeqLen, min totSeqLen, max totSeqLen, step size) for CUDA graphs to be
        captured.""",
        value_type=str,
        supported_actions=(Action.GenerateEngines,),
        supported_benchmarks=(Benchmark.BERT,))

    soft_drop: Field = Field(
        "soft_drop",
        """Inverse percentage of the longest BERT sequences to drop softly in Server scenario by deferring to the end.
        When soft_drop=1.0, no sequence is dropped. When soft_drop=0.992, the top 0.8% longest sequences are dropped
        softly, i.e. their inferences are pushed to the end of the test.""",
        value_type=float,
        supported_actions=_HARNESS_ACTIONS,
        supported_benchmarks=(Benchmark.BERT,))

    numa_config: Field = Field(
        "numa_config",
        """NUMA settings: GPU and CPU cores for each NUMA node as a string with the following convention:

        ```
        [Node 0 config]&[Node 1 config]&[Node 2 config]&...
        ```

        Each `[Node n config]` can be configured as `[GPU ID(s)]:[CPU ID(s)]`.
        Each `ID(s)` can be single digit, comma-separated digits, or digits with dash.
        ex.
            "numa_config": "3:0-15,64-79&2:16-31,80-95&1:32-47,96-111&0:48-63,112-127"

        In this example, example `3:0-15,64-79` means GPU 3 and CPU 0,1,2,...15,64,65,66,...79 are in the same node,
        and since this key is the very first of elements connected with &, they are in node 0.""",
        value_type=bool,
        supported_actions=_HARNESS_ACTIONS,
        supported_systems=(SystemClassifications.multi_gpu,))
    '''
    batch_size: Field = Field(
        "batch_size",  # TODO: Rename this to cpu_batch_size
        "Batch size to use for the CPU",
        value_type=int,
        supported_systems=(SystemClassifications.intel_openvino,),
        required=True)

    model_name: Field = Field(
        "model_name",
        "OpenVino model name",
        value_type=str,
        supported_systems=(SystemClassifications.intel_openvino,),
        required=True)
    
    num_instances: Field = Field(
        "num_instances",
        "Number of input request queues. Should be greater than or equal to CPU_THROUGHPUT_STREAMS in ov_parameters.",
        value_type=int,
        supported_systems=(SystemClassifications.intel_openvino,),
        required=True)

    ov_parameters: Field = Field(
        "ov_parameters",
        "OpenVino parameter dictionary",
        value_type=dict,
        supported_systems=(SystemClassifications.intel_openvino,),
        no_argparse=True,
        required=True)
    '''

def get_applicable_fields(action: Optional[Action], benchmark: Benchmark, scenario: Scenario, system: System,
                          workload_setting: WorkloadSetting) -> Tuple[List[Field], List[Field]]:
    """
    Gets the applicable fields for the workload from Fields.

    Args:
        action (Optional[Action]):
            The Action of the workload. Can be None.
        benchmark (Benchmark):
            The Benchmark of the workload
        scenario (Scenario):
            The Scenario of the workload
        system (System):
            The System running the workload
        workload_setting (WorkloadSetting):
            The WorkloadSetting field of the config to get fields for

    Returns:
        List[Field]: A list of the required fields for the workload to run
        List[Field]: A list of the optional (non-required) but supported fields for the workload to run
    """
    required = []
    optional = []
    for field in Fields:
        if field.value.supports(action, benchmark, scenario, system, workload_setting):
            if field.value.required:
                required.append(field.value)
            else:
                optional.append(field.value)
    return required, optional


@unique
class MainArgs(Enum):
    """
    Defines a set of known Fields used as CLI arguments for main.py. Note that Fields specified in this Enum will ignore
    all `Field.supported_*` attributes, since these values will not be known until the appropriate action is determined
    from MainArgs.action.
    """

    action: Field = Field(
        "action",
        "The phase in the benchmarking or submission process to run.",
        value_type=str,
        argparse_opts={
            "choices": Action.as_strings(),
            "required": True,
        })

    benchmarks: Field = Field(
        "benchmarks",
        "The names of benchmarks to run the current phase on. " +
        "If specifying multiple benchmarks, use a comma,separated,list of names (i.e. resnet50,bert,dlrm) " +
        "Default: Runs all benchmarks with valid configurations for the current system.",
        value_type=str)

    scenarios: Field = Field(
        "scenarios",
        "The names of scenarios to run the current phase for. " +
        "If specifying multiple scenarios, use a comma,separated,list of names (i.e. offline,server) " +
        "Default: Runs all scenarios with valid configurations for the current system.",
        value_type=str)

    gpu_only: Field = Field(
        "gpu_only",
        "Only used on Xavier machines where default behavior is to use the iGPU and both DLAs. " +
        "If set, then the current run will only use the iGPU for the specified action.",
        value_type=bool)

    no_gpu: Field = Field(
        "no_gpu",
        "Only used on Xavier machines where default behavior is to use the iGPU and both DLAs. " +
        "If set, then the current run will only use the DLA cores specified without the iGPU. " +
        "Default: DLA Core 0",
        value_type=bool)

    no_child_process: Field = Field(
        "no_child_process",
        "Generate engines in main process instead of child processes. Useful when running profilers or gdb " +
        "when debugging issues with generate_engines.",
        value_type=bool)

    power: Field = Field(
        "power",
        "Measure power during this harness run. Note that this does NOT set power_setting to MaxQ.",
        value_type=bool)

    # power_limit is specially handled, since it must be known *before* the action is run, when the ScopedMPS is
    # created. Hence, power_limit (which should be overridable), must be applied as an override at the initial main.py
    # level, not at the BenchmarkConfiguration level.
    # Server power args
    power_limit: Field = Fields.power_limit.value
    cpu_freq: Field = Fields.cpu_freq.value
    # Xavier power args
    #soc_gpu_freq: Field = Fields.soc_gpu_freq.value
    #soc_dla_freq: Field = Fields.soc_dla_freq.value
    #soc_cpu_freq: Field = Fields.soc_cpu_freq.value
    #soc_emc_freq: Field = Fields.soc_emc_freq.value
    #orin_num_cores: Field = Fields.orin_num_cores.value


    profile: Field = Field(
        "profile",
        "[INTERNAL ONLY] Select profiler to use.",
        value_type=str,
        argparse_opts={"choices": ["nsys", "nvprof", "ncu"]})

    audit_test: Field = Field(
        "audit_test",
        "The audit test to run, if an audit-related action is chosed.",
        value_type=str,
        argparse_opts={"choices": ["TEST01", "TEST04-A", "TEST04-B", "TEST05"]})

    system_name: Field = Field(
        "system_name",
        "Overrides the 'system name', which is generated from the detected system by default. " +
        "In most cases, it is better to set 'system_name' directly in the BenchmarkConfiguration " +
        "instead of using this field to keep the system_name consistent across runs.",
        value_type=str)

    harness_type: Field = Field(
        "harness_type",
        "Selects which harness to use during the run_harness phase. Note that this should also " +
        "be specified during the generate_engines phase, as changing harnesses may also change " +
        "engine-generation parameters. Use 'auto' to use the default harness (LWIS or custom). " +
        "Default: 'auto'",
        value_type=str,
        default="auto",
        argparse_opts={
            "choices": ["auto"] + HarnessType.as_strings(),
            "default": "auto",
        })

    accuracy_target: Field = Field(
        "accuracy_target",
        "Selects which accuracy target to compare the accuracy results against. Note that the " +
        "'high accuracy' (99.9%% of FP32) target is only available on BERT, DLRM, and 3D-UNET.",
        value_type=float,
        default=AccuracyTarget.k_99.value,
        argparse_opts={
            "choices": [at.value for at in AccuracyTarget],
            "default": AccuracyTarget.k_99.value,
        })

    power_setting: Field = Field(
        "power_setting",
        "Selects which power setting the harness is running on. Note that this setting does NOT " +
        "actually set the power setting itself, but instead is just used to select the appropriate " +
        "BenchmarkConfiguration. The power setting must be set with the appropriate commands " +
        "before this program is run.",
        value_type=str,
        default=PowerSetting.MaxP.value.name,
        argparse_opts={
            "choices": PowerSetting.as_strings(),
            "default": PowerSetting.MaxP.value.name,
        })

    config_ver: Field = Field(
        "config_ver",
        "[LEGACY OPTION] Specifies workload settings via the config_ver IDs from the old config.json " +
        "style configs. Using this flag will override the --harness_type, --accuracy_target, and " +
        "--power_setting flags.",
        value_type=str)
