/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <gflags/gflags.h>
#include <map>

// LWIS settings
DEFINE_bool(verbose, false, "Use verbose logging");

// QSL
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format "
    "(<full_image_name>.npy). Comma-separated "
    "list if there are more than one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.");
DEFINE_bool(coalesced_tensor, false, "Turn on if all the samples are coalesced into one single npy file");

// Loadgen test settings
DEFINE_string(scenario, "Server", "Scenario to run for Loadgen (Offline, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "resnet50", "Model name");

// configuration files
DEFINE_string(mlperf_conf_path, "", "Path to mlperf.conf");
DEFINE_string(user_conf_path, "", "Path to user.conf");

// Loadgen logging settings
DEFINE_string(logfile_outdir, "", "Specify the existing output directory for the LoadGen logs");
DEFINE_string(logfile_prefix, "", "Specify the filename prefix for the LoadGen log files");
DEFINE_string(logfile_suffix, "", "Specify the filename suffix for the LoadGen log files");
DEFINE_bool(logfile_prefix_with_datetime, false, "Prefix filenames for LoadGen log files");
DEFINE_bool(log_copy_detail_to_stdout, false, "Copy LoadGen detailed logging to stdout");
DEFINE_bool(disable_log_copy_summary_to_stdout, false, "Disable copy LoadGen summary logging to stdout");
DEFINE_string(log_mode, "AsyncPoll", "Logging mode for Loadgen");
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchronous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// LWIS SUT specific settings
DEFINE_string(gpu_engines, "", "Comma-separated list of gpu engines");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");
DEFINE_bool(use_spin_wait, false,
    "Actively wait for work completion.  This option may decrease "
    "multi-process "
    "synchronization time at the cost of additional CPU usage");
DEFINE_bool(use_device_schedule_spin, false,
    "Actively wait for results from the device.  May reduce latency at "
    "the the cost of "
    "less efficient CPU parallelization");
DEFINE_bool(use_graphs, false,
    "Enable cudaGraphs for TensorRT engines"); // TODO: Enable support
                                               // for Cuda Graphs
DEFINE_bool(use_deque_limit, false, "Enable a max number of elements dequed from work queue");
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");
DEFINE_bool(use_cuda_thread_per_device, true, "Enable a separate cuda thread per device");
DEFINE_bool(use_same_context, false,
    "Use the same TRT context for all copy streams (shape must be static and "
    "gpu_inference_streams must be 1).");

DEFINE_uint32(gpu_copy_streams, 4, "Number of copy streams for inference");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams for inference");
DEFINE_uint32(gpu_batch_size, 256, "Max Batch size to use for all devices and engines");

DEFINE_bool(run_infer_on_copy_streams, false, "Runs inference on copy streams");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");

DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");

// DELETEME: below parameters should not be used as it doesn't support (or
// tested for) DLA ; needs cleanup from infra
DEFINE_uint32(max_dlas, 0, "Max number of DLAs to use per device. Default: 0");

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum
 * value */
std::map<std::string, mlperf::TestScenario> scenarioMap
    = {{"Server", mlperf::TestScenario::Server}, {"Offline", mlperf::TestScenario::Offline}};