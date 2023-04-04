/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
 * Copyright 2018 Google LLC
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

#include "NvInferPlugin.h"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "loadgen.h"
#include "logger.h"
#include "test_settings.h"

#include "dlrm_qsl.hpp"
#include "dlrm_server.h"
#include "numpy.hpp"
#include "qsl.hpp"
#include "utils.hpp"

#include "cuda_profiler_api.h"

#include <chrono>
#include <dlfcn.h>
#include <thread>

DEFINE_string(gpu_engines, "", "Engine");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");

DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream, Server)");
DEFINE_string(test_mode, "PerformanceOnly", "Testing mode for Loadgen");
DEFINE_string(model, "dlrm", "Model name");
DEFINE_uint32(gpu_batch_size, 16384, "Max Batch size to use for all devices and engines");
DEFINE_bool(use_graphs, false, "Enable cudaGraphs for TensorRT engines"); // TODO: Enable support for Cuda Graphs
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");

DEFINE_uint32(gpu_copy_streams, 1, "[CURRENTLY NOT USED] Number of copy streams");
DEFINE_uint32(gpu_num_bundles, 2, "Number of event-buffer bundles per GPU");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");
DEFINE_uint32(gpu_inference_streams, 1, "Number of inference streams");

DEFINE_double(warmup_duration, 1.0, "Minimum duration to run warmup for");

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
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

// QSL arguments
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(sample_partition_path, "", "Path to sample partition file in npy format.");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated list if there are more than "
    "one input.");
DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");
DEFINE_bool(start_from_device, false, "Assuming that inputs start from device memory in QSL");

// Dataset arguments
DEFINE_uint32(min_sample_size, 100, "Minimum number of pairs a sample can contain.");
DEFINE_uint32(max_sample_size, 700, "Maximum number of pairs a sample can contain.");

// BatchMaker arguments
DEFINE_uint32(num_staging_threads, 8, "Number of staging threads in DLRM BatchMaker");
DEFINE_uint32(num_staging_batches, 4, "Number of staging batches in DLRM BatchMaker");
DEFINE_uint32(
    max_pairs_per_staging_thread, 0, "Maximum pairs to copy in one BatchMaker staging thread (0 = use default");
DEFINE_uint32(split_threshold, 500, "Sample size threshold to start round-robin on BatchMakers");
DEFINE_bool(check_contiguity, false,
    "Whether to use contiguity checking in BatchMaker (default: false, recommended: true for Offline)");
DEFINE_string(numa_config, "", "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices.");
DEFINE_bool(compress_categorical_inputs, false,
    "Compress categorical inputs to packed representation and decompress in TRT engine");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {
    {"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream},
    {"Server", mlperf::TestScenario::Server},
};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "DLRM_HARNESS";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    gLogger.reportTestStart(sampleTest);
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    // Load all the needed shared objects for plugins.
    std::vector<std::string> plugin_files = splitString(FLAGS_plugins, ",");
    for (auto& s : plugin_files)
    {
        void* dlh = dlopen(s.c_str(), RTLD_LAZY);
        if (nullptr == dlh)
        {
            gLogError << "Error loading plugin library " << s << std::endl;
            return 1;
        }
    }

    CHECK_EQ(FLAGS_compress_categorical_inputs,
        FLAGS_tensor_path.find("build/preprocessed_data/criteo/full_recalib/categorical_int32_compressed.npy")
            != std::string::npos);

    // Scope to force all smart objects destruction before CUDA context resets
    {
        int num_gpu;
        cudaGetDeviceCount(&num_gpu);
        LOG(INFO) << "Found " << num_gpu << " GPUs";
        // Configure the test settings
        mlperf::TestSettings testSettings;
        testSettings.scenario = scenarioMap[FLAGS_scenario];
        testSettings.mode = testModeMap[FLAGS_test_mode];
        testSettings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
        testSettings.server_coalesce_queries = true;

        // Configure the logging settings
        mlperf::LogSettings logSettings;
        logSettings.log_output.outdir = FLAGS_logfile_outdir;
        logSettings.log_output.prefix = FLAGS_logfile_prefix;
        logSettings.log_output.suffix = FLAGS_logfile_suffix;
        logSettings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
        logSettings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
        logSettings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
        logSettings.log_mode = logModeMap[FLAGS_log_mode];
        logSettings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
        logSettings.enable_trace = FLAGS_log_enable_trace;

        std::vector<int> gpus(num_gpu);
        std::iota(gpus.begin(), gpus.end(), 0);

        // Load the sample partition. We do this here to calculate the performance sample count of the underlying
        // LWIS QSL, since the super constructor must be in the constructor initialization list.
        std::vector<int> originalPartition;

        // Scope to automatically close the file
        {
            npy::NpyFile samplePartitionFile(FLAGS_sample_partition_path);
            CHECK_EQ(samplePartitionFile.getDims().size(), 1);

            // For now we do not allow numPartitions == FLAGS_performance_sample_count, since the TotalSampleCount
            // is determined at runtime in the underlying LWIS QSL.
            size_t numPartitions = samplePartitionFile.getDims()[0];
            CHECK_EQ(numPartitions > FLAGS_performance_sample_count, true);

            std::vector<char> tmp(samplePartitionFile.getTensorSize());
            samplePartitionFile.loadAll(tmp);

            originalPartition.resize(numPartitions);
            memcpy(originalPartition.data(), tmp.data(), tmp.size());
            LOG(INFO) << "Loaded " << originalPartition.size() - 1 << " sample partitions. (" << tmp.size()
                      << ") bytes.";
        }

        // Force underlying QSL to load all samples, since we want to be able to grab any partition given the sample
        // index.
        size_t perfPairCount = originalPartition.back();
        const auto numaConfig = parseNumaConfig(FLAGS_numa_config);
        std::vector<DLRMSampleLibraryPtr_t> qsls;
        if (numaConfig.empty())
        {
            auto oneQsl = std::make_shared<DLRMSampleLibrary>("DLRM QSL", FLAGS_map_path,
                splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count, perfPairCount,
                0, true, FLAGS_start_from_device, -1, 0);
            qsls.emplace_back(oneQsl);
        }
        else
        {
            const int32_t nbNumas = numaConfig.size();
            for (int32_t numaIdx = 0; numaIdx < nbNumas; numaIdx++)
            {
                // Use a thread to construct QSL so that the allocated memory is closer to that NUMA node.
                auto constructQsl = [&]() {
                    bindNumaMemPolicy(numaIdx, nbNumas);
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    auto oneQsl = std::make_shared<DLRMSampleLibrary>("DLRM QSL", FLAGS_map_path,
                        splitString(FLAGS_tensor_path, ","), originalPartition, FLAGS_performance_sample_count,
                        perfPairCount, 0, true, FLAGS_start_from_device, numaIdx, nbNumas);
                    qsls.emplace_back(oneQsl);
                    resetNumaMemPolicy();
                };
                std::thread th(constructQsl);
                bindThreadToCpus(th, numaConfig[numaIdx].second);
                th.join();
            }
        }

        auto dlrm_server
            = std::make_shared<DLRMServer>("DLRM SERVER", FLAGS_gpu_engines, qsls, gpus, FLAGS_gpu_batch_size,
                FLAGS_gpu_num_bundles, FLAGS_complete_threads, FLAGS_gpu_inference_streams, FLAGS_warmup_duration,
                FLAGS_num_staging_threads, FLAGS_num_staging_batches, FLAGS_max_pairs_per_staging_thread,
                FLAGS_split_threshold, FLAGS_check_contiguity, FLAGS_start_from_device, numaConfig, FLAGS_verbose_nvtx);

        LOG(INFO) << "Starting running actual test.";
        cudaProfilerStart();
        std::shared_ptr<DLRMSampleLibraryEnsemble> qslEnsemble(new DLRMSampleLibraryEnsemble(qsls));
        StartTest(dlrm_server.get(), qslEnsemble.get(), testSettings, logSettings);
        cudaProfilerStop();
        LOG(INFO) << "Finished running actual test.";
    }

    return 0;
}
