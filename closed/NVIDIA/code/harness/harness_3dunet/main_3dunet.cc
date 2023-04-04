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

/* Include necessary header files */
// Loadgen
#include "loadgen.h"

// TensorRT
#include "NvInfer.h"
#include "NvInferPlugin.h"
#include "logger.h"
#include "logging.h"

// LWIS, for 3D-UNet sliding window
#include "lwis_3dunet.hpp"

// Google Logging
#include <glog/logging.h>

// General C++
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <memory>
#include <sys/stat.h>
#include <thread>

#include "callback.hpp"
#include "utils.hpp"

#include "cuda_profiler_api.h"

/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
DEFINE_string(gpu_engines, "", "Comma-separated list of gpu engines");
DEFINE_string(dla_engines, "", "Comma-separated list of DLA engines");
DEFINE_string(plugins, "", "Comma-separated list of shared objects for plugins");
DEFINE_string(devices, "all", "Enable comma separated numbered devices");
DEFINE_int32(dla_core, -1,
    "Specify a DLA engine for layers that support DLA.  Value can range from 0 to n-1, "
    "where n is the number of DLA engines on the platform");
DEFINE_bool(verbose, false, "Use verbose logging");
DEFINE_bool(verbose_nvtx, false, "Turn ProfilingVerbosity to kDETAILED so that layer detail is printed in NVTX.");
DEFINE_bool(use_spin_wait, false,
    "Actively wait for work completion.  This option may decrease multi-process "
    "synchronization time at the cost of additional CPU usage");
DEFINE_bool(use_device_schedule_spin, false,
    "Actively wait for results from the device.  May reduce latency at the the cost of "
    "less efficient CPU parallelization");
DEFINE_string(map_path, "", "Path to map file for samples");
DEFINE_string(tensor_path, "",
    "Path to preprocessed samples in npy format (<full_image_name>.npy). Comma-separated "
    "list if there are more than one input.");
DEFINE_bool(coalesced_tensor, false, "Turn on if all the samples are coalesced into one single npy file");
DEFINE_bool(use_graphs, false,
    "Enable cudaGraphs for TensorRT engines"); // TODO: Enable support for Cuda Graphs
DEFINE_bool(
    use_direct_host_access, false, "Enable all devices to access host memory directly for input and output data.");
DEFINE_bool(use_deque_limit, false, "Enable a max number of elements dequed from work queue");
DEFINE_uint64(deque_timeout_usec, 10000, "Timeout for deque from work queue");
DEFINE_bool(use_batcher_thread_per_device, true, "Enable a separate batcher thread per device");
DEFINE_bool(use_cuda_thread_per_device, false, "Enable a separate cuda thread per device");
DEFINE_bool(start_from_device, false, "Start inference assuming the data is already in device memory");
DEFINE_bool(assume_contiguous, false, "Assume that the data in a query is already contiguous");
DEFINE_bool(use_same_context, false,
    "Use the same TRT context for all copy streams (shape must be static and "
    "gpu_inference_streams must be 1).");
DEFINE_string(numa_config, "", "NUMA settings: each NUMA node contains a pair of GPU indices and CPU indices.");

DEFINE_uint32(gpu_copy_streams, 1, "Number of copy streams for inference");
DEFINE_uint32(gpu_inference_streams, 1, "Number of streams for inference");
DEFINE_uint32(gpu_batch_size, 8, "Max Batch size to use for all devices and engines");

DEFINE_uint32(dla_copy_streams, 1, "Number of copy streams for inference");
DEFINE_uint32(dla_inference_streams, 1, "Number of streams for inference");
DEFINE_uint32(dla_batch_size, 8, "Max DLA Batch size to use for all devices and engines");
DEFINE_uint32(max_dlas, 2, "Max number of DLAs to use per device. Default: 2");

DEFINE_bool(run_infer_on_copy_streams, false, "Runs inference on copy streams");
DEFINE_uint32(complete_threads, 1, "Number of threads per device for sending responses");

DEFINE_double(warmup_duration, 5.0, "Minimum duration to run warmup for");

DEFINE_string(response_postprocess, "", "Enable imagenet post-processing on query sample responses.");

DEFINE_uint64(performance_sample_count, 0, "Number of samples to load in performance set.  0=use default");

DEFINE_bool(end_on_device, false, "Perform query D2H copy inside QuerySamplesComplete() callback (untimed)");
// Loadgen test settings
DEFINE_string(scenario, "Offline", "Scenario to run for Loadgen (Offline, SingleStream)");
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
DEFINE_uint64(log_mode_async_poll_interval_ms, 1000, "Specify the poll interval for asynchrounous logging");
DEFINE_bool(log_enable_trace, false, "Enable trace logging");

/* extra FLAGS 3D-UNet KiTS19 uses */
DEFINE_uint32(unet3d_sw_dhw, 128, "3DUNet sliding window dimension (uniform in D, H, W)");
DEFINE_uint32(
    unet3d_sw_overlap_pct, 50, "Percent value showing 3DUNet sliding window overlaps (uniform in all dimension)");
DEFINE_string(
    unet3d_sw_gaussian_patch_path, "", "Path to 3DUNet preconditioned, coalesced, Gaussian Patch in npy format (.npy)");
DEFINE_bool(slice_overlap_patch_kernel_cg_impl, false, "Use 3D-UNet patch kernel implemented using cooperative-group; if false, kernel impl using CPU implicit sync is used");

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestMode> testModeMap = {{"SubmissionRun", mlperf::TestMode::SubmissionRun},
    {"AccuracyOnly", mlperf::TestMode::AccuracyOnly}, {"PerformanceOnly", mlperf::TestMode::PerformanceOnly},
    {"FindPeakPerformance", mlperf::TestMode::FindPeakPerformance}};

/* Define a map to convert logging mode input string into its corresponding enum value */
std::map<std::string, mlperf::LoggingMode> logModeMap = {{"AsyncPoll", mlperf::LoggingMode::AsyncPoll},
    {"EndOfTestOnly", mlperf::LoggingMode::EndOfTestOnly}, {"Synchronous", mlperf::LoggingMode::Synchronous}};

/* Define a map to convert test mode input string into its corresponding enum value */
std::map<std::string, mlperf::TestScenario> scenarioMap = {{"Offline", mlperf::TestScenario::Offline},
    {"SingleStream", mlperf::TestScenario::SingleStream}, {"Server", mlperf::TestScenario::Server}};

/* Keep track of the GPU devices we are using */
std::vector<uint32_t> Devices;
std::vector<std::string> DeviceNames;

/* check if file exists; cannot use std::filesystem due to Xavier NX. Keeping as legacy behavior. */
inline bool does_exist(const std::string& name)
{
    struct stat buffer;
    return (stat(name.c_str(), &buffer) == 0);
}

/* Helper function to actually perform inference using MLPerf Loadgen */
void doInference()
{
    // Configure test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];

    gLogInfo << "mlperf.conf path: " << FLAGS_mlperf_conf_path << std::endl;
    gLogInfo << "user.conf path: " << FLAGS_user_conf_path << std::endl;
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);

    // Configure logging settings
    mlperf::LogSettings log_settings;
    log_settings.log_output.outdir = FLAGS_logfile_outdir;
    log_settings.log_output.prefix = FLAGS_logfile_prefix;
    log_settings.log_output.suffix = FLAGS_logfile_suffix;
    log_settings.log_output.prefix_with_datetime = FLAGS_logfile_prefix_with_datetime;
    log_settings.log_output.copy_detail_to_stdout = FLAGS_log_copy_detail_to_stdout;
    log_settings.log_output.copy_summary_to_stdout = !FLAGS_disable_log_copy_summary_to_stdout;
    log_settings.log_mode = logModeMap[FLAGS_log_mode];
    log_settings.log_mode_async_poll_interval_ms = FLAGS_log_mode_async_poll_interval_ms;
    log_settings.enable_trace = FLAGS_log_enable_trace;

    // Configure server settings
    lwis::ServerSettings_3DUNet sut_settings;
    sut_settings.GPUBatchSize = FLAGS_gpu_batch_size;
    sut_settings.GPUCopyStreams = FLAGS_gpu_copy_streams;
    sut_settings.GPUInferStreams = FLAGS_gpu_inference_streams;

    sut_settings.DLABatchSize = FLAGS_dla_batch_size;
    sut_settings.DLACopyStreams = FLAGS_dla_copy_streams;
    sut_settings.DLAInferStreams = FLAGS_dla_inference_streams;

    if (FLAGS_dla_core != -1)
    {
        sut_settings.MaxGPUs = 0;
        sut_settings.MaxDLAs = 1; // no interface to specify which DLA
    }
    else
    {
        sut_settings.MaxDLAs = FLAGS_max_dlas;
    }
    sut_settings.EnableSpinWait = FLAGS_use_spin_wait;
    sut_settings.EnableDeviceScheduleSpin = FLAGS_use_device_schedule_spin;
    sut_settings.RunInferOnCopyStreams = FLAGS_run_infer_on_copy_streams;
    sut_settings.EnableDirectHostAccess = FLAGS_use_direct_host_access;
    sut_settings.EnableDequeLimit = FLAGS_use_deque_limit;
    sut_settings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
    sut_settings.EnableBatcherThreadPerDevice = FLAGS_use_batcher_thread_per_device;
    sut_settings.EnableCudaThreadPerDevice = FLAGS_use_cuda_thread_per_device;
    sut_settings.EnableStartFromDeviceMem = FLAGS_start_from_device;
    sut_settings.CompleteThreads = FLAGS_complete_threads;
    sut_settings.UseSameContext = FLAGS_use_same_context;
    sut_settings.m_NumaConfig = parseNumaConfig(FLAGS_numa_config);
    sut_settings.m_GpuToNumaMap = getGpuToNumaMap(sut_settings.m_NumaConfig);
    sut_settings.EndOnDevice = FLAGS_end_on_device;
    sut_settings.SliceOverlapKernelCGImpl = FLAGS_slice_overlap_patch_kernel_cg_impl;
    sut_settings.VerboseNVTX = FLAGS_verbose_nvtx;

    // sliding window parameters
    // FIXME: SW_dhw/SW_overlap_pct all fixed value
    // FIXME: when possible, use std::filesystem::path()/std::filesystem::is_regular_file()
    sut_settings.SW_gaussian_patch_path = FLAGS_unet3d_sw_gaussian_patch_path;
    CHECK_EQ(does_exist(sut_settings.SW_gaussian_patch_path), 1) << "Cannot find Gaussian Patch file";

    lwis::ServerParams sut_params;
    sut_params.DeviceNames = FLAGS_devices;
    sut_params.EngineNames.resize(2);
    for (auto& engineName : splitString(FLAGS_gpu_engines, ","))
    {
        if (engineName == "")
            continue;
        std::vector<std::string> engines = {engineName};
        sut_params.EngineNames[0].emplace_back(engines);
    }
    for (auto& engineName : splitString(FLAGS_dla_engines, ","))
    {
        if (engineName == "")
            continue;
        std::vector<std::string> engines = {engineName};
        sut_params.EngineNames[1].emplace_back(engines);
    }

    // SANITY CHECK for this version's support of knobs
    // NOTE: GPUBatchSize/DLABatchSize == SW batch size, != sample batch size
    // FIXME: current version only supports batch size == 1, copy_streams == 1
    CHECK_EQ(sut_settings.GPUCopyStreams, 1) << "Only 1 copy stream is supported for now";
    CHECK_EQ(sut_settings.GPUInferStreams, 1) << "Only 1 infer stream is supported for now";
    CHECK_EQ(sut_settings.DLACopyStreams, 1) << "Only 1 copy stream is supported for now";
    CHECK_EQ(sut_settings.DLAInferStreams, 1) << "Only 1 infer stream is supported for now";

    // BatcherThreadPerDevice should be on, for performance on multi-GPU
    // FIXME: single batcher may need to be updated for good supply of samples to multi-GPUs
    CHECK_EQ(sut_settings.EnableBatcherThreadPerDevice, true) << "BatcherThreadPerDevice should be on";

    // Instantiate Server
    lwis::ServerPtr_t sut = std::make_shared<lwis::Server>("Server_3DUNet");

    // Instantiate QSL
    std::cout << "Creating QSL." << std::endl;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");
    std::vector<bool> start_from_device(tensor_paths.size(), FLAGS_start_from_device);
    const size_t padding = 0;
    std::shared_ptr<mlperf::QuerySampleLibrary> qsl;
    // NOTE: 3D-UNet KiTS19 tensor shapes are non-uniform, so disable the coalesced_tensor in QSL
    if (sut_settings.m_NumaConfig.empty())
    {
        // When NUMA is not used, create one QSL.
        auto oneQsl = std::make_shared<qsl::SampleLibrary3DUNet>("LWIS_SampleLibrary", FLAGS_map_path,
            splitString(FLAGS_tensor_path, ","),
            FLAGS_performance_sample_count ? FLAGS_performance_sample_count
                                           : std::max(FLAGS_gpu_batch_size, FLAGS_dla_batch_size), // Why batch size???
            /*padding*/ 0, /*FLAGS_coalesced_tensor*/ false, start_from_device);
        sut->AddSampleLibrary(oneQsl);
        qsl = oneQsl;
    }
    else
    {
        // When NUMA is used, create one QSL per NUMA node.
        std::cout << "Using NUMA. Config: " << FLAGS_numa_config << std::endl;
        const int32_t nbNumas = sut_settings.m_NumaConfig.size();
        std::vector<qsl::SampleLibrary3DUNetPtr_t> qsls;
        for (int32_t numaIdx = 0; numaIdx < nbNumas; numaIdx++)
        {
            // Use a thread to construct QSL so that the allocated memory is closer to that NUMA
            // node.
            auto constructQsl = [&]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                bindNumaMemPolicy(numaIdx, nbNumas);
                auto oneQsl = std::make_shared<qsl::SampleLibrary3DUNet>("LWIS_SampleLibrary", FLAGS_map_path,
                    splitString(FLAGS_tensor_path, ","),
                    FLAGS_performance_sample_count ? FLAGS_performance_sample_count
                                                   : std::max(FLAGS_gpu_batch_size, FLAGS_dla_batch_size),
                    /*padding*/ 0, /*FLAGS_coalesced_tensor*/ false, start_from_device);
                resetNumaMemPolicy();
                sut->AddSampleLibrary(oneQsl);
                qsls.emplace_back(oneQsl);
            };
            std::thread th(constructQsl);
            bindThreadToCpus(th, sut_settings.m_NumaConfig[numaIdx].second);
            th.join();
        }
        qsl = std::shared_ptr<qsl::SampleLibrary3DUNetEnsemble>(new qsl::SampleLibrary3DUNetEnsemble(qsls));
    }
    std::cout << "Finished Creating QSL." << std::endl;

    std::cout << "Setting up SUT." << std::endl;
    sut->Setup(sut_settings, sut_params); // Pass the requested sut settings and params to our SUT
    sut->SetResponseCallback(callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
                                                                       // post-processing callback
    std::cout << "Finished setting up SUT." << std::endl;

    // Perform a brief warmup
    std::cout << "Starting warmup. Running for a minimum of " << FLAGS_warmup_duration << " seconds." << std::endl;
    auto tStart = std::chrono::high_resolution_clock::now();
    sut->Warmup(FLAGS_warmup_duration);
    double elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    std::cout << "Finished warmup. Ran for " << elapsed << "s." << std::endl;

    // Perform the inference testing
    std::cout << "Starting running actual test." << std::endl;
    cudaProfilerStart();
    mlperf::StartTest(sut.get(), qsl.get(), test_settings, log_settings);
    cudaProfilerStop();
    std::cout << "Finished running actual test." << std::endl;

    // Log device stats
    auto devices = sut->GetDevices();
    for (auto& device : devices)
    {
        const auto& stats = device->GetStats();

        std::cout << "Device " << device->GetName() << " processed:" << std::endl;
        for (auto& elem : stats.m_BatchSizeHistogram)
        {
            std::cout << "  " << elem.second << " batches of size " << elem.first << std::endl;
        }

        std::cout << "  Memcpy Calls: " << stats.m_MemcpyCalls << std::endl;
        std::cout << "  PerSampleCudaMemcpy Calls: " << stats.m_PerSampleCudaMemcpyCalls << std::endl;
        std::cout << "  BatchedCudaMemcpy Calls: " << stats.m_BatchedCudaMemcpyCalls << std::endl;
    }

    // Inform the SUT that we are done
    sut->Done();

    // Make sure CUDA RT is still in scope when we free the memory
    qsl.reset();
    sut.reset();
}

int main(int argc, char* argv[])
{
    // Initialize logging
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "MLPerf_Inference_3DUNet_Harness";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    gLogger.reportTestStart(sampleTest);
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    else
    {
        setReportableSeverity(Severity::kINFO);
    }

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

    // Perform inference
    doInference();

    // Report pass
    return gLogger.reportPass(sampleTest);
}
