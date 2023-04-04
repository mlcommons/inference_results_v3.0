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

// Google Logging
#include <gflags/gflags.h>
#include <glog/logging.h>

// General C++
#include <chrono>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <memory>
#include <thread>

// helpers
#include "callback.hpp"
#include "common.hpp"
#include "utils.hpp"

// QDL
#include "qdl_intf_lwis.hpp"

// Various flags
#include "harness_lwis_flags.hpp"
#include "harness_net_flags.hpp"

// LON LWIS
#include "lwis_lon.hpp"

// CUDA
#include "cuda_profiler_api.h"

// Keep track of the GPU devices we are using
std::vector<uint32_t> Devices;
std::vector<std::string> DeviceNames;

// Primary thread control
std::atomic<bool> m_thread_active;

// poll the current state and let QDL/SUT work together in harmony
void PrimarySyncHandler(lwis_lon::QdlIntfPtr_t qdl, lwis_lon::ServerPtr_t sut)
{
    gLogVerbose << "QDL Interface::PrimarySyncHandler"
                << ": on CPU " << sched_getcpu() << std::endl;

    lwis_lon::SyncMsg PrimarySyncState = qdl->get_primary_state();
    while (PrimarySyncState != lwis_lon::SyncMsg::SHUTDOWN)
    {
        switch (PrimarySyncState)
        {
        case lwis_lon::SyncMsg::QUERY_NAME:
        {
            auto sut_name = sut->Name();
            qdl->send_sut_name(sut_name);
            qdl->set_qdl_state(lwis_lon::SyncMsg::SYNC_WAIT);
            qdl->set_primary_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        }
        case lwis_lon::SyncMsg::WARMUP_START:
        {
            // Perform a brief warmup
            std::cout << "Starting warmup. Running for a minimum of " << FLAGS_warmup_duration << " seconds."
                      << std::endl;
            auto tStart = std::chrono::high_resolution_clock::now();
            sut->Warmup(FLAGS_warmup_duration);
            double elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
            std::cout << "Finished warmup. Ran for " << elapsed << "s." << std::endl;
            std::cout << "LON node must have started running actual test." << std::endl;
            qdl->set_qdl_state(lwis_lon::SyncMsg::WARMUP_END);
            qdl->set_primary_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        }
        case lwis_lon::SyncMsg::ALL_DONE:
        {
            // Log device stats
            std::cout << "Test finished, cleaning up." << std::endl;
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
            // Inform QDL as well
            qdl->set_primary_state(lwis_lon::SyncMsg::SHUTDOWN);
            qdl->set_qdl_state(lwis_lon::SyncMsg::SHUTDOWN);
            qdl->Done();
            // shutdown myself
            m_thread_active = false;
            // sleep for a moment and change state
            sleep(1);
            break;
        }
        case lwis_lon::SyncMsg::SHUTDOWN:
        case lwis_lon::SyncMsg::SYNC_WAIT:
        case lwis_lon::SyncMsg::BUSY_WAIT: break;
        default:
            std::cerr << "ERROR: PrimaryThread got unexpected Sync State: " << PrimarySyncState << std::endl;
            break;
        }
        PrimarySyncState = qdl->get_primary_state();
    }
}

/* Helper function to actually perform inference */
void doInference()
{
    // sleep a little bit for IB to be stabilized
    sleep(5);

    bool const is_SUT = true;

    // QDL settings & params for QDL_Interface
    lwis_lon::QDLSettings qdl_settings;
    qdl_settings.m_SUTNumaConfig = lwis_lon::parse_numa_config(FLAGS_sut_numa_config);
    qdl_settings.m_SUTNicNumaMap = lwis_lon::parse_nic_numa_map(qdl_settings.m_SUTNumaConfig);
    qdl_settings.m_SUTNicGpuAffn = lwis_lon::parse_nic_gpu_affinity(FLAGS_sut_nic_gpu_affinity);
    qdl_settings.m_SUTGpuNumaMap = lwis_lon::parse_gpu_numa_map(qdl_settings.m_SUTNumaConfig);
    qdl_settings.m_Nic2NicMap = lwis_lon::parse_nic_mapping(FLAGS_nic_mapping);
    qdl_settings.m_EnMaxTrxPerConn = FLAGS_enable_max_transactions_per_connection;
    qdl_settings.m_MaxTrxPerConn = FLAGS_max_transactions_per_connection;
    qdl_settings.m_MaxWaitPerConnMicroSec = std::chrono::microseconds(FLAGS_max_wait_before_sending_us);
    qdl_settings.m_NumIBQPsPerNIC = FLAGS_num_ibqps_per_nic;
    qdl_settings.m_SutUsesHostMemForRdma = is_SUT && FLAGS_SUT_uses_host_mem_for_RDMA;
    qdl_settings.m_SutForceUsesHostMemForRecv = false;
    qdl_settings.m_SutForceUsesHostMemForSend = false;
    qdl_settings.m_CQ_wait_by_event = true;

    lwis_lon::QDLParams qdl_params;
    qdl_params.m_LON_NetID = FLAGS_lon_netid;
    qdl_params.m_TCP_port = FLAGS_tcp_port;
    qdl_params.m_is_SUT = is_SUT;

    // TODO: maybe set MaxTrxPerConn from Little's Law
    auto send_q_size = qdl_settings.m_EnMaxTrxPerConn ? qdl_settings.m_MaxTrxPerConn : 2048;
    auto recv_q_size = send_q_size;
    auto sample_size = lwis_lon::TensorSizeMap[FLAGS_model].first;
    auto num_input = lwis_lon::NumInputMap[FLAGS_model];
    auto response_size = lwis_lon::TensorSizeMap[FLAGS_model].second;
    uint8_t const num_resp{1};

    // Some info for NUMA mapping and NIC mapping
    std::cout << "Mapping -- " << std::endl;
    std::cout << "    SUT NIC : SUT NUMA node" << std::endl;
    for (auto& m : qdl_settings.m_SUTNicNumaMap)
    {
        std::cout << "    " << m.first << "  : " << m.second << std::endl;
    }
    std::cout << "----------------------------" << std::endl;
    std::cout << "Mapping -- " << std::endl;
    std::cout << "    SUT NIC : SUT GPU ID" << std::endl;
    for (auto& m : qdl_settings.m_SUTNicGpuAffn)
    {
        std::cout << "    " << m.first << "  : " << m.second << std::endl;
    }
    std::cout << "----------------------------" << std::endl;
    std::cout << "Mapping -- " << std::endl;
    std::cout << "    SUT GPU : SUT NUMA node" << std::endl;
    for (int i = 0; i < qdl_settings.m_SUTGpuNumaMap.size(); ++i)
    {
        std::cout << "    " << i << "       : " << qdl_settings.m_SUTGpuNumaMap[i] << std::endl;
    }
    std::cout << "----------------------------" << std::endl;
    std::cout << "Mapping -- " << std::endl;
    std::cout << "    LON NIC : SUT NIC" << std::endl;
    for (auto& m : qdl_settings.m_Nic2NicMap)
    {
        std::cout << "    " << m.first << "  : " << m.second << std::endl;
    }
    std::cout << "----------------------------" << std::endl;

    lwis_lon::IBConnsParams ibcp;
    int32_t num_NUMA_nodes = qdl_settings.m_SUTNumaConfig.size() > 0 ? qdl_settings.m_SUTNumaConfig.size() : 1;
    for (int i = 0; i < qdl_settings.m_Nic2NicMap.size(); ++i)
    {
        // SUT uses host CPUs with possible NUMA optimization
        int32_t NUMA_node = num_NUMA_nodes > 1 ? qdl_settings.m_SUTNicNumaMap[qdl_settings.m_Nic2NicMap[i].second] : 0;

        for (uint32_t j = 0; j < qdl_settings.m_NumIBQPsPerNIC; ++j)
        {
            // NIC may need more than one QPs
            ibcp.push_back(lwis_lon::get_ibconnection_param({qdl_settings.m_Nic2NicMap[i].first, -1},
                {qdl_settings.m_Nic2NicMap[i].second,
                    qdl_settings.m_SUTNicGpuAffn[qdl_settings.m_Nic2NicMap[i].second]},
                qdl_settings.m_SutUsesHostMemForRdma || qdl_settings.m_SutForceUsesHostMemForRecv,
                qdl_settings.m_SutUsesHostMemForRdma || qdl_settings.m_SutForceUsesHostMemForSend,
                qdl_settings.m_LON_memory_staging, qdl_settings.m_CQ_wait_by_event, NUMA_node, num_NUMA_nodes, j,
                send_q_size, num_resp, response_size, recv_q_size, num_input, sample_size));
        }
    }
    qdl_params.m_IBConns_params = std::move(ibcp);

    // Instantiate our QDL_Interface, work in lieu of QSL in SUT
    std::cout << "Creating QDL_Interface." << std::endl;
    auto qdl = std::make_shared<lwis_lon::QDL_Interface>("LWIS_SUT_QDL_Interface");
    std::cout << "Setting up QDL_Interface" << std::endl;
    qdl->SetupSettingsAndParams(qdl_settings, qdl_params);
    qdl->Setup();
    std::cout << "Finished Creating QDL_Interface." << std::endl;

    // Instantiate and configure our SUT
    lwis_lon::ServerPtr_t sut = std::make_shared<lwis_lon::Server>("LWIS_LON Network SUT");

    lwis_lon::ServerSettings sut_settings;
    sut_settings.EnableCudaGraphs = FLAGS_use_graphs;
    sut_settings.GPUBatchSize = FLAGS_gpu_batch_size;
    sut_settings.GPUCopyStreams = FLAGS_gpu_copy_streams;
    sut_settings.GPUInferStreams = FLAGS_gpu_inference_streams;
    sut_settings.EnableSpinWait = FLAGS_use_spin_wait;
    sut_settings.EnableDeviceScheduleSpin = FLAGS_use_device_schedule_spin;
    sut_settings.RunInferOnCopyStreams = FLAGS_run_infer_on_copy_streams;
    sut_settings.EnableDequeLimit = FLAGS_use_deque_limit;
    sut_settings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
    sut_settings.EnableCudaThreadPerDevice = FLAGS_use_cuda_thread_per_device;
    sut_settings.CompleteThreads = FLAGS_complete_threads;
    sut_settings.UseSameContext = FLAGS_use_same_context;
    sut_settings.SutUsesHostMemForRdma = is_SUT && FLAGS_SUT_uses_host_mem_for_RDMA;
    sut_settings.NumIBQPsPerNIC = FLAGS_num_ibqps_per_nic;
    sut_settings.m_GpuNumaMap = qdl_settings.m_SUTGpuNumaMap;
    sut_settings.m_NumaConfig = qdl_settings.m_SUTNumaConfig;

    // below should not be configurable as of yet
    sut_settings.EnableBatcherThreadPerDevice = true;
    sut_settings.ForceContiguous = false;

    lwis_lon::ServerParams sut_params;
    sut_params.DeviceNames = FLAGS_devices;
    sut_params.EngineNames.resize(2);
    for (auto& engineName : splitString(FLAGS_gpu_engines, ","))
    {
        if (engineName == "")
            continue;
        std::vector<std::string> engines = {engineName};
        sut_params.EngineNames[0].emplace_back(engines);
    }

    std::cout << "Setting up SUT." << std::endl;
    sut->Setup(sut_settings, sut_params); // Pass the requested sut settings and
                                          // params to our SUT
    // Disabling the callback as response has to be in GPU memory and callback
    // should be kernel (non-existent as of now)
    // sut->SetResponseCallback(
    //     callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
    // post-processing callback
    std::cout << "Finished setting up SUT." << std::endl;
    qdl->set_sut(sut);
    std::cout << "Registered SUT in QDL." << std::endl;

    std::cout << "Starting QDL worker threads." << std::endl;
    qdl->StartThreads();

    // QDL Interface has below threads:
    //
    // IBConnections TCP polling thread called Steward()
    // This responds Sync Messages like WARMUP, NAME, DONE, etc
    // When WARMUP is done, it creates WORKER THREADS
    // When NAME, it responds SUT name string through TCP
    // When DONE, it halts WORKER THREADS and shutdown the SUT and QDL_Interface
    //
    // SUT has below threads:
    // ProcessSamples() and ProcessBatches() are now thread per Device
    // IssueBatch()/CopySamples()/CopyResponses() are called by ProcessBatches()

    // QDL Interface transceives sync msgs, and PrimarySyncThread works it with
    // SUT
    // NOTE: FIXME: C++20 supports std::atomic<std::shared_ptr>> but not in C++17
    // Until moving on to C++20, let qdl to manage PrimarySyncState instead
    // std::atomic<std::shared_ptr<lwis_lon::SyncMsg>> m_CurrentSyncState =
    // SYNC_WAIT;
    // qdl->add_sync_msg_ptr(m_CurrentSyncState);
    auto m_PrimarySyncThread = std::thread(PrimarySyncHandler, qdl, sut);

    auto m_num_numa = qdl->GetNbNumas(is_SUT);
    auto m_num_gpus = qdl->GetNumGpus(is_SUT);

    // let the communication go
    qdl->set_qdl_state(lwis_lon::SyncMsg::SYNC_WAIT);

    // wait for PrimarySyncThread to join; this joins after SHUTDOWN from LON
    m_PrimarySyncThread.join();

    // Make sure CUDA RT is still in scope when we free the memory
    qdl.reset();
    sut.reset();
}

int main(int argc, char* argv[])
{
    // Initialize logging
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "LWIS_LON Harness Network SUT";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    else
    {
        setReportableSeverity(Severity::kINFO);
    }
    gLogger.reportTestStart(sampleTest);

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

    m_thread_active = true;

    // Perform inference
    doInference();

    return gLogger.reportPass(sampleTest);
}
