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

#include "bert_lon_server.h"

#include "cuda_profiler_api.h"

// helpers
#include "utils.hpp"

// QDL
#include "qdl_intf_bert.hpp"

// Various flags
#include "harness_bert_flags.hpp"
#include "harness_net_flags.hpp"

// Primary thread control
std::atomic<bool> m_thread_active;

// poll the current state and let QDL/SUT work together in harmony
void PrimarySyncHandler(bert_lon::QdlIntfPtr_t qdl, bert_lon::ServerPtr_t sut)
{
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
            std::cout << "LON node shall have started running actual test." << std::endl;
            qdl->set_qdl_state(lwis_lon::SyncMsg::WARMUP_END);
            qdl->set_primary_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        }
        case lwis_lon::SyncMsg::ALL_DONE:
        {
            std::cout << "Test finished, cleaning up." << std::endl;
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

// Helper function to actually perform inference using MLPerf Loadgen
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
    qdl_settings.m_SutUsesHostMemForRdma = is_SUT && FLAGS_SUT_uses_host_mem_for_RDMA;
    qdl_settings.m_SutForceUsesHostMemForRecv = true;
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
    auto SutRecvWithHostMemForRdma = qdl_settings.m_SutUsesHostMemForRdma || qdl_settings.m_SutForceUsesHostMemForRecv;
    auto SutSendWithHostMemForRdma = qdl_settings.m_SutUsesHostMemForRdma || qdl_settings.m_SutForceUsesHostMemForSend;

    CHECK_EQ(SutRecvWithHostMemForRdma, true) << "BERT LON HARNESS requires SUT "
                                                 "to receive input samples in "
                                                 "Host memory";

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
                SutRecvWithHostMemForRdma, SutSendWithHostMemForRdma, qdl_settings.m_LON_memory_staging,
                qdl_settings.m_CQ_wait_by_event, NUMA_node, num_NUMA_nodes, j, send_q_size, num_resp, response_size,
                recv_q_size, num_input, sample_size));
        }
    }
    qdl_params.m_IBConns_params = std::move(ibcp);

    // Instantiate our QDL_Interface, work in lieu of QSL in SUT
    std::cout << "Creating QDL_Interface." << std::endl;
    auto qdl = std::make_shared<bert_lon::QDL_Interface>("BERT_SUT_QDL_Interface");
    std::cout << "Setting up QDL_Interface." << std::endl;
    qdl->SetupSettingsAndParams(qdl_settings, qdl_params);
    qdl->Setup();
    std::cout << "Finished Creating QDL_Interface." << std::endl;

    std::cout << "Bringing up SUT." << std::endl;
    std::vector<int> gpus;
    if (FLAGS_devices == "all")
    {
        int numDevices = 0;
        cudaGetDeviceCount(&numDevices);
        LOG(INFO) << "Found " << numDevices << " GPUs";
        for (int i = 0; i < numDevices; i++)
        {
            gpus.emplace_back(i);
        }
    }
    else
    {
        LOG(INFO) << "Use GPUs: " << FLAGS_devices;
        auto deviceNames = splitStringLine(FLAGS_devices, ',');
        for (auto& n : deviceNames)
            gpus.emplace_back(std::stoi(n));
    }

    // fixing target latency percentile for now; SUT may not know the test
    // settings and the number may be fixed as per scenario
    // number of issue threads always go with number of GPUs, at least for now
    // (maybe multiple of num GPUs in the future)
    double const tgt_latency_percentile = 0.99;

    bert_lon::ServerSettings sut_settings;
    sut_settings.GPUBatchSize = FLAGS_gpu_batch_size;
    sut_settings.GPUCopyStreams = FLAGS_gpu_copy_streams;
    sut_settings.GPUInferStreams = FLAGS_gpu_inference_streams;
    sut_settings.EnableDequeLimit = FLAGS_use_deque_limit;
    sut_settings.Timeout = std::chrono::microseconds(FLAGS_deque_timeout_usec);
    sut_settings.SutRecvWithHostMemForRdma = SutRecvWithHostMemForRdma;
    sut_settings.SutSendWithHostMemForRdma = SutSendWithHostMemForRdma;
    sut_settings.NumIBQPsPerNIC = FLAGS_num_ibqps_per_nic;
    sut_settings.GpuNumaMap = qdl_settings.m_SUTGpuNumaMap;
    sut_settings.NumaConfig = qdl_settings.m_SUTNumaConfig;
    sut_settings.GraphMaxSeqLen = FLAGS_graphs_max_seqlen;
    sut_settings.SoftDrop = FLAGS_soft_drop;
    sut_settings.SoftDropTargetLatencyPercentile = tgt_latency_percentile;

    auto sut = std::make_shared<bert_lon::BERTServer>(
        "BERT_LON Network SUT", FLAGS_gpu_engines, gpus, FLAGS_use_graphs, FLAGS_graph_specs, sut_settings);

    std::cout << "Finished setting up SUT." << std::endl;
    qdl->set_sut(sut);
    std::cout << "Registered SUT in QDL." << std::endl;

    cudaProfilerStart();

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

    cudaProfilerStop();

    // reset smart pointers
    qdl.reset();
    sut.reset();
}

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "BERT_LON Harness Network SUT";
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

    if (FLAGS_load_plugins)
    {
        // TODO I can only prevent this from loading default plugins by commenting
        // this out
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    }

    m_thread_active = true;

    // Perform inference
    doInference();

    // Report pass or fail
    return gLogger.reportPass(sampleTest);
}
