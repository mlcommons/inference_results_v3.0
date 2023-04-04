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

#include "qsl_lon.hpp"

// helpers
#include "utils.hpp"

// QDL
#include "qdl_lon.hpp"

// commonly used
#include "bert_lon_common.h"

// Various flags
#include "harness_bert_flags.hpp"
#include "harness_net_flags.hpp"

/* Helper function to actually perform inference using MLPerf Loadgen */
void doInference()
{
    // sleep a little bit for IB to be stabilized
    sleep(5);

    bool const is_SUT = false;

    // Configure the test settings
    mlperf::TestSettings test_settings;
    test_settings.scenario = scenarioMap[FLAGS_scenario];
    test_settings.mode = testModeMap[FLAGS_test_mode];

    gLogInfo << "mlperf.conf path: " << FLAGS_mlperf_conf_path << std::endl;
    gLogInfo << "user.conf path: " << FLAGS_user_conf_path << std::endl;
    test_settings.FromConfig(FLAGS_mlperf_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.FromConfig(FLAGS_user_conf_path, FLAGS_model, FLAGS_scenario);
    test_settings.server_coalesce_queries = true;
    test_settings.server_num_issue_query_threads = FLAGS_server_num_issue_query_threads;

    // Configure the logging settings
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

    // QDL settings & params
    lwis_lon::QDLSettings qdl_settings;
    qdl_settings.m_LONNumaConfig = lwis_lon::parse_numa_config(FLAGS_lon_numa_config);
    qdl_settings.m_LONNicNumaMap = lwis_lon::parse_nic_numa_map(qdl_settings.m_LONNumaConfig);
    qdl_settings.m_Nic2NicMap = lwis_lon::parse_nic_mapping(FLAGS_nic_mapping);
    qdl_settings.m_EnMaxTrxPerConn = FLAGS_enable_max_transactions_per_connection;
    qdl_settings.m_MaxTrxPerConn = FLAGS_max_transactions_per_connection;
    qdl_settings.m_MaxWaitPerConnMicroSec = std::chrono::microseconds(FLAGS_max_wait_before_sending_us);
    qdl_settings.m_OneIssueQueue = FLAGS_lon_uses_one_issue_queue;
    qdl_settings.m_RoundRobinSamples = FLAGS_round_robin_samples_to_multi_issue_queue;
    qdl_settings.m_SmartBalanceSamples = FLAGS_smart_balance_samples_to_multi_issue_queue;
    CHECK_NE(FLAGS_round_robin_samples_to_multi_issue_queue && FLAGS_smart_balance_samples_to_multi_issue_queue, 1)
        << "Do not set round_robin_samples_to_multi_issue_queue and "
           "smart_balance_samples_to_multi_issue_queue "
           "together";
    CHECK(
        !(FLAGS_smart_balance_samples_to_multi_issue_queue && test_settings.scenario == mlperf::TestScenario::Offline))
        << "Smart Balancing won't work well with Offline scenario";
    qdl_settings.m_CQ_wait_by_event = true;
    qdl_settings.m_LON_memory_staging = false;

    lwis_lon::QDLParams qdl_params;
    qdl_params.m_SUT_NetID = FLAGS_sut_netid;
    qdl_params.m_TCP_port = FLAGS_tcp_port;
    qdl_params.m_is_SUT = is_SUT;
    // TODO: maybe set this MaxTrxPerConn using Little's Law
    auto send_q_size = qdl_settings.m_EnMaxTrxPerConn ? qdl_settings.m_MaxTrxPerConn : 2048;
    auto recv_q_size = send_q_size;
    auto sample_size = lwis_lon::TensorSizeMap[FLAGS_model].first;
    auto num_input = lwis_lon::NumInputMap[FLAGS_model];
    auto response_size = lwis_lon::TensorSizeMap[FLAGS_model].second;
    uint8_t const num_resp{1};

    // Some info for NUMA mapping and NIC mapping
    std::cout << "Mapping -- " << std::endl;
    std::cout << "    LON NIC : LON NUMA node" << std::endl;
    for (auto& m : qdl_settings.m_LONNicNumaMap)
    {
        std::cout << "    " << m.first << "  : " << m.second << std::endl;
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
    // BERT LON is not enabled for NUMA
    // int32_t num_NUMA_nodes = qdl_settings.m_LONNumaConfig.size() > 0 ?
    // qdl_settings.m_LONNumaConfig.size() : 1;
    int32_t const num_NUMA_nodes = 1;
    for (int i = 0; i < qdl_settings.m_Nic2NicMap.size(); ++i)
    {
        // LON uses host memory with possible NUMA optimization
        int32_t NUMA_node = num_NUMA_nodes > 1 ? qdl_settings.m_LONNicNumaMap[qdl_settings.m_Nic2NicMap[i].first] : 0;

        for (uint32_t j = 0; j < qdl_settings.m_NumIBQPsPerNIC; ++j)
        {
            // NIC may need more than one QPs
            ibcp.push_back(lwis_lon::get_ibconnection_param({qdl_settings.m_Nic2NicMap[i].first, -1},
                {qdl_settings.m_Nic2NicMap[i].second,
                    qdl_settings.m_SUTNicGpuAffn[qdl_settings.m_Nic2NicMap[i].second]},
                false, false, qdl_settings.m_LON_memory_staging, qdl_settings.m_CQ_wait_by_event, NUMA_node,
                num_NUMA_nodes, j, send_q_size, num_input, sample_size, recv_q_size, num_resp, response_size));
        }
    }
    qdl_params.m_IBConns_params = std::move(ibcp);

    // sanity check on parameters
    CHECK(FLAGS_performance_sample_count > 0) << "Performance sample count should be properly set";

    // Instantiate QDL
    std::cout << "Creating QDL." << std::endl;
    lwis_lon::QdlPtr_t qdl = std::make_shared<lwis_lon::QDL>("LON_QDL");
    // Pass the requested QDL settings and params
    qdl->SetupSettingsAndParams(qdl_settings, qdl_params);
    std::cout << "Setting up QDL." << std::endl;
    qdl->Setup();
    // DELETEME: QDL NOT ALLOWED TO DO ANY CALLBACK
    // qdl->SetResponseCallback(
    //     callbackMap[FLAGS_response_postprocess]); // Set QuerySampleResponse
    // post-processing callback
    std::cout << "Finished setting up QDL." << std::endl;

    // Instantiate our QSL
    std::cout << "Creating QSL." << std::endl;
    std::vector<std::string> tensor_paths = splitString(FLAGS_tensor_path, ",");

    const size_t padding = 0; // multistream is not supported for Network division

    std::shared_ptr<mlperf::QuerySampleLibrary> qsl;

    if (qdl->UseNuma(is_SUT))
    {
        // When NUMA is used, create one QSL per NUMA node.
        std::cout << "QSL using NUMA config: " << FLAGS_lon_numa_config << std::endl;
        const int32_t nbNumas = qdl->GetNbNumas(is_SUT);
        std::vector<qsl::SampleLibraryPtr_t> qsls;

        // if staging, need QSL as many as NUMA nodes
        if (qdl_settings.m_LON_memory_staging)
        {
            for (int32_t numaIdx = 0; numaIdx < nbNumas; numaIdx++)
            {
                // Use a thread to construct QSL so that the allocated memory is closer
                // to
                // that NUMA node.
                auto constructQsl = [&]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    bindNumaMemPolicy(numaIdx, nbNumas);
                    auto oneQsl = std::make_shared<qsl::SampleLibrary>("BERT_SampleLibrary_NUMA", FLAGS_map_path,
                        splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding,
                        FLAGS_coalesced_tensor, qdl_settings.m_LON_memory_staging, nullptr, 0);
                    resetNumaMemPolicy();
                    qdl->AddSampleLibrary(oneQsl);
                    std::cout << "Added QSL to QDL for NUMA " << numaIdx << std::endl;
                    qsls.emplace_back(oneQsl);
                };

                std::thread th(constructQsl);
                if (qdl->UseNuma(is_SUT))
                    bindThreadToCpus(th, qdl->GetClosestCpusFromNumaNode(numaIdx, is_SUT));
                th.join();
            }
        }
        // else, need QSL as many as connections as registered memory is used
        else
        {
            for (int i = 0; i < qdl_settings.m_Nic2NicMap.size(); ++i)
            {
                // LON uses host memory with possible NUMA optimization
                int32_t numaIdx = nbNumas > 1 ? qdl_settings.m_LONNicNumaMap[qdl_settings.m_Nic2NicMap[i].first] : 0;
                auto sbuf = qdl->GetConnections()->get_connection(i)->get_resources()->get_send_buffer();
                auto sbuf_size = qdl->GetConnections()->get_connection(i)->get_resources()->get_send_buffer_size();

                // Use a thread to construct QSL so that the allocated memory is closer
                // to
                // that NUMA node.
                auto constructQsl = [&]() {
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    bindNumaMemPolicy(numaIdx, nbNumas);
                    auto oneQsl = std::make_shared<qsl::SampleLibrary>("BERT_SampleLibrary_NUMA", FLAGS_map_path,
                        splitString(FLAGS_tensor_path, ","), FLAGS_performance_sample_count, padding,
                        FLAGS_coalesced_tensor, qdl_settings.m_LON_memory_staging, sbuf, sbuf_size);
                    resetNumaMemPolicy();
                    qdl->AddSampleLibrary(oneQsl);
                    std::cout << "Added QSL to QDL for Connection " << i << std::endl;
                    qsls.emplace_back(oneQsl);
                };

                std::thread th(constructQsl);
                if (qdl->UseNuma(is_SUT))
                    bindThreadToCpus(th, qdl->GetClosestCpusFromNumaNode(numaIdx, is_SUT));
                th.join();
            }
        }
        qsl = std::shared_ptr<qsl::SampleLibraryEnsemble>(new qsl::SampleLibraryEnsemble(qsls));
    }

    std::cout << "Finished Creating QSL." << std::endl;

    // QDL setup should have sync'ed with the SUT; SUT should have finished init
    // Send warmup signal and wait until it's done
    qdl->Warmup();

    // Perform the inference testing
    std::cout << "Starting running actual test." << std::endl;
    qdl->StartHeartbeat();
    mlperf::StartTest(qdl.get(), qsl.get(), test_settings, log_settings);
    qdl->StopHeartbeat();
    std::cout << "Finished running actual test." << std::endl;

    std::cout << "Test finished, cleaning up." << std::endl;
    // Inform the QDL that we are done
    qdl->Done();

    // reset smart pointers
    qsl.reset();
    qdl.reset();
}

int main(int argc, char* argv[])
{
    FLAGS_alsologtostderr = 1; // Log to console
    ::google::InitGoogleLogging("TensorRT mlperf");
    ::google::ParseCommandLineFlags(&argc, &argv, true);
    const std::string gSampleName = "BERT_LON Harness Network LON";
    auto sampleTest = gLogger.defineTest(gSampleName, argc, const_cast<const char**>(argv));
    if (FLAGS_verbose)
    {
        setReportableSeverity(Severity::kVERBOSE);
    }
    gLogger.reportTestStart(sampleTest);
    if (FLAGS_load_plugins)
    {
        // TODO I can only prevent this from loading default plugins by commenting
        // this out
        initLibNvInferPlugins(&gLogger.getTRTLogger(), "");
    }

    // Perform inference
    doInference();

    // Report pass or fail
    return gLogger.reportPass(sampleTest);
}
