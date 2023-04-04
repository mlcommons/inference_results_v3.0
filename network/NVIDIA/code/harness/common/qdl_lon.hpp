/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef __QDL_HPP__
#define __QDL_HPP__

#include <deque>
#include <functional>
#include <string>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "logger.h"
#include <glog/logging.h>

#include "half.h"

#include "common.hpp"
#include "lon_ib.hpp"
#include "qsl_lon.hpp"
#include "query_dispatch_library.h"
#include "utils.hpp"

#include <thread>
#include <unordered_map>
#include <vector>

// QDL (Query Distribution Library) is an implementation of the MLPerf Query Distribution Library.
// It works as a proxy to SUT in LON node

// if uncommented, disable memory copies for perf debugging
// #define QDL_DEBUG_DISABLE_MEMCOPY

#define THREAD_WAKEUP_TIME std::chrono::microseconds(100)
#define HEARTBEAT std::chrono::milliseconds(1000)
namespace lwis_lon
{

class QDL;

// Map for input and output tensor size, in Bytes
inline std::map<std::string, std::pair<std::size_t, std::size_t>> TensorSizeMap{
    {"resnet50", std::make_pair<std::size_t, std::size_t>(150528, 4)},
    {"bert", std::make_pair<std::size_t, std::size_t>(1536, 1536)},
};
// number of inputs, assuming same size (or max size)
inline std::map<std::string, std::uint8_t> NumInputMap{
    {"resnet50", 1},
    {"bert", 1},
};

// IBConnections parameters
using IBConnsParams = std::vector<lonib::conn_param_t>;

// IBConnections
using IBConnsPtr_t = std::shared_ptr<lonib::IBConnections>;

// pointer to class
using QdlPtr_t = std::shared_ptr<QDL>;

struct QDLSettings
{
    NumaConfig m_SUTNumaConfig;
    NumaConfig m_LONNumaConfig;
    NicNumaMap m_SUTNicNumaMap;
    NicNumaMap m_LONNicNumaMap;
    NicGpuAffn m_SUTNicGpuAffn;
    GpuNumaMap m_SUTGpuNumaMap;
    Nic2NicMap m_Nic2NicMap;
    bool m_EnMaxTrxPerConn{false};
    uint64_t m_MaxTrxPerConn{1};
    uint64_t m_NumIBQPsPerNIC{1};
    std::chrono::microseconds m_MaxWaitPerConnMicroSec{0};
    uint64_t m_NumBufferedRecvWQE{0};
    bool m_OneIssueQueue{true};
    bool m_RoundRobinSamples{false};
    bool m_SmartBalanceSamples{false};
    bool m_SutUsesHostMemForRdma{false};
    bool m_SutForceUsesHostMemForRecv{false};
    bool m_SutForceUsesHostMemForSend{false};
    bool m_LON_memory_staging{false};
    bool m_CQ_wait_by_event{true};
};

struct QDLParams
{
    IBConnsParams m_IBConns_params;
    std::string m_LON_NetID;
    std::string m_SUT_NetID;
    std::string m_TCP_port;
    bool m_is_SUT;
};

// This conn_param_t bundles the information to configure each IB connection class, i.e. IBConnection
// queue size should be the same for send/recv as current buffer management only happens for RDMA writes
//
// The number of sends always has to be the same to the number of receives
// Sample <=> SampleResponse, SampleResponse <=> ACK
//
// elem size wise, send and recv is from LON point of view; SUT side should be opposite in configuring this param
// LON is sending Samples (send) and waiting for Completions (recv)
// From SUT POV, samples are received and completions are sent
//
// con[0] (LON NIC_device/neighbor_GPU) <=> con[1] (SUT NIC_device/neighbor_GPU)
// con[2] whether SUT will use host memory for RDMA transactions
// con[3] if LON will stage memory to be used for RDMA transfer
// con[4] if CQ will be maintained by blocking wait events, instead of polling
// con[5] NUMA_node and con[6] total number of NUMA nodes
// con[7] number of QPs per NIC
// con[8] send queue size, con[9] number of send element(s), con[10] send element (max) size
// con[11] recv queue size, con[11] number of recv element(s), con[13] recv element (max) size
//
// LON side, this parameter may see neighboring GPU to NIC maybe N/A (-1)
//
// This will be set up after parsing cmdline arguments and used in QDL to instantiate resources and connections
//
inline lonib::conn_param_t get_ibconnection_param(std::pair<std::string, int> lon, std::pair<std::string, int> sut,
    bool SUT_recv_with_host_mem_for_RDMA, bool SUT_send_with_host_mem_for_RDMA, bool LON_memory_staging,
    bool CQ_wait_event, int32_t NUMA_node, int32_t num_NUMA_nodes, int32_t qp_idx, std::size_t send_queue_size,
    uint8_t send_elem_num, std::size_t send_elem_size, std::size_t recv_queue_size, uint8_t recv_elem_num,
    std::size_t recv_elem_size)
{
    return std::make_tuple(lon, sut, SUT_recv_with_host_mem_for_RDMA, SUT_send_with_host_mem_for_RDMA,
        LON_memory_staging, CQ_wait_event, NUMA_node, num_NUMA_nodes, qp_idx, send_queue_size, send_elem_num,
        send_elem_size, recv_queue_size, recv_elem_num, recv_elem_size);
}

class QDL : public mlperf::QueryDispatchLibrary
{
public:
    QDL()
        : m_Name("Default QDL")
        , m_issue_idx(0)
    {
    }

    QDL(std::string name)
        : m_Name(name)
        , m_issue_idx(0)
    {
    }

    ~QDL()
    {
        m_SampleLibraries.clear();
        m_IssueQueues.clear();
        m_RespQueues.clear();
        m_CleanupQueues.clear();
        m_AckQueues.clear();
        m_IssueThreads.clear();
        m_BundleThreads.clear();
        m_CompletionThreads.clear();
        m_CleanupThreads.clear();
        m_AckThreads.clear();
        m_PollCleanerThreads.clear();
        m_PollAckCleanerThreads.clear();
    }

    void AddSampleLibrary(qsl::SampleLibraryPtr_t sl)
    {
        m_SampleLibraries.emplace_back(sl);
    }

    void SetupSettingsAndParams(QDLSettings& settings, QDLParams& params)
    {
        m_qdl_settings = settings;
        m_qdl_params = params;
    }

    void Setup();
    void Warmup();
    void Done();

    lwis_lon::SyncMsg sync_with_peer(lwis_lon::SyncMsg send_msg = lwis_lon::SyncMsg::ACK);

    const std::string& GetName();

    // virtual interface, wrapping SUT via network comm
    virtual const std::string& Name() override;
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples);
    virtual void FlushQueries(){};

    // polling the completion from SUT
    void BundleQueryCompletion(int const conn_idx);
    void QueryCompletion(int const conn_idx);
    void ResourceCleanup(int const conn_idx);
    void SendAck(int const conn_idx);
    void SendCleaner(int const conn_idx);
    void SendAckCleaner(int const conn_idx);

    void StartHeartbeat();
    void StopHeartbeat();

    // Sending samples received from IssueQuery
    void ProcessSamples(int conn_idx);

    // Check if NUMA is used
    bool UseNuma(bool is_SUT)
    {
        return is_SUT ? !m_qdl_settings.m_SUTNumaConfig.empty() : !m_qdl_settings.m_LONNumaConfig.empty();
    }

    // Get number of NUMA nodes
    int GetNbNumas(bool is_SUT)
    {
        auto my_cfg = is_SUT ? m_qdl_settings.m_SUTNumaConfig : m_qdl_settings.m_LONNumaConfig;
        return my_cfg.size() > 0 ? my_cfg.size() : 1;
    }

    // Get NUMA node index of a NIC
    int GetNumaIdx(std::string const& nic, bool is_SUT)
    {
        auto my_cfg = is_SUT ? m_qdl_settings.m_SUTNicNumaMap : m_qdl_settings.m_LONNicNumaMap;
        return UseNuma(is_SUT) ? my_cfg[nic] : 0;
    }

    // Get closest CPUs to a NIC
    std::vector<int> GetClosestCpus(std::string const& nic, bool is_SUT)
    {
        CHECK(UseNuma(is_SUT));
        auto my_cfg = is_SUT ? m_qdl_settings.m_SUTNumaConfig : m_qdl_settings.m_LONNumaConfig;
        auto numa_idx = GetNumaIdx(nic, is_SUT);
        return std::get<2>(my_cfg[numa_idx]);
    }

    // Get closest CPUs to a NIC
    std::vector<int> GetClosestCpusFromNumaNode(int node, bool is_SUT)
    {
        CHECK(UseNuma(is_SUT));
        auto my_cfg = is_SUT ? m_qdl_settings.m_SUTNumaConfig : m_qdl_settings.m_LONNumaConfig;
        return std::get<2>(my_cfg[node]);
    }

    std::size_t GetNumGpus(bool is_SUT)
    {
        return is_SUT ? m_qdl_settings.m_SUTGpuNumaMap.size() : 0;
    }

    IBConnsPtr_t GetConnections()
    {
        return m_Connections;
    }

protected:
    std::string m_Name;
    mutable std::string m_SUT_Name;
    QDLSettings m_qdl_settings;
    QDLParams m_qdl_params;

    std::mutex m_send_data_mutex;
    std::mutex m_recv_data_mutex;
    std::mutex m_send_ack_mutex;
    std::mutex m_recv_ack_mutex;
    std::condition_variable m_send_data_cv;
    std::condition_variable m_recv_data_cv;
    std::condition_variable m_send_ack_cv;
    std::condition_variable m_recv_ack_cv;

    // indices of m_Connections and m_SampleLibraries are sync'ed
    std::shared_ptr<lonib::IBConnections> m_Connections;
    std::vector<qsl::SampleLibraryPtr_t> m_SampleLibraries;

    // Issue & Completion
    std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>> m_OneIssueQueue;
    std::vector<std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>>> m_IssueQueues;
    std::vector<std::thread> m_IssueThreads;

    std::vector<std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySampleResponse>>> m_RespQueues;
    std::vector<std::thread> m_BundleThreads;

    std::vector<std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySampleResponse>>> m_CleanupQueues;
    std::vector<std::thread> m_CleanupThreads;

    std::vector<std::shared_ptr<lwis_lon::SyncQueue<std::tuple<mlperf::ResponseId, uint32_t>>>> m_AckQueues;
    std::vector<std::thread> m_AckThreads;

    std::vector<std::thread> m_CompletionThreads;

    std::vector<std::thread> m_PollCleanerThreads;
    std::vector<std::thread> m_PollAckCleanerThreads;

    // thread management
    std::atomic<bool> m_thread_active;

    // alive marker
    std::thread m_heartbeat_thread;
    std::atomic<bool> m_keep_alive;
    void KeepAlive();

    // idx for next IssueQuery device
    std::atomic<uint32_t> m_issue_idx;
    uint32_t m_num_issue_connections;

    // Simple RoundRobin
    const uint32_t GetNextIssueIdx(bool smart_balance)
    {
        // Find device with smallest number of transactions on-the-fly
        // FIXME: this may prefer earlier index as tie-breaker;
        //        expecting dynamic nature would prevent this from happening
        if (smart_balance)
        {
            std::vector<size_t> num_trx;
            for (int i = 0; i < m_num_issue_connections; ++i)
            {
                auto occ = m_Connections->get_connection(i)->get_send_om()->get_occupancy();
                num_trx.push_back(occ);
            }
            m_issue_idx = std::min_element(num_trx.begin(), num_trx.end()) - num_trx.begin();
        }
        // Simple roundrobin
        else
        {
            ++m_issue_idx;

            // overflow
            if (m_issue_idx >= m_num_issue_connections)
            {
                m_issue_idx = 0;
            }
        }
        return m_issue_idx;
    }

    void SetIssueIdx(uint32_t num)
    {
        m_issue_idx = num;
    }
};

}; // namespace lwis_lon

#endif // __QDL_HPP__
