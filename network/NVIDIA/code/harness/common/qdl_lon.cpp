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

#ifndef __QDL_CPP__
#define __QDL_CPP__

#include "qdl_lon.hpp"

// QDL (Query Distribution Library) is an implementation of the MLPerf Query
// Distribution Library.
// It works as a proxy to SUT in LON node

namespace lwis_lon
{

void QDL::Setup()
{
    // install IBCommunications
    m_Connections = std::make_shared<lonib::IBConnections>(m_qdl_params.m_IBConns_params, m_qdl_params.m_is_SUT,
        m_qdl_params.m_SUT_NetID, m_qdl_params.m_LON_NetID, m_qdl_params.m_TCP_port);

    // enable thread
    m_thread_active = true;
    // enable CQ polling
    for (uint32_t i = 0; i < m_Connections->get_num_connections(); ++i)
    {
        m_Connections->get_connection(i)->set_continue_poll(true);
    }

    m_num_issue_connections = static_cast<uint32_t>(m_Connections->get_num_connections());
    m_IssueQueues.clear();
    m_CleanupQueues.clear();
    m_AckQueues.clear();
    m_RespQueues.clear();
    m_IssueQueues.resize(m_num_issue_connections, nullptr);
    m_CleanupQueues.resize(m_num_issue_connections, nullptr);
    m_AckQueues.resize(m_num_issue_connections, nullptr);
    m_RespQueues.resize(m_num_issue_connections, nullptr);

    SetIssueIdx(0);

    m_OneIssueQueue = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySample>>();

    auto const num_numa = GetNbNumas(m_qdl_params.m_is_SUT);
    std::vector<std::vector<int>> cpus_on_numa_node;
    cpus_on_numa_node.resize(num_numa);
    for (int i = 0; i < num_numa; ++i)
    {
        cpus_on_numa_node[i] = GetClosestCpusFromNumaNode(i, m_qdl_params.m_is_SUT);
    }

    // start threads for Issue & Completion
    for (uint32_t i = 0; i < m_num_issue_connections; ++i)
    {
        auto m_Con = m_Connections->get_connection(i);
        auto const dev_name = m_Con->get_resources()->get_configs().get_device_name();
        auto const numa_node = GetNumaIdx(dev_name, m_qdl_params.m_is_SUT);

        bindNumaMemPolicy(numa_node, num_numa);

        std::vector<int> cpu_to_bind;

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_PollAckCleanerThreads.emplace_back(std::thread(&QDL::SendAckCleaner, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_PollAckCleanerThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_PollCleanerThreads.emplace_back(std::thread(&QDL::SendCleaner, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_PollCleanerThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_BundleThreads.emplace_back(std::thread(&QDL::BundleQueryCompletion, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_BundleThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_CompletionThreads.emplace_back(std::thread(&QDL::QueryCompletion, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_CompletionThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_AckThreads.emplace_back(std::thread(&QDL::SendAck, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_AckThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_CleanupThreads.emplace_back(std::thread(&QDL::ResourceCleanup, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_CleanupThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_IssueThreads.emplace_back(std::thread(&QDL::ProcessSamples, this, i));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_IssueThreads.back(), cpu_to_bind);

        resetNumaMemPolicy();
    }
}

const std::string& QDL::Name()
{
    // Get SUT name through TCP comm
    int ret{-1};

    // sync message
    uint16_t my_msg{0x0101};
    uint16_t ur_msg{0x1010};

    // Expecting string, up to 31 characters + null termination(32B)
    std::string my_name(32, '\0');
    std::string ur_name(32, '\0');

    // LON sending QUERY_NAME, and expecting to get ACK
    my_msg = htons(static_cast<uint16_t>(lwis_lon::SyncMsg::QUERY_NAME));
    ret = m_Connections->sock_data_exchange(2, reinterpret_cast<char*>(&my_msg), reinterpret_cast<char*>(&ur_msg));
    CHECK(ret >= 0) << "TCP data exchange failed, ret=" + ret;
    CHECK(ntohs(ur_msg) == lwis_lon::SyncMsg::ACK) << "Sending SPELL_NAME failed";

    // LON sending ACK, and expecting to get name; no need for ntoh/hton here
    ret = m_Connections->sock_data_exchange(
        32, reinterpret_cast<char*>(my_name.data()), reinterpret_cast<char*>(ur_name.data()));
    CHECK(ret >= 0) << "TCP data exchange failed, ret=" + ret;

    m_SUT_Name = ur_name;
    m_SUT_Name.erase(std::find(m_SUT_Name.begin(), m_SUT_Name.end(), '\0'), m_SUT_Name.end());
    m_SUT_Name.shrink_to_fit();

    return m_SUT_Name;
}

lwis_lon::SyncMsg QDL::sync_with_peer(lwis_lon::SyncMsg send_msg)
{
    int ret{-1};

    // sync message
    uint16_t my_msg{0x0101};
    uint16_t ur_msg{0x1010};

    my_msg = htons(static_cast<uint16_t>(send_msg));
    ret = m_Connections->sock_data_exchange(2, reinterpret_cast<char*>(&my_msg), reinterpret_cast<char*>(&ur_msg));
    CHECK(ret >= 0) << "TCP data exchange failed, ret=" + ret;
    auto recvd_state = static_cast<lwis_lon::SyncMsg>(ntohs(ur_msg));

    return recvd_state;
}

void QDL::Warmup()
{
    // poll until SUT sends WARMUP done signal
    int ret{-1};
    lwis_lon::SyncMsg msg;

    msg = sync_with_peer(lwis_lon::SyncMsg::WARMUP_START);
    CHECK(msg == lwis_lon::SyncMsg::ACK) << "Sending WARMUP_START failed";
    std::cout << "Warmup request sent... ";

    msg = sync_with_peer(lwis_lon::SyncMsg::ACK);
    CHECK(msg == lwis_lon::SyncMsg::WARMUP_END) << "Warmup request failed";
    std::cout << "done." << std::endl;
}

void QDL::StartHeartbeat()
{
    m_keep_alive = true;
    m_heartbeat_thread = std::thread(&QDL::KeepAlive, this);
}

void QDL::StopHeartbeat()
{
    m_keep_alive = false;
    m_heartbeat_thread.join();
}

void QDL::KeepAlive()
{
    lwis_lon::SyncMsg msg;
    while (m_keep_alive)
    {
        std::this_thread::sleep_for(HEARTBEAT);

        msg = sync_with_peer();
        switch (msg)
        {
        case lwis_lon::SyncMsg::ACK: break;
        case lwis_lon::SyncMsg::ERROR:
            std::cerr << "ERROR: QDL Heartbeat got Sync State: ERROR, aborting..." << std::endl;
            std::abort();
            m_keep_alive = false;
            break;
        default:
            std::cerr << "ERROR: QDL Heartbeat got unexpected Sync State: " << std::hex << msg << std::dec << std::endl;
            CHECK(false) << "Terminating...";
            m_keep_alive = false;
            break;
        }
    }
}

void QDL::Done()
{
    // Send terminate signal, LON will shutdown/close the TCP
    lwis_lon::SyncMsg msg;
    msg = sync_with_peer(lwis_lon::SyncMsg::ALL_DONE);
    CHECK(msg == lwis_lon::SyncMsg::ACK) << "Sending ALL_DONE failed";

    // Shutdown/close socket so that SUT node can be gracefully release the port
    m_Connections->sock_shutdown();

    // Some messages helping closing threads
    for (uint32_t i = 0; i < m_Connections->get_num_connections(); ++i)
    {
        m_Connections->get_connection(i)->post_null_msgs();
        m_Connections->get_connection(i)->clear_continue_poll();
    }

    // Shutdown threads
    m_thread_active = false;

    // Wait for queue to deplete
    while (!m_OneIssueQueue->empty())
    {
    };
    for (auto& i : m_RespQueues)
        while (!i->empty())
        {
        };
    for (auto& i : m_IssueQueues)
        while (!i->empty())
        {
        };
    for (auto& i : m_CleanupQueues)
        while (!i->empty())
        {
        };
    for (auto& i : m_AckQueues)
        while (!i->empty())
        {
        };

    for (auto& thread : m_PollAckCleanerThreads)
        thread.join();
    for (auto& thread : m_PollCleanerThreads)
        thread.join();
    for (auto& thread : m_BundleThreads)
        thread.join();
    for (auto& thread : m_CompletionThreads)
        thread.join();
    for (auto& thread : m_AckThreads)
        thread.join();
    for (auto& thread : m_CleanupThreads)
        thread.join();
    for (auto& thread : m_IssueThreads)
        thread.join();
}

// send samples to SUT, to target NIC, round-robin on each sample
//  1. copy sample to correct send buffer in the RDMA registered memory region
//  2. post send
//  3. poll send completion - might be something ignored
//  4. push send OM
//  5. post recv
//  6. poll recv completion
//  7. Completes transactions with LoadGen via QuerySamplesComplete()
//  8. pop send OM
//  9. send ACK to SUT for SUT's OM to pop

// IssueQuery() pushes samples into SyncQueue called WorkQueue
// Sender threads as many as target SUT NICs spinning on WorkQueue
// Each thread grabs sample(s) from WorkQueue
// Each thread copies the sample(s) to proper buffer, post send, pushing send OM

// Each thread QuerySamplesComplete() / LoadGen will do log accuracy if needed
// pop send ringbuffer

void QDL::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    if (m_qdl_settings.m_OneIssueQueue)
    {
        TIMER_START(IssueQueryOneIssueQueueInsert);
        m_OneIssueQueue->insert(samples);
        TIMER_END(IssueQueryOneIssueQueueInsert);
    }
    else
    {
        if (m_qdl_settings.m_RoundRobinSamples)
        {
            TIMER_START(IssueQueryMultiIssueQueueRoundRobinPushBack);
            std::size_t samples_sz = samples.size();
            std::size_t stride = roundUp(samples_sz, m_num_issue_connections) / m_num_issue_connections;
            for (std::size_t it = 0; it < samples_sz; it += stride)
            {
                m_IssueQueues[GetNextIssueIdx(false)]->insert(samples, it, std::min(it + stride, samples_sz));
            }
            TIMER_END(IssueQueryMultiIssueQueueRoundRobinPushBack);
        }
        else
        {
            TIMER_START(IssueQueryMultiIssueQueueInsert);
            m_IssueQueues[GetNextIssueIdx(m_qdl_settings.m_SmartBalanceSamples)]->insert(samples);
            TIMER_END(IssueQueryMultiIssueQueueInsert);
        }
    }
}

void QDL::ProcessSamples(int const conn_idx)
{
    // run on thread, for a specific IBConnection, i.e. NIC-NIC connection
    // target GPU memory is decided by the target SUT NIC, to the neighboring one

    // threads on ProcessSamples(*) are competing on m_OneIssueQueue
    // this distributes requests randomly to target SUT accelerators

    // Or, round-robin distributor to m_IssueQueues

    std::stringstream ss;
    ss << "QDL::ProcessSamples CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    // create queue for this thread
    m_IssueQueues[conn_idx] = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySample>>();

    // expecting QDL on LON
    bool const is_SUT = false;

    // get my connection info using conn_id
    auto m_Con = m_Connections->get_connection(conn_idx);
    auto const m_TgtGpuIdx = m_Con->get_resources()->get_configs().get_gpu_idx();

    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // offset manager
    auto s_om = m_Con->get_send_om();

    //  base addresses of buffers for data exchange
    auto const sbuf = m_Con->get_resources()->get_send_buffer();
    auto const rbuf = m_Con->get_resources()->get_recv_buffer();

    // send element
    auto const send_elem_size = m_Con->get_resources()->get_send_elem_size();
    auto const send_elem_num = m_Con->get_resources()->get_send_elem_num();
    auto const send_size = send_elem_size * send_elem_num;

    auto const LON_memory_staging = m_Con->get_resources()->get_LON_memory_staging();

    // IB op to use
    auto ibv_op = IBV_WR_RDMA_WRITE_WITH_IMM;

    // QSL
    auto const dev_name = m_Con->get_resources()->get_configs().get_device_name();
    auto qsl_idx = LON_memory_staging ? GetNumaIdx(dev_name, is_SUT) : conn_idx;
    // wait untit registration is done
    do
    {
        std::this_thread::yield();
    } while (m_SampleLibraries.size() <= qsl_idx);
    auto m_QSL = m_SampleLibraries[qsl_idx];

    // choose issue queue
    auto myIssueQueue = m_qdl_settings.m_OneIssueQueue ? m_OneIssueQueue : m_IssueQueues[conn_idx];

    std::array<ibv_send_wr, MAX_BULK_SIZE + 1> sq_wr;
    std::array<ibv_recv_wr, MAX_BULK_SIZE + 1> rq_wr;
    std::array<ibv_sge, MAX_BULK_SIZE + 1> sq_sge;
    std::array<ibv_sge, MAX_BULK_SIZE + 1> rq_sge;

    auto const qsize
        = std::min(m_Con->get_resources()->get_recv_queue_size(), m_Con->get_resources()->get_send_queue_size());
    auto const my_bulksize = std::min(static_cast<std::size_t>(MAX_BULK_SIZE), qsize);

    while (m_thread_active)
    {
        std::deque<mlperf::QuerySample> samples;

        TIMER_START(ProcessSampleQueueEmpty);
        do
        {
            std::this_thread::yield();
        } while (myIssueQueue->empty() && m_thread_active);
        TIMER_END(ProcessSampleQueueEmpty);

        TIMER_START(ProcessSampleAcquireSamples);
        do
        {
            myIssueQueue->acquire(samples, m_qdl_settings.m_MaxWaitPerConnMicroSec, m_qdl_settings.m_MaxTrxPerConn,
                m_qdl_settings.m_EnMaxTrxPerConn);
        } while (samples.empty() && m_thread_active);
        TIMER_END(ProcessSampleAcquireSamples);

        std::size_t total = samples.size();
        std::size_t stride = total > my_bulksize ? my_bulksize : total;

        for (uint32_t j = 0; j < total; ++j)
        {
            auto const& sample = samples[j];

            auto cnt = j % stride;
            auto bulk_end = cnt >= stride - 1 || j >= total - 1;

            // Prep info
            auto resp_id = sample.id;

            // FIXME: also handle variable sample size and response size
            // such as: auto sample_size = m_QSL->GetSampleSize(sample.index); etc

            // exit condition
            if (resp_id == 0)
            {
                return;
            }

            TIMER_START(ProcessSampleOMPush);
            auto send_offset = s_om->push(resp_id) * send_size;
            TIMER_END(ProcessSampleOMPush);

            // prep send data from sample
            void* override_addr = 0;
            if (LON_memory_staging)
            {
                TIMER_START(ProcessSampleMemCpy);
                for (uint8_t i = 0; i < send_elem_num; ++i)
                {
                    // Multiple inputs are bundled in one consecutive memory region and
                    // are
                    // sent together
                    auto sample_addr = static_cast<char*>(m_QSL->GetSampleAddress(sample.index, i));
#ifndef QDL_DEBUG_DISABLE_MEMCOPY
                    memcpy(
                        reinterpret_cast<void*>(sbuf + send_offset + i * send_elem_size), sample_addr, send_elem_size);
#endif
                }
                TIMER_END(ProcessSampleMemCpy);
            }
            else
            {
                // multiple inputs are already packaged in consecutive memory location
                // in QSL::LoadSamplesToRam
                override_addr = m_QSL->GetSampleAddress(sample.index, 0);
            }
            TIMER_START(ProcessSampleBuildSendRecvWR);
            auto rtuple = m_Con->get_recv_wr(ibv_op, resp_id);
            auto stuple = m_Con->get_send_wr(ibv_op, resp_id, send_offset, send_elem_size * send_elem_num, send_offset,
                bulk_end, bulk_end, false, reinterpret_cast<uint64_t>(override_addr));
            auto& [rwr, rsge] = rtuple;
            auto& [swr, ssge] = stuple;
            rq_sge[cnt] = rsge;
            sq_sge[cnt] = ssge;
            rwr.next = &rq_wr[cnt + 1];
            swr.next = &sq_wr[cnt + 1];
            rwr.sg_list = &rq_sge[cnt];
            swr.sg_list = &sq_sge[cnt];
            rq_wr[cnt] = rwr;
            sq_wr[cnt] = swr;
            TIMER_END(ProcessSampleBuildSendRecvWR);

            if (bulk_end)
            {
                rq_wr[cnt].next = nullptr;
                sq_wr[cnt].next = nullptr;

                DLOG_IF(
                    INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                    << "Bulk post to RQ of " << (cnt + 1) << " WRs" << std::endl;

                // prep response by posting recv
                TIMER_START(ProcessSamplePostRQ);
                m_Con->bulk_post_to_RQ(rq_wr.front());
                TIMER_END(ProcessSamplePostRQ);

                DLOG_IF(
                    INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                    << "Bulk post to SQ of " << (cnt + 1) << " WRs" << std::endl;

                // send request by posting send
                TIMER_START(ProcessSamplePostSQ);
                m_Con->bulk_post_to_SQ(sq_wr.front());
                TIMER_END(ProcessSamplePostSQ);

                if (!cq_by_event)
                {
                    TIMER_START(ProcessSampleNotify);
                    std::lock_guard<std::mutex> lock(m_send_data_mutex);
                    m_send_data_cv.notify_one();
                    TIMER_END(ProcessSampleNotify);
                }
            }
        }
    }
}

void QDL::SendCleaner(int const conn_idx)
{
    // clean up the completions of DATA sent out

    std::stringstream ss;
    ss << "QDL::SendCleaner CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    auto m_Con = m_Connections->get_connection(conn_idx);

    auto const sq{true};
    auto const ack{false};
    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    while (m_thread_active)
    {
        if (!cq_by_event)
        {
            TIMER_START(SendCleanerLock);
            std::unique_lock l(m_send_data_mutex);
            TIMER_END(SendCleanerLock);

            TIMER_START(SendCleanerCV);
            m_send_data_cv.wait_for(l, THREAD_WAKEUP_TIME);
            TIMER_END(SendCleanerCV);
        }

        TIMER_START(SendCleanerPollCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(SendCleanerPollCQ);

        if (!results.empty())
        {
            auto r = results.back();
            auto& [id, imm, size] = r;

            // exit condition
            if (id == UINT64_MAX || imm == UINT32_MAX)
            {
                return;
            }
        }
    }
}

void QDL::SendAckCleaner(int const conn_idx)
{
    // clean up the completions of Acks sent out

    std::stringstream ss;
    ss << "QDL::SendAckCleaner CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    auto m_Con = m_Connections->get_connection(conn_idx);

    auto const sq{true};
    auto const ack{true};
    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    while (m_thread_active)
    {
        if (!cq_by_event)
        {
            TIMER_START(SendAckCleanerLock);
            std::unique_lock l(m_send_ack_mutex);
            TIMER_END(SendAckCleanerLock);

            TIMER_START(SendAckCleanerCV);
            m_send_ack_cv.wait_for(l, THREAD_WAKEUP_TIME);
            TIMER_END(SendAckCleanerCV);
        }

        TIMER_START(SendAckCleanerPollCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(SendAckCleanerPollCQ);

        if (!results.empty())
        {
            auto r = results.back();
            auto& [id, imm, size] = r;

            // exit condition
            if (id == UINT64_MAX || imm == UINT32_MAX)
            {
                return;
            }
        }
    }
}

void QDL::QueryCompletion(int const conn_idx)
{
    // Poll completion queue and if response arrives, get Immediate for ResponseId
    // (unique)
    // Then get QuerySampleResponse.data/.size from RDMA buffer and transfer size
    // With this, compile QuerySampleResponse and push it to CompletionQueue

    std::stringstream ss;
    ss << "QDL::QueryCompletion CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    // get my connection info using conn_id
    auto m_Con = m_Connections->get_connection(conn_idx);
    auto s_om = m_Con->get_send_om();

    auto const sq{false};
    auto const ack{false};
    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // buffer mgmt
    auto const rbuf = m_Con->get_resources()->get_recv_buffer();

    // recv element
    auto const recv_elem_size = m_Con->get_resources()->get_recv_elem_size();
    auto const recv_elem_num = m_Con->get_resources()->get_recv_elem_num();
    auto const recv_size = recv_elem_size * recv_elem_num;
    CHECK_EQ(recv_elem_num, 1) << "Does not support multiple responses per QuerySample yet";

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    // my completion queue to bundle
    do
    {
        std::this_thread::yield();
    } while (m_RespQueues[conn_idx] == nullptr);
    auto m_RespQueue = m_RespQueues[conn_idx];

    while (m_thread_active)
    {
        // completion will be taken care of through the queue here
        TIMER_START(QueryCompletionPollCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(QueryCompletionPollCQ);

        for (auto& r : results)
        {
            TIMER_START(QueryCompletionBuildResp);
            auto& [id_recvd, offset_recvd, size_recvd] = r;

            // exit condition
            if (id_recvd == UINT64_MAX || offset_recvd == UINT32_MAX)
            {
                return;
            }

            auto resp_addr = rbuf + offset_recvd;
            auto idx = offset_recvd / recv_size;

            mlperf::QuerySampleResponse resp;
            resp.id = s_om->peek(idx);
            resp.data = reinterpret_cast<uintptr_t>(resp_addr);
            resp.size = size_recvd;
            TIMER_END(QueryCompletionBuildResp);

            TIMER_START(QueryCompletionAddToBundleQueue);
            m_RespQueue->emplace_back(resp);
            TIMER_END(QueryCompletionAddToBundleQueue);
        }
    }
}

void QDL::BundleQueryCompletion(int const conn_idx)
{
    // This bundles completions into

    m_RespQueues[conn_idx] = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySampleResponse>>();

    std::stringstream ss;
    ss << "QDL::BundleQueryCompletion CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    // my completion queue to bundle
    auto m_BundleQueue = m_RespQueues[conn_idx];
    // resource cleanup queue to bundle
    do
    {
        std::this_thread::yield();
    } while (m_CleanupQueues[conn_idx] == nullptr);
    auto m_CleanupQueue = m_CleanupQueues[conn_idx];

    while (m_thread_active)
    {
        std::deque<mlperf::QuerySampleResponse> resps;

        TIMER_START(BundleQueryCompletionEmpty);
        do
        {
            std::this_thread::yield();
        } while (m_BundleQueue->empty() && m_thread_active);
        TIMER_END(BundleQueryCompletionEmpty);

        TIMER_START(BundleQueryCompletionAcquire);
        do
        {
            m_BundleQueue->acquire(resps, m_qdl_settings.m_MaxWaitPerConnMicroSec, m_qdl_settings.m_MaxTrxPerConn,
                m_qdl_settings.m_EnMaxTrxPerConn);
        } while (resps.empty() && m_thread_active);
        TIMER_END(BundleQueryCompletionAcquire);

        // bulk complete QuerySamples to LoadGen
        if (!resps.empty())
        {
            std::vector<mlperf::QuerySampleResponse> responses;
            TIMER_START(BundleQueryCompletionResponsesVectorCreation);
            for (auto& r : resps)
            {
                responses.push_back(r);
            }
            TIMER_END(BundleQueryCompletionResponsesVectorCreation);

            // Do not support callback at this moment
            // NOTE: may need to use callback, if response comes to GPU memory in LON
            // node: This is not supported yet
            TIMER_START(QuerySamplesComplete);
            mlperf::QuerySamplesComplete(&responses.front(), responses.size(), mlperf::ResponseCallback{});
            TIMER_END(QuerySamplesComplete);

            TIMER_START(BundleQueryCompletionAddToCleanupQueue);
            m_CleanupQueue->insert(responses);
            TIMER_END(BundleQueryCompletionAddToCleanupQueue);
        }
    }
}

void QDL::ResourceCleanup(int const conn_idx)
{
    // Clean up this connection's offset manager entry, so that new transaction
    // can be sent out

    // create queue
    m_CleanupQueues[conn_idx] = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySampleResponse>>();

    std::stringstream ss;
    ss << "QDL::ResourceCleanup CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    // get my connection info using conn_id
    auto m_Con = m_Connections->get_connection(conn_idx);
    auto s_om = m_Con->get_send_om();

    // buffer mgmt
    auto const rbuf = m_Con->get_resources()->get_recv_buffer();

    // recv element
    auto const recv_elem_size = m_Con->get_resources()->get_recv_elem_size();
    auto const recv_elem_num = m_Con->get_resources()->get_recv_elem_num();
    auto const recv_size = recv_elem_size * recv_elem_num;

    // queue to bundle ack messages to send
    do
    {
        std::this_thread::yield();
    } while (m_AckQueues[conn_idx] == nullptr);
    auto m_AckQueue = m_AckQueues[conn_idx];

    // resource cleanup queue to bundle
    auto m_CleanupQueue = m_CleanupQueues[conn_idx];

    while (m_thread_active)
    {
        std::deque<mlperf::QuerySampleResponse> resps;

        TIMER_START(ResourceCleanupEmpty);
        do
        {
            std::this_thread::yield();
        } while (m_CleanupQueue->empty() && m_thread_active);
        TIMER_END(ResourceCleanupEmpty);

        TIMER_START(ResourceCleanupAcquire);
        do
        {
            m_CleanupQueue->acquire(resps, m_qdl_settings.m_MaxWaitPerConnMicroSec, m_qdl_settings.m_MaxTrxPerConn,
                m_qdl_settings.m_EnMaxTrxPerConn);
        } while (resps.empty() && m_thread_active);
        TIMER_END(ResourceCleanupAcquire);

        if (!resps.empty())
        {
            TIMER_START(ResourceCleanupBuildAcks);
            std::vector<std::tuple<mlperf::ResponseId, uint32_t>> acks;
            acks.reserve(resps.size());
            for (auto& resp : resps)
            {
                uint32_t offset = static_cast<uint32_t>(resp.data - reinterpret_cast<uintptr_t>(rbuf));
                uint32_t index = offset / recv_size;

                // full roundtrip ensures send and recv buffer all consumed
                TIMER_START(ResourceCleanupOMPop);
                s_om->pop(index);
                TIMER_END(ResourceCleanupOMPop);

                acks.emplace_back(std::make_tuple(resp.id, offset));
            }
            TIMER_END(ResourceCleanupBuildAcks);

            TIMER_START(ResourceCleanupAckQueueInsert);
            m_AckQueue->insert(acks);
            TIMER_END(ResourceCleanupAckQueueInsert);
        }
    }
}

void QDL::SendAck(int const conn_idx)
{
    // Send ACK so that SUT can free the relevant offset manager entry

    m_AckQueues[conn_idx] = std::make_shared<lwis_lon::SyncQueue<std::tuple<mlperf::ResponseId, uint32_t>>>();

    std::stringstream ss;
    ss << "QDL::SendAck CON[" << conn_idx << "] -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    // get my connection info using conn_id
    auto m_Con = m_Connections->get_connection(conn_idx);

    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // queue to bundle ack messages to send
    auto m_AckQueue = m_AckQueues[conn_idx];

    // op
    auto const ibv_op = IBV_WR_RDMA_WRITE_WITH_IMM;

    // container for IB WRs
    std::array<ibv_send_wr, MAX_BULK_SIZE + 1> sq_wr;
    std::array<ibv_sge, MAX_BULK_SIZE + 1> sq_sge;

    auto const qsize
        = std::min(m_Con->get_resources()->get_recv_queue_size(), m_Con->get_resources()->get_send_queue_size());
    auto const my_bulksize = std::min(static_cast<std::size_t>(MAX_BULK_SIZE), qsize);

    while (m_thread_active)
    {
        std::deque<std::tuple<mlperf::ResponseId, uint32_t>> ack_info;

        TIMER_START(SendAckQueueEmpty);
        do
        {
            std::this_thread::yield();
        } while (m_AckQueue->empty() && m_thread_active);
        TIMER_END(SendAckQueueEmpty);

        TIMER_START(SendAckAcquire);
        do
        {
            m_AckQueue->acquire(ack_info, m_qdl_settings.m_MaxWaitPerConnMicroSec, m_qdl_settings.m_MaxTrxPerConn,
                m_qdl_settings.m_EnMaxTrxPerConn);
        } while (ack_info.empty() && m_thread_active);
        TIMER_END(SendAckAcquire);

        // bulk complete QuerySamples to LoadGen
        auto total = ack_info.size();

        std::size_t stride = total > my_bulksize ? my_bulksize : total;

        // send ACKs
        for (uint32_t j = 0; j < total; ++j)
        {
            TIMER_START(SendAckBuildSendWr);
            auto cnt = j % stride;
            auto bulk_end = cnt >= stride - 1 || j >= total - 1;

            auto r = ack_info[j];
            auto& [respId, offset] = r;

            auto tuple = m_Con->get_send_wr(ibv_op, respId, offset, 0, 0, bulk_end, bulk_end, true);
            auto& [wr, sge] = tuple;
            sq_sge[cnt] = sge;
            wr.next = &sq_wr[cnt + 1];
            wr.sg_list = &sq_sge[cnt];
            sq_wr[cnt] = wr;
            TIMER_END(SendAckBuildSendWr);

            if (bulk_end)
            {
                sq_wr[cnt].next = nullptr;

                DLOG_IF(
                    INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                    << "Bulk post to SQ of " << (cnt + 1) << " WRs" << std::endl;

                TIMER_START(SendAckBulkPostToSQ);
                m_Con->bulk_post_to_SQ(sq_wr.front(), true);
                TIMER_END(SendAckBulkPostToSQ);

                if (!cq_by_event)
                {
                    TIMER_START(SendAckNotify);
                    std::lock_guard<std::mutex> lock(m_send_ack_mutex);
                    m_send_ack_cv.notify_one();
                    TIMER_END(SendAckNotify);
                }
            }
        }
    }
}

}; // namespace lwis_lon

#endif // __QDL_HPP__
