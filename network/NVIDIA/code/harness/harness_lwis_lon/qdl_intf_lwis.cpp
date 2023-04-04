
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

#ifndef __QDL_INTF_LWIS_LON_CPP__
#define __QDL_INTF_LWIS_LON_CPP__

#include "qdl_intf_lwis.hpp"

// QDL INTF implements QDL functionality in SUT

namespace lwis_lon
{

// QDL_Interface inherits QDL with a bit of twist
void QDL_Interface::send_sut_name(std::string const& sut_name)
{
    // Get SUT name through TCP comm
    int ret{-1};

    // Expecting string, up to 31 characters + null termination(32B)
    std::string my_name(32, '\0');
    std::string ur_name(32, '\0');

    // copy sut_name to my_name
    std::strncpy(my_name.data(), sut_name.data(), (sut_name.length() > 31 ? 31 : sut_name.length()));

    // LON sending ACK, and expecting to get name; no need for ntoh/hton here
    ret = m_Connections->sock_data_exchange(
        32, reinterpret_cast<char*>(my_name.data()), reinterpret_cast<char*>(ur_name.data()));
    CHECK(ret >= 0) << "TCP data exchange failed, ret=" + ret;
}

void QDL_Interface::Setup()
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

    m_CurrentSyncState = m_PrimarySyncState = lwis_lon::SyncMsg::BUSY_WAIT;
}

void QDL_Interface::StartThreads()
{
    // start threads for Issue & Completion
    // this idx is for NIC order; worker threads are given GPU idx associated with
    // this NIC
    auto num_connections = static_cast<int>(m_Connections->get_num_connections());

    auto const num_numa = GetNbNumas(m_qdl_params.m_is_SUT);

    std::vector<std::vector<int>> cpus_on_numa_node;
    cpus_on_numa_node.resize(num_numa);
    for (int i = 0; i < num_numa; ++i)
    {
        cpus_on_numa_node[i] = GetClosestCpusFromNumaNode(i, m_qdl_params.m_is_SUT);
    }

    for (int i = 0; i < num_connections; ++i)
    {
        auto m_Con = m_Connections->get_connection(i);
        auto const dev_name = m_Con->get_resources()->get_configs().get_device_name();
        auto const m_GPU_idx = m_Con->get_resources()->get_gpu_idx();
        auto const m_QP_idx = m_Con->get_resources()->get_qp_idx();
        cudaSetDevice(m_GPU_idx);

        auto const numa_node = GetNumaIdx(dev_name, m_qdl_params.m_is_SUT);

        bindNumaMemPolicy(numa_node, num_numa);

        // manual control of CPU to map this thread
        std::vector<int> cpu_to_bind;

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_IssueThreads.emplace_back(std::thread(&QDL_Interface::Issue, this, i, m_QP_idx, m_GPU_idx));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_IssueThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_CompletionThreads.emplace_back(std::thread(&QDL_Interface::SampleCompletion, this, i, m_QP_idx, m_GPU_idx));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_CompletionThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_PollCleanerThreads.emplace_back(std::thread(&QDL_Interface::SampleCompletionCleaner, this, i, m_QP_idx));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_PollCleanerThreads.back(), cpu_to_bind);

        cpu_to_bind.clear();
        if (cpus_on_numa_node[numa_node].empty())
        {
            cpus_on_numa_node[numa_node] = GetClosestCpusFromNumaNode(numa_node, m_qdl_params.m_is_SUT);
        }
        cpu_to_bind.emplace_back(cpus_on_numa_node[numa_node].front());
        cpus_on_numa_node[numa_node].erase(cpus_on_numa_node[numa_node].begin());
        m_AckThreads.emplace_back(std::thread(&QDL_Interface::RecvAck, this, i, m_QP_idx));
        if (UseNuma(m_qdl_params.m_is_SUT))
            bindThreadToCpus(m_AckThreads.back(), cpu_to_bind);

        resetNumaMemPolicy();
    }

    m_PrimaryThread = std::thread(&QDL_Interface::Steward, this);
}

void QDL_Interface::Steward()
{
    // loops over TCP communication to initiate:
    // Warmup()
    // Name()
    // Done()
    // Inference requests are handled via IB communications

    std::stringstream ss;
    ss << "QDL_Interface::Steward -- on CPU " << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    lwis_lon::SyncMsg CurrentSyncState = get_qdl_state();
    bool m_exit{false};
    while ((lwis_lon::SyncMsg::SHUTDOWN != CurrentSyncState) && !m_exit)
    {
        switch (CurrentSyncState)
        {
        case lwis_lon::SyncMsg::SYNC_WAIT: set_qdl_state(sync_with_peer()); break;
        case lwis_lon::SyncMsg::QUERY_NAME:
            // already sent ACK, so ask for the sut name
            set_primary_state(lwis_lon::SyncMsg::QUERY_NAME);
            set_qdl_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        case lwis_lon::SyncMsg::WARMUP_START:
            set_primary_state(lwis_lon::SyncMsg::WARMUP_START);
            set_qdl_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        case lwis_lon::SyncMsg::WARMUP_END:
            sync_with_peer(lwis_lon::SyncMsg::WARMUP_END);
            set_qdl_state(lwis_lon::SyncMsg::SYNC_WAIT);
            break;
        case lwis_lon::SyncMsg::ALL_DONE:
            set_primary_state(lwis_lon::SyncMsg::ALL_DONE);
            set_qdl_state(lwis_lon::SyncMsg::BUSY_WAIT);
            break;
        case lwis_lon::SyncMsg::ACK: set_qdl_state(lwis_lon::SyncMsg::SYNC_WAIT); break;
        case lwis_lon::SyncMsg::BUSY_WAIT: break;
        case lwis_lon::SyncMsg::ERROR:
            std::cerr << "ERROR: QDL steward got Sync State: ERROR, aborting..." << std::endl;
            m_exit = true;
            std::abort();
            break;
        default:
            std::cerr << "ERROR: QDL steward got unexpected Sync State: " << std::hex << m_CurrentSyncState << std::dec
                      << std::endl;
            CHECK(false) << "Terminating...";
            m_exit = true;
            break;
        }
        CurrentSyncState = get_qdl_state();
    }
}

void QDL_Interface::Issue(int const con_idx, int const qp_idx, int const gpu_idx)
{
    // Receive requests to this GPU's memory via RDMA write
    // Immediate field gives offset; get data pointer using offset from Immediate
    // form mlperf::QuerySample
    // then push it to m_IssueQueue

    std::stringstream ss;
    ss << "QDL_Interface::Issue CON[" << con_idx << "]/QP[" << qp_idx << "]/GPU[" << gpu_idx << "] -- on CPU "
       << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    cudaSetDevice(gpu_idx);
    auto m_Con = m_Connections->get_connection(con_idx);
    auto m_IssueQueue = get_sut()->get_work_queue(gpu_idx);

    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // buffer mgmt
    auto const rbuf = m_Con->get_resources()->get_recv_buffer();
    auto s_om = m_Con->get_send_om();

    // Some sanity check for this SUT
    auto const recv_elem_num = m_Con->get_resources()->get_recv_elem_num();
    CHECK_EQ(recv_elem_num, 1) << "Expecting one input per QuerySample";

    // sample size
    auto const recv_elem_size = m_Con->get_resources()->get_recv_elem_size();

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    // op to use
    auto const ibv_op = IBV_WR_RDMA_WRITE_WITH_IMM;

    // Recv WR
    auto tuple = m_Con->get_recv_wr(ibv_op, gpu_idx);
    auto& [recv_wr, recv_sge] = tuple;
    recv_wr.sg_list = &recv_sge;

    // to push Recv WRs
    std::array<ibv_recv_wr, MAX_BULK_SIZE + 1> rq_wr;

    auto const qsize
        = std::min(m_Con->get_resources()->get_recv_queue_size(), m_Con->get_resources()->get_send_queue_size());
    auto const my_bulksize = std::min(static_cast<std::size_t>(MAX_BULK_SIZE), qsize);

    auto const sq{false};
    auto const ack{false};

    uint32_t my_seq_id = 0;

    // fill up the queue at the start (as opposed to bulksize)
    for (int j = 0; j < qsize; ++j)
    {
        auto cnt = j % my_bulksize;
        auto bulk_end = cnt >= my_bulksize - 1 || j >= qsize - 1;

        auto my_wr = recv_wr;
        my_wr.wr_id = my_wr.wr_id << 32 | my_seq_id;
        my_seq_id++;
        my_wr.next = cnt < my_bulksize - 1 ? &rq_wr[cnt + 1] : nullptr;
        rq_wr[cnt] = my_wr;

        if (bulk_end)
        {
            DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                << "Bulk post to RQ of " << my_bulksize << " WRs" << std::endl;

            TIMER_START(IssuePostToRQUpfront);
            m_Con->bulk_post_to_RQ(rq_wr.front());
            TIMER_END(IssuePostToRQUpfront);
        }
    }

    while (m_thread_active)
    {
        // prep for next sample
        // poll CQ for sample arrival
        TIMER_START(IssuePollCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(IssuePollCQ);

        if (!results.empty())
        {
            for (int i = 0; i < results.size(); i++)
            {
                auto my_wr = recv_wr;
                my_wr.wr_id = my_wr.wr_id << 32 | my_seq_id;
                my_seq_id++;
                my_wr.next = i < results.size() - 1 ? &rq_wr[i + 1] : nullptr;
                rq_wr[i] = my_wr;
            }

            DLOG_IF(INFO, std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                << "Bulk post to RQ of " << results.size() << " WRs" << std::endl;

            TIMER_START(IssuePostToRQ);
            m_Con->bulk_post_to_RQ(rq_wr.front());
            TIMER_END(IssuePostToRQ);

            std::vector<mlperf::QuerySample> samples;
            for (auto& r : results)
            {
                TIMER_START(IssueBuildQuerySample);
                auto& [id_recvd, offset_recvd, size_recvd] = r;

                // another exit condition
                if (id_recvd == UINT64_MAX || offset_recvd == UINT32_MAX)
                {
                    return;
                }

                auto rbuf_address = rbuf + offset_recvd;

                // sanity check... elem_size may be max size
                CHECK(size_recvd <= recv_elem_size * recv_elem_num);

                // compose mlperf::QuerySample; QuerySampleIndex is used for sample
                // address received
                mlperf::QuerySample sample;
                // encode QP idx to upper 32bit as offset is guaranteed to be in lower
                // 32bit
                sample.id
                    = (static_cast<mlperf::ResponseId>(qp_idx) << 32 | static_cast<mlperf::ResponseId>(offset_recvd));
                sample.index = reinterpret_cast<mlperf::QuerySampleIndex>(rbuf_address);
                TIMER_END(IssueBuildQuerySample);

                TIMER_START(IssueBuildQuerySampleInsert);
                samples.emplace_back(sample);
                TIMER_END(IssueBuildQuerySampleInsert);
            }

            TIMER_START(IssueIssueQueuePushBack);
            m_IssueQueue->insert(samples);
            TIMER_END(IssueIssueQueuePushBack);
        }
    }
}

void QDL_Interface::SampleCompletion(int const con_idx, int const qp_idx, int const gpu_idx)
{

    std::stringstream ss;
    ss << "QDL_Interface::SampleCompletion CON[" << con_idx << "]/QP[" << qp_idx << "]GPU[" << gpu_idx << "] -- on CPU "
       << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    cudaSetDevice(gpu_idx);

    auto m_Con = m_Connections->get_connection(con_idx);
    auto m_RespQueue = get_sut()->get_completion_queue(gpu_idx, qp_idx);
    auto m_RscRtnQueue = get_sut()->get_resource_return_queue(gpu_idx);

    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // buffer mgmt
    auto const sbuf = m_Con->get_resources()->get_send_buffer();
    auto s_om = m_Con->get_send_om();

    // RDMA may come to host mem7
    auto const SUT_send_with_host_mem_for_RDMA = m_Con->get_resources()->get_SUT_send_with_host_mem_for_RDMA();

    // send element
    auto const send_elem_size = m_Con->get_resources()->get_send_elem_size();
    auto const send_elem_num = m_Con->get_resources()->get_send_elem_num();
    auto const send_size = send_elem_size * send_elem_num;

    // recv size
    auto const recv_elem_size = m_Con->get_resources()->get_recv_elem_size();
    auto const recv_elem_num = m_Con->get_resources()->get_recv_elem_num();
    auto const recv_size = recv_elem_size * recv_elem_num;

    // sanity check for this SUT
    CHECK_EQ(send_elem_num, 1) << "Expecting one response per QuerySample";

    bool loop{true};

    auto ibv_op = IBV_WR_RDMA_WRITE_WITH_IMM;

    std::array<ibv_send_wr, MAX_BULK_SIZE + 1> sq_wr;
    std::array<ibv_recv_wr, MAX_BULK_SIZE + 1> rq_wr;
    std::array<ibv_sge, MAX_BULK_SIZE + 1> sq_sge;
    std::array<ibv_sge, MAX_BULK_SIZE + 1> rq_sge;

    std::deque<lwis_lon::Batch> batches;

    auto qsize = std::min(m_Con->get_resources()->get_recv_queue_size(), m_Con->get_resources()->get_send_queue_size());
    auto const my_bulksize = std::min(static_cast<std::size_t>(MAX_BULK_SIZE), qsize);

    cudaMemcpyKind const cpkind = SUT_send_with_host_mem_for_RDMA ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice;

    while (m_thread_active && loop)
    {
        batches.clear();

        TIMER_START(SampleCompletionEmpty);
        do
        {
            std::this_thread::yield();
        } while (m_RespQueue->empty() && m_thread_active);
        TIMER_END(SampleCompletionEmpty);

        TIMER_START(SampleCompletionAcquire);
        do
        {
            m_RespQueue->acquire(batches, m_qdl_settings.m_MaxWaitPerConnMicroSec, m_qdl_settings.m_MaxTrxPerConn,
                m_qdl_settings.m_EnMaxTrxPerConn);
        } while (batches.empty() && loop && m_thread_active);
        TIMER_END(SampleCompletionAcquire);

        for (auto& b : batches)
        {
            TIMER_START(SampleCompletionBuildResponse);
            // prepare resource to return
            lwis_lon::CudaResource resource;

            resource.Event = b.Event;
            resource.Stream = b.Stream;
            resource.ResId = b.ResId;
            resource.Terminate = b.Responses.empty();

            loop = !resource.Terminate;

#ifndef SUT_DEBUG_DISABLE_INFERENCE
            void* copy_src_addr{nullptr};
            void* copy_dst_addr{nullptr};
            std::size_t copy_size = 0;
            for (std::size_t i = 0; i < b.Responses.size(); ++i)
            {
                auto final = (i == (b.Responses.size() - 1));

                auto r = b.Responses[i];

                auto om_idx = static_cast<uint32_t>(r.id / recv_size);
                auto send_buff_offset = om_idx * send_size;
                auto sbuf_address = sbuf + send_buff_offset;
                auto data_size = r.size;

                if (i == 0)
                {
                    copy_src_addr = reinterpret_cast<void*>(r.data);
                    copy_dst_addr = reinterpret_cast<void*>(sbuf_address);
                }

                if ((static_cast<char*>(copy_src_addr) + copy_size == reinterpret_cast<char*>(r.data)
                        && static_cast<char*>(copy_dst_addr) + copy_size == sbuf_address))
                {
                    copy_size += data_size;
                }
                else
                {
#ifndef QDL_DEBUG_DISABLE_MEMCOPY
                    // copy response
                    TIMER_START(SampleCompletionMemCpy);
                    CHECK_EQ(cudaMemcpyAsync(copy_dst_addr, copy_src_addr, copy_size, cpkind, b.Stream), cudaSuccess);
                    TIMER_END(SampleCompletionMemCpy);
#endif
                    copy_src_addr = reinterpret_cast<void*>(r.data);
                    copy_dst_addr = reinterpret_cast<void*>(sbuf_address);
                    copy_size = data_size;
                }

                if (final)
                {
                    TIMER_START(SampleCompletionMemCpy);
                    CHECK_EQ(cudaMemcpyAsync(copy_dst_addr, copy_src_addr, copy_size, cpkind, b.Stream), cudaSuccess);
                    TIMER_END(SampleCompletionMemCpy);
                }
            }
            TIMER_START(SampleCompletionCudaStreamSync);
            CHECK_EQ(cudaStreamSynchronize(b.Stream), cudaSuccess);
            TIMER_END(SampleCompletionCudaStreamSync);
#endif
            TIMER_END(SampleCompletionBuildResponse);

            TIMER_START(SampleCompletionRscRtnQueueInsert);
            m_RscRtnQueue->emplace_back(resource);
            TIMER_END(SampleCompletionRscRtnQueueInsert);

            std::size_t total = b.Responses.size();
            std::size_t stride = total > my_bulksize ? my_bulksize : total;

            for (uint32_t j = 0; j < total; ++j)
            {
                TIMER_START(SampleCompletionBuildSendWr);
                auto r = b.Responses[j];

                auto cnt = j % stride;
                auto bulk_end = cnt >= stride - 1 || j >= total - 1;

                // prep buffer etc
                TIMER_START(SampleCompletionOMPush);
                auto om_idx = static_cast<uint32_t>(r.id / recv_size);
                s_om->push_by_index(r.id, om_idx);
                TIMER_END(SampleCompletionOMPush);

                auto data_size = r.size;
                auto send_buff_offset = om_idx * send_size;

                auto rtuple = m_Con->get_recv_wr(ibv_op, r.id, 0, 0, true);
                auto stuple = m_Con->get_send_wr(
                    ibv_op, r.id, send_buff_offset, data_size, send_buff_offset, bulk_end, bulk_end);
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
                TIMER_END(SampleCompletionBuildSendWr);

                if (bulk_end)
                {
                    rq_wr[cnt].next = nullptr;
                    sq_wr[cnt].next = nullptr;

                    DLOG_IF(INFO,
                        std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                        << "Bulk post to RQ of " << (cnt + 1) << " WRs" << std::endl;

                    // prep recv ACK for this send
                    TIMER_START(SampleCompletionPostRecvAck);
                    m_Con->bulk_post_to_RQ(rq_wr.front(), true);
                    TIMER_END(SampleCompletionPostRecvAck);

                    DLOG_IF(INFO,
                        std::greater<Logger::Severity>{}(gLogger.getReportableSeverity(), Logger::Severity::kINFO))
                        << "Bulk post to SQ of " << (cnt + 1) << " WRs" << std::endl;

                    // send data
                    TIMER_START(SampleCompletionPostToSQ);
                    m_Con->bulk_post_to_SQ(sq_wr.front());
                    TIMER_END(SampleCompletionPostToSQ);

                    if (!cq_by_event)
                    {
                        TIMER_START(SampleCompletionNotify);
                        std::lock_guard<std::mutex> lock(m_send_data_mutex);
                        m_send_data_cv.notify_one();
                        TIMER_END(SampleCompletionNotify);
                    }
                }
            }
        }
    }
}

// Recv Ack will clean up the send Offset Manager as it means the other side
// completed consuming the data sent over
void QDL_Interface::RecvAck(int const con_idx, int const qp_idx)
{
    std::stringstream ss;
    ss << "QDL_Interface::RecvAck CON[" << con_idx << "]/QP[" << qp_idx << "] -- on CPU " << sched_getcpu()
       << std::endl;
    gLogVerbose << ss.str();

    auto m_Con = m_Connections->get_connection(con_idx);
    auto s_om = m_Con->get_send_om();

    auto const sq{false};
    auto const ack{true};
    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // elements
    auto const send_elem_size = m_Con->get_resources()->get_send_elem_size();
    auto const send_elem_num = m_Con->get_resources()->get_send_elem_num();
    auto const send_size = send_elem_size * send_elem_num;

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    while (m_thread_active)
    {
        TIMER_START(RecvAckPollAckCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(RecvAckPollAckCQ);

        for (auto& r : results)
        {
            auto& [id, imm, size] = r;

            // exit condition
            if (id == UINT64_MAX || imm == UINT32_MAX)
            {
                return;
            }

            // Resp ACK uses offset for Resp, i.e. send offset from QDL Intf
            uint32_t index = imm / send_size;

            TIMER_START(RecvAckOMPop);
            s_om->pop(index);
            TIMER_END(RecvAckOMPop);
        }
    }
}

void QDL_Interface::SampleCompletionCleaner(int const con_idx, int const qp_idx)
{
    std::stringstream ss;
    ss << "QDL_Interface::SampleCompletionCleaner CON[" << con_idx << "]/QP[" << qp_idx << "] -- on CPU "
       << sched_getcpu() << std::endl;
    gLogVerbose << ss.str();

    auto m_Con = m_Connections->get_connection(con_idx);

    auto const sq{true};
    auto const ack{false};
    auto const cq_by_event = m_Con->get_resources()->get_CQ_wait_event();

    // completion collector
    auto wc_arr = std::make_shared<std::array<ibv_wc, MAX_BULK_SIZE>>();

    while (m_thread_active)
    {
        {
            TIMER_START(SampleCompletionCleanerLock);
            std::unique_lock l(m_send_data_mutex);
            TIMER_END(SampleCompletionCleanerLock);

            TIMER_START(SampleCompletionCleanerCV);
            m_send_data_cv.wait_for(l, THREAD_WAKEUP_TIME);
            TIMER_END(SampleCompletionCleanerCV);
        }

        TIMER_START(SampleCompletionCleanerPollCQ);
        std::vector<lonib::IBWC_data_t> results;
        if (cq_by_event)
        {
            results = *m_Con->bulk_poll_completion_by_event(sq, wc_arr, ack);
        }
        else
        {
            results = *m_Con->bulk_poll_completion(sq, wc_arr, ack);
        }
        TIMER_END(SampleCompletionCleanerPollCQ);

        if (!results.empty())
        {
            auto r = results.back();
            auto& [id, imm, size] = r;

            // another exit condition
            if (id == UINT64_MAX || imm == UINT32_MAX)
            {
                return;
            }
        }
    }
}

void QDL_Interface::Done()
{
    // Shutdown threads
    m_thread_active = false;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Some messages helping closing threads
    for (uint32_t i = 0; i < m_Connections->get_num_connections(); ++i)
    {
        m_Connections->get_connection(i)->post_null_msgs();
        m_Connections->get_connection(i)->clear_continue_poll();
    }

    // Shutdown/close socket
    m_Connections->sock_shutdown();

    for (uint32_t i = 0; i < m_Connections->get_num_connections(); ++i)
    {
        auto m_Con = m_Connections->get_connection(i);
        auto const m_GPU_idx = m_Con->get_resources()->get_gpu_idx();
        auto const m_QP_idx = m_Con->get_resources()->get_qp_idx();
        auto m_RespQueue = get_sut()->get_completion_queue(m_GPU_idx, m_QP_idx);
        auto m_IssueQueue = get_sut()->get_work_queue(m_GPU_idx);

        while (m_IssueQueue && !m_IssueQueue->empty())
        {
        };
        while (m_RespQueue && !m_RespQueue->empty())
        {
        };
    }

    for (auto& thread : m_AckThreads)
        thread.join();
    for (auto& thread : m_PollCleanerThreads)
        thread.join();
    for (auto& thread : m_CompletionThreads)
        thread.join();
    for (auto& thread : m_IssueThreads)
        thread.join();
    m_PrimaryThread.join();
}

}; // namespace lwis_lon

#endif // __QDL_INTF_LWIS_LON_CPP___