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

#ifndef __QDL_INTF_BERT_LON_HPP__
#define __QDL_INTF_BERT_LON_HPP__

#include <csignal>

#include "bert_lon_common.h"
#include "bert_lon_server.h"
#include "lwis_lon.hpp"
#include "qdl_lon.hpp"

// QDL INTF implements QDL functionality in SUT

namespace bert_lon
{

class QDL_Interface;

using QdlIntfPtr_t = std::shared_ptr<QDL_Interface>;

class QDL_Interface : public lwis_lon::QDL
{
public:
    QDL_Interface()
        : QDL()
    {
    }
    QDL_Interface(std::string name)
        : QDL(name)
    {
    }

    ~QDL_Interface()
    {
        m_SampleLibraries.clear();
        m_IssueThreads.clear();
        m_BundleThreads.clear();
        m_CompletionThreads.clear();
        m_PollCleanerThreads.clear();
        m_PollAckCleanerThreads.clear();
        m_AckThreads.clear();
    }

    void Setup();
    void StartThreads();

    void Steward();

    void send_sut_name(std::string const& name);

    void set_qdl_state(lwis_lon::SyncMsg state)
    {
        std::lock_guard<std::mutex> lock(cs_mutex);
        m_CurrentSyncState = state;
    }

    void set_primary_state(lwis_lon::SyncMsg state)
    {
        std::lock_guard<std::mutex> lock(ms_mutex);
        m_PrimarySyncState = state;
    }

    lwis_lon::SyncMsg get_qdl_state()
    {
        return m_CurrentSyncState;
    }

    lwis_lon::SyncMsg get_primary_state()
    {
        return m_PrimarySyncState;
    }

    const std::string& MyName()
    {
        return m_Name;
    }

    // virtual interface
    virtual const std::string& Name()
    {
        return m_Name;
    };
    virtual void IssueQuery(const std::vector<mlperf::QuerySample>& samples){};
    virtual void FlushQueries(){};

    void Issue(int const con_idx, int const qp_idx, int const gpu_idx);
    void SampleCompletion(int const con_idx, int const qp_idx, int const gpu_idx, int const bc_idx);
    void SampleCompletionCleaner(int const con_idx, int const qp_idx);
    void RecvAck(int const con_idx, int const qp_idx);
    void Done();
    void set_sut(ServerPtr_t sut)
    {
        m_SUT = sut;
    }
    ServerPtr_t get_sut()
    {
        return m_SUT;
    }

protected:
    std::thread m_PrimaryThread;

    std::vector<std::thread> m_AckThreads;

    std::mutex cs_mutex;
    std::mutex ms_mutex;
    std::atomic<lwis_lon::SyncMsg> m_CurrentSyncState;
    std::atomic<lwis_lon::SyncMsg> m_PrimarySyncState;
    ServerPtr_t m_SUT;
};

}; // namespace bert_lon

#endif // __QDL_INTF_BERT_LON_HPP__
