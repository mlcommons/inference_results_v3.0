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

#ifndef __LON_COMMON_HPP__
#define __LON_COMMON_HPP__

#include "NvInfer.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <functional>
#include <glog/logging.h>
#include <iostream>
#include <map>
#include <mutex>
#include <numaif.h>
#include <numeric>
#include <pthread.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "loadgen.h"
#include "query_sample_library.h"
#include "utils.hpp"

namespace lwis_lon
{

struct Batch
{
    std::vector<mlperf::QuerySampleResponse> Responses;
    cudaEvent_t Event;
    cudaStream_t Stream;
    std::size_t ResId;
    uint32_t QueuePairId;
    std::optional<mlperf::ResponseCallback> ResponseCb;
};

struct CudaResource
{
    cudaEvent_t Event;
    cudaStream_t Stream;
    std::size_t ResId;
    bool Terminate;
};

enum SyncMsg
{
    SYNC_WAIT = 0x0,
    BUSY_WAIT = 0x10,
    ERROR = 0x1010,
    ACK = 0x1111,
    ALL_DONE = 0xDEAD,
    QUERY_NAME = 0xDEAF,
    WARMUP_START = 0xEA5E,
    WARMUP_END = 0xFEED,
    SHUTDOWN = 0xFFFF,
};

using namespace std::chrono_literals;

template <typename T>
class SyncQueue
{
public:
    typedef typename std::deque<T>::iterator iterator;

    SyncQueue()
        : m_Size(0)
    {
    }

    bool empty()
    {
        // empty() is sometimes called in a spinloop
        // don't lock here because that can cause lock contention
        // when locking/unlocking in the spinloop
        return 0 == m_Size.load(std::memory_order_relaxed);
    }

    void insert(const std::vector<T>& values)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin(), values.end());
            m_Size.store(m_Queue.size(), std::memory_order_release);
        }
        m_Condition.notify_one();
    }

    void insert(const std::vector<T>& values, const size_t begin_idx, const size_t end_idx)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.insert(m_Queue.end(), values.begin() + begin_idx, values.begin() + end_idx);
            m_Size.store(m_Queue.size(), std::memory_order_release);
        }
        m_Condition.notify_one();
    }

    void acquire(
        std::deque<T>& values, std::chrono::microseconds duration = 10000us, size_t size = 1, bool limit = false)
    {
        size_t remaining = 0;

        TIMER_START(m_Mutex_create);
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            TIMER_END(m_Mutex_create);
            TIMER_START(m_Condition_wait_for);
            m_Condition.wait_for(l, duration, [=] { return m_Queue.size() >= size; });
            TIMER_END(m_Condition_wait_for);

            if (!limit || m_Queue.size() <= size)
            {
                TIMER_START(swap);
                values.swap(m_Queue);
                TIMER_END(swap);
            }
            else
            {
                auto beg = m_Queue.begin();
                auto end = beg + size;
                TIMER_START(values_insert);
                values.insert(values.end(), beg, end);
                TIMER_END(values_insert);
                TIMER_START(m_Queue_erase);
                m_Queue.erase(beg, end);
                TIMER_END(m_Queue_erase);
                remaining = m_Queue.size();
            }
            m_Size.store(m_Queue.size(), std::memory_order_release);
        }

        // wake up any waiting threads
        if (remaining)
            m_Condition.notify_one();
    }

    void push_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.push_back(v);
            m_Size.store(m_Queue.size(), std::memory_order_release);
        }
        m_Condition.notify_one();
    }

    void emplace_back(T const& v)
    {
        {
            std::unique_lock<std::mutex> l(m_Mutex);
            m_Queue.emplace_back(v);
            m_Size.store(m_Queue.size(), std::memory_order_release);
        }
        m_Condition.notify_one();
    }

    T front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        return r;
    }

    T front_then_pop()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Condition.wait(l, [=] { return !m_Queue.empty(); });
        T r(std::move(m_Queue.front()));
        m_Queue.pop_front();
        m_Size.store(m_Queue.size(), std::memory_order_release);
        return r;
    }

    void pop_front()
    {
        std::unique_lock<std::mutex> l(m_Mutex);
        m_Queue.pop_front();
        m_Size.store(m_Queue.size(), std::memory_order_release);
    }

private:
    mutable std::mutex m_Mutex;
    std::condition_variable m_Condition;

    std::deque<T> m_Queue;
    std::atomic<size_t> m_Size;
};

}; // namespace lwis_lon

#endif // __LON_COMMON_HPP__