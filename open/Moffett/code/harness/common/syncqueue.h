/*
 * Copyright (c) 1993-2021, NVIDIA CORPORATION.  All rights reserved.
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

/*
 * Copyright © 2023 Moffett System Inc. All rights reserved.
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

#include <chrono>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <vector>

#include "timer.h"

using namespace std::chrono_literals;

template<typename T>
class SyncQueue {
public:
  typedef typename std::deque<T>::iterator iterator;

  SyncQueue() {}

  bool empty() {
    std::unique_lock<std::mutex> l(m_Mutex);
    return m_Queue.empty();
  }

  size_t size() {
    std::unique_lock<std::mutex> l(m_Mutex);
    return m_Queue.size();
  }

  void insert(const std::vector<T>& values) {
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      m_Queue.insert(m_Queue.end(), values.begin(), values.end());
    }
    m_Condition.notify_one();
  }

  void insert(const std::vector<T>& values, const std::size_t begin_idx, const std::size_t end_index) {
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      m_Queue.insert(m_Queue.end(), values.begin() + begin_idx, values.begin() + end_index);
    }
    m_Condition.notify_one();
  }

  void acquire(std::deque<T>& values, std::chrono::microseconds duration = 10000us, std::size_t size = 1,
               bool limit = false) {
    std::size_t remaining = 0;

    TIMER_START(m_Mutex_create);
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      TIMER_END(m_Mutex_create);
      TIMER_START(m_Condition_wait_for);
      m_Condition.wait_for(l, duration, [=] { return m_Queue.size() >= size; });
      TIMER_END(m_Condition_wait_for);

      if (!limit || m_Queue.size() <= size) {
        TIMER_START(swap);
        values.swap(m_Queue);
        TIMER_END(swap);
      } else {
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
    }

    // wake up any waiting threads
    if (remaining)
      m_Condition.notify_one();
  }

  void push_back(T const& v) {
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      m_Queue.push_back(v);
    }
    m_Condition.notify_one();
  }
  void push_back(T&& v) {
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      m_Queue.push_back(std::move(v));
    }
    m_Condition.notify_one();
  }
  void emplace_back(T const& v) {
    {
      std::unique_lock<std::mutex> l(m_Mutex);
      m_Queue.emplace_back(v);
    }
    m_Condition.notify_one();
  }
  T front() {
    std::unique_lock<std::mutex> l(m_Mutex);
    m_Condition.wait(l, [=] { return !m_Queue.empty(); });
    T r(std::move(m_Queue.front()));
    return r;
  }
  T front_then_pop() {
    std::unique_lock<std::mutex> l(m_Mutex);
    m_Condition.wait(l, [=] { return !m_Queue.empty(); });
    T r(std::move(m_Queue.front()));
    m_Queue.pop_front();
    return r;
  }
  void pop_front() {
    std::unique_lock<std::mutex> l(m_Mutex);
    m_Queue.pop_front();
  }

private:
  mutable std::mutex m_Mutex;
  std::condition_variable m_Condition;

  std::deque<T> m_Queue;
};

#define QUEUE_TYPE SyncQueue
#define PUSH_QUEUE(q, element) (q).push_back(element)
#define POP_QUEUE(q) (q).front_then_pop()
#define ACQUIRE_QUEUE_TIMEOUT(q, element, timeout) (q).acquire(element, timeout, 1, true)
