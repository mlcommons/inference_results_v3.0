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
 * Copyright © 2022 Shanghai Biren Technology Co., Ltd. All rights reserved.
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

#pragma once

#include <atomic>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include <sys/time.h>

#include "system_under_test.h"

long get_nanosecond();

// For debugging the timing of each part
class Timer {
public:
  Timer(const std::string& tag_) : tag(tag_) { std::cout << "Timer " << tag << " created." << std::endl; }
  void add(const std::chrono::duration<double, std::milli>& in) {
    ++count;
    total += in;
  }
  ~Timer() {
    std::cout << "Timer " << tag << " reports " << total.count() / count << " ms per call for " << count << " times."
              << std::endl;
  }

private:
  std::string tag;
  std::chrono::duration<double, std::milli> total{0};
  std::size_t count{0};
};

#define TIMER_ON 0

#if TIMER_ON
#define TIMER_START(s)       \
  static Timer timer##s(#s); \
  auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_END(s)
#endif

namespace moffett {
namespace spu_backend {

class PerfTimer {
 public:
  explicit PerfTimer(const char *op, std::size_t device_idx)
      : op_(op), device_idx_(device_idx), k_start_(std::chrono::high_resolution_clock::now()) {
    char buff[60] = {0};
    printTime(k_start_, buff, 33);
    std::stringstream ss;
    ss << buff << "      " << op_ << " starts on " << device_idx_ << std::endl;
    std::cout << ss.str();
  }

  ~PerfTimer() {
    auto end = std::chrono::high_resolution_clock::now();
    char buff[60] = {0};
    printTime(end, buff, 33);
    std::stringstream ss;
    ss << buff << "      " << op_ << " ends on " << device_idx_
       << ". duration: " << std::chrono::duration_cast<std::chrono::microseconds>(end - k_start_).count() << std::endl;
    std::cout << ss.str();
    fflush(stdout);
  }

 private:
  std::string op_;
  const std::size_t device_idx_;
  const std::chrono::time_point<std::chrono::high_resolution_clock> k_start_;

  void printTime(const std::chrono::high_resolution_clock::time_point &p, char *buff, std::size_t size);
};

std::vector<std::string> ParseFileList(const char *arg);

} // namespace spu_backend
}
