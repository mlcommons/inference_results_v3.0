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

#pragma once

#include <pthread.h>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#define ALIGN(x, a) (((x) + (a)-1) & ~((a)-1))
#define FOCEINLINE __attribute__((always_inline))
#define LIKELY(x) __builtin_expect(!!(x), 1)
#define UNLIKELY(x) __builtin_expect(!!(x), 0)

using BertInputDataType = int32_t;
using BertOutputDataType = _Float32;
constexpr std::size_t BertMaxSeqLength = 512;
// constexpr std::size_t BertMaxSeqLength = 384;
constexpr std::size_t BertMaxSeqPerBatch = 3;

constexpr std::size_t BertInputIdsIdx = 0;
constexpr std::size_t BertSegmentIdsIdx = 1;
constexpr std::size_t BertPosIdsIdx = 2;
constexpr std::size_t BertPosMaskIdx = 3;

namespace spu_backend {
class PerfTimer;
}

#ifdef DEBUG
#define TIME(s, i) moffett::spu_backend::PerfTimer _(s, i)
#define TIME_START(s, i) moffett::spu_backend::PerfTimer _(s, i) {
#define TIME_END }
#else
#define TIME(s, i)
#define TIME_START(s, i)
#define TIME_END
#endif

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter) {
  std::vector<std::string> result;
  size_t start = 0;
  size_t next = 0;
  while (next != std::string::npos) {
    next = input.find(delimiter, start);
    result.emplace_back(input, start, next - start);
    start = next + 1;
  }
  return result;
}

inline std::vector<std::string> split(const std::string& s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

inline std::vector<int32_t> GetDevicesIdsFromString(std::string& device_ids) {
  std::vector<int32_t> devices;
  auto device_names = split(device_ids, ',');
  std::for_each(device_names.cbegin(), device_names.cend(),
                [&devices](const std::string& idx) { devices.emplace_back(std::stoi(idx)); });

  return devices;
}
