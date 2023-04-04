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

#include <cassert>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <dirent.h>
#include <sys/stat.h>

#include "timer.h"

long get_nanosecond() {
  struct timespec t;
  clock_gettime(CLOCK_REALTIME, &t);
  return t.tv_sec*1000000000+t.tv_nsec;
}

namespace moffett {
namespace spu_backend {

void PerfTimer::printTime(const std::chrono::high_resolution_clock::time_point &p, char *buff, std::size_t size) {
  static constexpr std::size_t
  k_day_time_length = 19;
  static constexpr std::size_t
  k_ms_length = 4;
  static constexpr std::size_t
  k_us_length = 7;
  assert(size >= k_day_time_length + k_us_length - 1);
  auto fraction = p - std::chrono::time_point_cast<std::chrono::seconds>(p);
  std::time_t t = std::chrono::high_resolution_clock::to_time_t(p);
  std::strftime(buff, k_day_time_length + 1, "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));
  snprintf(buff + k_day_time_length, k_us_length + 1, ".%06d",
           static_cast<int>(fraction / std::chrono_literals::operator ""us(1)));
}

static std::vector<uint32_t> SoftmaxShape(const std::vector<uint32_t> &in_shape, int32_t axis) {
  // convert in_data shape from {d0, d1, d2,...dn} to {N,D} matrix
  uint32_t N = 1;
  for (uint32_t i = 0; i < axis; ++i) {
    N *= in_shape[i];
  }

  uint32_t D = 1;
  for (uint32_t i = axis; i < in_shape.size(); ++i) {
    D *= in_shape[i];
  }

  return {N, D};
}

static std::vector<float> Softmax(const std::vector<float> &in_data, const std::vector<uint32_t> &in_shape,
                                  int32_t axis) {
  std::vector<float> out_data(in_data.size());
  auto ND = SoftmaxShape(in_shape, axis);
  auto N = ND[0];
  auto D = ND[1];

  for (uint32_t i = 0; i < N; ++i) {
    std::vector<float> exp_sum_list(D);

    for (uint32_t j = 0; j < D; ++j) {
      exp_sum_list[j] = 0.f;
      for (uint32_t k = 0; k < D; ++k) {
        exp_sum_list[j] += exp(in_data[i * D + k] - in_data[i * D + j]);
      }
    }

    for (uint32_t j = 0; j < D; ++j) {
      out_data[i * D + j] = 1.0f / exp_sum_list[j];
    }
  }

  return out_data;
}

std::vector<std::string> ParseFileList(const char *arg) {
  std::vector<std::string> file_names;

  struct stat file_stat;
  std::string path(arg);
  if (stat(path.c_str(), &file_stat)) {
    printf("Path %s doesn't exist.\n", path.c_str());
    if (!(file_stat.st_mode & (S_IFDIR | S_IFREG))) {
      printf("Path %s is not dir or regular file.\n", path.c_str());
      return file_names;
    }
  }

  if (file_stat.st_mode & S_IFDIR) {
    DIR *dir = opendir(path.c_str());
    if (dir == NULL) {
      printf("Couldn't open directory %s.\n", path.c_str());
      return file_names;
    }
    struct dirent *dir_ent;
    std::string real_file_name;
    while ((dir_ent = readdir(dir)) != NULL) {
      if (dir_ent->d_type == DT_REG) {
        real_file_name = path + '/';
        real_file_name += dir_ent->d_name;
        file_names.emplace_back(real_file_name);
      }
    }
  } else if (file_stat.st_mode & S_IFREG) {
    std::string real_file_name = path;
    file_names.emplace_back(real_file_name);
  }

  std::sort(file_names.begin(), file_names.end());

  return file_names;
}

} // namespace spu_backend
}
