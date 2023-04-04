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

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include "NvInfer.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <iostream>
#include <numaif.h>
#include <numeric>
#include <pthread.h>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "logger.h"
#include <glog/logging.h>

// set to 1 to collect latency stats
#define TIMER_ON 0

// For debugging the timing of each part
class Timer
{
public:
    Timer(const std::string& tag_)
        : tag(tag_)
    {
        gLogInfo << "Timer " << tag << " created." << std::endl;
    }
    void add(const std::chrono::duration<double, std::milli>& in)
    {
        ++count;
        total += in;
    }
    ~Timer()
    {
        gLogInfo << "Timer " << tag << " reports " << total.count() / count << " ms per call for " << count << " times."
                 << std::endl;
    }

private:
    std::string tag;
    std::chrono::duration<double, std::milli> total{0};
    size_t count{0};
};

#if TIMER_ON
#define TIMER_START(s)                                                                                                 \
    static Timer timer##s(#s);                                                                                         \
    auto start##s = std::chrono::high_resolution_clock::now();
#define TIMER_END(s) timer##s.add(std::chrono::high_resolution_clock::now() - start##s);
#else
#define TIMER_START(s)
#define TIMER_END(s)
#endif

/* Helper function to split a string based on a delimiting character */
inline std::vector<std::string> splitString(const std::string& input, const std::string& delimiter)
{
    std::vector<std::string> result;
    size_t start = 0;
    size_t next = 0;
    while (next != std::string::npos)
    {
        next = input.find(delimiter, start);
        result.emplace_back(input, start, next - start);
        start = next + 1;
    }
    return result;
}

/* Helper function to split a string based on a delimiting character, moving line by line*/
inline std::vector<std::string> splitStringLine(const std::string& s, char delim)
{
    std::vector<std::string> res;
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim))
    {
        res.push_back(item);
    }
    return res;
}

// Get element size of a data type.
inline unsigned int getElementSize(nvinfer1::DataType t)
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kINT8: return 1;
    // Fall through to error
    default: break;
    }
    throw std::runtime_error("Invalid DataType.");
    return 0;
}

// Return m rounded up to nearest multiple of n
inline int roundUp(int m, int n)
{
    return ((m + n - 1) / n) * n;
}

// Get the size of a binding dimensions
inline int64_t volume(
    const nvinfer1::Dims& d, const nvinfer1::TensorFormat& format, const bool hasImplicitBatch = false)
{
    nvinfer1::Dims d_new = d;
    // Get number of scalars per vector.
    int spv{1};
    switch (format)
    {
    case nvinfer1::TensorFormat::kCHW2: spv = 2; break;
    case nvinfer1::TensorFormat::kCHW4: spv = 4; break;
    case nvinfer1::TensorFormat::kHWC8: spv = 8; break;
    case nvinfer1::TensorFormat::kCHW16: spv = 16; break;
    case nvinfer1::TensorFormat::kCHW32: spv = 32; break;
    case nvinfer1::TensorFormat::kLINEAR:
    default: spv = 1; break;
    }
    if (spv > 1)
    {
        assert(d.nbDims >= 3); // Vectorized format only makes sense when nbDims>=3.
        d_new.d[d_new.nbDims - 3] = roundUp(d_new.d[d_new.nbDims - 3], spv);
    }
    // Skip the first dimension, which is batch dim.
    return std::accumulate(d_new.d + (hasImplicitBatch ? 0 : 1), d_new.d + d_new.nbDims, 1, std::multiplies<int64_t>());
}

inline int64_t volume(const nvinfer1::Dims& d, const bool hasImplicitBatch = false)
{
    return volume(d, nvinfer1::TensorFormat::kLINEAR, hasImplicitBatch);
}

// Create a shared pointer of an nvinfer1:: object which will be automatically destroyed when going out of scope.
struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

template <typename T>
inline std::shared_ptr<T> InferObject(T* obj)
{
    if (!obj)
    {
        throw std::runtime_error("Failed to create object");
    }
    return std::shared_ptr<T>(obj, InferDeleter());
}

namespace lwis_lon
{

// NUMA is expressed as a tuple of NIC IDs and GPU indices and CPU indices, in the order of NUMA node id
// NumaConfig[0]: NUMA node 0
// NumaConfig[0][0]: NIC IDs (device ID string)
// NumaConfig[0][1]: indices of GPU in node 0
// NumaConfig[0][2]: indices of CPU in node 0
using NumaConfig = std::vector<std::tuple<std::vector<std::string>, std::vector<int>, std::vector<int>>>;
// Find NUMA node from NIC ID
using NicNumaMap = std::unordered_map<std::string, int>;
// NIC pairs set as LON node NIC dev name : SUT node NIC dev name
using Nic2NicMap = std::vector<std::pair<std::string, std::string>>;
// SUT node NIC and GPU affinity info
using NicGpuAffn = std::unordered_map<std::string, int>;
// SUT node GPU NUMA map derived from NIC Numa map & NIC GPU affinity map
using GpuNumaMap = std::vector<int>;

// Helper to converts the range string (like "0,2-5,13-17") to a vector of ints.
inline std::vector<int> parseRange(const std::string& s)
{
    std::vector<int> results;
    auto ranges = splitString(s, ",");
    for (const auto& range : ranges)
    {
        auto startEnd = splitString(range, "-");
        CHECK(startEnd.size() <= 2) << "Invalid numa_config setting. Expects zero or one '-'.";
        if (startEnd.size() == 1)
        {
            results.push_back(std::stoi(startEnd[0]));
        }
        else
        {
            int start = std::stoi(startEnd[0]);
            int last = std::stoi(startEnd[1]);
            for (int i = start; i <= last; ++i)
            {
                results.push_back(i);
            }
        }
    }
    return results;
}

// Ex) "mlx5_0,mlx5_1:2,3:0-63&mlx5_2,mlx5_3:0,1:64,65,66,67" for 2 NUMA node
//     mlx5_0/mlx5_1/GPU2-3/CPU0-63 to node 0
inline NumaConfig parse_numa_config(std::string const& numa_cfg)
{
    NumaConfig cfg;
    if (!numa_cfg.empty())
    {
        auto nodes = splitString(numa_cfg, "&");
        for (auto const& node : nodes)
        {
            auto pair = splitString(node, ":");
            CHECK(pair.size() == 3) << "Invalid numa_config. Expecting two ':' per node";
            auto nics = splitString(pair[0], ",");
            auto gpus = parseRange(pair[1]);
            auto cpus = parseRange(pair[2]);
            cfg.emplace_back(std::make_tuple(nics, gpus, cpus));
        }
    }
    return cfg;
}

// Ex) mlx5_0:0, mlx5_1:0, mlx5_2:1, mlx5_3:1 in the above parse_numa_config() example
inline NicNumaMap parse_nic_numa_map(NumaConfig const& numa_cfg)
{
    NicNumaMap cfg;
    if (!numa_cfg.empty())
    {
        int i = 0;
        for (auto const& tuple : numa_cfg)
        {
            auto nics = std::get<0>(tuple);
            for (auto const& nic : nics)
            {
                CHECK(cfg.count(nic) == 0) << "Duplicate NIC found, possibly wrong numa_cfg";
                cfg[nic] = i;
            }
            i++;
        }
    }
    return cfg;
}

// Ex) "mlx5_0:mlx5_0&mlx5_1:mlx5_6" for LON mlx5_0 <=> SUT mlx5_0, LON mlx5_1 <=> SUT mlx5_6
inline Nic2NicMap parse_nic_mapping(std::string const& nic_map)
{
    CHECK(!nic_map.empty());
    Nic2NicMap map;
    auto pairs = splitString(nic_map, "&");
    for (auto const& pair : pairs)
    {
        auto nics = splitString(pair, ":");
        CHECK(nics.size() == 2) << "Invalid nic_mapping. Expecting exactly one ':' per nic pair";
        map.push_back(std::make_pair(nics[0], nics[1]));
    }
    return map;
}

// DON'T Ex) "mlx5_0:0,1&mlx5_1,mlx5_2:2" for mlx5_0 close to GPU0/1 while mlx5_1/2 close to GPU2
// Ex) "mlx5_0:0&mlx5_1:2" for mlx5_0 close to GPU0, mlx5_1 close to GPU2
inline NicGpuAffn parse_nic_gpu_affinity(std::string const& nic_gpu_aff)
{
    NicGpuAffn affn;
    CHECK(!nic_gpu_aff.empty());
    auto pairs = splitString(nic_gpu_aff, "&");
    for (auto const& pair : pairs)
    {
        auto map = splitString(pair, ":");
        CHECK(map.size() == 2) << "Invalid nic_gpu_affinity. Expecting exactly one ':' per nic-gpu pair";
        auto nics = splitString(map[0], ",");
        CHECK(nics.size() == 1) << "Need only one NIC per GPU";
        auto gpus = parseRange(map[1]);
        CHECK(gpus.size() == 1) << "Need only one GPU per NIC";
        CHECK(affn.count(nics[0]) == 0) << "Duplicate NIC found, possibly wrong nic_gpu_affinity";
        affn[nics[0]] = gpus[0];
    }
    return affn;
}

inline GpuNumaMap parse_gpu_numa_map(NumaConfig const& numa_cfg)
{
    GpuNumaMap map;
    if (!numa_cfg.empty())
    {
        int total_gpus = 0;
        for (auto const& tuple : numa_cfg)
        {
            total_gpus += std::get<1>(tuple).size();
        }
        map.resize(total_gpus);

        int i = 0;
        for (auto const& tuple : numa_cfg)
        {
            auto gpus = std::get<1>(tuple);
            for (auto const& gpu : gpus)
            {
                map[gpu] = i;
            }
            i++;
        }
    }
    return map;
}

}; // namespace lwis_lon

// Restrict mem allocation to specific NUMA node.
inline void bindNumaMemPolicy(const int32_t numaIdx, const int32_t nbNumas)
{
    unsigned long nodeMask = 1UL << numaIdx;
    long ret = set_mempolicy(MPOL_BIND, &nodeMask, nbNumas + 1);
    CHECK(ret >= 0) << std::strerror(errno);
}

// Reset mem allocation setting.
inline void resetNumaMemPolicy()
{
    long ret = set_mempolicy(MPOL_DEFAULT, nullptr, 0);
    CHECK(ret >= 0) << std::strerror(errno);
}

// Limit a thread to be on specific cpus.
inline void bindThreadToCpus(std::thread& th, const std::vector<int>& cpus, const bool ignore_esrch = false)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int cpu : cpus)
    {
        CPU_SET(cpu, &cpuset);
    }
    int ret = pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t), &cpuset);
    bool noerr = ignore_esrch ? ret == 0 || ret == ESRCH : ret == 0;
    CHECK(noerr) << std::strerror(ret);
}

class ScopedCudaStream
{
public:
    ScopedCudaStream()
    {
        CHECK_EQ(cudaStreamCreate(&stream), cudaSuccess);
    }
    ScopedCudaStream(const ScopedCudaStream&) = delete;

    ~ScopedCudaStream()
    {
        CHECK_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }

    cudaStream_t stream;
};

#endif // __UTILS_HPP__
