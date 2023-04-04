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

#ifndef __LWIS_HPP__
#define __LWIS_HPP__

#include <atomic>
#include <deque>
#include <map>
#include <thread>
#include <vector>

#include "NvInfer.h"
#include <cuda.h>
#include <cuda_runtime.h>

#include "common.hpp"
#include "lwis_buffers.h"
#include "system_under_test.h"
#include "test_settings.h"
#include "utils.hpp"

// LWIS (Light Weight Inference Server) is a simple server composed of the
// following components:
// Simple batcher
// Engine creation
// Host/Device Memory management
// Batch management and data reporting
//
// The goals of LWIS are:
// - Thin layer between MLPerf interface and Cuda/TRT APIs.
// - Single source file and minimal dependencies.

// SUT_DEBUG_DISABLE_INFERENCE disables all device operations including data copies, most events, and compute
// SUT_DEBUG_DISABLE_COMPUTE disables the compute (TRT enqueue) portion of inference
// #define SUT_DEBUG_DISABLE_INFERENCE
// #define SUT_DEBUG_DISABLE_COMPUTE

namespace lwis_lon
{
class BufferManager;

using namespace std::chrono_literals;

class Device;
class Engine;
class Server;

typedef std::shared_ptr<Device> DevicePtr_t;
typedef std::shared_ptr<Engine> EnginePtr_t;
typedef std::shared_ptr<Server> ServerPtr_t;

typedef std::shared_ptr<::nvinfer1::ICudaEngine> ICudaEnginePtr_t;
struct ServerSettings
{
    bool EnableCudaGraphs{false};
    bool EnableSyncOnEvent{false};
    bool EnableSpinWait{false};
    bool EnableDeviceScheduleSpin{false};
    bool EnableResponse{true};
    bool EnableDequeLimit{false};
    bool EnableBatcherThreadPerDevice{true};
    bool EnableCudaThreadPerDevice{false};
    bool EnableDma{true};
    bool EnableDmaStaging{false};
    bool RunInferOnCopyStreams{false};
    bool UseSameContext{false};
    bool SutUsesHostMemForRdma{false};
    bool SutRecvWithHostMemForRdma{false};
    bool SutSendWithHostMemForRdma{false};

    size_t GPUBatchSize{1};
    size_t GPUCopyStreams{4};
    size_t GPUInferStreams{4};
    int32_t MaxGPUs{-1};

    size_t NumIBQPsPerNIC{1};

    bool ForceContiguous{false};
    size_t CompleteThreads{1};

    std::chrono::microseconds Timeout{10000us};
    NumaConfig m_NumaConfig;
    GpuNumaMap m_GpuNumaMap;
};

struct ServerParams
{
    std::string DeviceNames;
    // <perDeviceType, perDevice, perIteration>
    std::vector<std::vector<std::vector<std::string>>> EngineNames;
};

// captures execution engine for performing inference
class Device
{
    friend Server;

public:
    Device(size_t id, size_t numCopyStreams, size_t numInferStreams, size_t numCompleteThreads, bool enableSpinWait,
        bool enableDeviceScheduleSpin, size_t batchSize, bool useSameContext, bool SutRecvWithHostMemForRdma,
        bool SutSendWithHostMemForRdma, uint32_t num_QPs)
        : m_Id(id)
        , m_CopyStreams(numCopyStreams)
        , m_InferStreams(numInferStreams)
        , m_EnableSpinWait(enableSpinWait)
        , m_EnableDeviceScheduleSpin(enableDeviceScheduleSpin)
        , m_BatchSize(batchSize)
        , m_UseSameContext(useSameContext)
        , m_SutRecvWithHostMemForRdma(SutRecvWithHostMemForRdma)
        , m_SutSendWithHostMemForRdma(SutSendWithHostMemForRdma)
        , m_num_QPs(num_QPs)
    {
        m_Dev_CompletionQueue.resize(num_QPs);
        for (int i = 0; i < num_QPs; i++)
        {
            m_Dev_CompletionQueue[i] = std::make_shared<lwis_lon::SyncQueue<lwis_lon::Batch>>();
        }
        m_Dev_ResourceReturnQueue = std::make_shared<lwis_lon::SyncQueue<lwis_lon::CudaResource>>();
        m_Dev_WorkQueue = std::make_shared<lwis_lon::SyncQueue<mlperf::QuerySample>>();

        for (int i = 0; i < numCompleteThreads; i++)
        {
            m_Threads.emplace_back(&Device::Completion, this);
        }
    }

    void AddEngine(EnginePtr_t engine);
    std::map<size_t, std::vector<EnginePtr_t>>& GetEngines()
    {
        return m_Engines;
    }

    size_t GetBatchSize()
    {
        return m_BatchSize;
    }
    void SetBatchSize(size_t batchSize)
    {
        m_BatchSize = batchSize;
    }

    std::string GetName()
    {
        return m_Name;
    }
    size_t GetId()
    {
        return m_Id;
    }

    bool GetSutRecvWithHostMemForRdma()
    {
        return m_SutRecvWithHostMemForRdma;
    }

    bool GetSutSendWithHostMemForRdma()
    {
        return m_SutSendWithHostMemForRdma;
    }

    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
            std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
            callback)
    {
        m_ResponseCallback = callback;
    }

    struct Stats
    {
        std::map<size_t, uint64_t> m_BatchSizeHistogram;
        uint64_t m_MemcpyCalls{0};
        uint64_t m_PerSampleCudaMemcpyCalls{0};
        uint64_t m_BatchedCudaMemcpyCalls{0};

        void reset()
        {
            m_BatchSizeHistogram.clear();
            m_MemcpyCalls = 0;
            m_PerSampleCudaMemcpyCalls = 0;
            m_BatchedCudaMemcpyCalls = 0;
        }
    };

    const Stats& GetStats() const
    {
        return m_Stats;
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::CudaResource>> get_resource_return_queue()
    {
        return m_Dev_ResourceReturnQueue;
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>> get_completion_queue(int qp_idx)
    {
        return m_Dev_CompletionQueue[qp_idx];
    }

    std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>> get_work_queue()
    {
        return m_Dev_WorkQueue;
    }

private:
    void BuildGraphs();

    void Setup();
    void Issue();
    void Done();

    void Completion();

    std::map<size_t, std::vector<EnginePtr_t>> m_Engines;

    size_t const m_Id{0};
    std::string m_Name{""};
    size_t m_BatchSize{0};

    bool const m_EnableSpinWait{false};
    bool const m_EnableDeviceScheduleSpin{false};
    bool const m_DLA{false};
    bool const m_UseSameContext{false};
    bool const m_SutRecvWithHostMemForRdma{false};
    bool const m_SutSendWithHostMemForRdma{false};
    size_t const m_num_QPs{1};

    std::vector<cudaStream_t> m_CopyStreams;
    std::vector<cudaStream_t> m_InferStreams;
    size_t m_InferStreamNum{0};
    std::map<cudaStream_t,
        std::tuple<std::shared_ptr<BufferManager>, cudaEvent_t, cudaEvent_t, cudaEvent_t,
            ::nvinfer1::IExecutionContext*>>
        m_StreamState;

    // Graphs
    typedef std::pair<cudaStream_t, size_t> t_GraphKey;
    std::map<t_GraphKey, cudaGraphExec_t> m_CudaGraphExecs;

    // Completion management
    std::vector<std::thread> m_Threads;

    std::vector<std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>>> m_Dev_CompletionQueue;
    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::CudaResource>> m_Dev_ResourceReturnQueue;
    std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>> m_Dev_WorkQueue;

    // Stream management
    lwis_lon::SyncQueue<cudaStream_t> m_StreamQueue;

    // Query sample response callback.
    std::function<void(::mlperf::QuerySampleResponse* responses, std::vector<::mlperf::QuerySampleIndex>& sample_ids,
        size_t response_count)>
        m_ResponseCallback;

    // Batch management
    lwis_lon::SyncQueue<std::pair<std::deque<mlperf::QuerySample>, cudaStream_t>> m_IssueQueue;

    Stats m_Stats;
};

// captures execution engine for performing inference
class Engine
{
public:
    Engine(::nvinfer1::ICudaEngine* cudaEngine)
        : m_CudaEngine(cudaEngine)
    {
    }

    ::nvinfer1::ICudaEngine* GetCudaEngine() const
    {
        return m_CudaEngine;
    }

private:
    ::nvinfer1::ICudaEngine* m_CudaEngine;
};

// Create buffers and other execution resources.
// Perform queuing, batching, and manage execution resources.
class Server : public mlperf::SystemUnderTest
{
public:
    Server()
        : m_Name("default SUT")
    {
    }
    Server(std::string name)
        : m_Name(name)
    {
    }
    ~Server()
    {
        m_Devices.clear();
        m_Threads.clear();
        m_IssueThreads.clear();
    }

    void Setup(ServerSettings& settings, ServerParams& params);
    void Warmup(double duration);
    void Done();

    std::vector<DevicePtr_t>& GetDevices()
    {
        return m_Devices;
    }

    DevicePtr_t GetDevice(int gpu_idx)
    {
        return m_Devices[gpu_idx];
    }

    // SUT virtual interface
    const std::string& Name() override
    {
        return m_Name;
    }
    void IssueQuery(const std::vector<mlperf::QuerySample>& samples) override;
    void FlushQueries() override;

    void IssueQuery(const std::vector<mlperf::QuerySample>& samples, int const gpu_idx);

    std::shared_ptr<lwis_lon::SyncQueue<mlperf::QuerySample>> get_work_queue(int gpu_idx)
    {
        return m_Devices[gpu_idx]->get_work_queue();
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::Batch>> get_completion_queue(int gpu_idx, int qp_idx)
    {
        return m_Devices[gpu_idx]->get_completion_queue(qp_idx);
    }

    std::shared_ptr<lwis_lon::SyncQueue<lwis_lon::CudaResource>> get_resource_return_queue(int gpu_idx)
    {
        return m_Devices[gpu_idx]->get_resource_return_queue();
    }

    // Set query sample response callback to all the devices.
    void SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
            std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
            callback);

private:
    void ProcessSamples(int DevId);
    void ProcessBatches(int DevId);

    void IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin,
        std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream);

    std::vector<void*> CopySamples(DevicePtr_t device, size_t batchSize,
        std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end,
        cudaStream_t stream);

    DevicePtr_t GetNextAvailableDevice(size_t deviceId);

    void Reset();

    const std::string m_Name;

    std::vector<DevicePtr_t> m_Devices;

    ServerSettings m_ServerSettings;

    // Query management
    std::vector<std::thread> m_Threads;

    // lwis_lon::Batch management
    std::vector<std::thread> m_IssueThreads;
    std::atomic<size_t> m_IssueNum{0};

    // Check if NUMA is used
    bool UseNuma()
    {
        return !m_ServerSettings.m_GpuNumaMap.empty();
    };

    // Get number of NUMA nodes
    int GetNbNumas()
    {
        return m_ServerSettings.m_GpuNumaMap.size();
    };

    // Get NUMA node index of a GPU
    int GetNumaIdxByGpuId(const int deviceId)
    {
        return UseNuma() ? m_ServerSettings.m_GpuNumaMap[deviceId] : 0;
    }

    // Get closest CPUs to a GPU
    std::vector<int> GetClosestCpusToGpu(const int deviceId)
    {
        CHECK(UseNuma());
        return std::get<2>(m_ServerSettings.m_NumaConfig[GetNumaIdxByGpuId(deviceId)]);
    }
};

}; // namespace lwis_lon

#endif
