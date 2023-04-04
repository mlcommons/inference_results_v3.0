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

#include "lwis_lon.hpp"

#include "NvInfer.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "logger.h"
#include <glog/logging.h>

namespace lwis_lon
{
using namespace std::chrono_literals;

std::vector<std::string> split(const std::string& s, char delim)
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

void enqueueShim(nvinfer1::IExecutionContext* context, int batchSize, void** bindings, cudaStream_t stream,
    cudaEvent_t* inputConsumed)
{
    // Assume the first dim is batch dim. Each profile has numBindings bindings.
    auto& engine = context->getEngine();
    if (engine.hasImplicitBatchDimension())
    {
        CHECK_EQ(context->enqueue(batchSize, bindings, stream, inputConsumed), true);
    }
    else
    {
        int profileNum = context->getOptimizationProfile();
        CHECK_EQ(profileNum >= 0 && profileNum < engine.getNbOptimizationProfiles(), true);
        int numBindings = engine.getNbBindings() / engine.getNbOptimizationProfiles();
        for (int i = 0; i < numBindings; i++)
        {
            if (engine.bindingIsInput(i))
            {
                int bindingIdx = numBindings * profileNum + i;
                auto inputDims = context->getBindingDimensions(bindingIdx);
                // Only set binding dimension if batch size changes.
                if (inputDims.d[0] != batchSize)
                {
                    inputDims.d[0] = batchSize;
                    CHECK_EQ(context->setBindingDimensions(bindingIdx, inputDims), true);
                }
            }
        }
        CHECK_EQ(context->allInputDimensionsSpecified(), true);
        CHECK_EQ(context->enqueueV2(bindings, stream, inputConsumed), true);
    }
}

//----------------
// Device
//----------------
void Device::AddEngine(EnginePtr_t engine)
{
    size_t batchSize{0};
    if (engine->GetCudaEngine()->hasImplicitBatchDimension())
    {
        batchSize = engine->GetCudaEngine()->getMaxBatchSize();
    }
    else
    {
        // Assuming the first dimension of the first input is batch dim.
        batchSize = engine->GetCudaEngine()->getProfileDimensions(0, 0, nvinfer1::OptProfileSelector::kMAX).d[0];
    }

    m_Engines[batchSize].emplace_back(engine);
    m_BatchSize = std::min(m_BatchSize, batchSize);
}

void Device::BuildGraphs()
{
    Issue();

    size_t batchSize = 1;
    for (auto& e : m_Engines)
    {
        auto maxBatchSize = e.first;
        auto engine = e.second;

        // build the graph by performing a single execution.  the engines are stored
        // in ascending order of maxBatchSize.  build graphs up to and including
        // this size
        while (batchSize <= maxBatchSize)
        {
            for (auto& streamState : m_StreamState)
            {
                auto& stream = streamState.first;
                auto& state = streamState.second;
                auto& context = std::get<4>(state);

                auto bufferManager = std::get<0>(state);
                auto buffers = &(bufferManager->getDeviceBindings()[0]);

                // need to issue enqueue to TRT to setup resources properly _before_
                // starting graph construction
                enqueueShim(context, batchSize, buffers, m_InferStreams[0], nullptr);

                cudaGraph_t graph;
#if (CUDA_VERSION >= 10010)
                CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0], cudaStreamCaptureModeThreadLocal), CUDA_SUCCESS);
#else
                CHECK_EQ(cudaStreamBeginCapture(m_InferStreams[0]), CUDA_SUCCESS);
#endif
                enqueueShim(context, batchSize, buffers, m_InferStreams[0], nullptr);
                CHECK_EQ(cudaStreamEndCapture(m_InferStreams[0], &graph), CUDA_SUCCESS);

                cudaGraphExec_t graphExec;
                CHECK_EQ(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0), CUDA_SUCCESS);

                t_GraphKey key = std::make_pair(stream, batchSize);
                m_CudaGraphExecs[key] = graphExec;

                CHECK_EQ(cudaGraphDestroy(graph), CUDA_SUCCESS);
            }

            batchSize++;
        }
    }
    gLogInfo << "Capture " << m_CudaGraphExecs.size() << " CUDA graphs" << std::endl;
}

void Device::Setup()
{
    cudaSetDevice(m_Id);
    if (m_EnableDeviceScheduleSpin)
        cudaSetDeviceFlags(cudaDeviceScheduleSpin);

    unsigned int cudaEventFlags
        = (m_EnableSpinWait ? cudaEventDefault : cudaEventBlockingSync) | cudaEventDisableTiming;

    for (auto& stream : m_InferStreams)
    {
        CHECK_EQ(cudaStreamCreate(&stream), CUDA_SUCCESS);
    }

    int profileIdx{0};
    auto engine = m_Engines.rbegin()->second.back()->GetCudaEngine();
    nvinfer1::IExecutionContext* context{nullptr};

    if (m_UseSameContext)
    {
        context = engine->createExecutionContext();
        CHECK_EQ(context->getOptimizationProfile() == 0, true);
        CHECK_EQ(m_InferStreams.size() == 1, true);
        CHECK_EQ(context->allInputDimensionsSpecified(), true);
    }

    for (auto& stream : m_CopyStreams)
    {
        CHECK_EQ(cudaStreamCreate(&stream), CUDA_SUCCESS);

        if (!m_UseSameContext)
        {
            context = engine->createExecutionContext();
            // Set optimization profile if necessary.
            CHECK_EQ(profileIdx < engine->getNbOptimizationProfiles(), true);
            if (context->getOptimizationProfile() < 0)
            {
                CHECK_EQ(context->setOptimizationProfile(profileIdx), true);
            }
        }
        CHECK_EQ(context->getOptimizationProfile() == profileIdx, true);

        std::shared_ptr<nvinfer1::ICudaEngine> emptyPtr{};
        std::shared_ptr<nvinfer1::ICudaEngine> aliasPtr(emptyPtr, engine);
        auto state = std::make_tuple(std::make_shared<BufferManager>(aliasPtr, m_BatchSize, profileIdx), cudaEvent_t(),
            cudaEvent_t(), cudaEvent_t(), context);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<1>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<2>(state), cudaEventFlags), CUDA_SUCCESS);
        CHECK_EQ(cudaEventCreateWithFlags(&std::get<3>(state), cudaEventFlags), CUDA_SUCCESS);

        m_StreamState.insert(std::make_pair(stream, state));

        m_StreamQueue.emplace_back(stream);

        // Engine with implicit batch only has one profile. DLA engine also only has
        // one profile.
        if (!engine->hasImplicitBatchDimension() && engine->getNbOptimizationProfiles() > 1 && !m_UseSameContext)
        {
            ++profileIdx;
        }
    }
}

void Device::Issue()
{
    CHECK_EQ(cudaSetDevice(m_Id), cudaSuccess);
}

void Device::Done()
{
    // join before destroying all members
    for (auto& thread : m_Threads)
    {
        thread.join();
    }

    // destroy member objects
    cudaSetDevice(m_Id);

    for (auto& stream : m_InferStreams)
    {
        cudaStreamDestroy(stream);
    }
    for (auto& stream : m_CopyStreams)
    {
        auto& state = m_StreamState[stream];

        cudaStreamDestroy(stream);
        cudaEventDestroy(std::get<1>(state));
        cudaEventDestroy(std::get<2>(state));
        cudaEventDestroy(std::get<3>(state));
        if (!m_UseSameContext)
        {
            std::get<4>(state)->destroy();
        }
    }
    if (m_UseSameContext)
    {
        std::get<4>(m_StreamState[m_CopyStreams[0]])->destroy();
    }
    for (auto& kv : m_CudaGraphExecs)
    {
        CHECK_EQ(cudaGraphExecDestroy(kv.second), CUDA_SUCCESS);
    }
}

void Device::Completion()
{
    gLogVerbose << "Device::Completion GPU[" << m_Id << "] -- on CPU " << sched_getcpu() << std::endl;

    // Completion handles completion of samples through IB and return of resources
    while (true)
    {
        do
        {
        } while (m_Dev_ResourceReturnQueue->empty());

        auto resource = m_Dev_ResourceReturnQueue->front_then_pop();

        if (resource.Terminate)
        {
            break;
        }

        m_StreamQueue.emplace_back(resource.Stream);
    }
}

//----------------
// Server
//----------------

//! Setup
//!
//! Perform all necessary (untimed) setup in order to perform inference
// including: building
//! graphs and allocating device memory.
void Server::Setup(ServerSettings& settings, ServerParams& params)
{
    initLibNvInferPlugins(&gLogger.getTRTLogger(), "");

    m_ServerSettings = settings;

    // enumerate devices
    std::vector<size_t> devices;
    if (params.DeviceNames == "all")
    {
        int numDevices = 0;
        cudaGetDeviceCount(&numDevices);
        for (int i = 0; i < numDevices; i++)
        {
            devices.emplace_back(i);
        }
    }
    else
    {
        auto deviceNames = split(params.DeviceNames, ',');
        for (auto& n : deviceNames)
            devices.emplace_back(std::stoi(n));
    }

    // check if an engine was specified
    if (!params.EngineNames.size())
        gLogError << "Engine file(s) not specified" << std::endl;

    auto runtime = nvinfer1::createInferRuntime(gLogger.getTRTLogger());

    for (auto& deviceNum : devices)
    {
        cudaSetDevice(deviceNum);

        size_t type = 0;
        for (auto& deviceTypes : params.EngineNames)
        {
            for (auto& batches : deviceTypes)
            {
                if (m_ServerSettings.MaxGPUs != -1 && deviceNum >= static_cast<size_t>(m_ServerSettings.MaxGPUs))
                    continue;

                for (int32_t deviceInstance = 0; deviceInstance < 1; deviceInstance++)
                {
                    size_t numCopyStreams = m_ServerSettings.GPUCopyStreams;
                    size_t numInferStreams = m_ServerSettings.GPUInferStreams;
                    size_t batchSize = m_ServerSettings.GPUBatchSize;

                    if (UseNuma())
                    {
                        bindNumaMemPolicy(GetNumaIdxByGpuId(deviceNum), GetNbNumas());
                    }
                    auto device = std::make_shared<lwis_lon::Device>(deviceNum, numCopyStreams, numInferStreams,
                        m_ServerSettings.CompleteThreads, m_ServerSettings.EnableSpinWait,
                        m_ServerSettings.EnableDeviceScheduleSpin, batchSize, m_ServerSettings.UseSameContext,
                        m_ServerSettings.SutRecvWithHostMemForRdma, m_ServerSettings.SutSendWithHostMemForRdma,
                        m_ServerSettings.NumIBQPsPerNIC);
                    m_Devices.emplace_back(device);

                    if (UseNuma())
                    {
                        resetNumaMemPolicy();
                    }

                    for (auto& engineName : batches)
                    {
                        std::vector<char> trtModelStream;
                        auto size = GetModelStream(trtModelStream, engineName);
                        auto engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);

                        device->AddEngine(std::make_shared<lwis_lon::Engine>(engine));

                        std::ostringstream deviceName;
                        deviceName << "Device:" << deviceNum;

                        device->m_Name = deviceName.str();
                        gLogInfo << device->m_Name << ": " << engineName << " has been successfully loaded."
                                 << std::endl;
                    }
                }
            }

            type++;
        }
    }

    runtime->destroy();

    CHECK(m_Devices.size()) << "No devices or engines available";

    for (auto& device : m_Devices)
        device->Setup();

    if (m_ServerSettings.EnableCudaGraphs)
    {
        gLogInfo << "Start creating CUDA graphs" << std::endl;
        std::vector<std::thread> tmpGraphsThreads;
        for (auto& device : m_Devices)
        {
            tmpGraphsThreads.emplace_back(&Device::BuildGraphs, device.get());
        }
        for (auto& thread : tmpGraphsThreads)
        {
            thread.join();
        }
        gLogInfo << "Finish creating CUDA graphs" << std::endl;
    }

    Reset();

    // create batchers
    auto num_thread_batcher_req = m_ServerSettings.EnableBatcherThreadPerDevice ? m_Devices.size() : 1;
    for (size_t deviceNum = 0; deviceNum < num_thread_batcher_req; deviceNum++)
    {
        auto device_id = m_Devices[deviceNum]->GetId();
        auto numa_node = GetNumaIdxByGpuId(device_id);
        auto num_numa = GetNbNumas();

        if (UseNuma() && m_ServerSettings.EnableBatcherThreadPerDevice)
        {
            bindNumaMemPolicy(numa_node, num_numa);
        }
        gLogInfo << "Creating batcher thread: " << deviceNum << " EnableBatcherThreadPerDevice: "
                 << (m_ServerSettings.EnableBatcherThreadPerDevice ? "true" : "false") << std::endl;
        m_Threads.emplace_back(std::thread(&Server::ProcessSamples, this, m_Devices[deviceNum]->GetId()));
        if (UseNuma() && m_ServerSettings.EnableBatcherThreadPerDevice)
        {
            // manual control of CPU to map this thread
            auto cpus = GetClosestCpusToGpu(device_id);
            std::vector<int> cpu_to_bind;
            if (cpus.size() > num_thread_batcher_req)
            {
                cpu_to_bind.emplace_back(cpus[cpus.size() - 1 - deviceNum]);
            }
            else
            {
                cpu_to_bind = cpus;
            }
            bindThreadToCpus(m_Threads.back(), cpu_to_bind);
            resetNumaMemPolicy();
        }
    }

    // create issue threads
    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        auto num_dev = m_Devices.size();
        for (size_t deviceNum = 0; deviceNum < num_dev; deviceNum++)
        {
            auto device_id = m_Devices[deviceNum]->GetId();
            auto numa_node = GetNumaIdxByGpuId(device_id);
            auto num_numa = GetNbNumas();

            if (UseNuma())
            {
                bindNumaMemPolicy(numa_node, num_numa);
            }
            gLogInfo << "Creating cuda thread: " << deviceNum << std::endl;
            m_IssueThreads.emplace_back(std::thread(&Server::ProcessBatches, this, m_Devices[deviceNum]->GetId()));
            if (UseNuma())
            {
                // manual control of CPU to map this thread
                auto cpus = GetClosestCpusToGpu(device_id);
                std::vector<int> cpu_to_bind;
                if (cpus.size() > deviceNum + num_thread_batcher_req)
                {
                    cpu_to_bind.emplace_back(cpus[cpus.size() - 1 - deviceNum - num_thread_batcher_req]);
                }
                else
                {
                    cpu_to_bind = cpus;
                }
                bindThreadToCpus(m_IssueThreads.back(), cpu_to_bind);
                resetNumaMemPolicy();
            }
        }
    }

    // If NUMA is used, make sure that the NUMA config makes sense.
    if (UseNuma())
    {
        CHECK(m_Devices.size() == m_ServerSettings.m_GpuNumaMap.size()) << "NUMA config does not match number of GPUs";
    }
}

void Server::Done()
{
    // send terminate signal
    for (auto& device : m_Devices)
    {
        for (size_t i = 0; i < m_ServerSettings.CompleteThreads; i++)
        {
            device->m_Dev_ResourceReturnQueue->push_back(lwis_lon::CudaResource{0, 0, 0, true});
        }

        while (!device->get_work_queue()->empty())
        {
        }
        device->get_work_queue()->emplace_back(mlperf::QuerySample{UINT64_MAX, UINT32_MAX});
    }

    for (auto& device : m_Devices)
    {
        device->Done();
    }

    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (auto& device : m_Devices)
        {
            std::deque<mlperf::QuerySample> batch;
            auto pair = std::make_pair(std::move(batch), nullptr);
            device->m_IssueQueue.emplace_back(pair);
        }
        for (auto& thread : m_IssueThreads)
            thread.join();
    }

    // join after we insert the dummy sample
    for (auto& thread : m_Threads)
        thread.join();
}

void Server::IssueQuery(std::vector<mlperf::QuerySample> const& samples, int const gpu_idx)
{
    TIMER_START(IssueQuery);
    get_work_queue(gpu_idx)->insert(samples);
    TIMER_END(IssueQuery);
}

DevicePtr_t Server::GetNextAvailableDevice(size_t deviceId)
{
    DevicePtr_t device;

    device = m_Devices[deviceId];
    while (device->m_StreamQueue.empty())
    {
    }

    return device;
}

void Server::IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin,
    std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream)
{
    auto enqueueBatchSize = batchSize;
    auto inferStream
        = (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : device->m_InferStreams[device->m_InferStreamNum];

    auto& state = device->m_StreamState[copyStream];
    auto [bufferManager, htod, inf, dtoh, context] = state;

    // setup Device
    device->Issue();

#ifndef SUT_DEBUG_DISABLE_INFERENCE
    // perform copy to device buffer
    std::vector<void*> buffers = bufferManager->getDeviceBindings();
    buffers = CopySamples(device, batchSize, begin, end, copyStream);
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
    }

#ifndef SUT_DEBUG_DISABLE_COMPUTE
    // perform inference
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
    }
    Device::t_GraphKey key = std::make_pair(copyStream, enqueueBatchSize);
    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
    if (g_it != device->m_CudaGraphExecs.end())
    {
        TIMER_START(cudaGraphLaunch);
        CHECK_EQ(cudaGraphLaunch(g_it->second, inferStream), CUDA_SUCCESS);
        TIMER_END(cudaGraphLaunch);
    }
    else
    {
        TIMER_START(enqueueShim);
        enqueueShim(context, enqueueBatchSize, &buffers[0], inferStream, nullptr);
        TIMER_END(enqueueShim);
    }
#endif
    // sync streams
    if (!m_ServerSettings.RunInferOnCopyStreams)
    {
        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    }
#endif

    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);

    // optional synchronization
    if (m_ServerSettings.EnableSyncOnEvent)
    {
        cudaEventSynchronize(dtoh);
    }

    // generate asynchronous response
    if (m_ServerSettings.EnableResponse)
    {
        // Assuming output binding is the last binding of the current profile.
        auto engine = device->m_Engines.rbegin()->second.back()->GetCudaEngine();
        int bindingIdx = engine->getNbBindings() / engine->getNbOptimizationProfiles() - 1;
        size_t respSize = volume(engine->getBindingDimensions(bindingIdx), engine->hasImplicitBatchDimension())
            * getElementSize(engine->getBindingDataType(bindingIdx));
        auto buffer = static_cast<int8_t*>(bufferManager->getDeviceBuffer(bindingIdx));

        TIMER_START(ServerIssueBatchCompletionQueuePushBack);
        std::vector<lwis_lon::Batch> batches;
        for (int i = 0; i < m_ServerSettings.NumIBQPsPerNIC; ++i)
        {
            lwis_lon::Batch batch;
            batch.Event = dtoh;
            batch.Stream = copyStream;
            batch.ResId = device->GetId();
            batch.QueuePairId = i;
            batches.emplace_back(batch);
        }
        for (auto it = begin; it != end; ++it)
        {
            uint32_t qp_id = static_cast<uint32_t>(it->id >> 32);
            uint32_t resp_id = static_cast<uint32_t>(it->id);
            batches[qp_id].Responses.emplace_back(mlperf::QuerySampleResponse{resp_id, (uintptr_t) buffer, respSize});
            buffer += respSize;
        }

        for (int i = 0; i < m_ServerSettings.NumIBQPsPerNIC; ++i)
        {
            if (!batches[i].Responses.empty())
            {
                device->get_completion_queue(i)->emplace_back(batches[i]);
            }
        }
        TIMER_END(ServerIssueBatchCompletionQueuePushBack);
    }

    // Simple round-robin across inference streams.  These don't need to be
    // managed like copy
    // streams since they are not tied with a resource that is re-used and not
    // managed by hardware.
    device->m_InferStreamNum = (device->m_InferStreamNum + 1) % device->m_InferStreams.size();
    device->m_Stats.m_BatchSizeHistogram[batchSize]++;
}

std::vector<void*> Server::CopySamples(DevicePtr_t device, std::size_t batchSize,
    std::deque<mlperf::QuerySample>::iterator begin, std::deque<mlperf::QuerySample>::iterator end, cudaStream_t stream)
{
    auto bufferManager = std::get<0>(device->m_StreamState[stream]);
    auto deviceId = device->GetId();

    cudaMemcpyKind const cpkind
        = device->GetSutRecvWithHostMemForRdma() ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;

    // setup default device buffers
    std::vector<void*> buffers = bufferManager->getDeviceBindings();

    // Currently assumes that input bindings always come before output bindings.
    auto engine = device->m_Engines.rbegin()->second.front()->GetCudaEngine();
    size_t num_inputs{0};
    int num_bindinges_per_profile{engine->getNbBindings() / engine->getNbOptimizationProfiles()};

    for (int i = 0; i < num_bindinges_per_profile; i++)
    {
        if (engine->bindingIsInput(i))
        {
            ++num_inputs;
        }
    }

    // no sample library.  copy to device memory if necessary
    for (size_t i = 0; i < num_inputs; i++)
    {
        size_t sampleSize
            = volume(engine->getBindingDimensions(i), engine->getBindingFormat(i), engine->hasImplicitBatchDimension())
            * getElementSize(engine->getBindingDataType(i));
        auto dev_buf = static_cast<int8_t*>(bufferManager->getDeviceBuffer(i));

        // copy samples in the batch
        void* copy_src_addr{nullptr};
        void* copy_dst_addr{nullptr};
        std::size_t copy_size = 0;
        for (auto it = begin; it != end; ++it)
        {
            auto final = (std::distance(it, end) == 1);
            auto index = std::distance(begin, it);

            if (index == 0)
            {
                copy_src_addr = reinterpret_cast<void*>(it->index);
                copy_dst_addr = dev_buf;
            }

            if (reinterpret_cast<std::size_t>(copy_src_addr) + copy_size == it->index)
            {
                copy_size += sampleSize;
            }
            else
            {
                TIMER_START(ServerCopySamplesCudaMemcpyAsync);
                CHECK_EQ(cudaMemcpyAsync(copy_dst_addr, copy_src_addr, copy_size, cpkind, stream), cudaSuccess);
                TIMER_END(ServerCopySamplesCudaMemcpyAsync);
                if (copy_size == sampleSize)
                {
                    device->m_Stats.m_PerSampleCudaMemcpyCalls++;
                }
                else
                {
                    device->m_Stats.m_BatchedCudaMemcpyCalls++;
                }
                copy_src_addr = reinterpret_cast<void*>(it->index);
                copy_dst_addr = dev_buf + copy_size;
                copy_size = sampleSize;
            }

            if (final)
            {
                TIMER_START(ServerCopySamplesCudaMemcpyAsync);
                CHECK_EQ(cudaMemcpyAsync(copy_dst_addr, copy_src_addr, copy_size, cpkind, stream), cudaSuccess);
                TIMER_END(ServerCopySamplesCudaMemcpyAsync);
                if (copy_size == sampleSize)
                {
                    device->m_Stats.m_PerSampleCudaMemcpyCalls++;
                }
                else
                {
                    device->m_Stats.m_BatchedCudaMemcpyCalls++;
                }
            }
        }
    }

    return buffers;
}

void Server::Reset()
{
    for (auto& device : m_Devices)
    {
        device->m_InferStreamNum = 0;
        device->m_Stats.reset();
    }
}

void Server::ProcessSamples(int devId)
{
    gLogVerbose << "Server::ProcessSamples GPU[" << devId << "] -- on CPU " << sched_getcpu() << std::endl;

    // initial device available
    auto device = GetNextAvailableDevice(devId);

    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        TIMER_START(m_WorkQueue_acquire_total);
        do
        {
            get_work_queue(devId)->acquire(
                samples, m_ServerSettings.Timeout, device->m_BatchSize, m_ServerSettings.EnableDequeLimit);
        } while (samples.empty());
        TIMER_END(m_WorkQueue_acquire_total);

        auto begin = samples.begin();
        auto end = samples.end();

        // Unusable numbers mean end of samples
        if (begin->id == UINT64_MAX || begin->index == UINT32_MAX)
        {
            break;
        }

        auto batch_begin = begin;

        // build batches up to maximum supported batchSize
        while (batch_begin != end)
        {
            auto batchSize = std::min(device->m_BatchSize, static_cast<size_t>(std::distance(batch_begin, end)));
            auto batch_end = batch_begin + batchSize;

            // Acquire resources
            TIMER_START(m_StreamQueue_pop_front);
            auto copyStream = device->m_StreamQueue.front();
            device->m_StreamQueue.pop_front();
            TIMER_END(m_StreamQueue_pop_front);

            // Issue this batch
            if (!m_ServerSettings.EnableCudaThreadPerDevice)
            {
                // issue on this thread
                TIMER_START(IssueBatch);
                IssueBatch(device, batchSize, batch_begin, batch_end, copyStream);
                TIMER_END(IssueBatch);
            }
            else
            {
                // issue on device specific thread
                std::deque<mlperf::QuerySample> batch(batch_begin, batch_end);
                auto pair = std::make_pair(std::move(batch), copyStream);
                device->m_IssueQueue.emplace_back(pair);
            }

            // Advance to next batch
            batch_begin = batch_end;

            // Get available device for next batch
            TIMER_START(GetNextAvailableDevice);
            device = GetNextAvailableDevice(devId);
            TIMER_END(GetNextAvailableDevice);
        }
    }
}

void Server::ProcessBatches(int deviceId)
{
    gLogVerbose << "Server::ProcessBatches GPU[" << deviceId << "] -- on CPU " << sched_getcpu() << std::endl;

    auto& device = m_Devices[deviceId];
    auto& issueQueue = device->m_IssueQueue;

    while (true)
    {
        auto pair = issueQueue.front();
        issueQueue.pop_front();

        auto [batch, stream] = pair;

        if (batch.empty() && stream == nullptr)
        {
            m_IssueNum--;
            break;
        }

        IssueBatch(device, batch.size(), batch.begin(), batch.end(), stream);
    }
}

void Server::Warmup(double duration)
{
    double elapsed = 0.0;
    auto tStart = std::chrono::high_resolution_clock::now();

    do
    {
        for (size_t deviceIndex = 0; deviceIndex < m_Devices.size(); ++deviceIndex)
        {
            // get next device to send batch to
            auto device = m_Devices[deviceIndex];

            for (auto copyStream : device->m_CopyStreams)
            {
                for (auto inferStream : device->m_InferStreams)
                {
                    auto& state = device->m_StreamState[copyStream];
                    auto [bufferManager, htod, inf, dtoh, context] = state;

                    device->Issue();
                    auto engine = device->m_Engines.rbegin()->second.front()->GetCudaEngine();

                    if (m_ServerSettings.EnableDma && !m_ServerSettings.EnableDmaStaging)
                    {
                        for (auto i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles(); i++)
                        {
                            if (engine->bindingIsInput(i))
                            {
                                bufferManager->copyInputToDeviceAsync(i, copyStream);
                            }
                        }
                    }

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                    {
                        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);
                        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);
                    }

                    Device::t_GraphKey key = std::make_pair(copyStream, device->m_BatchSize);
                    auto g_it = device->m_CudaGraphExecs.lower_bound(key);
                    if (g_it != device->m_CudaGraphExecs.end())
                    {
                        CHECK_EQ(cudaGraphLaunch(
                                     g_it->second, (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream),
                            CUDA_SUCCESS);
                    }
                    else
                    {
                        auto buffers = bufferManager->getDeviceBindings();
                        enqueueShim(context, device->m_BatchSize, &buffers[0],
                            (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream, nullptr);
                    }
                    if (!m_ServerSettings.RunInferOnCopyStreams)
                        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
                    if (m_ServerSettings.EnableDma && !m_ServerSettings.EnableDmaStaging)
                    {
                        for (auto i = 0; i < engine->getNbBindings() / engine->getNbOptimizationProfiles(); i++)
                        {
                            if (!engine->bindingIsInput(i))
                            {
                                bufferManager->copyOutputToHostAsync(i, copyStream);
                            }
                        }
                    }
                    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);
                }
            }
        }
        elapsed = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);

    for (auto& device : m_Devices)
    {
        device->Issue();
        cudaDeviceSynchronize();
    }

    // reset server state
    Reset();
}

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    // don't use
    CHECK(false);
}

void Server::FlushQueries()
{
    // This function is called at the end of a series of IssueQuery calls
    // (typically the end of a
    // region of queries that define a performance or accuracy test).  Its purpose
    // is to allow a
    // SUT to force all remaining queued samples out to avoid implementing
    // timeouts.

    // Currently, there is no use case for it in this IS.
}

void Server::SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
        std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
        callback)
{
    // don't use it for now
    CHECK(false);
    std::for_each(
        m_Devices.begin(), m_Devices.end(), [callback](DevicePtr_t device) { device->SetResponseCallback(callback); });
}

}; // namespace lwis_lon
