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

#include <algorithm>
#include <fstream>
#include <stdexcept>

#include "lwis_3dunet.hpp"

namespace lwis
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
        if (m_VerboseNVTX)
        {
            context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
        }
        else
        {
            context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
        }
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
            if (m_VerboseNVTX)
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
            }
            else
            {
                context->setNvtxVerbosity(nvinfer1::ProfilingVerbosity::kNONE);
            }
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

        // Engine with implicit batch only has one profile. DLA engine also only has one profile.
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
}

void Device::Completion()
{
    // Testing for completion needs to be based on the main thread finishing submission and
    // providing events for the completion thread to wait on.  The resources exist as part of the
    // Device class.
    //
    // Samples and responses are assumed to be contiguous and have the same sizes across samples.

    // Flow:
    // Main thread
    // - Find Device (check may be based on data buffer availability)
    // - Enqueue work
    // - Enqueue CompletionQueue batch
    // ...
    // - Enqueue CompletionQueue null batch

    // Completion thread(s)
    // - Wait for entry
    // - Wait for queue head to have data ready (wait for event)
    // - Dequeue CompletionQueue

    while (true)
    {
        // TODO: with multiple CudaStream inference it may be beneficial to handle these out of
        // order
        auto batch = m_CompletionQueue.front_then_pop();

        if (batch.Responses.empty())
            break;

        // wait on event completion
        CHECK_EQ(cudaEventSynchronize(batch.Event), cudaSuccess);

        // callback if it exists
        if (m_ResponseCallback)
        {
            CHECK(batch.SampleIds.size() == batch.Responses.size()) << "missing sample IDs";
            m_ResponseCallback(&batch.Responses[0], batch.SampleIds, batch.Responses.size());
        }

        // assume this function is reentrant for multiple devices
        TIMER_START(QuerySamplesComplete);
        mlperf::QuerySamplesComplete(
            &batch.Responses[0], batch.Responses.size(), batch.ResponseCb.value_or(mlperf::ResponseCallback{}));
        TIMER_END(QuerySamplesComplete);

        m_StreamQueue.emplace_back(batch.Stream);
    }
}

//----------------
// Server
//----------------

// Setup
//
// Perform all necessary (untimed) setup in order to perform inference including: building
// graphs and allocating device memory.
//
// Setup also takes care of populating preconditioned Gaussian patches for KiTS19
//
// NOTE: CUDA Graph & ForceContiguous means nothing for 3D UNet KiTS19 and hence removed
void Server::Setup(ServerSettings_3DUNet& settings, ServerParams& params)
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
                auto isDlaDevice = type == 1;

                if (!isDlaDevice && m_ServerSettings.MaxGPUs != -1
                    && deviceNum >= static_cast<size_t>(m_ServerSettings.MaxGPUs))
                    continue;

                for (int32_t deviceInstance = 0; deviceInstance < (isDlaDevice ? runtime->getNbDLACores() : 1);
                     deviceInstance++)
                {
                    if (isDlaDevice && m_ServerSettings.MaxDLAs != -1 && deviceInstance >= m_ServerSettings.MaxDLAs)
                        continue;

                    size_t numCopyStreams
                        = isDlaDevice ? m_ServerSettings.DLACopyStreams : m_ServerSettings.GPUCopyStreams;
                    size_t numInferStreams
                        = isDlaDevice ? m_ServerSettings.DLAInferStreams : m_ServerSettings.GPUInferStreams;

                    // this batchSize is  for SW batch size
                    size_t batchSize = isDlaDevice ? m_ServerSettings.DLABatchSize : m_ServerSettings.GPUBatchSize;

                    if (UseNuma())
                    {
                        bindNumaMemPolicy(GetNumaIdx(deviceNum), GetNbNumas());
                    }
                    auto device = std::make_shared<lwis::Device>(deviceNum, numCopyStreams, numInferStreams,
                        m_ServerSettings.CompleteThreads, m_ServerSettings.EnableSpinWait,
                        m_ServerSettings.EnableDeviceScheduleSpin, batchSize, isDlaDevice,
                        m_ServerSettings.UseSameContext, m_ServerSettings.SliceOverlapKernelCGImpl,
                        m_ServerSettings.VerboseNVTX);
                    m_Devices.emplace_back(device);
                    if (UseNuma())
                    {
                        resetNumaMemPolicy();
                    }

                    for (auto& engineName : batches)
                    {
                        std::vector<char> trtModelStream;
                        auto size = GetModelStream(trtModelStream, engineName);
                        if (isDlaDevice)
                            runtime->setDLACore(deviceInstance);
                        auto engine = runtime->deserializeCudaEngine(trtModelStream.data(), size, nullptr);

                        device->AddEngine(std::make_shared<lwis::Engine>(engine));

                        std::ostringstream deviceName;
                        deviceName << "Device:" << deviceNum;
                        if (isDlaDevice)
                        {
                            deviceName << ".DLA-" << deviceInstance;
                        }

                        device->m_Name = deviceName.str();
                        gLogInfo << device->m_Name << ": " << engineName << " has been successfully loaded."
                                 << std::endl;
                    }
                }
            }

            type++;
        }
    }

    // load Gaussian patches to memory
    CHECK(m_ServerSettings.SW_dhw == 128) << "3DUNet KiTS19 only supports 128x128x128 sliding window";
    CHECK(m_ServerSettings.SW_overlap_pct == 50) << "3DUNet KiTS19 only supports 50% overlap in SW";

    // set sliding window stride
    m_Stride = m_ServerSettings.SW_dhw * m_ServerSettings.SW_overlap_pct / 100;

    // buffer manager holding KiTS19 sample input/output, as well as buffers for the sliding window
    // inference copy each patch to dev buf and bookkeep the buffers
    auto numNUMADomains = UseNuma() ? GetNbNumas() : 1;
    m_kits19_buffer_manager.resize(m_Devices.size());
    for (auto& device : m_Devices)
    {
        int deviceNum = device->GetId();
        auto deviceNumNUMA = GetNumaIdx(deviceNum);
        auto sampleLibrary = m_SampleLibraries[deviceNumNUMA];
        const bool directHostAccess = ((device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess)
            || m_ServerSettings.EnableDirectHostAccess);
        const int numaIdx = UseNuma() ? deviceNumNUMA : -1;

        // change to desired batch size
        auto batchSize = device->GetBatchSize();

        // NOTE: m_kits19_buffer_manager is in device's ID order, NOT in device's NUMA Index order
        m_kits19_buffer_manager[deviceNum] = std::make_shared<KiTS19BufferManager>(m_ServerSettings.SW_dhw,
            device->m_InferStreams.size(), sampleLibrary->GetMaxSampleSize(), m_ServerSettings.SW_gaussian_patch_path,
            directHostAccess, deviceNum, numaIdx, numNUMADomains, batchSize);
    }

    runtime->destroy();

    CHECK(m_Devices.size()) << "No devices or engines available";

    for (auto& device : m_Devices)
        device->Setup();

    Reset();

    // create batchers
    for (size_t deviceNum = 0; deviceNum < (m_ServerSettings.EnableBatcherThreadPerDevice ? m_Devices.size() : 1);
         deviceNum++)
    {
        gLogInfo << "Creating batcher thread: " << deviceNum << " EnableBatcherThreadPerDevice: "
                 << (m_ServerSettings.EnableBatcherThreadPerDevice ? "true" : "false") << std::endl;
        m_Threads.emplace_back(std::thread(&Server::ProcessSamples, this));
        if (UseNuma() && m_ServerSettings.EnableBatcherThreadPerDevice)
        {
            bindThreadToCpus(m_Threads.back(), GetClosestCpus(m_Devices[deviceNum]->GetId()));
        }
    }

    // create issue threads
    if (m_ServerSettings.EnableCudaThreadPerDevice)
    {
        for (size_t deviceNum = 0; deviceNum < m_Devices.size(); deviceNum++)
        {
            gLogInfo << "Creating cuda thread: " << deviceNum << std::endl;
            m_IssueThreads.emplace_back(std::thread(&Server::ProcessBatches, this));
            if (UseNuma())
            {
                bindThreadToCpus(m_IssueThreads.back(), GetClosestCpus(m_Devices[deviceNum]->GetId()));
            }
        }
    }

    // If NUMA is used, make sure that the NUMA config makes sense.
    if (UseNuma())
    {
        CHECK(m_Devices.size() == m_ServerSettings.m_GpuToNumaMap.size())
            << "NUMA config does not match number of GPUs";
        CHECK(m_SampleLibraries.size() == m_ServerSettings.m_NumaConfig.size())
            << "Number of QSLs does not match NUMA config";
    }
}

void Server::Done()
{
    // send dummy batch to signal completion
    for (auto& device : m_Devices)
    {
        for (size_t i = 0; i < m_ServerSettings.CompleteThreads; i++)
        {
            device->m_CompletionQueue.push_back(Batch{});
        }
    }
    for (auto& device : m_Devices)
        device->Done();

    // send end sample to signal completion
    while (!m_WorkQueue.empty())
    {
    }

    while (m_DeviceNum)
    {
        size_t currentDeviceId = m_DeviceNum;
        m_WorkQueue.emplace_back(mlperf::QuerySample{0, 0});
        while (currentDeviceId == m_DeviceNum)
        {
        }
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

void Server::IssueQuery(const std::vector<mlperf::QuerySample>& samples)
{
    TIMER_START(IssueQuery);
    m_WorkQueue.insert(samples);
    TIMER_END(IssueQuery);
}

DevicePtr_t Server::GetNextAvailableDevice(size_t deviceId)
{
    DevicePtr_t device;
    if (!m_ServerSettings.EnableBatcherThreadPerDevice)
    {
        do
        {
            device = m_Devices[m_DeviceIndex];
            m_DeviceIndex = (m_DeviceIndex + 1) % m_Devices.size();
        } while (device->m_StreamQueue.empty());
    }
    else
    {
        device = m_Devices[deviceId];
        while (device->m_StreamQueue.empty())
        {
        }
    }

    return device;
}

void Server::CopyKiTS19Sample(const DevicePtr_t device, const std::shared_ptr<KiTS19BufferManager>& bufMgr,
    std::deque<mlperf::QuerySample>::iterator iter, const cudaStream_t stream, const bool directHostAccess)
{
    // 3DUNet KiTS19 cannot exploit contiguous input tensors
    // 3DUNet KiTS19 samples are in different shapes; need to return the shape info
    // This handles only one KiTS19 sample, and uses iter iterator pointing it
    // Unified memory: copy into the device buffer

    auto deviceId = device->GetId();
    auto sampleLibrary = m_SampleLibraries[GetNumaIdx(deviceId)];

    CHECK(sampleLibrary != nullptr) << "Need QSL KiTS19 properly instantiated";

    TIMER_START(host_to_device_copy);
    // only one input for KiTS19, so fixing input index to 0
    if (m_ServerSettings.EnableStartFromDeviceMem)
    {
        // copy from device buffer to staging device buffer
        CHECK_EQ(cudaMemcpyAsync(static_cast<int8_t*>(bufMgr->getSampleInputDeviceBuffer()),
                     sampleLibrary->GetSampleAddress(iter->index, 0, deviceId),
                     sampleLibrary->GetSampleSize(iter->index, 0), cudaMemcpyDeviceToDevice, stream),
            cudaSuccess);
        device->m_Stats.m_PerSampleCudaMemcpyCalls++;
    }
    else if (directHostAccess)
    {
        // copy to the host staging buffer which is used as device buffer
        std::memcpy(static_cast<int8_t*>(bufMgr->getSampleInputHostBuffer()),
            sampleLibrary->GetSampleAddress(iter->index, 0), sampleLibrary->GetSampleSize(iter->index, 0));
        device->m_Stats.m_MemcpyCalls++;
    }
    else
    {
        // copy direct to device buffer
        bufMgr->copyInputToDeviceAsync(
            stream, sampleLibrary->GetSampleAddress(iter->index, 0), sampleLibrary->GetSampleSize(iter->index, 0));
        device->m_Stats.m_PerSampleCudaMemcpyCalls++;
    }
    TIMER_END(host_to_device_copy);
}

void Server::IssueBatch(DevicePtr_t device, size_t batchSize, std::deque<mlperf::QuerySample>::iterator begin,
    std::deque<mlperf::QuerySample>::iterator end, cudaStream_t copyStream)
{
    CHECK_EQ(batchSize, 1) << "3D-UNet KiTS19 can only handle 1 sample at a time";

    const auto enqueueBatchSize = device->GetBatchSize();
    const auto patchKernelCGImpl = device->UsePatchKernelCGImpl();

    auto& state = device->m_StreamState[copyStream];
    // FIXME: Maybe remove this as it is used in warmup only
    //        Or reuse this for Sliding Window (currently it is not due to possible
    //        multi-inferStream)
    auto SWBufMgr = std::get<0>(state);
    // FIXME: need to figure how to use event(s) for multi-inferStream with overlap
    auto& htod = std::get<1>(state);
    auto& inf = std::get<2>(state);
    auto& dtoh = std::get<3>(state);
    auto& context = std::get<4>(state);

    auto deviceId = device->GetId();
    auto deviceIdNUMA = GetNumaIdx(deviceId);
    auto sampleLibrary = m_SampleLibraries[deviceIdNUMA];
    auto KiTS19bufMgr = m_kits19_buffer_manager[deviceId];

    // setup Device
    device->Issue();

    // perform copy to device
    // KiTS19 should be only one input, so grabbing one from first input index
    qsl::kits19_dim sample_dims = sampleLibrary->GetSampleDimension(begin->index, 0);

    const bool directHostAccess
        = ((device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess) || m_ServerSettings.EnableDirectHostAccess);

    CHECK_EQ(directHostAccess, KiTS19bufMgr->isDirectHostAccessEnabled());

    // setup buffers for this sample
    TIMER_START(SetupBuffers);
    KiTS19bufMgr->setupBuffers(sample_dims.d, sample_dims.h, sample_dims.w, copyStream, true);
    TIMER_END(SetupBuffers);

    TIMER_START(CopySamples);
    CopyKiTS19Sample(device, KiTS19bufMgr, begin, copyStream, directHostAccess);
    TIMER_END(CopySamples);
    if (!m_ServerSettings.RunInferOnCopyStreams)
        CHECK_EQ(cudaEventRecord(htod, copyStream), CUDA_SUCCESS);

    auto sample_in_buf = KiTS19bufMgr->getSampleInputBuffer();
    auto sample_out_buf = KiTS19bufMgr->getSampleOutputBuffer();
    auto sample_resp_buf = KiTS19bufMgr->getSampleResponseBuffer();
    // for input & output hence size 2 array
    std::vector<std::array<void*, 2>> sw_buffers;
    sw_buffers.resize(device->m_InferStreams.size());

    // perform inference on the asked KiTS19 sample
    // need to loop on, as many as needed, belows:
    // 1) slice and populate input buf
    // 2) point bindings to input/output buf
    // 3) get output and patch it, then accumulate it to final output buffer
    // when loop exits:
    // 4) do argmax before acknowledges sample done and initiates D2H copy
    // =======================================================================
    // loop starts
    // generate offsets, set UNet3DParams and feed kernels/TRT
    UNet3DParams u3p;
    u3p.image_d = sample_dims.d;
    u3p.image_h = sample_dims.h;
    u3p.image_w = sample_dims.w;
    u3p.image_size = u3p.image_d * u3p.image_h * u3p.image_w;

    auto inferStream
        = (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : device->m_InferStreams[device->m_InferStreamNum];

    // make sure buffers are ready
    if (!m_ServerSettings.RunInferOnCopyStreams)
        CHECK_EQ(cudaStreamWaitEvent(inferStream, htod, 0), CUDA_SUCCESS);

    // loop -- sliding window
    // NOTE: stride is guaranteed to be smaller than SW_dhw, i.e. sliding window d/h/w
    //       d/h/w are guaranteed to be divisible by stride

    // some control info
    TIMER_START(Prepare_SlidingWindow)
    int patch_idx = 0;
    int total_itr_left = ((sample_dims.d - m_Stride - 1) / m_Stride + 1)
        * ((sample_dims.h - m_Stride - 1) / m_Stride + 1) * ((sample_dims.w - m_Stride - 1) / m_Stride + 1);

    // return relative position of the slice
    auto get_slice_relative_position = [](int position, int total_length, int stride) {
        return static_cast<int>(position == 0 ? SliceRelativePosition::StartingCorner
                                              : (position >= total_length - stride ? SliceRelativePosition::EndingCorner
                                                                                   : SliceRelativePosition::Middle));
    };

    // return index to fetch a correct Gaussian patch from an array where
    // patches are contiguously packed
    auto get_gaussian_patch_index = [](int d_rel_pos, int h_rel_pos, int w_rel_pos)
    {
        return d_rel_pos * 9 + h_rel_pos * 3 + w_rel_pos;
    };

    int total_packed = 0;
    for (int d = 0; d < sample_dims.d - m_Stride; d += m_Stride)
    {
        int dd = get_slice_relative_position(d, sample_dims.d, m_ServerSettings.SW_dhw);
        for (int h = 0; h < sample_dims.h - m_Stride; h += m_Stride)
        {
            int hh = get_slice_relative_position(h, sample_dims.h, m_ServerSettings.SW_dhw);
            for (int w = 0; w < sample_dims.w - m_Stride; w += m_Stride)
            {
                int ww = get_slice_relative_position(w, sample_dims.w, m_ServerSettings.SW_dhw);
                patch_idx = get_gaussian_patch_index(dd, hh, ww);

                // have d, h, w, patch_idx all here
                device->m_Iterations.emplace_back(d, h, w, patch_idx);
                total_packed++;
            }
        }
    }

    CHECK_EQ(total_itr_left, total_packed) << "Iteration count mismatch";
    TIMER_END(Prepare_SlidingWindow)

    int total_enq = 0;
    // real sliding window inference happens here, using the iteration vector built above
    while (!device->m_Iterations.empty())
    {
        int slice_cnt = 0;
        while (slice_cnt < enqueueBatchSize && !device->m_Iterations.empty())
        {
            const auto [d, h, w, patch_idx] = device->m_Iterations.front();
            u3p.slice_to_off_d[slice_cnt] = d;
            u3p.slice_to_off_h[slice_cnt] = h;
            u3p.slice_to_off_w[slice_cnt] = w;
            u3p.patches[slice_cnt] = KiTS19bufMgr->getGaussianPatchBuffer(patch_idx);

            slice_cnt += 1;
            device->m_Iterations.pop_front();
        }
        CHECK(slice_cnt > 0) << "Collecting batch failed";
        u3p.actual_num_slices = slice_cnt;

        inferStream = (m_ServerSettings.RunInferOnCopyStreams)
            ? copyStream
            : device->m_InferStreams[device->m_InferStreamNum];

        sw_buffers[device->m_InferStreamNum][0] = KiTS19bufMgr->getSWInputBuffer(device->m_InferStreamNum);
        sw_buffers[device->m_InferStreamNum][1] = KiTS19bufMgr->getSWOutputBuffer(device->m_InferStreamNum);

        // slicing
        TIMER_START(UNET3DSW_sliceKernel);
        UNet3DKiTS19SliceKernelI8Linear_wrapper(sample_in_buf, sw_buffers[device->m_InferStreamNum][0], u3p, inferStream, deviceId);
        TIMER_END(UNET3DSW_sliceKernel);

        // Network run on TRT
        TIMER_START(enqueueShim);
        enqueueShim(
            context, u3p.actual_num_slices, &sw_buffers[device->m_InferStreamNum][0], inferStream, nullptr);
        TIMER_END(enqueueShim);

        // Gaussian patching & accumulation
        TIMER_START(UNET3DSW_patchKernel);
        UNet3DKiTS19PatchKernel_wrapper(sw_buffers[device->m_InferStreamNum][1], sample_out_buf, u3p, inferStream,
            deviceId, patchKernelCGImpl);
        TIMER_END(UNET3DSW_patchKernel);

        // stat handling
        TIMER_START(CollectSWBatchStat)
        device->m_Stats.m_BatchSizeHistogram[u3p.actual_num_slices]++;
        TIMER_END(CollectSWBatchStat)

        // Simple round-robin across inference streams.  These don't need to be managed like
        // copy streams since they are not tied with a resource that is re-used and not
        // managed by hardware.
        device->m_InferStreamNum = (device->m_InferStreamNum + 1) % device->m_InferStreams.size();

        // clear arrays for next batch
        for (int i = 0; i < u3p.actual_num_slices; i++)
        {
            u3p.patches[i] = nullptr;
            u3p.slice_to_off_d[i] = 0;
            u3p.slice_to_off_h[i] = 0;
            u3p.slice_to_off_w[i] = 0;
        }
    }

    // record event from final inferStream
    if (!m_ServerSettings.RunInferOnCopyStreams)
        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);

    // prepare final ArgMax on copyStream
    if (!m_ServerSettings.RunInferOnCopyStreams)
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);

    // final postprocess of the output, on sample_buffers[1]
    TIMER_START(UNET3DSW_ArgMaxKernel);
    UNet3DKiTS19ArgMaxKernel_wrapper(sample_out_buf, sample_resp_buf, u3p, copyStream, deviceId);
    TIMER_END(UNET3DSW_ArgMaxKernel);
    // coming here, 3D UNet KiTS19 inference on the given sample is done

    // perform copy back to host, if not DirectHostAccess
    if (m_ServerSettings.EnableDma && !directHostAccess)
    {
        auto engine = device->m_Engines.rbegin()->second.front()->GetCudaEngine();

        if (!m_ServerSettings.EndOnDevice)
        {
            KiTS19bufMgr->copyOutputToHostAsync(copyStream);
        }
    }
    else
    {
        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
    }
    CHECK_EQ(cudaEventRecord(dtoh, copyStream), CUDA_SUCCESS);

    // optional synchronization
    if (m_ServerSettings.EnableSyncOnEvent)
        cudaEventSynchronize(dtoh);

    // generate asynchronous response
    TIMER_START(asynchronous_response);
    if (m_ServerSettings.EnableResponse)
    {
        auto buffer = static_cast<int8_t*>(KiTS19bufMgr->getSampleResponseHostBuffer());
        size_t sampleSize = KiTS19bufMgr->getSampleResponseSize();

        Batch batch;
        batch.Event = dtoh;
        batch.Stream = copyStream;
        batch.Responses.emplace_back(mlperf::QuerySampleResponse{begin->id, (uintptr_t) buffer, sampleSize});
        if (device->m_ResponseCallback)
            batch.SampleIds.emplace_back(begin->index);
        if (m_ServerSettings.EndOnDevice)
        {
            auto baseDevice = reinterpret_cast<uintptr_t>(KiTS19bufMgr->getSampleResponseDeviceBuffer());
            auto baseHost = reinterpret_cast<uintptr_t>(KiTS19bufMgr->getSampleResponseHostBuffer());
            batch.ResponseCb = [=](mlperf::QuerySampleResponse* response) {
                auto dbuf = (response->data - baseHost) + baseDevice;
                CHECK_EQ(cudaMemcpyAsync(reinterpret_cast<void*>(response->data), reinterpret_cast<void*>(dbuf),
                             response->size, cudaMemcpyDeviceToHost, copyStream),
                    cudaSuccess);
                // in accuracy mode, response data is copied to accuracy log immediately after
                // callback, so sync stream before returning
                CHECK_EQ(cudaStreamSynchronize(copyStream), cudaSuccess);
            };
        }

        device->m_CompletionQueue.emplace_back(batch);
    }
    TIMER_END(asynchronous_response);
}

void Server::Reset()
{
    m_DeviceIndex = 0;

    for (auto& device : m_Devices)
    {
        device->m_InferStreamNum = 0;
        device->m_Stats.reset();
    }
}

void Server::ProcessSamples()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_DeviceNum++;

    // initial device available
    auto device = GetNextAvailableDevice(deviceId);

    // handle only one KiTS19 image at the same time
    const auto batchSize = 1;

    while (true)
    {
        std::deque<mlperf::QuerySample> samples;
        TIMER_START(m_WorkQueue_acquire_total);
        do
        {
            TIMER_START(m_WorkQueue_acquire);
            m_WorkQueue.acquire(samples, m_ServerSettings.Timeout, batchSize,
                m_ServerSettings.EnableDequeLimit || m_ServerSettings.EnableBatcherThreadPerDevice);
            TIMER_END(m_WorkQueue_acquire);
        } while (samples.empty());
        TIMER_END(m_WorkQueue_acquire_total);

        auto begin = samples.begin();
        auto end = samples.end();

        // Use a null (0) id to represent the end of samples
        if (!begin->id)
        {
            m_DeviceNum--;
            break;
        }
        auto batch_begin = begin;
        while (batch_begin < end)
        {
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
            device = GetNextAvailableDevice(deviceId);
            TIMER_END(GetNextAvailableDevice);
        }
    }
}

void Server::ProcessBatches()
{
    // until Setup is called we may not have valid devices
    size_t deviceId = m_IssueNum++;
    auto& device = m_Devices[deviceId];
    auto& issueQueue = device->m_IssueQueue;

    while (true)
    {
        auto pair = issueQueue.front();
        issueQueue.pop_front();

        auto& batch = pair.first;
        auto& stream = pair.second;

        if (batch.empty())
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
                    // FIXME: Maybe retire this -- Not sure why needed copy between H2D/D2H for
                    // warmup
                    auto bufferManager = std::get<0>(state);
                    // FIXME: find a way of using events for multi-inferStream
                    auto& htod = std::get<1>(state);
                    auto& inf = std::get<2>(state);
                    auto& dtoh = std::get<3>(state);
                    auto& context = std::get<4>(state);

                    bool directHostAccess = (device->m_DLA && m_ServerSettings.EnableDLADirectHostAccess)
                        || m_ServerSettings.EnableDirectHostAccess;

                    device->Issue();
                    auto engine = device->m_Engines.rbegin()->second.front()->GetCudaEngine();

                    if (m_ServerSettings.EnableDma && !directHostAccess)
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

                    Device::t_GraphKey key = std::make_pair(inferStream, device->m_BatchSize);
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
                        if (directHostAccess)
                        {
                            buffers = bufferManager->getHostBindings();
                        }
                        enqueueShim(context, device->m_BatchSize, &buffers[0],
                            (m_ServerSettings.RunInferOnCopyStreams) ? copyStream : inferStream, nullptr);
                    }
                    if (!m_ServerSettings.RunInferOnCopyStreams)
                        CHECK_EQ(cudaEventRecord(inf, inferStream), CUDA_SUCCESS);

                    if (!m_ServerSettings.RunInferOnCopyStreams)
                        CHECK_EQ(cudaStreamWaitEvent(copyStream, inf, 0), CUDA_SUCCESS);
                    if (m_ServerSettings.EnableDma && !directHostAccess)
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
        elapsed = std::chrono::duration<float>(std::chrono::high_resolution_clock::now() - tStart).count();
    } while (elapsed < duration);

    for (auto& device : m_Devices)
    {
        device->Issue();
        cudaDeviceSynchronize();
    }

    // reset server state
    Reset();
}

void Server::FlushQueries()
{
    // This function is called at the end of a series of IssueQuery calls (typically the end of a
    // region of queries that define a performance or accuracy test).  Its purpose is to allow a
    // SUT to force all remaining queued samples out to avoid implementing timeouts.

    // Currently, there is no use case for it in this IS.
}

void Server::SetResponseCallback(std::function<void(::mlperf::QuerySampleResponse* responses,
        std::vector<::mlperf::QuerySampleIndex>& sample_ids, size_t response_count)>
        callback)
{
    std::for_each(
        m_Devices.begin(), m_Devices.end(), [callback](DevicePtr_t device) { device->SetResponseCallback(callback); });
}

} // namespace lwis
