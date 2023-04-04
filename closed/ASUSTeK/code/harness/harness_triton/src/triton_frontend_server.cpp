/*
 * Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
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

#include "triton_frontend_server.hpp"
#include <google/protobuf/util/json_util.h>

#include "loadgen.h"

namespace triton_frontend
{

void Triton_Server_SUT::Init(size_t min_sample_size, size_t max_sample_size, size_t buffer_manager_thread_count,
    bool batch_triton_requests, bool check_contiguity, const std::string& numa_config_str,
    const std::string& backend_path)
{
    auto numa_config = parseNumaConfig(numa_config_str);
#if TRITON_FRONTEND_TRACE
    /* Set up trace manager */
    ni::TraceManager* manager = nullptr;
    FAIL_IF_ERR(triton::server::TraceManager::Create(
                    &manager, TRITONSERVER_TRACE_LEVEL_MAX, 80 /* rate, one sample per batch*/, "triton_trace.log"),
        "creating trace manger");
    m_TraceManager.reset(manager);
#endif // TRITON_FRONTEND_TRACE
    m_BatchTritonRequests = batch_triton_requests;
    m_CheckContiguity = check_contiguity;

    /* Create the options for the server */
    TRITONSERVER_ServerOptions* server_options = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsNew(&server_options), "creating server options");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelControlMode(
                    server_options, TRITONSERVER_ModelControlMode::TRITONSERVER_MODEL_CONTROL_EXPLICIT),
        "Setting model control mode");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetModelRepositoryPath(server_options, m_ModelRepositoryPath.c_str()),
        "setting model repository path");

    // FIXME currently don't need to pass in valid directory as TensorRT backend
    // has not yet decoupled from Triton core, will need to fix once it is
    // separated as dynamically loaded library
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBackendDirectory(server_options, backend_path.c_str()),
        "setting backend directory");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetPinnedMemoryPoolByteSize(server_options, (((uint64_t) 1) << 29)),
        "setting pinned memory pool size");

    FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetBufferManagerThreadCount(server_options, buffer_manager_thread_count),
        "setting buffer manager thread count");

    for (size_t node_idx = 0; node_idx < numa_config.size(); ++node_idx)
    {
        // Convert the NUMA config to form expected by Triton:
        // NUMA node id and CPU cores are set to a host policy which
        // is associated with particular Triton model instances based
        // on device placement.
        // CPU cores are specified in a range format, i.e. if the NUMA node
        // has CPU core 0, 2, 4, 5, 6, the following string should be passed to
        // Triton: "0-0,2-2,4-6"
        const auto& gpu_ids = numa_config[node_idx].first;
        auto cpu_ids = numa_config[node_idx].second;
        std::sort(cpu_ids.begin(), cpu_ids.end());
        std::string cpu_str;
        int lower = -1;
        int upper = -1;
        for (size_t cpu_idx = 0; cpu_idx < cpu_ids.size(); ++cpu_idx)
        {
            if (lower == -1)
            {
                lower = cpu_ids[cpu_idx];
                upper = lower;
            }
            else if (cpu_ids[cpu_idx] == (upper + 1))
            {
                upper = cpu_ids[cpu_idx];
            }
            else
            {
                if (!cpu_str.empty())
                {
                    cpu_str += ",";
                }
                cpu_str += (std::to_string(lower) + "-" + std::to_string(upper));
                lower = -1;
                upper = -1;
            }
        }
        if (lower != -1)
        {
            if (!cpu_str.empty())
            {
                cpu_str += ",";
            }
            cpu_str += (std::to_string(lower) + "-" + std::to_string(upper));
        }
        for (const auto gpu_id : gpu_ids)
        {
            std::string policy_name = "gpu_" + std::to_string(gpu_id);
            FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetHostPolicy(
                            server_options, policy_name.c_str(), "numa-node", std::to_string(node_idx).c_str()),
                "setting NUMA node for host policy");
            FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetHostPolicy(
                            server_options, policy_name.c_str(), "cpu-cores", cpu_str.c_str()),
                "setting CPU cores for host policy");
        }
    }
    // Uncomment this to get detailed verbose logs from triton
    // FAIL_IF_ERR(TRITONSERVER_ServerOptionsSetLogVerbose(server_options, 1),
    // "Setting verbose level 1");

    /* Actually create the server now */
    TRITONSERVER_Server* server_ptr = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerNew(&server_ptr, server_options), "creating server");
    FAIL_IF_ERR(TRITONSERVER_ServerOptionsDelete(server_options), "deleting server options");
    m_Server = std::shared_ptr<TRITONSERVER_Server>(server_ptr, [](TRITONSERVER_Server*){}); // MLPINF-1837

    FAIL_IF_ERR(TRITONSERVER_ServerLoadModel(m_Server.get(), m_ModelName.c_str()), "Loading model");
    /* Wait until the server is both live and ready, and the model is ready. */
    size_t health_iters = 0;
    while (true)
    {
        bool live, ready, model_ready;
        FAIL_IF_ERR(TRITONSERVER_ServerIsLive(m_Server.get(), &live), "unable to get server liveness");
        FAIL_IF_ERR(TRITONSERVER_ServerIsReady(m_Server.get(), &ready), "unable to get server readiness");
        FAIL_IF_ERR(TRITONSERVER_ServerModelIsReady(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_ready),
            "unable to get model readiness");
        std::cout << "Server Health status: live " << live << ", ready " << ready << ", model ready " << model_ready
                  << std::endl;
        if (live && ready && model_ready)
        {
            std::cout << "Server is live and ready. Model is ready" << std::endl;
            break;
        }

        if (++health_iters >= 200)
        {
            FAIL("failed to find healthy inference server within 200 tries");
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Initalize enough pinned output buffer in front
    // We want to have instance # sets of output buffers, each set has
    // output # buffers, and each buffer has max batch bytes for the output
    TRITONSERVER_Message* model_config_message = nullptr;
    FAIL_IF_ERR(TRITONSERVER_ServerModelConfig(
                    m_Server.get(), m_ModelName.c_str(), m_ModelVersion, 1 /* config_version */, &model_config_message),
        "unable to get model config message");
    auto lcm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_config_message, TRITONSERVER_MessageDelete);

    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_config_message, &buffer, &byte_size),
        "unable to serialize model metadata message to JSON");

    ::google::protobuf::util::JsonStringToMessage({buffer, (int) byte_size}, &m_Config);
    std::cout << "Model Config:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
}

void Triton_Server_SUT::ModelMetadata()
{
    TRITONSERVER_Message* model_metadata_message = nullptr;
    FAIL_IF_ERR(
        TRITONSERVER_ServerModelMetadata(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_metadata_message),
        "obtaining metadata message");
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(TRITONSERVER_MessageSerializeToJson(model_metadata_message, &buffer, &byte_size),
        "serializing model metadata message");

    std::cout << "Model Metadata:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
    auto lmm = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_metadata_message, TRITONSERVER_MessageDelete);
}

void Triton_Server_SUT::ModelStats()
{
    TRITONSERVER_Message* model_stats_message = nullptr;
    auto err
        = TRITONSERVER_ServerModelStatistics(m_Server.get(), m_ModelName.c_str(), m_ModelVersion, &model_stats_message);
    if (err != nullptr)
    {
        std::cerr << "failed to obtain stats message: " << TRITONSERVER_ErrorMessage(err) << std::endl;
        TRITONSERVER_ErrorDelete(err);
        return;
    }
    auto lms = std::unique_ptr<TRITONSERVER_Message, decltype(&TRITONSERVER_MessageDelete)>(
        model_stats_message, TRITONSERVER_MessageDelete);
    const char* buffer;
    size_t byte_size;
    FAIL_IF_ERR(
        TRITONSERVER_MessageSerializeToJson(model_stats_message, &buffer, &byte_size), "serializing stats message");

    std::cout << "Model Stats:" << std::endl;
    std::cout << std::string(buffer, byte_size) << std::endl;
}

void Triton_Server_SUT::TraceCaptureTimeStamp(TRITONSERVER_InferenceTrace* trace_ptr, const std::string comment)
{
#if TRITON_FRONTEND_TRACE
    if (m_TraceManager != nullptr)
    {
        if (trace_ptr != nullptr)
        {
            uint64_t trace_id = 0;
            TRITONSERVER_InferenceTraceId(trace_ptr, &trace_id);
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            m_TraceManager->CaptureTimestamp(trace_id, TRITONSERVER_TRACE_LEVEL_MIN, comment, TIMESPEC_TO_NANOS(ts));
        }
    }
#endif
}

void Triton_Server_SUT::QuerySamplesComplete(
    mlperf::QuerySampleResponse* responses, size_t num_responses, int64_t pool_idx)
{
    if (m_EndOnDevice)
    {
        auto pool = (*m_OutputBufferPoolEndOnDevice)[pool_idx];
        auto block = pool->Obtain();
        const auto end_on_device_cb = [=](::mlperf::QuerySampleResponse* response) {
            const auto offset = response->data - responses[0].data;
            auto host_buf = block->m_Data + offset;
            const auto stream = this->m_EndOnDeviceCopyStream.stream;
            FAIL_IF_CUDA_ERR(
                cudaMemcpyAsync(host_buf, (const void*) response->data, response->size, cudaMemcpyDeviceToHost, stream),
                "failed to memcpy async");
            response->data = (uintptr_t) host_buf;
            FAIL_IF_CUDA_ERR(cudaStreamSynchronize(stream), "failed to synchronize stream");
        };
        mlperf::QuerySamplesComplete(responses, num_responses, end_on_device_cb);
        pool->Release(block);
    }
    else
    {
        mlperf::QuerySamplesComplete(responses, num_responses);
    }
}

} // namespace triton_frontend