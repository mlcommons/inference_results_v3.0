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

#ifndef __QSL_3DUNET_HPP__
#define __QSL_3DUNET_HPP__

#include <chrono>
#include <deque>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <sstream>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>

#include "NvInfer.h"
#include "NvInferPlugin.h"

#include "numpy.hpp"

#include "logger.h"
#include <glog/logging.h>

#include "loadgen.h"
#include "query_sample_library.h"
#include "test_settings.h"

// For KiTS19, QSL also need to bookkeep the dimensions of each sample
// Rewriting QSL, adding some variables and functions for KiTS19 samples
namespace qsl
{
struct kits19_dim
{
    int c;
    int d;
    int h;
    int w;

    kits19_dim()
    {
        c = 1;
        d = 128;
        h = 128;
        w = 128;
    }

    kits19_dim(int _c, int _d, int _h, int _w)
        : c(_c)
        , d(_d)
        , h(_h)
        , w(_w)
    {
    }
};

class LookupableQuerySampleLibrary : public mlperf::QuerySampleLibrary
{
public:
    virtual void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx, size_t device_idx = 0) = 0;
    virtual size_t GetSampleSize(mlperf::QuerySampleIndex sample_index, size_t input_idx) const = 0;
    virtual size_t GetMaxSampleSize() const = 0;
    virtual kits19_dim GetSampleDimension(mlperf::QuerySampleIndex sample_index, size_t input_idx) const = 0;
};

class SampleLibrary3DUNet : public LookupableQuerySampleLibrary
{
public:
    SampleLibrary3DUNet(std::string name, std::string mapPath, std::vector<std::string> tensorPaths,
        size_t perfSampleCount, size_t padding = 0, bool coalesced = false,
        std::vector<bool> startFromDevice = std::vector<bool>(1, false))
        : m_Name(name)
        , m_PerfSampleCount(perfSampleCount)
        , m_PerfSamplePadding(padding)
        , m_Coalesced(coalesced)
        , m_MapPath(mapPath)
        , m_TensorPaths(tensorPaths)
    {
        CHECK(m_Coalesced == false) << "3DUNet-KiTS19 cannot force use of contiguous input tensors";
        CHECK(m_PerfSamplePadding == 0) << "3DUNet-KiTS19 cannot handle any additional perf sample "
                                           "(PerfSamplePadding not allowed)";

        m_StartFromDevice.swap(startFromDevice);

        // load and read in the sample map
        std::ifstream fs(m_MapPath);
        CHECK(fs) << "Unable to open sample map file: " << m_MapPath;

        char s[1024];
        while (fs.getline(s, 1024))
        {
            std::istringstream iss(s);
            std::vector<std::string> r((std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

            m_FileLabelMap.insert(
                std::make_pair(m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
            m_SampleCount++;
        }

        // Get input size and allocate memory
        m_NumInputs = m_TensorPaths.size();
        // m_SampleMemory is going to be contiguous
        m_SampleMemory.resize(m_NumInputs);
        m_SampleSizes.resize(m_SampleCount);
        m_SampleDimensions.resize(m_SampleCount);
        for (int sample_cnt = 0; sample_cnt < m_SampleCount; ++sample_cnt)
        {
            m_SampleSizes[sample_cnt].resize(m_NumInputs);
            m_SampleDimensions[sample_cnt].resize(m_NumInputs);
        }

        // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
        m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);

        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            size_t sampleSizeSum = 0;
            for (size_t sample_idx = 0; sample_idx < m_SampleCount; sample_idx++)
            {
                std::string path = m_TensorPaths[input_idx] + "/" + std::get<0>(m_FileLabelMap[sample_idx]) + ".npy";
                npy::NpyFile npy(path);
                auto thisSampleSize = npy.getTensorSize();
                if (thisSampleSize > m_SampleSizeMax)
                    m_SampleSizeMax = thisSampleSize;
                m_SampleSizes[sample_idx][input_idx] = thisSampleSize;
                sampleSizeSum += thisSampleSize;
                std::vector<size_t> sample_dim = npy.getDims();
                CHECK(sample_dim.size() == 4) << "Expected 4D tensor, but got " << sample_dim.size()
                                              << "D, for sample idx: " << sample_idx << ", input idx: " << input_idx;
                m_SampleDimensions[sample_idx][input_idx]
                    = kits19_dim(sample_dim[0], sample_dim[1], sample_dim[2], sample_dim[3]);
            }
            // allocate buffers
            if (!m_StartFromDevice[input_idx])
            {
                m_SampleMemory[input_idx].resize(1);
                CHECK_EQ(cudaMallocHost(&m_SampleMemory[input_idx][0], sampleSizeSum), cudaSuccess);
            }
            else
            {
                CHECK_EQ(cudaGetDeviceCount(&m_NumDevices), cudaSuccess);
                m_SampleMemory[input_idx].resize(m_NumDevices);

                for (int device_idx = 0; device_idx < m_NumDevices; device_idx++)
                {
                    CHECK_EQ(cudaSetDevice(device_idx), cudaSuccess);
                    CHECK_EQ(cudaMalloc(&m_SampleMemory[input_idx][device_idx], sampleSizeSum), cudaSuccess);
                }
            }
        }
    }

    ~SampleLibrary3DUNet()
    {
        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            if (!m_StartFromDevice[input_idx])
            {
                CHECK_EQ(cudaFreeHost(m_SampleMemory[input_idx][0]), cudaSuccess);
            }
            else
            {
                for (int device_idx = 0; device_idx < m_NumDevices; device_idx++)
                {
                    CHECK_EQ(cudaFree(m_SampleMemory[input_idx][device_idx]), cudaSuccess);
                }
            }
        }
    }

    const std::string& Name() override
    {
        return m_Name;
    }
    size_t TotalSampleCount() override
    {
        return m_SampleCount;
    }
    size_t PerformanceSampleCount() override
    {
        return m_PerfSampleCount;
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        // prep containers
        for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
        {
            auto& sampleId = samples[sampleIndex];
            m_SampleAddressMapHost[sampleId].push_back(std::vector<void*>(m_NumInputs, nullptr));
            m_SampleAddressMapDevice[sampleId].push_back(std::vector<std::vector<void*>>(m_NumInputs));
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
            {
                if (m_StartFromDevice[input_idx])
                {
                    m_SampleAddressMapDevice[sampleId].back()[input_idx].resize(m_NumDevices, nullptr);
                }
            }
        }

        // copy the samples into pinned memory and construct sample address map
        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++)
        {
            size_t addressOffset = 0;
            for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++)
            {
                // sampleId is unique ID given to original sample set; sampleIndex is the index for
                // samples vector
                auto& sampleId = samples[sampleIndex];

                std::string path = m_TensorPaths[input_idx] + "/" + std::get<0>(m_FileLabelMap[sampleId]) + ".npy";
                npy::NpyFile npy(path);
                std::vector<char> data;
                npy.loadAll(data);

                if (!m_StartFromDevice[input_idx])
                {
                    auto sampleAddress = static_cast<int8_t*>(m_SampleMemory[input_idx][0]) + addressOffset;
                    std::memcpy((char*) sampleAddress, data.data(), m_SampleSizes[sampleId][input_idx]);
                    m_SampleAddressMapHost[sampleId].back()[input_idx] = sampleAddress;
                }
                else
                {
                    for (int device_idx = 0; device_idx < m_NumDevices; device_idx++)
                    {
                        auto sampleAddress
                            = static_cast<int8_t*>(m_SampleMemory[input_idx][device_idx]) + addressOffset;
                        CHECK_EQ(cudaSetDevice(device_idx), cudaSuccess);
                        CHECK_EQ(cudaMemcpy(sampleAddress, data.data(), m_SampleSizes[sampleId][input_idx],
                                     cudaMemcpyHostToDevice),
                            cudaSuccess);
                        m_SampleAddressMapDevice[sampleId].back()[input_idx][device_idx] = sampleAddress;
                    }
                }
                addressOffset += m_SampleSizes[sampleId][input_idx];
            }
        }

        if (std::any_of(m_StartFromDevice.begin(), m_StartFromDevice.end(), [](bool i) { return i == true; }))
        {
            for (int device_idx = 0; device_idx < m_NumDevices; device_idx++)
            {
                CHECK_EQ(cudaSetDevice(device_idx), cudaSuccess);
                CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
            }
        }
    }

    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        // due to the removal of freelisting this code is currently a check and not required for
        // functionality.
        for (auto& sampleId : samples)
        {
            {
                auto it = m_SampleAddressMapHost.find(sampleId);
                CHECK(it != m_SampleAddressMapHost.end()) << "Sample: " << sampleId << " not allocated properly";
                auto& sampleAddresses = it->second;
                CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";
                sampleAddresses.pop_back();
                if (sampleAddresses.empty())
                {
                    m_SampleAddressMapHost.erase(it);
                }
            }
            {
                auto it = m_SampleAddressMapDevice.find(sampleId);
                CHECK(it != m_SampleAddressMapDevice.end()) << "Sample: " << sampleId << " not allocated properly";
                auto& sampleAddresses = it->second;
                CHECK(!sampleAddresses.empty()) << "Sample: " << sampleId << " not loaded";
                sampleAddresses.pop_back();
                if (sampleAddresses.empty())
                {
                    m_SampleAddressMapDevice.erase(it);
                }
            }
        }

        CHECK(m_SampleAddressMapHost.empty() && m_SampleAddressMapDevice.empty())
            << "Unload did not remove all samples";
    }

    void* GetSampleAddress(mlperf::QuerySampleIndex sample_idx, size_t input_idx, size_t device_idx = 0) override
    {
        if (!m_StartFromDevice[input_idx])
        {
            auto it = m_SampleAddressMapHost.find(sample_idx);
            CHECK(it != m_SampleAddressMapHost.end()) << "Sample: " << sample_idx << " missing from RAM";
            CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";
            return it->second.front()[input_idx];
        }

        auto it = m_SampleAddressMapDevice.find(sample_idx);
        CHECK(it != m_SampleAddressMapDevice.end()) << "Sample: " << sample_idx << " missing from Device RAM";
        CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";
        return it->second.front()[input_idx][device_idx];
    }

    size_t GetSampleSize(mlperf::QuerySampleIndex sample_idx, size_t input_idx) const override
    {
        return (m_SampleSizes.empty() ? 0 : m_SampleSizes[sample_idx][input_idx]);
    }

    size_t GetMaxSampleSize() const override
    {
        return m_SampleSizeMax;
    }

    kits19_dim GetSampleDimension(mlperf::QuerySampleIndex sample_idx, size_t input_idx) const override
    {
        return (m_SampleDimensions.empty() ? kits19_dim{0, 0, 0, 0} : m_SampleDimensions[sample_idx][input_idx]);
    }

protected:
    size_t m_NumInputs{0};
    int m_NumDevices{1};

private:
    const std::string m_Name;
    size_t m_PerfSampleCount{0};
    size_t m_PerfSamplePadding{0};
    std::string m_MapPath;
    std::vector<std::string> m_TensorPaths;
    bool m_Coalesced;
    std::vector<bool> m_StartFromDevice;
    std::vector<std::vector<void*>> m_SampleMemory;
    std::vector<std::vector<size_t>> m_SampleSizes;
    size_t m_SampleSizeMax{0};
    std::vector<std::vector<kits19_dim>> m_SampleDimensions;
    std::vector<std::unique_ptr<npy::NpyFile>> m_NpyFiles;
    size_t m_SampleCount{0};
    // maps sampleId to <fileName, label>
    std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;
    // maps sampleId to num_inputs of <address>
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<void*>>> m_SampleAddressMapHost;
    // maps sampleId to num_inputs of num_devices of <address>
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<std::vector<void*>>>> m_SampleAddressMapDevice;

    nvinfer1::DataType m_Precision{nvinfer1::DataType::kINT8};
};

typedef std::shared_ptr<qsl::SampleLibrary3DUNet> SampleLibrary3DUNetPtr_t;

class SampleLibrary3DUNetEnsemble : public mlperf::QuerySampleLibrary
{
public:
    SampleLibrary3DUNetEnsemble(const std::vector<SampleLibrary3DUNetPtr_t> qsls)
        : m_qsls(qsls){};
    const std::string& Name() override
    {
        return m_qsls[0]->Name();
    }
    size_t TotalSampleCount() override
    {
        return m_qsls[0]->TotalSampleCount();
    }
    size_t PerformanceSampleCount() override
    {
        return m_qsls[0]->PerformanceSampleCount();
    }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        for (auto qsl : m_qsls)
        {
            qsl->LoadSamplesToRam(samples);
        }
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override
    {
        for (auto qsl : m_qsls)
        {
            qsl->UnloadSamplesFromRam(samples);
        }
    }

private:
    std::vector<SampleLibrary3DUNetPtr_t> m_qsls;
};

} // namespace qsl

#endif // __QSL_3DUNET_HPP__