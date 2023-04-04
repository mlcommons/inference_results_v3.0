/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
 *
 * Modified by NEUCHIPS on 2023.
 *
 */

#ifndef __QSL_HPP__
#define __QSL_HPP__

#include "config.h"

#include <chrono>
#include <fcntl.h>
#include <fstream>

#include <iostream>
#include <fstream>
#include <string>

#include <sstream>
#include <string>
#include <thread>
#include <iostream>
#include <iterator>
#include <map>
#include <deque>
#include <vector>
#include <set>

#include <cuda.h>
#include <cuda_runtime.h>

#include <glog/logging.h>

#include "query_sample_library.h"
#include "test_settings.h"
#include "numpy.hpp"

#include <unistd.h>
#include <sys/mman.h>

#include "neuchips_user.h"

// QSL (Query Sample Library) is an implementation of the MLPerf Query Sample Library.  It's purpose
// is to:
// 1) Allow samples to be loaded and unloaded dynamically at runtime from Loadgen.
// 2) Support lookup of currently loaded tensor addresses in memory.

namespace qsl {
  class LookupableQuerySampleLibrary : public mlperf::QuerySampleLibrary {
    public:
        virtual void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx, size_t device_idx = 0) = 0;
        virtual size_t GetSampleSize(size_t input_idx) const = 0;
    };
  class SampleLibrary : public LookupableQuerySampleLibrary {
  public:
    SampleLibrary(std::string name, std::string mapPath, std::vector<std::string> tensorPaths, size_t perfSampleCount,
      size_t padding = 0, bool coalesced = false, std::vector<bool> startFromDevice = std::vector<bool>(1,false) )
        : m_Name(name), m_PerfSampleCount(perfSampleCount), m_PerfSamplePadding(padding), m_MapPath(mapPath), m_TensorPaths(tensorPaths),
          m_Coalesced(coalesced), m_1st_sample_size(0) {

      for (auto it = begin (startFromDevice); it != end (startFromDevice); ++it) {
        LOG(INFO) << "startFromDevice " << *it;
      }

      m_StartFromDevice.swap(startFromDevice);
      // Get input size and allocate memory
      for (auto it = begin (m_TensorPaths); it != end (m_TensorPaths); ++it) {
        LOG(INFO) << "m_TensorPath: " << *it;
      }
      m_NumInputs = m_TensorPaths.size();
      m_SampleSizes.resize(m_NumInputs);
      LOG(INFO) << "m_SampleSizes.size(): " << m_SampleSizes.size();

      m_SampleMemory.resize(m_NumInputs);
      LOG(INFO) << "m_SampleMemory.size(): " << m_SampleMemory.size();

      // Initialize npy file caches in coalescing mode
      LOG(INFO) << "m_Coalesced: " << m_Coalesced;
      if (m_Coalesced) {
        m_NpyFiles.resize(m_NumInputs);
        for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
          m_NpyFiles[input_idx].reset(new npy::NpyFile(m_TensorPaths[input_idx]));
        }
      }

      // Get number of samples
      if (!m_Coalesced) {
        // load and read in the sample map
        std::ifstream fs(m_MapPath);
        CHECK(fs) << "Unable to open sample map file: " << m_MapPath;

        char s[1024];
        while (fs.getline(s, 1024)) {
          std::istringstream iss(s);
          std::vector<std::string> r((std::istream_iterator<std::string>{iss}), std::istream_iterator<std::string>());

          m_FileLabelMap.insert(std::make_pair(m_SampleCount, std::make_tuple(r[0], (r.size() > 1 ? std::stoi(r[1]) : 0))));
          m_SampleCount++;
        }
      }
      else {
        // In coalescing mode, the first dimension is number of samples
        m_SampleCount = m_NpyFiles[0]->getDims()[0];
      }

      // as a safety, don't allow the perfSampleCount to be larger than sampleCount.
      m_PerfSampleCount = std::min(m_PerfSampleCount, m_SampleCount);

      for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
        if (!m_Coalesced) {
          std::string path = m_TensorPaths[input_idx] + "/" + std::get<0>(m_FileLabelMap[0]) + ".npy";
          npy::NpyFile npy(path);
          m_SampleSizes[input_idx] = npy.getTensorSize();
        }
        else {
          LOG(INFO) << "m_NpyFiles[input_idx]->getTensorSize() " << input_idx << ", getTensorSize() " << m_NpyFiles[input_idx]->getTensorSize() << ", m_SampleCount " << m_SampleCount;
          m_SampleSizes[input_idx] = m_NpyFiles[input_idx]->getTensorSize() / m_SampleCount;
        }

        for (auto it = begin (m_SampleSizes); it != end (m_SampleSizes); ++it) {
          LOG(INFO) << "m_SampleSizes " << *it;
	}
        m_SampleMemory[input_idx].resize(1);
        CHECK_EQ(ncsMallocHost(&m_SampleMemory[input_idx][0], (m_PerfSampleCount + m_PerfSamplePadding) * m_SampleSizes[input_idx]), cudaSuccess);
      }
    }

    int ncsMallocHost(void** memptr, size_t size)
    {
        void* ptr;
        ptr = malloc(size);
        if(!ptr)
        {
          return -1;
        }
        else
        {
          *memptr = ptr;
        }

        return 0;
    }

    int ncsFreeHost(void* ptr)
    {
        free(ptr);
        return 0;
    }

    int ncsFree(void* ptr)
    {
        free(ptr);
        return 0;
    }

    ~SampleLibrary() {
      for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
        if (!m_StartFromDevice[input_idx]) {
          CHECK_EQ(ncsFreeHost(m_SampleMemory[input_idx][0]), cudaSuccess);
        }
        else {
          for (int device_idx = 0; device_idx < m_NumDevices; device_idx++) {
            CHECK_EQ(ncsFree(m_SampleMemory[input_idx][device_idx]), cudaSuccess);
          }
        }
      }
    }

    const std::string &Name() /*const*/ override { return m_Name; }
    size_t TotalSampleCount() override { return m_SampleCount; }
    size_t PerformanceSampleCount() override { return m_PerfSampleCount; }

    void set_1st_sample_size(u32 first_sample_size)
    {
        if(first_sample_size > 0)
        {
            m_1st_sample_size = first_sample_size;
            FILE *fp;
            fp = fopen("mlperf_sample.bin", "wb");
            DLOG(INFO) << "qsl set_1st_sample_size " << first_sample_size;
            fwrite(&first_sample_size, sizeof(first_sample_size), 1, fp);
            fclose(fp);
        }
    }

    inline u32 get_1st_sample_size()
    {
        return m_1st_sample_size;
    }

    #define MAX_PATH_LEN 128

    static int ncs_preload_init_multithread(const std::vector<mlperf::QuerySampleIndex>& flatten_samples, u32 numa_idx, u32 device_id, std::vector<u32>& partition_sizes, 
                            std::vector<size_t>& sampleSizes, int skipLoadSamples, size_t m_NumInputs, std::vector<char>& data_aligner, std::vector<char>& data_emb, u32* first_sample_size)
    {
        int fd;
        char devname_path[MAX_PATH_LEN];
        struct ncs_arg_dma_buf_init arg_dma_buf_init;
        struct neuchip_ioctl_arg ioctl_data;
        u32 num_of_partitions = partition_sizes.size();

        arg_dma_buf_init.numa_node = numa_idx;
        arg_dma_buf_init.nents = num_of_partitions;
        arg_dma_buf_init.partition_sizes = partition_sizes.data(); //partitionSizes->getRawArray(); //batchGenerator->getPartitionSizesArray();

        snprintf(devname_path, MAX_PATH_LEN, "/dev/%s-%d", DRIVER_PATH_NAME, device_id);
        LOG(INFO) << "ncs_preload_init_multithread: " << devname_path << ", num_of_partitions:" << num_of_partitions;
        fd = open(devname_path, O_RDWR);
        if(fd < 0) {
            LOG(INFO) << "Cannot open device file... /dev/ " << device_id;
            return -1;
        }

        ioctl(fd, IOCX_DMA_BUF_INIT, &arg_dma_buf_init);


        if(!skipLoadSamples)
        {
            LOG(INFO) << "ncs_preload_init_multithread: LoadSamplesToRamNeuchips m_NumInputs " << m_NumInputs;
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
                //std::vector<char>& data;
                //m_NpyFiles[input_idx]->loadSamples(data, flatten_samples);
                if(0 == input_idx)
                {
                    LOG(INFO) << "ncs_preload_init_multithread: input_idx(" << input_idx << ") lwis qsl.hpp m_NpyFiles[" << input_idx << "]->loadSamples() data.size() " << data_aligner.size();
                    //data = data_aligner;
                }
                else
                {
                    LOG(INFO) << "ncs_preload_init_multithread: input_idx(" << input_idx << ") lwis qsl.hpp m_NpyFiles[" << input_idx << "]->loadSamples() data.size() " << data_emb.size();
                    //data = data_emb;
                }

                if(input_idx == 0)
                {
                    char* host_buf_numeric = data_aligner.data();
                    struct ncs_arg_fill_dma_data arg_fill_dma_data;

                    LOG(INFO) << "ncs_preload_init_multithread: IOCX_FILL_DMA_DATA flatten_samples.size() " << flatten_samples.size() << " sampleSizes[" << input_idx << "] " << sampleSizes[input_idx];
                    for(int i = 0; i < arg_dma_buf_init.nents; i++) //3 ch * 204800 buffers times of IOCX_FILL_DMA_DATA
                    {
                        arg_fill_dma_data.user_host_buffer = (u8*) host_buf_numeric;
                        arg_fill_dma_data.input_index = ALIGN_PING_IDX;
                        arg_fill_dma_data.len = arg_dma_buf_init.partition_sizes[i] * sampleSizes[input_idx];
                        arg_fill_dma_data.sample_index = i;

                        if(0 == i)
                        {
                            LOG(INFO) << "ncs_preload_init: partition_sizes[0] " << arg_dma_buf_init.partition_sizes[0] << " set_1st_sample_size " << arg_dma_buf_init.partition_sizes[0];
                            //set_1st_sample_size(arg_dma_buf_init.partition_sizes[0]);
                            *first_sample_size = arg_dma_buf_init.partition_sizes[0];
                        }


                        ioctl(fd, IOCX_FILL_DMA_DATA, &arg_fill_dma_data);
                        host_buf_numeric += arg_fill_dma_data.len;
                    }
                }              
                else if(input_idx == 1) //special handling to split to emb top & bottom
                {
                    char* npy_file_data_emb = data_emb.data();
                    u8* host_buf_top = new u8 [1000*26*8]; //208*96
                    u8* host_buf_bottom = new u8 [1000*96*26*8]; //208*96
                    u32 total_bytes = 0;

                    struct ncs_arg_fill_dma_data arg_fill_dma_data;
                    size_t emb_full_size = sampleSizes[input_idx];


                    LOG(INFO) << "ncs_preload_init: IOCX_FILL_DMA_DATA flatten_samples.size() " << flatten_samples.size() << " sampleSizes[" << input_idx << "] " << sampleSizes[input_idx];
                    for(u32 i = 0; i < arg_dma_buf_init.nents; i++) //3 ch * 204800 buffers times of IOCX_FILL_DMA_DATA,
                    {
                        CHECK(26 * sizeof(int64_t) == sampleSizes[input_idx]) << "Unexpected m_SampleSizes[" << input_idx << "] " << sampleSizes[input_idx];
                        u8* ptr_top = host_buf_top;
                        u8* ptr_bot = host_buf_bottom;

                        memcpy(ptr_top, npy_file_data_emb, arg_dma_buf_init.partition_sizes[i]* emb_full_size);
                        npy_file_data_emb += (arg_dma_buf_init.partition_sizes[i]* emb_full_size);

                        arg_fill_dma_data.input_index = EMB_TOP_PING_IDX;
                        arg_fill_dma_data.sample_index = i;
                        arg_fill_dma_data.len = arg_dma_buf_init.partition_sizes[i] * emb_full_size;
                        arg_fill_dma_data.user_host_buffer = (u8*) host_buf_top;
                        //LOG(INFO) << "arg_fill_dma_data(" << arg_fill_dma_data.input_index << ")[" << i << "], size " << arg_dma_buf_init.partition_sizes[i] << " * " << m_SampleSizes[input_idx]/2;
                        ioctl(fd, IOCX_FILL_DMA_DATA, &arg_fill_dma_data);
                    }
                    LOG(INFO) << "IOCX_FILL_DMA_DATA total_bytes " << total_bytes << " data_emb.size() " << data_emb.size();

                    delete host_buf_top;
                    delete host_buf_bottom;
                }
            }
        }
       
        return fd;
    }

    int ncs_preload_init(const std::vector<mlperf::QuerySampleIndex>& flatten_samples, u32 numa_idx, u32 device_id, std::vector<u32>& partition_sizes, std::vector<size_t>& sampleSizes, int skipLoadSamples)
    {
        int fd;
        char devname_path[MAX_PATH_LEN];
        struct ncs_arg_dma_buf_init arg_dma_buf_init;
        struct neuchip_ioctl_arg ioctl_data;
        u32 num_of_partitions = partition_sizes.size();

        arg_dma_buf_init.numa_node = numa_idx;
        arg_dma_buf_init.nents = num_of_partitions;
        arg_dma_buf_init.partition_sizes = partition_sizes.data(); //partitionSizes->getRawArray(); //batchGenerator->getPartitionSizesArray();

        snprintf(devname_path, MAX_PATH_LEN, "/dev/%s-%d", DRIVER_PATH_NAME, device_id);
        LOG(INFO) << "ncs_preload_init: " << devname_path << ", num_of_partitions:" << num_of_partitions;
        fd = open(devname_path, O_RDWR);
        if(fd < 0) {
            LOG(INFO) << "Cannot open device file...";
            return -1;
        }

        ioctl(fd, IOCX_DMA_BUF_INIT, &arg_dma_buf_init);

        if(!skipLoadSamples)
        {
            LOG(INFO) << "ncs_preload_init: LoadSamplesToRamNeuchips m_NumInputs " << m_NumInputs;
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
                std::vector<char> data;
                m_NpyFiles[input_idx]->loadSamples(data, flatten_samples);
                LOG(INFO) << "ncs_preload_init: input_idx(" << input_idx << ") lwis qsl.hpp m_NpyFiles[" << input_idx << "]->loadSamples() data.size() " << data.size();
                if(input_idx == 0)
                {
                    char* host_buf_numeric = data.data();
                    struct ncs_arg_fill_dma_data arg_fill_dma_data;

                    LOG(INFO) << "ncs_preload_init: IOCX_FILL_DMA_DATA flatten_samples.size() " << flatten_samples.size() << " m_SampleSizes[" << input_idx << "] " << m_SampleSizes[input_idx];
                    for(int i = 0; i < arg_dma_buf_init.nents; i++) //3 ch * 204800 buffers times of IOCX_FILL_DMA_DATA
                    {
                        arg_fill_dma_data.user_host_buffer = (u8*) host_buf_numeric;
                        arg_fill_dma_data.input_index = ALIGN_PING_IDX;
                        arg_fill_dma_data.len = arg_dma_buf_init.partition_sizes[i] * m_SampleSizes[input_idx];
                        arg_fill_dma_data.sample_index = i;

                        if(0 == i)
                        {
                            LOG(INFO) << "ncs_preload_init: partition_sizes[0] " << arg_dma_buf_init.partition_sizes[0];
                            set_1st_sample_size(arg_dma_buf_init.partition_sizes[0]);
                        }

                        ioctl(fd, IOCX_FILL_DMA_DATA, &arg_fill_dma_data);
                        host_buf_numeric += arg_fill_dma_data.len;
                    }
                }
                else if(input_idx == 1) //special handling to split to emb top & bottom
                {
                    char* npy_file_data_emb = data.data();
                    u8* host_buf_top = new u8 [1000*26*8]; //208*96
                    u8* host_buf_bottom = new u8 [1000*96*26*8]; //208*96
                    u32 total_bytes = 0;

                    struct ncs_arg_fill_dma_data arg_fill_dma_data;
                    size_t emb_full_size = m_SampleSizes[input_idx];
                    //size_t emb_half_size = m_SampleSizes[input_idx]/2;

                    LOG(INFO) << "ncs_preload_init: IOCX_FILL_DMA_DATA flatten_samples.size() " << flatten_samples.size() << " m_SampleSizes[" << input_idx << "] " << m_SampleSizes[input_idx];
                    for(u32 i = 0; i < arg_dma_buf_init.nents; i++) //3 ch * 204800 buffers times of IOCX_FILL_DMA_DATA,
                    {
                        CHECK(26 * sizeof(int64_t) == m_SampleSizes[input_idx]) << "Unexpected m_SampleSizes[" << input_idx << "] " << m_SampleSizes[input_idx];
                        u8* ptr_top = host_buf_top;
                        u8* ptr_bot = host_buf_bottom;
/*
                        memcpy(ptr_top, npy_file_data_emb, arg_dma_buf_init.partition_sizes[i]* emb_half_size);
                        npy_file_data_emb += (arg_dma_buf_init.partition_sizes[i]* emb_half_size);
                        memcpy(ptr_bot, npy_file_data_emb, arg_dma_buf_init.partition_sizes[i]* emb_half_size);
                        npy_file_data_emb += (arg_dma_buf_init.partition_sizes[i]* emb_half_size);
*/
                        memcpy(ptr_top, npy_file_data_emb, arg_dma_buf_init.partition_sizes[i]* emb_full_size);
                        npy_file_data_emb += (arg_dma_buf_init.partition_sizes[i]* emb_full_size);

                        arg_fill_dma_data.input_index = EMB_TOP_PING_IDX;
                        arg_fill_dma_data.sample_index = i;
                        arg_fill_dma_data.len = arg_dma_buf_init.partition_sizes[i] * emb_full_size;
                        arg_fill_dma_data.user_host_buffer = (u8*) host_buf_top;
                        //LOG(INFO) << "arg_fill_dma_data(" << arg_fill_dma_data.input_index << ")[" << i << "], size " << arg_dma_buf_init.partition_sizes[i] << " * " << m_SampleSizes[input_idx]/2;
                        ioctl(fd, IOCX_FILL_DMA_DATA, &arg_fill_dma_data);
                    }
                    LOG(INFO) << "IOCX_FILL_DMA_DATA total_bytes " << total_bytes << " data.size() " << data.size();

                    delete host_buf_top;
                    delete host_buf_bottom;
                }
            }
        }
        return fd;
    }

    int ncs_preload_uninit(int device_idx)
    {
        int fd;
        char devname_path[MAX_PATH_LEN];
        struct neuchip_ioctl_arg ioctl_arg;

        snprintf(devname_path, MAX_PATH_LEN, "/dev/%s-%d", DRIVER_PATH_NAME, device_idx);
        LOG(INFO) << "ncs_preload_uninit: Opening Driver "  << devname_path;
        fd = open(devname_path, O_RDWR);
        if(fd < 0) {
                printf("Cannot open device file...\n");
                return -1;
        }

        LOG(INFO) << "IOCX_DMA_BUF_RELEASE +";
        ioctl(fd, IOCX_DMA_BUF_RELEASE, NULL);
        LOG(INFO) << "IOCX_DMA_BUF_RELEASE -";

        close(fd);

        return 0;

    }

    void LoadSamplesToRamNeuchips(const std::vector<mlperf::QuerySampleIndex>& flatten_samples, const std::vector<mlperf::QuerySampleIndex>& samples, std::vector<int>& sampleStartIdxs, size_t totalSampleCount,
                    int32_t numaIdx, int32_t numNuma, std::vector<int> deviceIDs) /*override*/ {

        LOG(INFO) << "LoadSamplesToRamNeuchips: std::vector<mlperf::QuerySampleIndex> samples.size() " << samples.size()
                    << " sampleStartIdxs.size() " << sampleStartIdxs.size() << " totalSampleCount " << totalSampleCount << " numaIdx " << numaIdx;

        std::vector<u32> partition_sizes;
        for (auto index : samples)
        {
            CHECK_EQ((index >= 0) && (index < totalSampleCount), true);
            int start = sampleStartIdxs[index];
            int end = sampleStartIdxs[index + 1];
            partition_sizes.push_back(end - start);
        }

        int skipLoadSample = 0;

#ifdef USE_MULTI_THREAD_NCS_PRELOAD
        u32 first_sample_size;
        if(0 == numaIdx)
        {
            std::vector<char> data_aligner;
            std::vector<char> data_emb;
            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
                if(0 == input_idx)
                {
                    m_NpyFiles[input_idx]->loadSamples(data_aligner, flatten_samples);
                }
                else if(1 == input_idx)
                {
                    m_NpyFiles[input_idx]->loadSamples(data_emb, flatten_samples);
                }            
            }

            std::vector<std::thread> downloadThreads;
            downloadThreads.reserve(9);
            //for(auto id : deviceIDs)
            LOG(INFO) << "LoadSamplesToRamNeuchips numaIdx " << numaIdx;

            for(int id = 0; id < 9; id++)
            {
                LOG(INFO) << "creating loading thread " << id;
                std::thread t(&ncs_preload_init_multithread, std::ref(flatten_samples), numaIdx, id, std::ref(partition_sizes), std::ref(m_SampleSizes), skipLoadSample, 
                            m_NumInputs, std::ref(data_aligner), std::ref(data_emb), &first_sample_size);
                downloadThreads.emplace_back(std::move(t));
            }

            for (int i = 0; i < downloadThreads.size(); ++i) {
                LOG(INFO) << "wait loading thread " << i;
                downloadThreads[i].join();
                LOG(INFO) << "done loading thread " << i;
            }

            LOG(INFO) << "finish multithread loading.";

            LOG(INFO) << "set_1st_sample_size " << first_sample_size;
            if(first_sample_size > 0)
            {
                set_1st_sample_size(first_sample_size);
            }
        }



#else
        //currently device_id is set to as the same as numa_idx
        for(auto id : deviceIDs)
        {
            ncs_preload_init(flatten_samples, numaIdx, id, partition_sizes, m_SampleSizes, skipLoadSample);
            skipLoadSample++;
        }
#endif  //#ifdef USE_MULTI_THREAD_NCS_PRELOAD


        // construct sample address map
        for (size_t sampleIndex = 0; sampleIndex < samples.size(); sampleIndex++) {
            auto& sampleId = samples[sampleIndex];

            m_SampleAddressMapHost[sampleStartIdxs[sampleId]].push_back(std::vector<void*>(m_NumInputs, nullptr)); //3) Constructs the container with count copies of elements with value value.
            //m_SampleAddressMapHost[sampleStartIdxs[sampleId]].push_back(std::vector<u32>(m_NumInputs, 0xdeadbeef)); //3) Constructs the container with count copies of elements with value value.

            for (size_t input_idx = 0; input_idx < m_NumInputs; input_idx++) {
                m_SampleAddressMapHost[sampleStartIdxs[sampleId]].back()[input_idx] = static_cast<int8_t *>((void *)((ssize_t)sampleIndex)); //used to construct kernel driver's sg_list indices
                //m_SampleAddressMapHost[sampleStartIdxs[sampleId]].back()[input_idx] = sampleIndex; //used to construct kernel driver's sg_list indices
            }

        }

    }

    void UnloadSamplesFromRamNeuchips(const std::vector<mlperf::QuerySampleIndex>& samples, std::vector<int>& sampleStartIdxs, int32_t numaIdx, std::vector<int> deviceIDs) {
        LOG(INFO) << "UnloadSamplesFromRamNeuchips samples size " << samples.size() << " numaIdx " << numaIdx;
        for(auto id : deviceIDs)
        {
            LOG(INFO) << "UnloadSamplesFromRamNeuchips device id " << id;
            ncs_preload_uninit(id);
        }
        LOG(INFO) << "UnloadSamplesFromRamNeuchips  m_SampleAddressMapHost.clear()+";
        m_SampleAddressMapHost.clear();
        LOG(INFO) << "UnloadSamplesFromRamNeuchips  m_SampleAddressMapHost.clear()-";
    }

    virtual void* GetSampleAddress(mlperf::QuerySampleIndex sample_index, size_t input_idx,
                                   size_t device_idx = 0) override
    {
        {
            //neuchips+[
            auto it = m_SampleAddressMapHost.find(sample_index);

            CHECK(it != m_SampleAddressMapHost.end())
                << "Sample: " << sample_index << " missing from RAM";

            //LOG(INFO) << "qsl GetSampleAddress m_SampleAddressMapHost.find(" << sample_index << "):" << it->second.front()[input_idx];
            CHECK(input_idx <= it->second.front().size()) << "invalid input_idx";

            return it->second.front()[input_idx];

        }

    }

    virtual size_t GetSampleSize(size_t input_idx) const override
    {
        return (m_SampleSizes.empty() ? 0 : m_SampleSizes[input_idx]);
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
    std::vector<size_t> m_SampleSizes;
    std::vector<std::vector<void *>> m_SampleMemory;
    std::vector<std::unique_ptr<npy::NpyFile>> m_NpyFiles;
    size_t m_SampleCount{0};
    // maps sampleId to <fileName, label>
    std::map<mlperf::QuerySampleIndex, std::tuple<std::string, size_t>> m_FileLabelMap;
    // maps sampleId to num_inputs of <address>
  
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<void *>>> m_SampleAddressMapHost;

    // maps sampleId to num_inputs of num_devices of <address>
    //std::map<mlperf::QuerySampleIndex, std::vector<std::vector<std::vector<void *>>>> m_SampleAddressMapDevice;

    //neuchips+[ moved to dlrm_qsl class
    // maps sampleId to num_inputs of neuchips kernel driver's dma linked-list table
    std::map<mlperf::QuerySampleIndex, std::vector<std::vector<size_t>>> m_SampleAddressMapKernelDriver;
    std::set<size_t> m_SampleAddressSetKernelDriver;
    u32 m_1st_sample_size;
    //neuchips-]
  };

  typedef std::shared_ptr<qsl::SampleLibrary> SampleLibraryPtr_t;

  class SampleLibraryEnsemble : public mlperf::QuerySampleLibrary {
  public:
    SampleLibraryEnsemble(const std::vector<SampleLibraryPtr_t> qsls) : m_qsls(qsls) {};
    const std::string &Name() const /*override*/ { return m_qsls[0]->Name(); }
    size_t TotalSampleCount() override { return m_qsls[0]->TotalSampleCount(); }
    size_t PerformanceSampleCount() override { return m_qsls[0]->PerformanceSampleCount(); }

    void LoadSamplesToRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
      int i = 0;
      for (auto qsl : m_qsls) {
        LOG(INFO) << "SampleLibraryEnsemble::LoadSamplesToRam qsl[" << i++ << "] qsl->LoadSamplesToRam";
        qsl->LoadSamplesToRam(samples);
      }
    }
    void UnloadSamplesFromRam(const std::vector<mlperf::QuerySampleIndex>& samples) override {
        int i = 0;
      for (auto qsl : m_qsls) {
        LOG(INFO) << "SampleLibraryEnsemble::UnloadSamplesFromRam qsl[" << i++ << "] qsl->UnloadSamplesFromRam";
        qsl->UnloadSamplesFromRam(samples);
      }
    }

  private:
    std::vector<SampleLibraryPtr_t> m_qsls;
  };

};

#endif // __QSL_HPP__
