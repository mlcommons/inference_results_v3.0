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

#include "dlrm_server.h"

#include "dlrm_qsl.hpp"
#include "glog/logging.h"
#include "loadgen.h"

#include <chrono>
#include <fstream>
#include <set>

bool operator==(const ncsinfer1::Dims& d1, const ncsinfer1 ::Dims& d2)
{
    if (d1.nbDims != d2.nbDims)
        return false;
    for (int it = 0; it < d1.nbDims; it++) {
        if (d1.d[it] != d2.d[it])
            return false;
    }
    return true;
}

size_t GetEffectiveBatchSize(const std::vector<DLRMTask>& tasks) {
    return std::accumulate(tasks.begin(), tasks.end(), 0ULL,
        [] (const size_t curr, const DLRMTask& t) {
            return curr + t.numIndividualPairs;
        });
}

#define NUMERIC_VOLUME_SIZE (13)
#define CATEGORY_VOLUME_SIZE (26)

DLRMCore::DLRMCore(int maxBatchSize, int numBundles,
    int numCompleteThreads, int profileIdx)
    : mMaxBatchSize(maxBatchSize)
    , mResHandlerPool(numCompleteThreads)
    , mBundleCounter(0)
    , mNewStartPingPong(true)
    , mStageNo(0)
    , m_1st_sample_size(0) {

    SetBatchSize(maxBatchSize);
    size_t numBindings = 3;
    size_t firstBinding = 0;

    mNumInVol = NUMERIC_VOLUME_SIZE;
    mCatInVol = CATEGORY_VOLUME_SIZE;
    LOG(INFO) << "DLRMCore::DLRMCore mNumInVol " << mNumInVol
              << " mCatInVol " << mCatInVol;
    mOutVol = 8;

    mBindings.resize(numBundles);
    for (int i = 0; i < numBundles; ++i) {
        LOG(INFO) << "DLRMCore::DLRMCore init DLRMEventBufferBundle " << i;
        auto bundle = std::make_shared<DLRMEventBufferBundle>(i, mNumInVol,
                            mCatInVol, mOutVol, mMaxBatchSize);
        mEventBufferBundle.push_back(bundle);

        // set this profile's bindings to DevicePtrs, set rest to nullptr
        mBindings[i].assign(numBindings, nullptr);
        mBindings[i][firstBinding + 0] =
                        bundle->numericInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 1] =
                        bundle->categoricalInputBuf.GetDevicePtr();
        mBindings[i][firstBinding + 2] =
                        bundle->outputBuf.GetDevicePtr();
    }

    LOG(INFO) << "Created copy streams and buffers";
    LOG(INFO) << "DLRMCore::DLRMCore Setup complete";
}

inline void DLRMCore::h2d_renew_sgl_init(
    struct ncs_arg_h2d_renew_sgl* arg_h2d_renew_sgl,
    u32 chan_idx,
    Batch* batch,
    bool useDummyIndex,
    size_t batchSize) {
    size_t cur_size = batchSize;

    DLOG(INFO) << "h2d_renew_sgl_init: chan_idx=" << chan_idx 
                << " batchSize " << batchSize;

    arg_h2d_renew_sgl->chan_idx     = chan_idx;
    if (ALIGN_PING_IDX == chan_idx || ALIGN_PONG_IDX == chan_idx) {
        arg_h2d_renew_sgl->nents        = batch->getH2DIndex();
        arg_h2d_renew_sgl->h2d_indices  = batch->getH2DIndices();
    } else {
        if (!useDummyIndex) {
            DLOG(INFO) << "*h2d_renew_sgl_init EMB getH2DIndex "
                       << batch->getH2DIndex();
            arg_h2d_renew_sgl->nents        = batch->getH2DIndex();
            arg_h2d_renew_sgl->h2d_indices  = batch->getH2DIndices();
            // fill dummy EMB indices to tail
            for (int i = arg_h2d_renew_sgl->nents;
                    i < (N3000_MAX_BATCH_SIZE/100); i++)
            {
                if (cur_size >= N3000_MAX_BATCH_SIZE) {
                    DLOG(INFO) << "break h2d_renew_sgl_init chan_idx="
                               << chan_idx << ", h2d_indices["
                               << i << "] batchSzie " << batchSize
                               << " nents " << arg_h2d_renew_sgl->nents;
                    break;
                }
                arg_h2d_renew_sgl->nents++;
                arg_h2d_renew_sgl->h2d_indices[i] = 0;
                cur_size += get_1st_sample_size();
                DLOG(INFO) << "append h2d_renew_sgl_init m_1st_sample_size "
                           << get_1st_sample_size() << " chan_idx="
                           << chan_idx << ", h2d_indices[" << i << "]"
                           << " cur_size "  << cur_size;
            }
        }
    }

    if (useDummyIndex) {
        arg_h2d_renew_sgl->nents =
            (N3000_MAX_BATCH_SIZE + get_1st_sample_size() - 1) /
            get_1st_sample_size();
        for (int i = 0; i < arg_h2d_renew_sgl->nents; i++) {
            arg_h2d_renew_sgl->h2d_indices[i] = 0;
        }
    }
}

void DLRMCore::SetBatchSize(int batchSize)
{
}

#define MAX_PATH_LEN 128
int DLRMCore::ncs_setup_output_ping_pong_addr(u32 device_id, int fd)
{
        struct neuchip_ioctl_arg ioctl_data;
        struct dma_mmap_param mmap_param;

        LOG(INFO) << "ncs_setup_output_ping_pong_addr: device_id << "
                  << device_id;
        {
            // read chan setting
            if (NUM_INF_PER_PIPE_STAGE == 192) {
                ioctl_data.dma_buf_len  = (NUM_INF_PER_PIPE_STAGE * 1010);
            } else {
                ioctl_data.dma_buf_len  = (NUM_INF_PER_PIPE_STAGE * 760);
            }

            ioctl_data.dma_chan     = 0;
            ioctl_data.rd_buf_ofs   = DLRM_OUTPUT_ADDR_0;
            ioctl_data.rd_buf       =
                (unsigned char*) malloc(ioctl_data.dma_buf_len * sizeof(char));

            DLOG(INFO) <<
                "ncs_setup_output_ping_pong_addr IOCX_DLRM_READ_CH_SET";
            if (ioctl(fd, IOCX_DLRM_READ_CH_SET, &ioctl_data ))
                printf("Failed IOCX_READ_CH_SET, fd = %d, error: %s\n",
                    fd, strerror(errno));

            DLOG(INFO) <<
                "ncs_setup_output_ping_pong_addr IOCX_DLRM_READ_CH_SET";
            ioctl_data.dma_chan     = 1;
            ioctl_data.rd_buf_ofs   = DLRM_OUTPUT_ADDR_1;
            if (ioctl(fd, IOCX_DLRM_READ_CH_SET, &ioctl_data ))
                printf("Failed IOCX_READ_CH_SET, fd = %d, error: %s\n",
                        fd, strerror(errno));

            // ----  get read chan mmap address and mmap  ----//
            mmap_param.dma_buf_idx = 0;
            mmap_param.dma_chan    = 0;
            mmap_param.dma_mmap_en = 1;
            if (NUM_INF_PER_PIPE_STAGE == 192) {
                mmap_param.mmap_size   = (NUM_INF_PER_PIPE_STAGE * 1010);
            } else {
                mmap_param.mmap_size   = (NUM_INF_PER_PIPE_STAGE * 760);
            }
            mmap_param.rd_wr = READ_EP;

            DLOG(INFO) <<
                "ncs_setup_output_ping_pong_addr IOCX_GET_DMA_BUF_MMAP_ADDR";
            if (ioctl(fd, IOCX_GET_DMA_BUF_MMAP_ADDR, &mmap_param ))
                printf("Failed IOCX_DMA_BUFFER_MMAP, fd = %d, error: %s\n",
                    fd, strerror(errno));

            unsigned int mapped_size = mmap_param.mmap_size;

            mN3000PingOutAddr = mmap(NULL, mapped_size,
                                       PROT_READ | PROT_WRITE,
                                       MAP_SHARED, fd,
                                       0);

            LOG(INFO) << "ncs_setup_output_ping_pong_addr mmapped_size "
                      << mapped_size << " n3000_ping_out_addr "
                      << reinterpret_cast<void *>(mN3000PingOutAddr);

            mmap_param.dma_buf_idx = 0;
            mmap_param.dma_chan    = 1;
            mmap_param.dma_mmap_en = 1;
            if (NUM_INF_PER_PIPE_STAGE == 192) {
                mmap_param.mmap_size   = (NUM_INF_PER_PIPE_STAGE * 1010);
            } else {
                mmap_param.mmap_size   = (NUM_INF_PER_PIPE_STAGE * 760);
            }
            mmap_param.rd_wr = READ_EP;

            if (ioctl(fd, IOCX_GET_DMA_BUF_MMAP_ADDR, &mmap_param ))
                printf("Failed IOCX_DMA_BUFFER_MMAP, fd = %d, error: %s\n",
                        fd, strerror(errno));

            mN3000PongOutAddr = mmap(NULL,
                                     mapped_size,
                                     PROT_READ | PROT_WRITE,
                                     MAP_SHARED,
                                     fd,
                                     0);
        }

        LOG(INFO) << "ncs_setup_output_ping_pong_addr-";

        return 0;
    }

    int DLRMCore::CalculateStages(int batch_size) {
        int stages = (batch_size + NUM_INF_PER_PIPE_STAGE - 1) /
                                       NUM_INF_PER_PIPE_STAGE;
        if (stages < 16) {
            return 16;
        } else {
            return stages;
        }
    }

void DLRMCore::infer(std::shared_ptr<DLRMEventBufferBundle> ebBundle,
    size_t batchSize,
    std::vector<DLRMTask>& tasks,
    Batch* batch, void (*h2dCallBack)(void*), DLRMInferCallback resultCallback,
    DLRMNumericInputType* numericInputPtr,
    DLRMCategoricalInputType* categoricalInputPtr,
    int fdN3000, int deviceId, int profileIdx) {
    struct ncs_arg_h2d_renew_sgl arg_h2d_renew_sgl;
    int is_final_send = 0;

    set_current_thread_infering(ebBundle->idx, true);

    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
            << profileIdx << "): infer thread("
            << ebBundle->idx << ") *1 batchSize = "
            << batchSize << " mMaxBatchSize" << mMaxBatchSize;

    bool oddBatch = batch ? batch->isOddBatch() : false;
    bool emptyPipeStage = false;
    bool lastPipeStage = false;

    // Copy buffers, calls IOCX_H2D_RENEW_SGL & IOCX_SG_WRITE_DATA
    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
            << profileIdx << "): infer thread(" << ebBundle->idx
            << ") *2 mNewStartPingPong " << mNewStartPingPong;

    if (1 == ebBundle->idx && mNewStartPingPong) {
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << "): infer thread("
                << ebBundle->idx << ") *3 workaround GET_RESULT pong dummy";
        struct neuchip_ioctl_arg ioctl_arg;
        ioctl_arg.ping_pong_sel = ebBundle->idx;

        // Get output/result
        void* output_ptr;
        if (NUM_INF_PER_PIPE_STAGE == 192) {
            ioctl_arg.dma_buf_len = NUM_INF_PER_PIPE_STAGE * 1010;
        } else {
            ioctl_arg.dma_buf_len = NUM_INF_PER_PIPE_STAGE * 760;
        }

        output_ptr = ioctl_arg.ping_pong_sel ?
                        mN3000PongOutAddr : mN3000PingOutAddr;
        ioctl_arg.dma_chan = ebBundle->idx;
        ioctl_arg.rd_buf_ofs = DLRM_OUTPUT_ADDR_1;

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << ") *4 workaround IOCX_GET_RESULT start";
        if (ioctl(fdN3000, IOCX_GET_RESULT, &ioctl_arg)) {
            LOG(INFO) << "ProcessTasks ERROR: infer thread " << ebBundle->idx
                        << " ERROR: IOCX_GET_RESULT failed";
            set_current_thread_infering(ebBundle->idx, false);
            return;
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << ") *4 workaround IOCX_GET_RESULT done";
        mNewStartPingPong = false;
    }

    if (1 == batch->getTasks().size()) {
        auto lastTask = batch->getTasks().back();

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer(){" << ebBundle->idx
                    << " batchSize " << batchSize
                    << "} *5 batch's lastTask #" << lastTask.dbgSeq
                    << " num " << lastTask.numIndividualPairs
                    << " skipSamples " << lastTask.skipSamples;

        if (lastTask.numIndividualPairs == N3000_MAX_BATCH_SIZE) {
            lastPipeStage = true;
            emptyPipeStage = true;
            DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                        << profileIdx << "): **infer thread(" << ebBundle->idx
                        << ") lastPipeStage is true";
        }
    }

    int ping_pong_sel = ebBundle->idx;

    {
        int align_idx;
        int emb_top_idx;
        int emb_bottom_idx;

        if (0 == ping_pong_sel) {
            align_idx = ALIGN_PING_IDX;
            emb_top_idx = EMB_TOP_PING_IDX;
            emb_bottom_idx = EMB_BOTTOM_PING_IDX;
        } else {
            align_idx = ALIGN_PONG_IDX;
            emb_top_idx = EMB_TOP_PONG_IDX;
            emb_bottom_idx = EMB_BOTTOM_PONG_IDX;
        }

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << ") *6 ping_pong_sel:" << ping_pong_sel
                    << " h2d_renew_sgl_init";

        h2d_renew_sgl_init(&arg_h2d_renew_sgl, align_idx, batch,
                            emptyPipeStage, batchSize);
        if (ioctl(fdN3000, IOCX_H2D_RENEW_SGL_ALIGNER, &arg_h2d_renew_sgl)) {
            LOG(INFO) << "ERROR: IOCX_H2D_RENEW_SGL_ALIGNER Failed to CHAN"
                      << align_idx;
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << ") *7 ping_pong_sel:" << ping_pong_sel
                    << " IOCX_H2D_RENEW_SGL_ALIGNER done";

        h2d_renew_sgl_init(&arg_h2d_renew_sgl, emb_top_idx, batch,
                            emptyPipeStage, batchSize);
        if (ioctl(fdN3000, IOCX_H2D_RENEW_SGL_EMB, &arg_h2d_renew_sgl)) {
            LOG(INFO) << "ERROR: IOCX_H2D_RENEW_SGL_EMB Failed to CHAN"
                      << emb_top_idx;
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << ") *8 ping_pong_sel:" << ping_pong_sel
                    << " IOCX_H2D_RENEW_SGL_EMB done";

    }

    h2dCallBack(batch);

    // Run inference
    struct neuchip_ioctl_arg ioctl_arg;
    ioctl_arg.ping_pong_sel = ebBundle->idx;
    if (false == lastPipeStage) {
        ioctl_arg.final_send = 0;
        is_final_send = 0;
    } else {
        ioctl_arg.final_send = 1;
        is_final_send = 1;
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << "): final_send " << ioctl_arg.final_send;
        resetStageNo();
        m_1st_sample_size = 0;
    }

    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << "): infer thread("
                << ebBundle->idx << "): *9 IOCX_PREDICT+";
    ioctl_arg.offset = (0xABCD0000 | 0);
    if (ioctl(fdN3000, IOCX_PREDICT, &ioctl_arg)) {
        printf("Failed to IOCX_PREDICT, fd = %d, error: %s\n",
                fdN3000, strerror(errno));
    }
    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): infer thread(" << ebBundle->idx
                    << "): *10 IOCX_PREDICT-";

    // Get output/result
    void* output_ptr;
    if (NUM_INF_PER_PIPE_STAGE == 192) {
        ioctl_arg.dma_buf_len = NUM_INF_PER_PIPE_STAGE * 1010;
    } else {
        ioctl_arg.dma_buf_len = NUM_INF_PER_PIPE_STAGE * 760;
    }

final_send_repeat:

    if (0 == ping_pong_sel) {
        output_ptr = mN3000PingOutAddr;
        ioctl_arg.dma_chan = 0;
        ioctl_arg.rd_buf_ofs = DLRM_OUTPUT_ADDR_0;
    } else {
        output_ptr = mN3000PongOutAddr;
        ioctl_arg.dma_chan = 1;
        ioctl_arg.rd_buf_ofs = DLRM_OUTPUT_ADDR_1;
    }

    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << "): infer thread(" << ebBundle->idx
                << "): *11 IOCX_GET_RESULT+";
    if (ioctl(fdN3000, IOCX_GET_RESULT, &ioctl_arg)) {
        LOG(INFO) << "ERROR: infer thread(" << ebBundle->idx << "): ERROR IOCX_GET_RESULT:"
                    << strerror(errno);
    }
    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << "): infer thread(" << ebBundle->idx
                << "): *12 IOCX_GET_RESULT-";

    batchSize = N3000_MAX_BATCH_SIZE;

#ifdef HANDLE_RESULT_USE_OUTPUT_HOST_PTR
    memcpy(ebBundle->outputBuf.GetHostPtr(), output_ptr, batchSize);
#endif

    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << "): infer thread(" << ebBundle->idx
                << "): *13 neuchips_deferred_result+ batchSize " << batchSize;
#ifdef HANDLE_RESULT_USE_OUTPUT_HOST_PTR
    DLRMDeferredResult* deferredResult =
        new DLRMDeferredResult {batchSize, ebBundle->outputBuf.GetHostPtr(),
#else
    DLRMDeferredResult* deferredResult =
        new DLRMDeferredResult {batchSize, (const DLRMOutputType*)output_ptr,
#endif

        std::move(tasks),
        resultCallback, [=](const DLRMResult& r) {
            mResHandlerPool.Enqueue(r);
        }};

    auto neuchips_deferred_result = [](void* deferredResult,
                    int batchSize, int stage_no, int deviceId,
                    int ping_pong_sel, int is_final_send) -> void {
                    DLRMDeferredResult* res =
                        reinterpret_cast<DLRMDeferredResult*>(deferredResult);
                    DLRMResult r
                        = {std::make_shared<std::vector<DLRMOutputType>>(
                            res->outputs, res->outputs + res->batchSize),
                            std::move(res->tasks), res->callback,
                            batchSize,
                            };
                    res->resultCallback(r);
                    delete res;
                 };

    if (1 == is_final_send) {
        for (int i = 0; i < 1000; i++) {
            if (is_the_other_thread_infering(ping_pong_sel)) {
                LOG(INFO) <<
                "ProcessTasks the_other_thread_infering(), count down " << i;
            } else {
                break;
            }
        }
    }
    neuchips_deferred_result(
        deferredResult, batchSize, getStageNo(),
        deviceId, ping_pong_sel, is_final_send);
    set_current_thread_infering(ebBundle->idx, false);

    if (lastPipeStage) {
        if (ping_pong_sel) {
            ping_pong_sel = 0;
        } else {
            ping_pong_sel = 1;
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
            << profileIdx << "): infer thread(" << ebBundle->idx
            << ") one more time to get the last result, using ping_pong_sel:"
            << ping_pong_sel;

        LOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                  << profileIdx << "): infer thread(" << ebBundle->idx
                  << ") emptyPipeStage(true). set back mNewStartPingPong(true)";
        mNewStartPingPong = true;
        emptyPipeStage = 0;
        lastPipeStage = 0;
        resetStageNo();
    }

    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId
        << "/" << profileIdx << "):  infer thread(" << ebBundle->idx
        << "): *14 neuchips_deferred_result- batchSize " << batchSize;
}

void DLRMCore::inferFromDevice(
    std::shared_ptr<DLRMEventBufferBundle> ebBundle,
    size_t batchSize,
    std::vector<DLRMTask>& tasks,
    DLRMInferCallback resultCallback) {
    // LOG(INFO) << "inferFromDevice() batchSize = " << batchSize;
}

// void DLRMCore::WarmUp(double duration)
// {
//     double elapsed = 0.0;
//     auto tStart = std::chrono::high_resolution_clock::now();
//
//     std::vector<DLRMTask> dummyTasks(mMaxBatchSize, {{0, 0}, 1});
//     std::vector<DLRMNumericInputType> dummyNumIn(mMaxBatchSize * mNumInVol);
//     std::vector<DLRMCategoricalInputType> dummyCatIn(
//                                          mMaxBatchSize * mCatInVol);
//
//     LOG(INFO) << "Running warmup for " << duration << "s.";
//     do {
//         auto bundle = NextForegroundBundle();
//         std::this_thread::sleep_for(std::chrono::nanoseconds(1000));
//         elapsed = std::chrono::duration<float>(
//              std::chrono::high_resolution_clock::now() - tStart).count();
//     } while (elapsed < duration);
//
//     LOG(INFO) << "Running warmup #";
//     for (size_t i = 0; i < mEventBufferBundle.size(); ++i) {
//         LOG(INFO) << "Running warmup NextForegroundBundle()->syncD2H()";
//         NextForegroundBundle()->syncD2H();
//     }
//     LOG(INFO) << "Warmup complete, ran for " << elapsed << "s.";
// }

std::shared_ptr<DLRMEventBufferBundle> DLRMCore::NextForegroundBundle() {
    size_t idx = mBundleCounter;
    mBundleCounter = (mBundleCounter + 1) % mEventBufferBundle.size();
    return mEventBufferBundle[idx];
}

std::shared_ptr<DLRMEventBufferBundle> DLRMCore::getBundleByIdx(size_t idx) {
    return mEventBufferBundle[idx];
}

DLRMCore::~DLRMCore() {
}

DLRMServer::DLRMServer(const std::string name,
                       const std::string enginePath,
                       std::vector<DLRMSampleLibraryPtr_t> qsls,
                       const std::vector<int>& gpus,
                       int maxBatchSize,
                       int numBundles,
                       int numCompleteThreads,
                       int numDLRMCores,
                       double warmupDuration,
                       int numStagingThreads,
                       int numStagingBatches,
                       int maxPairsPerThread,
                       int splitThreshold,
                       bool checkContiguity,
                       bool startFromDevice,
                       NumaConfig numaConfig,
                       const std::string scenario,
                       int num_issue_query_threads)
    : mName{name}
    , mQsls{qsls}
    , mStartFromDevice(startFromDevice)
    , mStopWork{false}
    , mDLRMCores{gpus.size() * numDLRMCores}
    , mNumInVol(0)
    , mCatInVol(0)
    , mSplitThreshold(splitThreshold)
    , mNumaConfig(numaConfig)
    , mGpuToNumaMap(getGpuToNumaMap(mNumaConfig)) {
    LOG(INFO) << "Using " << numDLRMCores << " DLRM Core(s)";

    if (UseNuma()) {
        LOG(INFO) << "Using NUMA nodes";
        CHECK(mNumaConfig.size() == qsls.size())
            << "Number of QSLs should match number of NUMA nodes!";
    }

    mMaxBatchSize = maxBatchSize;
    // Enforce that max batch size is even due to Top MLP plugin
    if (mMaxBatchSize % 2 == 1) {
        mMaxBatchSize = mMaxBatchSize - 1;
    }

    if (0 == scenario.compare("Server")) {
        LOG(INFO) << "BatchMaker scenario is Server.";
        mServerMode = true;
    } else {
        mServerMode = false;
    }

    std::vector<std::thread> setupThreads;
    for (const auto& deviceId : gpus) {
        setupThreads.emplace_back(&DLRMServer::SetupDevice,
                this, enginePath, numBundles, numCompleteThreads,
                numDLRMCores, warmupDuration, deviceId);
    }

    for (auto& t : setupThreads) {
        t.join();
    }

    if (!startFromDevice) {
        LOG(INFO) << "DLRMServer !startFromDevice" << "DLRMServer mNumInVol "
                  << mNumInVol << "DLRMServer mCatInVol " << mCatInVol;
        int numBatchMakers = UseNuma() ? mNumaConfig.size() : 1;
        for (int i = 0; i < numBatchMakers; ++i) {
            // Construct BatchMaker
            if (UseNuma()) {
                LOG(INFO) << "DLRMServer Construct BatchMaker: NUM_NUMA_ZONES "
                    << NUM_NUMA_ZONES << "deviceId " << mNumaConfig[i].first[0];
            }
            mBatchMakers.emplace_back(std::make_shared<BatchMaker>(
                /* numStagingThreads = */ numStagingThreads,
                /* numBatches = */ numStagingBatches,
                /* maxBatchSize = */ maxBatchSize,
                /* maxPairsPerThread = */ maxPairsPerThread,
                /* numericVolume = */ mNumInVol,
                /* categoricalVolume = */ mCatInVol,
                /* checkContiguity = */ checkContiguity,
                /* qsl = */ UseNuma() ? qsls[i] : qsls[0],
                /* numaIdx = */ UseNuma() ? mNumaConfig[i].first[0] : -1,
                /* numaNum = */ UseNuma() ? NUM_NUMA_ZONES : 0,
                /* cpus = */ UseNuma() ?
                                mNumaConfig[i].second : std::vector<int>(),
                /* scenario = */ scenario));
        }
    }

    if (mServerMode) {
        if (num_issue_query_threads) {
            LOG(INFO) << "Start " << num_issue_query_threads
                      << " IssueQuery() threads";
            for (int i = 0; i < num_issue_query_threads; ++i) {
                LOG(INFO) << "StartIssueThread()"
                          << num_issue_query_threads << ") created.";
                mIssueQueryThreads.emplace_back(
                    &DLRMServer::StartIssueThread, this, i);
            }
        } else {
            LOG(INFO) << "Start IssueQuery() in main thread";
            mThreadMap[std::this_thread::get_id()] = 0;
        }
    }

    int current_count = 0;
    for (int32_t numaIdx = 0; numaIdx < mNumaConfig.size(); numaIdx++) {
        for (const auto deviceId : mNumaConfig[numaIdx].first) {
            // Bind the worker to the NUMA memory if needed
            if (UseNuma()) {
                LOG(INFO) << "bindNumaMemPolicy: mGpuToNumaMap["
                        << deviceId << "]=" << mGpuToNumaMap[deviceId];
                bindNumaMemPolicy(deviceId, NUM_NUMA_ZONES);
            }
            mWorkerThreads.reserve(numDLRMCores);
            LOG(INFO) << "DLRMServer::DLRMServer numDLRMCores " << numDLRMCores;
            for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx) {
                auto dlrmCore =
                    mDLRMCores[current_count * numDLRMCores + profileIdx];
                LOG(INFO) << "deviceId " << deviceId
                          << ", profileIdx " << profileIdx
                          << " , startFromDevice " << startFromDevice;
                mWorkerThreads.emplace_back(
                    &DLRMServer::ProcessTasks, this, dlrmCore,
                    deviceId, profileIdx);

                // Limit the worker thread to the closest CPUs.
                if (UseNuma()) {
                    bindThreadToCpus(mWorkerThreads.back(),
                        mNumaConfig[mGpuToNumaMap[deviceId]].second);
                }

                LOG(INFO) << "deviceId " << deviceId << ", profileIdx "
                    << (profileIdx + 1)
                    << " , startFromDevice " << startFromDevice;
                mWorkerThreads.emplace_back(
                    &DLRMServer::ProcessTasks, this, dlrmCore,
                    deviceId, profileIdx+1);

                // Limit the worker thread to the closest CPUs.
                if (UseNuma()) {
                    bindThreadToCpus(mWorkerThreads.back(),
                        mNumaConfig[mGpuToNumaMap[deviceId]].second);
                }
            }

            current_count++;
        }
    }
    // Reset memory allocation setting
    if (UseNuma()) {
        resetNumaMemPolicy();
    }
}

void DLRMServer::SetupDevice(const std::string enginePath,
    int numBundles, int numCompleteThreads, int numDLRMCores,
    int warmupDuration, int deviceId) {
    LOG(INFO) << "DLRMServer::SetupDevice " << enginePath
        << " numBundles " << numBundles
        << " numCompleteThreads " << numCompleteThreads
        << " numDLRMCores " << numDLRMCores
        << " warmupDuration " << warmupDuration << " deviceId " << deviceId;

    mNumInVol = NUMERIC_VOLUME_SIZE;
    mCatInVol = CATEGORY_VOLUME_SIZE;
    LOG(INFO) << "DLRMServer::SetupDevice mNumInVol " << mNumInVol
              << " mCatInVol " << mCatInVol;
    for (size_t profileIdx = 0; profileIdx < numDLRMCores; ++profileIdx) {
        LOG(INFO) << "DLRMServer::SetupDevice profileIdx " << profileIdx;
        auto dlrmCore =
            std::make_shared<DLRMCore>(mMaxBatchSize, numBundles,
                                        numCompleteThreads, profileIdx);
        mDLRMCores[deviceId * numDLRMCores + profileIdx] = dlrmCore;
        CHECK_LE(mMaxBatchSize, dlrmCore->GetMaxBatchSize());
    }
}

DLRMServer::~DLRMServer()
{
    LOG(INFO) << "~DLRMServer";

    DLOG(INFO) << "~DLRMServer";
    {
        std::unique_lock<std::mutex> lck(mMtx);
        mStopWork = true;
        mCondVar.notify_all();
    }
    for (auto batchMaker : mBatchMakers) {
        if (batchMaker) {
            batchMaker->StopWork();
        }
    }

    for (auto& workerThread : mWorkerThreads) {
        workerThread.join();
    }

    if (mServerMode) {
        for (auto& thread : mIssueQueryThreads) {
            thread.join();
        }
    }
}

const std::string& DLRMServer::Name() {
    return mName;
}

void DLRMServer::StartIssueThread(int threadIdx) {
    LOG(INFO) << "StartIssueThread:" << threadIdx;
    {
        std::lock_guard<std::mutex> lock(mMtxIssueQuery);
        mThreadMap[std::this_thread::get_id()] = threadIdx;
    }

    LOG(INFO) << "StartIssueThread:" << threadIdx 
              << " RegisterIssueQueryThread";
    mlperf::RegisterIssueQueryThread();
}

void DLRMServer::IssueQueryServer(
    const std::vector<mlperf::QuerySample>& samples) {
    int threadIdx;
    const int64_t numSamplesTotal = samples.size();

    {
        std::unique_lock<std::mutex> lock(mMtxIssueQuery);
        threadIdx = mThreadMap[std::this_thread::get_id()];
        lock.unlock();
    }

    {
        int nextBatchMakerIdx = UseNuma() ? threadIdx : 0;
        DLOG(INFO) << "IQ[" << nextBatchMakerIdx << "] " << numSamplesTotal;
        mBatchMakers[nextBatchMakerIdx]->IssueQuery(
                                            samples, 0, numSamplesTotal);
    }
}

void DLRMServer::IssueQueryOffline(
                    const std::vector<mlperf::QuerySample>& samples) {
    {
        const int64_t numSamplesTotal = samples.size();

        DLOG(INFO) << "DLRMServer::IssueQuery numSamplesTotal "
                    << numSamplesTotal
                    << " mSplitThreshold " << mSplitThreshold;
        // If num sample is too small, we don't want to round robin across batchMakers.
        if (UseNuma() && numSamplesTotal > mSplitThreshold) {
            const int64_t numBatchMakers = mNumaConfig.size();
            const int64_t numSamplesPerNode = numSamplesTotal / numBatchMakers;
            const int64_t remainder = numSamplesTotal % numBatchMakers;

            // Use a thread per batchMaker to issue queries in parallel. Each batchMaker has its own lock, so running
            // with multiple threads should not cause lock contention.
            auto issueQueryOneBatchMaker
                = [](std::shared_ptr<BatchMaker> batchMaker, const std::vector<mlperf::QuerySample>& samples,
                      const int64_t offset, const int64_t size) { batchMaker->IssueQuery(samples, offset, size); };
            std::vector<std::thread> issueThreads;
            issueThreads.reserve(numBatchMakers);

            int64_t offset{0};
            uint64_t startIdx{(mPrevBatchMakerIdx + 1) % mNumaConfig.size()};
            for (int64_t myIdx = 0; myIdx < numBatchMakers; ++myIdx) {
                int64_t numaIdx = (startIdx + myIdx) % mNumaConfig.size();
                int64_t size = numSamplesPerNode + (numaIdx < remainder ? 1 : 0);
                if (size == 0) {
                    break;
                }

                DLOG(INFO) << "issueQueryOneBatchMaker " << numaIdx << "/" << myIdx;
                issueThreads.emplace_back(
                    issueQueryOneBatchMaker, mBatchMakers[numaIdx], std::ref(samples), offset, size);
                // Might be the thread so lightweight, it is already done before attempting the NUMA binding,
                // unfortunately In this case, bind may trap the ESRCH upon pthread_attr_t, so let us ignore the ESRCH
                // here
                bindThreadToCpus(issueThreads.back(), mNumaConfig[numaIdx].second, true);

                offset += size;
            }
            CHECK(offset == numSamplesTotal);

            for (auto& th : issueThreads) {
                th.join();
                DLOG(INFO) << "issueQueryOneBatchMaker join() finish.";
            }
            mPrevBatchMakerIdx = startIdx;
        } else {
            int nextBatchMakerIdx = UseNuma() ?
                        (mPrevBatchMakerIdx + 1) % mNumaConfig.size() : 0;

            mPrevBatchMakerIdx = nextBatchMakerIdx;
            DLOG(INFO)
                << "DLRMServer::IssueQuery mBatchMakers[nextBatchMakerIdx "
                << nextBatchMakerIdx << "]->IssueQuery mPrevBatchMakerIdx "
                << mPrevBatchMakerIdx;
            mBatchMakers[nextBatchMakerIdx]->IssueQuery(
                                            samples, 0, numSamplesTotal);
        }
    }
}

void DLRMServer::IssueQuery(const std::vector<mlperf::QuerySample>& samples) {
    if (mServerMode) {
        IssueQueryServer(samples);
    } else {
        IssueQueryOffline(samples);
    }
}

void DLRMServer::FlushQueries()
{
    LOG(INFO) << "DLRMServer::FlushQueries";

    if (!mStartFromDevice) {
        for (auto batchMaker : mBatchMakers) {
            batchMaker->FlushQueries();
        }
    }
}

void DLRMServer::ProcessTasks(std::shared_ptr<DLRMCore> dlrmCore,
                                int deviceId, int profileIdx) {
    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                << profileIdx << ")";

    char devname_path[128];
    snprintf(devname_path, MAX_PATH_LEN, "/dev/%s-%d",
                    DRIVER_PATH_NAME, deviceId);
    DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): Opening Driver " << devname_path;

    int mFdN3000 = open(devname_path, O_RDWR);
    if (mFdN3000 < 0) {
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): ERROR Opening Driver " << devname_path;
        return;
    }

    if (0 == dlrmCore->getBundleByIdx(profileIdx)->idx) {
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): ncs_setup_output_ping_pong_addr";
        dlrmCore->ncs_setup_output_ping_pong_addr(deviceId, mFdN3000);
    }

    // Process samples in batches
    while (true) {
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): #1 getBundleByIdx()";
        auto ebBundle = dlrmCore->getBundleByIdx(profileIdx);

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                    << profileIdx << "): #2 wait its own turn ebBundle.idx:"
                    << ebBundle->idx << ", getStageNo():"
                    << dlrmCore->getStageNo();

        while (1) {
            if (ebBundle->idx == (dlrmCore->getStageNo()%2)) {
                DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId
                           << "/" << profileIdx
                           << ") #3 got stageNo ebBundle.idx:" << ebBundle->idx
                           << ", stageNo " << dlrmCore->getStageNo();
                break;
            }

            if (mStopWork) {
                DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                            << profileIdx << ") #3 mStopWork " << mStopWork;
                break;
            }
        }

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") #4 granted its turn ebBundle.idx:"
                   << ebBundle->idx;

        int batchMakerIdx = UseNuma() ? mGpuToNumaMap[deviceId] : 0;

        Batch* batch = mBatchMakers[batchMakerIdx]->GetBatch();
        if (!batch) {
            DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                       << profileIdx << "): !batch";
            break;
        }

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") #5 increaseStageNo";
        dlrmCore->increaseStageNo(); // advance stage no

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") #6 ebBundle->idx " << ebBundle->idx
                   << " get batch: " << batch->mDebugId << " "
                   << "getTasks.size()" << batch->getTasks().size()
                   << ", front dbgSeq " << batch->getTasks().front().dbgSeq
                   << ", back dbgSeq " << batch->getTasks().back().dbgSeq;

        size_t actualBatchSize = batch->getCommittedCopies();
        auto tasks = batch->getTasks();
        auto numericHostPtr = batch->getNumericHostPtr();
        auto categoricalHostPtr = batch->getCategoricalHostPtr();
        bool oddBatch = batch->isOddBatch();

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId  << "/"
                   << profileIdx << ") #7 Batch Size : " << actualBatchSize;

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") #8 infer()+";

        dlrmCore->infer(
            ebBundle, actualBatchSize,
            tasks,
            batch,
            [](void* batch) -> void {
                reinterpret_cast<Batch*>(batch)->mBatchMaker->NotifyH2D(
                    reinterpret_cast<Batch*>(batch)); },
            [=](std::vector<mlperf::QuerySampleResponse>& responses) {
                DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                           << profileIdx << ") resultCallback responses.size() "
                           << responses.size();

            mlperf::QuerySamplesComplete(responses.data(), responses.size());},
            numericHostPtr, categoricalHostPtr, mFdN3000, deviceId, profileIdx);

        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                            << profileIdx << ") #9 infer()- loop end";
    }

    if (0 == dlrmCore->getBundleByIdx(profileIdx)->idx) {
        struct neuchip_ioctl_arg ioctl_arg;
        memset(&ioctl_arg, 0, sizeof(struct neuchip_ioctl_arg));
        ioctl_arg.dma_chan = 0;
        if (ioctl(mFdN3000, IOCX_FREE_DMA_CHANNEL, &ioctl_arg)) {
            LOG(INFO) << "Failed IOCX_FREE_RD_CHANNEL 0";
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") O.K. IOCX_FREE_RD_CHANNEL 0";

        ioctl_arg.dma_chan = 1;
        if (ioctl(mFdN3000, IOCX_FREE_DMA_CHANNEL, &ioctl_arg )) {
            LOG(INFO) << "Failed IOCX_FREE_RD_CHANNEL 1";
        }
        DLOG(INFO) << "DLRMServer::ProcessTasks(" << deviceId << "/"
                   << profileIdx << ") O.K. IOCX_FREE_RD_CHANNEL 1";
    }

    if (mFdN3000) {
        close(mFdN3000);
    }
}

std::vector<DLRMTask> DLRMServer::GetBatch()
{
    std::vector<DLRMTask> res;
    // Wait for the new work to arrive
    std::unique_lock<std::mutex> lck(mMtx);
    mCondVar.wait(lck, [&] { return (!mTasks.empty()) || mStopWork; });

    // Consume up to mMaxBatchSize pairs
    int currentBatchSize = 0;
    while (!mTasks.empty()) {
        const auto& topTask = mTasks.front();
        currentBatchSize += topTask.numIndividualPairs;
        if (currentBatchSize > mMaxBatchSize)
            break;
        res.push_back(topTask);
        mTasks.pop_front();
    }

    // Let some other thread to consume more tasks if this one got any
    if (!res.empty())
        mCondVar.notify_one();

    return res;
}
