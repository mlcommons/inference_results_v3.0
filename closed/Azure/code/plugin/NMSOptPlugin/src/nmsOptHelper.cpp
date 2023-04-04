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

#include "ssdOpt.h"
#include "ssdOptMacros.h"
#include <cassert>
#include <algorithm>

void reportAssertion(const char* msg, const char* file, int line)
{
    std::ostringstream stream;
    stream << "Assertion failed: " << msg << std::endl
           << file << ':' << line << std::endl
           << "Aborting..." << std::endl;
    getLogger()->log(nvinfer1::ILogger::Severity::kINTERNAL_ERROR, stream.str().c_str());
    cudaDeviceReset();
    abort();
}

namespace nvinfer1
{
namespace plugin
{

size_t detectionForwardBBoxDataSize(int N, int C1, DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return N * C1 * sizeof(float);
    }

    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardBBoxPermuteSize(bool shareLocation, int N, int C1, DType_t DT_BBOX)
{
    if (DT_BBOX == DataType::kFLOAT)
    {
        return shareLocation ? 0 : N * C1 * sizeof(float);
    }
    printf("Only FP32 type bounding boxes are supported.\n");
    return (size_t) -1;
}

size_t detectionForwardPreNMSSize(int N, int C2)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return (size_t) N * (size_t) C2 * sizeof(float);
}

size_t detectionForwardPostNMSSize(int N, int numClasses, int topK)
{
    static_assert(sizeof(float) == sizeof(int), "Must run on a platform where sizeof(int) == sizeof(float)");
    return N * numClasses * topK * sizeof(float);
}

size_t detectionInferenceWorkspaceSize(bool shareLocation, int N, int C1, int C2, int numClasses, int numPredsPerClass,
    int topK, DType_t DT_BBOX, DType_t DT_SCORE)
{
    size_t temp_active_counts_size, temp_sort_scores_size, temp_sort_indicies_size;
    size_t temp_cub_storage_bytes = 0;
    temp_cub_storage_bytes = topKScoresPerClassWorkspaceSize(N, numClasses, numPredsPerClass, topK, DT_SCORE,
        temp_active_counts_size, temp_sort_scores_size, temp_sort_indicies_size);
    temp_cub_storage_bytes
        = temp_cub_storage_bytes - temp_active_counts_size - temp_sort_scores_size - temp_sort_indicies_size;

    size_t wss[12];
    wss[0] = detectionForwardBBoxDataSize(N, C1, DT_BBOX);                         // bboxData
    wss[1] = detectionForwardBBoxPermuteSize(shareLocation, N, C1, DT_BBOX);       // bboxPermute
    wss[2] = std::max(detectionForwardPreNMSSize(N, C2), temp_sort_scores_size);   // scores
    wss[3] = std::max(detectionForwardPreNMSSize(N, C2), temp_sort_scores_size);   // softmaxScores
    wss[4] = std::max(detectionForwardPreNMSSize(N, C2), temp_sort_indicies_size); // indices
    wss[5] = temp_sort_indicies_size;                                              // temp indices
    wss[6] = temp_active_counts_size;                                              // temp active counts
    wss[7] = temp_cub_storage_bytes;
    wss[8] = detectionForwardPostNMSSize(N, numClasses, topK); // postNMSScores
    wss[9] = detectionForwardPostNMSSize(N, numClasses, topK); // postNMSIndices
    wss[10] = N * numClasses * sizeof(int) + N * sizeof(int);  // activeCount, activeCountPerBatch
    wss[11] = 0; // N * numClasses * numPredsPerClass * sizeof(float); //sortingWorkspace
    return calculateTotalWorkspaceSize(wss, 12);
}
} // namespace plugin
} // namespace nvinfer1
