/*
 * Copyright (c) 2023, NEUCHIPS .  All rights reserved.
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

#pragma once

#define NUM_INF_PER_PIPE_STAGE 256

/* TODO: dynamic switch to corresponding settings */
// #define INTEL_GEN5_SERVER 1

#ifdef INTEL_GEN5_SERVER
#define DLRM_OUTPUT_ADDR_0 0x650000000
#define DLRM_OUTPUT_ADDR_1 0x660000000

#define NUMA_NOT_SUPPORTED

#else
#define DLRM_OUTPUT_ADDR_0 0x2050000000
#define DLRM_OUTPUT_ADDR_1 0x2060000000
#endif

#define DRIVER_PATH_NAME "neuchips_ai_epr"

#define STAGE_BATCH_SIZE   750
#define BATCH_SZ_REG 0x18800b0
#define N3000_MAX_BATCH_SIZE (NUM_INF_PER_PIPE_STAGE*STAGE_BATCH_SIZE)

#define NUM_NUMA_ZONES 8

#define HANDLE_RESULT_USE_OUTPUT_HOST_PTR 1
#define USE_MULTI_THREAD_NCS_PRELOAD 1

#define DEBUG_GLOG_INFO


