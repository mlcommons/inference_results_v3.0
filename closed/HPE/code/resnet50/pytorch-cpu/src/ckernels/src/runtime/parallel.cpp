/*******************************************************************************
 * Copyright 2020-2022 Intel Corporation
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
 *******************************************************************************/

#include <memory>
#include "config.hpp"
#include "context.hpp"
#include <runtime/generic_val.hpp>
#include <runtime/parallel.hpp>
#include <util/simple_math.hpp>
#include <omp.h>


// todo: handle signed integers
extern "C" void sc_parallel_call_cpu(void (*pfunc)(int64_t, sc::generic_val *),
        int64_t begin, int64_t end, int64_t step, sc::generic_val *args) {
#pragma omp parallel for
    for (int64_t i = begin; i < end; i += step) {
        pfunc(i, args);
    }
}

#define get_num_threads omp_get_max_threads
#define set_num_threads omp_set_num_threads
#define get_thread_num omp_get_thread_num
#define get_in_parallel omp_in_parallel

// omp or sequential
extern "C" void sc_parallel_call_cpu_with_env_impl(
        void (*pfunc)(void *, void *, int64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, sc::generic_val *args) {
#pragma omp parallel for
    for (int64_t i = begin; i < end; i += step) {
        pfunc(rtl_ctx, module_env, i, args);
    }
}

extern "C" void sc_parallel_call_cpu_with_env(
        void (*pfunc)(void *, void *, int64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, sc::generic_val *args) {
    sc::runtime::stream_t *stream
            = reinterpret_cast<sc::runtime::stream_t *>(rtl_ctx);
    stream->vtable()->parallel_call(
            pfunc, rtl_ctx, module_env, begin, end, step, args);
}

namespace sc {
thread_pool_table sc_pool_table {&sc_parallel_call_cpu_with_env_impl, nullptr,
        &get_num_threads, &set_num_threads, &get_thread_num, &get_in_parallel};
}
