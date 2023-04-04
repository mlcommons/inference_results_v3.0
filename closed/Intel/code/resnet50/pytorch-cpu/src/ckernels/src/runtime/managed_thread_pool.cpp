/*******************************************************************************
 * Copyright 2022 Intel Corporation
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <immintrin.h>
#include "config.hpp"
#include "managed_thread_pool.hpp"
#include "memorypool.hpp"
#include "runtime.hpp"
#include "thread_locals.hpp"
#include <cpu/x64/amx_tile_configure.hpp>
#include <util/simple_math.hpp>

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
// clang-format off
#define TBB_PREVIEW_GLOBAL_CONTROL 1
#include <tbb/parallel_for_each.h>
// clang-format on
#endif

using namespace sc;
using sc::runtime::thread_manager;
static void do_dispatch(thread_manager *s, int tid);
namespace sc {
namespace runtime {

void thread_manager::thread_pool_state::wait_all() {
    for (;;) {
        if (remaining.load(std::memory_order_acquire) == 0) { break; }
        _mm_pause();
    }
}

void thread_manager::thread_pool_state::reset_scoreboard() {
    remaining.store(num_threads - 1, std::memory_order_release);
}

thread_manager::thread_manager() {
    state.trigger = 1;
}

static void do_cleanup() {
    auto &tls = thread_local_buffer_t::tls_buffer_;
    tls.in_managed_thread_pool_ = false;
    auto &need_release_amx = tls.amx_buffer_.need_release_tile_;
    if (need_release_amx) {
        dnnl::impl::cpu::x64::amx_tile_release();
        need_release_amx = false;
        // force to re-configure next time
        tls.amx_buffer_.cur_palette = nullptr;
    }
}

static void worker_func(thread_manager *ths, int tid) {
    int st;
    auto &task = ths->state.task;
    int current_job_id = 2;
    // auto prev = std::chrono::high_resolution_clock::now();
    while ((st = ths->state.trigger) != -1) {
        if (st == current_job_id) {
            // printf("DT=%ld\n",
            //         std::chrono::duration_cast<std::chrono::microseconds>(
            //                 std::chrono::high_resolution_clock::now() - prev)
            //                 .count());
            do_dispatch(ths, tid);
            // prev = std::chrono::high_resolution_clock::now();
            --ths->state.remaining;
            current_job_id++;
        }
        _mm_pause();
    }
    do_cleanup();
}

void thread_manager::run_main_function(main_func_t f, runtime::stream_t *stream,
        void *mod_data, generic_val *args) {
    int threads = runtime_config_t::get().get_num_threads();
    state.num_threads = threads;
    if (threads > 1) {
        state.trigger = 1;
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
#pragma omp parallel for
        for (int i = 0; i < threads; i++) {
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        tbb::parallel_for(0, threads, 1, [&](int64_t i) {
#endif
            thread_local_buffer_t::tls_buffer_.in_managed_thread_pool_ = true;
            if (i == 0) {
                f(stream, mod_data, args);
                state.trigger = -1;
                do_cleanup();
            } else {
                worker_func(this, i);
            }
#if SC_CPU_THREADPOOL == SC_THREAD_POOL_OMP
        }
#elif SC_CPU_THREADPOOL == SC_THREAD_POOL_TBB
        });
#endif

#if SC_CPU_THREADPOOL == SC_THREAD_POOL_SEQ
        throw std::runtime_error("Running SEQ in thread pool");
#endif
    } else {
        thread_local_buffer_t::tls_buffer_.in_managed_thread_pool_ = true;
        f(stream, mod_data, args);
        do_cleanup();
    }
}

thread_local thread_manager thread_manager::cur_mgr;
} // namespace runtime
} // namespace sc

// using balance211 to dispatch the workloads
static void do_dispatch(thread_manager *s, int tid) {
    size_t end = s->state.task.end;
    size_t begin = s->state.task.begin;
    size_t step = s->state.task.step;
    size_t len = end - begin;
    size_t num_jobs = utils::divide_and_ceil(len, s->state.task.step);
    size_t my_jobs = utils::divide_and_ceil(num_jobs, s->state.num_threads);
    assert(my_jobs > 0);
    size_t my_jobs_2 = my_jobs - 1;
    size_t the_tid = num_jobs - my_jobs_2 * s->state.num_threads;
    size_t cur_jobs = (size_t)tid < the_tid ? my_jobs : my_jobs_2;
    size_t my_begin = (size_t)tid <= the_tid
            ? tid * my_jobs
            : the_tid * my_jobs + (tid - the_tid) * my_jobs_2;
    my_begin *= step;
    size_t my_end = my_begin + cur_jobs * step;
    for (size_t i = my_begin; i < my_end; i += step) {
        s->state.task.pfunc(s->state.task.stream, s->state.task.module_env, i,
                s->state.task.args);
    }
}

void sc_parallel_call_managed(
        void (*pfunc)(void *, void *, int64_t, sc::generic_val *),
        void *rtl_ctx, void *module_env, int64_t begin, int64_t end,
        int64_t step, sc::generic_val *args) {
    thread_manager *stream = &thread_manager::cur_mgr;
    stream->state.reset_scoreboard();
    stream->state.task = thread_manager::thread_pool_state::task_type {
            pfunc, rtl_ctx, module_env, begin, end, step, args};
    stream->state.trigger++;
    do_dispatch(stream, 0);
    stream->state.wait_all();
}
