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

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <runtime/config.hpp>
#include <runtime/context.hpp>
#include <runtime/data_type.hpp>
#include <runtime/microkernel/cpu/microkernel.hpp>
#include <util/assert.hpp>

using namespace sc;
typedef sc::sc_data_etype sc_dtype;

extern "C" {

static int sc_brgemm_update_fp32(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b) {
    float *Abuf = (float *)A;
    float *Bbuf = (float *)B;
    float *Cbuf = (float *)C;
    for (int i = 0; i < num; i++) {
        // a is MxK
        for (int m = 0; m < M; m++) {
            for (int n = 0; n < N; n++) {
                for (int k = 0; k < K; k++) {
                    Cbuf[m * LDC + n] += Abuf[m * LDA + k] * Bbuf[k * LDB + n];
                }
            }
        }
        Abuf += stride_a;
        Bbuf += stride_b;
    }
    return 0;
}

SC_API int sc_brgemm_init_update(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, const void *attrs, char *bd_mask,
        const void *op_setting, const void *postop_data, void *c_buf,
        sc::runtime::stream_t *stream) {
    COMPILE_ASSERT(sc_dtype(dtypeA) == sc_data_etype::F32
                    && sc_dtype(dtypeB) == sc_data_etype::F32,
            "Only implemeted fp32");
    dnnl_brgemm_init(C, M, N, LDC, dtypeA, 0);
    sc_brgemm_update_fp32(
            A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a, stride_b);
    return 0;
}

SC_API int sc_brgemm_update(const void *A, const void *B, void *C, int num,
        int M, int N, int K, int LDA, int LDB, int LDC, int stride_a,
        int stride_b, int dtypeA, int dtypeB, const void *attrs, char *bd_mask,
        const void *op_setting, const void *postop_data, void *c_buf,
        sc::runtime::stream_t *stream) {
    COMPILE_ASSERT(sc_dtype(dtypeA) == sc_data_etype::F32
                    && sc_dtype(dtypeB) == sc_data_etype::F32,
            "Only implemeted fp32");
    sc_brgemm_update_fp32(
            A, B, C, num, M, N, K, LDA, LDB, LDC, stride_a, stride_b);
    return 0;
}

SC_API int sc_brgemm_list_update(const void **A_list, const void **B_list,
        void *C, int num, int M, int N, int K, int LDA, int LDB, int LDC,
        int stride_a, int stride_b, int len, int dtypeA, int dtypeB,
        const void *attrs, char *bd_mask, const void *op_setting,
        const void *postop_data, void *c_buf, sc::runtime::stream_t *stream) {
    COMPILE_ASSERT(sc_dtype(dtypeA) == sc_data_etype::F32
                    && sc_dtype(dtypeB) == sc_data_etype::F32,
            "Only implemeted fp32");
    for (int i = 0; i < len; ++i) {
        sc_brgemm_update_fp32(A_list[i], B_list[i], C, num, M, N, K, LDA, LDB,
                LDC, stride_a, stride_b);
    }
    return 0;
}
}
