#include <runtime/kernel_include/cpu_include.hpp>
#include <runtime/config.hpp>

extern "C"
{
    void *sc_aligned_malloc(void *stream, uint64_t size);
    void sc_aligned_free(void *stream, void *ptr);
    void sc_parallel_call_cpu_with_env(void *func, void *stream, int8_t *env,
                                       uint64_t begin, uint64_t end, uint64_t step, generic_val *args);

    void *sc_thread_aligned_malloc(void *stream, uint64_t size);
    int32_t sc_brgemm_init_update(void *A, void *B, void *C, int32_t num, int32_t M,
                                  int32_t N, int32_t K, int32_t LDA, int32_t LDB, int32_t LDC,
                                  int32_t stride_a, int32_t stride_b, int32_t dtypeA, int32_t dtypeB,
                                  void *brg_attrs, void *bd_mask, void *postops_setting,
                                  void *postops_data, void *c_buf, void *stream);
    void sc_thread_aligned_free(void *stream, void *ptr);

    void dnnl_brgemm_list_call(void *brg_desc, void *A_list, void *B_list, void *C, int len, int num,
                               int stride_a, int stride_b, int dtypeA, int dtypeB, void *stream);
    void dnnl_brgemm_list_call_postops(void *brg_desc, void *A_list, void *B_list, void *C, int len, int num,
                                       int stride_a, int stride_b, int dtypeA, int dtypeB,
                                       void *postops_data, void *c_buf, void *stream);
    void *dnnl_brgemm_func(int M, int N, int K, int LDA, int LDB, int LDC,
                           int stride_a, int stride_b, float beta, int dtypeA, int dtypeB,
                           const void *brg_attrs, void *bd_mask, const void *postops_setting);
    void dnnl_brgemm_call(void *brg_desc, void *A, void *B, void *C, int num, void *stream);
    void dnnl_brgemm_call_postops(void *brg_desc, void *A, void *B, void *C, int num,
                                  void *postops_data, void *c_buf, void *stream);
    void dnnl_brgemm_postops_data_init(void *dnnl_data, void *bias, void *scales,
                                       void *binary_post_ops_rhs, uint64_t oc_logical_off,
                                       uint64_t dst_row_logical_off, void *data_C_ptr_,
                                       uint64_t first_mb_matrix_addr_off, void *a_zp_compensations,
                                       void *b_zp_compensations, void *c_zp_values, bool skip_accumulation);
}

namespace sc
{
    namespace runtime
    {
        extern char default_stream;
    }
} // namespace sc
