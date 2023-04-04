#ifndef MLA_RT_CLIENT_HANDLE_
#define MLA_RT_CLIENT_HANDLE_

#include <inttypes.h>
#include <cstdint>

extern "C" {
#include <simaai/gst-api.h>
}
#include <simaai/simaai_memory.h>

#include "gstsimaaimlfilter.h"

typedef int64_t simamm_buffer_id_t;

#ifdef __cplusplus
extern "C" {
#endif

void mla_rt_client_load_model (const char * path);

int mla_rt_client_run_model (simamm_buffer_id_t buffer_id,
			     uint32_t in_buf_offset,
			     simamm_buffer_id_t out_buf_id,
			     uint32_t out_buf_offset);

int32_t
mla_rt_client_write_output_buffer (const void * data, size_t size, int id);

int32_t
mla_rt_client_write_input_buffer (const void * data, size_t size, int id);
     
int mla_rt_client_init (const char * model_path);

simaai_memory_t * mla_rt_client_prepare_outbuf (size_t output_sz);
     
void mla_rt_client_cleanup(void);

int mla_rt_client_batch_run_model (uint64_t in_addr_list[],
                                   simamm_buffer_id_t out_buf_id,
                                   uint32_t out_buf_offset,
                                   size_t batch_size,
                                   size_t in_sz,
                                   size_t out_sz);
    
#ifdef __cplusplus
}
#endif

#endif // MLA_RT_CLIENT_HANDLE_
