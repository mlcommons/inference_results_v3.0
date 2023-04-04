#include <inttypes.h>
#include <stdio.h>
#include <sys/time.h>

#include "mla_rt_client.h"

extern "C" {
#include <simaai/gst-api.h>
#include <simaai/simaailog.h>
}
#include <simaai/gstsimaaiallocator.h>
#include <simaai/simaai_memory.h>

#include "gstsimaaimlfilter.h"

// #define ML_FILTER_EVAL_PERF

static uint64_t model = 0;
static uint64_t handle = 0;

typedef struct {
  int id;
  uint64_t addr;
} buffer_mapping;

static buffer_mapping mappings[sizeof(uint64_t) * 8] = {{ .id = -1, .addr = 0 }};
static int64_t current_mappings = 0;
static pthread_mutex_t mapping_mutex = PTHREAD_MUTEX_INITIALIZER;

static uint64_t buffer_id_to_paddr(unsigned int id);

/**
 * @brief Mapping buffer id to physical address
 * @param id, input buffer id
 */
static uint64_t buffer_id_to_paddr(unsigned int id)
{
  uint32_t i;
  simaai_memory_t * m;
  uint64_t res = 0;

  pthread_mutex_lock(&mapping_mutex);

  for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
    if((id == mappings[i].id) && (current_mappings & (1 << i))) {
      res = mappings[i].addr;
      break;
    }
  }

  if(res == 0) {
    for(i = 0; i < (sizeof(mappings) / sizeof(mappings[0])); i++) {
      if(!(current_mappings & (1 << i))) {
        m = simaai_memory_attach(id);
        mappings[i].addr = simaai_memory_get_phys(m);
        mappings[i].id = id;
        current_mappings |= 1 << i;
        // simaailog(SIMAAILOG_DEBUG, "Storing new mapping of"
        //           "%#x to %#lx at index %d", mappings[i].id, mappings[i].addr, i);
        res = mappings[i].addr;
        break;
      }
    }
  }

  if(res == 0) {
    simaailog(SIMAAILOG_DEBUG, "Ran out of space to store buffer address,"
              "this may have impact on performance or stability");
    m = simaai_memory_attach(id);
    res = simaai_memory_get_phys(m);
  }

  pthread_mutex_unlock(&mapping_mutex);

  return res;
}

int32_t
mla_rt_client_write_input_buffer (const void * data, size_t size, int id)
{
  FILE *ofp;
  size_t osz = size, wosz;
  void *ovaddr;
  int32_t res = 0, i;
  char *dumpaddr;
  char full_opath[256];

  snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%d.in", "ml_filter", id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL) {
    res = -1;
    goto err_ofp;
  }

  wosz = fwrite(data, 1, osz, ofp);
  if(osz != wosz) {
    res = -3;
    goto err_owrite;
  }

err_owrite:
err_omap:
  fclose(ofp);
err_ofp:
  return res;
}

int32_t
mla_rt_client_write_output_buffer (const void * data, size_t size, int id)
{
  FILE *ofp;
  size_t osz = size, wosz;
  void *ovaddr;
  int32_t res = 0, i;
  char *dumpaddr;
  char full_opath[256];

  snprintf(full_opath, sizeof(full_opath) - 1, "/tmp/%s-%d.out", "ml_filter", id);

  ofp = fopen(full_opath, "w");
  if(ofp == NULL) {
    res = -1;
    goto err_ofp;
  }

  wosz = fwrite(data, 1, osz, ofp);
  if(osz != wosz) {
    res = -3;
    goto err_owrite;
  }

err_owrite:
err_omap:
  fclose(ofp);
err_ofp:
  return res;
}


/**
 * @brief Abstract API to call mla-rt load model 
 * @param path, the lm file or model path
 */
void mla_rt_client_load_model (const char * path)
{
  model = (uint64_t) mla_load_model((mla_handle_p) handle, path);
  simaailog(SIMAAILOG_DEBUG, "Loaded model handle 0x%x", model);
}

/**
 * @brief Abstract API to call mla-rt run_model with physical address
 * 
 * API to run a model on the mla with input and output as physical buffers to do
 * 'zero' copy run. This has to be called after load_model
 * @param in_buffer_id, The simaai buffer id input
 * @param in_buf_offset, the offset if a circular buffer is used
 * @param out_buf_id, the simaai output buffer id
 * @param out_buf_offset, the buffer offset as to where the output should be written
 * @return 0 on success, -1 on failure
 */
int mla_rt_client_run_model (simamm_buffer_id_t in_buffer_id,
			     uint32_t in_buf_offset,
			     simamm_buffer_id_t out_buf_id,
			     uint32_t out_buf_offset)
{
#ifdef ML_FILTER_EVAL_PERF
  struct timeval start, end;
  long seconds, micros;
  gettimeofday(&start, NULL);
#endif

  uint64_t in_addr = buffer_id_to_paddr(in_buffer_id) + in_buf_offset;
  uint64_t out_addr = buffer_id_to_paddr(out_buf_id) + out_buf_offset;

  simaailog(SIMAAILOG_DEBUG, "in_addr: 0x%x, out_addr: 0x%x\n", in_addr, out_addr);
  
  if (mla_run_model_phys((mla_model_p) model, in_addr , out_addr) != 0) {
    simaailog(SIMAAILOG_ERR, "Model run failed");
    return -1;
  }

#ifdef ML_FILTER_EVAL_PERF
  gettimeofday(&end, NULL);

  seconds = (end.tv_sec - start.tv_sec);
  micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
  printf("[%d]:[%s] mla_run_model_phys runtime %ld secs, %ld micros\n", __LINE__, __func__, seconds , micros);
#endif
  return 0;
}

/**
 * @brief Abstract API to call mla-rt run_model with physical address list
 * 
 * API to run a model on mla-rt with batch supported mode
 * @param in_addr_list, The list of input physical dram address.
 * @param out_buf_id, The output buffer id, this is allocated using gstsimaallocator
 * @param out_buff_offset, The buffer offset used by the system
 * @param batch_size, The batch size of the model
 * @param in_sz, is input size, for 'a' data in the batch
 * @param out_sz, is output size, for 'a' data in the batch
 * @return 0 on success, -1 on failure
 */
int mla_rt_client_batch_run_model (uint64_t in_addr_list[],
                                   simamm_buffer_id_t out_buf_id,
                                   uint32_t out_buf_offset,
                                   size_t batch_size,
                                   size_t in_sz,
                                   size_t out_sz)
{
#ifdef ML_FILTER_EVAL_PERF
  struct timeval start, end;
  long seconds, micros;
  gettimeofday(&start, NULL);
#endif

  uint64_t out_addr = buffer_id_to_paddr(out_buf_id) + out_buf_offset;
  uint64_t * out_addr_list = new uint64_t[batch_size];
  uint32_t offset = 0;
  
  for (int i = 0; i < batch_size; i++) {
      offset = i * out_sz;
      out_addr_list[i] = out_addr + offset;
  }

  simaailog(SIMAAILOG_DEBUG, "[%d]:[%s], in_sz:[%ld], out_sz:[%ld], batch_size: [%d]\n", __LINE__, __func__, in_sz, out_sz, batch_size);

  if (mla_run_batch_model_phys((mla_model_p) model, batch_size,
                               in_addr_list , in_sz,
                               out_addr_list, out_sz) != 0) {
    simaailog(SIMAAILOG_ERR, "Model run failed");
    return -1;
  }

#ifdef ML_FILTER_EVAL_PERF
  gettimeofday(&end, NULL);

  seconds = (end.tv_sec - start.tv_sec);
  micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
  simaailog(SIMAAILOG_INFO"[%d]:[%s] mla_run_model_phys runtime %ld secs, %ld micros\n", __LINE__, __func__, seconds , micros);
#endif
  delete [] out_addr_list;
  return 0;
}

/**
 * @brief API to prepare the output buffer for mla
 *
 * Prepares the output buffer, returns the buffer id which is later used for
 * mapping and attaching
 * @param[in] out_mem, the simaai_memory handle
 * @param[in] output_sz, size of the buffer to be allocated
 * @param[out] out_buf_id filled after allocating buffer
 * @return 0 on success < 0 on failure
 */
simaai_memory_t * mla_rt_client_prepare_outbuf (size_t output_sz)
{
  int32_t res = 0;
  simaai_memory_t * output_buff;

  // Logical number of buffers
  output_buff = simaai_memory_alloc(output_sz, SIMAAI_MEM_TARGET_GENERIC);
  if (output_buff == NULL) {
    res = -3;
    simaailog(SIMAAILOG_ERR, "MLA_prepare_out_buff 0x%x", res);
    return NULL;
  }

  simaailog(SIMAAILOG_INFO, "self->mem 0x%x", output_buff);
  return output_buff;
}

/** 
 * @brief Initialize mlart client
 *
 * The mla_rt_client_init, initializes the output buffer and then pre-loads the
 * lm file into the mla. This has to be called before running the model using run_model 
 * @param[in] out_mem the output memory handle
 * @param[in] output_sz, size of the output
 * @param[in] model_path, filesystem path of the mla model file
 * @param[out] out_buf_id, output buffer id used for attaching
 * @return 0 on success < 0 on failure
 */
int mla_rt_client_init (const char * model_path)
{
  simaailog_init(NULL);
  handle = (uint64_t) mla_get_handle();
  mla_rt_client_load_model(model_path);
  simaailog(SIMAAILOG_INFO, "mla_model : 0x%x\n", model);
  return 0;

err:
  return -1;
}

void mla_rt_client_cleanup (void) {
  mla_free_model((mla_model_p)model);
  mla_free_handle((mla_handle_p)handle);
}
